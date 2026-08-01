"""ntfy.sh push notifications for sampling start, completion, and errors."""

from __future__ import annotations

import logging
import os
import queue
import re
import threading
import uuid

import requests

from cloudposterior.progress import (
    JobPhase,
    PhaseUpdate,
    SamplingProgress,
)

_log = logging.getLogger(__name__)

# ntfy's own topic charset. Validated because the topic is interpolated into a
# URL path (a '/' or '?' would retarget the POST) and rendered into a link.
_TOPIC_RE = re.compile(r"[A-Za-z0-9_-]{1,64}")


def _format_time(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = int(seconds // 60)
    secs = seconds % 60
    return f"{minutes}m {secs:.0f}s"


def _ascii_bar(fraction: float, width: int = 12) -> str:
    filled = int(fraction * width)
    return "=" * filled + ">" + "." * max(0, width - filled - 1)


def _model_topic_name(model) -> str:
    """Generate an ntfy topic like 'eight-schools-subtle-pug-1f3c9a2b8d7e4f60'.

    ntfy topics are world-readable and world-writable: anyone who knows the
    topic reads every notification and can publish spoofed ones. The wordhash
    alone is ~22 bits behind a guessable model-name prefix, so a random 64-bit
    suffix carries the actual secrecy; the readable part is for the user.
    """
    from cloudposterior.naming import get_model_name, slugify
    from cloudposterior.wordhash import wordhash

    name = slugify(get_model_name(model), separator="-")
    secret = uuid.uuid4().hex[:16]
    words = wordhash(uuid.uuid4().bytes)
    # ntfy caps topics at 64 chars; the readable prefix yields first.
    budget = 64 - len(secret) - len(words) - 2
    name = name[:max(0, budget)].strip("-")
    return "-".join(p for p in (name, words, secret) if p)


class NtfyNotifier:
    """Send live-updating progress notifications via ntfy.

    Defaults to ntfy.sh. Override the server with ``server`` param
    or ``CLOUDPOSTERIOR_NTFY_SERVER`` env var to use a self-hosted instance.
    """

    def __init__(
        self,
        topic: str | None = None,
        server: str | None = None,
        model=None,
        instance_desc: str = "",
    ):
        self.topic = topic or self._resolve_topic(model)
        if not _TOPIC_RE.fullmatch(self.topic):
            raise ValueError(
                f"invalid ntfy topic {self.topic!r}: must be 1-64 characters "
                "of A-Z, a-z, 0-9, '_' or '-'"
            )
        self.server = (
            server
            or os.environ.get("CLOUDPOSTERIOR_NTFY_SERVER")
            or "https://ntfy.sh"
        ).rstrip("/")
        self._base_url = f"{self.server}/{self.topic}"
        self._instance_desc = instance_desc
        self._phases: list[tuple[str, str, str]] = []
        self._sampling: SamplingProgress | None = None
        # Sends run on a worker thread: show_phase is called from the thread
        # consuming the progress stream, where a slow or unreachable ntfy
        # server would otherwise stall the display for up to the full timeout.
        self._queue: queue.Queue = queue.Queue()
        self._worker: threading.Thread | None = None

    def _resolve_topic(self, model=None) -> str:
        env_topic = os.environ.get("CLOUDPOSTERIOR_NTFY_TOPIC")
        if env_topic:
            return env_topic
        return _model_topic_name(model)

    @property
    def url(self) -> str:
        return self._base_url

    def show_phase(self, update: PhaseUpdate):
        detail = update.message
        if update.status == "done" and update.elapsed > 0.1:
            detail += f" ({_format_time(update.elapsed)})"

        found = False
        for i, (s, label, d) in enumerate(self._phases):
            if label == update.phase.value:
                self._phases[i] = (update.status, update.phase.value, detail)
                found = True
                break
        if not found:
            self._phases.append((update.status, update.phase.value, detail))

        # Send on sampling start/completion, and on any error phase.
        if update.phase == JobPhase.SAMPLING or update.status == "error":
            self._send_update()

    def show_sampling(self, progress: SamplingProgress):
        # Track latest progress for the completion summary, but don't send
        self._sampling = progress

    def _build_body(self) -> str:
        lines = []

        # Phase summary
        phase_parts = []
        for status, label, detail in self._phases:
            icon = {
                "done": "done",
                "in_progress": "...",
                "error": "ERR",
            }.get(status, "?")
            phase_parts.append(f"[{icon}] {label}: {detail}")
        lines.append(" | ".join(phase_parts))

        # Sampling table
        if self._sampling and self._sampling.chains:
            lines.append("")
            lines.append("| Chain | Progress | Draws | Div | Step | Speed |")
            lines.append("|-------|----------|-------|-----|------|-------|")
            for chain_id in sorted(self._sampling.chains.keys()):
                cp = self._sampling.chains[chain_id]
                pct = cp.draw / cp.total if cp.total > 0 else 0
                bar = _ascii_bar(pct, width=10)
                speed = f"{cp.draws_per_sec:.0f}/s" if cp.draws_per_sec > 0 else "--"
                lines.append(
                    f"| {chain_id} [{cp.phase[:4]}] | `{bar}` | "
                    f"{cp.draw}/{cp.total} | {cp.divergences} | "
                    f"{cp.step_size:.3f} | {speed} |"
                )
            lines.append("")
            lines.append(
                f"Divergences: {self._sampling.total_divergences} | "
                f"Elapsed: {_format_time(self._sampling.elapsed)}"
            )

        return "\n".join(lines)

    def _is_complete(self) -> bool:
        # Sampling finishing is the completion signal: sends are triggered by
        # SAMPLING phase events, so a later phase (e.g. the remote download)
        # can never be "done" by the time the last send fires.
        return any(
            label == JobPhase.SAMPLING.value and status == "done"
            for status, label, _ in self._phases
        )

    def _has_error(self) -> bool:
        return any(status == "error" for status, _, _ in self._phases)

    def _send_update(self):
        title = "cloudposterior"
        if self._instance_desc:
            title += f" -- {self._instance_desc}"

        if self._has_error():
            tags = "rotating_light"
            priority = "4"
            title += " [failed]"
        elif self._is_complete():
            tags = "white_check_mark"
            priority = "3"
            title += " [complete]"
        else:
            tags = "hourglass_flowing_sand"
            priority = "2"

        self._enqueue(
            self._build_body().encode("utf-8"),
            {
                "X-Title": title,
                "X-Markdown": "yes",
                "X-Priority": priority,
                "X-Tags": tags,
            },
        )

    def _enqueue(self, body: bytes, headers: dict) -> None:
        if self._worker is None:
            self._worker = threading.Thread(
                target=self._drain, name="cloudposterior-ntfy", daemon=True
            )
            self._worker.start()
        self._queue.put((body, headers))

    def _drain(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:  # close() sentinel
                return
            body, headers = item
            try:
                resp = requests.post(
                    self._base_url, data=body, headers=headers, timeout=5
                )
                if resp.status_code >= 400:
                    # Silently swallowing this hid a permanently misconfigured
                    # server, and a 403 from a protected topic, completely.
                    _log.debug(
                        "ntfy POST to %s returned %s: %s",
                        self._base_url, resp.status_code, resp.text[:200],
                    )
            except Exception as exc:
                _log.debug("ntfy POST to %s failed: %s", self._base_url, exc)

    def stop(self) -> None:
        """Flush pending notifications and retire the worker thread."""
        if self._worker is None:
            return
        self._queue.put(None)
        self._worker.join(timeout=5)
        self._worker = None
