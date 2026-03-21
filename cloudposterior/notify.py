"""ntfy.sh push notifications with live-updating progress."""

from __future__ import annotations

import os
import uuid
from typing import Any

import requests

from cloudposterior.progress import (
    ChainProgress,
    JobPhase,
    PhaseUpdate,
    SamplingProgress,
)


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
    """Generate a readable ntfy topic name from a PyMC model.

    Uses the model name if set, otherwise derives a name from
    the free random variable names. Appends a short hash for uniqueness.
    """
    import re

    suffix = uuid.uuid4().hex[:6]

    if model is not None and hasattr(model, "name") and model.name:
        # Clean the model name for URL safety
        slug = re.sub(r"[^a-zA-Z0-9]+", "-", model.name).strip("-").lower()
        return f"pd-{slug}-{suffix}"

    # Derive from RV names
    if model is not None and hasattr(model, "free_RVs") and model.free_RVs:
        rv_names = [rv.name.split("::")[-1] for rv in model.free_RVs[:3]]
        slug = "-".join(rv_names)
        slug = re.sub(r"[^a-zA-Z0-9]+", "-", slug).strip("-").lower()
        if len(model.free_RVs) > 3:
            slug += f"-plus{len(model.free_RVs) - 3}"
        return f"pd-{slug}-{suffix}"

    return f"pd-{suffix}"


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
        self.server = (
            server
            or os.environ.get("CLOUDPOSTERIOR_NTFY_SERVER")
            or "https://ntfy.sh"
        ).rstrip("/")
        self.event_id = f"pd-{uuid.uuid4().hex[:8]}"
        self._base_url = f"{self.server}/{self.topic}"
        self._instance_desc = instance_desc
        self._phases: list[tuple[str, str, str]] = []
        self._sampling: SamplingProgress | None = None
        self._announced = False

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
        if update.status == "done":
            detail += f" ({_format_time(update.elapsed)})"

        found = False
        for i, (s, label, d) in enumerate(self._phases):
            if label == update.phase.value:
                self._phases[i] = (update.status, update.phase.value, detail)
                found = True
                break
        if not found:
            self._phases.append((update.status, update.phase.value, detail))

        self._send_update()

    def show_sampling(self, progress: SamplingProgress):
        self._sampling = progress
        self._send_update()

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
        for status, label, _ in self._phases:
            if label == JobPhase.DOWNLOADING.value and status == "done":
                return True
            if label == JobPhase.CACHE_HIT.value:
                return True
        return False

    def _send_update(self):
        complete = self._is_complete()
        title = "cloudposterior"
        if self._instance_desc:
            title += f" -- {self._instance_desc}"

        if complete:
            tags = "white_check_mark"
            priority = "3"
            title += " [complete]"
        else:
            tags = "hourglass_flowing_sand"
            priority = "2"

        try:
            requests.post(
                self._base_url,
                data=self._build_body().encode("utf-8"),
                headers={
                    "X-Title": title,
                    "X-Markdown": "yes",
                    "X-Priority": priority,
                    "X-Tags": tags,
                },
                timeout=5,
            )
        except Exception:
            pass  # notifications are best-effort
