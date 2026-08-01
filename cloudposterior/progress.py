"""Progress tracking data structures and PyMC callback factory."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from queue import Empty, Queue
from typing import Iterator


class JobPhase(str, Enum):
    SERIALIZING = "serializing"
    CACHE_HIT = "cache_hit"
    DATA_UPLOADING = "data_uploading"
    UPLOADING = "uploading"
    PROVISIONING = "provisioning"
    CONTAINER_READY = "container_ready"
    DEVICE = "device"
    COMPILING = "compiling"
    SAMPLING = "sampling"
    DOWNLOADING = "downloading"


@dataclass
class PhaseUpdate:
    phase: JobPhase
    status: str  # "in_progress", "done", "error"
    message: str
    elapsed: float
    progress: float | None = None  # 0-1 fraction, None if indeterminate


@dataclass
class ChainProgress:
    draw: int
    total: int
    phase: str  # "tuning" | "sampling" | "done"
    draws_per_sec: float = 0.0
    eta_seconds: float = 0.0
    divergences: int = 0
    mean_tree_depth: float = 0.0
    step_size: float = 0.0
    tree_size: int = 0  # grad evals (leapfrog steps)


@dataclass
class SamplingProgress:
    chains: dict[int, ChainProgress]
    total_divergences: int = 0
    elapsed: float = 0.0
    total_draws: int = 0
    warnings: list[str] = field(default_factory=list)


@dataclass
class ParamConvergence:
    rhat: float
    ess_bulk: int
    ess_tail: int


@dataclass
class ConvergenceUpdate:
    params: dict[str, ParamConvergence]
    draws: int = 0
    traces: dict[str, list[list[float]]] = field(default_factory=dict)  # param -> [[chain0_vals], [chain1_vals], ...]


# Union type for progress events streamed from remote
ProgressEvent = PhaseUpdate | SamplingProgress | ConvergenceUpdate


def _as_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_int(value, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def decode_progress_event(data: dict) -> ProgressEvent | None:
    """Convert a decoded msgpack dict into a typed ProgressEvent.

    Shared by the client streaming path and the cp.map worker (which writes
    progress server-side), so it lives here with no Modal/backend dependency.

    Every field is read defensively and numerics are coerced: a partial or
    stringly payload from a newer/older worker previously raised KeyError
    mid-decode, and a null slipping through reached the dashboard where
    ``.toFixed`` on it read as the run having gone offline.
    """
    msg_type = data.get("type")

    if msg_type == "phase":
        # Tolerate phase names this client doesn't know (a newer worker may add
        # one); skipping the event is better than the caller mistaking the
        # undecodable chunk for the result payload.
        try:
            phase = JobPhase(data["phase"])
        except (KeyError, ValueError):
            return None
        return PhaseUpdate(
            phase=phase,
            status=data.get("status", "in_progress"),
            message=data.get("message", ""),
            elapsed=_as_float(data.get("elapsed")),
        )

    if msg_type == "sampling":
        chains = {}
        for chain_id_str, cdata in data.get("chains", {}).items():
            try:
                chain_id = int(chain_id_str)
            except (TypeError, ValueError):
                continue
            chains[chain_id] = ChainProgress(
                draw=_as_int(cdata.get("draw")),
                total=_as_int(cdata.get("total")),
                phase=cdata.get("phase", "sampling"),
                draws_per_sec=_as_float(cdata.get("draws_per_sec")),
                eta_seconds=_as_float(cdata.get("eta_seconds")),
                divergences=_as_int(cdata.get("divergences")),
                mean_tree_depth=_as_float(cdata.get("mean_tree_depth")),
                step_size=_as_float(cdata.get("step_size")),
                tree_size=_as_int(cdata.get("tree_size")),
            )
        return SamplingProgress(
            chains=chains,
            total_divergences=_as_int(data.get("total_divergences")),
            elapsed=_as_float(data.get("elapsed")),
            total_draws=_as_int(data.get("total_draws")),
        )

    if msg_type == "convergence":
        params = {}
        for name, pdata in data.get("params", {}).items():
            params[name] = ParamConvergence(
                rhat=_as_float(pdata.get("rhat")),
                ess_bulk=_as_float(pdata.get("ess_bulk")),
                ess_tail=_as_float(pdata.get("ess_tail")),
            )
        traces = data.get("traces", {})
        return ConvergenceUpdate(
            params=params, draws=_as_int(data.get("draws")), traces=traces
        )

    return None


def dispatch_event(event: ProgressEvent | None, sinks) -> None:
    """Route a ProgressEvent to the matching method on each sink.

    Single source of truth for the event->sink fan-out used by both the
    client display loop (api.py) and the cp.map worker's dashboard writer.
    """
    if event is None:
        return
    for sink in sinks:
        if isinstance(event, PhaseUpdate):
            sink.show_phase(event)
        elif isinstance(event, SamplingProgress):
            sink.show_sampling(event)
        elif isinstance(event, ConvergenceUpdate) and hasattr(sink, "show_convergence"):
            sink.show_convergence(event)


def make_sampling_callback(queue: Queue, tune: int, draws: int):
    """Create a PyMC sampling callback that pushes progress to a queue.

    The callback receives (trace, draw) on each MCMC iteration.
    PyMC's draw object provides: chain, tuning, stats, etc.

    NOTE: this mirrors the per-draw bookkeeping in the remote worker's
    ``progress_callback`` (cloudposterior/remote/worker.py) -- the worker
    version additionally accumulates traces and honors the stop flag. Keep the
    draw counting / phase reset / windowed tree-depth logic in sync.
    """
    chain_draw_counts: dict[int, int] = {}
    chain_start_times: dict[int, float] = {}
    chain_divergences: dict[int, int] = {}
    chain_tree_depths: dict[int, list[float]] = {}
    chain_phase: dict[int, bool] = {}  # chain -> is_tuning

    def callback(trace, draw):
        chain = draw.chain
        is_tuning = draw.tuning

        if chain not in chain_start_times:
            chain_start_times[chain] = time.time()
            chain_draw_counts[chain] = 0
            chain_divergences[chain] = 0
            chain_tree_depths[chain] = []
            chain_phase[chain] = is_tuning

        # Tuning -> sampling transition: restart the draw count and clock so
        # progress shows draws out of `draws` (not tune+draws) with a sane ETA.
        if chain_phase.get(chain) and not is_tuning:
            chain_draw_counts[chain] = 0
            chain_start_times[chain] = time.time()
            chain_phase[chain] = False

        chain_draw_counts[chain] += 1
        current_draw = chain_draw_counts[chain]

        stats = draw.stats[0] if draw.stats else {}
        diverging = stats.get("diverging", False)
        tree_depth = stats.get("tree_depth", 0)
        tree_size = stats.get("n_steps", stats.get("tree_size", 0))
        step_size = stats.get("step_size", 0.0)

        if diverging:
            chain_divergences[chain] += 1
        chain_tree_depths[chain].append(tree_depth)

        elapsed = time.time() - chain_start_times[chain]
        dps = current_draw / elapsed if elapsed > 0 else 0.0
        total = tune if is_tuning else draws
        remaining = max(0, total - current_draw)
        eta = remaining / dps if dps > 0 else 0.0

        recent_td = chain_tree_depths[chain][-100:]
        mean_td = sum(recent_td) / len(recent_td)

        progress = ChainProgress(
            draw=current_draw,
            total=total,
            phase="tuning" if is_tuning else "sampling",
            draws_per_sec=dps,
            eta_seconds=eta,
            divergences=chain_divergences[chain],
            mean_tree_depth=mean_td,
            step_size=step_size,
            tree_size=tree_size,
        )
        queue.put((chain, progress))

    return callback


class ProgressAggregator:
    """Reads per-draw events from queue, emits batched SamplingProgress snapshots."""

    def __init__(self, queue: Queue, interval: float = 0.5):
        self._queue = queue
        self._interval = interval
        self._chains: dict[int, ChainProgress] = {}
        self._start_time = time.time()
        self._stopped = False

    def snapshots(self) -> Iterator[SamplingProgress]:
        """Yield aggregated snapshots at regular intervals."""
        while not self._stopped:
            deadline = time.time() + self._interval
            # Drain the queue until the interval elapses. `continue` on an
            # empty queue, not `break`: breaking out on the first timeout made
            # an idle chain re-emit an identical snapshot every ~0.1s instead
            # of once per interval -- ~5x the intended widget re-renders.
            while time.time() < deadline and not self._stopped:
                try:
                    chain, progress = self._queue.get(timeout=0.1)
                    self._chains[chain] = progress
                except Empty:
                    continue
                except Exception:
                    break

            snapshot = self._snapshot()
            if snapshot is not None:
                yield snapshot

        # One final snapshot after stop(): the last per-draw events can land
        # between the previous emit and the stop, which left the display
        # frozen just short of the total (e.g. 980/1000).
        final = self._snapshot()
        if final is not None:
            yield final

    def _snapshot(self) -> SamplingProgress | None:
        if not self._chains:
            return None
        total_div = sum(c.divergences for c in self._chains.values())
        warnings = []
        if total_div > 0:
            warnings.append(f"{total_div} divergence(s) so far")
        return SamplingProgress(
            chains=dict(self._chains),
            total_divergences=total_div,
            elapsed=time.time() - self._start_time,
            warnings=warnings,
        )

    def stop(self):
        self._stopped = True
