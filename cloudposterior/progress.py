"""Progress tracking data structures and PyMC callback factory."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from queue import Queue
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


def decode_progress_event(data: dict) -> ProgressEvent | None:
    """Convert a decoded msgpack dict into a typed ProgressEvent.

    Shared by the client streaming path and the cp.map worker (which writes
    progress server-side), so it lives here with no Modal/backend dependency.
    """
    msg_type = data.get("type")

    if msg_type == "phase":
        return PhaseUpdate(
            phase=JobPhase(data["phase"]),
            status=data["status"],
            message=data["message"],
            elapsed=data["elapsed"],
        )

    if msg_type == "sampling":
        chains = {}
        for chain_id_str, cdata in data.get("chains", {}).items():
            chain_id = int(chain_id_str) if isinstance(chain_id_str, str) else chain_id_str
            chains[chain_id] = ChainProgress(
                draw=cdata["draw"],
                total=cdata["total"],
                phase=cdata["phase"],
                draws_per_sec=cdata.get("draws_per_sec", 0.0),
                eta_seconds=cdata.get("eta_seconds", 0.0),
                divergences=cdata.get("divergences", 0),
                mean_tree_depth=cdata.get("mean_tree_depth", 0.0),
                step_size=cdata.get("step_size", 0.0),
                tree_size=cdata.get("tree_size", 0),
            )
        return SamplingProgress(
            chains=chains,
            total_divergences=data.get("total_divergences", 0),
            elapsed=data.get("elapsed", 0.0),
            total_draws=data.get("total_draws", 0),
        )

    if msg_type == "convergence":
        params = {}
        for name, pdata in data.get("params", {}).items():
            params[name] = ParamConvergence(
                rhat=pdata["rhat"],
                ess_bulk=pdata["ess_bulk"],
                ess_tail=pdata["ess_tail"],
            )
        traces = data.get("traces", {})
        return ConvergenceUpdate(params=params, draws=data.get("draws", 0), traces=traces)

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
    """
    chain_draw_counts: dict[int, int] = {}
    chain_start_times: dict[int, float] = {}
    chain_divergences: dict[int, int] = {}
    chain_tree_depths: dict[int, list[float]] = {}

    def callback(trace, draw):
        chain = draw.chain
        is_tuning = draw.tuning

        if chain not in chain_start_times:
            chain_start_times[chain] = time.time()
            chain_draw_counts[chain] = 0
            chain_divergences[chain] = 0
            chain_tree_depths[chain] = []

        chain_draw_counts[chain] += 1
        current_draw = chain_draw_counts[chain]

        stats = draw.stats[0] if draw.stats else {}
        diverging = stats.get("diverging", False)
        tree_depth = stats.get("tree_depth", 0)
        step_size = stats.get("step_size", 0.0)

        if diverging:
            chain_divergences[chain] += 1
        chain_tree_depths[chain].append(tree_depth)

        elapsed = time.time() - chain_start_times[chain]
        dps = current_draw / elapsed if elapsed > 0 else 0.0
        total = tune if is_tuning else draws
        remaining = total - current_draw
        eta = remaining / dps if dps > 0 else 0.0

        mean_td = sum(chain_tree_depths[chain]) / len(chain_tree_depths[chain])

        progress = ChainProgress(
            draw=current_draw,
            total=total,
            phase="tuning" if is_tuning else "sampling",
            draws_per_sec=dps,
            eta_seconds=eta,
            divergences=chain_divergences[chain],
            mean_tree_depth=mean_td,
            step_size=step_size,
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
            # Drain queue
            while time.time() < deadline:
                try:
                    chain, progress = self._queue.get(timeout=0.1)
                    self._chains[chain] = progress
                except Exception:
                    break

            if self._chains:
                total_div = sum(c.divergences for c in self._chains.values())
                warnings = []
                if total_div > 0:
                    warnings.append(f"{total_div} divergence(s) so far")
                yield SamplingProgress(
                    chains=dict(self._chains),
                    total_divergences=total_div,
                    elapsed=time.time() - self._start_time,
                    warnings=warnings,
                )

    def stop(self):
        self._stopped = True
