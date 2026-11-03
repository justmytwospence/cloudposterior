"""Cloud transport compatibility for pymc-monitor's typed progress events."""

from __future__ import annotations

from pymc_monitor.events import (  # noqa: F401
    ChainProgress,
    ConvergenceUpdate,
    JobPhase,
    ParamConvergence,
    PhaseUpdate,
    ProgressAggregator,
    ProgressEvent,
    RunFinished,
    SamplingProgress,
    _as_float,
    _as_int,
    dispatch_event,
    event_from_dict as _decode,
    event_to_dict,
    make_sampling_callback,
)


def decode_progress_event(data: dict) -> ProgressEvent | None:
    """Read the shared schema, skipping phases unknown to this cloud client."""
    if not isinstance(data, dict):
        return None
    if data.get("type") == "phase":
        try:
            JobPhase(data["phase"])
        except (KeyError, ValueError):
            return None
    return _decode(data)
