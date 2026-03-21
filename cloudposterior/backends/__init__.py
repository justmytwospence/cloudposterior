"""Compute backend abstraction."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterator

import arviz as az

from cloudposterior.progress import ProgressEvent
from cloudposterior.serialize import SamplingPayload


class SamplingJob(ABC):
    """Handle to a running or completed remote sampling job."""

    @abstractmethod
    def stream_progress(self) -> Iterator[ProgressEvent]:
        """Yield progress events as sampling runs."""
        ...

    @abstractmethod
    def result(self) -> az.InferenceData:
        """Block until sampling completes and return the trace."""
        ...

    @abstractmethod
    def cancel(self) -> None:
        """Cancel the running job."""
        ...


class ComputeBackend(ABC):
    """Abstract interface for compute backends."""

    @abstractmethod
    def submit(self, payload: SamplingPayload) -> SamplingJob:
        """Submit a sampling job and return a handle."""
        ...
