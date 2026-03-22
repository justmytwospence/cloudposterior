"""Compute backend abstraction."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Iterator

import arviz as az

from cloudposterior.progress import ProgressEvent
from cloudposterior.serialize import SamplingPayload

if TYPE_CHECKING:
    from cloudposterior.config import RemoteConfig


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


class RemoteEnvironment(ABC):
    """A provisioned environment that can run multiple sampling jobs.

    Model payloads are stored in a Volume. Per-call, only the payload path
    and sampling kwargs are sent -- the model is loaded from the Volume on
    the remote side.
    """

    @abstractmethod
    def submit(
        self, model_bytes: bytes, sample_kwargs: dict, nuts_sampler: str
    ) -> SamplingJob:
        """Submit a sampling job.

        model_bytes is used to check if the payload needs uploading to the
        Volume. Only the payload path + kwargs are sent to the remote.
        """
        ...

    @abstractmethod
    def teardown(self) -> None:
        """Tear down the compute session. Volumes persist."""
        ...


class ComputeBackend(ABC):
    """Abstract interface for compute backends."""

    @abstractmethod
    def submit(self, payload: SamplingPayload) -> SamplingJob:
        """Submit a one-shot sampling job and return a handle."""
        ...

    def provision(
        self,
        model_bytes: bytes,
        model,
        version_manifest: dict[str, str],
        config: RemoteConfig,
        project: str = "cloudposterior",
        idle_timeout: int = 600,
    ) -> RemoteEnvironment:
        """Provision a reusable environment.

        Creates the compute environment (app, Volume reference) but does not
        upload the model payload. Upload is deferred to the first cache miss.

        Args:
            model_bytes: Serialized model (cloudpickle + lz4, includes data).
            model: The PyMC model object (for deriving human-readable names).
            version_manifest: Package versions for image building.
            config: Resource configuration.
            project: Project name for Volume scoping.
            idle_timeout: Container idle timeout in seconds.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support persistent environments"
        )
