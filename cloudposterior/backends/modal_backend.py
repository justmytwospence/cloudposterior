"""Modal compute backend."""

from __future__ import annotations

import io
import time
from contextlib import nullcontext as _nullcontext
from typing import Iterator

import msgpack

from cloudposterior.backends import ComputeBackend, SamplingJob
from cloudposterior.config import DEFAULT_PACKAGES, OPTIONAL_PACKAGES, RemoteConfig
from cloudposterior.progress import (
    ChainProgress,
    JobPhase,
    PhaseUpdate,
    ProgressEvent,
    SamplingProgress,
)
from cloudposterior.serialize import SamplingPayload


def _build_pip_specs(manifest: dict[str, str]) -> list[str]:
    """Convert a version manifest into pinned pip install specs."""
    specs = []
    for pkg in DEFAULT_PACKAGES:
        if pkg in manifest:
            specs.append(f"{pkg}=={manifest[pkg]}")
        else:
            specs.append(pkg)

    # Add optional sampler packages if present in manifest
    for key, pip_name in OPTIONAL_PACKAGES.items():
        if key in manifest:
            specs.append(f"{pip_name}=={manifest[key]}")

    return specs


def _create_modal_app(manifest: dict[str, str], config: RemoteConfig):
    """Create a Modal app with an image matching the version manifest."""
    import modal

    python_version = manifest.get("python", "3.11.0")
    py_major_minor = ".".join(python_version.split(".")[:2])

    pip_specs = _build_pip_specs(manifest)

    image = (
        modal.Image.debian_slim(python_version=py_major_minor)
        .uv_pip_install(pip_specs)
        .add_local_python_source("cloudposterior")
    )

    app = modal.App("cloudposterior")

    # Use serialized=True to allow defining the function dynamically
    # (not at module scope). Modal will cloudpickle the function.
    @app.function(
        image=image,
        serialized=True,
        cpu=config.cpu,
        memory=config.memory,
        timeout=config.timeout,
        **({"gpu": config.gpu} if config.gpu else {}),
    )
    def remote_sample(model_bytes: bytes, data_bytes: bytes, sample_kwargs: dict, nuts_sampler: str = "pymc"):
        from cloudposterior.remote.worker import run_sampling
        yield from run_sampling(model_bytes, data_bytes, sample_kwargs, nuts_sampler)

    return app, remote_sample


class ModalSamplingJob(SamplingJob):
    """Handle to a Modal sampling job."""

    def __init__(
        self,
        payload: SamplingPayload,
        config: RemoteConfig,
        nuts_sampler: str,
    ):
        self._payload = payload
        self._config = config
        self._nuts_sampler = nuts_sampler
        self._idata_bytes: bytes | None = None
        self._events: list[ProgressEvent] = []

    def stream_progress(self, output_widget=None) -> Iterator[ProgressEvent]:
        """Submit to Modal and yield progress events.

        Args:
            output_widget: Optional ipywidgets.Output to capture Modal's stdout into.
                           If provided, Modal output goes into the widget instead of stdout.
        """
        import modal

        app, remote_sample = _create_modal_app(
            self._payload.version_manifest,
            self._config,
        )

        # In notebooks, capture Modal output into the widget
        # In terminal, let Modal render normally
        modal_ctx = modal.enable_output()
        widget_ctx = output_widget if output_widget is not None else _nullcontext()

        with modal_ctx as output_manager:
            if output_widget is not None:
                output_manager.set_quiet_mode(True)
            with widget_ctx, app.run():
                gen = remote_sample.remote_gen(
                    self._payload.model_bytes,
                    self._payload.data_bytes,
                    self._payload.sample_kwargs,
                    self._nuts_sampler,
                )

                for chunk in gen:
                    try:
                        decoded = msgpack.unpackb(chunk, raw=False)
                    except Exception:
                        self._idata_bytes = chunk
                        continue

                    event = _decode_progress_event(decoded)
                    if event is not None:
                        self._events.append(event)
                        yield event
                    elif decoded.get("type") == "result":
                        pass

    def result(self):
        """Return the InferenceData. Must call stream_progress first."""
        import arviz as az
        import lz4.frame

        if self._idata_bytes is None:
            # If stream_progress wasn't called, run it now
            for _ in self.stream_progress():
                pass

        if self._idata_bytes is None:
            raise RuntimeError("Sampling did not produce results")

        import os
        import tempfile

        raw = lz4.frame.decompress(self._idata_bytes)
        with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as tmp:
            tmp.write(raw)
            tmp_path = tmp.name
        try:
            # Load eagerly so we can delete the temp file
            idata = az.from_netcdf(tmp_path)
            idata.load()
            return idata
        finally:
            os.unlink(tmp_path)

    def cancel(self):
        # Modal doesn't have a clean cancel API for generators yet
        pass


def _decode_progress_event(data: dict) -> ProgressEvent | None:
    """Convert a decoded msgpack dict into a typed ProgressEvent."""
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
        )

    return None


class ModalBackend(ComputeBackend):
    """Modal compute backend."""

    def __init__(self, instance: str | None = None, nuts_sampler: str = "pymc"):
        self._config = RemoteConfig.from_instance(instance)
        self._nuts_sampler = nuts_sampler

    def submit(self, payload: SamplingPayload) -> ModalSamplingJob:
        return ModalSamplingJob(payload, self._config, self._nuts_sampler)
