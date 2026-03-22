"""Modal compute backend."""

from __future__ import annotations

import contextlib
import io
import tempfile
import time
from contextlib import nullcontext as _nullcontext
from typing import Iterator

import msgpack

from cloudposterior.backends import ComputeBackend, RemoteEnvironment, SamplingJob
from cloudposterior.config import DEFAULT_PACKAGES, OPTIONAL_PACKAGES, RemoteConfig
from cloudposterior.progress import (
    ChainProgress,
    JobPhase,
    PhaseUpdate,
    ProgressEvent,
    SamplingProgress,
)
from cloudposterior.serialize import SamplingPayload


_MODAL_SETUP_MSG = (
    "Modal is not authenticated. To set up cloud execution:\n"
    "\n"
    "  uv add modal\n"
    "  uv run modal setup\n"
    "\n"
    "This opens a browser window to link your Modal account.\n"
    "See https://modal.com/docs/guide for details."
)


def _handle_modal_error(exc: Exception) -> Exception:
    """Wrap Modal auth/connection errors with a friendly message."""
    msg = str(exc).lower()
    if "authenticate" in msg or "token" in msg or "credential" in msg or "setup" in msg:
        err = RuntimeError(_MODAL_SETUP_MSG)
        err.__cause__ = exc
        return err
    return exc


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
        """Submit to Modal and yield progress events."""
        import modal

        app, remote_sample = _create_modal_app(
            self._payload.version_manifest,
            self._config,
        )

        # Don't call modal.enable_output() -- it enables a spinner and status
        # lines that interleave with our own progress display. Without it,
        # Modal runs silently and we show progress via our own Rich/ipywidgets UI.
        try:
            run_ctx = app.run()
        except Exception as exc:
            raise _handle_modal_error(exc)
        with run_ctx:
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
            # Load every group eagerly into memory so the temp file can be deleted
            idata = az.from_netcdf(tmp_path)
            for group in idata.groups():
                getattr(idata, group).load()
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


def _build_image(manifest: dict[str, str]):
    """Build a Modal image with packages matching the version manifest."""
    import modal

    python_version = manifest.get("python", "3.11.0")
    py_major_minor = ".".join(python_version.split(".")[:2])
    pip_specs = _build_pip_specs(manifest)

    return (
        modal.Image.debian_slim(python_version=py_major_minor)
        .uv_pip_install(pip_specs)
        .add_local_python_source("cloudposterior")
    )


def _create_persistent_app(
    manifest: dict[str, str],
    config: RemoteConfig,
    volume,
    dashboard_dict_name: str | None = None,
    model_label: str = "model",
):
    """Create a Modal app with a class-based sampler and mounted Volume.

    The Volume contains model payloads at human-readable paths. The sampler
    loads a payload by path on each call (fast local read from mounted volume).
    If dashboard_dict is provided, a web endpoint is added for live progress.
    """
    import modal

    image = _build_image(manifest)
    app = modal.App("cloudposterior-persistent")

    max_scaledown = 1200  # Modal caps at 20 minutes
    scaledown = min(config.idle_timeout, max_scaledown)

    @app.cls(
        image=image,
        serialized=True,
        cpu=config.cpu,
        memory=config.memory,
        timeout=config.timeout,
        scaledown_window=scaledown,
        volumes={"/data": volume},
        **({"gpu": config.gpu} if config.gpu else {}),
    )
    class Sampler:
        @modal.method(is_generator=True)
        def sample(self, payload_path: str, sample_kwargs: dict, nuts_sampler: str = "pymc"):
            from cloudposterior.remote.worker import run_sampling_from_volume

            yield from run_sampling_from_volume(
                f"/data/{payload_path}", sample_kwargs, nuts_sampler,
            )

    # Add dashboard web endpoints if requested
    dashboard_fn = None
    progress_fn = None
    if dashboard_dict_name is not None:
        _dict_name = dashboard_dict_name
        _uid = dashboard_dict_name.replace("cp-dash-", "")[:6]

        @app.function(serialized=True, image=image)
        @modal.fastapi_endpoint(method="GET", label=f"{model_label}-{_uid}")
        def serve_dashboard():
            from fastapi.responses import HTMLResponse
            from cloudposterior.dashboard import render_dashboard_html
            import modal as _modal
            try:
                d = _modal.Dict.from_name(_dict_name)
                progress_url = d["progress_url"]
            except (KeyError, Exception):
                progress_url = ""
            return HTMLResponse(render_dashboard_html(progress_url))

        @app.function(serialized=True, image=image)
        @modal.fastapi_endpoint(method="GET", label=f"{model_label}-{_uid}-progress")
        def serve_progress():
            from fastapi.responses import JSONResponse
            import modal as _modal
            try:
                d = _modal.Dict.from_name(_dict_name)
                data = d["progress"]
            except (KeyError, Exception):
                data = {"phases": [], "sampling": None, "complete": False}
            return JSONResponse(data)

        dashboard_fn = serve_dashboard
        progress_fn = serve_progress

    return app, Sampler, dashboard_fn, progress_fn


class PersistentModalSamplingJob(SamplingJob):
    """Sampling job that uses an already-provisioned Modal environment."""

    def __init__(
        self,
        sampler_cls,
        payload_path: str,
        sample_kwargs: dict,
        nuts_sampler: str,
    ):
        self._sampler_cls = sampler_cls
        self._payload_path = payload_path
        self._sample_kwargs = sample_kwargs
        self._nuts_sampler = nuts_sampler
        self._idata_bytes: bytes | None = None
        self._events: list[ProgressEvent] = []

    def stream_progress(self, output_widget=None) -> Iterator[ProgressEvent]:
        sampler = self._sampler_cls()
        gen = sampler.sample.remote_gen(
            self._payload_path,
            self._sample_kwargs,
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
        import arviz as az
        import lz4.frame

        if self._idata_bytes is None:
            for _ in self.stream_progress():
                pass

        if self._idata_bytes is None:
            raise RuntimeError("Sampling did not produce results")

        import os

        raw = lz4.frame.decompress(self._idata_bytes)
        with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as tmp:
            tmp.write(raw)
            tmp_path = tmp.name
        try:
            idata = az.from_netcdf(tmp_path)
            for group in idata.groups():
                getattr(idata, group).load()
            return idata
        finally:
            os.unlink(tmp_path)

    def cancel(self):
        pass


def _compute_payload_path(m_slug: str, model_bytes: bytes, data_bytes: bytes) -> str:
    """Compute the Volume path for a model payload.

    Human-readable directory structure, machine-correct filename.
    Args:
        m_slug: Pre-computed model slug (stable, not stack-dependent).
    """
    from cloudposterior.naming import data_slug, payload_hash

    d_slug = data_slug(data_bytes)
    p_hash = payload_hash(model_bytes)
    return f"{m_slug}/{d_slug}/payload-{p_hash}.bin"


class ModalEnvironment(RemoteEnvironment):
    """A provisioned Modal environment with payloads in a Volume."""

    def __init__(self, app, sampler_cls, volume, project: str, model_slug: str,
                 dashboard_dict=None, dashboard_fn=None, progress_fn=None):
        self._app = app
        self._sampler_cls = sampler_cls
        self._volume = volume
        self._project = project
        self._model_slug = model_slug
        self._dashboard_dict = dashboard_dict
        self._dashboard_fn = dashboard_fn
        self._progress_fn = progress_fn
        self._dashboard_url: str | None = None
        self._progress_url: str | None = None
        self._exit_stack = contextlib.ExitStack()
        self._running = False
        self._uploaded_hashes: set[str] = set()

    def _ensure_running(self):
        if not self._running:
            try:
                self._exit_stack.enter_context(self._app.run())
            except Exception as exc:
                raise _handle_modal_error(exc)
            self._running = True

            # Capture dashboard URLs after app starts
            if self._dashboard_fn is not None:
                try:
                    self._dashboard_url = self._dashboard_fn.get_web_url()
                except Exception:
                    pass
            if self._progress_fn is not None:
                try:
                    self._progress_url = self._progress_fn.get_web_url()
                    # Store progress URL in Dict so dashboard endpoint can find it
                    if self._dashboard_dict is not None and self._progress_url:
                        self._dashboard_dict["progress_url"] = self._progress_url
                except Exception:
                    pass

    def _upload_if_needed(self, model_bytes: bytes, payload_path: str) -> bool:
        """Upload model payload to Volume if not already there. Returns True if uploaded."""
        from cloudposterior.naming import payload_hash

        p_hash = payload_hash(model_bytes)
        if p_hash in self._uploaded_hashes:
            return False

        # Check Volume
        try:
            dir_path = "/".join(payload_path.split("/")[:-1])
            entries = self._volume.listdir(f"/{dir_path}")
            filename = payload_path.split("/")[-1]
            if any(e.path == filename for e in entries):
                self._uploaded_hashes.add(p_hash)
                return False
        except Exception:
            pass

        # Upload with force=True to overwrite if already exists
        import os

        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(model_bytes)
            tmp_path = tmp.name
        try:
            with self._volume.batch_upload(force=True) as upload:
                upload.put_file(tmp_path, f"/{payload_path}")
        finally:
            os.unlink(tmp_path)

        self._uploaded_hashes.add(p_hash)
        return True

    def submit(
        self, model_bytes: bytes, sample_kwargs: dict, nuts_sampler: str,
        payload_path: str | None = None,
    ) -> PersistentModalSamplingJob:
        self._ensure_running()

        if payload_path is None:
            raise ValueError("payload_path is required for persistent environments")

        # Upload is handled by the caller (after cache check) via _upload_if_needed
        return PersistentModalSamplingJob(
            self._sampler_cls,
            payload_path,
            sample_kwargs,
            nuts_sampler,
        )

    def teardown(self) -> None:
        self._exit_stack.close()
        self._running = False


class ModalBackend(ComputeBackend):
    """Modal compute backend."""

    def __init__(self, config: RemoteConfig | None = None, nuts_sampler: str = "pymc"):
        self._config = config or RemoteConfig()
        self._nuts_sampler = nuts_sampler

    def submit(self, payload: SamplingPayload) -> ModalSamplingJob:
        return ModalSamplingJob(payload, self._config, self._nuts_sampler)

    def provision(
        self,
        model_bytes: bytes,
        model,
        version_manifest: dict[str, str],
        config: RemoteConfig,
        project: str = "cloudposterior",
        idle_timeout: int = 600,
        dashboard: bool = False,
    ) -> ModalEnvironment:
        """Provision a persistent environment (no upload -- deferred to first cache miss)."""
        import modal
        import uuid

        from cloudposterior.naming import model_slug as compute_model_slug

        config.idle_timeout = idle_timeout
        volume_name = f"cp-{project}"
        try:
            volume = modal.Volume.from_name(volume_name, create_if_missing=True)
        except Exception as exc:
            raise _handle_modal_error(exc)

        m_slug = compute_model_slug(model)

        # Create dashboard Dict if requested
        dashboard_dict = None
        dashboard_dict_name = None
        if dashboard:
            dashboard_dict_name = f"cp-dash-{uuid.uuid4().hex[:8]}"
            dashboard_dict = modal.Dict.from_name(dashboard_dict_name, create_if_missing=True)

        app, sampler_cls, dashboard_fn, progress_fn = _create_persistent_app(
            version_manifest, config, volume,
            dashboard_dict_name=dashboard_dict_name,
            model_label=m_slug.replace("_", "-"),
        )
        return ModalEnvironment(
            app, sampler_cls, volume, project, m_slug,
            dashboard_dict=dashboard_dict,
            dashboard_fn=dashboard_fn,
            progress_fn=progress_fn,
        )

    @staticmethod
    def cleanup_volumes(project: str = "cloudposterior") -> None:
        """Delete the Volume for a project."""
        import modal

        volume_name = f"cp-{project}"
        try:
            modal.Volume.objects.delete(volume_name)
        except Exception:
            pass
