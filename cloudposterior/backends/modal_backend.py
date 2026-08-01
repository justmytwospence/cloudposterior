"""Modal compute backend."""

from __future__ import annotations

import contextlib
import tempfile
from typing import Iterator

import msgpack

from cloudposterior.backends import ComputeBackend, RemoteEnvironment, SamplingJob
from cloudposterior.config import DEFAULT_PACKAGES, OPTIONAL_PACKAGES, RemoteConfig
from cloudposterior.progress import (
    ProgressEvent,
    decode_progress_event,
)
from cloudposterior.serialize import SamplingPayload


# Keep the N most recent payload-*.bin files per model directory in the
# Volume. Older payloads from past model edits are pruned on upload.
_PAYLOAD_KEEP_PER_MODEL = 5


def _run_blocking(fn, *args, **kwargs):
    """Run a blocking Modal call, off the event loop if one is active.

    Inside a running asyncio loop (marimo's kernel, an async web app) a blocking
    Modal interface raises noisy AsyncUsageWarnings and stalls the loop. Running
    it in a worker thread (which has no event loop) avoids both. Outside a loop,
    call directly. Use for one-shot client calls (provision, upload, web URLs).
    """
    import asyncio

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return fn(*args, **kwargs)

    import concurrent.futures

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        return ex.submit(fn, *args, **kwargs).result()


def _run_blocking_op(env, method_name: str, *args):
    """Invoke a blocking @modal.method() on the env's Sampler Cls by name.

    Ensures the app is running, instantiates the Cls, and runs the named method
    off the event loop. Shared by the predictive, SMC, and compute_log_likelihood
    client paths (all blocking, non-streaming remote ops).
    """
    env._ensure_running()
    sampler = env._sampler_cls()
    return _run_blocking(getattr(sampler, method_name).remote, *args)


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


def _build_pip_specs(
    manifest: dict[str, str],
    gpu: str | None = None,
    nuts_sampler: str = "pymc",
) -> list[str]:
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

    # nutpie is the default sampler -- install it on every CPU image so the
    # default works even if the user doesn't have nutpie locally. It is NOT
    # pinned to the local version: nutpie recompiles the model remotely, so only
    # the model pickle (pymc/pytensor/numpy) must version-match. The >=0.13 floor
    # guarantees the progress_callback API used for live monitoring.
    if not gpu:
        specs.append("nutpie>=0.13")

    # GPU containers: always install numpyro + jax[cuda12] since GPU
    # is only useful for JAX-based samplers. This ensures the container
    # is ready when pm.sample(nuts_sampler="numpyro") is called.
    if gpu:
        if "numpyro" not in manifest:
            specs.append("numpyro")
        jax_version = manifest.get("jax", "")
        if jax_version:
            specs.append(f"jax[cuda12]=={jax_version}")
        else:
            specs.append("jax[cuda12]")
    # Non-GPU: install numpyro/jax if requested by sampler
    elif nuts_sampler in ("numpyro", "blackjax"):
        if "numpyro" not in manifest:
            specs.append("numpyro")
        if "jax" not in manifest:
            specs.append("jax")

    return specs


def _create_modal_app(manifest: dict[str, str], config: RemoteConfig,
                      nuts_sampler: str = "pymc"):
    """Create a Modal app with an image matching the version manifest."""
    import modal

    python_version = manifest.get("python", "3.11.0")
    py_major_minor = ".".join(python_version.split(".")[:2])

    pip_specs = _build_pip_specs(manifest, gpu=config.gpu, nuts_sampler=nuts_sampler)

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
    def remote_sample(model_bytes: bytes, sample_kwargs: dict, nuts_sampler: str = "pymc"):
        from cloudposterior.remote.worker import run_sampling
        yield from run_sampling(model_bytes, sample_kwargs, nuts_sampler)

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
        self._streamed = False

    def stream_progress(self, output_widget=None) -> Iterator[ProgressEvent]:
        """Submit to Modal and yield progress events."""
        self._streamed = True

        app, remote_sample = _create_modal_app(
            self._payload.version_manifest,
            self._config,
            self._nuts_sampler,
        )

        # Don't call modal.enable_output() -- it enables a spinner and status
        # lines that interleave with our own progress display. Without it,
        # Modal runs silently and we show progress via our own Rich/ipywidgets UI.
        # Enter/exit app.run() off the event loop (like the persistent path's
        # _ensure_running): its blocking Modal calls warn and stall inside an
        # async host (marimo, async web apps).
        try:
            run_ctx = app.run()
            _run_blocking(run_ctx.__enter__)
        except Exception as exc:
            raise _handle_modal_error(exc)
        try:
            gen = remote_sample.remote_gen(
                self._payload.model_bytes,
                self._payload.sample_kwargs,
                self._nuts_sampler,
            )
            self._idata_bytes = yield from _stream_events(gen, self._events)
        finally:
            try:
                _run_blocking(run_ctx.__exit__, None, None, None)
            except Exception:
                pass

    def result(self):
        """Return the InferenceData. Must call stream_progress first."""
        from cloudposterior.serialize import deserialize_inference_data

        if self._idata_bytes is None and not self._streamed:
            # stream_progress was never called -- drive it once.
            for _ in self.stream_progress():
                pass

        if self._idata_bytes is None:
            # Streaming ran but yielded no trace. Re-streaming here would
            # silently launch a second *paid* sampling run, so fail instead.
            raise RuntimeError(
                "the sampling stream finished without producing a trace; "
                "refusing to re-run the remote job"
            )

        return deserialize_inference_data(self._idata_bytes)

    def cancel(self):
        # Modal doesn't have a clean cancel API for generators yet
        pass


# Decoding moved to cloudposterior.progress (shared with the cp.map worker).
# Back-compat alias for any external importer of the old private name.
_decode_progress_event = decode_progress_event


def _stream_events(gen, events_list) -> Iterator[ProgressEvent]:
    """Decode a worker generator into ProgressEvents and return the compressed
    InferenceData bytes.

    The worker yields msgpack progress events, then a ``{"type": "result"}``
    sentinel, then one raw (non-msgpack) chunk of lz4-compressed NetCDF. We use
    the sentinel as the primary signal for the result chunk (so a genuine decode
    error is not silently mistaken for the result), with a non-msgpack chunk as a
    defensive fallback. A sampling failure in the worker re-raises here.

    The blocking Modal generator is drained in a background thread -- which has
    no event loop, so it doesn't emit AsyncUsageWarnings in async hosts (marimo,
    async web apps). Decoding and ``yield`` stay on the calling thread, so the
    progress-display sinks update in the right runtime context (the marimo
    widget's trait writes need the main thread).
    """
    import queue
    import threading

    chunk_q: queue.Queue = queue.Queue()
    _SENTINEL = object()
    err: dict = {}

    def _drain():
        try:
            for chunk in gen:
                chunk_q.put(chunk)
        except Exception as exc:  # surface worker/Modal errors on the main thread
            err["exc"] = exc
        finally:
            chunk_q.put(_SENTINEL)

    consumer = threading.Thread(target=_drain, daemon=True)
    consumer.start()

    idata_bytes = None
    expecting_result = False
    while True:
        chunk = chunk_q.get()
        if chunk is _SENTINEL:
            break
        if expecting_result:
            idata_bytes = chunk
            expecting_result = False
            continue
        try:
            unpacker = msgpack.Unpacker(raw=False)
            unpacker.feed(chunk)
            decoded_any = False
            for decoded in unpacker:
                decoded_any = True
                if isinstance(decoded, dict) and decoded.get("type") == "result":
                    expecting_result = True
                    continue
                # A single undecodable event (e.g. malformed fields) must not
                # make the whole chunk be mistaken for the result bytes below.
                try:
                    event = _decode_progress_event(decoded)
                except Exception:
                    continue
                if event is not None:
                    events_list.append(event)
                    yield event
            if not decoded_any:
                idata_bytes = chunk
        except Exception:
            # Non-msgpack chunk: the compressed InferenceData bytes.
            idata_bytes = chunk

    consumer.join()
    if "exc" in err:
        raise err["exc"]
    return idata_bytes


def _build_image(manifest: dict[str, str], gpu: str | None = None,
                 nuts_sampler: str = "pymc"):
    """Build a Modal image with packages matching the version manifest."""
    import modal

    python_version = manifest.get("python", "3.11.0")
    py_major_minor = ".".join(python_version.split(".")[:2])
    pip_specs = _build_pip_specs(manifest, gpu=gpu, nuts_sampler=nuts_sampler)

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
    dashboard: bool = False,
    model_label: str = "model",
    stop_token: str | None = None,
    nuts_sampler: str = "pymc",
):
    """Create a Modal app with a class-based sampler and mounted Volume.

    The Volume contains model payloads at human-readable paths. The sampler
    loads a payload by path on each call (fast local read from mounted volume).
    When ``dashboard_dict_name`` is set a token-gated ``/stop`` endpoint is added
    (used by both the in-notebook stop button and the dashboard); the dashboard
    and progress endpoints are added only when ``dashboard`` is True.
    """
    import modal

    image = _build_image(manifest, gpu=config.gpu, nuts_sampler=nuts_sampler)
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
        def sample(self, payload_path: str, sample_kwargs: dict, nuts_sampler: str = "pymc",
                   stop_dict_name: str | None = None):
            from cloudposterior.remote.worker import run_sampling_from_volume

            yield from run_sampling_from_volume(
                f"/data/{payload_path}", sample_kwargs, nuts_sampler,
                stop_dict_name=stop_dict_name,
            )

        @modal.method()
        def prior_predictive(self, payload_path: str, sample_kwargs: dict) -> bytes:
            from cloudposterior.remote.worker import run_prior_predictive

            return run_prior_predictive(f"/data/{payload_path}", sample_kwargs)

        @modal.method()
        def posterior_predictive(self, payload_path: str, idata_bytes: bytes,
                                 sample_kwargs: dict) -> bytes:
            from cloudposterior.remote.worker import run_posterior_predictive

            return run_posterior_predictive(f"/data/{payload_path}", idata_bytes, sample_kwargs)

        @modal.method()
        def sample_blocking(self, payload_path: str, sample_kwargs: dict,
                            nuts_sampler: str = "nutpie",
                            progress_dict_name: str | None = None,
                            progress_key: str | None = None,
                            stop_dict_name: str | None = None) -> bytes:
            from cloudposterior.remote.worker import run_sampling_blocking

            return run_sampling_blocking(
                f"/data/{payload_path}", sample_kwargs, nuts_sampler,
                progress_dict_name=progress_dict_name, progress_key=progress_key,
                stop_dict_name=stop_dict_name,
            )

        @modal.method()
        def sample_smc(self, payload_path: str, sample_kwargs: dict) -> bytes:
            from cloudposterior.remote.worker import run_smc

            return run_smc(f"/data/{payload_path}", sample_kwargs)

        @modal.method()
        def compute_log_likelihood(self, payload_path: str, idata_bytes: bytes,
                                   sample_kwargs: dict) -> bytes:
            from cloudposterior.remote.worker import run_compute_log_likelihood

            return run_compute_log_likelihood(
                f"/data/{payload_path}", idata_bytes, sample_kwargs
            )

    # Add dashboard web endpoints if requested
    dashboard_fn = None
    progress_fn = None
    stop_fn = None
    if dashboard_dict_name is not None:
        _dict_name = dashboard_dict_name
        # Full uuid hex, not a 6-char slice. These labels become the public
        # hostname, and the endpoints below serve posterior draws -- 24 bits
        # behind a guessable model-name prefix is not a secret.
        _uid = _endpoint_uid(dashboard_dict_name)
        _label_stem = f"{_truncate_label(model_label)}-{_uid}"
        _progress_label = f"{_label_stem}-progress"
        _stop_label = f"{_label_stem}-stop"
        _dash_label = _label_stem
        _stop_token = stop_token or ""

        # Stop endpoint backs both the dashboard's stop button and the
        # in-notebook stop button, so it exists whenever control infra does.
        @app.function(serialized=True, image=image)
        @modal.fastapi_endpoint(method="POST", label=_stop_label)
        async def serve_stop(token: str = "", model: str = ""):
            import hmac

            from fastapi.responses import JSONResponse
            import modal as _modal
            # Require the token baked into the page so a random caller who
            # finds the stop URL can't kill someone's sampling run.
            if not _stop_token or not hmac.compare_digest(token, _stop_token):
                return JSONResponse({"stopped": False, "error": "invalid token"}, status_code=403)
            try:
                # from_name is a lazy reference (no I/O); only get/put have .aio.
                d = _modal.Dict.from_name(_dict_name)
                # cp.map's per-model Stop targets "stop:<label>"; the global
                # "Stop all" (and single-model runs) use the bare "stop" key.
                await d.put.aio(f"stop:{model}" if model else "stop", True)
            except Exception as exc:
                # Reporting success here made a failed stop (deleted Dict,
                # network error) indistinguishable from a working one.
                return JSONResponse(
                    {"stopped": False, "error": str(exc)}, status_code=500
                )
            return JSONResponse({"stopped": True})

        stop_fn = serve_stop

        # Dashboard + progress endpoints only when the live dashboard is on.
        if dashboard:
            @app.function(serialized=True, image=image)
            @modal.fastapi_endpoint(method="GET", label=_dash_label)
            def serve_dashboard(token: str = ""):
                import hmac

                from fastapi.responses import HTMLResponse, JSONResponse
                from cloudposterior.dashboard import render_dashboard_html
                if not _stop_token or not hmac.compare_digest(token, _stop_token):
                    return JSONResponse({"error": "invalid token"}, status_code=403)
                return HTMLResponse(render_dashboard_html(
                    progress_label=_progress_label,
                    stop_label=_stop_label,
                    dashboard_label=_dash_label,
                    stop_token=_stop_token,
                ))

            @app.function(serialized=True, image=image)
            @modal.fastapi_endpoint(method="GET", label=_progress_label)
            async def serve_progress(token: str = ""):
                # Async handler + .aio() Modal calls: FastAPI runs this on an
                # event loop, so blocking calls here would warn and stall it.
                import hmac

                from fastapi.responses import JSONResponse
                import modal as _modal
                # This returns parameter names and raw posterior draws, so it
                # is gated on the same token as /stop.
                if not _stop_token or not hmac.compare_digest(token, _stop_token):
                    return JSONResponse({"error": "invalid token"}, status_code=403)
                default = {"phases": [], "sampling": None, "complete": False}
                try:
                    # from_name is a lazy reference (no I/O); only get/put have .aio.
                    d = _modal.Dict.from_name(_dict_name)
                    # cp.map writes a "models" manifest plus one state key per
                    # model; single-model runs write the flat "progress" state.
                    # Return {models, runs} for the former (the page normalizes
                    # the latter to N=1), so the dashboard has one data shape.
                    manifest = await d.get.aio("models", None)
                    if manifest:
                        runs = {}
                        for m in manifest:
                            runs[m["label"]] = await d.get.aio(m["label"], {})
                        data = {"models": manifest, "runs": runs}
                    else:
                        data = await d.get.aio("progress", default)
                except Exception:
                    data = default
                return JSONResponse(data)

            dashboard_fn = serve_dashboard
            progress_fn = serve_progress

    return app, Sampler, dashboard_fn, progress_fn, stop_fn


class PersistentModalSamplingJob(SamplingJob):
    """Sampling job that uses an already-provisioned Modal environment."""

    def __init__(
        self,
        sampler_cls,
        payload_path: str,
        sample_kwargs: dict,
        nuts_sampler: str,
        stop_dict_name: str | None = None,
    ):
        self._sampler_cls = sampler_cls
        self._payload_path = payload_path
        self._sample_kwargs = sample_kwargs
        self._nuts_sampler = nuts_sampler
        self._stop_dict_name = stop_dict_name
        self._idata_bytes: bytes | None = None
        self._events: list[ProgressEvent] = []
        self._streamed = False

    def stream_progress(self, output_widget=None) -> Iterator[ProgressEvent]:
        self._streamed = True
        try:
            sampler = self._sampler_cls()
            gen = sampler.sample.remote_gen(
                self._payload_path,
                self._sample_kwargs,
                self._nuts_sampler,
                stop_dict_name=self._stop_dict_name,
            )
        except Exception as exc:
            raise _handle_modal_error(exc)
        self._idata_bytes = yield from _stream_events(gen, self._events)

    def result(self):
        from cloudposterior.serialize import deserialize_inference_data

        if self._idata_bytes is None and not self._streamed:
            for _ in self.stream_progress():
                pass

        if self._idata_bytes is None:
            # See ModalSamplingJob.result: re-streaming would start a second
            # paid run.
            raise RuntimeError(
                "the sampling stream finished without producing a trace; "
                "refusing to re-run the remote job"
            )

        return deserialize_inference_data(self._idata_bytes)

    def cancel(self):
        pass


# Modal caps endpoint labels at 63 chars; a 32-char uid plus the "-progress"
# suffix leaves this much for the model name.
_MAX_MODEL_LABEL = 20


def _truncate_label(model_label: str) -> str:
    """Bound the model part of an endpoint label.

    model_slug is unbounded (an unnamed model yields e.g.
    "mu-tau-theta-sigma-plus3"); once the label exceeds Modal's limit the
    endpoint is rejected or silently renamed at provision time.
    """
    return (model_label[:_MAX_MODEL_LABEL].rstrip("-")) or "model"


def _endpoint_uid(dashboard_dict_name: str) -> str:
    """Full-entropy suffix for public endpoint labels."""
    import uuid

    stem = dashboard_dict_name.replace("cp-dash-", "")
    # from_name suffixes are already random; extend to a full uuid worth of
    # entropy so the hostname itself is not guessable.
    return f"{stem}{uuid.uuid4().hex}"[:32]


def _compute_payload_path(m_slug: str, model_bytes: bytes) -> str:
    """Compute the Volume path for a model payload.

    The model_bytes hash already captures observed-data identity (cloudpickle
    bundles the data into the model), so a separate data slug isn't needed.
    """
    from cloudposterior.naming import payload_hash

    p_hash = payload_hash(model_bytes)
    return f"{m_slug}/payload-{p_hash}.bin"


class ModalEnvironment(RemoteEnvironment):
    """A provisioned Modal environment with payloads in a Volume."""

    def __init__(self, app, sampler_cls, volume, project: str, model_slug: str,
                 dashboard_dict=None, dashboard_dict_name: str | None = None,
                 dashboard_fn=None, progress_fn=None, stop_fn=None,
                 stop_token: str | None = None, config=None):
        # The config this env was sized/built with. Kept-warm envs are reused
        # across runs (and across cp.map -> cp.cloud), and callers read it back
        # to decide reusability and to report resize drift -- so it must be set
        # at construction, not patched on by one of the two provisioning paths.
        self.config = config
        self._app = app
        self._sampler_cls = sampler_cls
        self._volume = volume
        self._project = project
        self._model_slug = model_slug
        self._dashboard_dict = dashboard_dict
        self._dashboard_dict_name = dashboard_dict_name
        self._dashboard_fn = dashboard_fn
        self._progress_fn = progress_fn
        self._stop_fn = stop_fn
        self._stop_token = stop_token
        self._dashboard_url: str | None = None
        self._progress_url: str | None = None
        self._stop_url: str | None = None
        self._exit_stack = contextlib.ExitStack()
        self._running = False
        self._uploaded_hashes: set[str] = set()

    def _ensure_running(self):
        if not self._running:
            try:
                # Enter app.run() off the event loop: its __enter__ is a blocking
                # Modal call that otherwise warns/stalls inside an async host
                # (marimo). Enter manually in a worker thread, then register the
                # context's __exit__ on the stack for teardown.
                cm = self._app.run()
                _run_blocking(cm.__enter__)
                self._exit_stack.push(cm)
            except Exception as exc:
                raise _handle_modal_error(exc)
            self._running = True

            # Capture dashboard URLs after app starts (off-loop: avoids async warnings)
            if self._dashboard_fn is not None:
                try:
                    url = _run_blocking(self._dashboard_fn.get_web_url)
                    # The page and its /progress feed are token-gated, so the
                    # shareable URL has to carry the token.
                    if url and self._stop_token:
                        url = f"{url}?token={self._stop_token}"
                    self._dashboard_url = url
                except Exception:
                    pass
            if self._progress_fn is not None:
                try:
                    self._progress_url = _run_blocking(self._progress_fn.get_web_url)
                except Exception:
                    pass

            if self._stop_fn is not None:
                try:
                    self._stop_url = _run_blocking(self._stop_fn.get_web_url)
                except Exception:
                    pass

    def _upload_if_needed(self, model_bytes: bytes, payload_path: str) -> bool:
        """Upload model payload to Volume if not already there. Returns True if uploaded."""
        from cloudposterior.naming import payload_hash

        p_hash = payload_hash(model_bytes)
        if p_hash in self._uploaded_hashes:
            return False

        # Check Volume (off-loop: avoids async warnings in marimo/async hosts)
        try:
            dir_path = "/".join(payload_path.split("/")[:-1])
            entries = _run_blocking(self._volume.listdir, f"/{dir_path}")
            filename = payload_path.split("/")[-1]
            # TODO(e2e): confirm whether Volume.listdir entries carry bare
            # filenames or dir-prefixed paths; comparing basenames is correct
            # under either shape.
            if any(e.path.split("/")[-1] == filename for e in entries):
                self._uploaded_hashes.add(p_hash)
                return False
        except Exception:
            pass

        # Upload with force=True to overwrite if already exists
        import os

        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(model_bytes)
            tmp_path = tmp.name

        def _do_upload():
            with self._volume.batch_upload(force=True) as upload:
                upload.put_file(tmp_path, f"/{payload_path}")

        try:
            _run_blocking(_do_upload)
        finally:
            os.unlink(tmp_path)

        self._uploaded_hashes.add(p_hash)
        self._prune_old_payloads(payload_path)
        return True

    def _prune_old_payloads(self, payload_path: str) -> None:
        """Best-effort LRU prune: keep the N most recent payload-*.bin files
        in the same directory as ``payload_path``. Without this, every model
        edit accumulates a new payload until the user runs cleanup_volumes().

        ``payload_path`` is the payload that was *just* uploaded and is never a
        prune candidate: volume mtimes have one-second granularity, so a tie
        could otherwise sort the fresh payload into the stale tail and delete
        the file the next submit is about to load.
        """
        import logging

        try:
            dir_path = "/".join(payload_path.split("/")[:-1])
            entries = list(_run_blocking(self._volume.listdir, f"/{dir_path}"))
        except Exception as exc:
            logging.getLogger(__name__).debug("listdir failed during prune: %s", exc)
            return

        # Compare basenames: modal's FileEntry.path is the full path from the
        # volume root, so strip the directory before matching.
        keep_name = payload_path.split("/")[-1]
        all_payloads = [
            e for e in entries
            if e.path.split("/")[-1].startswith("payload-") and e.path.endswith(".bin")
        ]
        payloads = [e for e in all_payloads if e.path.split("/")[-1] != keep_name]
        # When the kept payload is in the listing it occupies one of the N slots.
        budget = _PAYLOAD_KEEP_PER_MODEL - (len(all_payloads) - len(payloads))
        if len(payloads) <= budget:
            return

        payloads.sort(key=lambda e: getattr(e, "mtime", 0), reverse=True)
        for stale in payloads[budget:]:
            try:
                name = stale.path.split("/")[-1]
                _run_blocking(self._volume.remove_file, f"/{dir_path}/{name}")
            except Exception as exc:
                logging.getLogger(__name__).debug(
                    "failed to prune %s: %s", stale.path, exc,
                )

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
            stop_dict_name=self._dashboard_dict_name,
        )

    def teardown(self) -> None:
        # Off the event loop: app.run().__exit__ is a blocking Modal call.
        try:
            _run_blocking(self._exit_stack.close)
        except Exception:
            pass
        # The control Dict is a named, persistent Modal object created fresh
        # per provision -- delete it or orphans accumulate in the workspace.
        if self._dashboard_dict_name:
            try:
                import modal

                _run_blocking(modal.Dict.objects.delete, self._dashboard_dict_name)
            except Exception:
                pass
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
        idle_timeout: int = 1200,
        dashboard: bool = False,
        stop_enabled: bool = False,
        nuts_sampler: str = "pymc",
    ) -> ModalEnvironment:
        """Provision a persistent environment (no upload -- deferred to first cache miss).

        ``stop_enabled`` provisions the control Dict + token-gated ``/stop``
        endpoint for the in-notebook stop button even when the full ``dashboard``
        is off.
        """
        import dataclasses
        import modal
        import uuid

        from cloudposterior.naming import model_slug as compute_model_slug

        # Copy rather than mutate: the caller owns its RemoteConfig, and the
        # copy is what gets stored on the env as the as-provisioned record.
        config = dataclasses.replace(config, idle_timeout=idle_timeout)
        volume_name = f"cp-{project}"
        try:
            volume = _run_blocking(modal.Volume.from_name, volume_name, create_if_missing=True)
        except Exception as exc:
            raise _handle_modal_error(exc)

        m_slug = compute_model_slug(model)

        # Control infra (a Modal Dict + secret token) backs both the live
        # dashboard and the in-notebook stop button; create it for either.
        dashboard_dict = None
        dashboard_dict_name = None
        stop_token = None
        if dashboard or stop_enabled:
            dashboard_dict_name = f"cp-dash-{uuid.uuid4().hex[:8]}"
            dashboard_dict = _run_blocking(
                modal.Dict.from_name, dashboard_dict_name, create_if_missing=True
            )
            stop_token = uuid.uuid4().hex  # secret, baked into the page only

        app, sampler_cls, dashboard_fn, progress_fn, stop_fn = _create_persistent_app(
            version_manifest, config, volume,
            dashboard_dict_name=dashboard_dict_name,
            dashboard=dashboard,
            model_label=m_slug.replace("_", "-"),
            stop_token=stop_token,
            nuts_sampler=nuts_sampler,
        )
        return ModalEnvironment(
            app, sampler_cls, volume, project, m_slug,
            dashboard_dict=dashboard_dict,
            dashboard_dict_name=dashboard_dict_name,
            dashboard_fn=dashboard_fn,
            progress_fn=progress_fn,
            stop_fn=stop_fn,
            stop_token=stop_token,
            config=config,
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
