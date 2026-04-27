"""Public API for cloudposterior."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import arviz as az

from cloudposterior.config import RemoteConfig
from cloudposterior.progress import JobPhase, PhaseUpdate, SamplingProgress

if TYPE_CHECKING:
    import pymc as pm


def _detect_project_name() -> str:
    """Detect a project name from the environment.

    Tries (in order):
    1. Notebook filename (VS Code sets __vsc_ipynb_file__)
    2. Current working directory basename
    """
    import os
    from pathlib import Path

    # VS Code notebook
    vsc_file = os.environ.get("__vsc_ipynb_file__")
    if vsc_file:
        return Path(vsc_file).stem

    # Fall back to cwd basename
    return Path.cwd().name


class cloud:
    """Context manager that intercepts PyMC operations with remote execution,
    caching, live dashboard, and push notifications.

    Remote containers stay warm for 20 minutes after the last run.

    Usage::

        with cp.cloud(model, remote=True):               # cloud + live dashboard
        with cp.cloud(model, remote=True, notify=True):   # cloud + dashboard + ntfy
        with cp.cloud(model, remote=True, dashboard=False): # cloud, no dashboard
        with cp.cloud(model, cache="disk"):               # local + disk cache
        with cp.cloud(model, notify=True):                # local + ntfy notifications
    """

    def __init__(
        self,
        model: pm.Model,
        *,
        remote: bool = False,
        cache: bool | str = True,
        dashboard: bool | None = None,
        notify: bool | str | dict = False,
        instance: str | None = None,
        progress: bool = True,
        project: str | None = None,
    ):
        # dashboard=None means "default": on for remote, off for local with no
        # warning. dashboard=True with remote=False is a user mistake -- warn.
        if dashboard is True and not remote:
            import warnings

            warnings.warn(
                "dashboard=True has no effect without remote=True; the live "
                "dashboard requires cloud execution. Pass remote=True or omit "
                "dashboard.",
                stacklevel=2,
            )
        if dashboard is None:
            dashboard = True
        self.model = model
        self.remote = remote
        self.cache = cache
        self.dashboard = dashboard and remote  # dashboard only works with remote
        self.notify = notify
        self.instance = instance
        self.progress = progress
        self.project = project or _detect_project_name()
        self._originals: dict[str, object] = {}
        self._env = None
        self._model_bytes: bytes | None = None

    def __enter__(self):
        import pymc as pm

        self._originals["sample"] = pm.sample
        pm.sample = self._make_intercepted_sample()
        self.model.__enter__()
        return self.model

    def __exit__(self, *exc):
        import pymc as pm

        pm.sample = self._originals["sample"]
        if self._env is not None:
            self._env.teardown()
            self._env = None
        return self.model.__exit__(*exc)

    def _provision_environment(self, nuts_sampler: str, sample_kwargs: dict):
        from cloudposterior.backends.modal_backend import ModalBackend
        from cloudposterior.serialize import get_version_manifest

        config = RemoteConfig.from_instance(
            self.instance, model=self.model, sample_kwargs=sample_kwargs,
            nuts_sampler=nuts_sampler,
        )
        manifest = get_version_manifest()
        backend = ModalBackend(config=config)
        self._env = backend.provision(
            self._model_bytes, self.model, manifest, config,
            project=self.project, idle_timeout=config.idle_timeout,
            dashboard=self.dashboard,
        )
        # Stash the resolved config so the displayed instance_desc matches
        # what was actually provisioned (no recomputation drift).
        self._env.config = config

    def destroy(self):
        """Tear down the environment and clean up the project volume.

        Call after the ``with`` block to immediately stop the container
        and delete the project's volume::

            session = cp.cloud(model, remote=True)
            with session:
                idata = pm.sample(draws=2000)
            session.destroy()
        """
        if self._env is not None:
            self._env.teardown()
            self._env = None
        from cloudposterior.backends.modal_backend import ModalBackend
        ModalBackend.cleanup_volumes(project=self.project)

    def _make_intercepted_sample(self):
        ctx = self

        def intercepted_sample(**kwargs):
            nuts_sampler = kwargs.pop("nuts_sampler", "pymc")

            # Lazy first-touch serialization: pay the cloudpickle cost only
            # when we actually need it. Memoize on the model so repeat calls
            # in the same session don't re-serialize.
            if ctx._model_bytes is None:
                ctx._model_bytes = _ensure_model_bytes(ctx.model)

            if ctx.remote and ctx._env is None:
                # First sample call -- provision sized to these kwargs.
                ctx._provision_environment(nuts_sampler, kwargs)
            elif ctx._env is not None:
                # Later call -- warn if auto-sizing would have picked
                # different resources (first call wins; container is fixed).
                _warn_if_resize_drift(ctx, nuts_sampler, kwargs)

            if ctx._env is not None:
                return _run_sample_persistent(
                    model=ctx.model,
                    env=ctx._env,
                    model_bytes=ctx._model_bytes,
                    cache=ctx.cache,
                    dashboard=ctx.dashboard,
                    notify=ctx.notify,
                    nuts_sampler=nuts_sampler,
                    progress=ctx.progress,
                    **kwargs,
                )
            return _run_sample(
                model=ctx.model,
                remote=ctx.remote,
                cache=ctx.cache,
                notify=ctx.notify,
                instance=ctx.instance,
                nuts_sampler=nuts_sampler,
                progress=ctx.progress,
                original_sample=ctx._originals["sample"],
                model_bytes=ctx._model_bytes,
                **kwargs,
            )

        return intercepted_sample


def _ensure_model_bytes(model) -> bytes:
    """Serialize a model once and memoize the bytes on the model object.

    pm.sample() mutates the model in place (compiled functions, etc.), which
    would change the cloudpickle output. Serializing once before the first
    sample and reusing across calls keeps the cache key stable.
    """
    from cloudposterior.serialize import serialize_model

    if not hasattr(model, "_cp_model_bytes"):
        model._cp_model_bytes = serialize_model(model)
    return model._cp_model_bytes


def _warn_if_resize_drift(ctx, nuts_sampler: str, sample_kwargs: dict) -> None:
    """If a later pm.sample() call would auto-size to a different VM than
    what was provisioned on the first call, warn the user.

    Container sizing is fixed at provision time (first sample call). Subsequent
    calls reuse the same VM. If the user changes chains/draws to something the
    auto-sizer would have provisioned differently, they should know.
    """
    if ctx.instance is not None:
        # Explicit preset -- user made the choice; no auto-sizing happened.
        return
    provisioned = getattr(ctx._env, "config", None)
    if provisioned is None:
        return
    import warnings

    would_be = RemoteConfig.from_instance(
        None, model=ctx.model, sample_kwargs=sample_kwargs, nuts_sampler=nuts_sampler,
    )
    if (would_be.cpu, would_be.memory, would_be.gpu) != (
        provisioned.cpu, provisioned.memory, provisioned.gpu,
    ):
        warnings.warn(
            f"pm.sample kwargs would auto-size to {would_be.describe()}, but "
            f"the container was already provisioned as {provisioned.describe()} "
            f"on the first call. Container size is fixed for the duration of "
            f"the cp.cloud(...) block. Use a new cp.cloud() block to resize.",
            stacklevel=3,
        )


def _run_sample(
    model,
    *,
    remote: bool,
    cache: bool,
    notify: bool | str,
    instance: str | None,
    nuts_sampler: str,
    progress: bool,
    original_sample,
    model_bytes: bytes | None = None,
    **sample_kwargs,
) -> az.InferenceData:
    """Core sampling logic with cache, remote, and notification support."""
    from cloudposterior.cache import resolve_cache
    from cloudposterior.naming import cache_key as compute_cache_key

    if model_bytes is None:
        model_bytes = _ensure_model_bytes(model)

    # -- Check cache (include nuts_sampler so different samplers don't collide) --
    cache_kwargs = {**sample_kwargs, "nuts_sampler": nuts_sampler}
    cache_backend = resolve_cache(cache, model=model)
    cache_key = None

    if cache_backend is not None:
        cache_key = compute_cache_key(model_bytes, cache_kwargs)
        cached = cache_backend.load(cache_key, sample_kwargs=cache_kwargs)
        if cached is not None:
            if progress:
                from cloudposterior.display import _is_notebook
                if _is_notebook():
                    from IPython.display import display, HTML
                    display(HTML(
                        '<div style="font-family:monospace;font-size:13px;color:#888;padding:2px 0;">'
                        '<span style="color:#5cb85c;">&#10003;</span> cached result'
                        '</div>'
                    ))
                else:
                    from rich.console import Console
                    Console().print("[green]\u2713[/green] [dim]cached result[/dim]")
            return cached

    # -- Build sinks (only needed for cache miss) --
    if remote:
        config = RemoteConfig.from_instance(instance, model=model, sample_kwargs=sample_kwargs, nuts_sampler=nuts_sampler)
        instance_desc = f"Modal ({config.describe()})"
    else:
        instance_desc = "local"

    # For local runs, skip progress display -- let PyMC show its native output.
    if remote:
        sinks = _build_sinks(
            progress=progress,
            notify=notify,
            instance_desc=instance_desc,
            model=model,
        )
    else:
        sinks = _build_sinks(
            progress=False,
            notify=notify,
            instance_desc=instance_desc,
            model=model,
        )

    def emit(event):
        from cloudposterior.progress import ConvergenceUpdate
        for sink in sinks:
            if isinstance(event, PhaseUpdate):
                sink.show_phase(event)
            elif isinstance(event, SamplingProgress):
                sink.show_sampling(event)
            elif isinstance(event, ConvergenceUpdate) and hasattr(sink, "show_convergence"):
                sink.show_convergence(event)

    # -- Run sampling --
    if remote:
        idata = _run_remote(
            model=model,
            model_bytes=model_bytes,
            config=config,
            nuts_sampler=nuts_sampler,
            sinks=sinks,
            emit=emit,
            **sample_kwargs,
        )
    else:
        idata = _run_local(
            model=model,
            original_sample=original_sample,
            sinks=sinks,
            emit=emit,
            **sample_kwargs,
        )

    # -- Cache store --
    if cache_backend is not None and cache_key:
        cache_backend.save(cache_key, idata, sample_kwargs=cache_kwargs)

    _stop_sinks(sinks)
    return idata


_NOTIFY_DICT_KEYS = {"topic", "server"}


def _parse_notify(notify) -> tuple[str | None, str | None]:
    """Resolve the ``notify`` kwarg into ``(topic, server)``.

    Accepts ``True`` (auto-topic), a topic string, or
    ``{"topic": ..., "server": ...}``. Rejects unknown dict keys and unknown
    types so typos like ``notify={"channel": "x"}`` fail loudly.
    """
    if notify is True:
        return None, None
    if isinstance(notify, str):
        return notify, None
    if isinstance(notify, dict):
        extras = set(notify) - _NOTIFY_DICT_KEYS
        if extras:
            raise ValueError(
                f"notify dict accepts keys {sorted(_NOTIFY_DICT_KEYS)}; "
                f"got unexpected keys: {sorted(extras)}"
            )
        return notify.get("topic"), notify.get("server")
    raise TypeError(
        f"notify must be bool | str | dict; got {type(notify).__name__}={notify!r}"
    )


def _build_sinks(*, progress: bool, dashboard: bool = False, notify=False,
                 instance_desc: str, model=None, dashboard_dict=None) -> list:
    """Create display, dashboard, and notification sinks."""
    sinks = []

    if progress:
        from cloudposterior.display import _is_notebook, NotebookDisplay, TerminalDisplay

        if _is_notebook():
            display = NotebookDisplay(instance_desc)
        else:
            display = TerminalDisplay(instance_desc)
            display.start()
        sinks.append(display)

    # Live dashboard (convergence, traces, stop button)
    if dashboard and dashboard_dict is not None:
        from cloudposterior.dashboard import DashboardSink
        sinks.append(DashboardSink(dashboard_dict))

    # Push notifications (ntfy)
    if notify:
        from cloudposterior.notify import NtfyNotifier

        topic, server = _parse_notify(notify)

        auto_generated = topic is None
        notifier = NtfyNotifier(
            topic=topic,
            server=server,
            model=model,
            instance_desc=instance_desc,
        )
        sinks.append(notifier)

        _show_link(notifier.url, label="Notifications", show_qr=auto_generated)

    return sinks


def _show_link(url: str, label: str = "Link", show_qr: bool = False):
    """Display a URL with optional QR code."""
    from cloudposterior.display import _is_notebook

    if _is_notebook():
        from IPython.display import display as ipy_display, HTML

        parts = [
            f'<div style="font-family:monospace;font-size:12px;padding:4px 0;">',
            f'{label}: <a href="{url}" target="_blank">{url}</a>',
        ]
        if show_qr:
            try:
                import qrcode
                import qrcode.image.svg
                import io

                qr = qrcode.make(url, image_factory=qrcode.image.svg.SvgPathImage, box_size=6)
                buf = io.BytesIO()
                qr.save(buf)
                svg = buf.getvalue().decode("utf-8")
                parts.append(f'<div style="padding:4px 0;">{svg}</div>')
            except Exception:
                pass
        parts.append("</div>")
        ipy_display(HTML("".join(parts)))
    else:
        from rich.console import Console

        console = Console()
        console.print(f"[dim]{label}: {url}[/dim]")
        if show_qr:
            try:
                import qrcode

                qr = qrcode.QRCode(border=1)
                qr.add_data(url)
                qr.make(fit=True)
                qr.print_ascii(out=None)  # prints to stdout
            except Exception:
                pass


def _stop_sinks(sinks: list):
    for sink in sinks:
        if hasattr(sink, "stop"):
            sink.stop()


def _run_remote(
    *,
    model,
    model_bytes: bytes,
    config: RemoteConfig,
    nuts_sampler: str,
    sinks: list,
    emit,
    **sample_kwargs,
) -> az.InferenceData:
    """Run sampling on a remote Modal VM."""
    from cloudposterior.backends.modal_backend import ModalBackend
    from cloudposterior.serialize import SamplingPayload, get_version_manifest

    payload = SamplingPayload(
        model_bytes=model_bytes,
        version_manifest=get_version_manifest(),
        sample_kwargs=sample_kwargs,
    )

    backend = ModalBackend(config=config, nuts_sampler=nuts_sampler)
    job = backend.submit(payload)

    # Stream with upload/download phases
    emit(PhaseUpdate(
        phase=JobPhase.UPLOADING,
        status="in_progress",
        message="sending to Modal",
        elapsed=0.0,
    ))

    upload_start = time.time()
    first_event = True

    for event in job.stream_progress():
        if first_event:
            emit(PhaseUpdate(
                phase=JobPhase.UPLOADING,
                status="done",
                message="payload uploaded",
                elapsed=time.time() - upload_start,
            ))
            first_event = False
        emit(event)

    emit(PhaseUpdate(
        phase=JobPhase.DOWNLOADING,
        status="done",
        message="trace received",
        elapsed=0.0,
    ))

    return job.result()


def _run_sample_persistent(
    model,
    *,
    env,
    model_bytes: bytes,
    cache: bool,
    dashboard: bool = False,
    notify: bool | str | dict = False,
    nuts_sampler: str = "pymc",
    progress: bool,
    **sample_kwargs,
) -> az.InferenceData:
    """Sampling via a persistent environment.

    Model payload is in the Volume. Per-call sends only kwargs + a path
    identifying which payload to load. Volume upload is deferred until
    after cache check (no upload needed on cache hit).
    """
    from cloudposterior.backends.modal_backend import _compute_payload_path
    from cloudposterior.cache import resolve_cache
    from cloudposterior.naming import cache_key as compute_cache_key

    # Cache check -- include nuts_sampler in key so different samplers don't collide
    cache_kwargs = {**sample_kwargs, "nuts_sampler": nuts_sampler}

    cache_backend = resolve_cache(cache, model=model)
    cache_key = None
    if cache_backend is not None:
        cache_key = compute_cache_key(model_bytes, cache_kwargs)
        cached = cache_backend.load(cache_key, sample_kwargs=cache_kwargs)
        if cached is not None:
            if progress:
                from cloudposterior.display import _is_notebook
                if _is_notebook():
                    from IPython.display import display, HTML
                    display(HTML(
                        '<div style="font-family:monospace;font-size:13px;color:#888;padding:2px 0;">'
                        '<span style="color:#5cb85c;">&#10003;</span> cached result'
                        '</div>'
                    ))
                else:
                    from rich.console import Console
                    Console().print("[green]\u2713[/green] [dim]cached result[/dim]")
            return cached

    # Cache miss -- read the actually-provisioned config from the env so the
    # displayed instance description matches what's running (no recomputation drift).
    config = env.config
    instance_desc = f"Modal ({config.describe()})"

    sinks = _build_sinks(
        progress=progress,
        dashboard=dashboard,
        notify=notify,
        instance_desc=instance_desc,
        model=model,
        dashboard_dict=getattr(env, "_dashboard_dict", None),
    )

    # Start the app early if dashboard is requested, so we can show the URL
    if dashboard and env._dashboard_fn is not None:
        env._ensure_running()
        dashboard_url = env._dashboard_url
        if dashboard_url:
            # Ensure URL ends with / so the ASGI app serves from root
            if not dashboard_url.endswith("/"):
                dashboard_url += "/"
            _show_link(dashboard_url, label="Dashboard", show_qr=True)

    def emit(event):
        from cloudposterior.progress import ConvergenceUpdate
        for sink in sinks:
            if isinstance(event, PhaseUpdate):
                sink.show_phase(event)
            elif isinstance(event, SamplingProgress):
                sink.show_sampling(event)
            elif isinstance(event, ConvergenceUpdate) and hasattr(sink, "show_convergence"):
                sink.show_convergence(event)

    # Upload payload to Volume if needed
    payload_path = _compute_payload_path(env._model_slug, model_bytes)

    payload_mb = len(model_bytes) / (1024 * 1024)
    emit(PhaseUpdate(
        phase=JobPhase.DATA_UPLOADING,
        status="in_progress",
        message=f"uploading to volume ({payload_mb:.1f} MB)",
        elapsed=0.0,
    ))
    upload_start = time.time()
    uploaded = env._upload_if_needed(model_bytes, payload_path)
    if uploaded:
        emit(PhaseUpdate(
            phase=JobPhase.DATA_UPLOADING,
            status="done",
            message="uploaded to volume",
            elapsed=time.time() - upload_start,
        ))
    else:
        emit(PhaseUpdate(
            phase=JobPhase.DATA_UPLOADING,
            status="done",
            message="volume up to date",
            elapsed=time.time() - upload_start,
        ))

    # Submit to container (env no longer uploads -- we already did)
    job = env.submit(model_bytes, sample_kwargs, nuts_sampler, payload_path=payload_path)

    emit(PhaseUpdate(
        phase=JobPhase.PROVISIONING,
        status="in_progress",
        message="provisioning container",
        elapsed=0.0,
    ))

    provision_start = time.time()
    first_event = True
    download_start = None
    for event in job.stream_progress():
        if first_event:
            emit(PhaseUpdate(
                phase=JobPhase.PROVISIONING,
                status="done",
                message="container ready",
                elapsed=time.time() - provision_start,
            ))
            first_event = False
        emit(event)
        # Start download timer when sampling completes (remote compression + transfer follows)
        if isinstance(event, PhaseUpdate) and event.phase == JobPhase.SAMPLING and event.status == "done":
            download_start = time.time()
            emit(PhaseUpdate(
                phase=JobPhase.DOWNLOADING,
                status="in_progress",
                message="compressing and transferring trace",
                elapsed=0.0,
            ))

    # result() does local lz4 decompression + netcdf parsing
    idata = job.result()

    emit(PhaseUpdate(
        phase=JobPhase.DOWNLOADING,
        status="done",
        message="trace loaded",
        elapsed=time.time() - (download_start or time.time()),
    ))

    if cache_backend is not None and cache_key:
        cache_backend.save(cache_key, idata, sample_kwargs=cache_kwargs)

    _stop_sinks(sinks)
    return idata


def _run_local(
    *,
    model,
    original_sample,
    sinks: list,
    emit,
    **sample_kwargs,
) -> az.InferenceData:
    """Run sampling locally using the original pm.sample."""
    from queue import Queue
    from threading import Thread

    from cloudposterior.progress import ProgressAggregator, make_sampling_callback

    emit(PhaseUpdate(
        phase=JobPhase.SAMPLING,
        status="in_progress",
        message="local sampling started",
        elapsed=0.0,
    ))

    sample_start = time.time()

    # If we have sinks (notifications), inject a progress callback
    if sinks:
        tune = sample_kwargs.get("tune", 1000)
        draws = sample_kwargs.get("draws", 1000)
        progress_queue: Queue = Queue()
        callback = make_sampling_callback(progress_queue, tune, draws)
        aggregator = ProgressAggregator(progress_queue)

        def stream_progress():
            for snapshot in aggregator.snapshots():
                emit(snapshot)

        progress_thread = Thread(target=stream_progress, daemon=True)
        progress_thread.start()

        with model:
            idata = original_sample(
                callback=callback,
                **sample_kwargs,
            )

        aggregator.stop()
        progress_thread.join(timeout=2)
    else:
        with model:
            idata = original_sample(**sample_kwargs)

    emit(PhaseUpdate(
        phase=JobPhase.SAMPLING,
        status="done",
        message="sampling complete",
        elapsed=time.time() - sample_start,
    ))

    return idata


# -- Explicit API (backwards-compatible) --

def sample(
    model: pm.Model,
    *,
    draws: int = 1000,
    tune: int = 1000,
    chains: int | None = None,
    cores: int | None = None,
    nuts_sampler: str = "pymc",
    instance: str | None = None,
    progress: bool = True,
    cache: bool = True,
    notify: bool | str | dict = False,
    **pm_sample_kwargs,
) -> az.InferenceData:
    """Run a single remote PyMC sampling job on Modal.

    For repeated sampling with the same model, use the ``cp.cloud()`` context
    manager instead -- it keeps the container warm and only ships sample
    kwargs after the first call.
    """
    import pymc as pm

    return _run_sample(
        model=model,
        remote=True,
        cache=cache,
        notify=notify,
        instance=instance,
        nuts_sampler=nuts_sampler,
        progress=progress,
        original_sample=pm.sample,
        draws=draws,
        tune=tune,
        **({"chains": chains} if chains is not None else {}),
        **({"cores": cores} if cores is not None else {}),
        **pm_sample_kwargs,
    )


def cleanup_volumes(project: str | None = None) -> None:
    """Delete the Modal Volume for a single project.

    Defaults to the current project (auto-detected from notebook filename or
    working directory). Pass ``project="..."`` to target a specific one.

    Examples::

        cp.cleanup_volumes()                        # delete the current project's volume
        cp.cleanup_volumes(project="my-research")   # delete a specific project's volume
    """
    from cloudposterior.backends.modal_backend import ModalBackend

    ModalBackend.cleanup_volumes(project=project or _detect_project_name())
