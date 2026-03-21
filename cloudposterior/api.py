"""Public API for cloudposterior."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Iterator

import arviz as az

from cloudposterior.backends import SamplingJob
from cloudposterior.config import RemoteConfig
from cloudposterior.progress import JobPhase, PhaseUpdate, ProgressEvent, SamplingProgress
from cloudposterior.serialize import create_payload, payload_size_mb

if TYPE_CHECKING:
    import pymc as pm


class wrap:
    """Context manager that intercepts PyMC operations with remote execution,
    caching, and notifications.

    Usage::

        with pd.wrap(model):
            idata = pm.sample(draws=2000, chains=4)  # local with in-memory caching

        with pd.wrap(model, remote=True):             # run on Modal VM
        with pd.wrap(model, cache="disk"):            # persistent disk cache
        with pd.wrap(model, cache="/path/to/cache"):  # custom cache directory
        with pd.wrap(model, cache=False):             # disable caching
        with pd.wrap(model, notify=True):                          # ntfy.sh, auto topic
        with pd.wrap(model, notify="my-topic"):                    # ntfy.sh, custom topic
        with pd.wrap(model, notify={"server": "https://ntfy.example.com"}):  # self-hosted
        with pd.wrap(model, notify={"server": "https://ntfy.example.com", "topic": "my-topic"}):
    """

    def __init__(
        self,
        model: pm.Model,
        *,
        remote: bool = False,
        cache: bool = True,
        notify: bool | str | dict = False,
        instance: str | None = None,
        nuts_sampler: str = "pymc",
        progress: bool = True,
    ):
        self.model = model
        self.remote = remote
        self.cache = cache
        self.notify = notify
        self.instance = instance
        self.nuts_sampler = nuts_sampler
        self.progress = progress
        self._originals: dict[str, object] = {}

    def __enter__(self):
        import pymc as pm

        self._originals["sample"] = pm.sample
        pm.sample = self._make_intercepted_sample()
        self.model.__enter__()
        return self.model

    def __exit__(self, *exc):
        import pymc as pm

        pm.sample = self._originals["sample"]
        return self.model.__exit__(*exc)

    def _make_intercepted_sample(self):
        ctx = self

        def intercepted_sample(**kwargs):
            return _run_sample(
                model=ctx.model,
                remote=ctx.remote,
                cache=ctx.cache,
                notify=ctx.notify,
                instance=ctx.instance,
                nuts_sampler=ctx.nuts_sampler,
                progress=ctx.progress,
                original_sample=ctx._originals["sample"],
                **kwargs,
            )

        return intercepted_sample


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
    **sample_kwargs,
) -> az.InferenceData:
    """Core sampling logic with cache, remote, and notification support."""
    from cloudposterior.cache import compute_cache_key, resolve_cache
    from cloudposterior.serialize import serialize_model, serialize_observed_data

    # -- Resolve resource config (auto-size or preset) --
    config = RemoteConfig.from_instance(instance, model=model, sample_kwargs=sample_kwargs)
    if remote:
        instance_desc = f"Modal ({config.describe()})"
    else:
        instance_desc = "local"

    # -- Build sinks --
    sinks = _build_sinks(
        progress=progress,
        notify=notify,
        instance_desc=instance_desc,
        model=model,
    )

    def emit(event):
        for sink in sinks:
            if isinstance(event, PhaseUpdate):
                sink.show_phase(event)
            elif isinstance(event, SamplingProgress):
                sink.show_sampling(event)

    # -- Serialize for cache key --
    serial_start = time.time()
    model_bytes = serialize_model(model)
    data_bytes = serialize_observed_data(model)
    emit(PhaseUpdate(
        phase=JobPhase.SERIALIZING,
        status="done",
        message="model + data packaged",
        elapsed=time.time() - serial_start,
    ))

    # -- Resolve cache backend --
    cache_backend = resolve_cache(cache, model=model)
    cache_key = None

    if cache_backend is not None:
        cache_key = compute_cache_key(model_bytes, data_bytes, sample_kwargs)
        cached = cache_backend.load(cache_key, data_bytes=data_bytes, sample_kwargs=sample_kwargs)
        if cached is not None:
            emit(PhaseUpdate(
                phase=JobPhase.CACHE_HIT,
                status="done",
                message="returning cached result",
                elapsed=0.0,
            ))
            _stop_sinks(sinks)
            return cached

    # -- Run sampling --
    if remote:
        idata = _run_remote(
            model=model,
            model_bytes=model_bytes,
            data_bytes=data_bytes,
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
        cache_backend.save(cache_key, idata, data_bytes=data_bytes, sample_kwargs=sample_kwargs)

    _stop_sinks(sinks)
    return idata


def _build_sinks(*, progress: bool, notify, instance_desc: str, model=None) -> list:
    """Create display + notification sinks."""
    sinks = []

    if progress:
        from cloudposterior.display import _is_notebook, NotebookDisplay, TerminalDisplay

        if _is_notebook():
            display = NotebookDisplay(instance_desc)
        else:
            display = TerminalDisplay(instance_desc)
            display.start()
        sinks.append(display)

    if notify:
        from cloudposterior.notify import NtfyNotifier

        if isinstance(notify, dict):
            topic = notify.get("topic")
            server = notify.get("server")
        elif isinstance(notify, str):
            topic = notify
            server = None
        else:
            topic = None
            server = None

        auto_generated = topic is None
        notifier = NtfyNotifier(
            topic=topic,
            server=server,
            model=model,
            instance_desc=instance_desc,
        )
        sinks.append(notifier)

        if progress:
            _show_ntfy_link(notifier.url, show_qr=auto_generated)

    return sinks


def _show_ntfy_link(url: str, show_qr: bool = False):
    """Display the ntfy notification URL, with optional QR code."""
    from cloudposterior.display import _is_notebook

    if _is_notebook():
        from IPython.display import display as ipy_display, HTML

        parts = [
            f'<div style="font-family:monospace;font-size:12px;padding:4px 0;">',
            f'Notifications: <a href="{url}" target="_blank">{url}</a>',
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
        console.print(f"[dim]Notifications: {url}[/dim]")
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
    data_bytes: bytes,
    config: RemoteConfig,
    nuts_sampler: str,
    sinks: list,
    emit,
    **sample_kwargs,
) -> az.InferenceData:
    """Run sampling on a remote Modal VM."""
    from cloudposterior.backends.modal_backend import ModalBackend
    from cloudposterior.display import NotebookDisplay
    from cloudposterior.serialize import SamplingPayload, get_version_manifest

    payload = SamplingPayload(
        model_bytes=model_bytes,
        data_bytes=data_bytes,
        version_manifest=get_version_manifest(),
        sample_kwargs=sample_kwargs,
    )

    backend = ModalBackend(config=config, nuts_sampler=nuts_sampler)
    job = backend.submit(payload)

    # Get output widget from notebook display if present
    output_widget = None
    for sink in sinks:
        if isinstance(sink, NotebookDisplay):
            output_widget = sink.modal_output_widget
            break

    # Stream with upload/download phases
    emit(PhaseUpdate(
        phase=JobPhase.UPLOADING,
        status="in_progress",
        message="sending to Modal",
        elapsed=0.0,
    ))

    upload_start = time.time()
    first_event = True

    for event in job.stream_progress(output_widget=output_widget):
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


def _run_local(
    *,
    model,
    original_sample,
    sinks: list,
    emit,
    **sample_kwargs,
) -> az.InferenceData:
    """Run sampling locally using the original pm.sample."""
    emit(PhaseUpdate(
        phase=JobPhase.SAMPLING,
        status="in_progress",
        message="local sampling started",
        elapsed=0.0,
    ))

    sample_start = time.time()
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
    """Run PyMC sampling on a remote cloud VM.

    Drop-in replacement for pm.sample() that runs on Modal.
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


def submit(
    model: pm.Model,
    *,
    draws: int = 1000,
    tune: int = 1000,
    chains: int | None = None,
    cores: int | None = None,
    nuts_sampler: str = "pymc",
    instance: str | None = None,
    **pm_sample_kwargs,
) -> SamplingJob:
    """Submit a sampling job to a remote cloud VM without blocking."""
    from cloudposterior.backends.modal_backend import ModalBackend

    sample_kwargs = {
        "draws": draws,
        "tune": tune,
        **pm_sample_kwargs,
    }
    if chains is not None:
        sample_kwargs["chains"] = chains
    if cores is not None:
        sample_kwargs["cores"] = cores

    payload = create_payload(model, sample_kwargs)
    config = RemoteConfig.from_instance(instance, model=model, sample_kwargs=sample_kwargs)
    backend = ModalBackend(config=config, nuts_sampler=nuts_sampler)
    return backend.submit(payload)
