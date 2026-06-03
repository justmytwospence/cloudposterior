"""Public API for cloudposterior."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import arviz as az

from cloudposterior.config import RemoteConfig
from cloudposterior.progress import JobPhase, PhaseUpdate, dispatch_event

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


# Remote environments kept warm AFTER the `with cp.cloud(...)` block exits, so
# the dashboard stays browsable and repeat runs of the same model reuse the warm
# container (Modal's scaledown_window idles it out, ~20 min). Keyed by
# (project, model_slug). Torn down by cp.cleanup_volumes(), cloud.destroy(), or
# atexit on interpreter/kernel shutdown.
_LIVE_ENVS: dict = {}


def _env_key(project: str, model) -> tuple:
    from cloudposterior.naming import model_slug

    return (project, model_slug(model))


def _normalize_until(until) -> dict | None:
    """Resolve an adaptive-stop target into a dict (or None).

    ``True`` -> Vehtari defaults; a dict overrides ``r_hat`` / ``ess`` over
    those defaults; anything else (None / False) -> None. Shared by
    ``cp.cloud(until=...)`` and ``cp.map(until=...)`` so the worker always
    receives a dict (it calls ``until.get(...)``), never a bare ``True``.
    """
    if until is True:
        return {"r_hat": 1.01, "ess": 400}
    if isinstance(until, dict):
        return {"r_hat": 1.01, "ess": 400, **until}
    return None


def _teardown_live_envs(project: str | None = None) -> None:
    """Tear down kept-warm environments (all, or just one project)."""
    for key in list(_LIVE_ENVS):
        if project is not None and key[0] != project:
            continue
        env = _LIVE_ENVS.pop(key, None)
        if env is not None:
            try:
                env.teardown()
            except Exception:
                pass


import atexit as _atexit  # noqa: E402

_atexit.register(_teardown_live_envs)


class cloud:
    """Context manager that intercepts PyMC operations with remote execution,
    caching, live dashboard, and push notifications.

    Remote containers stay warm for 20 minutes after the last run.

    Runs the full MCMC workflow in the cloud: ``pm.sample`` (NUTS via nutpie /
    pymc / numpyro / blackjax, and custom ``step=`` methods), ``pm.sample_smc``,
    ``pm.sample_prior_predictive`` / ``pm.sample_posterior_predictive``, and
    ``pm.compute_log_likelihood`` (for ``az.loo`` / ``az.waic`` / ``az.compare``).
    Optimization-based inference (``pm.fit`` variational, ``pm.find_MAP``) and
    non-InferenceData utilities (``pm.compute_deterministics``, ``pm.draw``) are
    not yet routed to the cloud and still run locally. For remote ``pm.sample``,
    ``return_inferencedata=False`` and a per-draw ``callback=`` can't be matched
    exactly and warn instead of silently diverging.

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
        until: dict | bool | None = None,
        overwrite: bool = False,
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
        self.overwrite = overwrite
        self.dashboard = dashboard and remote  # dashboard only works with remote
        self.notify = notify
        self.instance = instance
        self.progress = progress
        self.project = project or _detect_project_name()
        # Adaptive convergence target (remote only). The worker early-stops once
        # every scalar param clears it (draws= is the cap).
        self.until = _normalize_until(until)
        self._originals: dict[str, object] = {}
        self._env = None
        self._model_bytes: bytes | None = None

    def __enter__(self):
        import pymc as pm

        self._originals["sample"] = pm.sample
        self._originals["sample_prior_predictive"] = pm.sample_prior_predictive
        self._originals["sample_posterior_predictive"] = pm.sample_posterior_predictive
        self._originals["sample_smc"] = pm.sample_smc
        self._originals["compute_log_likelihood"] = pm.compute_log_likelihood
        pm.sample = self._make_intercepted_sample()
        pm.sample_prior_predictive = self._make_intercepted_predictive("prior")
        pm.sample_posterior_predictive = self._make_intercepted_predictive("posterior")
        pm.sample_smc = self._make_intercepted_smc()
        pm.compute_log_likelihood = self._make_intercepted_cll()
        self.model.__enter__()
        return self.model

    def __exit__(self, *exc):
        import pymc as pm

        pm.sample = self._originals["sample"]
        pm.sample_prior_predictive = self._originals["sample_prior_predictive"]
        pm.sample_posterior_predictive = self._originals["sample_posterior_predictive"]
        pm.sample_smc = self._originals["sample_smc"]
        pm.compute_log_likelihood = self._originals["compute_log_likelihood"]
        # Leave the remote env warm (kept in _LIVE_ENVS) so the dashboard stays
        # browsable and a repeat run reuses the container. Stopped by
        # cp.cleanup_volumes(), session.destroy(), or atexit on shutdown.
        return self.model.__exit__(*exc)

    def _provision_environment(self, nuts_sampler: str, sample_kwargs: dict):
        from cloudposterior.backends.modal_backend import ModalBackend
        from cloudposterior.serialize import get_version_manifest

        # Reuse a kept-warm env for this model if one is still up -- unless this
        # run needs a dashboard the warm env wasn't provisioned with.
        key = _env_key(self.project, self.model)
        existing = _LIVE_ENVS.get(key)
        if existing is not None and not (self.dashboard and existing._dashboard_fn is None):
            self._env = existing
            return
        if existing is not None:  # insufficient for this run -- replace it
            try:
                existing.teardown()
            except Exception:
                pass
            _LIVE_ENVS.pop(key, None)

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
            # Provision the /stop endpoint for the in-notebook stop button even
            # when the full dashboard is off (any remote run with a progress UI).
            stop_enabled=self.progress,
        )
        # Stash the resolved config so the displayed instance_desc matches
        # what was actually provisioned (no recomputation drift).
        self._env.config = config
        _LIVE_ENVS[key] = self._env  # keep warm past the `with` block

    def destroy(self):
        """Tear down the environment and clean up the project volume.

        Call after the ``with`` block to immediately stop the container
        and delete the project's volume::

            session = cp.cloud(model, remote=True)
            with session:
                idata = pm.sample(draws=2000)
            session.destroy()
        """
        from cloudposterior.backends.modal_backend import ModalBackend

        _teardown_live_envs(self.project)  # stop the kept-warm session(s)
        self._env = None
        ModalBackend.cleanup_volumes(project=self.project)

    def _make_intercepted_sample(self):
        ctx = self

        def intercepted_sample(**kwargs):
            _validate_sample_kwargs(kwargs)
            nuts_sampler = kwargs.pop("nuts_sampler", None)
            has_step = kwargs.get("step") is not None
            if has_step:
                # A custom step= requires PyMC's own sampler -- nutpie and the
                # JAX samplers ignore step= and would silently run NUTS instead.
                nuts_sampler = "pymc"
            elif nuts_sampler is None:
                nuts_sampler = _default_sampler(ctx.model, local=not ctx.remote)

            if ctx.remote:
                _warn_remote_sample_fidelity(kwargs)

            # Adaptive early-stop: worker-side, so remote + nutpie/pymc only.
            if ctx.until is not None:
                import warnings

                if not ctx.remote:
                    warnings.warn(
                        "until= is remote-only (worker-side early-stop); "
                        "ignored for local sampling.", stacklevel=2,
                    )
                elif nuts_sampler in ("numpyro", "blackjax"):
                    warnings.warn(
                        f"until= requires the nutpie or pymc sampler; ignored "
                        f"for nuts_sampler={nuts_sampler!r}.", stacklevel=2,
                    )
                else:
                    kwargs["until"] = ctx.until

            # Lazy first-touch serialization: pay the cloudpickle cost only
            # when we actually need it. Memoize on the model so repeat calls
            # in the same session don't re-serialize.
            if ctx._model_bytes is None:
                ctx._model_bytes = _ensure_model_bytes(ctx.model)
            else:
                _warn_if_model_data_changed(ctx)

            # Remote step= rides as a combined {model, step} payload so the
            # step's value variables stay identity-linked to the model the
            # worker deserializes (a separately-pickled step would not match).
            call_model_bytes = ctx._model_bytes
            if ctx.remote and has_step:
                from cloudposterior.serialize import serialize_model_with_step

                call_model_bytes = serialize_model_with_step(
                    ctx.model, kwargs.pop("step")
                )

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
                    model_bytes=call_model_bytes,
                    cache=ctx.cache,
                    overwrite=ctx.overwrite,
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
                overwrite=ctx.overwrite,
                notify=ctx.notify,
                instance=ctx.instance,
                nuts_sampler=nuts_sampler,
                progress=ctx.progress,
                original_sample=ctx._originals["sample"],
                model_bytes=call_model_bytes,
                **kwargs,
            )

        return intercepted_sample

    def _make_intercepted_predictive(self, kind: str):
        """Intercept pm.sample_prior_predictive / pm.sample_posterior_predictive.

        ``kind`` is "prior" or "posterior". Forward passes (no MCMC) -- routed to
        a non-streaming worker entry that loads the model (and, for posterior, the
        trace) and returns the result. Local runs defer to the original.
        """
        ctx = self
        orig_key = (
            "sample_prior_predictive" if kind == "prior" else "sample_posterior_predictive"
        )

        def intercepted_predictive(*args, **kwargs):
            if not ctx.remote:
                return ctx._originals[orig_key](*args, **kwargs)

            if ctx._model_bytes is None:
                ctx._model_bytes = _ensure_model_bytes(ctx.model)
            if ctx._env is None:
                ctx._provision_environment(_default_sampler(ctx.model, local=False), kwargs)

            trace = None
            if kind == "posterior":
                if args:
                    trace = args[0]
                elif "trace" in kwargs:
                    trace = kwargs.pop("trace")
                else:
                    raise TypeError("sample_posterior_predictive requires a trace")
            elif args:  # prior predictive: first positional is draws
                kwargs["draws"] = args[0]

            return _run_predictive(ctx, kind, trace, kwargs)

        return intercepted_predictive

    def _make_intercepted_smc(self):
        """Intercept pm.sample_smc. Returns InferenceData, like pm.sample.

        SMC has no per-draw callback, so the remote call is blocking (no live
        streaming). Local runs defer to the original.
        """
        ctx = self

        def intercepted_smc(draws=2000, **kwargs):
            if not ctx.remote:
                return ctx._originals["sample_smc"](draws, **kwargs)

            kwargs["draws"] = draws
            _validate_sample_kwargs(kwargs)
            if ctx._model_bytes is None:
                ctx._model_bytes = _ensure_model_bytes(ctx.model)
            if ctx._env is None:
                ctx._provision_environment(_default_sampler(ctx.model, local=False), kwargs)
            return _run_smc(ctx, kwargs)

        return intercepted_smc

    def _make_intercepted_cll(self):
        """Intercept pm.compute_log_likelihood (idata in -> idata out).

        Matches PyMC exactly: with the default extend_inferencedata=True the
        caller's idata is extended in place with a log_likelihood group and
        returned; with extend_inferencedata=False a standalone idata holding
        just that group is returned. Local runs defer to the original.
        """
        ctx = self

        def intercepted_cll(idata=None, **kwargs):
            if not ctx.remote:
                return ctx._originals["compute_log_likelihood"](idata, **kwargs)
            if idata is None:
                raise TypeError(
                    "compute_log_likelihood() missing required argument: 'idata'"
                )
            if ctx._model_bytes is None:
                ctx._model_bytes = _ensure_model_bytes(ctx.model)
            if ctx._env is None:
                ctx._provision_environment(_default_sampler(ctx.model, local=False), kwargs)

            extend = kwargs.get("extend_inferencedata", True)
            out = _run_idata_op(ctx, "compute_log_likelihood", idata, kwargs)
            from cloudposterior._idata import add_group, get_group, group_names

            if extend:
                # Match PyMC's in-place semantics: add the new group(s) to the
                # caller's idata and return that same object.
                existing = set(group_names(idata))
                for name in group_names(out):
                    if name not in existing:
                        add_group(idata, name, get_group(out, name))
                return idata
            # extend_inferencedata=False: PyMC returns the bare log_likelihood
            # Dataset, not an InferenceData.
            return get_group(out, "log_likelihood")

        return intercepted_cll


def _ensure_model_bytes(model) -> bytes:
    """Serialize a model once and memoize the bytes on the model object.

    pm.sample() mutates the model in place (compiled functions, etc.), which
    would change the cloudpickle output. Serializing once before the first
    sample and reusing across calls keeps the cache key stable.
    """
    from cloudposterior.serialize import serialize_model

    if not hasattr(model, "_cp_model_bytes"):
        model._cp_model_bytes = serialize_model(model)
        model._cp_data_fp = _observed_data_fingerprint(model)
    return model._cp_model_bytes


def _has_discrete_free_rvs(model) -> bool:
    """True if any free RV is discrete (int/bool), which nutpie/JAX NUTS can't handle."""
    try:
        for rv in model.free_RVs:
            if str(getattr(rv, "dtype", "") or "").startswith(("int", "uint", "bool")):
                return True
    except Exception:
        return True  # be conservative -- the pymc sampler handles anything
    return False


def _nutpie_available() -> bool:
    import importlib.util

    return importlib.util.find_spec("nutpie") is not None


def _default_sampler(model, *, local: bool) -> str:
    """Resolve the default NUTS sampler.

    nutpie is PyMC 6's own default and ~2x faster on CPU, so prefer it for
    fully-continuous models. Fall back to the pymc sampler for models with
    discrete variables, and -- for local runs only -- when nutpie isn't
    installed (remote images always ship nutpie).
    """
    if _has_discrete_free_rvs(model):
        return "pymc"
    if local and not _nutpie_available():
        return "pymc"
    return "nutpie"


_CORE_INT_KWARGS = ("draws", "tune", "chains", "cores")


def _validate_sample_kwargs(kwargs: dict) -> None:
    """Catch the common typo class early: the core sampling counts must be
    positive ints. Everything else passes through to pm.sample unchanged.
    """
    for key in _CORE_INT_KWARGS:
        if key in kwargs and kwargs[key] is not None:
            val = kwargs[key]
            if isinstance(val, bool) or not isinstance(val, int) or val <= 0:
                raise TypeError(f"{key} must be a positive int, got {val!r}")


def _warn_remote_sample_fidelity(kwargs: dict) -> None:
    """Warn (don't silently diverge) about the two pm.sample behaviors that
    can't be matched exactly for remote execution."""
    import warnings

    if kwargs.get("return_inferencedata") is False:
        warnings.warn(
            "return_inferencedata=False is not yet supported for remote "
            "sampling (a MultiTrace can't be transported back); an "
            "InferenceData is returned instead.",
            stacklevel=3,
        )
    if kwargs.get("callback") is not None:
        warnings.warn(
            "callback= can't run per-draw against local state inside a remote "
            "container; it is ignored for remote sampling (use remote=False to "
            "run a callback locally).",
            stacklevel=3,
        )


def _observed_data_fingerprint(model):
    """Best-effort fingerprint of the model's mutable (``pm.Data``) arrays.

    The model is cloudpickled once on the first ``pm.sample`` call; if the user
    mutates observed data (e.g. ``pm.set_data``) mid-block, the stale bytes are
    reused. This lets us warn instead of silently returning the wrong cache hit.
    """
    try:
        import numpy as np
        from pytensor.compile.sharedvalue import SharedVariable

        fp = 0
        for var in model.named_vars.values():
            if isinstance(var, SharedVariable):
                try:
                    arr = np.asarray(var.get_value(borrow=True))
                    fp ^= hash((var.name, arr.shape, str(arr.dtype), float(np.asarray(arr, dtype="float64").sum())))
                except Exception:
                    continue
        return fp
    except Exception:
        return None


def _warn_if_model_data_changed(ctx) -> None:
    """Warn if the model's mutable data changed after the model was serialized."""
    stored = getattr(ctx.model, "_cp_data_fp", None)
    if stored is None:
        return
    current = _observed_data_fingerprint(ctx.model)
    if current is not None and current != stored:
        import warnings

        warnings.warn(
            "The model's observed/mutable data changed after the first pm.sample() "
            "call in this cp.cloud(...) block. The model is serialized once per block, "
            "so this change is ignored remotely and the cache key is unchanged. Start a "
            "new cp.cloud(...) block to sample the updated data.",
            stacklevel=3,
        )


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
    overwrite: bool = False,
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
        # overwrite= forces a re-run: skip the load (still save below to replace).
        cached = None if overwrite else cache_backend.load(cache_key, sample_kwargs=cache_kwargs)
        if cached is not None:
            if progress:
                from cloudposterior.display import _emit_oneshot_html

                def _cached_terminal():
                    from rich.console import Console
                    Console().print("[green]\u2713[/green] [dim]cached result[/dim]")

                _emit_oneshot_html(
                    [
                        '<div style="font-family:monospace;font-size:13px;color:#888;padding:2px 0;">'
                        '<span style="color:#5cb85c;">&#10003;</span> cached result'
                        '</div>'
                    ],
                    terminal_fallback=_cached_terminal,
                )
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
        dispatch_event(event, sinks)

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
            nuts_sampler=nuts_sampler,
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
                 instance_desc: str, model=None, dashboard_dict=None,
                 stop_url: str | None = None, stop_token: str | None = None) -> list:
    """Create display, dashboard, and notification sinks."""
    sinks = []

    if progress:
        from cloudposterior.display import (
            NotebookDisplay,
            TerminalDisplay,
            _is_marimo,
            _is_notebook,
        )

        if _is_marimo() or _is_notebook():
            display = NotebookDisplay(instance_desc, stop_url=stop_url, stop_token=stop_token)
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
    from cloudposterior.display import _emit_oneshot_html

    # HTML fragments for browser frontends (Jupyter + marimo). The SVG QR
    # renders in both; the terminal fallback prints an ASCII QR instead.
    parts = [
        '<div style="font-family:monospace;font-size:12px;padding:4px 0;">',
        f'{label}: <a href="{url}" target="_blank">{url}</a>',
    ]
    if show_qr:
        try:
            import io

            import qrcode
            import qrcode.image.svg

            qr = qrcode.make(url, image_factory=qrcode.image.svg.SvgPathImage, box_size=6)
            buf = io.BytesIO()
            qr.save(buf)
            svg = buf.getvalue().decode("utf-8")
            parts.append(f'<div style="padding:4px 0;">{svg}</div>')
        except Exception:
            pass
    parts.append("</div>")

    def _terminal():
        from rich.console import Console

        Console().print(f"[dim]{label}: {url}[/dim]")
        if show_qr:
            try:
                import qrcode

                qr = qrcode.QRCode(border=1)
                qr.add_data(url)
                qr.make(fit=True)
                qr.print_ascii(out=None)  # prints to stdout
            except Exception:
                pass

    _emit_oneshot_html(parts, terminal_fallback=_terminal)


def _stop_sinks(sinks: list):
    for sink in sinks:
        if hasattr(sink, "stop"):
            sink.stop()


def _run_predictive(ctx, kind: str, trace, sample_kwargs: dict):
    """Run prior/posterior predictive on the remote persistent environment.

    A blocking (non-streaming) call: uploads the model if needed, invokes the
    worker's predictive method, and returns the decoded InferenceData.
    """
    from cloudposterior.backends.modal_backend import (
        _compute_payload_path,
        _run_blocking_op,
    )
    from cloudposterior.serialize import (
        deserialize_inference_data,
        serialize_inference_data,
    )

    env = ctx._env
    env._ensure_running()
    payload_path = _compute_payload_path(env._model_slug, ctx._model_bytes)
    env._upload_if_needed(ctx._model_bytes, payload_path)

    # The worker loads the model from the Volume -- never ship one in kwargs.
    sample_kwargs.pop("model", None)

    if kind == "prior":
        idata_bytes = _run_blocking_op(env, "prior_predictive", payload_path, sample_kwargs)
    else:
        trace_bytes = serialize_inference_data(trace)
        idata_bytes = _run_blocking_op(
            env, "posterior_predictive", payload_path, trace_bytes, sample_kwargs
        )
    return deserialize_inference_data(idata_bytes)


def _run_smc(ctx, sample_kwargs: dict) -> az.InferenceData:
    """Run pm.sample_smc on the remote persistent environment (blocking).

    Caches the result like pm.sample, namespacing the key with the op so SMC
    output never collides with a same-kwargs pm.sample / pm.fit run.
    """
    from cloudposterior.backends.modal_backend import (
        _compute_payload_path,
        _run_blocking_op,
    )
    from cloudposterior.cache import resolve_cache
    from cloudposterior.naming import cache_key as compute_cache_key
    from cloudposterior.serialize import deserialize_inference_data

    cache_kwargs = {**sample_kwargs, "_cp_op": "smc"}
    cache_backend = resolve_cache(ctx.cache, model=ctx.model)
    cache_key = None
    if cache_backend is not None:
        cache_key = compute_cache_key(ctx._model_bytes, cache_kwargs)
        # overwrite= forces a re-run: skip the load (still save below to replace).
        cached = None if ctx.overwrite else cache_backend.load(cache_key, sample_kwargs=cache_kwargs)
        if cached is not None:
            return cached

    env = ctx._env
    env._ensure_running()
    payload_path = _compute_payload_path(env._model_slug, ctx._model_bytes)
    env._upload_if_needed(ctx._model_bytes, payload_path)
    sample_kwargs.pop("model", None)

    idata_bytes = _run_blocking_op(env, "sample_smc", payload_path, sample_kwargs)
    idata = deserialize_inference_data(idata_bytes)

    if cache_backend is not None and cache_key:
        cache_backend.save(cache_key, idata, sample_kwargs=cache_kwargs)
    return idata


def _run_idata_op(ctx, op: str, in_idata, sample_kwargs: dict):
    """Run an idata-in / idata-out op (e.g. compute_log_likelihood) remotely.

    Ships the input idata to the worker, invokes ``op``, and returns the
    decoded result. Not cached (matches the predictive precedent; the input
    idata isn't folded into a key).
    """
    from cloudposterior.backends.modal_backend import (
        _compute_payload_path,
        _run_blocking_op,
    )
    from cloudposterior.serialize import (
        deserialize_inference_data,
        serialize_inference_data,
    )

    env = ctx._env
    env._ensure_running()
    payload_path = _compute_payload_path(env._model_slug, ctx._model_bytes)
    env._upload_if_needed(ctx._model_bytes, payload_path)
    sample_kwargs.pop("model", None)

    in_bytes = serialize_inference_data(in_idata)
    out_bytes = _run_blocking_op(env, op, payload_path, in_bytes, sample_kwargs)
    return deserialize_inference_data(out_bytes)


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
    overwrite: bool = False,
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
        # overwrite= forces a re-run: skip the load (still save below to replace).
        cached = None if overwrite else cache_backend.load(cache_key, sample_kwargs=cache_kwargs)
        if cached is not None:
            if progress:
                from cloudposterior.display import _emit_oneshot_html

                def _cached_terminal():
                    from rich.console import Console
                    Console().print("[green]\u2713[/green] [dim]cached result[/dim]")

                _emit_oneshot_html(
                    [
                        '<div style="font-family:monospace;font-size:13px;color:#888;padding:2px 0;">'
                        '<span style="color:#5cb85c;">&#10003;</span> cached result'
                        '</div>'
                    ],
                    terminal_fallback=_cached_terminal,
                )
            return cached

    # Cache miss -- read the actually-provisioned config from the env so the
    # displayed instance description matches what's running (no recomputation drift).
    config = env.config
    instance_desc = f"Modal ({config.describe()})"

    # Start the app early when control infra (dashboard or stop button) exists,
    # so the stop/dashboard URLs are captured before we build the display.
    if env._stop_fn is not None or (dashboard and env._dashboard_fn is not None):
        env._ensure_running()

    # Clear a stale stop flag left in the control Dict by a prior run on this
    # kept-warm env -- otherwise the worker would abort immediately.
    control_dict = getattr(env, "_dashboard_dict", None)
    if control_dict is not None:
        try:
            from cloudposterior.backends.modal_backend import _run_blocking

            _run_blocking(control_dict.__setitem__, "stop", False)
        except Exception:
            pass

    # Show the dashboard URL *above* the live display, so it stays put as the
    # progress widget grows (phase steps + chain rows get added below it).
    if dashboard and env._dashboard_fn is not None:
        dashboard_url = env._dashboard_url
        if dashboard_url:
            # Ensure URL ends with / so the ASGI app serves from root
            if not dashboard_url.endswith("/"):
                dashboard_url += "/"
            _show_link(dashboard_url, label="Dashboard", show_qr=True)

    sinks = _build_sinks(
        progress=progress,
        dashboard=dashboard,
        notify=notify,
        instance_desc=instance_desc,
        model=model,
        dashboard_dict=getattr(env, "_dashboard_dict", None),
        stop_url=getattr(env, "_stop_url", None),
        stop_token=getattr(env, "_stop_token", None),
    )

    def emit(event):
        dispatch_event(event, sinks)

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
    nuts_sampler: str = "pymc",
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

    sampler_kwargs = {"nuts_sampler": nuts_sampler} if nuts_sampler else {}

    # PyMC's per-draw callback only fires for the pymc sampler; external samplers
    # (nutpie/numpyro/blackjax) raise if a callback is passed (PyMC 6), so only
    # attach it when the resolved sampler is "pymc".
    if sinks and nuts_sampler == "pymc":
        tune = sample_kwargs.get("tune", 1000)
        draws = sample_kwargs.get("draws", 1000)
        progress_queue: Queue = Queue()
        callback = make_sampling_callback(progress_queue, tune, draws)
        aggregator = ProgressAggregator(progress_queue)

        def stream_progress():
            for snapshot in aggregator.snapshots():
                emit(snapshot)

        # A plain threading.Thread has no marimo runtime context, so the
        # widget's trait writes would no-op there; marimo.Thread propagates the
        # context so local progress animates live. (Remote sampling emits on the
        # main thread, so it's unaffected.) Fall back to threading.Thread
        # elsewhere (Jupyter/terminal).
        from cloudposterior.display import _is_marimo

        if _is_marimo():
            import marimo

            thread_cls = marimo.Thread
        else:
            thread_cls = Thread
        progress_thread = thread_cls(target=stream_progress, daemon=True)
        progress_thread.start()

        with model:
            idata = original_sample(
                callback=callback,
                **sampler_kwargs,
                **sample_kwargs,
            )

        aggregator.stop()
        progress_thread.join(timeout=2)
    else:
        with model:
            idata = original_sample(**sampler_kwargs, **sample_kwargs)

    emit(PhaseUpdate(
        phase=JobPhase.SAMPLING,
        status="done",
        message="sampling complete",
        elapsed=time.time() - sample_start,
    ))

    from cloudposterior.serialize import sanitize_inference_data

    return sanitize_inference_data(idata)


# -- Explicit API (backwards-compatible) --

def sample(
    model: pm.Model,
    *,
    draws: int = 1000,
    tune: int = 1000,
    chains: int | None = None,
    cores: int | None = None,
    nuts_sampler: str | None = None,
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

    if nuts_sampler is None:
        nuts_sampler = _default_sampler(model, local=False)

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

    Also stops any kept-warm session for the project (the dashboard goes offline).
    """
    from cloudposterior.backends.modal_backend import ModalBackend

    project = project or _detect_project_name()
    _teardown_live_envs(project)  # stop the kept-warm session first
    ModalBackend.cleanup_volumes(project=project)


def map(models, sample_kwargs=None, *, cache: bool | str = True,
        project: str | None = None, nuts_sampler: str | None = None,
        instance: str | None = None, progress: bool = True,
        dashboard: bool | None = None, until: dict | bool | None = None,
        overwrite: bool = False) -> list:
    """Fit many models in parallel on the cloud.

    ``models`` is a list of ``pm.Model`` -- vary priors, structure, or data for
    model comparison / sensitivity. ``sample_kwargs`` is a single dict of
    ``pm.sample`` kwargs applied to all, or a list aligned with ``models``. Each
    model is uploaded once and the fits run concurrently (Modal ``spawn``).
    Returns InferenceData in input order.

    ``until`` enables adaptive early-stop (``True`` -> Vehtari defaults, or a
    dict overriding ``r_hat`` / ``ess``): each fit stops once every scalar param
    clears the target, with ``draws=`` as the cap. Applies to the whole batch
    (nutpie / pymc samplers only); per-model targets are still possible by
    putting ``until`` in each model's ``sample_kwargs`` dict.

    ``overwrite=True`` ignores any cached result, re-runs every model, and
    replaces the stored entry (``cache=False`` instead skips the cache entirely,
    saving nothing).

    A live dashboard (on by default; pass ``dashboard=False`` to opt out) serves
    an overview of all models with drill-in per-model pages (chains, convergence,
    traces) and global / per-model Stop. Each worker writes its own progress into
    a shared Modal Dict keyed by its model, so the spawned fits surface live
    without streaming back to the client. The printed line-level progress is
    job-level only.

    Example::

        import arviz as az
        idatas = cp.map([pooled, hierarchical, per_county], {"draws": 1000})
        az.compare({"pooled": idatas[0], "hier": idatas[1], "county": idatas[2]})
    """
    import pymc as pm

    from cloudposterior.backends.modal_backend import (
        ModalBackend,
        _compute_payload_path,
        _run_blocking,
    )
    from cloudposterior.cache import resolve_cache
    from cloudposterior.naming import cache_key as compute_cache_key
    from cloudposterior.naming import model_slug
    from cloudposterior.serialize import (
        deserialize_inference_data,
        get_version_manifest,
    )

    if isinstance(models, pm.Model):
        models = [models]
    models = list(models)
    n = len(models)
    if n == 0:
        return []

    # Normalize sample_kwargs to one dict per model.
    if sample_kwargs is None:
        kwargs_list = [{} for _ in range(n)]
    elif isinstance(sample_kwargs, dict):
        kwargs_list = [dict(sample_kwargs) for _ in range(n)]
    else:
        kwargs_list = [dict(k) for k in sample_kwargs]
        if len(kwargs_list) != n:
            raise ValueError(
                f"sample_kwargs list ({len(kwargs_list)}) must align with models ({n})"
            )

    model_bytes_list = [_ensure_model_bytes(m) for m in models]
    samplers = [nuts_sampler or _default_sampler(m, local=False) for m in models]
    project = project or _detect_project_name()
    dashboard_on = True if dashboard is None else bool(dashboard)
    until_param = _normalize_until(until)
    _until_warned = False

    # Unique per-model dashboard key (slugs collide for identical/unnamed models)
    # doubling as the per-model stop key; human name for display.
    labels = [f"{model_slug(m)}-{i}" for i, m in enumerate(models)]
    names = [getattr(m, "name", "") or model_slug(m) for m in models]

    def _log(msg):
        if progress:
            print(f"cp.map: {msg}")

    # -- Cache check first: all local (cloudpickle + key hash + cache load touch
    # no Modal), so only cache *misses* need a remote container. An all-cached
    # map therefore provisions nothing -- no env, no dashboard. --
    cache_backend = resolve_cache(cache, model=models[0]) if cache else None
    results: list = [None] * n
    cached_idx: list[int] = []
    run_idx: list[int] = []
    # aligned with run_idx: (model_bytes, payload_path, kwargs, sampler, key, key_kwargs, label)
    run_meta: list = []
    for i, m in enumerate(models):
        # Adaptive-stop target: the until= param wins; otherwise normalize (and
        # de-footgun) any `until` left in this model's kwargs. Injected before the
        # cache key so it participates in caching, matching cp.cloud.
        raw_until = kwargs_list[i].pop("until", None)
        eff_until = until_param if until is not None else _normalize_until(raw_until)
        if eff_until is not None:
            if samplers[i] in ("numpyro", "blackjax"):
                if not _until_warned:
                    import warnings

                    warnings.warn(
                        "until= requires the nutpie or pymc sampler; ignored for "
                        f"nuts_sampler={samplers[i]!r}.", stacklevel=2,
                    )
                    _until_warned = True
            else:
                kwargs_list[i]["until"] = eff_until

        mb = model_bytes_list[i]
        payload_path = _compute_payload_path(model_slug(m), mb)
        ck_kwargs = {**kwargs_list[i], "nuts_sampler": samplers[i]}
        ckey = None
        if cache_backend is not None:
            # Still compute the key (so the fresh result is saved) but skip the
            # load when overwrite= forces a re-run.
            ckey = compute_cache_key(mb, ck_kwargs)
            if not overwrite:
                hit = cache_backend.load(ckey, sample_kwargs=ck_kwargs)
                if hit is not None:
                    results[i] = hit
                    cached_idx.append(i)
                    continue
        run_idx.append(i)
        run_meta.append((mb, payload_path, kwargs_list[i], samplers[i], ckey, ck_kwargs, labels[i]))

    n_cached = len(cached_idx)
    if not run_idx:
        _log(f"all {n} model(s) cached")
        return results

    # -- At least one cache miss: provision the shared env (image + Volume, and
    # the dashboard/stop infra when on) sized to the first model. --
    config = RemoteConfig.from_instance(
        instance, model=models[0], sample_kwargs=kwargs_list[0], nuts_sampler=samplers[0],
    )
    backend = ModalBackend(config=config)
    env = backend.provision(
        model_bytes_list[0], models[0], get_version_manifest(), config,
        project=project, idle_timeout=config.idle_timeout,
        dashboard=dashboard_on, stop_enabled=dashboard_on,
    )

    def _dash_write(key, value):
        d = getattr(env, "_dashboard_dict", None)
        if d is None:
            return
        try:
            _run_blocking(d.__setitem__, key, value)
        except Exception:
            pass

    teardown = True
    try:
        env._ensure_running()

        if dashboard_on:
            # Clear stale stop flags left by a prior run on a kept-warm env.
            _dash_write("stop", False)
            for lbl in labels:
                _dash_write(f"stop:{lbl}", False)
            # Publish the manifest so the dashboard renders one panel per model;
            # cached models show as already complete.
            _dash_write("models", [{"label": labels[i], "name": names[i]} for i in range(n)])
            for i in cached_idx:
                _dash_write(labels[i], {
                    "phases": [{"label": "cache_hit", "status": "done",
                                "detail": "loaded from cache"}],
                    "sampling": None, "complete": True,
                })
            if env._dashboard_fn is not None and env._dashboard_url:
                url = env._dashboard_url
                if not url.endswith("/"):
                    url += "/"
                _show_link(url, label="Dashboard", show_qr=True)

        _log(f"fitting {len(run_idx)} model(s) in parallel"
             + (f" ({n_cached} cached)" if n_cached else "") + " ...")
        if dashboard_on and env._dashboard_url:
            _log(f"dashboard: {env._dashboard_url}")

        # Upload the miss payloads now that the env is running.
        for (mb, payload_path, _kw, _sp, _ck, _cak, _lbl) in run_meta:
            env._upload_if_needed(mb, payload_path)

        sampler_obj = env._sampler_cls()
        dash_name = env._dashboard_dict_name if dashboard_on else None
        # spawn() is a blocking Modal call -- run it off the event loop so it
        # doesn't warn/stall inside an async host (marimo), mirroring how the
        # .get() result fetch below is already wrapped. Each worker writes its
        # own progress under its label and honors global / per-model stop.
        calls = [
            _run_blocking(
                sampler_obj.sample_blocking.spawn, pp, kw, sp,
                progress_dict_name=dash_name,
                progress_key=(lbl if dashboard_on else None),
                stop_dict_name=dash_name,
            )
            for (_mb, pp, kw, sp, _ck, _cak, lbl) in run_meta
        ]
        for j, i in enumerate(run_idx):
            idata = deserialize_inference_data(_run_blocking(calls[j].get))
            results[i] = idata
            _log(f"[{j + 1}/{len(run_idx)}] done")
            _mb, _pp, _kw, _sp, ckey, ck_kwargs, _lbl = run_meta[j]
            if cache_backend is not None and ckey is not None:
                cache_backend.save(ckey, idata, sample_kwargs=ck_kwargs)

        if dashboard_on:
            # Keep the env warm (reusing the cp.cloud registry) so the dashboard
            # stays browsable after the run; Modal idles it out via
            # scaledown_window / the atexit hook.
            _LIVE_ENVS[_env_key(project, models[0])] = env
            teardown = False
        return results
    finally:
        if teardown:
            env.teardown()
