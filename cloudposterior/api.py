"""Public API for cloudposterior."""

from __future__ import annotations

import threading
import time
import warnings
import weakref
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
_LIVE_ENVS: "dict[tuple[str, str], object]" = {}

# Guards _LIVE_ENVS and the pm.* monkeypatch, both of which are process-global.
_PATCH_LOCK = threading.Lock()

# Serialized-model memos. Keyed weakly so a model the user drops is not pinned
# in memory, and stored here rather than as attributes on the model itself --
# see _ensure_model_bytes for why an attribute would be actively harmful.
# model -> (observed_data_fingerprint, model_bytes)
_MODEL_BYTES_CACHE: "weakref.WeakKeyDictionary" = weakref.WeakKeyDictionary()
# step -> ((id(model), observed_data_fingerprint), combined_model_step_bytes)
_STEP_BYTES_CACHE: "weakref.WeakKeyDictionary" = weakref.WeakKeyDictionary()
# model -> (observed_data_fingerprint, structural cache identity)
_MODEL_DIGEST_CACHE: "weakref.WeakKeyDictionary" = weakref.WeakKeyDictionary()

# The real pm.* functions, captured the first time a block patches them.
_TRUE_ORIGINALS: dict = {}


def _true_original(name: str):
    """The genuine PyMC function, even while a cp.cloud block has it patched."""
    import pymc as pm

    return _TRUE_ORIGINALS.get(name) or getattr(pm, name)


def _forget_model_bytes(model) -> None:
    """Drop memoized serializations for a model (used when tearing down)."""
    _MODEL_BYTES_CACHE.pop(model, None)
    for step in [s for s, v in _STEP_BYTES_CACHE.items() if v[0][0] == id(model)]:
        _STEP_BYTES_CACHE.pop(step, None)


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
    exactly and warn instead of silently diverging. The Stop button can abort
    nutpie/pymc runs early (keeping the partial trace); JAX samplers
    (numpyro/blackjax) and SMC have no per-draw hook and run to completion.

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
        self._entered = False
        self._patched_sample = None

    # The five pm.* functions this block swaps out. Patching is process-global,
    # so entry/exit is guarded and strictly non-reentrant: re-entering the same
    # instance would capture the interceptors as the "originals", leaving
    # pm.sample patched after exit and recursing into itself forever.
    _PATCHED_NAMES = (
        "sample",
        "sample_prior_predictive",
        "sample_posterior_predictive",
        "sample_smc",
        "compute_log_likelihood",
    )

    def _restore(self, pm) -> None:
        for name in self._PATCHED_NAMES:
            if name in self._originals:
                setattr(pm, name, self._originals[name])

    def __enter__(self):
        import pymc as pm

        if self._entered:
            raise RuntimeError(
                "cp.cloud(...) is not reentrant -- this block is already "
                "active. Create a separate cp.cloud(...) instance instead."
            )
        self._entered = True
        with _PATCH_LOCK:
            for name in self._PATCHED_NAMES:
                self._originals[name] = getattr(pm, name)
                # Remember the genuine PyMC functions the first time we see
                # them, so cp.sample() called from inside a block doesn't take
                # an interceptor as its "original" and recurse.
                _TRUE_ORIGINALS.setdefault(name, self._originals[name])
            pm.sample = self._make_intercepted_sample()
            pm.sample_prior_predictive = self._make_intercepted_predictive("prior")
            pm.sample_posterior_predictive = self._make_intercepted_predictive("posterior")
            pm.sample_smc = self._make_intercepted_smc()
            pm.compute_log_likelihood = self._make_intercepted_cll()
            # Identity marker so __exit__ can detect out-of-order teardown.
            self._patched_sample = pm.sample
        try:
            self.model.__enter__()
        except BaseException:
            # Entering the model context failed, so __exit__ will never run --
            # unpatch here or PyMC stays intercepted for the whole process.
            with _PATCH_LOCK:
                self._restore(pm)
            self._entered = False
            raise
        return self.model

    def __exit__(self, *exc):
        import pymc as pm

        with _PATCH_LOCK:
            if pm.sample is not self._patched_sample:
                import warnings

                warnings.warn(
                    "cp.cloud(...) blocks were exited out of order; restoring "
                    "the original PyMC functions anyway. Nested blocks must be "
                    "exited innermost-first.",
                    stacklevel=2,
                )
            self._restore(pm)
        self._entered = False
        # Leave the remote env warm (kept in _LIVE_ENVS) so the dashboard stays
        # browsable and a repeat run reuses the container. Stopped by
        # cp.cleanup_volumes(), session.destroy(), or atexit on shutdown.
        return self.model.__exit__(*exc)

    _JAX_SAMPLERS = ("numpyro", "blackjax")

    def _can_reuse_env(self, env, nuts_sampler: str) -> bool:
        """Whether a kept-warm env satisfies this run's image/feature needs."""
        # A run that wants the live dashboard can't reuse an env without one.
        if self.dashboard and env._dashboard_fn is None:
            return False
        # jax/numpyro ship only in images built for a JAX sampler (or for GPU),
        # so a warm image built for pymc/nutpie can't serve a JAX run. Be
        # conservative when the warm env doesn't say what it was built for.
        if nuts_sampler in self._JAX_SAMPLERS:
            cfg = getattr(env, "config", None)
            built_for = getattr(env, "nuts_sampler", None)
            has_jax = (cfg is not None and cfg.gpu is not None) or (
                built_for in self._JAX_SAMPLERS
            )
            if not has_jax:
                return False
        return True

    def _provision_environment(self, nuts_sampler: str, sample_kwargs: dict):
        from cloudposterior.backends.modal_backend import ModalBackend
        from cloudposterior.serialize import get_version_manifest

        # Reuse a kept-warm env for this model if one is still up -- unless it
        # lacks what this run needs (a dashboard, or a jax/GPU image).
        key = _env_key(self.project, self.model)
        existing = _LIVE_ENVS.get(key)
        if existing is not None and self._can_reuse_env(existing, nuts_sampler):
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
            # The image needs jax/numpyro for the JAX samplers on CPU presets.
            nuts_sampler=nuts_sampler,
        )
        # Stash the resolved config so the displayed instance_desc matches
        # what was actually provisioned (no recomputation drift).
        self._env.config = config
        _LIVE_ENVS[key] = self._env  # keep warm past the `with` block

    def destroy(self, delete_volume: bool = False):
        """Tear down this session's environment.

        Call after the ``with`` block to immediately stop the container::

            session = cp.cloud(model, remote=True)
            with session:
                idata = pm.sample(draws=2000)
            session.destroy()

        Only this session's model is affected. ``delete_volume=True`` also
        deletes the *project-wide* Volume, discarding every model's uploaded
        payload -- opt-in because the project is shared with other sessions.
        """
        from cloudposterior.backends.modal_backend import ModalBackend

        key = _env_key(self.project, self.model)
        with _PATCH_LOCK:
            env = _LIVE_ENVS.pop(key, None)
        if env is not None:
            try:
                env.teardown()
            except Exception:
                pass
        self._env = None
        self._model_bytes = None
        _forget_model_bytes(self.model)
        if delete_volume:
            ModalBackend.cleanup_volumes(project=self.project)

    def _make_intercepted_sample(self):
        ctx = self

        def intercepted_sample(**kwargs):
            if not _call_targets_ctx_model(ctx, kwargs, "pm.sample()"):
                return ctx._originals["sample"](**kwargs)
            # The target is the wrapped model -- drop a redundant model= kwarg
            # so it doesn't leak into cache keys or the remote payload.
            kwargs.pop("model", None)

            _validate_sample_kwargs(kwargs)
            nuts_sampler = kwargs.pop("nuts_sampler", None)
            has_step = kwargs.get("step") is not None
            if has_step:
                # A custom step= requires PyMC's own sampler -- nutpie and the
                # JAX samplers ignore step= and would silently run NUTS instead.
                nuts_sampler = "pymc"
            elif nuts_sampler is None:
                nuts_sampler = _default_sampler(ctx.model, local=not ctx.remote)

            # Send tune explicitly. Under PyMC 6 an unset tune resolves per
            # sampler (400 for nutpie, 1000 for pymc), so leaving it out made
            # progress totals and the auto-sized timeout disagree with what
            # the worker actually ran.
            kwargs["tune"] = resolve_tune(kwargs, nuts_sampler)

            if ctx.remote:
                _warn_remote_sample_fidelity(kwargs)
                # The warning says callback= is ignored remotely -- actually
                # remove it: a function kwarg would poison the cache key (its
                # repr embeds a memory address) and ride the wire for nothing.
                kwargs.pop("callback", None)

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
            # when we actually need it. Memoized, and re-done automatically if
            # the observed data changed since the last call.
            ctx._model_bytes = _ensure_model_bytes(ctx.model)

            # Remote step= rides as a combined {model, step} payload so the
            # step's value variables stay identity-linked to the model the
            # worker deserializes (a separately-pickled step would not match).
            call_model_bytes = ctx._model_bytes
            if ctx.remote and has_step:
                call_model_bytes = _ensure_step_bytes(ctx.model, kwargs.pop("step"))

            if ctx.remote:
                # Check the cache BEFORE provisioning: a hit must not create
                # Modal infra (Volume / control Dict / app) just to be skipped.
                cache_kwargs = {**kwargs, "nuts_sampler": nuts_sampler}
                cache_backend, cache_key, cached = _cache_lookup(
                    ctx.cache, ctx.model, call_model_bytes, cache_kwargs,
                    overwrite=ctx.overwrite, progress=ctx.progress,
                )
                if cached is not None:
                    return cached

                if ctx._env is None:
                    # First sample call -- provision sized to these kwargs.
                    ctx._provision_environment(nuts_sampler, kwargs)
                else:
                    # Later call -- warn if auto-sizing would have picked
                    # different resources (first call wins; container is fixed).
                    _warn_if_resize_drift(ctx, nuts_sampler, kwargs)

                return _run_sample_persistent(
                    model=ctx.model,
                    env=ctx._env,
                    model_bytes=call_model_bytes,
                    cache_backend=cache_backend,
                    cache_key=cache_key,
                    cache_kwargs=cache_kwargs,
                    dashboard=ctx.dashboard,
                    notify=ctx.notify,
                    nuts_sampler=nuts_sampler,
                    progress=ctx.progress,
                    notify_topic_holder=ctx,
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
                notify_topic_holder=ctx,
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
            if not ctx.remote or not _call_targets_ctx_model(
                ctx, kwargs, f"pm.{orig_key}()"
            ):
                return ctx._originals[orig_key](*args, **kwargs)
            kwargs.pop("model", None)
            _warn_predictive_kwarg_drift(kind, kwargs)

            ctx._model_bytes = _ensure_model_bytes(ctx.model)
            if ctx._env is None:
                ctx._provision_environment(_default_sampler(ctx.model, local=False), kwargs)

            trace = None
            extend = False
            if kind == "posterior":
                if args:
                    trace = args[0]
                elif "trace" in kwargs:
                    trace = kwargs.pop("trace")
                else:
                    raise TypeError("sample_posterior_predictive requires a trace")
                # PyMC's extend_inferencedata=True extends the caller's idata
                # in place; the worker only ever computes standalone, so merge
                # client-side (mirrors intercepted_cll).
                extend = bool(kwargs.pop("extend_inferencedata", False))
            elif args:  # prior predictive: first positional is draws
                kwargs["draws"] = args[0]

            out = _run_predictive(ctx, kind, trace, kwargs)
            if kind == "posterior" and extend:
                from cloudposterior._idata import add_group, get_group, group_names

                existing = set(group_names(trace))
                for name in group_names(out):
                    if name not in existing:
                        add_group(trace, name, get_group(out, name))
                return trace
            return out

        return intercepted_predictive

    def _make_intercepted_smc(self):
        """Intercept pm.sample_smc. Returns InferenceData, like pm.sample.

        SMC has no per-draw callback, so the remote call is blocking (no live
        streaming). Local runs defer to the original.
        """
        ctx = self

        def intercepted_smc(draws=2000, **kwargs):
            if not ctx.remote or not _call_targets_ctx_model(
                ctx, kwargs, "pm.sample_smc()"
            ):
                return ctx._originals["sample_smc"](draws, **kwargs)
            kwargs.pop("model", None)

            kwargs["draws"] = draws
            _validate_sample_kwargs(kwargs)
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
            if not ctx.remote or not _call_targets_ctx_model(
                ctx, kwargs, "pm.compute_log_likelihood()"
            ):
                return ctx._originals["compute_log_likelihood"](idata, **kwargs)
            kwargs.pop("model", None)
            if idata is None:
                raise TypeError(
                    "compute_log_likelihood() missing required argument: 'idata'"
                )
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


_STREAM_DONE = object()


def _prepend(first, rest):
    """Re-attach an already-pulled first item to the front of an iterator."""
    if first is not _STREAM_DONE:
        yield first
        yield from rest


def _is_missing_payload_error(exc: Exception) -> bool:
    """Whether a worker error means the Volume payload is gone.

    The worker opens the payload path directly, so a pruned payload surfaces
    as a plain file-not-found from inside the container.
    """
    msg = str(exc).lower()
    return (
        "payload" in msg and ("not found" in msg or "no such file" in msg)
    ) or "filenotfounderror" in msg


def _model_identity(model) -> str:
    """Cache identity for a model, memoized against its data fingerprint.

    Structural, not a hash of the pickle bytes: rebuilding the same model in a
    new session yields different bytes (fresh RNG state on every shared
    variable), which made the persistent disk cache miss every time.
    """
    from cloudposterior.naming import model_digest

    fp = _observed_data_fingerprint(model)
    cached = _MODEL_DIGEST_CACHE.get(model)
    if cached is not None and cached[0] == fp:
        return cached[1]
    digest = model_digest(model)
    _MODEL_DIGEST_CACHE[model] = (fp, digest)
    return digest


def _call_targets_ctx_model(ctx, kwargs: dict, what: str) -> bool:
    """Whether an intercepted PyMC call actually targets the wrapped model.

    A call that names a *different* model (an explicit ``model=`` kwarg, or an
    inner ``with other_model:`` context) must not be intercepted -- we would
    silently run the wrapped model instead and hand back results for the wrong
    model. Warn and let the caller fall back to native PyMC.
    """
    target_model = kwargs.get("model")
    if target_model is None:
        try:
            import pymc as pm

            target_model = pm.modelcontext(None)
        except Exception:
            target_model = None
    if target_model is not None and target_model is not ctx.model:
        import warnings

        warnings.warn(
            f"{what} inside cp.cloud(...) targets a different model than the "
            "one the block wraps; running it with native PyMC (no cloud "
            "execution or caching) for this call. Wrap that model in its own "
            "cp.cloud(...) block to run it remotely.",
            stacklevel=3,
        )
        return False
    return True


def _ensure_model_bytes(model) -> bytes:
    """Serialize a model once and memoize the bytes, keyed by observed data.

    pm.sample() mutates the model in place (compiled functions, etc.), which
    would change the cloudpickle output. Serializing once and reusing across
    calls keeps the cache key stable.

    The memo lives in a module-level WeakKeyDictionary rather than on the model
    object: an attribute would be swept into the next cloudpickle (embedding the
    previous payload inside the new one), and it would outlive the data it was
    built from. Re-serializing when the observed-data fingerprint changes is
    what makes a post-``pm.set_data`` run produce the right bytes -- and the
    right cache key -- instead of silently reusing pre-mutation state.
    """
    from cloudposterior.serialize import serialize_model

    fp = _observed_data_fingerprint(model)
    cached = _MODEL_BYTES_CACHE.get(model)
    if cached is not None and cached[0] == fp:
        return cached[1]
    blob = serialize_model(model)
    _MODEL_BYTES_CACHE[model] = (fp, blob)
    return blob


def _ensure_step_bytes(model, step) -> bytes:
    """Serialize a combined ``{model, step}`` payload, memoized per step object.

    A separately-pickled step resolves its value variables against a different
    graph than the worker's model, so the two must ship in one blob (see
    ``serialize_model_with_step``). Memoizing on the step instance keeps repeat
    calls from re-pickling the whole model -- which would also mint a fresh
    payload hash and churn the remote Volume's prune window on every call.
    """
    from cloudposterior.serialize import serialize_model_with_step

    fp = _observed_data_fingerprint(model)
    cached = _STEP_BYTES_CACHE.get(step)
    if cached is not None and cached[0] == (id(model), fp):
        return cached[1]
    blob = serialize_model_with_step(model, step)
    _STEP_BYTES_CACHE[step] = ((id(model), fp), blob)
    return blob


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
    from cloudposterior._idata import pymc_major

    for key in _CORE_INT_KWARGS:
        if key in kwargs and kwargs[key] is not None:
            val = kwargs[key]
            if isinstance(val, bool) or not isinstance(val, int) or val <= 0:
                raise TypeError(f"{key} must be a positive int, got {val!r}")

    if kwargs.get("backend") is not None and pymc_major() < 6:
        raise TypeError(
            "backend= requires PyMC 6 or newer; the installed PyMC "
            f"{pymc_major()}.x has no such kwarg"
        )


def resolve_tune(sample_kwargs: dict, nuts_sampler: str) -> int:
    """Resolve the tune count the sampler will actually use.

    PyMC 6 made ``tune=None`` the default and resolves it per sampler: 400 for
    nutpie, 1000 for the pymc sampler. Progress totals and the auto-sized
    timeout both need the real number, so it is resolved client-side and sent
    explicitly rather than left for the worker to guess.
    """
    from cloudposterior._idata import pymc_major

    tune = sample_kwargs.get("tune")
    if tune is not None:
        return tune
    if nuts_sampler == "nutpie" and pymc_major() >= 6:
        return 400
    return 1000


def _warn_predictive_kwarg_drift(kind: str, kwargs: dict) -> None:
    """Flag predictive kwargs whose meaning changed in PyMC 6.

    PyMC 6 overhauled ``sample_posterior_predictive``: ``var_names`` now
    selects only which outputs are *stored*, while ``sample_vars`` /
    ``freeze_vars`` control what gets resampled. Code written against 5.x that
    passed ``var_names`` to force deterministics to be recomputed silently
    gets different results, so say so rather than let it pass.
    """
    from cloudposterior._idata import pymc_major

    if kind != "posterior" or pymc_major() < 6:
        return
    if kwargs.get("var_names") is not None and kwargs.get("sample_vars") is None:
        warnings.warn(
            "under PyMC 6, sample_posterior_predictive's var_names only "
            "selects which variables are stored -- it no longer decides what "
            "is resampled. Pass sample_vars= to control resampling.",
            stacklevel=3,
        )


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

    Drives memo invalidation in ``_ensure_model_bytes``: when this changes, the
    model is re-serialized so a ``pm.set_data`` mutation reaches the worker and
    produces a distinct cache key instead of silently reusing stale bytes.
    """
    from cloudposterior.naming import data_digest

    # Content hash, not a summary statistic: a summed fold was blind to a
    # permutation or a sign-symmetric edit of the same array.
    return data_digest(model) or None


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


def _emit_cached_indicator() -> None:
    """Show the one-line "cached result" indicator (notebook HTML or Rich)."""
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


def _cache_lookup(cache_arg, model, model_bytes: bytes, cache_kwargs: dict, *,
                  overwrite: bool, progress: bool):
    """Resolve the cache backend and look up an entry.

    Returns ``(backend, key, cached_or_None)``. ``overwrite=`` forces a re-run:
    the load is skipped but the key is still returned so the fresh result can
    replace the stored entry. Emits the "cached result" indicator on a hit.

    ``model_bytes`` is unused for identity -- see ``_model_identity``.
    """
    from cloudposterior.cache import resolve_cache
    from cloudposterior.naming import cache_key as compute_cache_key

    backend = resolve_cache(cache_arg, model=model)
    if backend is None:
        return None, None, None
    key = compute_cache_key(_model_identity(model), cache_kwargs)
    cached = None if overwrite else backend.load(key, sample_kwargs=cache_kwargs)
    if cached is not None and progress:
        _emit_cached_indicator()
    return backend, key, cached


def _run_sample(
    model,
    *,
    remote: bool,
    cache: bool | str,
    overwrite: bool = False,
    notify: bool | str,
    instance: str | None,
    nuts_sampler: str,
    progress: bool,
    original_sample,
    model_bytes: bytes | None = None,
    notify_topic_holder=None,
    **sample_kwargs,
) -> az.InferenceData:
    """Core sampling logic with cache, remote, and notification support."""
    if model_bytes is None:
        model_bytes = _ensure_model_bytes(model)

    # -- Check cache (include nuts_sampler so different samplers don't collide) --
    cache_kwargs = {**sample_kwargs, "nuts_sampler": nuts_sampler}
    cache_backend, cache_key, cached = _cache_lookup(
        cache, model, model_bytes, cache_kwargs,
        overwrite=overwrite, progress=progress,
    )
    if cached is not None:
        return cached

    # -- Build sinks (only needed for cache miss) --
    if remote:
        config = RemoteConfig.from_instance(instance, model=model, sample_kwargs=sample_kwargs, nuts_sampler=nuts_sampler)
        instance_desc = f"Modal ({config.describe()})"
    else:
        instance_desc = "local"

    # For local runs, skip progress display -- let PyMC show its native output.
    sinks = _build_sinks(
        progress=progress if remote else False,
        notify=notify,
        instance_desc=instance_desc,
        model=model,
        notify_topic_holder=notify_topic_holder,
    )

    def emit(event):
        dispatch_event(event, sinks)

    # -- Run sampling (sinks must be stopped even when it raises, or a
    # terminal Rich Live display would keep repainting over the traceback) --
    try:
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
        return idata
    finally:
        _stop_sinks(sinks)


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
                 stop_url: str | None = None, stop_token: str | None = None,
                 notify_topic_holder=None) -> list:
    """Create display, dashboard, and notification sinks.

    ``notify_topic_holder`` is the ``cloud`` instance (when there is one), used
    to reuse one auto-generated ntfy topic for the block: a fresh random topic
    per pm.sample() call meant a user who subscribed on their phone for the
    first run received nothing for the second.
    """
    sinks: list = []
    # Anything after a started display (a bad notify= value raises by design)
    # must not escape with the display still running: a live Rich Live keeps
    # its refresh thread going and paints over the traceback.
    try:
        if progress:
            from cloudposterior.display import (
                NotebookDisplay,
                TerminalDisplay,
                _is_marimo,
                _is_notebook,
            )

            display = None
            if _is_marimo() or _is_notebook():
                try:
                    display = NotebookDisplay(
                        instance_desc, stop_url=stop_url, stop_token=stop_token
                    )
                except Exception as exc:
                    # A broken anywidget/traitlets must not take down sampling.
                    warnings.warn(
                        f"cloudposterior: falling back to terminal progress "
                        f"display ({exc})",
                        stacklevel=2,
                    )
            if display is None:
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
            if auto_generated and notify_topic_holder is not None:
                # Reuse the block's topic across repeat pm.sample() calls.
                topic = getattr(notify_topic_holder, "_notify_topic", None)

            notifier = NtfyNotifier(
                topic=topic,
                server=server,
                model=model,
                instance_desc=instance_desc,
            )
            if auto_generated and notify_topic_holder is not None:
                notify_topic_holder._notify_topic = notifier.topic
            sinks.append(notifier)

            _show_link(notifier.url, label="Notifications", show_qr=auto_generated)
    except BaseException:
        _stop_sinks(sinks)
        raise

    return sinks


def _show_link(url: str, label: str = "Link", show_qr: bool = False):
    """Display a URL with optional QR code."""
    # HTML fragments for browser frontends (Jupyter + marimo). The SVG QR
    # renders in both; the terminal fallback prints an ASCII QR instead.
    import html as _html

    from cloudposterior.display import _emit_oneshot_html

    safe_url = _html.escape(url, quote=True)
    parts = [
        '<div style="font-family:monospace;font-size:12px;padding:4px 0;">',
        f'{_html.escape(label)}: <a href="{safe_url}" target="_blank">{safe_url}</a>',
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
            try:
                sink.stop()
            except Exception:
                # One sink failing to shut down must not prevent the others
                # from doing so -- a live Rich Live would wreck the terminal.
                pass


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
        cache_key = compute_cache_key(_model_identity(ctx.model), cache_kwargs)
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
    cache_backend=None,
    cache_key: str | None = None,
    cache_kwargs: dict | None = None,
    dashboard: bool = False,
    notify: bool | str | dict = False,
    nuts_sampler: str = "pymc",
    progress: bool,
    notify_topic_holder=None,
    **sample_kwargs,
) -> az.InferenceData:
    """Sampling via a persistent environment.

    Model payload is in the Volume. Per-call sends only kwargs + a path
    identifying which payload to load. The cache lookup already happened in
    the caller (before the env was provisioned -- a hit never creates Modal
    infra); ``cache_backend``/``cache_key`` are only used to save the result.
    """
    from cloudposterior.backends.modal_backend import _compute_payload_path

    # Read the actually-provisioned config from the env so the
    # displayed instance description matches what's running (no recomputation drift).
    config = getattr(env, "config", None) or RemoteConfig()
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
        notify_topic_holder=notify_topic_holder,
    )

    def emit(event):
        dispatch_event(event, sinks)

    # Sinks must be stopped even when the run raises (a worker error re-raised
    # mid-stream would otherwise leave a terminal Rich Live display running).
    try:
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
        stream = job.stream_progress()
        try:
            first = next(stream, _STREAM_DONE)
        except Exception as exc:
            if not _is_missing_payload_error(exc):
                raise
            # The Volume no longer holds the payload our upload memo claims is
            # there (another session pruned it, or cleanup_volumes ran from a
            # second kernel). Re-upload once and retry rather than failing.
            env.forget_upload(model_bytes)
            env._upload_if_needed(model_bytes, payload_path, force=True)
            job = env.submit(
                model_bytes, sample_kwargs, nuts_sampler, payload_path=payload_path
            )
            stream = job.stream_progress()
            first = next(stream, _STREAM_DONE)

        for event in _prepend(first, stream):
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

        return idata
    finally:
        _stop_sinks(sinks)


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
        progress_cb = make_sampling_callback(progress_queue, tune, draws)
        # pm.sample takes a single callback: compose ours with the user's
        # instead of passing callback= twice (TypeError).
        user_cb = sample_kwargs.pop("callback", None)
        if user_cb is not None:
            def callback(trace, draw, _progress_cb=progress_cb, _user_cb=user_cb):
                _progress_cb(trace, draw)
                _user_cb(trace, draw)
        else:
            callback = progress_cb
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

        try:
            with model:
                idata = original_sample(
                    callback=callback,
                    **sampler_kwargs,
                    **sample_kwargs,
                )
        finally:
            # On a sampling error these would otherwise never run, leaving the
            # aggregator blocked on its queue and the thread alive for the
            # rest of the session.
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
        # Not pm.sample: inside a cp.cloud block that is the interceptor, and
        # taking it as the "original" would recurse.
        original_sample=_true_original("sample"),
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
    model is uploaded once and the fits ``spawn`` concurrently; with no
    per-container input concurrency Modal fans them out across containers, so
    each fit is sized to *its own* model via ``Cls.with_options`` (cpu/memory)
    rather than to the first. Returns InferenceData in input order.

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
    from cloudposterior.naming import cache_key as compute_cache_key, model_slug
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
    for kw in kwargs_list:
        _validate_sample_kwargs(kw)

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
    # map therefore provisions nothing -- no env, no dashboard. Resolve one
    # backend per model: a disk cache derives its directory from the model's
    # slug, so a shared backend would file every result under models[0]. --
    cache_backends = [resolve_cache(cache, model=m) for m in models]
    results: list = [None] * n
    cached_idx: list[int] = []
    run_idx: list[int] = []
    # aligned with run_idx:
    #   (model_bytes, payload_path, kwargs, sampler, key, key_kwargs, label, config)
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
        if cache_backends[i] is not None:
            # Still compute the key (so the fresh result is saved) but skip the
            # load when overwrite= forces a re-run.
            ckey = compute_cache_key(_model_identity(m), ck_kwargs)
            if not overwrite:
                hit = cache_backends[i].load(ckey, sample_kwargs=ck_kwargs)
                if hit is not None:
                    results[i] = hit
                    cached_idx.append(i)
                    continue
        # Size each fit to its own model (each fans out to its own container);
        # narrowed per-spawn via with_options below.
        cfg = RemoteConfig.from_instance(
            instance, model=m, sample_kwargs=kwargs_list[i], nuts_sampler=samplers[i],
        )
        run_idx.append(i)
        run_meta.append(
            (mb, payload_path, kwargs_list[i], samplers[i], ckey, ck_kwargs, labels[i], cfg)
        )

    n_cached = len(cached_idx)
    if not run_idx:
        _log(f"all {n} model(s) cached")
        return results

    # -- At least one cache miss: provision the shared env (image + Volume, and
    # the dashboard/stop infra when on). Each fit fans out to its own container
    # (no per-container input concurrency), so size the shared image/gpu at the
    # batch high-water mark and narrow each spawn to its model via with_options. --
    run_configs = [meta[7] for meta in run_meta]
    base_config = RemoteConfig(
        cpu=max(c.cpu for c in run_configs),
        memory=max(c.memory for c in run_configs),
        timeout=max(c.timeout for c in run_configs),
        gpu=next((c.gpu for c in run_configs if c.gpu), None),
        auto_sized=any(c.auto_sized for c in run_configs),
    )
    backend = ModalBackend(config=base_config)
    # The shared image must satisfy every fit in the batch: if any model uses
    # a JAX sampler the image needs jax/numpyro installed.
    run_samplers = [meta[3] for meta in run_meta]
    image_sampler = next(
        (s for s in run_samplers if s in ("numpyro", "blackjax")), run_samplers[0]
    )
    env = backend.provision(
        model_bytes_list[0], models[0], get_version_manifest(), base_config,
        project=project, idle_timeout=base_config.idle_timeout,
        dashboard=dashboard_on, stop_enabled=dashboard_on,
        nuts_sampler=image_sampler,
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
    calls: list = []
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
        if any(meta[7].auto_sized for meta in run_meta):
            sizes = ", ".join(f"{m[7].cpu:.0f}c/{m[7].memory // 1024}GB" for m in run_meta)
            _log(f"per-model sizing: {sizes}")
        if dashboard_on and env._dashboard_url:
            _log(f"dashboard: {env._dashboard_url}")

        # Upload the miss payloads now that the env is running.
        for (mb, payload_path, _kw, _sp, _ck, _cak, _lbl, _cfg) in run_meta:
            env._upload_if_needed(mb, payload_path)

        dash_name = env._dashboard_dict_name if dashboard_on else None
        # spawn() is a blocking Modal call -- run it off the event loop so it
        # doesn't warn/stall inside an async host (marimo), mirroring how the
        # .get() result fetch below is already wrapped. with_options sizes each
        # spawned container to its own model (cpu/memory/gpu) -- without the gpu
        # override every fit in a mixed batch inherited the batch's GPU; the
        # image and Volume stay shared. Each worker writes its own progress and
        # honors global/per-model stop.
        calls[:] = [
            _run_blocking(
                env._sampler_cls.with_options(
                    cpu=cfg.cpu, memory=cfg.memory, gpu=cfg.gpu
                )()
                .sample_blocking.spawn,
                pp, kw, sp,
                progress_dict_name=dash_name,
                progress_key=(lbl if dashboard_on else None),
                stop_dict_name=dash_name,
            )
            for (_mb, pp, kw, sp, _ck, _cak, lbl, cfg) in run_meta
        ]
        for j, i in enumerate(run_idx):
            idata = deserialize_inference_data(_run_blocking(calls[j].get))
            results[i] = idata
            _log(f"[{j + 1}/{len(run_idx)}] done")
            _mb, _pp, _kw, _sp, ckey, ck_kwargs, _lbl, _cfg = run_meta[j]
            if cache_backends[i] is not None and ckey is not None:
                cache_backends[i].save(ckey, idata, sample_kwargs=ck_kwargs)

        if dashboard_on:
            # Keep the env warm (reusing the cp.cloud registry) so the dashboard
            # stays browsable after the run; Modal idles it out via
            # scaledown_window / the atexit hook. Tear down any *other* env
            # already registered under this key first -- silently overwriting
            # it would leak a warm container until its scaledown window.
            key = _env_key(project, models[0])
            previous = _LIVE_ENVS.get(key)
            if previous is not None and previous is not env:
                try:
                    previous.teardown()
                except Exception:
                    pass
            # A later cp.cloud run reusing this env reads env.config back (for
            # reuse checks and resize-drift reporting), so it must be recorded
            # here too -- not only on the cp.cloud provisioning path.
            if getattr(env, "config", None) is None:
                env.config = base_config
            with _PATCH_LOCK:
                _LIVE_ENVS[key] = env
            teardown = False
        return results
    except BaseException:
        # Don't leave the other spawned containers billing after a failure.
        for call in calls:
            try:
                call.cancel()
            except Exception:
                pass
        raise
    finally:
        if teardown:
            env.teardown()
