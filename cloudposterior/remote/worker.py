"""Remote worker that runs on Modal.

This module defines the Modal function that deserializes a PyMC model,
runs sampling with progress tracking, and streams results back.

It is NOT imported locally -- Modal serializes and runs it remotely.
The image is constructed dynamically based on the version manifest.

Live progress support depends on the sampler:
- ``pymc``  -- per-draw progress + convergence via PyMC's per-draw callback.
- ``nutpie`` -- per-draw progress + convergence via nutpie's native
  ``progress_callback`` and background sampler (the default fast sampler).
- ``numpyro``/``blackjax`` -- run entirely inside JAX with no per-draw hook,
  so only phase-level progress is available. PyMC raises if you pass a
  ``callback`` to these, so we never do.
"""

from __future__ import annotations

import time
from importlib import import_module

from pymc_monitor import policies as _policies

from cloudposterior._idata import add_group as _add_group, get_group as _get_group

# Consecutive nutpie is_finished failures tolerated before falling through to
# handle.wait(). At the 0.5s poll interval this is ~5s of persistent failure.
_MAX_POLL_FAILURES = 10


def _open_dict(name):
    """Resolve a Modal Dict by name (stop control or dashboard progress), if any."""
    if not name:
        return None
    try:
        import modal

        return modal.Dict.from_name(name)
    except Exception:
        return None


def _stop_requested(stop_dict, stop_key=None) -> bool:
    """Best-effort read of the stop flag. Called from the generator loop only
    (never the per-draw sampling hot loop) so the network round-trip can't slow
    sampling. Honors the global ``stop`` flag and, for cp.map, a per-model
    ``stop:<stop_key>`` flag set by the dashboard's per-model Stop button."""
    if stop_dict is None:
        return False
    try:
        if bool(stop_dict.get("stop", False)):
            return True
        if stop_key and bool(stop_dict.get(f"stop:{stop_key}", False)):
            return True
        return False
    except Exception:
        return False


def _resolve_chains(chains, cores) -> int:
    """The chain count pm.sample would use (it defaults to max(2, cores<=4)).

    Resolved here rather than left to the sampler so the convergence layer
    knows how many chains to wait for, and so nutpie (whose own default is 6)
    matches a local pm.sample run.
    """
    if chains is not None:
        return int(chains)
    import os

    c = cores if cores is not None else min(4, os.cpu_count() or 4)
    return max(2, int(c))


def _sub_seeds(random_seed, n: int) -> list:
    """n derived integer seeds (None when the run is unseeded)."""
    import numpy as np

    if random_seed is None or not isinstance(random_seed, (int, np.integer)):
        rng = np.random.default_rng()
    else:
        rng = np.random.default_rng(int(random_seed))
    return [int(s) for s in rng.integers(0, 2**31 - 1, size=n)]


def _monitored_names(model, until) -> list[str]:
    names = (until or {}).get("var_names")
    if names:
        return list(names)
    return [rv.name for rv in getattr(model, "free_RVs", [])]


def _scalar_traces(draws, max_points: int = 500) -> dict:
    """Downsampled per-chain traces of the scalar variables, for the dashboard."""
    counts: dict[str, int] = {}
    for name in draws.var_of:
        counts[name] = counts.get(name, 0) + 1
    traces = {}
    for i, name in enumerate(draws.var_of):
        if counts[name] != 1:
            continue
        vals = draws.values[:, :, i]
        if vals.shape[1] > max_points:
            step = max(1, vals.shape[1] // max_points)
            vals = vals[:, ::step][:, :max_points]
        traces[name] = [[float(v) for v in row] for row in vals]
    return traces


def _convergence_event(assessment: dict, draws, *, total_draws: int, final: bool) -> bytes:
    import msgpack

    from cloudposterior.convergence import _jsonable

    verdict = {
        k: assessment.get(k)
        for k in ("met", "decision", "consecutive", "projected_draws", "eta_seconds",
                  "cause", "hint", "next_check", "checks")
    }
    return msgpack.packb(_jsonable({
        "type": "convergence",
        "params": assessment.get("params", {}),
        "draws": assessment.get("draws", 0),
        "total_draws": total_draws,
        "traces": _scalar_traces(draws),
        "partial": not final,
        "verdict": verdict,
    }))


def _parts_to_draws(parts, names, superchain_ids):
    """Draws from one or more partial traces (superchains run as separate
    nutpie samplers), padding shorter parts with NaN so the convergence layer
    trims every chain to the shortest complete one."""
    import numpy as np

    from cloudposterior import convergence as cv

    if len(parts) == 1:
        return cv.draws_from_idata(parts[0], var_names=names, superchain_ids=superchain_ids)
    pieces = [cv.draws_from_idata(p, var_names=names) for p in parts]
    n = min(p.n_draws for p in pieces)
    values = np.concatenate([p.values[:, :n] for p in pieces], axis=0)
    first = pieces[0]
    out = cv.Draws(
        values=values, labels=list(first.labels), var_of=list(first.var_of),
        superchain_ids=list(superchain_ids) if superchain_ids is not None else None,
    )
    # Stats: re-trim each part to n so the counts cover the same draws.
    divs, energies, mds = [], [], []
    for p in parts:
        st = _get_group(p, "sample_stats")
        if st is None:
            continue
        if "diverging" in st.data_vars:
            divs.append(np.asarray(st["diverging"].values)[:, :n])
        if "energy" in st.data_vars:
            energies.append(np.asarray(st["energy"].values, dtype=float)[:, :n])
        for key in ("maxdepth_reached", "reached_max_treedepth"):
            if key in st.data_vars:
                mds.append(np.asarray(st[key].values)[:, :n])
                break
    if len(divs) == len(parts):
        out.diverging = int(np.nansum(np.concatenate(divs, axis=0).astype(float)))
    if len(energies) == len(parts):
        out.energy = np.concatenate(energies, axis=0)
    if len(mds) == len(parts):
        out.maxdepth_frac = float(np.nanmean(np.concatenate(mds, axis=0).astype(float)))
    return out


# Historical private imports remain aliases to the monitor's policy implementation.
_LADDER = _policies.LADDER
_next_rung = _policies.next_rung


def _sample_and_stream(model, sample_kwargs, nuts_sampler="nutpie", stop_dict_name=None,
                       stop_key=None):
    """Adapt the shared monitor generator to the existing Modal wire protocol."""
    import msgpack
    from pymc_monitor.control import RunControl
    from pymc_monitor.engine import iter_sample
    from pymc_monitor.events import event_to_dict

    from cloudposterior.serialize import serialize_inference_data

    kwargs = dict(sample_kwargs)
    target = kwargs.pop("until", None)
    initial_strategy = kwargs.pop("cp_init", None)
    if kwargs.pop("_cp_sampler_auto", False):
        nuts_sampler = None
    if kwargs.pop("_cp_tune_default", False):
        kwargs.pop("tune", None)
    kwargs.pop("progressbar", None)
    kwargs.pop("callback", None)  # client warned: local callbacks cannot run remotely
    kwargs.pop("return_inferencedata", None)  # MultiTrace cannot cross the wire
    stop_dict = _open_dict(stop_dict_name)
    control = RunControl(stop_requested=lambda: _stop_requested(stop_dict, stop_key))
    stream = iter_sample(model, kwargs, nuts_sampler=nuts_sampler, until=target,
                         initial_strategy=initial_strategy, control=control)
    try:
        while True:
            try:
                event = next(stream)
            except StopIteration as done:
                result = done.value
                break
            yield msgpack.packb(event_to_dict(event), use_bin_type=True)
    finally:
        stream.close()

    idata = result.idata
    from cloudposterior.convergence import attach_report
    attach_report(idata, result.report)
    from cloudposterior._idata import get_group
    post = get_group(idata, "posterior")
    if post is not None and "pymc_monitor_superchains" in post.attrs:
        post.attrs["cloudposterior_superchains"] = post.attrs["pymc_monitor_superchains"]

    compressed = serialize_inference_data(idata)
    yield msgpack.packb({"type": "result", "size_mb": round(len(compressed) / 1048576, 2)})
    yield compressed


_fixed_point = _policies.fixed_point
_split_nutpie_kwargs = _policies.split_nutpie_kwargs


def _unpack_model_payload(obj, sample_kwargs: dict, nuts_sampler: str):
    """Resolve a model payload that may bundle a step method.

    A combined ``{"model", "step"}`` payload (see ``serialize_model_with_step``)
    keeps the step's value variables identity-linked to the model. When a step
    is present it is injected into ``sample_kwargs`` and the pymc sampler is
    forced -- only it honors a custom ``step=`` and the per-draw progress
    callback. Returns ``(model, sample_kwargs, nuts_sampler)``.
    """
    if isinstance(obj, dict) and "model" in obj:
        model = obj["model"]
        step = obj.get("step")
        if step is not None:
            sample_kwargs = {**sample_kwargs, "step": step}
            nuts_sampler = "pymc"
        return model, sample_kwargs, nuts_sampler
    return obj, sample_kwargs, nuts_sampler


def run_sampling(
    model_bytes: bytes,
    sample_kwargs: dict,
    nuts_sampler: str = "nutpie",
    persistent: bool = False,
):
    """One-shot path: deserialize model from bytes and run sampling.

    Observed data is bundled inside the cloudpickled model, so no separate
    data payload is needed.
    """
    import msgpack

    phase_start = time.time()
    obj = _deserialize_payload(model_bytes)
    model, sample_kwargs, nuts_sampler = _unpack_model_payload(
        obj, sample_kwargs, nuts_sampler
    )

    elapsed = time.time() - phase_start
    phase_name = "container_ready" if persistent else "provisioning"
    phase_message = "container ready" if persistent else "environment ready"
    yield msgpack.packb({
        "type": "phase", "phase": phase_name, "status": "done",
        "message": phase_message, "elapsed": elapsed,
    })

    yield from _sample_and_stream(model, sample_kwargs, nuts_sampler)


def run_sampling_from_volume(
    payload_path: str,
    sample_kwargs: dict,
    nuts_sampler: str = "nutpie",
    stop_dict_name: str | None = None,
):
    """Persistent path: load model from Volume and run sampling."""
    import msgpack

    phase_start = time.time()
    with open(payload_path, "rb") as f:
        obj = _deserialize_payload(f.read())
    model, sample_kwargs, nuts_sampler = _unpack_model_payload(
        obj, sample_kwargs, nuts_sampler
    )

    elapsed = time.time() - phase_start
    yield msgpack.packb({
        "type": "phase", "phase": "container_ready", "status": "done",
        "message": "model loaded from volume", "elapsed": elapsed,
    })

    yield from _sample_and_stream(model, sample_kwargs, nuts_sampler, stop_dict_name=stop_dict_name)


def _deserialize_payload(data: bytes):
    """Unpickle a model payload, turning version skew into a readable error.

    A cloudpickle/pymc mismatch between the client that pickled the model and
    this container otherwise surfaces as a bare pickle error with no hint that
    versions are the problem.
    """
    from cloudposterior.serialize import deserialize_model

    try:
        return deserialize_model(data)
    except Exception as exc:
        versions = []
        for name in ("cloudpickle", "pymc", "pytensor", "numpy"):
            try:
                versions.append(f"{name}={import_module(name).__version__}")
            except Exception:
                versions.append(f"{name}=?")
        raise RuntimeError(
            f"could not deserialize the model payload ({exc}). The container "
            f"has {', '.join(versions)}; this usually means the local "
            "environment that pickled the model has different versions."
        ) from exc


def _load_model_from_volume(payload_path: str):
    with open(payload_path, "rb") as f:
        return _deserialize_payload(f.read())


def run_prior_predictive(payload_path: str, sample_kwargs: dict) -> bytes:
    """Load model from Volume, run prior predictive, return lz4 NetCDF bytes.

    A deterministic forward pass -- no MCMC, no per-chain streaming.
    """
    import pymc as pm

    from cloudposterior.serialize import serialize_inference_data

    model = _load_model_from_volume(payload_path)
    with model:
        idata = pm.sample_prior_predictive(**sample_kwargs)
    return serialize_inference_data(idata)


def run_posterior_predictive(payload_path: str, idata_bytes: bytes, sample_kwargs: dict) -> bytes:
    """Load model + posterior trace from Volume/args, run posterior predictive."""
    import pymc as pm

    from cloudposterior.serialize import deserialize_inference_data, serialize_inference_data

    model = _load_model_from_volume(payload_path)
    trace = deserialize_inference_data(idata_bytes)
    # Always compute standalone: extending the worker's deserialized copy of
    # the trace would be invisible to the caller. The client merges the new
    # group(s) into its own idata when extend_inferencedata=True was requested.
    sample_kwargs.pop("extend_inferencedata", None)
    with model:
        idata = pm.sample_posterior_predictive(
            trace, extend_inferencedata=False, **sample_kwargs
        )
    return serialize_inference_data(idata)


def run_smc(payload_path: str, sample_kwargs: dict) -> bytes:
    """Load model from Volume, run pm.sample_smc, return lz4 NetCDF bytes.

    SMC exposes no per-draw callback, so this is a blocking call (no streaming).
    """
    import pymc as pm

    from cloudposterior.serialize import serialize_inference_data

    sample_kwargs.pop("model", None)
    model = _load_model_from_volume(payload_path)
    with model:
        idata = pm.sample_smc(**sample_kwargs)
    return serialize_inference_data(idata)


def run_compute_log_likelihood(
    payload_path: str, idata_bytes: bytes, sample_kwargs: dict
) -> bytes:
    """Load model + trace, run pm.compute_log_likelihood, return lz4 NetCDF bytes.

    Always computes with ``extend_inferencedata=False`` so only the (small)
    log_likelihood group travels back; the client decides whether to merge it
    into the caller's idata in place (native default) or return it standalone.
    """
    import arviz as az
    import pymc as pm

    from cloudposterior.serialize import (
        deserialize_inference_data,
        serialize_inference_data,
    )

    sample_kwargs.pop("model", None)
    sample_kwargs.pop("extend_inferencedata", None)
    model = _load_model_from_volume(payload_path)
    idata = deserialize_inference_data(idata_bytes)
    with model:
        ll = pm.compute_log_likelihood(idata, extend_inferencedata=False, **sample_kwargs)
    # extend_inferencedata=False yields a bare Dataset; wrap it as a
    # log_likelihood group so it travels over the NetCDF path. arviz 0.x accepts
    # the group as a constructor kwarg; arviz 1.x (DataTree) rejects it, so fall
    # back to attaching it via the version-agnostic add_group shim.
    if not hasattr(ll, "groups"):
        try:
            ll = az.InferenceData(log_likelihood=ll)
        except TypeError:
            wrapped = az.InferenceData()
            _add_group(wrapped, "log_likelihood", ll)
            ll = wrapped
    return serialize_inference_data(ll)


def run_sampling_blocking(payload_path: str, sample_kwargs: dict,
                          nuts_sampler: str = "nutpie",
                          progress_dict_name: str | None = None,
                          progress_key: str | None = None,
                          stop_dict_name: str | None = None) -> bytes:
    """Run sampling to completion and return lz4 NetCDF bytes (no streaming).

    Used by cp.map via Modal spawn -- spawn can't stream, so we drive the
    streaming generator here and keep only the final result chunk. When a
    dashboard Dict is provided, the decoded progress events are written into it
    server-side under ``progress_key`` (this model's label), reusing the very
    same DashboardSink the client uses for single-model runs. ``stop_dict_name``
    + ``progress_key`` also wire up the dashboard's global / per-model Stop.
    """
    import msgpack

    from cloudposterior.progress import decode_progress_event, dispatch_event

    # Server-side dashboard sink: write this model's progress under its own key
    # so N concurrent map workers never clobber each other.
    sink = None
    if progress_dict_name and progress_key:
        from cloudposterior.dashboard import DashboardSink

        progress_dict = _open_dict(progress_dict_name)
        if progress_dict is not None:
            sink = DashboardSink(progress_dict, key=progress_key)

    model = _load_model_from_volume(payload_path)
    idata_bytes = None
    expecting_result = False
    for chunk in _sample_and_stream(
        model, sample_kwargs, nuts_sampler,
        stop_dict_name=stop_dict_name, stop_key=progress_key,
    ):
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
                if sink is not None and isinstance(decoded, dict):
                    # Best-effort: a sink hiccup must not corrupt result detection.
                    try:
                        dispatch_event(decode_progress_event(decoded), [sink])
                    except Exception:
                        pass
            if not decoded_any:
                idata_bytes = chunk
        except Exception:
            idata_bytes = chunk

    # Mark the run complete for the dashboard. Completion is signalled by a
    # terminal DOWNLOADING phase, which on single-model runs the *client*
    # emits after the stream -- but a cp.map sink lives here in the worker, so
    # without this the page never sees complete=true and polls a billed
    # endpoint at 1 Hz forever.
    if sink is not None:
        try:
            from cloudposterior.progress import JobPhase, PhaseUpdate

            sink.show_phase(PhaseUpdate(
                phase=JobPhase.DOWNLOADING,
                status="done",
                message="trace ready",
                elapsed=0.0,
            ))
        except Exception:
            pass

    if idata_bytes is None:
        raise RuntimeError("cp.map worker produced no result")
    return idata_bytes
