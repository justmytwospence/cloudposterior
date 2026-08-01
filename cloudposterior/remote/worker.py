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
from queue import Queue
from threading import Thread

from cloudposterior._idata import (
    add_group as _add_group,
    ess_tail as _ess_tail,
    get_group as _get_group,
    sanitize_inference_data as _sanitize_idata_attrs,
    to_inference_data as _to_inference_data,
)

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


def _sample_and_stream(model, sample_kwargs, nuts_sampler="nutpie", stop_dict_name=None,
                       stop_key=None):
    """Run MCMC sampling and yield msgpack-encoded progress + results.

    Shared core logic used by both one-shot and persistent paths. The caller
    loads the model; this function handles compilation, sampling, progress
    streaming, and result serialization.
    """
    import lz4.frame
    import msgpack
    import pymc as pm

    # -- Check JAX device for GPU samplers --
    if nuts_sampler in ("numpyro", "blackjax"):
        yield msgpack.packb({
            "type": "phase", "phase": "device", "status": "in_progress",
            "message": "initializing JAX", "elapsed": 0.0,
        })
        jax_start = time.time()
        try:
            import jax

            devices = jax.devices()
            device_types = [d.platform for d in devices]
            if "gpu" in device_types:
                gpu_devices = [d for d in devices if d.platform == "gpu"]
                device_msg = f"JAX using GPU ({gpu_devices[0].device_kind})"
            else:
                device_msg = "JAX using CPU (no GPU found)"
        except Exception as e:
            device_msg = f"JAX device check failed: {e}"
        yield msgpack.packb({
            "type": "phase", "phase": "device", "status": "done",
            "message": device_msg, "elapsed": time.time() - jax_start,
        })

    # -- Pop the kwargs the worker controls; the rest pass through to pm.sample --
    draws = sample_kwargs.pop("draws", 1000)
    user_tune = sample_kwargs.pop("tune", None)
    tune = user_tune if user_tune is not None else 1000
    chains = sample_kwargs.pop("chains", None)
    cores = sample_kwargs.pop("cores", None)
    random_seed = sample_kwargs.get("random_seed", None)
    sample_kwargs.pop("progressbar", None)
    sample_kwargs.pop("callback", None)
    # The client warns that return_inferencedata=False is unsupported remotely
    # (a MultiTrace can't be transported back); drop it here so the pymc path
    # doesn't produce a MultiTrace that crashes serialization after sampling.
    sample_kwargs.pop("return_inferencedata", None)
    # Adaptive: early-stop once every scalar param hits this convergence target.
    # cp-only kwarg (never passed to the sampler); applies to nutpie + pymc.
    until = sample_kwargs.pop("until", None)

    # Convergence is evaluated over scalar (chain, draw) parameters only. On a
    # purely vector-valued model there are none, so until= can never fire --
    # say so rather than silently running to the draw cap.
    if until and not _has_scalar_free_rvs(model):
        until = None
        yield msgpack.packb({
            "type": "phase", "phase": "sampling", "status": "in_progress",
            "message": (
                "until= has no scalar parameters to evaluate on this model; "
                "adaptive early-stop disabled, sampling the full draws"
            ),
            "elapsed": 0.0,
        })

    # -- nutpie compile phase, with fallback to the pymc sampler --
    compiled = None
    if nuts_sampler == "nutpie":
        compile_start = time.time()
        yield msgpack.packb({
            "type": "phase", "phase": "compiling", "status": "in_progress",
            "message": "compiling model with nutpie", "elapsed": 0.0,
        })
        try:
            import nutpie

            compiled = nutpie.compile_pymc_model(model)
            yield msgpack.packb({
                "type": "phase", "phase": "compiling", "status": "done",
                "message": "nutpie compilation complete",
                "elapsed": time.time() - compile_start,
            })
        except Exception as exc:
            nuts_sampler = "pymc"
            yield msgpack.packb({
                "type": "phase", "phase": "compiling", "status": "done",
                "message": f"nutpie unavailable ({exc}); falling back to the pymc sampler",
                "elapsed": time.time() - compile_start,
            })

    # -- Sampling phase --
    yield msgpack.packb({
        "type": "phase", "phase": "sampling", "status": "in_progress",
        "message": "MCMC sampling started", "elapsed": 0.0,
    })

    progress_queue: Queue = Queue()
    chain_traces: dict[int, dict[str, list]] = {}  # pymc path: chain -> {param: [post-warmup values]}
    all_chain_states: dict[str, dict] = {}
    counters = {"total_draws": 0, "last_conv_draws": 0}
    sample_start = time.time()
    stop_dict = _open_dict(stop_dict_name)

    sampling_error = None
    idata = None
    stopped_early = False
    converged = False

    def _drain_and_yield():
        """Collapse queued per-chain updates into a single 'sampling' event."""
        updated = False
        while not progress_queue.empty():
            try:
                update = progress_queue.get_nowait()
                all_chain_states[str(update["chain"])] = update
                updated = True
            except Exception:
                break
        if all_chain_states and updated:
            total_div = sum(c["divergences"] for c in all_chain_states.values())
            return msgpack.packb({
                "type": "sampling",
                "chains": dict(all_chain_states),
                "total_divergences": total_div,
                "elapsed": round(time.time() - sample_start, 1),
                "total_draws": counters["total_draws"],
            })
        return None

    def _convergence_from_idata(partial, *, every_draws=50):
        """Compute rank-normalized R-hat / bulk+tail ESS on the post-warmup
        partial trace (scalar params only). Used by both nutpie (via inspect())
        and as the shared formatter. Vehtari 2021 thresholds (1.01 / 400) are
        applied in the display, not here."""
        import arviz as az

        post = _get_group(partial, "posterior")
        if post is None:
            return None
        ndraw = int(post.sizes.get("draw", 0))
        nchain = int(post.sizes.get("chain", 0))
        if ndraw < every_draws or nchain < 2:
            return None
        convergence = {}
        traces = {}
        max_trace_points = 500
        for name in post.data_vars:
            da = post[name]
            if da.ndim != 2:  # (chain, draw) scalar params only
                continue
            try:
                convergence[name] = {
                    "rhat": round(float(az.rhat(da)), 4),
                    "ess_bulk": round(float(az.ess(da))),
                    "ess_tail": round(_ess_tail(da)),
                }
            except Exception:
                continue
            vals = da.values  # (chain, draw)
            if vals.shape[1] > max_trace_points:
                step = max(1, vals.shape[1] // max_trace_points)
                vals = vals[:, ::step][:, :max_trace_points]
            traces[name] = [row.tolist() for row in vals]
        if convergence:
            return msgpack.packb({
                "type": "convergence",
                "params": convergence,
                "draws": ndraw,
                "total_draws": counters["total_draws"],
                "traces": traces,
                "partial": True,
            })
        return None

    # ===================== nutpie: native progress + background sampler =====================
    if nuts_sampler == "nutpie":
        def nutpie_cb(progress_list):
            total = 0
            for i, p in enumerate(progress_list):
                total_draws = int(p.total_draws or 0)
                finished = int(p.finished_draws or 0)
                tune_used = max(0, total_draws - draws)
                if p.tuning:
                    cur, tot, phase = finished, tune_used, "tuning"
                else:
                    cur, tot, phase = max(0, finished - tune_used), draws, "sampling"
                total += finished
                elapsed = float(p.runtime_ms or 0) / 1000.0
                dps = finished / elapsed if elapsed > 0 else 0.0
                remaining = max(0, total_draws - finished)
                eta = remaining / dps if dps > 0 else 0.0
                progress_queue.put({
                    "chain": i,
                    "draw": cur,
                    "total": tot,
                    "phase": phase,
                    "draws_per_sec": round(dps, 1),
                    "eta_seconds": round(eta, 1),
                    "divergences": int(p.divergences or 0),
                    "mean_tree_depth": 0.0,
                    "step_size": round(float(p.step_size or 0.0), 4),
                    "tree_size": int(p.num_steps or 0),
                })
            counters["total_draws"] = total

        handle = None
        nutpie_kwargs = dict(
            draws=draws,
            # Match pm.sample's tune default (1000); nutpie's own default (~300)
            # would silently diverge from an equivalent local run.
            tune=tune,
            chains=chains,
            cores=cores,
            seed=random_seed,
            save_warmup=False,
            progress_bar=False,
            blocking=False,
        )
        # Forward the remaining pm.sample kwargs nutpie understands (e.g.
        # target_accept, maxdepth) instead of silently dropping them; report
        # whatever nutpie can't accept so the user knows it was ignored.
        forwarded, dropped = _split_nutpie_kwargs(nutpie, sample_kwargs)
        forwarded.pop("random_seed", None)  # already mapped to seed=
        dropped.pop("random_seed", None)
        nutpie_kwargs.update(forwarded)
        if dropped:
            yield msgpack.packb({
                "type": "phase", "phase": "sampling", "status": "in_progress",
                "message": "ignoring kwargs unsupported by nutpie: "
                           + ", ".join(sorted(dropped)),
                "elapsed": 0.0,
            })
        try:
            handle = nutpie.sample(compiled, progress_callback=nutpie_cb, **nutpie_kwargs)
        except Exception:
            # Some nutpie builds (e.g. 0.16.8) accept progress_callback but fail to
            # wire it ("'PyNutsSettings' object has no attribute
            # 'progress_callback'"). Fall back to sampling without live per-draw
            # progress rather than aborting the whole run.
            try:
                handle = nutpie.sample(compiled, **nutpie_kwargs)
            except Exception as exc:
                sampling_error = exc

        if handle is not None:
            _poll_failures = {"n": 0}

            def _finished():
                # nutpie's is_finished property raises (e.g. TimeoutError) while
                # the sampler is still running, so treat a raise as "not done" --
                # but only up to a point. A property that raises persistently
                # would otherwise spin at 0.5s until the container timeout kills
                # the run with no result at all.
                try:
                    done = bool(handle.is_finished)
                except Exception:
                    _poll_failures["n"] += 1
                    return _poll_failures["n"] >= _MAX_POLL_FAILURES
                _poll_failures["n"] = 0
                return done

            while not _finished():
                time.sleep(0.5)
                snap = _drain_and_yield()
                if snap:
                    yield snap
                if counters["total_draws"] - counters["last_conv_draws"] >= 100:
                    counters["last_conv_draws"] = counters["total_draws"]
                    try:
                        conv = _convergence_from_idata(_to_inference_data(handle.inspect()))
                        if conv:
                            yield conv
                            if until and _conv_meets_target(conv, until):
                                idata = _to_inference_data(handle.abort())
                                stopped_early = True
                                converged = True
                    except Exception:
                        pass
                    if converged:
                        break
                if _stop_requested(stop_dict, stop_key):
                    try:
                        idata = _to_inference_data(handle.abort())  # returns partial trace
                        stopped_early = True
                    except Exception:
                        pass
                    break

            if idata is None:
                try:
                    idata = _to_inference_data(handle.wait())
                except Exception as exc:
                    sampling_error = exc
            snap = _drain_and_yield()
            if snap:
                yield snap

    # ===================== numpyro/blackjax: JAX, phase-level progress only =====================
    elif nuts_sampler in ("numpyro", "blackjax"):
        yield msgpack.packb({
            "type": "phase", "phase": "sampling", "status": "in_progress",
            "message": f"sampling on {nuts_sampler} (JAX) -- live per-chain progress unavailable",
            "elapsed": round(time.time() - sample_start, 1),
        })

        def do_sample_jax():
            nonlocal idata, sampling_error
            try:
                with model:
                    idata = pm.sample(
                        draws=draws, tune=tune, chains=chains,
                        nuts_sampler=nuts_sampler, progressbar=False, **sample_kwargs,
                    )
            except Exception as exc:
                sampling_error = exc

        jax_thread = Thread(target=do_sample_jax)
        jax_thread.start()
        stop_noted = False
        while jax_thread.is_alive():
            time.sleep(0.5)
            # JAX samplers expose no per-draw hook, so an early stop can't
            # abort the run -- at least acknowledge the request once instead
            # of silently ignoring the dashboard's Stop button.
            if not stop_noted and _stop_requested(stop_dict, stop_key):
                stop_noted = True
                yield msgpack.packb({
                    "type": "phase", "phase": "sampling", "status": "in_progress",
                    "message": f"stop requested -- {nuts_sampler} (JAX) cannot "
                               "stop early; waiting for completion",
                    "elapsed": round(time.time() - sample_start, 1),
                })
        jax_thread.join()

    # ===================== pymc: per-draw callback (only sampler that supports it) =====================
    else:
        chain_draw_counts: dict[int, int] = {}
        chain_start_times: dict[int, float] = {}
        chain_divergences: dict[int, int] = {}
        chain_tree_depths: dict[int, list[float]] = {}
        chain_phase: dict[int, bool] = {}
        should_stop = {"v": False}

        def progress_callback(trace, draw):
            # NOTE: the draw counting / phase reset / windowed tree-depth logic
            # mirrors make_sampling_callback in cloudposterior/progress.py
            # (used for local runs) -- keep the two in sync.
            # Cheap local flag only -- the network poll lives in the generator loop (D3).
            if should_stop["v"]:
                raise KeyboardInterrupt("early stop requested")
            counters["total_draws"] += 1
            chain = draw.chain
            is_tuning = draw.tuning

            if chain not in chain_start_times:
                chain_start_times[chain] = time.time()
                chain_draw_counts[chain] = 0
                chain_divergences[chain] = 0
                chain_tree_depths[chain] = []
                chain_phase[chain] = is_tuning

            if chain_phase.get(chain) and not is_tuning:
                chain_draw_counts[chain] = 0
                chain_start_times[chain] = time.time()
                chain_phase[chain] = False

            chain_draw_counts[chain] += 1
            current_draw = chain_draw_counts[chain]

            stats = draw.stats[0] if draw.stats else {}
            diverging = stats.get("diverging", False)
            tree_depth = stats.get("tree_depth", 0)
            tree_size = stats.get("n_steps", stats.get("tree_size", 0))
            step_size = stats.get("step_size", 0.0)

            if diverging:
                chain_divergences[chain] += 1
            chain_tree_depths[chain].append(tree_depth)

            chain_elapsed = time.time() - chain_start_times[chain]
            dps = current_draw / chain_elapsed if chain_elapsed > 0 else 0.0
            total = tune if is_tuning else draws
            remaining = total - current_draw
            eta = remaining / dps if dps > 0 else 0.0
            mean_td = sum(chain_tree_depths[chain][-100:]) / min(len(chain_tree_depths[chain]), 100)

            # Accumulate POST-WARMUP scalar params for live convergence (C3).
            if not is_tuning and hasattr(draw, "point") and draw.point:
                import numpy as _np

                ct = chain_traces.setdefault(chain, {})
                for param_name, value in draw.point.items():
                    val = _np.asarray(value)
                    if val.ndim == 0:
                        ct.setdefault(param_name, []).append(float(val))

            progress_queue.put({
                "chain": chain,
                "draw": current_draw,
                "total": total,
                "phase": "tuning" if is_tuning else "sampling",
                "draws_per_sec": round(dps, 1),
                "eta_seconds": round(eta, 1),
                "divergences": chain_divergences[chain],
                "mean_tree_depth": round(mean_td, 1),
                "step_size": round(step_size, 4),
                "tree_size": tree_size,
            })

        def do_sample_pymc():
            nonlocal idata, sampling_error, stopped_early
            try:
                with model:
                    idata = pm.sample(
                        draws=draws, tune=tune, chains=chains, cores=cores,
                        nuts_sampler="pymc",  # force the pure-pymc sampler: only it
                        callback=progress_callback,  # supports the per-draw callback
                        progressbar=False, **sample_kwargs,
                    )
            except KeyboardInterrupt:
                stopped_early = True  # PyMC preserves the partial trace in idata
            except Exception as exc:
                sampling_error = exc

        sample_thread = Thread(target=do_sample_pymc)
        sample_thread.start()
        while sample_thread.is_alive():
            time.sleep(0.5)
            snap = _drain_and_yield()
            if snap:
                yield snap
            conv = _pymc_convergence(chain_traces, counters, msgpack)
            if conv:
                yield conv
                if until and _conv_meets_target(conv, until):
                    converged = True
                    should_stop["v"] = True  # callback raises -> keeps partial trace
            if not should_stop["v"] and _stop_requested(stop_dict, stop_key):
                should_stop["v"] = True  # callback raises KeyboardInterrupt on its next draw
        sample_thread.join()
        snap = _drain_and_yield()
        if snap:
            yield snap

    # -- Error / stop / completion phase --
    if sampling_error is not None:
        import traceback

        tb = "".join(traceback.format_exception(type(sampling_error), sampling_error, sampling_error.__traceback__))
        yield msgpack.packb({
            "type": "phase", "phase": "sampling", "status": "error",
            "message": str(sampling_error), "traceback": tb,
            "elapsed": time.time() - sample_start,
        })
        raise sampling_error

    if not stopped_early and stop_dict is not None:
        stopped_early = _stop_requested(stop_dict, stop_key)

    if converged:
        yield msgpack.packb({
            "type": "phase", "phase": "sampling", "status": "done",
            "message": f"converged ({counters['total_draws']} draws, target met)",
            "elapsed": round(time.time() - sample_start, 1),
        })
    elif stopped_early:
        yield msgpack.packb({
            "type": "phase", "phase": "sampling", "status": "done",
            "message": f"stopped early ({counters['total_draws']} draws)",
            "elapsed": round(time.time() - sample_start, 1),
        })
    else:
        yield msgpack.packb({
            "type": "phase", "phase": "sampling", "status": "done",
            "message": "sampling complete",
            "elapsed": round(time.time() - sample_start, 1),
        })

    # -- Serialize and return InferenceData --
    if idata is None:
        if stopped_early:
            raise RuntimeError(
                "sampling was stopped before any draws could be returned "
                "(the partial trace was not recoverable)"
            )
        raise RuntimeError("Sampling produced no results")

    _sanitize_idata_attrs(idata)

    import os
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        idata.to_netcdf(tmp_path)
        with open(tmp_path, "rb") as f:
            idata_compressed = lz4.frame.compress(f.read())
    finally:
        os.unlink(tmp_path)

    yield msgpack.packb({
        "type": "result",
        "size_mb": round(len(idata_compressed) / (1024 * 1024), 2),
    })

    yield idata_compressed


# Sampler settings nutpie accepts through ``nutpie.sample(**kwargs)``. Its
# signature is ``**kwargs``-open but unknown names raise at runtime ("Unknown
# settings attribute"), so only forward names known to be settings.
_NUTPIE_SETTINGS_KWARGS = {
    "target_accept",
    "maxdepth",
    "max_energy_error",
    "store_divergences",
    "store_gradient",
    "store_unconstrained",
    "store_mass_matrix",
    "use_grad_based_mass_matrix",
}


def _split_nutpie_kwargs(nutpie, sample_kwargs: dict) -> tuple[dict, dict]:
    """Partition leftover pm.sample kwargs into ``(forwarded, dropped)`` for
    ``nutpie.sample``.

    Forwards names in nutpie's explicit signature plus the known settings
    kwargs (e.g. ``target_accept``, ``maxdepth``); everything else is dropped
    (and reported by the caller) rather than crashing nutpie's settings parser.
    """
    import inspect

    accepted = set(_NUTPIE_SETTINGS_KWARGS)
    try:
        params = inspect.signature(nutpie.sample).parameters
        accepted |= {
            name for name, p in params.items()
            if p.kind is not inspect.Parameter.VAR_KEYWORD
        }
    except Exception:
        pass
    forwarded = {k: v for k, v in sample_kwargs.items() if k in accepted}
    dropped = {k: v for k, v in sample_kwargs.items() if k not in accepted}
    return forwarded, dropped


def _has_scalar_free_rvs(model) -> bool:
    """Whether the model has at least one scalar free RV.

    Convergence (and therefore ``until=``) is computed over (chain, draw)
    variables only; a model whose every parameter is vector-valued yields no
    convergence entries at all.
    """
    try:
        shapes = model.eval_rv_shapes()
    except Exception:
        shapes = {}
    for rv in getattr(model, "free_RVs", []):
        shape = shapes.get(rv.name)
        if shape is None:
            shape = tuple(d for d in (rv.type.shape or ()) if d is not None)
        if not shape:  # () -> scalar
            return True
    return False


def _conv_meets_target(conv_bytes, until) -> bool:
    """Whether a packed convergence event clears the adaptive target for EVERY
    scalar param: rhat <= until['r_hat'] and bulk/tail ESS >= until['ess']."""
    import msgpack

    try:
        d = msgpack.unpackb(conv_bytes, raw=False)
    except Exception:
        return False
    params = d.get("params") or {}
    if not params:
        return False
    rhat_max = float(until.get("r_hat", 1.01))
    ess_min = float(until.get("ess", 400))
    for p in params.values():
        if float(p.get("rhat", 1e9)) > rhat_max:
            return False
        if float(p.get("ess_bulk", 0)) < ess_min or float(p.get("ess_tail", 0)) < ess_min:
            return False
    return True


def _pymc_convergence(chain_traces, counters, msgpack):
    """Compute live R-hat / ESS from the pymc path's accumulated post-warmup
    scalar traces. Throttled to every 50 total draws; needs >=2 chains."""
    import numpy as _np

    if len(chain_traces) < 2:
        return None
    all_lens = [len(v) for ct in chain_traces.values() for v in ct.values()]
    min_draws = min(all_lens) if all_lens else 0
    if min_draws < 20:
        return None
    if counters["total_draws"] - counters["last_conv_draws"] < 50:
        return None
    counters["last_conv_draws"] = counters["total_draws"]

    try:
        import arviz as az

        param_names = set()
        for ct in chain_traces.values():
            param_names.update(ct.keys())

        convergence = {}
        traces = {}
        max_trace_points = 500
        for param in sorted(param_names):
            chain_values = [
                chain_traces[cid][param]
                for cid in sorted(chain_traces.keys())
                if param in chain_traces[cid]
            ]
            if len(chain_values) < 2:
                continue
            min_len = min(len(cv) for cv in chain_values)
            if min_len < 50:
                continue
            arr = _np.array([cv[:min_len] for cv in chain_values])  # (chains, draws)
            try:
                convergence[param] = {
                    "rhat": round(float(az.rhat(arr)), 4),
                    "ess_bulk": round(float(az.ess(arr))),
                    "ess_tail": round(_ess_tail(arr)),
                }
            except Exception:
                continue
            sub = []
            for cv in chain_values:
                vals = cv
                if len(vals) > max_trace_points:
                    step = max(1, len(vals) // max_trace_points)
                    vals = vals[::step][:max_trace_points]
                sub.append(list(vals))
            traces[param] = sub

        if convergence:
            return msgpack.packb({
                "type": "convergence",
                "params": convergence,
                "draws": min_draws,
                "total_draws": counters["total_draws"],
                "traces": traces,
                "partial": True,
            })
    except Exception:
        return None
    return None


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
    import pickle

    import lz4.frame
    import msgpack

    phase_start = time.time()
    obj = pickle.loads(lz4.frame.decompress(model_bytes))
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
    import pickle

    import lz4.frame
    import msgpack

    phase_start = time.time()
    with open(payload_path, "rb") as f:
        model_bytes = f.read()

    obj = pickle.loads(lz4.frame.decompress(model_bytes))
    model, sample_kwargs, nuts_sampler = _unpack_model_payload(
        obj, sample_kwargs, nuts_sampler
    )

    elapsed = time.time() - phase_start
    yield msgpack.packb({
        "type": "phase", "phase": "container_ready", "status": "done",
        "message": "model loaded from volume", "elapsed": elapsed,
    })

    yield from _sample_and_stream(model, sample_kwargs, nuts_sampler, stop_dict_name=stop_dict_name)


def _load_model_from_volume(payload_path: str):
    import pickle

    import lz4.frame

    with open(payload_path, "rb") as f:
        return pickle.loads(lz4.frame.decompress(f.read()))


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
