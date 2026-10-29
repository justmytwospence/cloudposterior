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
from queue import Queue
from threading import Thread

from cloudposterior._idata import (
    add_group as _add_group,
    concat_chains as _concat_chains,
    get_group as _get_group,
    sanitize_inference_data as _sanitize_idata_attrs,
    to_inference_data as _to_inference_data,
    trim_incomplete_draws as _trim_incomplete,
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


# Escalation ladder: what to change on a re-run, by the likely cause of failure.
# Each step is tried once, in order; steps a sampler can't apply are skipped.
_LADDER = {
    "early_drift": [("tune", 2.0), ("tune", 2.0)],
    "divergences": [("target_accept", 0.95), ("target_accept", 0.99)],
    "low_bfmi": [("tune", 2.0), ("adaptation", "low_rank")],
    "chains_disagree": [("tune", 2.0), ("target_accept", 0.95)],
    "slow_mixing": [("draws", "projected")],
    "ess_stalled": [("adaptation", "low_rank"), ("tune", 2.0)],
}


def _next_rung(report: dict, settings: dict, nuts_sampler: str, applied: dict):
    """Pick and apply the next escalation step. Returns a description or None."""
    import math

    cause = report.get("cause")
    steps = _LADDER.get(cause or "", [])
    i = applied.get(cause, 0)
    extra = settings["extra"]
    while i < len(steps):
        key, value = steps[i]
        i += 1
        applied[cause] = i
        if key == "tune":
            old = settings["tune"]
            settings["tune"] = int(math.ceil(old * value))
            return f"tune {old} -> {settings['tune']}"
        if key == "target_accept":
            old = extra.get("target_accept", 0.8)
            if float(old) >= value:
                continue
            extra["target_accept"] = value
            return f"target_accept {old} -> {value}"
        if key == "adaptation":
            if nuts_sampler != "nutpie" or extra.get("adaptation") == value:
                continue
            extra["adaptation"] = value
            if settings["tune"] < 800:
                settings["tune"] = 800  # nutpie's low-rank default warmup
            return f"nutpie adaptation -> {value}"
        if key == "draws":
            proj = report.get("projected_draws")
            old = settings["draws"]
            if not proj or proj <= old:
                continue
            settings["draws"] = int(min(4 * old, math.ceil(proj * 1.2)))
            return f"draws cap {old} -> {settings['draws']} (projected need ~{proj})"
    return None


def _sample_and_stream(model, sample_kwargs, nuts_sampler="nutpie", stop_dict_name=None,
                       stop_key=None):
    """Run MCMC sampling and yield msgpack-encoded progress + results.

    Shared core logic used by both one-shot and persistent paths. The caller
    loads the model; this function handles compilation, sampling (with the
    convergence layer's adaptive stopping and escalation), progress streaming,
    and result serialization.
    """
    import lz4.frame
    import msgpack

    from cloudposterior import convergence as cv

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
    sample_kwargs = dict(sample_kwargs)
    draws = sample_kwargs.pop("draws", 1000)
    user_tune = sample_kwargs.pop("tune", None)
    tune = user_tune if user_tune is not None else 1000
    chains = sample_kwargs.pop("chains", None)
    cores = sample_kwargs.pop("cores", None)
    random_seed = sample_kwargs.pop("random_seed", None)
    sample_kwargs.pop("progressbar", None)
    sample_kwargs.pop("callback", None)
    # The client warns that return_inferencedata=False is unsupported remotely
    # (a MultiTrace can't be transported back); drop it here so the pymc path
    # doesn't produce a MultiTrace that crashes serialization after sampling.
    sample_kwargs.pop("return_inferencedata", None)
    # cp-only kwargs (never passed to the sampler): the convergence target and
    # the initialization strategy.
    until = cv.normalize_target(sample_kwargs.pop("until", None))
    init = sample_kwargs.pop("cp_init", None)
    if nuts_sampler in ("numpyro", "blackjax"):
        until = None  # no per-draw hook: the client already warned
    chains = _resolve_chains(chains, cores)

    superchains = (until or {}).get("superchains")
    if superchains and chains % superchains:
        raise ValueError(
            f"until superchains={superchains} must divide chains={chains} evenly"
        )

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

            compile_kwargs = {}
            if init == "prior":
                compile_kwargs["default_initialization_strategy"] = "prior"
            compiled = nutpie.compile_pymc_model(model, **compile_kwargs)
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

    settings = {"draws": draws, "tune": tune, "seed": random_seed, "extra": sample_kwargs}
    attempts: list = []
    applied: dict = {}
    tried: list[str] = []
    max_escalations = int((until or {}).get("escalate", 0))
    attempt = 0
    total_start = time.time()
    while True:
        outcome = yield from _sample_once(
            model, compiled, nuts_sampler, settings, chains, cores, until, init,
            stop_dict_name, stop_key, attempt,
        )
        idata, report = outcome["idata"], outcome["report"]
        if report is not None:
            attempts.append({
                "attempt": attempt + 1,
                "draws": settings["draws"], "tune": settings["tune"],
                "target_accept": settings["extra"].get("target_accept"),
                "adaptation": settings["extra"].get("adaptation"),
                "status": report["status"], "summary": report["summary"],
                "cause": report.get("cause"),
            })
        description = None
        if report is not None and tried:
            report["hint"] = (report.get("hint") or "").rstrip() + (
                " Already re-ran with: " + "; ".join(tried) + "."
            )
        if (
            until and report is not None and attempt < max_escalations
            and report["status"] in ("futile", "cap") and not report["met"]
        ):
            description = _next_rung(report, settings, nuts_sampler, applied)
        if description is None:
            break
        tried.append(description)
        attempt += 1
        if isinstance(settings["seed"], int):
            settings["seed"] = settings["seed"] + 1000
        yield msgpack.packb({
            "type": "phase", "phase": "sampling", "status": "in_progress",
            "message": (
                f"re-running ({attempt}/{max_escalations}): {description} -- "
                f"{report.get('cause', '').replace('_', ' ')}"
            ),
            "elapsed": round(time.time() - total_start, 1),
        })

    if report is not None:
        if len(attempts) > 1:
            report["attempts"] = attempts
        cv.attach_report(idata, report)
        message = cv.verdict_message(report)
    else:
        message = outcome.get("message", "sampling complete")
    yield msgpack.packb({
        "type": "phase", "phase": "sampling", "status": "done",
        "message": message,
        "elapsed": round(time.time() - total_start, 1),
    })

    # -- Serialize and return InferenceData --
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


def _sample_once(model, compiled, nuts_sampler, settings, chains, cores, until, init,
                 stop_dict_name, stop_key, attempt):
    """One sampling run. Yields progress; returns {"idata", "report"}.

    Raises (after emitting an error phase) when the sampler fails, and when a
    stop left no recoverable draws.
    """
    import msgpack
    import pymc as pm

    from cloudposterior import convergence as cv

    draws = settings["draws"]
    tune = settings["tune"]
    random_seed = settings["seed"]
    sample_kwargs = dict(settings["extra"])

    yield msgpack.packb({
        "type": "phase", "phase": "sampling", "status": "in_progress",
        "message": "MCMC sampling started" if attempt == 0 else f"MCMC sampling (attempt {attempt + 1})",
        "elapsed": 0.0,
    })

    names = _monitored_names(model, until)
    target = until or cv.normalize_target(True)
    controller = cv.Controller(target, n_chains=chains, draws_cap=draws, stop=bool(until))
    superchains = (until or {}).get("superchains")
    per_super = chains // superchains if superchains else None
    sc_ids = [c // per_super for c in range(chains)] if superchains else None
    if sc_ids is not None and nuts_sampler in ("numpyro", "blackjax"):
        sc_ids = None

    progress_queue: Queue = Queue()
    all_chain_states: dict[str, dict] = {}
    counters = {"total_draws": 0}
    sample_start = time.time()
    stop_dict = _open_dict(stop_dict_name)

    sampling_error = None
    idata = None
    stopped_by_user = False
    decision = None  # "met" | "futile" once the controller stops the run
    last_step = None

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

    def _checkpoint(make_draws):
        """Assess at a due checkpoint; returns (event_bytes, assessment) or (None, None)."""
        try:
            d = make_draws()
            a = controller.step(d)
            # Time to the target at the current per-chain draw rate.
            rates = [c.get("draws_per_sec", 0.0) for c in all_chain_states.values()
                     if c.get("phase") == "sampling"]
            proj = a.get("projected_draws")
            if rates and proj and proj > d.n_draws and min(rates) > 0:
                a["eta_seconds"] = round((proj - d.n_draws) / min(rates), 1)
            return _convergence_event(a, d, total_draws=counters["total_draws"], final=False), a
        except Exception:
            # A diagnostic failure must never take down sampling; push the next
            # checkpoint out so a persistent failure isn't retried every poll.
            controller.next_at = max(controller.next_at, 1) * 2
            return None, None

    # ===================== nutpie: native progress + background sampler(s) =====================
    if nuts_sampler == "nutpie":
        import nutpie

        post_counts: dict[int, int] = {}
        handle_totals: dict[int, int] = {}

        def make_cb(offset: int, part: int):
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
                    post_counts[offset + i] = 0 if p.tuning else cur
                    total += finished
                    elapsed = float(p.runtime_ms or 0) / 1000.0
                    dps = finished / elapsed if elapsed > 0 else 0.0
                    remaining = max(0, total_draws - finished)
                    eta = remaining / dps if dps > 0 else 0.0
                    progress_queue.put({
                        "chain": offset + i,
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
                handle_totals[part] = total
                counters["total_draws"] = sum(handle_totals.values())
            return nutpie_cb

        # Forward the remaining pm.sample kwargs nutpie understands (e.g.
        # target_accept, maxdepth, adaptation) instead of silently dropping
        # them; report whatever nutpie can't accept so the user knows.
        forwarded, dropped = _split_nutpie_kwargs(nutpie, sample_kwargs)
        forwarded.pop("random_seed", None)
        dropped.pop("random_seed", None)
        if dropped and attempt == 0:
            yield msgpack.packb({
                "type": "phase", "phase": "sampling", "status": "in_progress",
                "message": "ignoring kwargs unsupported by nutpie: "
                           + ", ".join(sorted(dropped)),
                "elapsed": 0.0,
            })

        # One sampler, or one per superchain: every chain in a superchain
        # starts from the same point (Margossian et al. 2024).
        if superchains:
            import dataclasses

            seeds = _sub_seeds(random_seed, 2 * superchains)
            jobs = []
            for k in range(superchains):
                start = compiled.initial_point_func(seeds[k])
                try:
                    part_model = dataclasses.replace(
                        compiled, initial_point_func=_fixed_point(start)
                    )
                except Exception as exc:
                    raise ValueError(
                        "superchains need nutpie's default (numba) PyMC backend"
                    ) from exc
                jobs.append((part_model, per_super, seeds[superchains + k], k * per_super))
        else:
            seed = random_seed if random_seed is None or isinstance(random_seed, int) else None
            jobs = [(compiled, chains, seed, 0)]

        handles = []
        for part, (part_model, n_ch, seed, offset) in enumerate(jobs):
            kw = dict(
                draws=draws,
                # Match pm.sample's tune default (1000); nutpie's own default
                # (~300) would silently diverge from an equivalent local run.
                tune=tune,
                chains=n_ch,
                cores=(min(cores, n_ch) if cores else n_ch) if superchains else cores,
                seed=seed,
                save_warmup=False,
                progress_bar=False,
                blocking=False,
                **forwarded,
            )
            try:
                handles.append(nutpie.sample(part_model, progress_callback=make_cb(offset, part), **kw))
            except Exception:
                # Some nutpie builds (e.g. 0.16.8) accept progress_callback but
                # fail to wire it. Fall back to sampling without live progress
                # (and so without adaptive stopping) rather than aborting.
                try:
                    handles.append(nutpie.sample(part_model, **kw))
                except Exception as exc:
                    sampling_error = exc
                    break

        def _abort_all():
            parts = []
            for h in handles:
                try:
                    parts.append(_trim_incomplete(_to_inference_data(h.abort())))
                except Exception:
                    pass
            return parts

        if handles and sampling_error is None:
            _poll_failures = {"n": 0}

            def _finished():
                # nutpie's is_finished property raises (e.g. TimeoutError) while
                # the sampler is still running, so treat a raise as "not done" --
                # but only up to a point. A property that raises persistently
                # would otherwise spin at 0.5s until the container timeout kills
                # the run with no result at all.
                try:
                    done = all(bool(h.is_finished) for h in handles)
                except Exception:
                    _poll_failures["n"] += 1
                    return _poll_failures["n"] >= _MAX_POLL_FAILURES
                _poll_failures["n"] = 0
                return done

            parts = None
            while not _finished():
                time.sleep(0.5)
                snap = _drain_and_yield()
                if snap:
                    yield snap
                post_min = min(post_counts.values()) if len(post_counts) == chains else 0
                if controller.due(post_min):
                    event, a = _checkpoint(lambda: _parts_to_draws(
                        [_to_inference_data(h.inspect()) for h in handles], names, sc_ids,
                    ))
                    if event:
                        yield event
                        last_step = a
                        if a["decision"] in ("met", "futile"):
                            decision = a["decision"]
                            parts = _abort_all()
                            break
                if _stop_requested(stop_dict, stop_key):
                    stopped_by_user = True
                    parts = _abort_all()
                    break

            if parts is None:
                parts = []
                for h in handles:
                    try:
                        parts.append(_to_inference_data(h.wait()))
                    except Exception as exc:
                        sampling_error = exc
                        break
            if sampling_error is None and parts:
                idata = parts[0] if len(parts) == 1 else _concat_chains(parts)
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
        if init == "prior" and "initvals" not in sample_kwargs:
            sample_kwargs["initvals"] = {rv.name: "prior" for rv in model.free_RVs}

        def do_sample_jax():
            nonlocal idata, sampling_error
            try:
                with model:
                    idata = pm.sample(
                        draws=draws, tune=tune, chains=chains,
                        nuts_sampler=nuts_sampler, progressbar=False,
                        random_seed=random_seed, **sample_kwargs,
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
        import numpy as _np

        chain_draw_counts: dict[int, int] = {}
        chain_start_times: dict[int, float] = {}
        chain_divergences: dict[int, int] = {}
        chain_tree_depths: dict[int, list[float]] = {}
        chain_phase: dict[int, bool] = {}
        # Post-warmup values of the monitored variables and HMC stats, per chain.
        value_to_rv = {}
        for rv in getattr(model, "free_RVs", []):
            try:
                value_to_rv[model.rvs_to_values[rv].name] = rv.name
            except Exception:
                pass
        wanted = {v for v, r in value_to_rv.items() if r in set(names)}
        chain_values: dict[int, dict[str, list]] = {}
        chain_stats: dict[int, dict[str, list]] = {}
        should_stop = {"v": False}
        # Set when the convergence layer decides: interrupt once every chain has
        # this many post-warmup draws. Interrupting at once would let PyMC's
        # _choose_chains drop the chains that were behind (it maximizes total
        # draws), changing the chain count and breaking superchain groups.
        stop_at = {"n": None}

        # Initialization: per-chain starting points from the prior, and/or one
        # shared start per superchain (jitter off so the chains really share it).
        if superchains or init == "prior":
            if sc_ids is not None:
                from pymc.initial_point import make_initial_point_fn

                strategy = "prior" if init == "prior" else "support_point"
                jitter = set() if init == "prior" else set(model.free_RVs)
                fn = make_initial_point_fn(
                    model=model, default_strategy=strategy, jitter_rvs=jitter,
                    return_transformed=False,
                )
                seeds = _sub_seeds(random_seed, superchains)
                starts = [fn(s) for s in seeds]
                sample_kwargs["initvals"] = [starts[k] for k in sc_ids]
                user_init = sample_kwargs.get("init", "jitter+adapt_diag")
                plain = str(user_init).replace("jitter+", "")
                sample_kwargs["init"] = "adapt_diag" if plain == "auto" else plain
            elif "initvals" not in sample_kwargs:
                sample_kwargs["initvals"] = {rv.name: "prior" for rv in model.free_RVs}

        def progress_callback(trace, draw):
            # NOTE: the draw counting / phase reset / windowed tree-depth logic
            # mirrors make_sampling_callback in cloudposterior/progress.py
            # (used for local runs) -- keep the two in sync.
            # Cheap local flag only -- the network poll lives in the generator loop (D3).
            if should_stop["v"]:
                raise KeyboardInterrupt("early stop requested")
            if stop_at["n"] is not None and len(chain_stats) == chains and min(
                len(cs["diverging"]) for cs in chain_stats.values()
            ) >= stop_at["n"]:
                raise KeyboardInterrupt("convergence target decided")
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

            # Accumulate POST-WARMUP monitored values + stats for the
            # convergence layer (vectors included, not just scalars).
            if not is_tuning and getattr(draw, "point", None):
                cv_ = chain_values.setdefault(chain, {})
                for vname in wanted:
                    if vname in draw.point:
                        cv_.setdefault(vname, []).append(_np.array(draw.point[vname], dtype=float))
                cs = chain_stats.setdefault(chain, {"diverging": [], "energy": [], "maxdepth": []})
                cs["diverging"].append(bool(diverging))
                cs["energy"].append(float(stats.get("energy", _np.nan)))
                cs["maxdepth"].append(bool(stats.get("reached_max_treedepth", False)))

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

        def _pymc_draws():
            n = min(len(chain_stats[c]["diverging"]) for c in range(chains))
            arrays = {}
            for vname in sorted(wanted):
                rows = []
                for c in range(chains):
                    vals = chain_values.get(c, {}).get(vname, [])[:n]
                    if len(vals) < n:
                        raise ValueError("incomplete chain")
                    rows.append(_np.stack(vals))
                arrays[value_to_rv[vname]] = _np.stack(rows)
            stat = {
                key: _np.array([chain_stats[c][key][:n] for c in range(chains)])
                for key in ("diverging", "energy", "maxdepth")
            }
            return cv.draws_from_arrays(
                arrays, diverging=stat["diverging"],
                energy=stat["energy"] if _np.isfinite(stat["energy"]).all() else None,
                maxdepth=stat["maxdepth"], superchain_ids=sc_ids,
            )

        def do_sample_pymc():
            nonlocal idata, sampling_error
            try:
                with model:
                    idata = pm.sample(
                        draws=draws, tune=tune, chains=chains, cores=cores,
                        nuts_sampler="pymc",  # force the pure-pymc sampler: only it
                        callback=progress_callback,  # supports the per-draw callback
                        progressbar=False, random_seed=random_seed, **sample_kwargs,
                    )
            except KeyboardInterrupt:
                pass  # PyMC keeps the partial trace when the callback interrupts
            except Exception as exc:
                sampling_error = exc

        sample_thread = Thread(target=do_sample_pymc)
        sample_thread.start()
        while sample_thread.is_alive():
            time.sleep(0.5)
            snap = _drain_and_yield()
            if snap:
                yield snap
            post_min = (
                min(len(chain_stats[c]["diverging"]) for c in range(chains))
                if wanted and all(c in chain_stats for c in range(chains)) else 0
            )
            if stop_at["n"] is None and not should_stop["v"] and controller.due(post_min):
                event, a = _checkpoint(_pymc_draws)
                if event:
                    yield event
                    last_step = a
                    if a["decision"] in ("met", "futile"):
                        decision = a["decision"]
                        stop_at["n"] = max(len(cs["diverging"]) for cs in chain_stats.values())
            if not should_stop["v"] and _stop_requested(stop_dict, stop_key):
                stopped_by_user = True
                should_stop["v"] = True  # callback raises KeyboardInterrupt on its next draw
        sample_thread.join()
        snap = _drain_and_yield()
        if snap:
            yield snap

    # -- Error / stop phase --
    if sampling_error is not None:
        import traceback

        tb = "".join(traceback.format_exception(type(sampling_error), sampling_error, sampling_error.__traceback__))
        yield msgpack.packb({
            "type": "phase", "phase": "sampling", "status": "error",
            "message": str(sampling_error), "traceback": tb,
            "elapsed": time.time() - sample_start,
        })
        raise sampling_error

    if not stopped_by_user and decision is None and stop_dict is not None:
        stopped_by_user = _stop_requested(stop_dict, stop_key)

    if idata is None:
        if stopped_by_user or decision:
            raise RuntimeError(
                "sampling was stopped before any draws could be returned "
                "(the partial trace was not recoverable)"
            )
        raise RuntimeError("Sampling produced no results")

    try:
        n_chains_out = int(_get_group(idata, "posterior").sizes["chain"])
    except Exception:
        n_chains_out = chains
    if sc_ids is not None and n_chains_out != len(sc_ids):
        sc_ids = None  # a stop dropped chains; the superchain groups are gone
    if sc_ids is not None:
        try:
            _get_group(idata, "posterior").attrs[cv.SUPERCHAIN_ATTR] = list(sc_ids)
        except Exception:
            pass

    # -- Final verdict on the complete trace --
    if decision == "met":
        status = "met"
    elif decision == "futile":
        status = "futile"
    elif stopped_by_user:
        status = "stopped"
    elif until:
        status = "cap"
    else:
        status = "complete"
    report = None
    try:
        d = cv.draws_from_idata(idata, var_names=names, superchain_ids=sc_ids)
        a = cv.assess(d, target)
        if status == "futile" and last_step is not None:
            a["cause"], a["hint"] = last_step.get("cause"), last_step.get("hint")
        elif not a["met"]:
            a["cause"], a["hint"] = cv.diagnose(d, a)
        a["projected_draws"] = cv.projected_draws(a)
        a["decision"] = decision or "complete"
        yield _convergence_event(a, d, total_draws=counters["total_draws"], final=True)
        report = cv.make_report(a, status=status, target=until)
    except Exception:
        report = None
    return {"idata": idata, "report": report,
            "message": "stopped early" if stopped_by_user else "sampling complete"}


def _fixed_point(start):
    """An initial-point function that always returns ``start`` (a superchain's
    shared starting point), whatever seed nutpie passes."""
    def initial_point(_seed):
        return start.copy()
    return initial_point


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
