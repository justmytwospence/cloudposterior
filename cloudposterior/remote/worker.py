"""Remote worker that runs on Modal.

This module defines the Modal function that deserializes a PyMC model,
runs sampling with progress tracking, and streams results back.

It is NOT imported locally -- Modal serializes and runs it remotely.
The image is constructed dynamically based on the version manifest.
"""

from __future__ import annotations

import io
import time
from queue import Queue
from threading import Thread


def _sample_and_stream(model, sample_kwargs, nuts_sampler="pymc", stop_dict_name=None):
    """Run MCMC sampling and yield msgpack-encoded progress + results.

    Shared core logic used by both one-shot and persistent paths.
    The caller is responsible for loading the model; this function
    handles compilation, sampling, progress streaming, and result serialization.

    If stop_dict_name is provided, the callback checks for an early stop
    signal every 10 draws via a Modal Dict.
    """
    import lz4.frame
    import msgpack
    import pymc as pm

    # -- Check JAX device for GPU samplers --
    if nuts_sampler in ("numpyro", "blackjax"):
        yield msgpack.packb({
            "type": "phase",
            "phase": "device",
            "status": "in_progress",
            "message": "initializing JAX",
            "elapsed": 0.0,
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
                device_msg = f"JAX using CPU (no GPU found)"
        except Exception as e:
            device_msg = f"JAX device check failed: {e}"
        yield msgpack.packb({
            "type": "phase",
            "phase": "device",
            "status": "done",
            "message": device_msg,
            "elapsed": time.time() - jax_start,
        })

    # -- Phase: compiling (if nutpie/numpyro) --
    if nuts_sampler == "nutpie":
        compile_start = time.time()
        yield msgpack.packb({
            "type": "phase",
            "phase": "compiling",
            "status": "in_progress",
            "message": "compiling model with nutpie",
            "elapsed": 0.0,
        })
        import nutpie
        compiled = nutpie.compile_pymc_model(model)
        yield msgpack.packb({
            "type": "phase",
            "phase": "compiling",
            "status": "done",
            "message": "nutpie compilation complete",
            "elapsed": time.time() - compile_start,
        })

    # -- Phase: sampling --
    yield msgpack.packb({
        "type": "phase",
        "phase": "sampling",
        "status": "in_progress",
        "message": "MCMC sampling started",
        "elapsed": 0.0,
    })

    tune = sample_kwargs.pop("tune", 1000)
    draws = sample_kwargs.pop("draws", 1000)
    chains = sample_kwargs.pop("chains", None)
    cores = sample_kwargs.pop("cores", None)

    progress_queue: Queue = Queue()
    sampling_error = None
    idata = None

    # Set up early stop check via Modal Dict
    _stop_dict = None
    if stop_dict_name:
        try:
            import modal
            _stop_dict = modal.Dict.from_name(stop_dict_name)
        except Exception:
            pass

    chain_draw_counts: dict[int, int] = {}
    chain_start_times: dict[int, float] = {}
    chain_divergences: dict[int, int] = {}
    chain_tree_depths: dict[int, list[float]] = {}
    chain_phase: dict[int, bool] = {}
    # Accumulate trace values for convergence diagnostics
    chain_traces: dict[int, dict[str, list]] = {}  # chain -> {param_name: [values...]}
    _total_draws: int = 0
    _last_convergence_draw: int = 0
    sample_start = time.time()

    def progress_callback(trace, draw):
        nonlocal _total_draws
        _total_draws += 1

        # Check for early stop signal every 10 draws
        if _stop_dict is not None and _total_draws % 10 == 0:
            try:
                if _stop_dict.get("stop", False):
                    raise KeyboardInterrupt("early stop requested")
            except KeyboardInterrupt:
                raise
            except Exception:
                pass
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

        # Accumulate parameter values for convergence and trace plots
        if hasattr(draw, 'point') and draw.point:
            if chain not in chain_traces:
                chain_traces[chain] = {}
            for param_name, value in draw.point.items():
                import numpy as _np
                val = _np.asarray(value)
                if val.ndim == 0:  # scalar params only
                    if param_name not in chain_traces[chain]:
                        chain_traces[chain][param_name] = []
                    chain_traces[chain][param_name].append(float(val))

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

    stopped_early = False

    def do_sample():
        nonlocal idata, sampling_error, stopped_early
        try:
            if nuts_sampler == "nutpie":
                idata = nutpie.sample(compiled, draws=draws, tune=tune, chains=chains, progress_bar=False)
            else:
                with model:
                    idata = pm.sample(
                        draws=draws,
                        tune=tune,
                        chains=chains,
                        cores=cores,
                        callback=progress_callback,
                        progressbar=False,
                        **sample_kwargs,
                    )
        except KeyboardInterrupt:
            # Early stop: PyMC preserves partial trace in idata
            stopped_early = True
        except Exception as e:
            sampling_error = e

    sample_thread = Thread(target=do_sample)
    sample_thread.start()

    all_chain_states: dict[str, dict] = {}

    def _drain_and_yield():
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
            })
        return None

    def _compute_convergence():
        """Compute R-hat and ESS on accumulated traces. Returns msgpack or None."""
        nonlocal _last_convergence_draw
        import numpy as _np

        # Need at least 2 chains with draws
        if len(chain_traces) < 2:
            return None

        # Find minimum draws across all params in all chains
        all_lens = [len(v) for ct in chain_traces.values() for v in ct.values()]
        min_draws = min(all_lens) if all_lens else 0
        if min_draws < 20:
            return None
        # Only compute every 50 total draws
        if _total_draws - _last_convergence_draw < 50:
            return None
        _last_convergence_draw = _total_draws

        try:
            import arviz as az

            # Build a dict of {param: array(chains, draws)} from accumulated traces
            param_names = set()
            for ct in chain_traces.values():
                param_names.update(ct.keys())

            convergence = {}
            for param in sorted(param_names):
                # Get draws per chain, truncate to shortest
                chain_values = []
                for chain_id in sorted(chain_traces.keys()):
                    if param in chain_traces[chain_id]:
                        chain_values.append(chain_traces[chain_id][param])
                if len(chain_values) < 2:
                    continue
                min_len = min(len(cv) for cv in chain_values)
                if min_len < 50:
                    continue
                arr = _np.array([cv[:min_len] for cv in chain_values])  # (chains, draws)
                rhat = float(az.rhat(arr))
                ess_bulk = float(az.ess(arr))
                ess_tail = float(az.ess(arr, method="tail"))
                convergence[param] = {
                    "rhat": round(rhat, 4),
                    "ess_bulk": round(ess_bulk),
                    "ess_tail": round(ess_tail),
                }

            if convergence:
                # Subsampled trace values for live traceplots
                traces = {}
                max_trace_points = 500
                for param in sorted(param_names):
                    chain_values = []
                    for chain_id in sorted(chain_traces.keys()):
                        if param in chain_traces[chain_id]:
                            vals = chain_traces[chain_id][param]
                            if len(vals) > max_trace_points:
                                step = len(vals) // max_trace_points
                                vals = vals[::step][:max_trace_points]
                            chain_values.append(vals)
                    if chain_values:
                        traces[param] = chain_values

                return msgpack.packb({
                    "type": "convergence",
                    "params": convergence,
                    "draws": min_draws,
                    "traces": traces,
                })
        except Exception:
            pass
        return None

    while sample_thread.is_alive():
        time.sleep(0.5)
        snapshot = _drain_and_yield()
        if snapshot:
            yield snapshot
        # Periodically compute and yield convergence diagnostics
        conv = _compute_convergence()
        if conv:
            yield conv

    sample_thread.join()

    snapshot = _drain_and_yield()
    if snapshot:
        yield snapshot

    if sampling_error is not None:
        yield msgpack.packb({
            "type": "phase",
            "phase": "sampling",
            "status": "error",
            "message": str(sampling_error),
            "elapsed": time.time() - sample_start,
        })
        raise sampling_error

    # Check if stop was requested (either via exception or Dict flag)
    if not stopped_early and _stop_dict is not None:
        try:
            stopped_early = _stop_dict.get("stop", False)
        except Exception:
            pass

    if stopped_early:
        yield msgpack.packb({
            "type": "phase",
            "phase": "sampling",
            "status": "done",
            "message": f"stopped early ({_total_draws} draws)",
            "elapsed": round(time.time() - sample_start, 1),
        })
    else:
        yield msgpack.packb({
            "type": "phase",
            "phase": "sampling",
            "status": "done",
            "message": "sampling complete",
            "elapsed": round(time.time() - sample_start, 1),
        })

    # -- Serialize and return InferenceData --
    if idata is None:
        raise RuntimeError("Sampling produced no results")

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


def run_sampling(
    model_bytes: bytes,
    data_bytes: bytes,
    sample_kwargs: dict,
    nuts_sampler: str = "pymc",
    persistent: bool = False,
):
    """One-shot path: deserialize model from bytes and run sampling."""
    import lz4.frame
    import msgpack
    import numpy as np

    phase_start = time.time()
    model_raw = lz4.frame.decompress(model_bytes)
    import pickle
    model = pickle.loads(model_raw)

    data_raw = lz4.frame.decompress(data_bytes)
    buf = io.BytesIO(data_raw)
    observed = dict(np.load(buf))

    elapsed = time.time() - phase_start
    phase_name = "container_ready" if persistent else "provisioning"
    phase_message = "container ready" if persistent else "environment ready"
    yield msgpack.packb({
        "type": "phase",
        "phase": phase_name,
        "status": "done",
        "message": phase_message,
        "elapsed": elapsed,
    })

    yield from _sample_and_stream(model, sample_kwargs, nuts_sampler)


def run_sampling_from_volume(
    payload_path: str,
    sample_kwargs: dict,
    nuts_sampler: str = "pymc",
    stop_dict_name: str | None = None,
):
    """Persistent path: load model from Volume and run sampling."""
    import lz4.frame
    import msgpack
    import pickle

    phase_start = time.time()
    with open(payload_path, "rb") as f:
        model_bytes = f.read()

    model_raw = lz4.frame.decompress(model_bytes)
    model = pickle.loads(model_raw)

    elapsed = time.time() - phase_start
    yield msgpack.packb({
        "type": "phase",
        "phase": "container_ready",
        "status": "done",
        "message": "model loaded from volume",
        "elapsed": elapsed,
    })

    yield from _sample_and_stream(model, sample_kwargs, nuts_sampler, stop_dict_name=stop_dict_name)
