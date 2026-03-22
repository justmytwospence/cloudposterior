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


def _sample_and_stream(model, sample_kwargs, nuts_sampler="pymc"):
    """Run MCMC sampling and yield msgpack-encoded progress + results.

    Shared core logic used by both one-shot and persistent paths.
    The caller is responsible for loading the model; this function
    handles compilation, sampling, progress streaming, and result serialization.
    """
    import lz4.frame
    import msgpack
    import pymc as pm

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

    chain_draw_counts: dict[int, int] = {}
    chain_start_times: dict[int, float] = {}
    chain_divergences: dict[int, int] = {}
    chain_tree_depths: dict[int, list[float]] = {}
    chain_phase: dict[int, bool] = {}
    sample_start = time.time()

    def progress_callback(trace, draw):
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

    def do_sample():
        nonlocal idata, sampling_error
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

    while sample_thread.is_alive():
        time.sleep(0.5)
        snapshot = _drain_and_yield()
        if snapshot:
            yield snapshot

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

    yield msgpack.packb({
        "type": "phase",
        "phase": "sampling",
        "status": "done",
        "message": "sampling complete",
        "elapsed": round(time.time() - sample_start, 1),
    })

    # -- Serialize and return InferenceData --
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

    yield from _sample_and_stream(model, sample_kwargs, nuts_sampler)
