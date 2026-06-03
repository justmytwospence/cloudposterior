"""Custom step= methods. A step instance pickled separately from the model has
value variables in a different graph than a separately-deserialized model, so
the worker would raise "not a value variable in the model". cloudposterior ships
a combined {model, step} payload to keep them identity-linked."""

import numpy as np
import pytest


def _model():
    import pymc as pm

    with pm.Model() as m:
        mu = pm.Normal("mu", 0, 1)
        sigma = pm.HalfNormal("sigma", 1)
        pm.Normal("y", mu, sigma, observed=np.array([0.1, -0.2, 0.3, 0.5, -0.1, 0.2]))
    return m


def test_unpack_model_payload_injects_step_and_forces_pymc():
    pm = pytest.importorskip("pymc")
    from cloudposterior.remote.worker import _unpack_model_payload

    m = _model()
    with m:
        step = pm.Metropolis()
    model, kwargs, sampler = _unpack_model_payload(
        {"model": m, "step": step}, {"draws": 10}, "nutpie"
    )
    assert model is m
    assert kwargs["step"] is step
    assert sampler == "pymc"  # nutpie/JAX can't honor a custom step


def test_bare_model_payload_passes_through():
    pytest.importorskip("pymc")
    from cloudposterior.remote.worker import _unpack_model_payload

    m = _model()
    model, kwargs, sampler = _unpack_model_payload(m, {"draws": 10}, "nutpie")
    assert model is m and "step" not in kwargs and sampler == "nutpie"


def test_combined_payload_samples_with_step_through_worker():
    """End-to-end: serialize {model, step} together, run the streaming worker,
    and confirm it samples with the Metropolis step (no graph-identity error)."""
    pytest.importorskip("pymc")
    import msgpack

    from cloudposterior.remote.worker import run_sampling
    from cloudposterior.serialize import serialize_model_with_step
    import pymc as pm

    m = _model()
    with m:
        step = pm.Metropolis()
    blob = serialize_model_with_step(m, step)

    idata_bytes = None
    for chunk in run_sampling(blob, {"draws": 40, "tune": 40, "chains": 2,
                                     "random_seed": 0}, "nutpie"):
        try:
            msgpack.unpackb(chunk, raw=False)
        except Exception:
            idata_bytes = chunk

    assert idata_bytes is not None
    from cloudposterior.serialize import deserialize_inference_data

    idata = deserialize_inference_data(idata_bytes)
    assert {"mu", "sigma"} <= set(idata.posterior.data_vars)
    # Metropolis emits an 'accept'/'scaling' stat (NUTS would emit 'step_size').
    stats = set(idata.sample_stats.data_vars)
    assert stats & {"accept", "accepted", "scaling"}


def test_local_sample_with_step_works():
    pm = pytest.importorskip("pymc")
    import cloudposterior as cp
    from cloudposterior._idata import group_names

    m = _model()
    with cp.cloud(m, remote=False):
        idata = pm.sample(draws=30, tune=30, chains=2, step=pm.Metropolis(),
                          progressbar=False, random_seed=0,
                          compute_convergence_checks=False)
    assert "posterior" in group_names(idata)
