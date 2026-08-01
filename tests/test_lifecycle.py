"""Lifecycle correctness for process-global state.

The monkeypatch, the serialized-model memo, and the remote job handles all
outlive a single call, and each has a failure mode that silently produces a
wrong result rather than an error. These tests pin the guards down.
"""

import numpy as np
import pymc as pm
import pytest

import cloudposterior as cp
from cloudposterior import api

PATCHED = (
    "sample",
    "sample_prior_predictive",
    "sample_posterior_predictive",
    "sample_smc",
    "compute_log_likelihood",
)


def _model(mu_val=0.0):
    with pm.Model() as m:
        mu = pm.Normal("mu", mu_val, 1)
        pm.Normal("obs", mu, 1, observed=np.array([1.0, 2.0, 3.0]))
    return m


def _originals():
    return {name: getattr(pm, name) for name in PATCHED}


# -- monkeypatch lifecycle --------------------------------------------------

def test_reentering_the_same_block_raises_and_leaves_pymc_clean():
    """Re-entry would capture the interceptors as the "originals", leaving
    pm.sample patched after exit and recursing into itself forever."""
    before = _originals()
    session = cp.cloud(_model())
    with session:
        with pytest.raises(RuntimeError, match="not reentrant"):
            with session:
                pass
    assert _originals() == before


def test_enter_failure_restores_pymc(monkeypatch):
    """The five pm.* rebinds happen before the model context is entered; if
    that raises, __exit__ never runs and PyMC must not stay patched."""
    before = _originals()
    model = _model()
    monkeypatch.setattr(
        type(model), "__enter__", lambda self: (_ for _ in ()).throw(ValueError("boom"))
    )
    with pytest.raises(ValueError, match="boom"):
        with cp.cloud(model):
            pass
    assert _originals() == before


def test_normal_exit_restores_every_patched_function():
    before = _originals()
    with cp.cloud(_model()):
        assert pm.sample is not before["sample"]
    assert _originals() == before


# -- model-bytes memoization ------------------------------------------------

def test_model_bytes_memoized_when_data_unchanged():
    model = _model()
    first = api._ensure_model_bytes(model)
    assert api._ensure_model_bytes(model) is first


def test_model_bytes_reserialized_after_set_data():
    """A pm.set_data mutation must reach the worker and change the cache key,
    not silently reuse pre-mutation bytes."""
    with pm.Model() as model:
        x = pm.Data("x", np.array([1.0, 2.0, 3.0]))
        mu = pm.Normal("mu", 0, 1)
        pm.Normal("obs", mu * x, 1, observed=np.array([1.0, 2.0, 3.0]))

    first = api._ensure_model_bytes(model)
    with model:
        pm.set_data({"x": np.array([10.0, 20.0, 30.0])})
    second = api._ensure_model_bytes(model)

    assert second != first


def test_model_bytes_not_stored_on_the_model_object():
    """An attribute would be swept into the next cloudpickle, embedding the
    previous payload inside the new one."""
    model = _model()
    api._ensure_model_bytes(model)
    assert not hasattr(model, "_cp_model_bytes")
    assert not hasattr(model, "_cp_data_fp")


def test_step_blob_memoized_per_step_instance():
    from cloudposterior import serialize

    model = _model()
    with model:
        step = pm.Metropolis()

    calls = []
    real = serialize.serialize_model_with_step

    def counting(m, s):
        calls.append((m, s))
        return real(m, s)

    serialize.serialize_model_with_step = counting
    try:
        first = api._ensure_step_bytes(model, step)
        second = api._ensure_step_bytes(model, step)
    finally:
        serialize.serialize_model_with_step = real

    assert first is second
    assert len(calls) == 1


# -- interceptor model targeting --------------------------------------------

@pytest.mark.parametrize(
    "fn_name,call",
    [
        ("sample", lambda other: pm.sample(model=other)),
        (
            "sample_posterior_predictive",
            lambda other: pm.sample_posterior_predictive(object(), model=other),
        ),
        ("sample_smc", lambda other: pm.sample_smc(10, model=other)),
        (
            "compute_log_likelihood",
            lambda other: pm.compute_log_likelihood(object(), model=other),
        ),
    ],
)
def test_ops_targeting_another_model_defer_to_native_pymc(fn_name, call):
    """Without the guard these run the *wrapped* model remotely and hand back
    results for a model the caller never named."""
    wrapped, other = _model(), _model(mu_val=5.0)
    seen = {}

    def spy(*args, **kwargs):
        seen["model"] = kwargs.get("model")
        return "native"

    session = cp.cloud(wrapped, remote=True)
    with session:
        # Stand in for the captured original so no real sampling happens. It
        # must be put back before the block exits: _originals is what __exit__
        # restores onto the pymc module.
        real = session._originals[fn_name]
        session._originals[fn_name] = spy
        try:
            with pytest.warns(UserWarning, match="different model"):
                assert call(other) == "native"
        finally:
            session._originals[fn_name] = real

    assert seen["model"] is other
    assert getattr(pm, fn_name) is real


# -- remote job result() ----------------------------------------------------

def _job_that_streams_nothing():
    """A PersistentModalSamplingJob whose stream yields events but no trace."""
    from cloudposterior.backends.modal_backend import PersistentModalSamplingJob

    calls = {"n": 0}

    class FakeSampler:
        class sample:
            @staticmethod
            def remote_gen(*a, **k):
                calls["n"] += 1
                return iter(())

    job = PersistentModalSamplingJob(
        lambda: FakeSampler(), "p", {}, "pymc", stop_dict_name=None
    )
    job._sampler_cls = FakeSampler
    return job, calls


def test_result_refuses_to_rerun_a_stream_that_produced_no_trace():
    """Re-streaming here would silently launch a second *paid* sampling run."""
    job, calls = _job_that_streams_nothing()

    for _ in job.stream_progress():
        pass
    assert calls["n"] == 1

    with pytest.raises(RuntimeError, match="refusing to re-run"):
        job.result()
    assert calls["n"] == 1, "result() must not re-invoke the remote function"


def test_result_still_drives_the_stream_when_never_started():
    job, calls = _job_that_streams_nothing()

    with pytest.raises(RuntimeError, match="refusing to re-run"):
        job.result()
    assert calls["n"] == 1
