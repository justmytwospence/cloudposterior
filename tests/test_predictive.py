"""Predictive interception: local defers to the original, swaps restore."""

import numpy as np
import pytest


def _model():
    import pymc as pm

    with pm.Model() as m:
        mu = pm.Normal("mu", 0, 1)
        pm.Normal("y", mu, 1, observed=np.random.randn(20))
    return m


def test_local_predictive_defers_and_restores():
    pm = pytest.importorskip("pymc")
    import cloudposterior as cp
    from cloudposterior._idata import group_names

    m = _model()
    orig_prior = pm.sample_prior_predictive
    orig_post = pm.sample_posterior_predictive
    with cp.cloud(m, remote=False):
        assert pm.sample_prior_predictive is not orig_prior  # patched
        idata = pm.sample_prior_predictive(draws=25)
        assert "prior" in group_names(idata)
    # restored on exit
    assert pm.sample_prior_predictive is orig_prior
    assert pm.sample_posterior_predictive is orig_post


def test_remote_posterior_predictive_requires_trace(monkeypatch):
    pytest.importorskip("pymc")
    import cloudposterior as cp

    c = cp.cloud(_model(), remote=True)
    c._env = object()  # truthy -> skip real provisioning
    intercepted = c._make_intercepted_predictive("posterior")
    with pytest.raises(TypeError):
        intercepted()  # no trace


def test_remote_posterior_predictive_extend_merges_into_callers_idata(monkeypatch):
    """extend_inferencedata=True matches PyMC: the caller's idata is extended
    in place and returned (the worker always computes standalone)."""
    pytest.importorskip("pymc")
    import arviz as az

    import cloudposterior as cp
    import cloudposterior.api as api
    from cloudposterior._idata import add_group, group_names

    def _idata_with(group, values):
        idata = az.InferenceData()
        add_group(idata, group, az.dict_to_dataset(values))
        return idata

    trace = _idata_with("posterior", {"mu": np.zeros((2, 5))})
    remote_out = _idata_with("posterior_predictive", {"y": np.ones((2, 5, 3))})

    captured = {}

    def fake_run_predictive(ctx, kind, t, kwargs):
        captured["kwargs"] = kwargs
        return remote_out

    monkeypatch.setattr(api, "_run_predictive", fake_run_predictive)

    c = cp.cloud(_model(), remote=True)
    c._env = object()      # truthy -> skip real provisioning
    c._model_bytes = b"x"  # skip real serialization
    intercepted = c._make_intercepted_predictive("posterior")

    result = intercepted(trace, extend_inferencedata=True)
    assert result is trace
    assert "posterior_predictive" in group_names(trace)
    # the flag is client-side only; it must not ship to the worker
    assert "extend_inferencedata" not in captured["kwargs"]

    # default (False): standalone result, caller's idata untouched
    trace2 = _idata_with("posterior", {"mu": np.zeros((2, 5))})
    result2 = intercepted(trace2)
    assert result2 is remote_out
    assert "posterior_predictive" not in group_names(trace2)
