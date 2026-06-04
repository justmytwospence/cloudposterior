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
