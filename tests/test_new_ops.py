"""sample_smc + compute_log_likelihood: interception, restore, and the worker
pipeline run locally (no Modal). These ops return InferenceData, so they ride
the existing predictive blocking template."""

import numpy as np
import pytest

from cloudposterior._idata import group_names


def _model():
    import pymc as pm

    with pm.Model() as m:
        mu = pm.Normal("mu", 0, 1)
        sigma = pm.HalfNormal("sigma", 1)
        pm.Normal("y", mu, sigma, observed=np.array([0.1, -0.2, 0.3, 0.5, -0.1, 0.2]))
    return m


# -- interception + restore -------------------------------------------------

def test_smc_and_cll_patched_and_restored():
    pm = pytest.importorskip("pymc")
    import cloudposterior as cp

    orig_smc = pm.sample_smc
    orig_cll = pm.compute_log_likelihood
    with cp.cloud(_model(), remote=False):
        assert pm.sample_smc is not orig_smc
        assert pm.compute_log_likelihood is not orig_cll
    assert pm.sample_smc is orig_smc
    assert pm.compute_log_likelihood is orig_cll


def test_local_smc_defers_to_original():
    pm = pytest.importorskip("pymc")
    import cloudposterior as cp
    from cloudposterior._idata import group_names

    m = _model()
    with cp.cloud(m, remote=False):
        idata = pm.sample_smc(draws=40, chains=2, progressbar=False, random_seed=0)
    assert "posterior" in group_names(idata)
    assert {"mu", "sigma"} <= set(idata.posterior.data_vars)


def test_local_compute_log_likelihood_defers_and_extends_in_place():
    pm = pytest.importorskip("pymc")
    import cloudposterior as cp
    from cloudposterior._idata import group_names

    m = _model()
    with cp.cloud(m, remote=False):
        idata = pm.sample(draws=40, tune=40, chains=2, nuts_sampler="pymc",
                          progressbar=False, random_seed=0,
                          compute_convergence_checks=False)
        out = pm.compute_log_likelihood(idata, progressbar=False)
    # native extend_inferencedata=True returns the same object, extended.
    assert out is idata
    assert "log_likelihood" in group_names(idata)


def test_remote_compute_log_likelihood_requires_idata():
    pytest.importorskip("pymc")
    import cloudposterior as cp

    c = cp.cloud(_model(), remote=True)
    c._env = object()  # truthy -> skip real provisioning
    intercepted = c._make_intercepted_cll()
    with pytest.raises(TypeError):
        intercepted()  # no idata


# -- worker pipeline (no Modal) ---------------------------------------------

def _serialize_model_to_file(model, tmp_path):
    from cloudposterior.serialize import serialize_model

    path = tmp_path / "payload.bin"
    path.write_bytes(serialize_model(model))
    return str(path)


def test_run_smc_worker_pipeline(tmp_path):
    pytest.importorskip("pymc")
    from cloudposterior.remote.worker import run_smc
    from cloudposterior.serialize import deserialize_inference_data

    path = _serialize_model_to_file(_model(), tmp_path)
    idata = deserialize_inference_data(
        run_smc(path, {"draws": 60, "chains": 2, "progressbar": False, "random_seed": 0})
    )
    assert {"mu", "sigma"} <= set(idata.posterior.data_vars)
    # SMC's object-dtype sample_stats must survive the NetCDF round-trip.
    assert "beta" in idata.sample_stats.data_vars


def test_run_compute_log_likelihood_matches_native(tmp_path):
    pm = pytest.importorskip("pymc")
    from cloudposterior._idata import get_group
    from cloudposterior.remote.worker import run_compute_log_likelihood
    from cloudposterior.serialize import (
        deserialize_inference_data,
        serialize_inference_data,
    )

    m = _model()
    path = _serialize_model_to_file(m, tmp_path)
    with m:
        idata = pm.sample(draws=60, tune=60, chains=2, nuts_sampler="pymc",
                          progressbar=False, random_seed=0,
                          compute_convergence_checks=False)
        native = pm.compute_log_likelihood(idata.copy(), extend_inferencedata=False,
                                           progressbar=False)

    out = deserialize_inference_data(
        run_compute_log_likelihood(path, serialize_inference_data(idata.copy()),
                                   {"progressbar": False})
    )
    worker_ll = get_group(out, "log_likelihood")["y"].values
    native_ll = native["y"].values
    assert np.allclose(worker_ll, native_ll, atol=1e-8)


def test_run_prior_predictive_worker_pipeline(tmp_path):
    """Sibling of run_smc / run_compute_log_likelihood, previously untested."""
    pytest.importorskip("pymc")
    from cloudposterior.remote.worker import run_prior_predictive
    from cloudposterior.serialize import deserialize_inference_data

    path = _serialize_model_to_file(_model(), tmp_path)
    idata = deserialize_inference_data(
        run_prior_predictive(path, {"draws": 20, "random_seed": 0})
    )
    assert "prior" in group_names(idata)
    assert {"mu", "sigma"} <= set(idata.prior.data_vars)


def test_run_posterior_predictive_worker_pipeline(tmp_path):
    pytest.importorskip("pymc")
    import pymc as pm

    from cloudposterior.remote.worker import run_posterior_predictive
    from cloudposterior.serialize import (
        deserialize_inference_data,
        serialize_inference_data,
    )

    m = _model()
    with m:
        idata = pm.sample(draws=30, tune=30, chains=2, progressbar=False,
                          random_seed=0, compute_convergence_checks=False)

    path = _serialize_model_to_file(m, tmp_path)
    out = deserialize_inference_data(run_posterior_predictive(
        path, serialize_inference_data(idata), {"random_seed": 0},
    ))
    assert "posterior_predictive" in group_names(out)
    assert "y" in out.posterior_predictive.data_vars
