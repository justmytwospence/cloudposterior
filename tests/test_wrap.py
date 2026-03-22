"""Test the cp.cloud() context manager."""

from unittest.mock import patch, MagicMock

import numpy as np
import pymc as pm

import cloudposterior as cp


def _make_model():
    y = np.array([28, 8, -3, 7, -1, 1, 18, 12], dtype=np.float64)
    sigma = np.array([15, 10, 16, 11, 9, 11, 10, 18], dtype=np.float64)
    with pm.Model() as model:
        mu = pm.Normal("mu", 0, 5)
        tau = pm.HalfCauchy("tau", 5)
        theta = pm.Normal("theta", mu=mu, sigma=tau, shape=8)
        pm.Normal("obs", mu=theta, sigma=sigma, observed=y)
    return model


def test_pm_sample_is_patched_inside_cloud():
    """pm.sample should be replaced inside cp.cloud()."""
    model = _make_model()
    original_sample = pm.sample

    with cp.cloud(model):
        assert pm.sample is not original_sample, "pm.sample should be patched inside cloud"

    assert pm.sample is original_sample, "pm.sample should be restored after cloud"


def test_pm_sample_restored_on_exception():
    """pm.sample must be restored even if an exception occurs inside the block."""
    model = _make_model()
    original_sample = pm.sample

    try:
        with cp.cloud(model):
            assert pm.sample is not original_sample
            raise ValueError("intentional error")
    except ValueError:
        pass

    assert pm.sample is original_sample, "pm.sample should be restored after exception"


def test_cloud_enters_model_context():
    """The model context should be active inside cp.cloud()."""
    model = _make_model()

    with cp.cloud(model):
        # PyMC uses a context variable to track the active model
        # If the model context is entered, we can add variables to it
        active = pm.modelcontext(None)
        assert active is model


def test_cloud_delegates_to_run_sample():
    """The patched pm.sample should call _run_sample under the hood."""
    model = _make_model()

    with patch("cloudposterior.api._run_sample") as mock_run:
        mock_run.return_value = MagicMock()  # fake InferenceData

        with cp.cloud(model, instance="large", nuts_sampler="nutpie"):
            pm.sample(draws=500, chains=2)

        mock_run.assert_called_once()
        call_kwargs = mock_run.call_args[1]
        assert call_kwargs["model"] is model
        assert call_kwargs["instance"] == "large"
        assert call_kwargs["nuts_sampler"] == "nutpie"
        assert call_kwargs["draws"] == 500
        assert call_kwargs["chains"] == 2
