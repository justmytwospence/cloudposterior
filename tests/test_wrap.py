"""Test the cp.cloud() context manager."""

import warnings
from unittest.mock import patch, MagicMock

import numpy as np
import pymc as pm
import pytest

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

        with cp.cloud(model, instance="large"):
            pm.sample(draws=500, chains=2, nuts_sampler="nutpie")

        mock_run.assert_called_once()
        call_kwargs = mock_run.call_args[1]
        assert call_kwargs["model"] is model
        assert call_kwargs["instance"] == "large"
        assert call_kwargs["nuts_sampler"] == "nutpie"
        assert call_kwargs["draws"] == 500
        assert call_kwargs["chains"] == 2


def test_cloud_default_does_not_warn_about_dashboard():
    """The default cp.cloud(model) (no remote, no explicit dashboard) must not
    spam users with a dashboard warning -- only explicit dashboard=True without
    remote=True should warn."""
    model = _make_model()
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # promote warnings to errors
        with cp.cloud(model):
            pass


def test_cloud_explicit_dashboard_without_remote_warns():
    """Explicit dashboard=True without remote=True is a user error; warn."""
    model = _make_model()
    with pytest.warns(UserWarning, match="dashboard=True has no effect"):
        with cp.cloud(model, dashboard=True):
            pass


def test_remote_pops_callback_from_kwargs_and_cache_key():
    """callback= is warned about and removed for remote runs: a function repr
    embeds a memory address, which would make disk-cache keys unstable."""
    model = _make_model()
    captured = {}

    def fake_lookup(cache_arg, m, mb, cache_kwargs, *, overwrite, progress):
        captured["cache_kwargs"] = cache_kwargs
        return None, None, None

    def fake_provision(self, nuts_sampler, kwargs):
        self._env = MagicMock()

    with patch("cloudposterior.api._cache_lookup", side_effect=fake_lookup), \
         patch("cloudposterior.api._run_sample_persistent") as mock_run, \
         patch.object(cp.cloud, "_provision_environment", fake_provision):
        mock_run.return_value = MagicMock()
        with cp.cloud(model, remote=True):
            with pytest.warns(UserWarning, match="callback"):
                pm.sample(draws=10, callback=lambda *a, **k: None)

    assert "callback" not in captured["cache_kwargs"]
    assert "callback" not in mock_run.call_args[1]


def test_remote_cache_hit_skips_provisioning():
    """A cache hit must not create Modal infra (Volume/Dict/app)."""
    model = _make_model()
    sentinel = MagicMock(name="cached_idata")

    def fake_lookup(*args, **kwargs):
        return MagicMock(), "key", sentinel

    def boom(self, nuts_sampler, kwargs):
        raise AssertionError("provisioned despite cache hit")

    with patch("cloudposterior.api._cache_lookup", side_effect=fake_lookup), \
         patch.object(cp.cloud, "_provision_environment", boom):
        with cp.cloud(model, remote=True):
            out = pm.sample(draws=10)

    assert out is sentinel


def test_sample_inside_other_model_falls_back_to_native():
    """pm.sample() targeting a model other than the wrapped one must run
    natively (intercepting would silently sample the wrong model)."""
    m1 = _make_model()
    m2 = _make_model()
    native = MagicMock(name="native_sample")
    real_sample = pm.sample
    pm.sample = native
    try:
        with cp.cloud(m1):
            with m2:
                with pytest.warns(UserWarning, match="different model"):
                    pm.sample(draws=10)
        native.assert_called_once_with(draws=10)
    finally:
        pm.sample = real_sample


def test_sinks_stopped_when_sampling_raises():
    """A failed run must still stop the sinks (a Rich Live display left
    running garbles the terminal)."""
    from cloudposterior.api import _run_sample

    model = _make_model()
    sink = MagicMock()

    with patch("cloudposterior.api._build_sinks", return_value=[sink]), \
         patch("cloudposterior.api._run_local", side_effect=RuntimeError("boom")):
        with pytest.raises(RuntimeError, match="boom"):
            _run_sample(
                model=model, remote=False, cache=False, notify=False,
                instance=None, nuts_sampler="pymc", progress=False,
                original_sample=MagicMock(), draws=10,
            )

    sink.stop.assert_called_once()


def test_local_user_callback_composes_with_progress_callback():
    """notify/progress sinks attach our per-draw callback; a user callback=
    must compose with it instead of raising a duplicate-kwarg TypeError."""
    from cloudposterior.api import _run_local

    model = _make_model()
    calls = {"n": 0}

    def user_cb(trace, draw):
        calls["n"] += 1

    _run_local(
        model=model, original_sample=pm.sample, sinks=[MagicMock()],
        emit=lambda e: None, nuts_sampler="pymc",
        draws=10, tune=10, chains=1, callback=user_cb, progressbar=False,
    )
    assert calls["n"] > 0
