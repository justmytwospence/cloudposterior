"""Test the resize-mismatch warning logic in isolation.

The warning fires when a later pm.sample() inside the same cp.cloud(...)
block uses kwargs that would have auto-sized to a different VM than the one
already provisioned. This test exercises the logic with a fake env so it
runs without Modal.
"""

from types import SimpleNamespace

import numpy as np
import pymc as pm
import pytest

import cloudposterior as cp
from cloudposterior.api import _warn_if_resize_drift
from cloudposterior.config import RemoteConfig


def _make_model():
    with pm.Model() as model:
        mu = pm.Normal("mu", 0, 1)
        pm.Normal("y", mu, 1, observed=np.array([1.0, 2.0, 3.0]))
    return model


def _ctx(model, env_config: RemoteConfig, instance: str | None = None):
    """Build a minimal stand-in for the cloud context manager."""
    return SimpleNamespace(
        model=model,
        instance=instance,
        _env=SimpleNamespace(config=env_config),
    )


def test_no_warning_when_kwargs_match_provisioned_size():
    model = _make_model()
    provisioned = RemoteConfig.from_instance(
        None, model=model, sample_kwargs={"chains": 4, "draws": 1000}, nuts_sampler="pymc",
    )
    ctx = _ctx(model, provisioned)
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _warn_if_resize_drift(ctx, "pymc", {"chains": 4, "draws": 1000})


def test_warning_when_chains_would_change_auto_size():
    model = _make_model()
    provisioned = RemoteConfig.from_instance(
        None, model=model, sample_kwargs={"chains": 2, "draws": 1000}, nuts_sampler="pymc",
    )
    ctx = _ctx(model, provisioned)
    with pytest.warns(UserWarning, match="auto-size"):
        _warn_if_resize_drift(ctx, "pymc", {"chains": 16, "draws": 1000})


def test_no_warning_when_user_pinned_an_instance_preset():
    """If the user explicitly chose instance="large", they accepted the size.
    Auto-size drift is irrelevant -- don't pester them."""
    model = _make_model()
    pinned = RemoteConfig.from_instance("large")
    ctx = _ctx(model, pinned, instance="large")
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _warn_if_resize_drift(ctx, "pymc", {"chains": 16, "draws": 1000})


def test_no_warning_when_env_has_no_config_attribute():
    """Defensive: older envs (or test mocks) without .config should be a no-op."""
    model = _make_model()
    ctx = SimpleNamespace(model=model, instance=None, _env=SimpleNamespace())
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _warn_if_resize_drift(ctx, "pymc", {"chains": 16})
