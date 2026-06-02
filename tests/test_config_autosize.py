"""Auto-sizing edge cases and instance-preset validation (config.py)."""

import numpy as np
import pymc as pm
import pytest

from cloudposterior.config import RemoteConfig


def _model(n_obs=100):
    with pm.Model() as m:
        mu = pm.Normal("mu", 0, 1)
        pm.Normal("y", mu, 1, observed=np.zeros(n_obs))
    return m


def test_unknown_instance_preset_raises():
    with pytest.raises(ValueError, match="unknown instance preset"):
        RemoteConfig.from_instance("xlrge")  # typo for "xlarge"


def test_known_presets():
    assert RemoteConfig.from_instance("gpu").gpu == "A100"
    assert RemoteConfig.from_instance("small").cpu == 4
    assert RemoteConfig.from_instance("xlarge").memory == 65536


def test_autosize_provisions_gpu_for_jax_samplers():
    cfg = RemoteConfig.from_instance(
        None, model=_model(), sample_kwargs={"chains": 4, "draws": 1000}, nuts_sampler="numpyro",
    )
    assert cfg.gpu == "A10G"
    assert cfg.auto_sized


def test_autosize_no_gpu_for_nutpie_default():
    cfg = RemoteConfig.from_instance(
        None, model=_model(), sample_kwargs={"chains": 4, "draws": 1000}, nuts_sampler="nutpie",
    )
    assert cfg.gpu is None


def test_autosize_handles_model_without_observed_rvs():
    with pm.Model() as m:
        pm.Normal("mu", 0, 1)  # prior-only model, no observed data
    cfg = RemoteConfig.from_instance(None, model=m, sample_kwargs={"chains": 2}, nuts_sampler="pymc")
    assert cfg.cpu >= 4
    assert cfg.memory >= 4096


def test_autosize_memory_rounds_to_power_of_two_gb():
    cfg = RemoteConfig.from_instance(
        None, model=_model(n_obs=1), sample_kwargs={"chains": 4, "draws": 1000}, nuts_sampler="pymc",
    )
    gb = cfg.memory // 1024
    assert gb & (gb - 1) == 0  # power of two
    assert cfg.memory <= 65536


def test_describe_mentions_gpu_when_present():
    cfg = RemoteConfig.from_instance("gpu")
    assert "A100" in cfg.describe()
