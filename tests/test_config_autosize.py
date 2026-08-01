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


def test_autosize_does_not_provision_a_gpu_for_jax_samplers():
    """GPU is opt-in: a JAX sampler alone is not evidence the model is big
    enough to be worth GPU rates, and auto-provisioning billed users for a
    three-parameter model. CPU images install jax/numpyro when the sampler
    needs them, so the run still works."""
    cfg = RemoteConfig.from_instance(
        None, model=_model(), sample_kwargs={"chains": 4, "draws": 1000}, nuts_sampler="numpyro",
    )
    assert cfg.gpu is None
    assert cfg.auto_sized


def test_gpu_is_available_via_the_preset():
    assert RemoteConfig.from_instance("gpu").gpu == "A100"


def test_autosize_scales_the_timeout_with_the_work():
    """A long run was killed at the fixed one-hour limit, losing the trace."""
    short = RemoteConfig.from_instance(
        None, model=_model(), sample_kwargs={"chains": 4, "draws": 1000}, nuts_sampler="pymc",
    )
    long = RemoteConfig.from_instance(
        None, model=_model(), sample_kwargs={"chains": 8, "draws": 200_000},
        nuts_sampler="pymc",
    )
    assert short.timeout == 3600  # the floor still applies to ordinary runs
    assert long.timeout > short.timeout


def test_autosize_counts_the_log_likelihood_group():
    """n_obs x chains x draws x 8 is routinely the largest consumer and the
    usual cause of an OOM."""
    import numpy as np
    import pymc as pm

    with pm.Model() as big_obs:
        mu = pm.Normal("mu", 0, 1)
        pm.Normal("obs", mu, 1, observed=np.zeros(200_000))

    with_ll = RemoteConfig.from_instance(
        None, model=big_obs, sample_kwargs={"chains": 4, "draws": 2000},
        nuts_sampler="pymc",
    )
    without_ll = RemoteConfig.from_instance(
        None, model=big_obs,
        sample_kwargs={"chains": 4, "draws": 2000,
                       "idata_kwargs": {"log_likelihood": False}},
        nuts_sampler="pymc",
    )
    assert with_ll.memory > without_ll.memory


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
