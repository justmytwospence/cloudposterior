"""Cross-package contracts exercised through cloudposterior without Modal billing."""

from __future__ import annotations

from unittest.mock import patch

import pymc as pm

import cloudposterior as cp
from cloudposterior import api
from cloudposterior.convergence import read_report


def _model():
    with pm.Model() as model:
        pm.Normal("theta", 0, 1)
    return model


def test_local_monitor_uses_original_sampler_and_writes_legacy_report():
    model = _model()
    with cp.cloud(model, remote=False, cache=False, progress=False, until="draft"):
        idata = pm.sample(draws=12, tune=12, chains=2, cores=1,
                          nuts_sampler="pymc", random_seed=42,
                          compute_convergence_checks=False, progressbar=False)
    attrs = idata.posterior.attrs
    assert attrs["pymc_monitor_convergence"] == attrs["cloudposterior_convergence"]
    assert read_report(idata)["status"] in {"met", "futile", "cap"}
    assert pm.sample is api._true_original("sample")


def test_cache_hit_skips_local_serialization_and_sampling():
    model = _model()
    args = dict(draws=12, tune=12, chains=2, cores=1, nuts_sampler="pymc",
                random_seed=42, compute_convergence_checks=False, progressbar=False)
    with cp.cloud(model, remote=False, cache=True, progress=False):
        first = pm.sample(**args)
        with patch("cloudposterior.api._ensure_model_bytes",
                   side_effect=AssertionError("cache hit serialized model")):
            with patch.object(api, "_run_local", side_effect=AssertionError("cache hit sampled")):
                second = pm.sample(**args)
    assert first.posterior.sizes == second.posterior.sizes


def test_cache_false_does_not_hash_or_serialize_locally():
    model = _model()
    with patch("cloudposterior.api._ensure_model_bytes",
               side_effect=AssertionError("local model serialization")):
        with patch("pymc_cache.identity.model_digest",
                   side_effect=AssertionError("result cache identity")):
            with cp.cloud(model, remote=False, cache=False, progress=False):
                idata = pm.sample(draws=8, tune=8, chains=2, cores=1,
                                  nuts_sampler="pymc", progressbar=False,
                                  compute_convergence_checks=False)
    assert idata.posterior.sizes["draw"] == 8


def test_smc_cache_hit_precedes_cloud_serialization():
    model = _model()
    sentinel = object()

    class HitCache:
        def load(self, key, **kwargs):
            return sentinel

        def save(self, key, result, **kwargs):
            raise AssertionError("cache hit was saved again")

    with patch("cloudposterior.api._ensure_model_bytes",
               side_effect=AssertionError("SMC cache hit serialized model")):
        with cp.cloud(model, remote=True, cache=HitCache(), progress=False):
            assert pm.sample_smc(draws=20) is sentinel


def test_local_multitrace_bypasses_monitor_with_warning():
    import pytest

    model = _model()
    with cp.cloud(model, cache=True, progress=False):
        with pytest.warns(UserWarning, match="enhanced monitoring and caching"):
            trace = pm.sample(draws=8, tune=8, chains=1, cores=1,
                              nuts_sampler="pymc", progressbar=False,
                              return_inferencedata=False)
    assert not hasattr(trace, "posterior")
