"""Regression tests for per-sampler dispatch and default-sampler selection.

These guard the headline bug: numpyro/blackjax were silently ignored (the worker
ran the pymc CPU sampler on a paid GPU box), and a callback was passed to external
samplers, which PyMC 6 rejects with a ValueError.
"""

from unittest.mock import patch

import numpy as np
import pymc as pm
import pytest

from cloudposterior.api import _default_sampler, _validate_sample_kwargs
from cloudposterior.remote.worker import run_sampling
from cloudposterior.serialize import serialize_model


def _model():
    with pm.Model() as m:
        pm.Normal("mu", 0, 1)
        pm.Normal("y", 0, 1, observed=np.array([0.1, -0.2, 0.3]))
    return m


def _real_idata():
    """A tiny real InferenceData/DataTree to hand back from the mocked pm.sample."""
    with _model():
        return pm.sample(
            draws=5, tune=5, chains=1, nuts_sampler="pymc",
            progressbar=False, random_seed=0,
        )


def _drain(gen):
    for _ in gen:
        pass


# -- worker dispatch --------------------------------------------------------

@pytest.mark.parametrize("sampler", ["numpyro", "blackjax"])
def test_external_sampler_forwarded_without_callback(sampler):
    """numpyro/blackjax must reach pm.sample as nuts_sampler, with NO callback."""
    idata = _real_idata()
    captured = {}

    def fake_sample(*args, **kwargs):
        captured.update(kwargs)
        return idata

    model_bytes = serialize_model(_model())
    with patch("pymc.sample", side_effect=fake_sample):
        _drain(run_sampling(model_bytes, {"draws": 5, "tune": 5, "chains": 1}, sampler))

    assert captured.get("nuts_sampler") == sampler
    assert "callback" not in captured  # PyMC 6 raises if a callback is passed here


def test_pymc_sampler_forces_pymc_and_uses_callback():
    """The pymc path must pin nuts_sampler='pymc' (so PyMC 6 doesn't auto-pick
    nutpie) and attach the per-draw callback."""
    idata = _real_idata()
    captured = {}

    def fake_sample(*args, **kwargs):
        captured.update(kwargs)
        return idata

    model_bytes = serialize_model(_model())
    with patch("pymc.sample", side_effect=fake_sample):
        _drain(run_sampling(model_bytes, {"draws": 5, "tune": 5, "chains": 1}, "pymc"))

    assert captured.get("nuts_sampler") == "pymc"
    assert callable(captured.get("callback"))


# -- default sampler selection ----------------------------------------------

def test_default_sampler_continuous_is_nutpie():
    assert _default_sampler(_model(), local=False) == "nutpie"


def test_default_sampler_discrete_is_pymc():
    with pm.Model() as m:
        p = pm.Beta("p", 1, 1)
        pm.Bernoulli("z", p=p)  # discrete free RV -> nutpie/JAX can't handle it
        pm.Normal("y", 0, 1, observed=np.zeros(3))
    assert _default_sampler(m, local=False) == "pymc"


def test_default_sampler_local_falls_back_without_nutpie(monkeypatch):
    monkeypatch.setattr("cloudposterior.api._nutpie_available", lambda: False)
    assert _default_sampler(_model(), local=True) == "pymc"   # local: nutpie missing
    assert _default_sampler(_model(), local=False) == "nutpie"  # remote: always shipped


# -- kwarg validation -------------------------------------------------------

@pytest.mark.parametrize("bad", [{"chains": "4"}, {"draws": 0}, {"tune": -5}, {"cores": 2.5}])
def test_validate_rejects_bad_core_counts(bad):
    with pytest.raises(TypeError):
        _validate_sample_kwargs(dict(bad))


def test_validate_allows_passthrough_kwargs():
    # target_accept / random_seed and valid ints must pass through untouched.
    _validate_sample_kwargs({"chains": 4, "draws": 1000, "target_accept": 0.95, "random_seed": 42})
