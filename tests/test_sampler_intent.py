"""Execution-only sampler intent must survive transport without changing cache keys."""

from types import SimpleNamespace
from unittest.mock import Mock

import pymc as pm
import pytest

import cloudposterior as cp
from cloudposterior import api


def _model():
    with pm.Model() as model:
        pm.Normal("x", 0, 1)
    return model


@pytest.mark.parametrize("remote", [False, True])
@pytest.mark.parametrize("provided,expected", [
    ({}, (True, True)),
    ({"nuts_sampler": "nutpie", "tune": 61}, (False, False)),
    ({"tune": 61}, (True, False)),
    ({"nuts_sampler": "nutpie"}, (False, True)),
])
def test_interceptor_preserves_intent(remote, provided, expected, monkeypatch):
    model = _model()
    captured = {}

    def lookup(_cache, _model, kwargs, **options):
        captured["key"] = dict(kwargs)
        return object(), SimpleNamespace(hit=False)

    def run(**kwargs):
        captured["execution"] = kwargs
        return object()

    monkeypatch.setattr(api, "_cache_lookup", lookup)
    monkeypatch.setattr(api, "_run_sample_persistent" if remote else "_run_sample", run)
    if remote:
        monkeypatch.setattr(api, "_ensure_model_bytes", lambda _: b"model")
        monkeypatch.setattr(api.cloud, "_provision_environment", lambda self, *args: setattr(self, "_env", object()))
    with cp.cloud(model, remote=remote, cache=False, progress=False, dashboard=False):
        pm.sample(draws=5, **provided)

    execution = captured["execution"]
    assert (execution["_cp_sampler_auto"], execution["_cp_tune_default"]) == expected
    assert execution["tune"] == provided.get("tune", api.resolve_tune({}, execution["nuts_sampler"]))
    assert captured["key"]["nuts_sampler"] == execution["nuts_sampler"]


def test_cache_lookup_excludes_execution_flags(monkeypatch):
    import pymc_cache

    calls = []

    class Session:
        def __init__(self, *args, **kwargs):
            pass

        def lookup(self, kwargs, **options):
            calls.append((kwargs, options))
            return SimpleNamespace(hit=False)

    monkeypatch.setattr(pymc_cache, "CacheSession", Session)
    monkeypatch.setattr(api, "_emit_cached_indicator", Mock(side_effect=AssertionError("hit")))
    api._cache_lookup(False, _model(), {"nuts_sampler": "nutpie", "tune": 400,
                                       "_cp_sampler_auto": True, "_cp_tune_default": True},
                      overwrite=False, progress=False)
    kwargs, options = calls.pop()
    assert kwargs == {"nuts_sampler": "nutpie", "tune": 400}
    assert options["sampler_identity"] == "nutpie"


@pytest.mark.parametrize("sampler_auto,tune_default,expected_sampler,expected_tune", [
    (True, True, None, None),
    (False, False, "nutpie", 61),
])
def test_local_monitor_receives_unresolved_intent(monkeypatch, sampler_auto, tune_default,
                                                   expected_sampler, expected_tune):
    seen = []

    def engine(model, kwargs, **options):
        seen.append((dict(kwargs), options))
        if False:
            yield
        return SimpleNamespace(idata=object())

    monkeypatch.setattr("pymc_monitor.engine.iter_sample", engine)
    result = api._run_local(model=_model(), original_sample=Mock(), sinks=[], emit=Mock(),
                            nuts_sampler="nutpie", draws=5, tune=61,
                            _cp_sampler_auto=sampler_auto, _cp_tune_default=tune_default)
    assert result is not None
    kwargs, options = seen.pop()
    assert options["nuts_sampler"] == expected_sampler
    assert kwargs.get("tune") == expected_tune
    assert not {"_cp_sampler_auto", "_cp_tune_default"} & kwargs.keys()


def test_fallback_not_stored_under_nutpie_preflight_key():
    import arviz as az
    import numpy as np
    from pymc_cache import CacheSession, MemoryCache

    model = _model()
    backend = MemoryCache()
    session = CacheSession(backend, model=model)
    lookup = session.lookup({"draws": 5, "tune": 400, "chains": 1,
                             "random_seed": 7, "nuts_sampler": "nutpie"},
                            sampler_identity="nutpie")
    assert not lookup.hit
    groups = {"posterior": {"x": np.zeros((1, 5))}}
    try:  # ArviZ 0.x takes group kwargs, 1.x one dict
        idata = az.from_dict(**groups)
    except TypeError:
        idata = az.from_dict(groups)
    assert session.store(lookup, idata, status="complete", effective_sampler="pymc") is False
    assert session.lookup({"draws": 5, "tune": 400, "chains": 1,
                           "random_seed": 7, "nuts_sampler": "nutpie"},
                          sampler_identity="nutpie").hit is False


def test_auto_nutpie_compilation_fallback_is_not_cached_as_nutpie(monkeypatch):
    import nutpie
    from pymc_cache import MemoryCache

    model = _model()
    cache = MemoryCache()
    monkeypatch.setattr(nutpie, "compile_pymc_model", Mock(side_effect=RuntimeError("compile failed")))
    with cp.cloud(model, remote=False, cache=cache, progress=False):
        with pytest.warns(RuntimeWarning, match="falling back to pymc"):
            result = pm.sample(draws=5, tune=4, chains=1, cores=1, random_seed=2,
                               progressbar=False, compute_convergence_checks=False)
    from cloudposterior.convergence import read_report

    report = read_report(result)
    assert report["requested_sampler"] == "nutpie"
    assert report["effective_sampler"] == "pymc"
    assert len(cache) == 0


def test_fallback_effective_sampler_passed_to_cache_store(monkeypatch):
    from cloudposterior import convergence

    report = {"status": "complete", "effective_sampler": "pymc"}
    cache = Mock()
    lookup = SimpleNamespace(hit=False, sampler_identity="nutpie")
    idata = object()
    monkeypatch.setattr(api, "_build_sinks", lambda **kwargs: [])
    monkeypatch.setattr(api, "_run_local", lambda **kwargs: idata)
    monkeypatch.setattr(api, "_legacy_report_aliases", lambda result: None)
    monkeypatch.setattr(convergence, "read_report", lambda result: report)
    assert api._run_sample(model=_model(), remote=False, cache=False, notify=False,
                           instance=None, nuts_sampler="nutpie", progress=False,
                           original_sample=Mock(), cache_session=cache,
                           cache_lookup=lookup, draws=5, tune=400) is idata
    cache.store.assert_called_once_with(lookup, idata, status="complete",
                                        effective_sampler="pymc")
