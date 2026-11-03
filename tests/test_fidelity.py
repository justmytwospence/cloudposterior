"""Fidelity guardrails: cloudposterior must not silently diverge from native
PyMC. The two pm.sample behaviors that can't be matched for remote execution
(return_inferencedata=False -> MultiTrace, and a per-draw local callback) warn
loudly instead of being dropped in silence."""

from unittest.mock import Mock

import pymc as pm
import pytest

import cloudposterior as cp
from cloudposterior import api


def test_return_inferencedata_false_warns():
    pytest.importorskip("pymc")
    from cloudposterior.api import _warn_remote_sample_fidelity

    with pytest.warns(UserWarning, match="return_inferencedata=False"):
        _warn_remote_sample_fidelity({"return_inferencedata": False})


def test_callback_warns():
    pytest.importorskip("pymc")
    from cloudposterior.api import _warn_remote_sample_fidelity

    with pytest.warns(UserWarning, match="callback"):
        _warn_remote_sample_fidelity({"callback": lambda *a, **k: None})


def test_clean_kwargs_do_not_warn(recwarn):
    pytest.importorskip("pymc")
    from cloudposterior.api import _warn_remote_sample_fidelity

    _warn_remote_sample_fidelity({"draws": 1000, "return_inferencedata": True})
    assert len(recwarn) == 0


def _trace_model():
    with pm.Model() as model:
        pm.Normal("x", 0, 1)
    return model


def test_local_trace_native_backend_and_callback():
    from pymc.backends.ndarray import NDArray

    model = _trace_model()
    trace = NDArray(model=model)
    calls = []

    def callback(trace, draw):
        calls.append((trace, draw))

    with cp.cloud(model, remote=False, cache=True, dashboard=True, progress=False):
        with pytest.warns(UserWarning, match="trace=.*enhanced monitoring, dashboard, and caching"):
            idata = pm.sample(draws=6, tune=4, chains=1, cores=1, trace=trace,
                              callback=callback, nuts_sampler="pymc", progressbar=False,
                              compute_convergence_checks=False, random_seed=13)
    assert idata.posterior.sizes["draw"] == 6
    assert calls and all(isinstance(seen, NDArray) for seen, _ in calls)


def test_local_trace_skips_cache_serialization_and_monitor(monkeypatch):
    model = _trace_model()
    trace = object()
    callback = Mock()
    native = Mock(return_value=object())
    for name in ("_cache_lookup", "_ensure_model_bytes", "_ensure_step_bytes",
                 "_payload_fingerprint", "_validate_sample_kwargs", "_run_local"):
        monkeypatch.setattr(api, name, Mock(side_effect=AssertionError(name)))
    session = cp.cloud(model, remote=False, cache=True, dashboard=True, progress=False)
    with session, monkeypatch.context() as guard:
        guard.setitem(session._originals, "sample", native)
        with pytest.warns(UserWarning, match="trace=.*native PyMC"):
            result = pm.sample(trace=trace, callback=callback, draws=7)
    assert result is native.return_value
    native.assert_called_once_with(trace=trace, callback=callback, draws=7)


@pytest.mark.parametrize("target", ["draft", {"ess": 30}])
def test_local_trace_rejects_adaptive_before_sampling(target, monkeypatch):
    model = _trace_model()
    native = Mock(side_effect=AssertionError("sampled"))
    monkeypatch.setattr(api, "_cache_lookup", Mock(side_effect=AssertionError("hashed")))
    session = cp.cloud(model, remote=False, until=target, progress=False)
    with session, monkeypatch.context() as guard:
        guard.setitem(session._originals, "sample", native)
        with pytest.raises(ValueError, match="adaptive sampling is unavailable with trace"):
            pm.sample(trace=object(), draws=7)
    native.assert_not_called()
