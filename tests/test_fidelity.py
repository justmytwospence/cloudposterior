"""Fidelity guardrails: cloudposterior must not silently diverge from native
PyMC. The two pm.sample behaviors that can't be matched for remote execution
(return_inferencedata=False -> MultiTrace, and a per-draw local callback) warn
loudly instead of being dropped in silence."""

import pytest


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
