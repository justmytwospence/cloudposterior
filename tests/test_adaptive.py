"""Adaptive `until=` convergence target: normalization + worker target check."""

import msgpack
import pytest

from cloudposterior.remote.worker import _conv_meets_target


def _conv(params):
    return msgpack.packb({"type": "convergence", "params": params, "draws": 500})


def test_meets_target_when_all_params_pass():
    c = _conv({"mu": {"rhat": 1.005, "ess_bulk": 800, "ess_tail": 600}})
    assert _conv_meets_target(c, {"r_hat": 1.01, "ess": 400}) is True


def test_fails_on_rhat():
    c = _conv({"mu": {"rhat": 1.05, "ess_bulk": 800, "ess_tail": 600}})
    assert _conv_meets_target(c, {"r_hat": 1.01, "ess": 400}) is False


def test_fails_if_any_param_below_ess():
    c = _conv({
        "a": {"rhat": 1.0, "ess_bulk": 800, "ess_tail": 600},
        "b": {"rhat": 1.0, "ess_bulk": 100, "ess_tail": 600},  # bulk too low
    })
    assert _conv_meets_target(c, {"r_hat": 1.01, "ess": 400}) is False


def test_empty_params_not_converged():
    assert _conv_meets_target(_conv({}), {"r_hat": 1.01, "ess": 400}) is False


def test_until_normalization():
    pm = pytest.importorskip("pymc")
    import cloudposterior as cp

    with pm.Model() as m:
        pm.Normal("x", 0, 1)
    assert cp.cloud(m, until=True).until == {"r_hat": 1.01, "ess": 400}
    assert cp.cloud(m, until={"ess": 1000}).until == {"r_hat": 1.01, "ess": 1000}
    assert cp.cloud(m, until=None).until is None


def test_normalize_until():
    """Shared normalizer used by cp.cloud + cp.map: True/dict/None all become a
    dict-or-None so the worker never sees a bare True."""
    from cloudposterior.api import _normalize_until

    assert _normalize_until(True) == {"r_hat": 1.01, "ess": 400}
    assert _normalize_until({"ess": 1000}) == {"r_hat": 1.01, "ess": 1000}
    assert _normalize_until({"r_hat": 1.005, "ess": 800}) == {"r_hat": 1.005, "ess": 800}
    assert _normalize_until(None) is None
    assert _normalize_until(False) is None
