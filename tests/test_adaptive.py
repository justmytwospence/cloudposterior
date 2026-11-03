"""Adaptive `until=`: normalization, worker-side stopping, superchains, escalation.

The worker runs locally here (no Modal): ``run_sampling`` is the same
generator the container drives.
"""

import json

import msgpack
import numpy as np
import pytest

from cloudposterior import convergence as cv


def _from_dict(az, **groups):
    """az.from_dict across majors (0.x: group kwargs; 1.x: one dict)."""
    try:
        return az.from_dict(**groups)
    except TypeError:
        return az.from_dict(groups)



def test_until_normalization():
    pm = pytest.importorskip("pymc")
    import cloudposterior as cp

    with pm.Model() as m:
        pm.Normal("x", 0, 1)
    assert cp.cloud(m, until=True).until == cv.normalize_target(True)
    assert cp.cloud(m, until={"ess": 1000}).until["ess"] == 1000
    assert cp.cloud(m, until="draft").until["ess"] == 100
    assert cp.cloud(m, until=None).until is None
    with pytest.raises(ValueError):
        cp.cloud(m, until={"rhat": 1.01})  # typo fails loudly


def test_normalize_until_shared_by_cloud_and_map():
    from cloudposterior.api import _normalize_until

    assert _normalize_until(True) == cv.normalize_target(True)
    assert _normalize_until({"r_hat": 1.005, "ess": 800})["r_hat"] == 1.005
    assert _normalize_until(None) is None
    assert _normalize_until(False) is None


# -- worker, run locally ------------------------------------------------------


def _run(model, kwargs, sampler):
    from cloudposterior.remote.worker import run_sampling
    from cloudposterior.serialize import create_payload, deserialize_inference_data

    payload = create_payload(model, kwargs)
    events, result = [], None
    for chunk in run_sampling(payload.model_bytes, payload.sample_kwargs, sampler):
        try:
            events.append(msgpack.unpackb(chunk, raw=False))
        except Exception:
            result = chunk
    idata = deserialize_inference_data(result)
    return events, idata


def _regression():
    pm = pytest.importorskip("pymc")
    rng = np.random.default_rng(0)
    x = rng.normal(size=(500, 3))
    with pm.Model() as m:
        b = pm.Normal("b", 0, 1, shape=3)  # vector-only free RVs + one scalar
        s = pm.HalfNormal("s", 2)
        pm.Normal("y", x @ b, s, observed=x @ np.array([1.0, -1.0, 0.5]) + rng.normal(size=500))
    return m


@pytest.mark.parametrize("sampler", ["nutpie", "pymc"])
def test_worker_stops_once_targets_met(sampler):
    if sampler == "nutpie":
        pytest.importorskip("nutpie")
    events, idata = _run(
        _regression(),
        {"draws": 400000 if sampler == "nutpie" else 20000, "tune": 300, "chains": 4,
         "until": True, "random_seed": 1},
        sampler,
    )
    report = cv.read_report(idata)
    assert report["status"] == "met" and report["met"] is True
    n = idata.posterior.sizes["draw"]
    assert n < (400000 if sampler == "nutpie" else 20000)
    assert not np.isnan(idata.posterior["b"].values).any()  # aborted tails trimmed
    conv = [e for e in events if e.get("type") == "convergence"]
    assert conv and "b" in conv[-1]["params"]  # vector variables are monitored
    done = [e for e in events if e.get("type") == "phase" and e.get("status") == "done"
            and e.get("phase") == "sampling"]
    assert "targets met" in done[-1]["message"]


def test_worker_reports_without_until():
    pytest.importorskip("nutpie")
    _, idata = _run(_regression(), {"draws": 300, "tune": 200, "chains": 2, "random_seed": 2}, "nutpie")
    report = cv.read_report(idata)
    assert report["status"] == "complete" and report["target"] is None
    assert idata.posterior.sizes["draw"] == 300


@pytest.mark.parametrize("sampler", ["nutpie", "pymc"])
def test_worker_superchains_share_starts_and_use_nested_rhat(sampler):
    if sampler == "nutpie":
        pytest.importorskip("nutpie")
    events, idata = _run(
        _regression(),
        {"draws": 200, "tune": 200, "chains": 8, "cores": 8,
         "until": {"superchains": 4, "patience": 1}, "cp_init": "prior", "random_seed": 3},
        sampler,
    )
    assert idata.posterior.sizes["chain"] == 8
    report = cv.read_report(idata)
    assert report["checks"]["r_hat"]["nested"] is True
    ids = idata.posterior.attrs[cv.SUPERCHAIN_ATTR]
    if isinstance(ids, str):
        ids = json.loads(ids)
    assert list(np.asarray(ids).tolist()) == [0, 0, 1, 1, 2, 2, 3, 3]


def test_superchains_must_divide_chains():
    with pytest.raises(ValueError, match="divide"):
        _run(_regression(), {"draws": 10, "tune": 10, "chains": 6,
                             "until": {"superchains": 4}}, "pymc")


def test_escalation_ladder_steps():
    from cloudposterior.remote.worker import _next_rung

    settings = {"draws": 1000, "tune": 1000, "extra": {}}
    applied = {}
    rep = {"cause": "divergences"}
    assert "0.95" in _next_rung(rep, settings, "nutpie", applied)
    assert "0.99" in _next_rung(rep, settings, "nutpie", applied)
    assert _next_rung(rep, settings, "nutpie", applied) is None

    applied = {}
    assert "tune 1000 -> 2000" in _next_rung({"cause": "early_drift"}, settings, "pymc", applied)
    # low-rank adaptation is nutpie-only: pymc skips to the next step
    applied = {}
    desc = _next_rung({"cause": "ess_stalled"}, settings, "pymc", applied)
    assert desc.startswith("tune")
    desc = _next_rung({"cause": "slow_mixing", "projected_draws": 1500}, settings, "nutpie", {})
    assert settings["draws"] == 1800 and "draws 1000 -> 1800" in desc


def test_trim_incomplete_draws_and_concat_chains():
    az = pytest.importorskip("arviz")
    from cloudposterior._idata import concat_chains, trim_incomplete_draws

    post = np.random.default_rng(0).normal(size=(2, 10))
    post[1, 7:] = np.nan  # aborted chain padded with NaN
    idata = _from_dict(az, posterior={"a": post}, sample_stats={"diverging": np.zeros((2, 10), bool)})
    trimmed = trim_incomplete_draws(idata)
    assert trimmed.posterior.sizes["draw"] == 7
    assert trimmed.sample_stats.sizes["draw"] == 7

    other = _from_dict(az, posterior={"a": np.ones((3, 9))}, sample_stats={"diverging": np.zeros((3, 9), bool)})
    joined = concat_chains([trimmed, other])
    assert joined.posterior.sizes["chain"] == 5 and joined.posterior.sizes["draw"] == 7
    assert list(joined.posterior.chain.values) == [0, 1, 2, 3, 4]


# -- client side --------------------------------------------------------------


def test_init_kwarg_is_validated():
    pm = pytest.importorskip("pymc")
    import cloudposterior as cp

    with pm.Model() as m:
        pm.Normal("x", 0, 1)
    assert cp.cloud(m, init="prior").init == "prior"
    with pytest.raises(ValueError, match="init="):
        cp.cloud(m, init="pathfinder")


def test_superchain_and_chain_count_checks():
    from cloudposterior.api import _check_superchains

    with pytest.raises(ValueError, match="divide"):
        _check_superchains(cv.normalize_target({"superchains": 4}), {"chains": 6})
    _check_superchains(cv.normalize_target({"superchains": 4}), {"chains": 8})
    with pytest.warns(UserWarning, match="at least 4"):
        _check_superchains(cv.normalize_target(True), {"chains": 2})


def test_map_threads_init_and_until(monkeypatch):
    pytest.importorskip("pymc")
    import cloudposterior as cp
    from tests.test_map import _fake_modal, _models

    state = _fake_modal(monkeypatch)
    cp.map(_models(1), {"draws": 10, "chains": 4}, cache=False, until="draft", init="prior")
    kw = state["spawned"][0]["kw"]
    assert kw["cp_init"] == "prior" and kw["until"]["ess"] == 100


def test_remote_verdict_warns_with_hint():
    az = pytest.importorskip("arviz")
    from cloudposterior.api import _report_verdict

    idata = _from_dict(az, posterior={"a": np.zeros((2, 5))})
    cv.attach_report(idata, {
        "status": "cap", "met": False, "draws": 5, "summary": "max R-hat 1.2",
        "hint": "chains disagree", "failing": ["r_hat"],
    })
    with pytest.warns(UserWarning, match="targets NOT met.*chains disagree"):
        _report_verdict(idata)
    cv.attach_report(idata, {"status": "met", "met": True, "summary": "ok"})
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _report_verdict(idata)  # met -> silent


def test_escalation_budgets_the_timeout():
    pm = pytest.importorskip("pymc")
    from cloudposterior.config import RemoteConfig

    with pm.Model() as m:
        pm.Normal("x", 0, 1, shape=3)
    base = {"draws": 200000, "tune": 1000, "chains": 4}
    plain = RemoteConfig._auto(m, dict(base))
    esc = RemoteConfig._auto(m, {**base, "until": cv.normalize_target({"escalate": 2})})
    assert esc.timeout >= plain.timeout


def test_convergence_event_decodes_verdict_and_reaches_dashboard():
    from cloudposterior.dashboard import DashboardSink
    from cloudposterior.progress import decode_progress_event

    event = decode_progress_event({
        "type": "convergence", "params": {"mu": {"rhat": 1.02, "ess_bulk": 90, "ess_tail": 80}},
        "draws": 100, "partial": True,
        "verdict": {"met": False, "decision": "continue", "projected_draws": 450,
                    "checks": {"r_hat": {"threshold": 1.01}}},
    })
    assert event.verdict["projected_draws"] == 450 and event.final is False
    store = {}
    DashboardSink(store).show_convergence(event)
    assert store["progress"]["convergence"]["verdict"]["projected_draws"] == 450
