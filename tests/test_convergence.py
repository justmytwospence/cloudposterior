"""The convergence layer: diagnostics match ArviZ, targets, controller decisions."""

import json
import math

import numpy as np
import pytest

from cloudposterior import convergence as cv


def _from_dict(az, **groups):
    """az.from_dict across majors (0.x: group kwargs; 1.x: one dict)."""
    try:
        return az.from_dict(**groups)
    except TypeError:
        return az.from_dict(groups)



def _ar1(rng, chains, draws, k, rho=0.6, shift=None, scale=None):
    x = np.zeros((chains, draws, k))
    eps = rng.normal(size=(chains, draws, k))
    for t in range(1, draws):
        x[:, t] = rho * x[:, t - 1] + eps[:, t]
    if shift is not None:
        x += np.asarray(shift)[:, None, None]
    if scale is not None:
        x *= np.asarray(scale)[:, None, None]
    return x


@pytest.fixture(params=[(4, 500, 3), (4, 301, 2), (2, 60, 2), (8, 200, 2)])
def sample(request):
    rng = np.random.default_rng(sum(request.param))
    c, n, k = request.param
    x = _ar1(rng, c, n, k)
    x[0] *= 1.3  # one chain with a different scale
    x[:, :, -1] = np.exp(x[:, :, -1] / 3)  # skewed element
    return x


def _az_2d(fn, x, **kw):
    return np.array([float(fn(x[:, :, j], **kw)) for j in range(x.shape[2])])


def test_rhat_rank_matches_arviz(sample):
    az = pytest.importorskip("arviz")
    if int(az.__version__.split(".")[0]) >= 1:
        pytest.skip("reference values are ArviZ 0.x")
    np.testing.assert_allclose(cv.rhat_rank(sample), _az_2d(az.rhat, sample), rtol=1e-10)


def test_ess_matches_arviz(sample):
    az = pytest.importorskip("arviz")
    if int(az.__version__.split(".")[0]) >= 1:
        pytest.skip("reference values are ArviZ 0.x")
    np.testing.assert_allclose(cv.ess_bulk(sample), _az_2d(az.ess, sample), rtol=1e-8)
    np.testing.assert_allclose(
        cv.ess_tail(sample), _az_2d(az.ess, sample, method="tail"), rtol=1e-8
    )
    np.testing.assert_allclose(
        cv.ess_mean(sample), _az_2d(az.ess, sample, method="mean"), rtol=1e-8
    )


def test_mcse_quantile_matches_arviz(sample):
    az = pytest.importorskip("arviz")
    if int(az.__version__.split(".")[0]) >= 1:
        pytest.skip("reference values are ArviZ 0.x")
    for p in (0.05, 0.95):
        np.testing.assert_allclose(
            cv.mcse_quantile(sample, p),
            _az_2d(az.mcse, sample, method="quantile", prob=p),
            rtol=1e-8,
        )


def test_ess_constant_element_is_full_size():
    x = np.ones((4, 100, 1))
    assert cv.ess_basic(x)[0] == 400


def test_nested_rhat_detects_superchain_offset():
    rng = np.random.default_rng(0)
    ids = np.repeat(np.arange(4), 8)  # 4 superchains x 8 chains
    iid = rng.normal(size=(32, 20, 2))
    ok = cv.nested_rhat(iid, ids)
    assert np.all(ok < 1.05)
    bad = iid + np.repeat(np.array([0.0, 0.0, 0.0, 2.0]), 8)[:, None, None]
    assert np.all(cv.nested_rhat(bad, ids) > 1.1)


def test_nested_rhat_rejects_unequal_superchains():
    x = np.zeros((5, 10, 1))
    with pytest.raises(ValueError):
        cv.nested_rhat(x, [0, 0, 0, 1, 1])


def test_many_short_chains_classic_rhat_fails_nested_passes():
    """The nested-R-hat paper's motivating case: stationary, very short chains.
    Classic split R-hat over many short chains reads 'not converged'; nested
    R-hat with superchains doesn't."""
    rng = np.random.default_rng(1)
    x = _ar1(rng, 256, 12, 1, rho=0.9)  # stationary-ish start not needed: iid inits
    ids = np.repeat(np.arange(8), 32)
    assert cv.rhat_rank(x)[0] > 1.01
    assert cv.nested_rhat(x, ids)[0] < cv.rhat_rank(x)[0]


# -- targets ---------------------------------------------------------------


def test_normalize_target_presets_and_overrides():
    t = cv.normalize_target(True)
    assert t["r_hat"] == "auto" and t["ess"] == 400 and t["divergences"] == 0
    d = cv.normalize_target("draft")
    assert d["ess"] == 100 and d["divergences"] is None
    o = cv.normalize_target({"ess": 1000, "escalate": True, "var_names": "mu"})
    assert o["ess"] == 1000 and o["escalate"] == 2 and o["var_names"] == ["mu"]
    assert cv.normalize_target({"r_hat": 1.005})["r_hat"] == 1.005
    assert cv.normalize_target(None) is None
    assert cv.normalize_target(False) is None


def test_normalize_target_rejects_typos():
    with pytest.raises(ValueError, match="unknown until= keys"):
        cv.normalize_target({"rhat": 1.01})
    with pytest.raises(ValueError):
        cv.normalize_target("publication")
    with pytest.raises(ValueError):
        cv.normalize_target({"ess": None, "mcse": None})


def test_rhat_threshold_scales_with_chains():
    t = cv.normalize_target(True)
    assert cv.rhat_threshold(t, 4) == pytest.approx(1.01)
    assert cv.rhat_threshold(t, 32) == pytest.approx(math.sqrt(1 + 64 / 400))
    assert cv.rhat_threshold(t, 32, n_superchains=4) == pytest.approx(1.01)
    assert cv.rhat_threshold({**t, "r_hat": 1.02}, 32) == 1.02
    # An MCSE target implies an ESS target.
    m = cv.normalize_target({"ess": None, "mcse": 0.05})
    assert cv.effective_ess_target(m) == pytest.approx(400)


# -- assessment ------------------------------------------------------------


def _draws(x, names=None, **kw):
    names = names or ["a"]
    arrays = {}
    k = x.shape[2]
    per = k // len(names)
    for i, n in enumerate(names):
        arrays[n] = x[:, :, i * per:(i + 1) * per]
    return cv.draws_from_arrays(arrays, **kw)


def test_assess_checks_vector_elements():
    rng = np.random.default_rng(2)
    x = rng.normal(size=(4, 1000, 6))
    x[0, :, 4] += 3.0  # one element of the vector is stuck apart in chain 0
    d = _draws(x, names=["mu", "theta"])
    a = cv.assess(d, cv.normalize_target(True))
    assert a["checks"]["r_hat"]["ok"] is False
    assert a["checks"]["r_hat"]["worst"][0] == "theta[1]"
    assert a["params"]["mu"]["rhat"] < 1.01
    assert a["met"] is False


def test_assess_met_on_iid_draws():
    rng = np.random.default_rng(3)
    d = _draws(rng.normal(size=(4, 1000, 4)))
    a = cv.assess(d, cv.normalize_target(True))
    assert a["met"] is True, a["checks"]


def test_divergences_gate_and_cause():
    rng = np.random.default_rng(4)
    d = _draws(rng.normal(size=(4, 1000, 2)), diverging=np.zeros((4, 1000)))
    t = cv.normalize_target(True)
    assert cv.assess(d, t)["met"] is True
    div = np.zeros((4, 1000))
    div[1, 10] = 1
    d2 = _draws(rng.normal(size=(4, 1000, 2)), diverging=div)
    a = cv.assess(d2, t)
    assert a["met"] is False and a["checks"]["divergences"]["value"] == 1
    assert cv.diagnose(d2, a)[0] == "divergences"
    # draft tolerates (reports) divergences
    assert cv.assess(d2, cv.normalize_target({"preset": "draft"}))["checks"]["divergences"]["ok"] is None


def test_unequal_chain_lengths_are_trimmed():
    x = np.random.default_rng(5).normal(size=(3, 100, 1))
    x[0, 80:] = np.nan  # an aborted chain padded with NaN
    div = np.ones((3, 100))
    d = cv.draws_from_arrays({"a": x}, diverging=div)
    assert d.n_draws == 80 and d.diverging == 3 * 80


def test_mcse_and_quantile_targets():
    rng = np.random.default_rng(6)
    d = _draws(rng.normal(size=(4, 500, 2)))
    a = cv.assess(d, cv.normalize_target({"ess": None, "mcse": 0.1, "quantiles": True}))
    assert a["checks"]["mcse"]["ok"] is True
    a2 = cv.assess(d, cv.normalize_target({"ess": None, "mcse": 0.01}))
    assert a2["checks"]["mcse"]["ok"] is False
    assert cv.projected_draws(a2) > 500


def test_early_drift_vs_persistent_disagreement():
    rng = np.random.default_rng(7)
    n = 800
    drift = rng.normal(size=(4, n, 1))
    # chains start apart and converge during the first quarter
    starts = np.array([-6.0, -2.0, 2.0, 6.0])[:, None, None]
    decay = np.exp(-np.arange(n) / 60.0)[None, :, None]
    drift = drift + starts * decay
    a = cv.assess(_draws(drift), cv.normalize_target(True))
    assert a["checks"]["r_hat"]["ok"] is False
    assert cv.diagnose(_draws(drift), a)[0] == "early_drift"

    stuck = rng.normal(size=(4, n, 1)) + np.array([0, 0, 0, 3.0])[:, None, None]
    a2 = cv.assess(_draws(stuck), cv.normalize_target(True))
    assert cv.diagnose(_draws(stuck), a2)[0] == "chains_disagree"


def test_slow_but_stationary_chains_read_as_slow_mixing():
    counts = {"slow_mixing": 0}
    for seed in range(10):
        rng = np.random.default_rng(seed)
        n, rho = 800, 0.97
        x = np.zeros((4, n, 1))
        e = rng.normal(size=(4, n, 1)) * np.sqrt(1 - rho**2)
        x[:, 0] = rng.normal(size=(4, 1))
        for i in range(1, n):
            x[:, i] = rho * x[:, i - 1] + e[:, i]
        d = _draws(x)
        a = cv.assess(d, cv.normalize_target(True))
        cause = cv.diagnose(d, a)[0]
        counts[cause] = counts.get(cause, 0) + 1
    assert counts["slow_mixing"] >= 7, counts


# -- controller ------------------------------------------------------------


def test_checkpoints_are_geometric_and_need_patience():
    rng = np.random.default_rng(8)
    t = cv.normalize_target(True)
    ctl = cv.Controller(t, n_chains=4, draws_cap=5000)
    assert ctl.first_at == 100
    assert not ctl.due(99) and ctl.due(100)
    a1 = ctl.step(_draws(rng.normal(size=(4, 1000, 2))))
    assert a1["met"] and a1["decision"] == "continue"  # patience 2
    assert ctl.next_at == 1250
    a2 = ctl.step(_draws(rng.normal(size=(4, 1250, 2))))
    assert a2["decision"] == "met"


def test_display_only_controller_never_stops():
    rng = np.random.default_rng(9)
    ctl = cv.Controller(cv.normalize_target({"patience": 1}), 4, 5000, stop=False)
    assert ctl.step(_draws(rng.normal(size=(4, 1000, 1))))["decision"] == "continue"


def test_futility_on_divergence_is_immediate():
    rng = np.random.default_rng(10)
    div = np.zeros((4, 200))
    div[0, 5] = 1
    ctl = cv.Controller(cv.normalize_target(True), 4, 5000)
    a = ctl.step(_draws(rng.normal(size=(4, 200, 1)), diverging=div))
    assert a["decision"] == "futile" and a["cause"] == "divergences"


def test_futility_when_rhat_stalls():
    rng = np.random.default_rng(11)
    ctl = cv.Controller(cv.normalize_target(True), 4, 100000)
    decisions = []
    n = 100
    for _ in range(6):
        x = rng.normal(size=(4, n, 1)) + np.array([0, 0, 0, 2.0])[:, None, None]
        decisions.append(ctl.step(_draws(x))["decision"])
        n = ctl.next_at
    assert decisions[-1] == "futile"
    assert "continue" in decisions[:2]


def test_no_futility_when_futility_disabled():
    rng = np.random.default_rng(12)
    ctl = cv.Controller(cv.normalize_target({"futility": False}), 4, 5000)
    div = np.zeros((4, 200))
    div[0, 5] = 1
    assert ctl.step(_draws(rng.normal(size=(4, 200, 1)), diverging=div))["decision"] == "continue"


# -- reports ---------------------------------------------------------------


def test_report_roundtrip_through_idata():
    az = pytest.importorskip("arviz")
    rng = np.random.default_rng(13)
    idata = _from_dict(az,
        posterior={"mu": rng.normal(size=(4, 500)), "theta": rng.normal(size=(4, 500, 3))},
        sample_stats={"diverging": np.zeros((4, 500), bool), "energy": rng.normal(size=(4, 500))},
    )
    rep = cv.check_convergence(idata)
    assert rep["met"] is True and rep["status"] == "met"
    cv.attach_report(idata, rep)
    assert json.loads(idata.posterior.attrs[cv.REPORT_ATTR])["met"] is True
    assert cv.read_report(idata)["summary"] == rep["summary"]
    assert "targets met" in cv.verdict_message({**rep, "status": "met"})
