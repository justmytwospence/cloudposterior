"""Convergence assessment and adaptive stopping for MCMC runs.

This is cloudposterior's convergence layer. It decides when a run has met its
targets, when more draws cannot fix it, why, and what to try next. The worker
uses it to stop sampling early; ``cp.check_convergence`` uses it after the fact
on any InferenceData.

Design (see the references at the bottom):

- Every element of every monitored variable is checked, not only scalars. A
  hierarchical model's group-level vectors are where funnels live.
- Diagnostics are vectorized numpy reimplementations of ArviZ 0.x's
  rank-normalized split R-hat, bulk/tail ESS, mean ESS and quantile MCSE
  (tested against ArviZ). That keeps the layer identical across ArviZ 0.x/1.x
  and fast enough to run on thousands of elements during sampling.
- The stop decision is taken only at geometrically spaced checkpoints and must
  hold at ``patience`` consecutive checkpoints. Checking every few draws and
  stopping on the first pass selects favourable noise (Cowles, Roberts and
  Rosenthal 1998).
- HMC pathologies gate the verdict: post-warmup divergences and E-BFMI.
- The R-hat threshold accounts for the chain count. At stationarity split R-hat
  satisfies R^2 - 1 ~= 2m / ESS (Vats and Knudson 2021), so the fixed 1.01
  forces ~100 ESS per chain: fine for 4 chains, wasteful for 32.
- Superchains (groups of chains sharing a starting point) switch the check to
  nested R-hat (Margossian et al. 2024), which works for many short chains.
- ESS should grow linearly with draws (Vehtari et al. 2021). The controller
  extrapolates the time to the target and stops for futility when R-hat or ESS
  stall, naming the likely cause (warmup too short, chains disagreeing,
  divergences, slow mixing) instead of burning the draw cap.

References:
  Vehtari, Gelman, Simpson, Carpenter, Buerkner (2021). Rank-normalization,
    folding, and localization: an improved R-hat. Bayesian Analysis.
  Margossian, Hoffman, Sountsov, Riou-Durand, Vehtari, Gelman (2024). Nested
    R-hat: assessing the convergence of MCMC when running many short chains.
    Bayesian Analysis.
  Margossian and Gelman (2024). For how many iterations should we run MCMC?
    Handbook of MCMC, 2nd ed.
  Vats and Knudson (2021). Revisiting the Gelman-Rubin diagnostic. Stat. Sci.
  Cowles, Roberts, Rosenthal (1998). Possible biases induced by MCMC
    convergence diagnostics. J. Stat. Comput. Simul.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field

import numpy as np

# ---------------------------------------------------------------------------
# Targets
# ---------------------------------------------------------------------------

#: Convergence targets. ``r_hat="auto"`` scales with the chain count.
PRESETS: dict[str, dict] = {
    # Publication-grade: Vehtari et al. (2021) thresholds plus HMC gates.
    "final": {
        "r_hat": "auto",
        "ess": 400,
        "mcse": None,
        "quantiles": False,
        "divergences": 0,
        "bfmi": 0.3,
        "treedepth": None,
    },
    # Model development: enough to see whether the model is broken
    # (Margossian and Gelman 2024, "inference precision vs. workflow stage").
    "draft": {
        "r_hat": 1.05,
        "ess": 100,
        "mcse": None,
        "quantiles": False,
        "divergences": None,
        "bfmi": None,
        "treedepth": None,
    },
}

#: Controller settings shared by every preset.
CONTROL_DEFAULTS: dict = {
    "var_names": None,  # None -> every free random variable
    "patience": 2,  # consecutive passing checkpoints required to stop
    "growth": 1.25,  # checkpoint spacing factor (draws per chain)
    "futility": True,  # stop when more draws cannot meet the target
    "escalate": 0,  # automatic re-runs with adjusted settings on failure
    "superchains": None,  # K groups of chains sharing a start -> nested R-hat
}

_TARGET_KEYS = set(PRESETS["final"]) | set(CONTROL_DEFAULTS)


def normalize_target(until) -> dict | None:
    """Resolve an ``until=`` value into a complete target dict (or None).

    - ``None`` / ``False`` -> None (no adaptive stopping)
    - ``True`` -> the ``"final"`` preset
    - ``"final"`` / ``"draft"`` -> that preset
    - a dict -> overrides on top of ``dict["preset"]`` (default ``"final"``)

    Unknown keys raise, so a typo cannot silently disable a check.
    """
    if until is None or until is False:
        return None
    if until is True:
        until = "final"
    if isinstance(until, str):
        if until not in PRESETS:
            raise ValueError(
                f"until={until!r} is not a preset; use one of {sorted(PRESETS)} "
                "or a dict of targets"
            )
        until = {"preset": until}
    if not isinstance(until, dict):
        raise TypeError(
            f"until must be bool | str | dict; got {type(until).__name__}={until!r}"
        )
    raw = dict(until)
    preset = raw.pop("preset", "final")
    if preset not in PRESETS:
        raise ValueError(f"unknown preset {preset!r}; use one of {sorted(PRESETS)}")
    unknown = set(raw) - _TARGET_KEYS
    if unknown:
        raise ValueError(
            f"unknown until= keys {sorted(unknown)}; valid keys are "
            f"{sorted(_TARGET_KEYS | {'preset'})}"
        )
    target = {**PRESETS[preset], **CONTROL_DEFAULTS, **raw}

    esc = target["escalate"]
    target["escalate"] = 2 if esc is True else int(esc or 0)
    if target["var_names"] is not None:
        names = target["var_names"]
        target["var_names"] = [names] if isinstance(names, str) else list(names)
    if target["r_hat"] != "auto" and target["r_hat"] is not None:
        target["r_hat"] = float(target["r_hat"])
    if target["ess"] is None and target["mcse"] is None:
        raise ValueError("until= needs at least one precision target: ess or mcse")
    if target["patience"] < 1:
        raise ValueError("until patience must be >= 1")
    if target["growth"] <= 1:
        raise ValueError("until growth must be > 1")
    if target["superchains"] is not None:
        target["superchains"] = int(target["superchains"])
        if target["superchains"] < 2:
            raise ValueError("until superchains must be >= 2")
    return target


def effective_ess_target(target: dict) -> float:
    """The ESS the target implies (an MCSE of e*sd needs ESS ~ 1/e^2)."""
    ess = target.get("ess")
    mcse = target.get("mcse")
    candidates = [float(ess)] if ess else []
    if mcse:
        candidates.append(1.0 / float(mcse) ** 2)
    return max(candidates) if candidates else 400.0


def rhat_threshold(target: dict, n_chains: int, n_superchains: int | None = None) -> float:
    """R-hat threshold for a run with ``n_chains`` chains.

    ``"auto"``: 1.01 (Vehtari et al. 2021) for up to ~4 chains. Beyond that the
    threshold is the value R-hat takes at stationarity once the ESS target is
    reached, sqrt(1 + 2m/ESS) for split R-hat (Vats and Knudson 2021), so more
    chains do not force more draws than the precision target needs. For
    nested R-hat with K superchains it is sqrt(1 + (K + 0.2)/ESS): Margossian
    et al. (2024, sec. 3.4) with the nonstationary-variance tolerance set to a
    fifth of the target's relative variance.
    """
    r_hat = target.get("r_hat", "auto")
    if r_hat not in (None, "auto"):
        return float(r_hat)
    ess = effective_ess_target(target)
    if n_superchains:
        return max(1.01, math.sqrt(1.0 + (n_superchains + 0.2) / ess))
    return max(1.01, math.sqrt(1.0 + 2.0 * max(1, n_chains) / ess))


# ---------------------------------------------------------------------------
# Vectorized diagnostics. Arrays are (chain, draw, element).
# ---------------------------------------------------------------------------


def split_chains(x: np.ndarray) -> np.ndarray:
    """Split each chain in half: (C, N, K) -> (2C, N//2, K)."""
    half = x.shape[1] // 2
    return np.concatenate([x[:, :half], x[:, x.shape[1] - half:]], axis=0)


def z_scale(x: np.ndarray) -> np.ndarray:
    """Rank-normalize over all chains and draws, per element (Blom offsets)."""
    from scipy import stats
    from scipy.special import ndtri

    c, n, k = x.shape
    flat = x.reshape(c * n, k)
    ranks = stats.rankdata(flat, method="average", axis=0)
    size = c * n
    z = ndtri((ranks - 3.0 / 8.0) / (size - 2 * 3.0 / 8.0 + 1.0))
    return z.reshape(c, n, k)


def _rhat_basic(x: np.ndarray) -> np.ndarray:
    n = x.shape[1]
    chain_mean = x.mean(axis=1)
    chain_var = x.var(axis=1, ddof=1)
    between = n * chain_mean.var(axis=0, ddof=1)
    within = chain_var.mean(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.sqrt((between / within + n - 1) / n)


def rhat_rank(x: np.ndarray) -> np.ndarray:
    """Rank-normalized split R-hat: max of bulk and folded (ArviZ ``method="rank"``)."""
    xs = split_chains(x)
    bulk = _rhat_basic(z_scale(xs))
    k = xs.shape[2]
    folded = np.abs(xs - np.median(xs.reshape(-1, k), axis=0))
    tail = _rhat_basic(z_scale(folded))
    return np.fmax(bulk, tail)


def _autocov(x: np.ndarray) -> np.ndarray:
    """Biased autocovariance along the draw axis via FFT (ArviZ convention)."""
    from scipy.fft import next_fast_len

    n = x.shape[1]
    m = next_fast_len(2 * n)
    centered = x - x.mean(axis=1, keepdims=True)
    f = np.fft.rfft(centered, n=m, axis=1)
    f *= np.conjugate(f)
    cov = np.fft.irfft(f, n=m, axis=1)[:, :n]
    return cov / n


def ess_basic(x: np.ndarray) -> np.ndarray:
    """Multi-chain ESS with Geyer's initial positive + monotone sequence.

    A vectorized port of ArviZ 0.x ``_ess`` (Vehtari et al. 2021, eq. 3.10-3.13).
    """
    x = np.asarray(x, dtype=float)
    c, n, k = x.shape
    out = np.full(k, np.nan)
    if n < 4:
        return out
    flat = x.reshape(c * n, k)
    finite = np.isfinite(flat).all(axis=0)
    const = finite & ((flat.max(axis=0) - flat.min(axis=0)) < np.finfo(float).resolution)
    out[const] = c * n
    live = finite & ~const
    if not live.any():
        return out
    y = x[:, :, live]
    kk = y.shape[2]

    acov = _autocov(y)  # (C, N, K)
    chain_mean = y.mean(axis=1)
    mean_var = acov[:, 0, :].mean(axis=0) * n / (n - 1.0)
    var_plus = mean_var * (n - 1.0) / n
    if c > 1:
        var_plus = var_plus + chain_mean.var(axis=0, ddof=1)
    macov = acov.mean(axis=0)  # (N, K)
    with np.errstate(divide="ignore", invalid="ignore"):
        rho = 1.0 - (mean_var - macov) / var_plus  # (N, K)

    rho_hat = np.zeros((n, kk))
    rho_hat[0] = 1.0
    rho_hat[1] = rho[1]
    even = np.ones(kk)
    odd = rho[1].copy()
    active = (even + odd) > 0.0
    t_end = np.ones(kk, dtype=int)
    t = 1
    while t < n - 3 and active.any():
        e, o = rho[t + 1], rho[t + 2]
        even = np.where(active, e, even)
        odd = np.where(active, o, odd)
        store = active & ((e + o) >= 0)
        rho_hat[t + 1] = np.where(store, e, rho_hat[t + 1])
        rho_hat[t + 2] = np.where(store, o, rho_hat[t + 2])
        t += 2
        t_end = np.where(active, t, t_end)
        active = active & ((e + o) > 0.0)
    max_t = t_end - 2
    cols = np.arange(kk)
    fix = even > 0
    idx = max_t + 1
    rho_hat[idx[fix], cols[fix]] = even[fix]

    # Geyer's initial monotone sequence (sequential along lags).
    t = 1
    top = int(max_t.max()) if kk else 0
    while t <= top - 2:
        go = t <= (max_t - 2)
        prev = rho_hat[t - 1] + rho_hat[t]
        cur = rho_hat[t + 1] + rho_hat[t + 2]
        shrink = go & (cur > prev)
        rho_hat[t + 1] = np.where(shrink, prev / 2.0, rho_hat[t + 1])
        rho_hat[t + 2] = np.where(shrink, prev / 2.0, rho_hat[t + 2])
        t += 2

    cs = np.vstack([np.zeros((1, kk)), np.cumsum(rho_hat, axis=0)])
    j = max_t + 1  # sum(rho_hat[:j])
    head = cs[j, cols]
    tail_idx = np.minimum(j, n - 1)
    extra = np.where(j < n, rho_hat[tail_idx, cols], 0.0)
    tau = -1.0 + 2.0 * head + extra
    size = c * n
    tau = np.maximum(tau, 1.0 / np.log10(size))
    ess = size / tau
    ess[np.isnan(rho_hat).any(axis=0)] = np.nan
    out[live] = ess
    return out


def ess_bulk(x: np.ndarray) -> np.ndarray:
    return ess_basic(z_scale(split_chains(x)))


def ess_mean(x: np.ndarray) -> np.ndarray:
    return ess_basic(split_chains(x))


def _quantiles(x: np.ndarray, prob: float) -> np.ndarray:
    k = x.shape[2]
    return np.quantile(x.reshape(-1, k), prob, axis=0)


def ess_quantile(x: np.ndarray, prob: float) -> np.ndarray:
    q = _quantiles(x, prob)
    return ess_basic(split_chains((x <= q).astype(float)))


def ess_tail(x: np.ndarray) -> np.ndarray:
    """Tail ESS: the smaller of the 5% and 95% quantile ESS."""
    return np.fmin(ess_quantile(x, 0.05), ess_quantile(x, 0.95))


def mcse_quantile(x: np.ndarray, prob: float) -> np.ndarray:
    """Monte Carlo standard error of a quantile (Vehtari et al. 2021, sec. 4.4)."""
    from scipy import stats

    k = x.shape[2]
    ess = ess_quantile(x, prob)
    probability = np.array([0.1586553, 0.8413447])[:, None]
    with np.errstate(invalid="ignore"):
        ppf = stats.beta.ppf(probability, ess * prob + 1, ess * (1 - prob) + 1)
    srt = np.sort(x.reshape(-1, k), axis=0)
    size = srt.shape[0]
    ppf_size = ppf * size - 1
    lo = np.floor(np.nan_to_num(np.maximum(ppf_size[0], 0), nan=0)).astype(int)
    hi = np.ceil(np.nan_to_num(np.minimum(ppf_size[1], size - 1), nan=size - 1)).astype(int)
    cols = np.arange(k)
    return (srt[hi, cols] - srt[lo, cols]) / 2.0


def _nested_basic(x: np.ndarray, ids: np.ndarray) -> np.ndarray:
    c, n, k = x.shape
    chain_mean = x.mean(axis=1)
    chain_var = x.var(axis=1, ddof=1) if n > 1 else np.zeros((c, k))
    sc_means, within = [], []
    for g in np.unique(ids):
        sel = ids == g
        m = int(sel.sum())
        sc_means.append(chain_mean[sel].mean(axis=0))
        b_tilde = chain_mean[sel].var(axis=0, ddof=1) if m > 1 else np.zeros(k)
        within.append(b_tilde + chain_var[sel].mean(axis=0))
    b_nu = np.var(np.stack(sc_means), axis=0, ddof=1)
    w_nu = np.mean(np.stack(within), axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.sqrt(1.0 + b_nu / w_nu)


def nested_rhat(x: np.ndarray, superchain_ids) -> np.ndarray:
    """Rank-normalized, split, folded nested R-hat (Margossian et al. 2024).

    ``superchain_ids[c]`` is chain c's superchain. Every superchain must hold
    the same number of chains, and chains in a superchain must have started at
    the same point for the diagnostic to mean anything.
    """
    ids = np.asarray(superchain_ids)
    if ids.shape[0] != x.shape[0]:
        raise ValueError("superchain_ids must have one entry per chain")
    counts = np.unique(ids, return_counts=True)[1]
    if len(counts) < 2 or len(set(counts.tolist())) != 1:
        raise ValueError("nested R-hat needs >= 2 superchains of equal size")
    xs = split_chains(x)
    ids2 = np.concatenate([ids, ids])
    bulk = _nested_basic(z_scale(xs), ids2)
    k = xs.shape[2]
    folded = np.abs(xs - np.median(xs.reshape(-1, k), axis=0))
    tail = _nested_basic(z_scale(folded), ids2)
    return np.fmax(bulk, tail)


def bfmi(energy: np.ndarray) -> np.ndarray:
    """E-BFMI per chain from a (chain, draw) energy array."""
    energy = np.atleast_2d(np.asarray(energy, dtype=float))
    num = np.square(np.diff(energy, axis=1)).mean(axis=1)
    den = np.var(energy, axis=1, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return num / den


# ---------------------------------------------------------------------------
# Draws: what the assessment runs on
# ---------------------------------------------------------------------------


@dataclass
class Draws:
    """Post-warmup draws flattened to (chain, draw, element) plus HMC stats."""

    values: np.ndarray
    labels: list[str]
    var_of: list[str]
    diverging: int | None = None
    energy: np.ndarray | None = None
    maxdepth_frac: float | None = None
    superchain_ids: list[int] | None = None

    @property
    def n_chains(self) -> int:
        return int(self.values.shape[0])

    @property
    def n_draws(self) -> int:
        return int(self.values.shape[1])


def element_labels(name: str, shape: tuple[int, ...]) -> list[str]:
    if not shape:
        return [name]
    return [
        f"{name}[{', '.join(str(i) for i in idx)}]"
        for idx in np.ndindex(*shape)
    ]


def draws_from_arrays(
    arrays: dict[str, np.ndarray],
    *,
    diverging=None,
    energy=None,
    maxdepth=None,
    superchain_ids=None,
) -> Draws:
    """Build Draws from ``{name: (chain, draw, *shape)}`` arrays.

    Chains of unequal length (a sampler aborted mid-run pads with NaN) are
    truncated to the shortest complete chain, and the stats with them.
    """
    if not arrays:
        raise ValueError("no variables to assess")
    blocks, labels, var_of = [], [], []
    n_min = None
    for name, arr in arrays.items():
        arr = np.asarray(arr, dtype=float)
        if arr.ndim < 2:
            raise ValueError(f"{name}: expected (chain, draw, ...) array")
        c, n = arr.shape[:2]
        flat = arr.reshape(c, n, -1)
        lengths = []
        for ch in range(c):
            bad = ~np.isfinite(flat[ch]).any(axis=1)
            lengths.append(int(np.argmax(bad)) if bad.any() else n)
        m = min(lengths)
        n_min = m if n_min is None else min(n_min, m)
        blocks.append(flat)
        labels += element_labels(name, arr.shape[2:])
        var_of += [name] * flat.shape[2]
    values = np.concatenate([b[:, :n_min] for b in blocks], axis=2)

    n_div = None
    if diverging is not None:
        div = np.asarray(diverging)[:, :n_min]
        n_div = int(np.nansum(div.astype(float)))
    en = None
    if energy is not None:
        en = np.asarray(energy, dtype=float)[:, :n_min]
    md = None
    if maxdepth is not None:
        mdv = np.asarray(maxdepth)[:, :n_min].astype(float)
        md = float(np.nanmean(mdv)) if mdv.size else None
    return Draws(
        values=values, labels=labels, var_of=var_of, diverging=n_div,
        energy=en, maxdepth_frac=md,
        superchain_ids=list(superchain_ids) if superchain_ids is not None else None,
    )


SUPERCHAIN_ATTR = "cloudposterior_superchains"
REPORT_ATTR = "cloudposterior_convergence"


def draws_from_idata(idata, var_names=None, superchain_ids=None) -> Draws:
    """Build Draws from an InferenceData / DataTree's posterior + sample_stats."""
    from cloudposterior._idata import get_group

    post = get_group(idata, "posterior")
    if post is None:
        raise ValueError("idata has no posterior group")
    names = list(var_names) if var_names else list(post.data_vars)
    arrays = {}
    for name in names:
        if name not in post.data_vars:
            continue
        da = post[name]
        dims = list(da.dims)
        if dims[:2] != ["chain", "draw"]:
            da = da.transpose("chain", "draw", ...)
        if not np.issubdtype(da.dtype, np.number) and da.dtype != bool:
            continue
        arrays[name] = np.asarray(da.values)
    if superchain_ids is None:
        raw = getattr(post, "attrs", {}).get(SUPERCHAIN_ATTR)
        if raw is not None:
            superchain_ids = json.loads(raw) if isinstance(raw, str) else list(np.asarray(raw).tolist())

    stats = get_group(idata, "sample_stats")
    div = energy = maxdepth = None
    if stats is not None:
        svars = stats.data_vars
        if "diverging" in svars:
            div = np.asarray(stats["diverging"].values)
        if "energy" in svars:
            energy = np.asarray(stats["energy"].values)
        for key in ("maxdepth_reached", "reached_max_treedepth"):
            if key in svars:
                maxdepth = np.asarray(stats[key].values)
                break
    return draws_from_arrays(
        arrays, diverging=div, energy=energy, maxdepth=maxdepth,
        superchain_ids=superchain_ids,
    )


# ---------------------------------------------------------------------------
# Assessment
# ---------------------------------------------------------------------------


def _worst(labels, values, *, largest: bool, n: int = 3) -> list[str]:
    vals = np.where(np.isfinite(values), values, np.inf if largest else -np.inf)
    order = np.argsort(-vals if largest else vals)
    return [labels[i] for i in order[:n]]


def _rhat_values(x: np.ndarray, superchain_ids) -> np.ndarray:
    if superchain_ids is not None:
        return nested_rhat(x, superchain_ids)
    return rhat_rank(x)


def _rhat_and_bulk(x: np.ndarray, superchain_ids) -> tuple[np.ndarray, np.ndarray]:
    """R-hat (rank, or nested rank) and bulk ESS sharing one rank-normalization.

    Ranking is the dominant cost on big models, and R-hat's bulk half and bulk
    ESS rank-normalize the very same split draws.
    """
    xs = split_chains(x)
    zb = z_scale(xs)
    k = xs.shape[2]
    zf = z_scale(np.abs(xs - np.median(xs.reshape(-1, k), axis=0)))
    if x.shape[0] < 2:
        rh = np.full(k, np.nan)
    elif superchain_ids is not None:
        ids = np.asarray(superchain_ids)
        counts = np.unique(ids, return_counts=True)[1]
        if ids.shape[0] != x.shape[0] or len(counts) < 2 or len(set(counts.tolist())) != 1:
            raise ValueError("nested R-hat needs >= 2 superchains of equal size")
        ids2 = np.concatenate([ids, ids])
        rh = np.fmax(_nested_basic(zb, ids2), _nested_basic(zf, ids2))
    else:
        rh = np.fmax(_rhat_basic(zb), _rhat_basic(zf))
    return rh, ess_basic(zb)


def assess(draws: Draws, target: dict) -> dict:
    """Evaluate every check in ``target`` on ``draws``. Stateless.

    Returns a JSON/msgpack-safe dict: per-check value/threshold/ok/worst
    elements, ``met`` (all gating checks pass), and per-variable worst values.
    """
    x_all = draws.values
    c, n, _ = x_all.shape
    sc = draws.superchain_ids
    n_sc = len(set(sc)) if sc is not None else None
    result: dict = {"draws": n, "chains": c, "checks": {}, "params": {}, "met": False}
    if n < 4 or c < 1:
        result["insufficient"] = True
        return result

    flat = x_all.reshape(c * n, -1)
    finite = np.isfinite(flat).all(axis=0)
    with np.errstate(invalid="ignore"):
        var = np.var(np.where(np.isfinite(flat), flat, 0.0), axis=0, ddof=1)
    live = ~finite | (var > 0)  # non-finite elements stay in and fail
    x = x_all[:, :, live]
    labels = [lbl for lbl, keep in zip(draws.labels, live) if keep]
    var_of = [v for v, keep in zip(draws.var_of, live) if keep]
    checks = result["checks"]

    if x.shape[2]:
        thr = rhat_threshold(target, c, n_sc)
        rh, eb = _rhat_and_bulk(x, sc)
        rh = np.where(np.isfinite(rh), rh, np.inf) if c > 1 else rh
        if c > 1:
            checks["r_hat"] = {
                "value": float(np.max(rh)), "threshold": round(thr, 4),
                "ok": bool(np.max(rh) <= thr), "worst": _worst(labels, rh, largest=True),
                "nested": sc is not None,
            }
        eb = np.nan_to_num(eb, nan=0.0)
        et = np.nan_to_num(ess_tail(x), nan=0.0)
        ess_t = target.get("ess")
        if ess_t:
            checks["ess_bulk"] = {
                "value": float(np.min(eb)), "threshold": float(ess_t),
                "ok": bool(np.min(eb) >= ess_t), "worst": _worst(labels, eb, largest=False),
            }
            checks["ess_tail"] = {
                "value": float(np.min(et)), "threshold": float(ess_t),
                "ok": bool(np.min(et) >= ess_t), "worst": _worst(labels, et, largest=False),
            }
        mcse_t = target.get("mcse")
        if mcse_t:
            em = np.nan_to_num(ess_mean(x), nan=0.0)
            with np.errstate(divide="ignore"):
                rel = 1.0 / np.sqrt(em)
            if target.get("quantiles"):
                sd = x.reshape(-1, x.shape[2]).std(axis=0, ddof=1)
                with np.errstate(divide="ignore", invalid="ignore"):
                    q = np.fmax(mcse_quantile(x, 0.05), mcse_quantile(x, 0.95)) / sd
                rel = np.fmax(rel, np.nan_to_num(q, nan=np.inf))
            checks["mcse"] = {
                "value": float(np.max(rel)), "threshold": float(mcse_t),
                "ok": bool(np.max(rel) <= mcse_t), "worst": _worst(labels, rel, largest=True),
            }

        # Per-variable worst values for the dashboard table.
        params: dict = {}
        for i, name in enumerate(var_of):
            p = params.setdefault(name, {"rhat": 0.0, "ess_bulk": math.inf, "ess_tail": math.inf})
            if c > 1 and math.isfinite(rh[i]):
                p["rhat"] = max(p["rhat"], float(rh[i]))
            elif c > 1:
                p["rhat"] = math.inf
            p["ess_bulk"] = min(p["ess_bulk"], float(eb[i]))
            p["ess_tail"] = min(p["ess_tail"], float(et[i]))
        result["params"] = {
            k: {
                "rhat": round(v["rhat"], 4) if math.isfinite(v["rhat"]) else 99.0,
                "ess_bulk": round(v["ess_bulk"]) if math.isfinite(v["ess_bulk"]) else 0,
                "ess_tail": round(v["ess_tail"]) if math.isfinite(v["ess_tail"]) else 0,
            }
            for k, v in params.items()
        }

    if draws.diverging is not None:
        tol = target.get("divergences")
        checks["divergences"] = {
            "value": int(draws.diverging), "threshold": tol,
            "ok": None if tol is None else bool(draws.diverging <= tol),
        }
    if draws.energy is not None and draws.energy.shape[1] > 2:
        b = bfmi(draws.energy)
        thr_b = target.get("bfmi")
        low = float(np.nanmin(b)) if np.isfinite(b).any() else float("nan")
        checks["bfmi"] = {
            "value": round(low, 4), "threshold": thr_b,
            "ok": None if thr_b is None or not math.isfinite(low) else bool(low >= thr_b),
            "worst": [f"chain {int(i)}" for i in np.argsort(np.nan_to_num(b, nan=np.inf))[:3]],
        }
    if draws.maxdepth_frac is not None:
        thr_t = target.get("treedepth")
        checks["treedepth"] = {
            "value": round(float(draws.maxdepth_frac), 4), "threshold": thr_t,
            "ok": None if thr_t is None else bool(draws.maxdepth_frac <= thr_t),
        }

    gating = [ch["ok"] for ch in checks.values() if ch.get("ok") is not None]
    result["met"] = bool(gating) and all(gating) and bool(x.shape[2] if x.ndim == 3 else 0)
    return result


def projected_draws(assessment: dict) -> int | None:
    """Draws per chain at which the failing sampling-fixable checks would pass,
    assuming ESS grows linearly and R-hat^2 - 1 shrinks as 1/draws."""
    n = assessment.get("draws", 0)
    if not n:
        return None
    checks = assessment.get("checks", {})
    for key in ("divergences", "bfmi"):
        ch = checks.get(key)
        if ch and ch.get("ok") is False:
            return None  # more draws cannot fix these
    factors = [1.0]
    for key in ("ess_bulk", "ess_tail"):
        ch = checks.get(key)
        if ch and ch.get("ok") is False:
            factors.append(ch["threshold"] / max(ch["value"], 1.0))
    ch = checks.get("mcse")
    if ch and ch.get("ok") is False and math.isfinite(ch["value"]):
        factors.append((ch["value"] / ch["threshold"]) ** 2)
    ch = checks.get("r_hat")
    if ch and ch.get("ok") is False and math.isfinite(ch["value"]):
        thr = ch["threshold"]
        factors.append((ch["value"] ** 2 - 1.0) / max(thr ** 2 - 1.0, 1e-9))
    return int(math.ceil(n * max(factors)))


def diagnose(draws: Draws, assessment: dict) -> tuple[str | None, str | None]:
    """Name the most likely reason the target is not met, with a hint.

    One R-hat cannot tell a warmup that was too short from a sampling phase
    that was too short (Margossian and Gelman 2024). Comparing R-hat on the
    late half of the draws with R-hat on all of them separates them: early
    drift that has died out means warmup was too short; disagreement that
    persists late means chains are stuck apart, which more draws won't fix.
    """
    checks = assessment.get("checks", {})

    def failed(key):
        ch = checks.get(key)
        return bool(ch) and ch.get("ok") is False

    if failed("divergences"):
        n_div = checks["divergences"]["value"]
        return "divergences", (
            f"{n_div} divergent transition(s) after warmup: the sampler cannot "
            "reach part of the posterior, so estimates are biased. Raise "
            "target_accept (e.g. 0.95) or reparameterize (non-centered "
            "hierarchical priors)."
        )
    if failed("bfmi"):
        return "low_bfmi", (
            f"E-BFMI {checks['bfmi']['value']:.2f} is below "
            f"{checks['bfmi']['threshold']}: the energy distribution is poorly "
            "explored (heavy tails or a funnel). Reparameterize; longer runs won't help."
        )
    if failed("r_hat"):
        worst = ", ".join(checks["r_hat"].get("worst", [])[:3])
        regime = _rhat_regime(draws, checks["r_hat"].get("worst", []))
        if regime == "early_drift":
            return "early_drift", (
                f"chains were still drifting early in sampling ({worst}): warmup "
                "was too short. Increase tune."
            )
        if regime == "slow_mixing":
            proj = projected_draws(assessment)
            more = f" About {proj} draws per chain would meet the target." if proj else ""
            return "slow_mixing", (
                f"chains mix slowly ({worst}): R-hat is high but consistent with "
                "stationary chains that need more draws." + more
            )
        return "chains_disagree", (
            f"chains disagree even late in sampling ({worst}): multimodality, a "
            "stuck chain, or a funnel. More draws won't fix it: reparameterize, "
            "add prior information, or try init='prior' to probe for modes."
        )
    if failed("treedepth"):
        return "treedepth", (
            f"max tree depth hit in {checks['treedepth']['value']:.0%} of draws: "
            "raise maxdepth or reparameterize."
        )
    if failed("ess_bulk") or failed("ess_tail") or failed("mcse"):
        proj = projected_draws(assessment)
        more = f" About {proj} draws per chain would meet it." if proj else ""
        return "slow_mixing", "chains agree but mix slowly." + more
    return None, None


def _rhat_regime(draws: Draws, worst: list[str]) -> str:
    """Classify an R-hat failure from how it changes over the late half.

    For the worst elements, compare R^2 - 1 on the late half of the draws with
    R^2 - 1 on all of them (ratio r). Chains that are stationary but mix slowly
    roughly double it on half the draws (r ~ 2); chains stuck apart keep it
    (r ~ 1); chains that drifted in from their starting points and then agreed
    show much less disagreement late (r << 1). Calibrated on AR(1) and offset
    simulations: early drift r < 0.5, stuck 0.9-1.1, slow mixing mostly > 1.4.
    """
    x = draws.values
    n = x.shape[1]
    if n < 16 or x.shape[0] < 2:
        return "chains_disagree"
    idx = [draws.labels.index(w) for w in worst if w in draws.labels]
    if not idx:
        return "chains_disagree"
    full = _rhat_values(x[:, :, idx], draws.superchain_ids)
    late = _rhat_values(x[:, n // 2:, idx], draws.superchain_ids)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.nanmax(late ** 2 - 1) / np.nanmax(full ** 2 - 1)
    if not np.isfinite(ratio):
        return "chains_disagree"
    if ratio < 0.5:
        return "early_drift"
    if ratio >= 1.4:
        return "slow_mixing"
    return "chains_disagree"


# ---------------------------------------------------------------------------
# Controller: when to check, when to stop
# ---------------------------------------------------------------------------


@dataclass
class Controller:
    """Stateful stop/continue decisions at geometric checkpoints.

    ``step(draws)`` assesses, records history and returns the assessment with
    ``decision`` in {"continue", "met", "futile"} plus ``projected_draws``,
    ``cause`` and ``hint``.
    """

    target: dict
    n_chains: int
    draws_cap: int
    stop: bool = True  # False: assess for display only, never stop
    history: list = field(default_factory=list)
    consecutive: int = 0
    next_at: int = 0

    def __post_init__(self):
        per_chain = math.ceil(effective_ess_target(self.target) / max(1, self.n_chains))
        self.first_at = max(50, per_chain)
        if self.draws_cap:
            self.first_at = min(self.first_at, max(4, self.draws_cap))
        self.next_at = self.first_at

    def due(self, n_draws: int) -> bool:
        return n_draws >= self.next_at

    def step(self, draws: Draws) -> dict:
        a = assess(draws, self.target)
        n = draws.n_draws
        self.next_at = max(n + 1, int(math.ceil(n * float(self.target["growth"]))))
        self.consecutive = self.consecutive + 1 if a["met"] else 0
        checks = a.get("checks", {})
        ess_vals = [checks[k]["value"] for k in ("ess_bulk", "ess_tail") if k in checks]
        self.history.append({
            "n": n,
            "rhat": checks.get("r_hat", {}).get("value"),
            "rhat_ok": checks.get("r_hat", {}).get("ok"),
            "ess": min(ess_vals) if ess_vals else None,
            "ess_ok": all(checks[k]["ok"] for k in ("ess_bulk", "ess_tail") if k in checks),
            "bfmi_ok": checks.get("bfmi", {}).get("ok"),
            "met": a["met"],
        })

        decision = "continue"
        cause = hint = None
        if a["met"] and self.consecutive >= int(self.target["patience"]):
            decision = "met"
        elif not a["met"] and self.target.get("futility", True):
            futile = self._futile(a)
            if futile:
                decision = "futile"
                cause, hint = diagnose(draws, a)
                if futile == "rhat_stalled" and cause == "slow_mixing":
                    worst = ", ".join(a["checks"]["r_hat"].get("worst", [])[:3])
                    cause = "chains_disagree"
                    hint = (
                        f"R-hat stopped improving as draws grew ({worst}): chains "
                        "are stuck apart (multimodality, a stuck chain, or a "
                        "funnel). Reparameterize or add prior information."
                    )
                if futile == "ess_stalled":
                    cause = "ess_stalled"
                    hint = (
                        "ESS is not growing with the number of draws, so running "
                        "longer won't reach the target. Reparameterize, or try "
                        "nutpie's adaptation='low_rank' for correlated posteriors."
                    )
        if decision == "continue" and not a["met"]:
            cause, hint = diagnose(draws, a)
        a["decision"] = decision if self.stop else "continue"
        a["consecutive"] = self.consecutive
        a["projected_draws"] = projected_draws(a)
        a["cause"] = cause
        a["hint"] = hint
        a["next_check"] = self.next_at
        return a

    def _futile(self, a: dict) -> str | None:
        checks = a.get("checks", {})
        div = checks.get("divergences")
        if div and div.get("ok") is False:
            return "divergences"  # divergences only accumulate
        patience = int(self.target["patience"])
        recent = self.history[-patience:]
        if len(recent) >= patience and all(h["bfmi_ok"] is False for h in recent):
            return "low_bfmi"
        # Stall detection over a window that spans at least 2x the draws.
        n_now = self.history[-1]["n"]
        start = None
        for i in range(len(self.history) - 1, -1, -1):
            if self.history[i]["n"] * 2 <= n_now:
                start = i
                break
        if start is None or len(self.history) - start < 3:
            return None
        window = self.history[start:]
        first, last = window[0], window[-1]
        if (
            all(h["rhat_ok"] is False for h in window)
            and first["rhat"] is not None and last["rhat"] is not None
            and math.isfinite(first["rhat"]) and math.isfinite(last["rhat"])
            and (last["rhat"] ** 2 - 1) > 0.8 * (first["rhat"] ** 2 - 1)
        ):
            return "rhat_stalled"
        if (
            not last["ess_ok"] and first["ess"] is not None and last["ess"] is not None
            and last["ess"] < 1.2 * max(first["ess"], 1.0)
        ):
            return "ess_stalled"
        return None


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------


def summarize(assessment: dict) -> str:
    """One-line human summary of an assessment."""
    checks = assessment.get("checks", {})
    parts = []
    rh = checks.get("r_hat")
    if rh:
        name = "nested R-hat" if rh.get("nested") else "R-hat"
        parts.append(f"max {name} {rh['value']:.3f} (<= {rh['threshold']:.3f})")
    eb, et = checks.get("ess_bulk"), checks.get("ess_tail")
    if eb and et:
        parts.append(f"min ESS {min(eb['value'], et['value']):.0f} (>= {eb['threshold']:.0f})")
    mc = checks.get("mcse")
    if mc:
        parts.append(f"max MCSE/sd {mc['value']:.3f} (<= {mc['threshold']:.3f})")
    dv = checks.get("divergences")
    if dv and dv["value"]:
        parts.append(f"{dv['value']} divergences")
    return ", ".join(parts)


def failing(assessment: dict) -> list[str]:
    return [k for k, ch in assessment.get("checks", {}).items() if ch.get("ok") is False]


def make_report(assessment: dict, *, status: str, target: dict | None,
                attempts: list | None = None) -> dict:
    """The verdict stored on the result (``posterior.attrs`` JSON)."""
    report = {
        "status": status,  # met | futile | cap | complete | stopped
        "met": bool(assessment.get("met")),
        "draws": assessment.get("draws"),
        "chains": assessment.get("chains"),
        "summary": summarize(assessment),
        "failing": failing(assessment),
        "cause": assessment.get("cause"),
        "hint": assessment.get("hint"),
        "projected_draws": assessment.get("projected_draws"),
        "checks": assessment.get("checks", {}),
        "target": target,
    }
    if attempts:
        report["attempts"] = attempts
    return report


def verdict_message(report: dict) -> str:
    """Short message for the sampling phase / notifications."""
    status = report.get("status")
    summ = report.get("summary") or ""
    n = report.get("draws")
    if status == "met":
        return f"targets met at {n} draws/chain ({summ})"
    if status == "futile":
        return f"target unreachable, stopped at {n} draws/chain: {report.get('hint') or summ}"
    if status == "cap":
        if report.get("met"):
            return f"draw cap reached, targets met ({summ})"
        return f"draw cap reached, targets NOT met ({summ}). {report.get('hint') or ''}".strip()
    if status == "stopped":
        return f"stopped by user at {n} draws/chain ({summ})"
    if report.get("met") is False and report.get("failing"):
        return f"sampling complete; diagnostics flag problems ({summ}). {report.get('hint') or ''}".strip()
    return f"sampling complete ({summ})" if summ else "sampling complete"


def attach_report(idata, report: dict) -> None:
    """Store the report on the posterior group as a JSON attr (NetCDF-safe)."""
    from cloudposterior._idata import get_group

    post = get_group(idata, "posterior")
    if post is None:
        return
    try:
        post.attrs[REPORT_ATTR] = json.dumps(_jsonable(report))
    except Exception:
        pass


def read_report(idata) -> dict | None:
    """The convergence report a cloudposterior run attached, if any."""
    from cloudposterior._idata import get_group

    post = get_group(idata, "posterior")
    raw = getattr(post, "attrs", {}).get(REPORT_ATTR) if post is not None else None
    if raw is None:
        return None
    try:
        return json.loads(raw) if isinstance(raw, str) else dict(raw)
    except Exception:
        return None


def _jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        f = float(obj)
        return f if math.isfinite(f) else None
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def check_convergence(idata, until=True, *, var_names=None) -> dict:
    """Assess any InferenceData against a convergence target.

    Works on local or remote results. Returns the same report a remote
    ``until=`` run stores on its result: ``met``, a one-line ``summary``,
    ``failing`` checks, the likely ``cause`` with a ``hint``, and
    ``projected_draws`` (draws per chain at which the target would be met if
    the problem is just slow mixing).
    """
    target = normalize_target(until) or normalize_target(True)
    names = var_names or target.get("var_names")
    draws = draws_from_idata(idata, var_names=names)
    a = assess(draws, target)
    a["cause"], a["hint"] = diagnose(draws, a) if not a["met"] else (None, None)
    a["projected_draws"] = projected_draws(a)
    return make_report(a, status="met" if a["met"] else "complete", target=target)
