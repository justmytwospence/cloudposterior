"""Mirror examples/caching.ipynb -- exercises cp.cloud(model) end-to-end
without Modal.

These run real pm.sample() against tiny hierarchical models, so they take a
few seconds each but cost nothing.
"""

import time

import numpy as np
import pymc as pm
import pytest

import cloudposterior as cp
from cloudposterior.cache import _default_memory_cache


def _make_radon_intercepts(seed: int = 0):
    """A small hierarchical radon-style model. Mirrors caching.ipynb's
    radon_intercepts but with synthetic data sized for fast tests."""
    rng = np.random.default_rng(seed)
    n_obs = 30
    n_county = 5
    county_idx = rng.integers(0, n_county, n_obs)
    floor = rng.integers(0, 2, n_obs).astype(float)
    log_radon = rng.normal(0, 1, n_obs)
    coords = {"county": [f"c{i}" for i in range(n_county)]}
    with pm.Model(name="radon_intercepts", coords=coords) as model:
        mu_a = pm.Normal("mu_a", 0, 5)
        sigma_a = pm.HalfNormal("sigma_a", 2)
        a_raw = pm.Normal("a_raw", 0, 1, dims="county")
        a = pm.Deterministic("a", mu_a + sigma_a * a_raw, dims="county")
        b_floor = pm.Normal("b_floor", 0, 5)
        mu = a[county_idx] + b_floor * floor
        sigma_y = pm.HalfNormal("sigma_y", 2)
        pm.Normal("obs", mu=mu, sigma=sigma_y, observed=log_radon)
    return model


def _make_radon_slopes():
    """A structurally different model -- mirrors caching.ipynb's radon_slopes
    cell, used to verify distinct models get distinct cache entries."""
    rng = np.random.default_rng(1)
    n_obs = 30
    n_county = 5
    county_idx = rng.integers(0, n_county, n_obs)
    floor = rng.integers(0, 2, n_obs).astype(float)
    log_radon = rng.normal(0, 1, n_obs)
    coords = {"county": [f"c{i}" for i in range(n_county)]}
    with pm.Model(name="radon_slopes", coords=coords) as model:
        mu_a = pm.Normal("mu_a", 0, 5)
        sigma_a = pm.HalfNormal("sigma_a", 2)
        a_raw = pm.Normal("a_raw", 0, 1, dims="county")
        a = pm.Deterministic("a", mu_a + sigma_a * a_raw, dims="county")
        mu_b = pm.Normal("mu_b", 0, 5)
        sigma_b = pm.HalfNormal("sigma_b", 2)
        b_raw = pm.Normal("b_raw", 0, 1, dims="county")
        b = pm.Deterministic("b", mu_b + sigma_b * b_raw, dims="county")
        mu = a[county_idx] + b[county_idx] * floor
        sigma_y = pm.HalfNormal("sigma_y", 2)
        pm.Normal("obs", mu=mu, sigma=sigma_y, observed=log_radon)
    return model


@pytest.fixture(autouse=True)
def _clear_default_memory_cache():
    """Each test starts with a clean in-memory cache so prior tests don't
    accidentally serve cached idata."""
    _default_memory_cache._store.clear()
    yield
    _default_memory_cache._store.clear()


def _mu_a(idata):
    """Look up the mu_a posterior values, accounting for PyMC's model-name
    prefix (``pm.Model(name="radon_intercepts")`` produces variable names
    like ``radon_intercepts::mu_a``)."""
    for name in idata.posterior.data_vars:
        if name.endswith("mu_a"):
            return idata.posterior[name].values
    raise AssertionError(f"no mu_a variable in posterior; got {list(idata.posterior.data_vars)}")


# -- Mirror of cells 2 + 3 in caching.ipynb (memory cache miss, then hit) ----

def test_memory_cache_hit_on_repeat_call():
    """Second cp.cloud(model) block with identical kwargs returns the cached
    result instantly without sampling."""
    model = _make_radon_intercepts()
    with cp.cloud(model):
        idata1 = pm.sample(draws=10, tune=10, chains=1, progressbar=False)

    t0 = time.time()
    with cp.cloud(model):
        idata2 = pm.sample(draws=10, tune=10, chains=1, progressbar=False)
    elapsed = time.time() - t0

    assert elapsed < 1.0, f"cache hit took {elapsed:.2f}s -- expected sub-second"
    np.testing.assert_array_equal(_mu_a(idata1), _mu_a(idata2))


# -- Mirror of cells 4 + 5 in caching.ipynb (disk cache miss, then hit across
#    a fresh in-memory cache to prove persistence) -----------------------------

def test_disk_cache_persists_across_memory_cache_clear(tmp_path):
    """Disk cache survives a wiped in-memory cache (i.e. a kernel restart)."""
    model = _make_radon_intercepts()
    with cp.cloud(model, cache=tmp_path):
        idata1 = pm.sample(draws=10, tune=10, chains=1, progressbar=False)

    nc_files = list(tmp_path.rglob("*.nc"))
    assert nc_files, "expected a NetCDF file written to disk cache"

    # Simulate a fresh kernel by wiping the in-memory cache.
    _default_memory_cache._store.clear()

    t0 = time.time()
    with cp.cloud(model, cache=tmp_path):
        idata2 = pm.sample(draws=10, tune=10, chains=1, progressbar=False)
    elapsed = time.time() - t0

    assert elapsed < 2.0, f"disk cache hit took {elapsed:.2f}s -- expected fast"
    np.testing.assert_array_equal(_mu_a(idata1), _mu_a(idata2))


def test_disk_cache_layout_uses_model_name(tmp_path):
    """Cached files live under {base_dir}/{model_slug}/..., matching the
    README's claimed disk layout."""
    model = _make_radon_intercepts()
    with cp.cloud(model, cache=tmp_path):
        pm.sample(draws=10, tune=10, chains=1, progressbar=False)

    nc_files = list(tmp_path.rglob("*.nc"))
    assert len(nc_files) == 1
    assert nc_files[0].parent.name == "radon_intercepts"
    assert "draws10_tune10_chains1" in nc_files[0].name


# -- Mirror of cells 6 + 7 in caching.ipynb (different model -> different file) -

def test_distinct_models_get_distinct_cache_files(tmp_path):
    """Switching to a structurally different model writes a separate cache
    entry rather than colliding with the first model's."""
    intercepts = _make_radon_intercepts()
    slopes = _make_radon_slopes()

    with cp.cloud(intercepts, cache=tmp_path):
        pm.sample(draws=10, tune=10, chains=1, progressbar=False)
    with cp.cloud(slopes, cache=tmp_path):
        pm.sample(draws=10, tune=10, chains=1, progressbar=False)

    parents = {f.parent.name for f in tmp_path.rglob("*.nc")}
    assert "radon_intercepts" in parents
    assert "radon_slopes" in parents


# -- Cache=False truly disables caching --------------------------------------

def test_cache_false_does_not_write_anything(tmp_path):
    """cache=False skips both lookup and storage."""
    model = _make_radon_intercepts()
    with cp.cloud(model, cache=False):
        pm.sample(draws=10, tune=10, chains=1, progressbar=False)

    # Default disk cache directory should not have been created.
    assert not (tmp_path / ".cloudposterior").exists()
    assert _default_memory_cache._store == {}


# -- Cache key sensitivity ---------------------------------------------------

def test_different_random_seed_misses_cache():
    """Same model + same draws/tune/chains but different random_seed must
    produce a distinct cache entry (the README claims this guarantee).

    Verified by sampling values rather than timing: two different seeds must
    produce two different posterior traces; if the cache had served the first
    result back, the values would be identical.
    """
    model = _make_radon_intercepts()
    with cp.cloud(model):
        idata1 = pm.sample(draws=10, tune=10, chains=1, random_seed=1, progressbar=False)
    with cp.cloud(model):
        idata2 = pm.sample(draws=10, tune=10, chains=1, random_seed=2, progressbar=False)

    assert not np.array_equal(_mu_a(idata1), _mu_a(idata2)), (
        "different random_seed values produced identical traces -- the second "
        "call must have hit the cache, contradicting cache-key sensitivity to seed"
    )
