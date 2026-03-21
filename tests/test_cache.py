"""Test result caching."""

import shutil
import numpy as np
import pymc as pm

from cloudposterior.cache import (
    MemoryCache,
    DiskCache,
    compute_cache_key,
    resolve_cache,
    _model_slug,
)
from cloudposterior.serialize import serialize_model, serialize_observed_data


def _make_model(name=""):
    y = np.array([28, 8, -3, 7, -1, 1, 18, 12], dtype=np.float64)
    sigma = np.array([15, 10, 16, 11, 9, 11, 10, 18], dtype=np.float64)
    with pm.Model(name=name) as model:
        mu = pm.Normal("mu", 0, 5)
        tau = pm.HalfCauchy("tau", 5)
        theta = pm.Normal("theta", mu=mu, sigma=tau, shape=8)
        pm.Normal("obs", mu=theta, sigma=sigma, observed=y)
    return model


def test_cache_key_deterministic():
    """Same model + data + kwargs should produce the same key."""
    model = _make_model()
    mb = serialize_model(model)
    db = serialize_observed_data(model)
    kwargs = {"draws": 100, "tune": 50, "chains": 2}

    key1 = compute_cache_key(mb, db, kwargs)
    key2 = compute_cache_key(mb, db, kwargs)
    assert key1 == key2
    assert len(key1) == 64  # SHA-256 hex


def test_cache_key_changes_with_kwargs():
    """Different kwargs should produce a different key."""
    model = _make_model()
    mb = serialize_model(model)
    db = serialize_observed_data(model)

    key1 = compute_cache_key(mb, db, {"draws": 100})
    key2 = compute_cache_key(mb, db, {"draws": 200})
    assert key1 != key2


def test_memory_cache_roundtrip():
    """MemoryCache stores and retrieves objects."""
    cache = MemoryCache()
    assert cache.load("nonexistent") is None

    cache.save("key1", {"data": 42})
    assert cache.load("key1") == {"data": 42}
    assert cache.load("key2") is None


def test_disk_cache_roundtrip(tmp_path):
    """DiskCache stores and retrieves InferenceData."""
    model = _make_model("test_model")
    with model:
        idata = pm.sample(draws=10, tune=10, chains=1, progressbar=False)

    cache = DiskCache(base_dir=tmp_path, model=model)
    assert cache.load("testkey123456") is None

    cache.save("testkey123456", idata)
    loaded = cache.load("testkey123456")
    assert loaded is not None
    assert "posterior" in loaded.groups()
    # Named models prefix RVs with "model_name::"
    rv_names = list(loaded.posterior.data_vars)
    assert any("mu" in name for name in rv_names)

    # Check directory structure
    nc_files = list(tmp_path.rglob("*.nc"))
    assert len(nc_files) == 1
    assert "test_model" in str(nc_files[0].parent)


def test_model_slug_named():
    model = _make_model("Eight Schools")
    assert _model_slug(model) == "eight_schools"


def test_model_slug_unnamed():
    model = _make_model()
    slug = _model_slug(model)
    assert "mu" in slug
    assert "tau" in slug


def test_resolve_cache_true():
    """cache=True returns the default MemoryCache."""
    backend = resolve_cache(True)
    assert isinstance(backend, MemoryCache)


def test_resolve_cache_false():
    """cache=False returns None."""
    assert resolve_cache(False) is None


def test_resolve_cache_disk_string():
    """cache='disk' returns a DiskCache with default path."""
    backend = resolve_cache("disk")
    assert isinstance(backend, DiskCache)


def test_resolve_cache_custom_path(tmp_path):
    """cache=Path returns a DiskCache at that path."""
    backend = resolve_cache(tmp_path)
    assert isinstance(backend, DiskCache)
