"""Test result caching."""

import shutil
import numpy as np
import pymc as pm
import pytest

from cloudposterior.cache import (
    MemoryCache,
    DiskCache,
    resolve_cache,
)
from cloudposterior.naming import cache_key, model_slug
from cloudposterior.serialize import serialize_model


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
    """Same model + kwargs should produce the same key."""
    model = _make_model()
    mb = serialize_model(model)
    kwargs = {"draws": 100, "tune": 50, "chains": 2}

    key1 = cache_key(mb, kwargs)
    key2 = cache_key(mb, kwargs)
    assert key1 == key2
    assert len(key1) == 64  # SHA-256 hex


def test_cache_key_changes_with_kwargs():
    """Different kwargs should produce a different key."""
    model = _make_model()
    mb = serialize_model(model)

    key1 = cache_key(mb, {"draws": 100})
    key2 = cache_key(mb, {"draws": 200})
    assert key1 != key2


def test_memory_cache_roundtrip():
    """MemoryCache stores and retrieves objects."""
    cache = MemoryCache()
    assert cache.load("nonexistent") is None

    cache.save("key1", {"data": 42})
    assert cache.load("key1") == {"data": 42}
    assert cache.load("key2") is None


def test_disk_cache_roundtrip(tmp_path):
    """DiskCache stores and retrieves InferenceData with structured paths."""
    model = _make_model("test_model")
    with model:
        idata = pm.sample(draws=10, tune=10, chains=1, progressbar=False)

    sample_kwargs = {"draws": 10, "tune": 10, "chains": 1}
    key = "a1b2c3d4e5f6g7h8"  # simulate a real cache key hash

    cache = DiskCache(base_dir=tmp_path, model=model)
    assert cache.load(key, sample_kwargs=sample_kwargs) is None

    cache.save(key, idata, sample_kwargs=sample_kwargs)
    loaded = cache.load(key, sample_kwargs=sample_kwargs)
    assert loaded is not None
    assert "posterior" in loaded.groups()
    rv_names = list(loaded.posterior.data_vars)
    assert any("mu" in name for name in rv_names)

    # Check directory structure: model_name/params-hash.nc
    nc_files = list(tmp_path.rglob("*.nc"))
    assert len(nc_files) == 1
    path = nc_files[0]
    assert "test_model" in str(path)
    assert "draws10_tune10_chains1" in path.name
    assert key[:8] in path.name  # hash suffix for uniqueness


def test_model_slug_named():
    model = _make_model("Eight Schools")
    assert model_slug(model) == "eight_schools"


def test_model_slug_unnamed():
    model = _make_model()
    slug = model_slug(model)
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


def test_resolve_cache_custom_backend():
    """A duck-typed backend with load/save methods passes through unchanged."""
    backend = MemoryCache()
    assert resolve_cache(backend) is backend


@pytest.mark.parametrize("bad", [42, 3.14, object(), [], ()])
def test_resolve_cache_rejects_unknown_types(bad):
    """Unknown types raise TypeError instead of silently returning the default."""
    with pytest.raises(TypeError, match="cache must be"):
        resolve_cache(bad)
