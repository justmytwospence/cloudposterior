"""Test result caching."""

import numpy as np
import pymc as pm
import pytest

from cloudposterior.cache import (
    DiskCache,
    MemoryCache,
    resolve_cache,
)


def _make_model(name=""):
    y = np.array([28, 8, -3, 7, -1, 1, 18, 12], dtype=np.float64)
    sigma = np.array([15, 10, 16, 11, 9, 11, 10, 18], dtype=np.float64)
    with pm.Model(name=name) as model:
        mu = pm.Normal("mu", 0, 5)
        tau = pm.HalfCauchy("tau", 5)
        theta = pm.Normal("theta", mu=mu, sigma=tau, shape=8)
        pm.Normal("obs", mu=theta, sigma=sigma, observed=y)
    return model


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
    from cloudposterior._idata import group_names

    assert "posterior" in group_names(loaded)
    rv_names = list(loaded.posterior.data_vars)
    assert any("mu" in name for name in rv_names)

    # Check directory structure: model_name/params-hash.nc
    nc_files = list(tmp_path.rglob("*.nc"))
    assert len(nc_files) == 1
    path = nc_files[0]
    assert "test_model" in str(path)
    assert "draws10_tune10_chains1" in path.name
    assert key[:8] in path.name  # hash suffix for uniqueness


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


def test_resolve_cache_memory_alias():
    """cache="memory" is an alias for the default in-memory cache, not a disk
    cache rooted in a directory literally named "memory"."""
    from cloudposterior.cache import get_default_cache, resolve_cache

    assert resolve_cache("memory") is get_default_cache()


def test_resolve_cache_custom_dir_string(tmp_path):
    from cloudposterior.cache import DiskCache, resolve_cache

    backend = resolve_cache(str(tmp_path / "custom"))
    assert isinstance(backend, DiskCache)
    assert backend._base == tmp_path / "custom"


def test_disk_cache_load_miss_creates_no_directories(tmp_path):
    """A cache probe must not litter empty model directories on disk."""
    from cloudposterior.cache import DiskCache

    cache = DiskCache(base_dir=tmp_path / "cp-cache")
    assert cache.load("a" * 64, sample_kwargs={"draws": 10}) is None
    assert not (tmp_path / "cp-cache").exists()


# -- DiskCache hardening -----------------------------------------------------

def _disk(tmp_path):
    from cloudposterior.cache import DiskCache

    return DiskCache(base_dir=tmp_path)


def test_disk_cache_rejects_traversal_in_params(tmp_path):
    """sample_kwargs values become a path component; an unsanitized one used
    to escape the cache root."""
    cache = _disk(tmp_path)
    path = cache._path("a" * 64, {"nuts_sampler": "../../../etc/evil"})
    assert path.resolve().is_relative_to(tmp_path.resolve())
    assert ".." not in str(path)


def test_disk_cache_uses_a_16_char_key_prefix(tmp_path):
    cache = _disk(tmp_path)
    key = "b" * 64
    assert cache._path(key, {"draws": 10}).name.endswith(f"-{'b' * 16}.nc")


def test_disk_cache_verifies_the_full_key(tmp_path):
    """The filename holds only a prefix; a collision must be a miss, not the
    wrong posterior."""
    import arviz as az

    from cloudposterior._idata import add_group

    cache = _disk(tmp_path)
    idata = az.InferenceData()
    add_group(idata, "posterior", az.dict_to_dataset({"x": np.zeros((2, 5))}))

    key_a = "c" * 64
    cache.save(key_a, idata, sample_kwargs={"draws": 10})
    assert cache.load(key_a, sample_kwargs={"draws": 10}) is not None

    # Same 16-char prefix, different full key -> must not resolve to the entry.
    key_b = "c" * 16 + "d" * 48
    assert cache.load(key_b, sample_kwargs={"draws": 10}) is None


def test_disk_cache_treats_a_corrupt_file_as_a_miss(tmp_path):
    """A truncated entry should cost a re-run, not fail the user's sample."""
    cache = _disk(tmp_path)
    key = "e" * 64
    path = cache._path(key, {"draws": 10})
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"not a netcdf file")

    with pytest.warns(UserWarning, match="unreadable cache file"):
        assert cache.load(key, sample_kwargs={"draws": 10}) is None


def test_disk_cache_leaves_no_temp_file_when_writing_fails(tmp_path):
    from unittest.mock import patch

    import arviz as az

    from cloudposterior._idata import add_group

    cache = _disk(tmp_path)
    idata = az.InferenceData()
    add_group(idata, "posterior", az.dict_to_dataset({"x": np.zeros((2, 5))}))

    with patch.object(type(idata), "to_netcdf", side_effect=RuntimeError("disk full")):
        with pytest.raises(RuntimeError, match="disk full"):
            cache.save("f" * 64, idata, sample_kwargs={"draws": 10})

    assert list(tmp_path.rglob("*.tmp")) == []


def test_disk_cache_prunes_to_the_newest_entries(tmp_path):
    import arviz as az

    from cloudposterior._idata import add_group
    from cloudposterior.cache import _DISK_KEEP_PER_MODEL

    cache = _disk(tmp_path)
    idata = az.InferenceData()
    add_group(idata, "posterior", az.dict_to_dataset({"x": np.zeros((2, 5))}))

    for i in range(_DISK_KEEP_PER_MODEL + 3):
        cache.save(f"{i:064d}", idata, sample_kwargs={"draws": i})

    assert len(list(tmp_path.rglob("*.nc"))) == _DISK_KEEP_PER_MODEL


def test_cleanup_cache_removes_the_tree(tmp_path):
    import arviz as az

    from cloudposterior._idata import add_group
    from cloudposterior.cache import cleanup_cache

    cache = _disk(tmp_path)
    idata = az.InferenceData()
    add_group(idata, "posterior", az.dict_to_dataset({"x": np.zeros((2, 5))}))
    cache.save("a" * 64, idata, sample_kwargs={"draws": 10})

    assert cleanup_cache(tmp_path) == 1
    assert not tmp_path.exists()
