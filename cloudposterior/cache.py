"""Result caching based on model + sampling config.

Two backends:
- MemoryCache (default): fast, lives for the session
- DiskCache: persistent across sessions, project-local

Disk layout::

    .cloudposterior/
        radon_intercepts/
            draws2000_tune1000_chains4-a3f7b2c9.nc
        radon_slopes/
            draws2000_tune1000_chains4-7c2e5fa8.nc

Filenames combine human-readable params with a hash suffix for
uniqueness. Two runs with the same draws/tune/chains but different
random_seed or target_accept get different files.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol


def _params_label(sample_kwargs: dict) -> str:
    """Human-readable label from common MCMC sampling params."""
    parts = []
    for key in ("draws", "tune", "chains", "cores", "nuts_sampler", "target_accept"):
        if key in sample_kwargs and sample_kwargs[key] is not None:
            val = sample_kwargs[key]
            if key == "nuts_sampler" and val == "pymc":
                continue
            parts.append(f"{key}{val}")
    if not parts:
        parts.append("default")
    return "_".join(parts)


class CacheBackend(Protocol):
    def load(self, key: str, **kwargs): ...
    def save(self, key: str, idata, **kwargs) -> None: ...


class MemoryCache:
    """In-memory cache. Fast, lives for the session."""

    def __init__(self):
        self._store: dict[str, object] = {}

    def load(self, key: str, **kwargs):
        return self._store.get(key)

    def save(self, key: str, idata, **kwargs) -> None:
        self._store[key] = idata


class DiskCache:
    """Persistent disk cache with human-readable directory hierarchy.

    Layout: {base_dir}/{model_slug}/{params_label}-{key_prefix}.nc

    The filename combines human-readable params (draws, tune, chains) with
    a hash prefix from the full cache key for uniqueness. This ensures that
    runs differing only in non-displayed params (random_seed, init, etc.)
    never collide.

    Args:
        base_dir: Root cache directory. Defaults to ./.cloudposterior
        model: PyMC model, used to derive the top-level directory name
    """

    def __init__(self, base_dir: str | Path | None = None, model=None):
        from cloudposterior.naming import model_slug

        self._base = Path(base_dir) if base_dir else Path(".cloudposterior")
        self._model_slug = model_slug(model)

    def _path(self, key: str, sample_kwargs: dict | None = None) -> Path:
        cache_dir = self._base / self._model_slug
        cache_dir.mkdir(parents=True, exist_ok=True)
        # Use first 8 chars of cache key hash for uniqueness
        key_prefix = key[:8] if len(key) >= 8 else key
        if sample_kwargs is not None:
            label = _params_label(sample_kwargs)
            return cache_dir / f"{label}-{key_prefix}.nc"
        return cache_dir / f"{key_prefix}.nc"

    def load(self, key: str, sample_kwargs: dict | None = None):
        import arviz as az

        path = self._path(key, sample_kwargs=sample_kwargs)
        if path.exists():
            idata = az.from_netcdf(str(path))
            for group in idata.groups():
                getattr(idata, group).load()
            return idata
        return None

    def save(self, key: str, idata, sample_kwargs: dict | None = None) -> None:
        path = self._path(key, sample_kwargs=sample_kwargs)
        path.parent.mkdir(parents=True, exist_ok=True)
        idata.to_netcdf(str(path))


# Module-level default memory cache (shared across all calls in a session)
_default_memory_cache = MemoryCache()


def get_default_cache() -> MemoryCache:
    return _default_memory_cache


def resolve_cache(cache_arg, model=None) -> CacheBackend | None:
    """Resolve the cache argument from cp.cloud() into a CacheBackend.

    Args:
        cache_arg: True (memory), False (disabled), "disk" (project-local),
                   Path/str (custom disk path), or a CacheBackend instance
    """
    if cache_arg is False:
        return None
    if cache_arg is True:
        return get_default_cache()
    if isinstance(cache_arg, str) and cache_arg == "disk":
        return DiskCache(model=model)
    if isinstance(cache_arg, (str, Path)):
        return DiskCache(base_dir=cache_arg, model=model)
    if hasattr(cache_arg, "load") and hasattr(cache_arg, "save"):
        return cache_arg
    return get_default_cache()