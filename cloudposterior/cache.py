"""Result caching based on model + data + sampling config.

Two backends:
- MemoryCache (default): fast, lives for the session
- DiskCache: persistent across sessions, project-local

Disk layout::

    .cloudposterior/
        eight_schools/                     # model name
            gentle-fox/                    # wordhash of observed data
                draws2000_tune1000_chains4.nc   # human-readable MCMC params
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Protocol


def compute_cache_key(model_bytes: bytes, data_bytes: bytes, sample_kwargs: dict) -> str:
    """Deterministic SHA-256 hash of model + data + sampling config."""
    h = hashlib.sha256()
    h.update(model_bytes)
    h.update(data_bytes)
    for k, v in sorted(sample_kwargs.items()):
        h.update(f"{k}={v}".encode())
    return h.hexdigest()


def _model_slug(model) -> str:
    """Derive a filesystem-safe directory name from a PyMC model."""
    from cloudposterior.naming import get_model_name, slugify
    return slugify(get_model_name(model, stack_offset=4))


def _data_slug(data_bytes: bytes) -> str:
    """Wordhash of the observed data for the directory name."""
    from cloudposterior.wordhash import wordhash
    return f"data-{wordhash(data_bytes)}"


def _params_filename(sample_kwargs: dict) -> str:
    """Human-readable filename from MCMC sampling params."""
    parts = []
    for key in ("draws", "tune", "chains", "cores", "nuts_sampler", "target_accept"):
        if key in sample_kwargs and sample_kwargs[key] is not None:
            val = sample_kwargs[key]
            # Skip defaults that don't add info
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

    Layout: {base_dir}/{model_name}/{data_slug}/{params}.nc

    Args:
        base_dir: Root cache directory. Defaults to ./.cloudposterior
        model: PyMC model, used to derive the top-level directory name
    """

    def __init__(self, base_dir: str | Path | None = None, model=None):
        self._base = Path(base_dir) if base_dir else Path(".cloudposterior")
        self._model_slug = _model_slug(model)

    def _path(self, key: str, data_bytes: bytes | None = None, sample_kwargs: dict | None = None) -> Path:
        if data_bytes is not None and sample_kwargs is not None:
            data_dir = self._base / self._model_slug / _data_slug(data_bytes)
            data_dir.mkdir(parents=True, exist_ok=True)
            return data_dir / f"{_params_filename(sample_kwargs)}.nc"
        # Fallback: flat hash-based path
        from cloudposterior.wordhash import wordhash
        fallback_dir = self._base / self._model_slug
        fallback_dir.mkdir(parents=True, exist_ok=True)
        return fallback_dir / f"{wordhash(key)}.nc"

    def load(self, key: str, data_bytes: bytes | None = None, sample_kwargs: dict | None = None):
        import arviz as az

        path = self._path(key, data_bytes=data_bytes, sample_kwargs=sample_kwargs)
        if path.exists():
            idata = az.from_netcdf(str(path))
            for group in idata.groups():
                getattr(idata, group).load()
            return idata
        return None

    def save(self, key: str, idata, data_bytes: bytes | None = None, sample_kwargs: dict | None = None) -> None:
        path = self._path(key, data_bytes=data_bytes, sample_kwargs=sample_kwargs)
        path.parent.mkdir(parents=True, exist_ok=True)
        idata.to_netcdf(str(path))


# Module-level default memory cache (shared across all calls in a session)
_default_memory_cache = MemoryCache()


def get_default_cache() -> MemoryCache:
    return _default_memory_cache


def resolve_cache(cache_arg, model=None) -> CacheBackend | None:
    """Resolve the cache argument from cp.wrap() into a CacheBackend.

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
