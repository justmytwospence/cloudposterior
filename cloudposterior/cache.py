"""Result caching based on model + data + sampling config hash.

Two backends:
- MemoryCache (default): fast, lives for the session
- DiskCache: persistent across sessions, configurable directory

Disk layout::

    ~/.cache/cloudposterior/
        eight_schools/              # model name (or "unnamed")
            a1b2c3d4e5f6.nc        # short hash of full key
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
    if model is not None and hasattr(model, "name") and model.name:
        return re.sub(r"[^a-zA-Z0-9]+", "_", model.name).strip("_").lower()
    if model is not None and hasattr(model, "free_RVs") and model.free_RVs:
        names = [rv.name.split("::")[-1] for rv in model.free_RVs[:4]]
        slug = "_".join(names)
        return re.sub(r"[^a-zA-Z0-9]+", "_", slug).strip("_").lower()
    return "unnamed"


class CacheBackend(Protocol):
    def load(self, key: str): ...
    def save(self, key: str, idata) -> None: ...


class MemoryCache:
    """In-memory cache. Fast, lives for the session."""

    def __init__(self):
        self._store: dict[str, object] = {}

    def load(self, key: str):
        return self._store.get(key)

    def save(self, key: str, idata) -> None:
        self._store[key] = idata


class DiskCache:
    """Persistent disk cache with model-based directory hierarchy.

    Layout: {base_dir}/{model_slug}/{short_hash}.nc

    Args:
        base_dir: Root cache directory. Defaults to ~/.cache/cloudposterior
        model: PyMC model, used to derive the subdirectory name
    """

    def __init__(self, base_dir: str | Path | None = None, model=None):
        self._base = Path(base_dir) if base_dir else Path.home() / ".cache" / "cloudposterior"
        self._model_dir = self._base / _model_slug(model)
        self._model_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        short = key[:12]
        return self._model_dir / f"{short}.nc"

    def load(self, key: str):
        import arviz as az

        path = self._path(key)
        if path.exists():
            idata = az.from_netcdf(str(path))
            idata.load()
            return idata
        return None

    def save(self, key: str, idata) -> None:
        path = self._path(key)
        idata.to_netcdf(str(path))


# Module-level default memory cache (shared across all calls in a session)
_default_memory_cache = MemoryCache()


def get_default_cache() -> MemoryCache:
    return _default_memory_cache


def resolve_cache(cache_arg, model=None) -> CacheBackend | None:
    """Resolve the cache argument from pd.wrap() into a CacheBackend.

    Args:
        cache_arg: True (memory), False (disabled), "disk" (default disk path),
                   Path/str (custom disk path), or a CacheBackend instance
        model: PyMC model for directory naming

    Returns:
        A CacheBackend or None if disabled.
    """
    if cache_arg is False:
        return None
    if cache_arg is True:
        return get_default_cache()
    if isinstance(cache_arg, str) and cache_arg == "disk":
        return DiskCache(model=model)
    if isinstance(cache_arg, (str, Path)):
        return DiskCache(base_dir=cache_arg, model=model)
    # Assume it's a CacheBackend instance
    if hasattr(cache_arg, "load") and hasattr(cache_arg, "save"):
        return cache_arg
    return get_default_cache()
