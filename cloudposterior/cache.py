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

import copy
from collections import OrderedDict
from pathlib import Path
from typing import Protocol

_MAX_LABEL_LEN = 80
_DISK_KEEP_PER_MODEL = 20


def _params_label(sample_kwargs: dict) -> str:
    """Human-readable label from common MCMC sampling params.

    Slugified: these values become a path component, and an unsanitized one
    (``nuts_sampler="../../etc/x"``) would escape the cache root.
    """
    from cloudposterior.naming import slugify

    parts = []
    for key in ("draws", "tune", "chains", "cores", "nuts_sampler", "target_accept"):
        if key in sample_kwargs and sample_kwargs[key] is not None:
            val = sample_kwargs[key]
            if key == "nuts_sampler" and val == "pymc":
                continue
            parts.append(f"{key}{val}")
    if not parts:
        parts.append("default")
    return slugify("_".join(parts))[:_MAX_LABEL_LEN] or "default"


class CacheBackend(Protocol):
    def load(self, key: str, **kwargs): ...
    def save(self, key: str, idata, **kwargs) -> None: ...


class MemoryCache:
    """In-memory LRU cache. Fast, lives for the session.

    Entries are copied on load. Callers routinely extend a returned trace in
    place (``pm.compute_log_likelihood`` merges a ``log_likelihood`` group into
    it), which would otherwise mutate the cached entry itself and hand every
    later hit a trace the original run never produced. DiskCache already
    materializes a fresh object per load; this keeps the two consistent.

    ``max_entries`` bounds the store: each entry is a full posterior, and the
    default instance lives for the whole process.
    """

    def __init__(self, max_entries: int = 8):
        self._store: OrderedDict[str, object] = OrderedDict()
        self._max_entries = max_entries

    def load(self, key: str, **kwargs):
        if key not in self._store:
            return None
        self._store.move_to_end(key)
        return copy.deepcopy(self._store[key])

    def save(self, key: str, idata, **kwargs) -> None:
        self._store[key] = idata
        self._store.move_to_end(key)
        while len(self._store) > self._max_entries:
            self._store.popitem(last=False)


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
        # Pure path computation -- directories are created on save() only, so
        # a cache probe (load miss) doesn't litter empty directories.
        cache_dir = self._base / self._model_slug
        # 16 chars (64 bits), matching payload_hash. At 8 chars two genuinely
        # different runs sharing a params label could resolve to one file and
        # silently return the wrong posterior.
        key_prefix = key[:16]
        if sample_kwargs is not None:
            label = _params_label(sample_kwargs)
            path = cache_dir / f"{label}-{key_prefix}.nc"
        else:
            path = cache_dir / f"{key_prefix}.nc"
        # Defense in depth: _params_label is slugified, so nothing should be
        # able to traverse out, but the cache root is a hard boundary.
        resolved = path.resolve()
        if not resolved.is_relative_to(self._base.resolve()):
            raise ValueError(f"cache path escapes the cache root: {path}")
        return path

    @staticmethod
    def _key_path(path: Path) -> Path:
        """Sidecar holding the full cache key for the entry at ``path``."""
        return path.with_name(path.name + ".key")

    def load(self, key: str, sample_kwargs: dict | None = None):
        import warnings

        import arviz as az

        from cloudposterior._idata import load_all

        path = self._path(key, sample_kwargs=sample_kwargs)
        if not path.exists():
            return None

        # The filename only carries a prefix of the key; verify the whole thing
        # so a prefix collision is a miss rather than the wrong posterior.
        key_path = self._key_path(path)
        if key_path.exists():
            try:
                if key_path.read_text().strip() != key:
                    return None
            except OSError:
                return None

        try:
            idata = az.from_netcdf(str(path))
            load_all(idata)
            return idata
        except Exception as exc:
            # A truncated or unreadable entry (crash mid-write, half-synced
            # network drive) should cost a re-run, not fail the user's sample.
            warnings.warn(
                f"cloudposterior: ignoring unreadable cache file {path} ({exc})",
                stacklevel=2,
            )
            return None

    def save(self, key: str, idata, sample_kwargs: dict | None = None) -> None:
        import os
        import uuid

        from cloudposterior.serialize import sanitize_inference_data

        path = self._path(key, sample_kwargs=sample_kwargs)
        path.parent.mkdir(parents=True, exist_ok=True)
        # copy=True: caching must not change the type of the caller's
        # idata.attrs as a side effect of being enabled.
        idata = sanitize_inference_data(idata, copy=True)
        # Write to a temp file then atomically replace. A plain to_netcdf(path)
        # truncates in place, which fails ("unable to truncate a file which is
        # already open") when overwrite= re-saves an entry that an earlier load
        # left open via xarray's lazy file cache. os.replace also makes the write
        # atomic (no half-written cache on a crash).
        #
        # The temp name carries pid + random: a fixed ".tmp" let two processes
        # saving the same key interleave writes into one file and publish the
        # interleaved result.
        tmp = path.with_name(f"{path.name}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp")
        try:
            idata.to_netcdf(str(tmp))
            os.replace(tmp, path)
        finally:
            if tmp.exists():
                try:
                    tmp.unlink()
                except OSError:
                    pass
        self._key_path(path).write_text(key)
        self._prune(path.parent)

    def _prune(self, directory: Path, keep: int = _DISK_KEEP_PER_MODEL) -> None:
        """Keep the N most recent entries per model.

        The remote Volume already prunes its payloads; without this the local
        cache grows a full NetCDF posterior per run, forever.
        """
        try:
            entries = sorted(
                directory.glob("*.nc"), key=lambda p: p.stat().st_mtime, reverse=True
            )
        except OSError:
            return
        for stale in entries[keep:]:
            for victim in (stale, self._key_path(stale)):
                try:
                    victim.unlink()
                except OSError:
                    pass


# Module-level default memory cache (shared across all calls in a session)
_default_memory_cache = MemoryCache()


def get_default_cache() -> MemoryCache:
    return _default_memory_cache


def cleanup_cache(base_dir: str | Path | None = None) -> int:
    """Delete the local disk cache tree. Returns the number of entries removed.

    The disk counterpart of ``cleanup_volumes``.
    """
    import shutil

    base = Path(base_dir) if base_dir else Path(".cloudposterior")
    if not base.exists():
        return 0
    removed = len(list(base.rglob("*.nc")))
    shutil.rmtree(base)
    return removed


def resolve_cache(cache_arg, model=None) -> CacheBackend | None:
    """Resolve the cache argument from cp.cloud() into a CacheBackend.

    Args:
        cache_arg: True or "memory" (session memory cache), False (disabled),
                   "disk" (project-local ./.cloudposterior), any other
                   str/Path (custom disk cache directory), or a CacheBackend
                   instance.

    Raises:
        TypeError: if ``cache_arg`` is none of the above (e.g. ``cache=42``)
            rather than silently falling back to the default cache.
    """
    if cache_arg is False:
        return None
    if cache_arg is True or cache_arg == "memory":
        return get_default_cache()
    if cache_arg == "disk":
        return DiskCache(model=model)
    if isinstance(cache_arg, (str, Path)):
        return DiskCache(base_dir=cache_arg, model=model)
    if hasattr(cache_arg, "load") and hasattr(cache_arg, "save"):
        return cache_arg
    raise TypeError(
        f"cache must be bool | 'disk' | str | Path | CacheBackend; "
        f"got {type(cache_arg).__name__}={cache_arg!r}"
    )