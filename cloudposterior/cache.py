"""Backward-compatible cache backends backed by :mod:`pymc_cache`.

New entries use the versioned, full-key-verified format. Older cloudposterior
entries are deliberately not read; cleanup can remove verified old pairs.
"""

from __future__ import annotations

import re
from pathlib import Path

from pymc_cache.backends import (
    CacheBackend as CacheBackend,
    DiskCache as _DiskCache,
    MemoryCache as MemoryCache,
    get_default_cache as get_default_cache,
    resolve_cache as _resolve_cache,
)

_DISK_KEEP_PER_MODEL = 20
# Kept for callers that explicitly inspect/clear the historical shared cache.
_default_memory_cache = get_default_cache()


class DiskCache(_DiskCache):
    """The pymc-cache disk format under cloudposterior's historical root."""

    def __init__(self, base_dir: str | Path | None = None, model=None, **kwargs):
        super().__init__(base_dir if base_dir is not None else ".cloudposterior",
                         model=model, **kwargs)

    def _prune(self, directory: Path, keep: int = _DISK_KEEP_PER_MODEL) -> None:
        """Compatibility alias; prune only verified new-format entries."""
        self.prune(keep)

    @staticmethod
    def _key_path(path: Path) -> Path:
        return path.with_name(path.name + ".key")


def resolve_cache(cache_arg, model=None) -> CacheBackend | None:
    """Resolve the old aliases with ``.cloudposterior`` as the disk root."""
    if cache_arg == "disk":
        return DiskCache(model=model)
    if isinstance(cache_arg, (str, Path)) and cache_arg != "memory":
        return DiskCache(cache_arg, model=model)
    return _resolve_cache(cache_arg, model=model, default_directory=".cloudposterior")


def cleanup_cache(base_dir: str | Path | None = None) -> int:
    """Remove owned cache entries, preserving directories, symlinks and other files.

    Legacy entries are removed only when their adjacent ``.key`` file contains
    a full hexadecimal SHA-256 whose prefix matches the old filename. Never
    recurse into symlink directories or follow a symlinked cache root.
    """
    root = Path(base_dir) if base_dir is not None else Path(".cloudposterior")
    if root.is_symlink() or not root.is_dir():
        return 0
    removed = DiskCache(root).clear()
    for directory in root.iterdir():
        if directory.is_symlink() or not directory.is_dir():
            continue
        for path in directory.glob("*.nc"):
            if path.is_symlink() or not path.is_file():
                continue
            match = re.fullmatch(r"[a-z0-9_]+-([0-9a-f]{16})\.nc", path.name)
            if match is None:
                continue
            key_file = path.with_name(path.name + ".key")
            if key_file.is_symlink() or not key_file.is_file():
                continue
            try:
                key = key_file.read_text().strip()
                if re.fullmatch(r"[0-9a-f]{64}", key) and key.startswith(match[1]):
                    path.unlink()
                    key_file.unlink()
                    removed += 1
            except OSError:
                continue
    return removed
