from importlib.metadata import PackageNotFoundError, version

from cloudposterior.api import cleanup_volumes, cloud, map, sample
from cloudposterior.cache import CacheBackend, DiskCache, MemoryCache, cleanup_cache

try:
    __version__ = version("cloudposterior")
except PackageNotFoundError:  # not installed (e.g. vendored source checkout)
    __version__ = "0.0.0+unknown"

__all__ = [
    "cleanup_cache",
    "cleanup_volumes",
    "cloud",
    "map",
    "sample",
    "CacheBackend",
    "DiskCache",
    "MemoryCache",
    "__version__",
]
