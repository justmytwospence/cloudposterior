"""Human-readable names, safe cache identity, and cloud payload hashes.

Cache graph/key computation is owned by pymc-cache. The transport payload
hash intentionally hashes the actual serialized bytes, not the cache identity.
"""

from __future__ import annotations

import hashlib

from pymc_cache.identity import (  # noqa: F401
    _token as _identity_token,
    cache_key,
    data_digest,
    model_digest,
)
from pymc_cache.naming import (
    model_slug,  # noqa: F401
    slugify as _cache_slugify,
)


def slugify(name: str, separator: str = "_") -> str:
    """Compatibility wrapper for the cache's safe slug with custom separators."""
    return _cache_slugify(name).replace("_", separator)


def get_model_name(model) -> str:
    """Return a descriptive (not identity-bearing) name for a PyMC model."""
    name = getattr(model, "name", None)
    if name:
        return name
    variables = getattr(model, "free_RVs", ())
    if variables:
        name = "_".join(str(rv.name).split("::")[-1] for rv in variables[:4])
        if len(variables) > 4:
            name += f"_plus{len(variables) - 4}"
        return name
    return "unnamed"


def payload_hash(model_bytes: bytes) -> str:
    """The 16-character SHA-256 prefix identifying a serialized Volume blob."""
    return hashlib.sha256(model_bytes).hexdigest()[:16]


def _kwarg_token(value):
    """Compatibility spelling for pymc-cache's safe structured value token."""
    return _identity_token(value)
