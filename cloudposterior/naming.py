"""Derive human-readable and machine-correct identifiers from PyMC models.

Used by cache (directory names + keys), notify (topic names), and volumes.

Two layers:
- Human-readable: model_slug (for directory browsability)
- Machine-correct: payload_hash, cache_key (for identity checks)
"""

from __future__ import annotations

import hashlib
import re


def get_model_name(model) -> str:
    """Get the best human-readable name for a PyMC model.

    Tries two strategies:
    1. model.name (explicit PyMC model name)
    2. Free RV names (e.g. "mu_tau_theta")
    """
    # 1. Explicit model name
    if model is not None and hasattr(model, "name") and model.name:
        return model.name

    # 2. Derive from free RV names
    if model is not None and hasattr(model, "free_RVs") and model.free_RVs:
        names = [rv.name.split("::")[-1] for rv in model.free_RVs[:4]]
        result = "_".join(names)
        if len(model.free_RVs) > 4:
            result += f"_plus{len(model.free_RVs) - 4}"
        return result

    return "unnamed"


def slugify(name: str, separator: str = "_") -> str:
    """Convert a name to a filesystem/URL-safe slug."""
    return re.sub(r"[^a-zA-Z0-9]+", separator, name).strip(separator).lower()


def model_slug(model) -> str:
    """Filesystem-safe slug from a PyMC model name, e.g. 'radon'."""
    return slugify(get_model_name(model))


def payload_hash(model_bytes: bytes) -> str:
    """SHA-256 hex prefix of serialized model bytes (16 chars).

    Used for Volume payload filenames. Captures model + data identity
    since PyMC bundles observed data into the model pickle.
    """
    return hashlib.sha256(model_bytes).hexdigest()[:16]


def cache_key(model_bytes: bytes, sample_kwargs: dict) -> str:
    """Full SHA-256 of model + sampling config.

    Used for trace result caching. model_bytes already contains observed
    data (PyMC bundles it), so no separate data_bytes parameter needed.
    """
    h = hashlib.sha256()
    h.update(model_bytes)
    for k, v in sorted(sample_kwargs.items()):
        h.update(f"{k}={v}".encode())
    return h.hexdigest()
