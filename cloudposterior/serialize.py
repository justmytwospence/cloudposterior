"""Serialization of PyMC models for remote execution.

Cloudpickle captures observed data inside the model object, so we ship a
single compressed model blob — no separate data payload needed.
"""

from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass

import cloudpickle
import lz4.frame


@dataclass
class SamplingPayload:
    """Everything needed to run sampling on a remote machine."""

    model_bytes: bytes  # cloudpickle'd pm.Model (includes observed data), lz4 compressed
    version_manifest: dict[str, str]
    sample_kwargs: dict


def get_version_manifest() -> dict[str, str]:
    """Capture versions of key packages in the current environment."""
    packages = [
        "pymc",
        "pytensor",
        "numpy",
        "scipy",
        "cloudpickle",
        "arviz",
    ]
    manifest = {"python": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"}
    for pkg in packages:
        try:
            mod = importlib.import_module(pkg)
            manifest[pkg] = getattr(mod, "__version__", "unknown")
        except ImportError:
            pass

    # Check optional samplers
    for pkg in ("nutpie", "numpyro", "jax"):
        try:
            mod = importlib.import_module(pkg)
            manifest[pkg] = getattr(mod, "__version__", "unknown")
        except ImportError:
            pass

    return manifest


def serialize_model(model) -> bytes:
    """Serialize a PyMC model using cloudpickle + lz4 compression."""
    raw = cloudpickle.dumps(model)
    return lz4.frame.compress(raw)


def deserialize_model(data: bytes):
    """Deserialize a PyMC model from cloudpickle + lz4 bytes."""
    import pickle

    raw = lz4.frame.decompress(data)
    return pickle.loads(raw)


def create_payload(
    model,
    sample_kwargs: dict,
) -> SamplingPayload:
    """Create a complete serialized payload for remote sampling."""
    return SamplingPayload(
        model_bytes=serialize_model(model),
        version_manifest=get_version_manifest(),
        sample_kwargs=sample_kwargs,
    )


def payload_size_mb(payload: SamplingPayload) -> float:
    """Total payload size in MB."""
    return len(payload.model_bytes) / (1024 * 1024)
