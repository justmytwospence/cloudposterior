"""Serialization of PyMC models and data for remote execution."""

from __future__ import annotations

import importlib
import io
import sys
from dataclasses import dataclass, field

import cloudpickle
import lz4.frame
import numpy as np


@dataclass
class SamplingPayload:
    """Everything needed to run sampling on a remote machine."""

    model_bytes: bytes  # cloudpickle'd pm.Model, lz4 compressed
    data_bytes: bytes  # numpy .npz, lz4 compressed
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


def serialize_observed_data(model) -> bytes:
    """Extract and serialize observed data from a PyMC model.

    Observed data is pulled out and serialized separately as compressed
    numpy arrays for better compression than pickling inline.
    """
    observed = {}
    for rv in model.observed_RVs:
        obs_data = rv.tag.test_value if hasattr(rv.tag, "test_value") else None
        if obs_data is None:
            # Try getting from the observed variable's owner
            for v in model.rvs_to_values.values():
                if hasattr(v, "data") and hasattr(v, "name") and v.name == rv.name + "_observed":
                    obs_data = v.data
                    break
        if obs_data is not None:
            if hasattr(obs_data, "eval"):
                obs_data = obs_data.eval()
            observed[rv.name] = np.asarray(obs_data)

    buf = io.BytesIO()
    np.savez(buf, **observed)  # uncompressed -- lz4 handles compression in one pass
    return lz4.frame.compress(buf.getvalue())


def deserialize_observed_data(data: bytes) -> dict[str, np.ndarray]:
    """Deserialize observed data from compressed numpy bytes."""
    raw = lz4.frame.decompress(data)
    buf = io.BytesIO(raw)
    npz = np.load(buf)
    return dict(npz)


def create_payload(
    model,
    sample_kwargs: dict,
) -> SamplingPayload:
    """Create a complete serialized payload for remote sampling."""
    return SamplingPayload(
        model_bytes=serialize_model(model),
        data_bytes=serialize_observed_data(model),
        version_manifest=get_version_manifest(),
        sample_kwargs=sample_kwargs,
    )


def payload_size_mb(payload: SamplingPayload) -> float:
    """Total payload size in MB."""
    return (len(payload.model_bytes) + len(payload.data_bytes)) / (1024 * 1024)
