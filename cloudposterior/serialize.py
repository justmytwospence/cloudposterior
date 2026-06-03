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
    idata_bytes: bytes | None = None  # lz4 NetCDF trace, for posterior predictive


def get_version_manifest() -> dict[str, str]:
    """Capture versions of key packages in the current environment."""
    packages = [
        "pymc",
        "pytensor",
        "numpy",
        "scipy",
        "cloudpickle",
        "arviz",
        "numba",
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


def serialize_model_with_step(model, step) -> bytes:
    """Serialize a model together with a step method in one cloudpickle blob.

    A step method instance holds references into the model graph. Pickling it
    *separately* from the model produces a step whose value variables belong to
    a different graph instance than a separately-deserialized model, which makes
    ``pm.sample(step=...)`` raise "not a value variable in the model". Bundling
    both in a single pickle preserves shared object identity, so the worker can
    reconstruct a matching ``(model, step)`` pair. The worker detects this dict
    payload via :func:`deserialize_model` returning ``{"model", "step"}``.
    """
    raw = cloudpickle.dumps({"model": model, "step": step})
    return lz4.frame.compress(raw)


def serialize_inference_data(idata) -> bytes:
    """Serialize an arviz InferenceData to lz4-compressed NetCDF bytes.

    Mirrors the worker's result encoding; used client-side to ship a trace to a
    remote posterior-predictive call and to return cp.map results.
    """
    import os
    import tempfile

    sanitize_inference_data(idata)
    with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        idata.to_netcdf(tmp_path)
        with open(tmp_path, "rb") as f:
            return lz4.frame.compress(f.read())
    finally:
        os.unlink(tmp_path)


def deserialize_inference_data(data: bytes):
    """Load an arviz InferenceData from lz4-compressed NetCDF bytes."""
    import os
    import tempfile

    import arviz as az

    raw = lz4.frame.decompress(data)
    with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as tmp:
        tmp.write(raw)
        tmp_path = tmp.name
    try:
        return az.from_netcdf(tmp_path)
    finally:
        os.unlink(tmp_path)


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


# The attr sanitizer lives in cloudposterior._idata alongside the other
# arviz 0.x / 1.x compatibility shims; re-exported here for existing callers.
from cloudposterior._idata import sanitize_inference_data  # noqa: E402,F401
