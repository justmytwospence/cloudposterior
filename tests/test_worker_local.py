"""Test the remote worker locally (no Modal) to validate the full pipeline."""

import numpy as np
import pymc as pm
import msgpack

from cloudposterior.serialize import create_payload
from cloudposterior.remote.worker import run_sampling
from cloudposterior.progress import JobPhase


def test_worker_end_to_end():
    """Run the worker generator locally and verify we get progress + results."""
    y = np.array([28, 8, -3, 7, -1, 1, 18, 12], dtype=np.float64)
    sigma = np.array([15, 10, 16, 11, 9, 11, 10, 18], dtype=np.float64)

    with pm.Model() as model:
        mu = pm.Normal("mu", 0, 5)
        tau = pm.HalfCauchy("tau", 5)
        theta = pm.Normal("theta", mu=mu, sigma=tau, shape=8)
        pm.Normal("obs", mu=theta, sigma=sigma, observed=y)

    payload = create_payload(model, {"draws": 20, "tune": 20, "chains": 1})

    events = []
    idata_bytes = None

    for chunk in run_sampling(
        payload.model_bytes,
        payload.data_bytes,
        payload.sample_kwargs,
    ):
        try:
            decoded = msgpack.unpackb(chunk, raw=False)
            events.append(decoded)
            print(f"  Event: {decoded.get('type')} - {decoded.get('phase', decoded.get('elapsed', ''))}")
        except Exception:
            # Final chunk is compressed InferenceData
            idata_bytes = chunk
            print(f"  InferenceData: {len(chunk)} bytes")

    # Verify we got phase events
    phase_events = [e for e in events if e.get("type") == "phase"]
    assert len(phase_events) >= 2, f"Expected phase events, got {len(phase_events)}"

    # Verify we got sampling progress
    sampling_events = [e for e in events if e.get("type") == "sampling"]
    print(f"  Got {len(sampling_events)} sampling progress snapshots")

    # Verify we got the result metadata
    result_events = [e for e in events if e.get("type") == "result"]
    assert len(result_events) == 1

    # Verify InferenceData can be deserialized
    assert idata_bytes is not None
    import arviz as az
    import io
    import lz4.frame

    raw = lz4.frame.decompress(idata_bytes)
    idata = az.from_netcdf(io.BytesIO(raw))
    assert "posterior" in idata.groups()
    assert "mu" in idata.posterior
    print(f"  Posterior shape: {dict(idata.posterior.dims)}")
