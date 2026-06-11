"""Test the remote worker locally (no Modal) to validate the full pipeline."""

import msgpack
import numpy as np
import pymc as pm
import pytest

from cloudposterior.serialize import create_payload
from cloudposterior.remote.worker import run_sampling


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
    from cloudposterior._idata import group_names

    assert "posterior" in group_names(idata)
    assert "mu" in idata.posterior.data_vars
    print(f"  Posterior shape: {dict(idata.posterior.sizes)}")


def test_run_sampling_blocking_writes_dashboard_progress(monkeypatch):
    """cp.map's blocking worker writes per-model progress into the dashboard Dict
    (under its model label) and still returns the result bytes."""
    from cloudposterior.remote import worker

    store = {}
    events = [
        msgpack.packb({"type": "phase", "phase": "sampling", "status": "in_progress",
                       "message": "MCMC sampling started", "elapsed": 0.0}),
        msgpack.packb({"type": "sampling",
                       "chains": {"0": {"draw": 5, "total": 10, "phase": "sampling",
                                        "divergences": 0, "step_size": 0.1,
                                        "draws_per_sec": 1.0, "eta_seconds": 5.0,
                                        "tree_size": 7}},
                       "total_divergences": 0, "elapsed": 1.0, "total_draws": 5}),
        msgpack.packb({"type": "result", "size_mb": 0.01}),
        b"RESULT-BYTES",
    ]

    def fake_stream(model, sample_kwargs, nuts_sampler="nutpie", stop_dict_name=None, stop_key=None):
        assert stop_key == "pooled-0"  # per-model stop key threaded through
        for e in events:
            yield e

    monkeypatch.setattr(worker, "_load_model_from_volume", lambda p: object())
    monkeypatch.setattr(worker, "_open_dict", lambda name: store if name else None)
    monkeypatch.setattr(worker, "_sample_and_stream", fake_stream)
    monkeypatch.setattr(
        "cloudposterior.backends.modal_backend._run_blocking",
        lambda fn, *a, **k: fn(*a, **k),
    )

    out = worker.run_sampling_blocking(
        "/data/p.bin", {"draws": 10}, "nutpie",
        progress_dict_name="cp-dash-x", progress_key="pooled-0",
        stop_dict_name="cp-dash-x",
    )
    assert out == b"RESULT-BYTES"
    state = store["pooled-0"]                     # written under the model label
    assert state["sampling"]["chains"]["0"]["draw"] == 5
    assert any(p["label"] == "sampling" for p in state["phases"])


def test_run_sampling_blocking_no_dashboard_returns_bytes(monkeypatch):
    """Without progress args the worker behaves as before (just returns bytes)."""
    from cloudposterior.remote import worker

    events = [msgpack.packb({"type": "result", "size_mb": 0.01}), b"ONLY-BYTES"]
    monkeypatch.setattr(worker, "_load_model_from_volume", lambda p: object())
    monkeypatch.setattr(worker, "_sample_and_stream", lambda *a, **k: (e for e in events))
    out = worker.run_sampling_blocking("/data/p.bin", {"draws": 10}, "nutpie")
    assert out == b"ONLY-BYTES"


def test_stop_requested_honors_global_and_per_model():
    from cloudposterior.remote.worker import _open_dict, _stop_requested

    assert _open_dict(None) is None
    assert _stop_requested(None) is False
    assert _stop_requested({"stop": False}) is False
    assert _stop_requested({"stop": True}) is True                       # global stop
    assert _stop_requested({"stop:pooled-0": True}, "pooled-0") is True   # per-model
    assert _stop_requested({"stop:pooled-0": True}, "other") is False     # not mine
    assert _stop_requested({"stop": False}, "pooled-0") is False


def _tiny_model():
    with pm.Model() as model:
        pm.Normal("mu", 0, 1, observed=np.array([0.1, -0.2, 0.3]))
        pm.Normal("theta", 0, 1)
    return model


def _drain_worker(gen):
    """Run a worker generator; return (decoded_events, result_bytes)."""
    events, idata_bytes = [], None
    for chunk in gen:
        try:
            events.append(msgpack.unpackb(chunk, raw=False))
        except Exception:
            idata_bytes = chunk
    return events, idata_bytes


def test_worker_drops_return_inferencedata():
    """return_inferencedata=False must not crash the worker after sampling
    (the client warns an InferenceData is returned instead)."""
    payload = create_payload(
        _tiny_model(),
        {"draws": 10, "tune": 10, "chains": 1, "return_inferencedata": False},
    )
    events, idata_bytes = _drain_worker(
        run_sampling(payload.model_bytes, payload.sample_kwargs, nuts_sampler="pymc")
    )
    assert idata_bytes is not None
    assert any(e.get("type") == "result" for e in events)


def test_split_nutpie_kwargs_partitions_by_signature():
    from cloudposterior.remote.worker import _split_nutpie_kwargs

    class FakeNutpie:
        @staticmethod
        def sample(compiled_model, *, draws=1000, target_accept=0.8,
                   maxdepth=10, blocking=True):
            pass

    forwarded, dropped = _split_nutpie_kwargs(
        FakeNutpie, {"target_accept": 0.95, "idata_kwargs": {"log_likelihood": True}}
    )
    assert forwarded == {"target_accept": 0.95}
    assert set(dropped) == {"idata_kwargs"}

    class VarKwargsNutpie:
        @staticmethod
        def sample(compiled_model, **kwargs):
            pass

    # **kwargs signatures don't mean "accepts anything": nutpie's settings
    # parser raises on unknown names, so only known settings are forwarded.
    forwarded, dropped = _split_nutpie_kwargs(
        VarKwargsNutpie, {"target_accept": 0.9, "anything": 1}
    )
    assert forwarded == {"target_accept": 0.9} and set(dropped) == {"anything"}


def test_worker_nutpie_forwards_target_accept_and_reports_dropped():
    """The nutpie branch forwards supported kwargs and reports unsupported ones."""
    pytest.importorskip("nutpie")
    payload = create_payload(
        _tiny_model(),
        {"draws": 10, "tune": 10, "chains": 2, "target_accept": 0.95,
         "definitely_not_a_kwarg": 1},
    )
    events, idata_bytes = _drain_worker(
        run_sampling(payload.model_bytes, payload.sample_kwargs, nuts_sampler="nutpie")
    )
    assert idata_bytes is not None
    messages = [e.get("message", "") for e in events if e.get("type") == "phase"]
    assert any("definitely_not_a_kwarg" in m for m in messages)
    assert not any("target_accept" in m for m in messages)  # forwarded, not dropped
