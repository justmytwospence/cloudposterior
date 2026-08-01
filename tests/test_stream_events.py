"""Unit tests for the client-side worker-stream decoder (no Modal needed)."""

import msgpack
import pytest

from cloudposterior.backends.modal_backend import _stream_events
from cloudposterior.progress import PhaseUpdate


def _drive(chunks):
    """Run the _stream_events generator to completion, returning
    (yielded_events, returned_idata_bytes)."""
    events = []
    captured = []
    gen = _stream_events(iter(chunks), captured)
    result = None
    while True:
        try:
            events.append(next(gen))
        except StopIteration as stop:
            result = stop.value
            break
    return events, result


def test_stream_events_decodes_events_and_result():
    chunks = [
        msgpack.packb({"type": "phase", "phase": "sampling",
                       "status": "in_progress", "message": "go", "elapsed": 0.0}),
        msgpack.packb({"type": "result", "size_mb": 0.1}),
        b"RESULT-BYTES",
    ]
    events, result = _drive(chunks)
    assert len(events) == 1 and isinstance(events[0], PhaseUpdate)
    assert result == b"RESULT-BYTES"


def test_stream_events_skips_unknown_phase_without_corrupting_result():
    """An event with a phase this client doesn't know (newer worker) is
    skipped; the chunk must not be mistaken for the result payload."""
    chunks = [
        msgpack.packb({"type": "phase", "phase": "brand_new_phase",
                       "status": "done", "message": "x", "elapsed": 0.0}),
        msgpack.packb({"type": "phase", "phase": "sampling",
                       "status": "done", "message": "ok", "elapsed": 1.0}),
        msgpack.packb({"type": "result", "size_mb": 0.1}),
        b"RESULT-BYTES",
    ]
    events, result = _drive(chunks)
    assert len(events) == 1 and events[0].message == "ok"
    assert result == b"RESULT-BYTES"


def test_stream_events_reraises_worker_error():
    def gen():
        yield msgpack.packb({"type": "phase", "phase": "sampling",
                             "status": "in_progress", "message": "go",
                             "elapsed": 0.0})
        raise RuntimeError("worker exploded")

    with pytest.raises(RuntimeError, match="worker exploded"):
        _drive(gen())
