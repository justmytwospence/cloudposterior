"""Error paths and degraded-mode behavior.

Each of these was a case where a recoverable problem escalated: a broken
widget killed the sampling call, a partial progress payload raised mid-decode,
a stale error message misdiagnosed the failure.
"""

import queue

import pytest

from cloudposterior.progress import (
    ProgressAggregator,
    SamplingProgress,
    decode_progress_event,
)


# -- progress decoding -------------------------------------------------------

def test_partial_phase_payload_decodes_instead_of_raising():
    """The tolerance comment promised this; direct lookups raised KeyError."""
    event = decode_progress_event({"type": "phase", "phase": "sampling"})
    assert event is not None
    assert event.elapsed == 0.0


def test_partial_sampling_payload_decodes():
    event = decode_progress_event({"type": "sampling", "chains": {"0": {}}})
    assert event.chains[0].draw == 0
    assert event.chains[0].total == 0


def test_numeric_fields_are_coerced():
    """A null or stringly field reaching the dashboard hit .toFixed and read
    as the run having gone offline."""
    event = decode_progress_event({
        "type": "sampling",
        "chains": {"0": {"draw": "10", "total": "20", "step_size": None}},
        "elapsed": "1.5",
    })
    assert event.chains[0].draw == 10
    assert event.chains[0].step_size == 0.0
    assert event.elapsed == 1.5


def test_unknown_phase_is_skipped_not_raised():
    assert decode_progress_event({"type": "phase", "phase": "no-such-phase"}) is None


# -- progress aggregation ----------------------------------------------------

def test_aggregator_emits_a_final_snapshot_after_stop():
    """The last per-draw events land between the previous emit and the stop,
    which left the display frozen just short of the total."""
    from cloudposterior.progress import ChainProgress

    q: queue.Queue = queue.Queue()
    agg = ProgressAggregator(q, interval=0.05)
    q.put((0, ChainProgress(draw=1000, total=1000, phase="sampling")))

    snapshots = []
    for snap in agg.snapshots():
        snapshots.append(snap)
        agg.stop()

    assert len(snapshots) >= 2, "expected a final flush after stop()"
    assert snapshots[-1].chains[0].draw == 1000


def test_aggregator_respects_its_interval_when_idle():
    """Breaking out on the first empty-queue timeout re-emitted an identical
    snapshot every ~0.1s instead of once per interval."""
    import time

    from cloudposterior.progress import ChainProgress

    q: queue.Queue = queue.Queue()
    agg = ProgressAggregator(q, interval=0.4)
    q.put((0, ChainProgress(draw=1, total=10, phase="sampling")))

    start = time.time()
    count = 0
    for _ in agg.snapshots():
        count += 1
        if count == 2:
            agg.stop()
    # Two interval-spaced snapshots cannot arrive in well under one interval.
    assert time.time() - start >= 0.3


def test_snapshot_is_none_without_chains():
    agg = ProgressAggregator(queue.Queue(), interval=0.01)
    assert agg._snapshot() is None


# -- modal error classification ----------------------------------------------

def test_stop_token_in_a_message_is_not_an_auth_error():
    """Matching a bare "token" rewrote this codebase's own stop_token errors
    as "Modal is not authenticated", hiding the real cause."""
    from cloudposterior.backends.modal_backend import _handle_modal_error

    original = RuntimeError("invalid stop_token for this run")
    assert _handle_modal_error(original) is original


@pytest.mark.parametrize(
    "message",
    [
        "Token missing. Could not authenticate client.",
        "unauthenticated: no credentials found",
        "Run `modal setup` to get started",
    ],
)
def test_real_auth_errors_get_the_setup_message(message):
    from cloudposterior.backends.modal_backend import _handle_modal_error

    wrapped = _handle_modal_error(RuntimeError(message))
    assert "not authenticated" in str(wrapped)
    assert wrapped.__cause__ is not None


# -- endpoint labels ---------------------------------------------------------

def test_endpoint_labels_stay_within_modal_limits():
    """model_slug is unbounded ("mu-tau-theta-sigma-plus3"); once the label
    exceeded the limit the endpoint was rejected or silently renamed, and the
    dashboard's sibling-URL derivation broke."""
    from cloudposterior.backends.modal_backend import _endpoint_uid, _truncate_label

    uid = _endpoint_uid("cp-dash-deadbeef")
    stem = f"{_truncate_label('mu-tau-theta-sigma-lambda-plus7-and-more')}-{uid}"
    assert len(f"{stem}-progress") <= 63


def test_endpoint_uid_is_full_entropy():
    from cloudposterior.backends.modal_backend import _endpoint_uid

    uid = _endpoint_uid("cp-dash-deadbeef")
    assert len(uid) == 32
    assert _endpoint_uid("cp-dash-deadbeef") != uid  # random per provision


# -- dashboard completion ----------------------------------------------------

def test_dashboard_sink_completes_on_error():
    """An error is as terminal as a finished download; without this the page
    polls a billed endpoint at 1 Hz forever."""
    from cloudposterior.dashboard import DashboardSink
    from cloudposterior.progress import JobPhase, PhaseUpdate

    store: dict = {}
    sink = DashboardSink(store, key="k")
    sink.show_phase(PhaseUpdate(JobPhase.SAMPLING, "error", "kaboom", 1.0))
    assert store["k"]["complete"] is True


def test_dashboard_sink_state_is_initialized_upfront():
    from cloudposterior.dashboard import DashboardSink

    sink = DashboardSink({}, key="k")
    assert sink._convergence == {} and sink._traces == {}


# -- display fallback --------------------------------------------------------

def test_build_sinks_falls_back_when_the_widget_cannot_be_built(monkeypatch):
    """anywidget is a hard dep, but a version skew must not take down the run."""
    from cloudposterior import api
    from cloudposterior import display as display_mod

    monkeypatch.setattr(display_mod, "_is_notebook", lambda: True)
    monkeypatch.setattr(display_mod, "_is_marimo", lambda: False)

    def explode(*a, **k):
        raise ImportError("no anywidget here")

    monkeypatch.setattr(display_mod, "NotebookDisplay", explode)

    with pytest.warns(UserWarning, match="falling back to terminal"):
        sinks = api._build_sinks(progress=True, instance_desc="test")
    try:
        assert isinstance(sinks[0], display_mod.TerminalDisplay)
    finally:
        api._stop_sinks(sinks)


def test_build_sinks_stops_the_display_when_a_later_step_raises(monkeypatch):
    """A bad notify= value raises by design; the started Rich Live must not
    escape with it, still running and painting over the traceback."""
    from cloudposterior import api

    stopped = []

    class FakeDisplay:
        def __init__(self, *a, **k):
            pass

        def start(self):
            pass

        def stop(self):
            stopped.append(True)

    from cloudposterior import display as display_mod

    monkeypatch.setattr(display_mod, "_is_notebook", lambda: False)
    monkeypatch.setattr(display_mod, "_is_marimo", lambda: False)
    monkeypatch.setattr(display_mod, "TerminalDisplay", FakeDisplay)

    with pytest.raises((ValueError, TypeError)):
        api._build_sinks(
            progress=True, instance_desc="test", notify={"bogus_key": "x"}
        )
    assert stopped == [True]


# -- worker adaptive-stop guard ---------------------------------------------

def test_until_detects_a_model_with_no_scalar_parameters():
    """Convergence is computed over (chain, draw) variables only, so until=
    could never fire on a purely vector-valued model."""
    import pymc as pm

    from cloudposterior.remote.worker import _has_scalar_free_rvs

    with pm.Model() as vector_only:
        pm.Normal("beta", 0, 1, shape=5)
    assert _has_scalar_free_rvs(vector_only) is False

    with pm.Model() as has_scalar:
        pm.Normal("beta", 0, 1, shape=5)
        pm.HalfNormal("sigma", 1)
    assert _has_scalar_free_rvs(has_scalar) is True


def test_sampling_progress_defaults_are_sane():
    assert SamplingProgress(chains={}, total_divergences=0, elapsed=0.0).chains == {}
