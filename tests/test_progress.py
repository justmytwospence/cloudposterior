"""Test progress tracking with a real PyMC model."""

import numpy as np
import pymc as pm

from queue import Queue
from cloudposterior.progress import make_sampling_callback


def test_progress_callback_captures_draws():
    """Run pm.sample with our callback and verify it captures progress."""
    y = np.array([28, 8, -3, 7, -1, 1, 18, 12], dtype=np.float64)
    sigma = np.array([15, 10, 16, 11, 9, 11, 10, 18], dtype=np.float64)

    with pm.Model() as model:
        mu = pm.Normal("mu", 0, 5)
        tau = pm.HalfCauchy("tau", 5)
        theta = pm.Normal("theta", mu=mu, sigma=tau, shape=8)
        pm.Normal("obs", mu=theta, sigma=sigma, observed=y)

    queue = Queue()
    callback = make_sampling_callback(queue, tune=10, draws=10)

    with model:
        pm.sample(
            draws=10,
            tune=10,
            chains=1,
            nuts_sampler="pymc",
            callback=callback,
            progressbar=False,
        )

    # Check we got progress updates
    events = []
    while not queue.empty():
        events.append(queue.get())

    assert len(events) > 0, "Callback should have been called"
    # Each event is (chain_id, ChainProgress)
    chain_id, last_progress = events[-1]
    assert chain_id == 0
    assert last_progress.draw > 0
    assert last_progress.draws_per_sec > 0
    print(f"Captured {len(events)} progress events")
    print(f"Final: draw={last_progress.draw}, phase={last_progress.phase}, "
          f"dps={last_progress.draws_per_sec:.1f}")


def test_decode_progress_event_round_trips_each_type():
    """Raw msgpack-style dicts decode into the matching typed events."""
    from cloudposterior.progress import (
        ConvergenceUpdate,
        PhaseUpdate,
        SamplingProgress,
        decode_progress_event,
    )

    ph = decode_progress_event(
        {"type": "phase", "phase": "sampling", "status": "in_progress",
         "message": "go", "elapsed": 1.5}
    )
    assert isinstance(ph, PhaseUpdate) and ph.message == "go"

    sp = decode_progress_event(
        {"type": "sampling",
         "chains": {"0": {"draw": 5, "total": 10, "phase": "sampling"}},
         "total_divergences": 2, "elapsed": 1.0, "total_draws": 5}
    )
    assert isinstance(sp, SamplingProgress)
    assert sp.chains[0].draw == 5 and sp.total_divergences == 2  # chain id coerced to int

    cv = decode_progress_event(
        {"type": "convergence",
         "params": {"mu": {"rhat": 1.01, "ess_bulk": 400, "ess_tail": 380}},
         "draws": 100}
    )
    assert isinstance(cv, ConvergenceUpdate) and cv.params["mu"].rhat == 1.01

    assert decode_progress_event({"type": "result"}) is None


def test_dispatch_event_routes_and_tolerates_missing_methods():
    """dispatch_event calls the right sink method and skips convergence when absent."""
    from cloudposterior.progress import (
        ConvergenceUpdate,
        JobPhase,
        PhaseUpdate,
        SamplingProgress,
        dispatch_event,
    )

    class FullSink:
        def __init__(self):
            self.calls = []

        def show_phase(self, e):
            self.calls.append("phase")

        def show_sampling(self, e):
            self.calls.append("sampling")

        def show_convergence(self, e):
            self.calls.append("convergence")

    class NoConvSink(FullSink):
        show_convergence = property()  # remove

    full = FullSink()
    dispatch_event(PhaseUpdate(JobPhase.SAMPLING, "in_progress", "x", 0.0), [full])
    dispatch_event(SamplingProgress(chains={}), [full])
    dispatch_event(ConvergenceUpdate(params={}), [full])
    assert full.calls == ["phase", "sampling", "convergence"]

    # a sink lacking show_convergence must simply be skipped, not error
    class PhaseOnly:
        def __init__(self):
            self.calls = []

        def show_phase(self, e):
            self.calls.append("phase")

        def show_sampling(self, e):
            self.calls.append("sampling")

    bare = PhaseOnly()
    dispatch_event(ConvergenceUpdate(params={}), [bare])  # no-op, no crash
    assert bare.calls == []
    dispatch_event(None, [bare])  # None is ignored
    assert bare.calls == []
