"""Test progress tracking with a real PyMC model."""

import numpy as np
import pymc as pm

from queue import Queue
from cloudposterior.progress import make_sampling_callback, ProgressAggregator


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
        idata = pm.sample(
            draws=10,
            tune=10,
            chains=1,
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
