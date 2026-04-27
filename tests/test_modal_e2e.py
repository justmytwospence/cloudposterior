"""End-to-end tests against real Modal infrastructure.

Marked ``@pytest.mark.modal`` so they're SKIPPED by default. Pass
``--run-modal`` to opt in. Each test costs a small amount of Modal
credit (a few cents). Tests use the smallest possible model, instance,
and sampling config to keep costs minimal.

To run::

    uv run pytest tests/test_modal_e2e.py -v --run-modal
"""

import time
import uuid

import numpy as np
import pymc as pm
import pytest

import cloudposterior as cp


pytestmark = pytest.mark.modal


def _tiny_model():
    """Two free RVs, five observations -- the cheapest interesting model."""
    with pm.Model(name="tiny_e2e") as model:
        mu = pm.Normal("mu", 0, 1)
        sigma = pm.HalfNormal("sigma", 1)
        pm.Normal("obs", mu, sigma, observed=np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
    return model


@pytest.fixture
def isolated_project():
    """Each test gets a unique Modal Volume so it can't clash with the user's
    real volumes. Best-effort cleanup at teardown."""
    name = f"test-{uuid.uuid4().hex[:8]}"
    yield name
    try:
        cp.cleanup_volumes(project=name)
    except Exception:
        pass


def test_cloud_remote_end_to_end(isolated_project):
    """cp.cloud(model, remote=True) runs a full sample on Modal and returns
    a valid InferenceData with the requested shape."""
    model = _tiny_model()
    with cp.cloud(
        model,
        remote=True,
        cache=False,
        instance="small",
        progress=False,
        project=isolated_project,
    ):
        idata = pm.sample(draws=20, tune=20, chains=2, progressbar=False)

    assert "posterior" in idata.groups()
    assert "mu" in idata.posterior
    assert idata.posterior.mu.shape == (2, 20)
    assert "sigma" in idata.posterior
    # Sampled values should be finite (not NaN, not Inf)
    assert np.isfinite(idata.posterior.mu.values).all()


def test_remote_cache_hit_skips_modal_call(isolated_project):
    """A second pm.sample with identical kwargs in the same cp.cloud block
    must hit the in-memory cache and skip Modal entirely (sub-second)."""
    model = _tiny_model()
    with cp.cloud(
        model,
        remote=True,
        cache=True,
        instance="small",
        progress=False,
        project=isolated_project,
    ):
        idata1 = pm.sample(draws=20, tune=20, chains=2, random_seed=0, progressbar=False)
        t0 = time.time()
        idata2 = pm.sample(draws=20, tune=20, chains=2, random_seed=0, progressbar=False)
        elapsed = time.time() - t0

    assert elapsed < 1.0, (
        f"cache hit must be sub-second; took {elapsed:.2f}s -- did Modal run again?"
    )
    np.testing.assert_array_equal(
        idata1.posterior.mu.values, idata2.posterior.mu.values,
    )


def test_persistent_container_reuses_warm_vm(isolated_project):
    """Two pm.sample calls in the same cp.cloud block should reuse the same
    warm container. The second call has no provisioning or upload cost, so
    it should finish substantially faster than the first."""
    model = _tiny_model()
    with cp.cloud(
        model,
        remote=True,
        cache=False,
        instance="small",
        progress=False,
        project=isolated_project,
    ):
        t0 = time.time()
        pm.sample(draws=20, tune=20, chains=2, random_seed=1, progressbar=False)
        first = time.time() - t0

        t0 = time.time()
        pm.sample(draws=20, tune=20, chains=2, random_seed=2, progressbar=False)
        second = time.time() - t0

    assert second < first, (
        f"warm-container reuse should be faster than cold start: "
        f"first={first:.1f}s, second={second:.1f}s"
    )
