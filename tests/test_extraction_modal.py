"""Bounded CPU-only extraction smoke. NEVER run without explicit paid-test approval.

Only ``--run-modal`` opts these tests in. Each test owns a unique project; the
fixture destroys only that test's warm environment, Dict and Volume.
"""

import uuid

import numpy as np
import pymc as pm
import pytest

import cloudposterior as cp
from cloudposterior import api
from cloudposterior.config import RemoteConfig

pytestmark = pytest.mark.modal


@pytest.fixture
def isolated_cpu_project(monkeypatch):
    project = f"extraction-{uuid.uuid4().hex[:12]}"
    # Override auto-sizing in tests, never add a production instance preset.
    monkeypatch.setattr(RemoteConfig, "from_instance", classmethod(
        lambda cls, *args, **kwargs: cls(cpu=2, memory=4096, timeout=180,
                                         idle_timeout=30)))
    try:
        yield project
    finally:
        # cleanup_volumes first tears down only this project's kept-warm apps
        # and dashboard Dict, then deletes its own project-scoped Volume.
        cp.cleanup_volumes(project=project)


def _model():
    with pm.Model() as model:
        mu = pm.Normal("mu", 0, 1)
        pm.Normal("obs", mu, 1, observed=np.array([0.5, -0.2, 0.1]))
    return model


def test_real_modal_sample_monitor_wire_and_cache_hit(isolated_cpu_project, monkeypatch):
    model = _model()
    seed = int(isolated_cpu_project.rsplit("-", 1)[-1][:8], 16)
    session = cp.cloud(model, remote=True, cache=True, dashboard=False,
                       progress=False, project=isolated_cpu_project)
    with session:
        fixed = pm.sample(draws=30, tune=40, chains=2, cores=2,
                          nuts_sampler="pymc", random_seed=seed, progressbar=False)
        assert fixed.posterior.mu.shape == (2, 30)
        assert np.isfinite(fixed.posterior.mu.values).all()
        assert fixed.posterior.attrs["pymc_monitor_convergence"] == (
            fixed.posterior.attrs["cloudposterior_convergence"])
        original_env = session._env
        def unexpected_serialize(model):
            raise AssertionError("a cache hit attempted cloud serialization")

        with monkeypatch.context() as guard:
            guard.setattr(api, "_ensure_model_bytes", unexpected_serialize)
            cached = pm.sample(draws=30, tune=40, chains=2, cores=2,
                               nuts_sampler="pymc", random_seed=seed, progressbar=False)
        np.testing.assert_array_equal(cached.posterior.mu.values, fixed.posterior.mu.values)
        assert session._env is original_env



def test_real_modal_map_monitor_and_manifest(isolated_cpu_project, monkeypatch):
    from cloudposterior.backends.modal_backend import _run_blocking

    models = [_model(), _model()]
    models[0].name = "fixed"
    models[1].name = "adaptive"
    seed = int(isolated_cpu_project.rsplit("-", 1)[-1][:8], 16)
    fixed_kwargs = {"draws": 20, "tune": 20, "chains": 2, "cores": 2,
                    "nuts_sampler": "nutpie", "random_seed": seed, "progressbar": False}
    adaptive_target = {"ess": 30, "r_hat": 1.2, "patience": 1}
    adaptive_kwargs = {"draws": 200, "tune": 100, "chains": 2, "cores": 2,
                       "nuts_sampler": "nutpie", "random_seed": seed + 1,
                       "until": adaptive_target, "progressbar": False}
    results = cp.map(models, [fixed_kwargs, adaptive_kwargs],
                     project=isolated_cpu_project, cache=True, dashboard=True,
                     progress=False)
    assert len(results) == 2
    for idata in results:
        assert np.isfinite(idata.posterior.mu.values).all()
        assert idata.posterior.attrs["pymc_monitor_convergence"] == (
            idata.posterior.attrs["cloudposterior_convergence"])
    fixed_report, adaptive_report = map(cp.convergence.read_report, results)
    assert fixed_report["status"] == "complete"
    assert adaptive_report["status"] in {"met", "cap", "futile"}, adaptive_report
    assert adaptive_report["target"]["ess"] == adaptive_target["ess"]

    from cloudposterior.naming import model_slug

    env = api._LIVE_ENVS[(isolated_cpu_project, model_slug(models[0]))]
    store = env._dashboard_dict
    manifest = _run_blocking(store.__getitem__, "models")
    assert len(manifest) == 2
    for entry in manifest:
        state = _run_blocking(store.__getitem__, entry["label"])
        assert state["complete"] is True

    with monkeypatch.context() as guard:
        guard.setattr(api, "_ensure_model_bytes", lambda _: pytest.fail("cache hit serialized"))
        again = cp.map([models[0]], [fixed_kwargs], project=isolated_cpu_project,
                       cache=True, dashboard=False, progress=False)
    np.testing.assert_array_equal(again[0].posterior.mu.values, results[0].posterior.mu.values)


def test_real_modal_smc_wire_and_cache_hit(isolated_cpu_project, monkeypatch):
    model = _model()
    seed = int(isolated_cpu_project.rsplit("-", 1)[-1][:8], 16)
    with cp.cloud(model, remote=True, cache=True, dashboard=False,
                  progress=False, project=isolated_cpu_project):
        first = pm.sample_smc(draws=20, chains=1, cores=1, random_seed=seed,
                              progressbar=False)
        assert np.isfinite(first.posterior.mu.values).all()
        assert first.sample_stats is not None
        with monkeypatch.context() as guard:
            guard.setattr(api, "_ensure_model_bytes", lambda _: pytest.fail("cache hit serialized"))
            again = pm.sample_smc(draws=20, chains=1, cores=1, random_seed=seed,
                                  progressbar=False)
        np.testing.assert_array_equal(again.posterior.mu.values, first.posterior.mu.values)


def test_real_modal_dashboard_progress_cors_and_stop():
    from _modal_stop_worker import assert_stop_outcome, supervise

    # Do not use isolated_cpu_project: its unbounded finally/teardown can hang
    # pytest even after a watchdog failure. The supervisor owns precise cleanup.
    assert_stop_outcome(supervise())
