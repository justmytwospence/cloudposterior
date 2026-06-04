"""cp.map fan-out + live dashboard wiring (Modal mocked at the boundary)."""

import numpy as np
import pytest


def _posterior_idata(values):
    """Minimal InferenceData with a posterior group, across arviz majors.

    arviz 1.x dropped from_dict's per-group kwargs and made InferenceData a
    DataTree, so build the group via dict_to_dataset + the add_group shim."""
    import arviz as az

    from cloudposterior._idata import add_group

    idata = az.InferenceData()
    add_group(idata, "posterior", az.dict_to_dataset(values))
    return idata


def _fake_modal(monkeypatch):

    import cloudposterior.api as api
    from cloudposterior.backends import modal_backend as mb
    from cloudposterior.serialize import serialize_inference_data

    api._LIVE_ENVS.clear()  # avoid warm-env leakage across tests
    state = {"captured": [], "spawned": [], "envs": [], "provision_kwargs": [], "sizes": []}
    counter = {"i": 0}

    class FakeCall:
        def __init__(self, tag):
            self._tag = tag

        def get(self):
            state["captured"].append(self._tag)
            # tag each result by spawn order so the caller can verify ordering
            return serialize_inference_data(
                _posterior_idata({"x": np.full((2, 5), float(self._tag))})
            )

    class FakeMethod:
        def spawn(self, payload_path, kw, sampler, progress_dict_name=None,
                  progress_key=None, stop_dict_name=None):
            tag = counter["i"]
            counter["i"] += 1
            state["spawned"].append({
                "payload_path": payload_path,
                "kw": kw,
                "progress_dict_name": progress_dict_name,
                "progress_key": progress_key,
                "stop_dict_name": stop_dict_name,
            })
            return FakeCall(tag)

    class FakeSampler:
        sample_blocking = FakeMethod()

        def __init__(self, *a, **k):
            pass

        @classmethod
        def with_options(cls, *, cpu=None, memory=None):
            state["sizes"].append((cpu, memory))   # per-spawn resource sizing
            return cls

    class FakeEnv:
        _model_slug = "m"
        _sampler_cls = FakeSampler

        def __init__(self, dashboard):
            self.torn_down = False
            self._dashboard_dict_name = "cp-dash-test" if dashboard else None
            self._dashboard_dict = {} if dashboard else None
            self._dashboard_fn = None      # keeps _show_link out of the tests
            self._dashboard_url = None

        def _ensure_running(self):
            pass

        def _upload_if_needed(self, *a, **k):
            return True

        def teardown(self):
            self.torn_down = True

    class FakeBackend:
        def __init__(self, *a, **k):
            pass

        def provision(self, *a, **k):
            state["provision_kwargs"].append(k)
            env = FakeEnv(dashboard=k.get("dashboard", False))
            state["envs"].append(env)
            return env

    monkeypatch.setattr(mb, "ModalBackend", FakeBackend)
    monkeypatch.setattr(mb, "_compute_payload_path", lambda *a, **k: "p")
    monkeypatch.setattr(mb, "_run_blocking", lambda fn, *a, **k: fn(*a, **k))
    return state


def _models(n):
    import pymc as pm

    out = []
    for i in range(n):
        with pm.Model() as m:
            pm.Normal("x", 0, float(i + 1))  # distinct models
        out.append(m)
    return out


def test_cp_map_runs_all_in_input_order(monkeypatch):
    pytest.importorskip("pymc")
    import cloudposterior as cp

    state = _fake_modal(monkeypatch)
    out = cp.map(_models(3), {"draws": 10}, cache=False)
    # fake tags results by spawn order -> 0,1,2 returned in input order
    assert [float(o.posterior["x"].values.flat[0]) for o in out] == [0.0, 1.0, 2.0]
    assert len(state["captured"]) == 3


def test_cp_map_kwargs_list_must_align(monkeypatch):
    pytest.importorskip("pymc")
    import cloudposterior as cp

    _fake_modal(monkeypatch)
    with pytest.raises(ValueError):
        cp.map(_models(3), [{"draws": 10}, {"draws": 20}], cache=False)  # 2 != 3


def test_cp_map_empty_list(monkeypatch):
    pytest.importorskip("pymc")
    import cloudposterior as cp

    assert cp.map([], cache=False) == []


def test_cp_map_dashboard_on_by_default(monkeypatch):
    """Dashboard is provisioned, a manifest is published, and each spawn is wired
    to its own progress + stop keys; the env is kept warm afterward."""
    pytest.importorskip("pymc")
    import cloudposterior as cp
    import cloudposterior.api as api

    state = _fake_modal(monkeypatch)
    cp.map(_models(2), {"draws": 10}, cache=False)

    assert state["provision_kwargs"][0]["dashboard"] is True
    env = state["envs"][0]
    # one unique label per model, in input order
    manifest = env._dashboard_dict["models"]
    labels = [m["label"] for m in manifest]
    assert len(labels) == 2 and len(set(labels)) == 2
    # each spawn got its own progress key + the shared dict / stop names
    assert [s["progress_key"] for s in state["spawned"]] == labels
    assert all(s["progress_dict_name"] == "cp-dash-test" for s in state["spawned"])
    assert all(s["stop_dict_name"] == "cp-dash-test" for s in state["spawned"])
    # stale stop flags cleared at start
    assert env._dashboard_dict["stop"] is False
    # kept warm (not torn down) so the dashboard stays browsable
    assert env.torn_down is False
    assert env in api._LIVE_ENVS.values()


def test_cp_map_dashboard_false_opts_out(monkeypatch):
    """dashboard=False provisions no control Dict, wires no progress, tears down."""
    pytest.importorskip("pymc")
    import cloudposterior as cp
    import cloudposterior.api as api

    state = _fake_modal(monkeypatch)
    cp.map(_models(2), {"draws": 10}, cache=False, dashboard=False)

    assert state["provision_kwargs"][0]["dashboard"] is False
    env = state["envs"][0]
    assert env._dashboard_dict is None
    assert all(s["progress_dict_name"] is None for s in state["spawned"])
    assert all(s["progress_key"] is None for s in state["spawned"])
    assert all(s["stop_dict_name"] is None for s in state["spawned"])
    assert env.torn_down is True
    assert env not in api._LIVE_ENVS.values()


def test_cp_map_all_cached_skips_provisioning(monkeypatch):
    """When every model is a local cache hit, no Modal env / dashboard is spun up."""
    pytest.importorskip("pymc")
    import cloudposterior as cp
    import cloudposterior.api as api

    state = _fake_modal(monkeypatch)

    class AllHitCache:
        def load(self, key, **kw):
            return _posterior_idata({"x": np.zeros((2, 5))})

        def save(self, key, idata, **kw):
            raise AssertionError("save should not run when everything is cached")

    out = cp.map(_models(3), {"draws": 10}, cache=AllHitCache())
    assert len(out) == 3 and all(o is not None for o in out)
    # zero Modal work
    assert state["provision_kwargs"] == []
    assert state["spawned"] == []
    assert state["envs"] == []
    assert api._LIVE_ENVS == {}


def test_cp_map_partial_cache_provisions_and_spawns_only_misses(monkeypatch):
    """A partial cache still provisions, but spawns only the misses; the manifest
    lists all models (cached ones shown as complete)."""
    pytest.importorskip("pymc")
    import cloudposterior as cp

    state = _fake_modal(monkeypatch)

    class FirstHitCache:
        def __init__(self):
            self.calls = 0

        def load(self, key, **kw):
            self.calls += 1
            return _posterior_idata({"x": np.zeros((2, 5))}) if self.calls == 1 else None

        def save(self, key, idata, **kw):
            pass

    out = cp.map(_models(3), {"draws": 10}, cache=FirstHitCache())
    assert len(out) == 3 and all(o is not None for o in out)
    assert len(state["provision_kwargs"]) == 1       # provisioned once
    assert len(state["spawned"]) == 2                # only the two misses
    env = state["envs"][0]
    manifest = env._dashboard_dict["models"]
    assert len(manifest) == 3                         # manifest still covers all
    # the cached model (input index 0) gets a complete panel; misses are written
    # by the (faked) workers, not the client
    cached_label = manifest[0]["label"]
    assert env._dashboard_dict[cached_label]["complete"] is True
    complete_panels = [
        k for k, v in env._dashboard_dict.items()
        if isinstance(v, dict) and v.get("complete") is True
    ]
    assert len(complete_panels) == 1


def test_cp_map_until_true_injects_normalized_target(monkeypatch):
    """until=True is normalized to the Vehtari dict and threaded into every fit
    (no bare True reaching the worker)."""
    pytest.importorskip("pymc")
    import cloudposterior as cp

    state = _fake_modal(monkeypatch)
    cp.map(_models(2), {"draws": 10}, cache=False, until=True)
    assert len(state["spawned"]) == 2
    for s in state["spawned"]:
        assert s["kw"]["until"] == {"r_hat": 1.01, "ess": 400}


def test_cp_map_until_dict_merges_over_defaults(monkeypatch):
    pytest.importorskip("pymc")
    import cloudposterior as cp

    state = _fake_modal(monkeypatch)
    cp.map(_models(1), {"draws": 10}, cache=False, until={"ess": 1000})
    assert state["spawned"][0]["kw"]["until"] == {"r_hat": 1.01, "ess": 1000}


def test_cp_map_until_warns_and_skips_for_jax_samplers(monkeypatch):
    pytest.importorskip("pymc")
    import cloudposterior as cp

    state = _fake_modal(monkeypatch)
    with pytest.warns(UserWarning, match="nutpie or pymc"):
        cp.map(_models(2), {"draws": 10}, cache=False,
               nuts_sampler="numpyro", until=True)
    assert all("until" not in s["kw"] for s in state["spawned"])


def test_cp_map_until_in_sample_kwargs_is_normalized(monkeypatch):
    """A raw until left in sample_kwargs is normalized too (no True.get crash)."""
    pytest.importorskip("pymc")
    import cloudposterior as cp

    state = _fake_modal(monkeypatch)
    cp.map(_models(1), {"draws": 10, "until": True}, cache=False)
    assert state["spawned"][0]["kw"]["until"] == {"r_hat": 1.01, "ess": 400}


def test_cp_map_overwrite_forces_rerun_and_saves(monkeypatch):
    """overwrite=True ignores a matching cache entry: every model re-runs and is
    re-saved (load is never consulted)."""
    pytest.importorskip("pymc")
    import cloudposterior as cp

    state = _fake_modal(monkeypatch)

    class RecordingCache:
        def __init__(self):
            self.loads = 0
            self.saved = 0

        def load(self, key, **kw):
            self.loads += 1
            return _posterior_idata({"x": np.zeros((2, 5))})  # would hit

        def save(self, key, idata, **kw):
            self.saved += 1

    rc = RecordingCache()
    out = cp.map(_models(2), {"draws": 10}, cache=rc, overwrite=True)
    assert len(out) == 2
    assert rc.loads == 0                 # never loaded
    assert len(state["spawned"]) == 2    # both re-run
    assert rc.saved == 2                 # both overwritten


def test_cp_map_sizes_each_model_independently(monkeypatch):
    """Each spawned fit is sized to its own model (with_options), not models[0]."""
    pytest.importorskip("pymc")
    import cloudposterior as cp

    state = _fake_modal(monkeypatch)
    # same model, but the second asks for more chains -> more cores; auto-size
    # must reflect each model's own kwargs.
    cp.map(
        _models(2),
        [{"draws": 10, "chains": 4}, {"draws": 10, "chains": 16}],
        cache=False,
    )
    assert [cpu for cpu, _mem in state["sizes"]] == [4, 16]
