"""cp.map fan-out + live dashboard wiring (Modal mocked at the boundary)."""

import numpy as np
import pytest


def _fake_modal(monkeypatch):
    import arviz as az

    import cloudposterior.api as api
    from cloudposterior.backends import modal_backend as mb
    from cloudposterior.serialize import serialize_inference_data

    api._LIVE_ENVS.clear()  # avoid warm-env leakage across tests
    state = {"captured": [], "spawned": [], "envs": [], "provision_kwargs": []}
    counter = {"i": 0}

    class FakeCall:
        def __init__(self, tag):
            self._tag = tag

        def get(self):
            state["captured"].append(self._tag)
            # tag each result by spawn order so the caller can verify ordering
            return serialize_inference_data(
                az.from_dict(posterior={"x": np.full((2, 5), float(self._tag))})
            )

    class FakeMethod:
        def spawn(self, payload_path, kw, sampler, progress_dict_name=None,
                  progress_key=None, stop_dict_name=None):
            tag = counter["i"]
            counter["i"] += 1
            state["spawned"].append({
                "payload_path": payload_path,
                "progress_dict_name": progress_dict_name,
                "progress_key": progress_key,
                "stop_dict_name": stop_dict_name,
            })
            return FakeCall(tag)

    class FakeSampler:
        sample_blocking = FakeMethod()

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
