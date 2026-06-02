"""cp.map fan-out: results returned in input order (Modal mocked at the boundary)."""

import numpy as np
import pytest


def _fake_modal(monkeypatch, captured):
    import arviz as az

    from cloudposterior.backends import modal_backend as mb
    from cloudposterior.serialize import serialize_inference_data

    counter = {"i": 0}

    class FakeCall:
        def __init__(self, tag):
            self._tag = tag

        def get(self):
            captured.append(self._tag)
            # tag each result by spawn order so the caller can verify ordering
            return serialize_inference_data(
                az.from_dict(posterior={"x": np.full((2, 5), float(self._tag))})
            )

    class FakeMethod:
        def spawn(self, payload_path, kw, sampler):
            tag = counter["i"]
            counter["i"] += 1
            return FakeCall(tag)

    class FakeSampler:
        sample_blocking = FakeMethod()

    class FakeEnv:
        _model_slug = "m"
        _sampler_cls = FakeSampler

        def _ensure_running(self):
            pass

        def _upload_if_needed(self, *a, **k):
            return True

        def teardown(self):
            pass

    class FakeBackend:
        def __init__(self, *a, **k):
            pass

        def provision(self, *a, **k):
            return FakeEnv()

    monkeypatch.setattr(mb, "ModalBackend", FakeBackend)
    monkeypatch.setattr(mb, "_compute_payload_path", lambda *a, **k: "p")
    monkeypatch.setattr(mb, "_run_blocking", lambda fn, *a, **k: fn(*a, **k))


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

    captured = []
    _fake_modal(monkeypatch, captured)
    out = cp.map(_models(3), {"draws": 10}, cache=False)
    # fake tags results by spawn order -> 0,1,2 returned in input order
    assert [float(o.posterior["x"].values.flat[0]) for o in out] == [0.0, 1.0, 2.0]
    assert len(captured) == 3


def test_cp_map_kwargs_list_must_align(monkeypatch):
    pytest.importorskip("pymc")
    import cloudposterior as cp

    _fake_modal(monkeypatch, [])
    with pytest.raises(ValueError):
        cp.map(_models(3), [{"draws": 10}, {"draws": 20}], cache=False)  # 2 != 3


def test_cp_map_empty_list(monkeypatch):
    pytest.importorskip("pymc")
    import cloudposterior as cp

    assert cp.map([], cache=False) == []
