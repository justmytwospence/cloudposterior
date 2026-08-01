"""The public surface that had no tests: cp.sample, teardown, cleanup."""

import numpy as np
import pymc as pm
import pytest

import cloudposterior as cp
from cloudposterior import api


def _model():
    with pm.Model() as m:
        mu = pm.Normal("mu", 0, 1)
        pm.Normal("obs", mu, 1, observed=np.array([1.0, 2.0]))
    return m


def test_public_exports_are_importable():
    for name in cp.__all__:
        assert hasattr(cp, name), name


def test_cache_backends_are_exported():
    """resolve_cache's docstring advertises passing a CacheBackend instance,
    but they were only reachable via cloudposterior.cache."""
    assert cp.DiskCache is not None and cp.MemoryCache is not None


# -- cp.sample ---------------------------------------------------------------

def test_cp_sample_forwards_only_the_kwargs_it_was_given(monkeypatch):
    """chains/cores are splatted conditionally; passing None through would
    change what reaches pm.sample."""
    seen = {}

    def fake_run_sample(**kwargs):
        seen.update(kwargs)
        return "idata"

    monkeypatch.setattr(api, "_run_sample", fake_run_sample)
    assert cp.sample(_model(), draws=50) == "idata"

    assert seen["draws"] == 50 and seen["remote"] is True
    assert "chains" not in seen and "cores" not in seen


def test_cp_sample_passes_chains_when_given(monkeypatch):
    seen = {}
    monkeypatch.setattr(api, "_run_sample", lambda **kw: seen.update(kw))
    cp.sample(_model(), chains=2, cores=1)
    assert seen["chains"] == 2 and seen["cores"] == 1


def test_cp_sample_inside_a_block_does_not_take_the_interceptor_as_original(
    monkeypatch,
):
    """pm.sample is patched inside a cp.cloud block; using it as the "original"
    would recurse through the interceptor."""
    seen = {}
    monkeypatch.setattr(api, "_run_sample", lambda **kw: seen.update(kw))

    model = _model()
    with cp.cloud(model):
        patched = pm.sample
        cp.sample(model, draws=10)

    assert seen["original_sample"] is not patched


# -- teardown ----------------------------------------------------------------

def test_destroy_only_tears_down_its_own_model(monkeypatch):
    """destroy() used to stop every warm env in the project and delete the
    shared volume, discarding other models' payloads."""
    torn = []

    class FakeEnv:
        def __init__(self, tag):
            self.tag = tag
            self.config = None

        def teardown(self):
            torn.append(self.tag)

    mine = _model()
    with pm.Model(name="other_model") as other:
        pm.Normal("z", 0, 1)

    session = cp.cloud(mine, remote=True, project="proj-x")
    api._LIVE_ENVS[api._env_key("proj-x", mine)] = FakeEnv("mine")
    api._LIVE_ENVS[api._env_key("proj-x", other)] = FakeEnv("other")

    deleted = []
    from cloudposterior.backends import modal_backend as mb

    monkeypatch.setattr(
        mb.ModalBackend, "cleanup_volumes", staticmethod(lambda project: deleted.append(project))
    )

    session.destroy()

    assert torn == ["mine"]
    assert deleted == [], "volume deletion is project-wide, so it must be opt-in"


def test_destroy_can_delete_the_volume_when_asked(monkeypatch):
    deleted = []
    from cloudposterior.backends import modal_backend as mb

    monkeypatch.setattr(
        mb.ModalBackend, "cleanup_volumes", staticmethod(lambda project: deleted.append(project))
    )
    cp.cloud(_model(), remote=True, project="proj-y").destroy(delete_volume=True)
    assert deleted == ["proj-y"]


def test_cleanup_volumes_surfaces_failures(monkeypatch):
    """A swallowed failure looked exactly like a successful cleanup."""
    from cloudposterior.backends import modal_backend as mb

    class FakeVolume:
        class objects:
            @staticmethod
            def delete(name):
                raise RuntimeError("permission denied")

    fake_modal = type("M", (), {"Volume": FakeVolume})
    monkeypatch.setitem(__import__("sys").modules, "modal", fake_modal)

    with pytest.warns(UserWarning, match="could not delete volume"):
        mb.ModalBackend.cleanup_volumes(project="nope")


def test_cleanup_volumes_is_quiet_when_already_gone(monkeypatch):
    import warnings

    from cloudposterior.backends import modal_backend as mb

    class FakeVolume:
        class objects:
            @staticmethod
            def delete(name):
                raise RuntimeError("Volume not found")

    fake_modal = type("M", (), {"Volume": FakeVolume})
    monkeypatch.setitem(__import__("sys").modules, "modal", fake_modal)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        mb.ModalBackend.cleanup_volumes(project="already-gone")
