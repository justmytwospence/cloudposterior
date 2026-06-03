"""Remote envs are kept warm after the `with` block: registered, reused, torn down."""

import pytest


def _model():
    import pymc as pm

    with pm.Model() as m:
        pm.Normal("x", 0, 1)
    return m


def _fake_backend(monkeypatch, counter):
    from cloudposterior.backends import modal_backend as mb

    class FakeEnv:
        _dashboard_fn = object()  # looks dashboard-capable -> reuse is allowed

        def __init__(self):
            self.torn = False
            self.config = None

        def teardown(self):
            self.torn = True

    class FakeBackend:
        def __init__(self, *a, **k):
            pass

        def provision(self, *a, **k):
            counter["n"] += 1
            return FakeEnv()

    monkeypatch.setattr(mb, "ModalBackend", FakeBackend)


def test_env_kept_warm_reused_and_torn_down(monkeypatch):
    pytest.importorskip("pymc")
    import cloudposterior as cp
    from cloudposterior import api

    api._LIVE_ENVS.clear()
    counter = {"n": 0}
    _fake_backend(monkeypatch, counter)
    m = _model()
    key = api._env_key("kal-test", m)

    try:
        # First provision registers a kept-warm env (no teardown on exit).
        c1 = cp.cloud(m, remote=True, project="kal-test")
        c1._model_bytes = b"x"
        c1._provision_environment("nutpie", {})
        env1 = c1._env
        assert counter["n"] == 1
        assert api._LIVE_ENVS.get(key) is env1
        assert env1.torn is False

        # A second cloud for the same model reuses it -- no new provision.
        c2 = cp.cloud(m, remote=True, project="kal-test")
        c2._model_bytes = b"x"
        c2._provision_environment("nutpie", {})
        assert c2._env is env1
        assert counter["n"] == 1

        # Explicit teardown stops it and unregisters.
        api._teardown_live_envs("kal-test")
        assert env1.torn is True
        assert key not in api._LIVE_ENVS
    finally:
        api._LIVE_ENVS.clear()


def test_can_reuse_env_gates_on_gpu_for_jax_samplers():
    """A warm CPU env must not be reused for numpyro/blackjax (its image has no
    jax); GPU/CPU-default samplers reuse it fine."""
    import pymc as pm

    import cloudposterior as cp
    from cloudposterior.config import RemoteConfig

    with pm.Model() as m:
        pm.Normal("x", 0, 1)
    ctx = cp.cloud(m, remote=True)

    class CpuEnv:
        _dashboard_fn = object()           # has a dashboard
        config = RemoteConfig(gpu=None)    # CPU image -> no jax

    class GpuEnv:
        _dashboard_fn = object()
        config = RemoteConfig(gpu="A10G")  # jax-equipped

    class NoCfgEnv:
        _dashboard_fn = object()           # e.g. a cp.map env (no .config)

    cpu = CpuEnv()
    assert ctx._can_reuse_env(cpu, "nutpie") is True
    assert ctx._can_reuse_env(cpu, "pymc") is True
    assert ctx._can_reuse_env(cpu, "numpyro") is False   # CPU image lacks jax
    assert ctx._can_reuse_env(cpu, "blackjax") is False
    assert ctx._can_reuse_env(GpuEnv(), "numpyro") is True
    assert ctx._can_reuse_env(NoCfgEnv(), "numpyro") is False  # conservative
