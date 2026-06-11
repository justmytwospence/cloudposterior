"""Image pip-spec construction, including the JAX-sampler wiring (no Modal)."""

from unittest.mock import patch

from cloudposterior.backends.modal_backend import _build_pip_specs
from cloudposterior.config import RemoteConfig


def _manifest():
    return {"python": "3.12.1", "pymc": "5.20.0", "numpy": "2.1.0"}


def test_cpu_image_installs_nutpie_by_default():
    specs = _build_pip_specs(_manifest(), gpu=None)
    assert any(s.startswith("nutpie") for s in specs)
    assert not any("jax" in s for s in specs)


def test_cpu_image_installs_jax_for_jax_samplers():
    """A CPU preset + nuts_sampler="numpyro" must still get numpyro/jax
    (regression: callers never passed nuts_sampler, so this branch was dead
    and the remote run died on ImportError)."""
    for sampler in ("numpyro", "blackjax"):
        specs = _build_pip_specs(_manifest(), gpu=None, nuts_sampler=sampler)
        assert "numpyro" in specs, sampler
        assert "jax" in specs, sampler


def test_gpu_image_installs_cuda_jax():
    specs = _build_pip_specs(_manifest(), gpu="A10G")
    assert any(s.startswith("jax[cuda12]") for s in specs)
    assert "numpyro" in specs


def test_create_modal_app_threads_nuts_sampler(monkeypatch):
    """_create_modal_app passes its nuts_sampler to _build_pip_specs."""
    from cloudposterior.backends import modal_backend as mb

    captured = {}

    def fake_specs(manifest, gpu=None, nuts_sampler="pymc"):
        captured["nuts_sampler"] = nuts_sampler
        return ["pymc==5.20.0"]

    class FakeImage:
        @staticmethod
        def debian_slim(python_version=None):
            return FakeImage()

        def uv_pip_install(self, specs):
            return self

        def add_local_python_source(self, name):
            return self

    class FakeApp:
        def __init__(self, name):
            pass

        def function(self, **kwargs):
            return lambda fn: fn

    import types

    fake_modal = types.SimpleNamespace(Image=FakeImage, App=FakeApp)
    with patch.dict("sys.modules", {"modal": fake_modal}):
        monkeypatch.setattr(mb, "_build_pip_specs", fake_specs)
        mb._create_modal_app(_manifest(), RemoteConfig(), "numpyro")

    assert captured["nuts_sampler"] == "numpyro"
