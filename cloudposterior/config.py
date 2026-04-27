from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class RemoteConfig:
    """Configuration for remote sampling."""

    cpu: float = 8.0
    memory: int = 16384  # MB
    timeout: int = 3600  # seconds
    gpu: str | None = None
    auto_sized: bool = False  # True if config was auto-determined
    idle_timeout: int = 1200  # seconds before idle container is torn down (Modal max)

    @classmethod
    def from_instance(
        cls,
        instance: str | None,
        model=None,
        sample_kwargs: dict | None = None,
        nuts_sampler: str = "pymc",
    ) -> RemoteConfig:
        """Resolve resource config from instance hint, or auto-size from the model.

        If instance is a preset name ("small", "large", etc.), use the preset.
        If instance is None, auto-size based on model and sampling config.
        """
        if instance is not None:
            presets = {
                "small": cls(cpu=4, memory=8192),
                "medium": cls(cpu=8, memory=16384),
                "large": cls(cpu=16, memory=32768),
                "xlarge": cls(cpu=32, memory=65536),
                "gpu": cls(cpu=8, memory=16384, gpu="A100"),
            }
            if instance in presets:
                return presets[instance]
            return cls()

        # Auto-size from model + sampling config
        if model is not None and sample_kwargs is not None:
            return cls._auto(model, sample_kwargs, nuts_sampler)

        return cls()

    @classmethod
    def _auto(cls, model, sample_kwargs: dict, nuts_sampler: str = "pymc") -> RemoteConfig:
        """Estimate optimal resources from model characteristics.

        CPU honors both ``chains`` and ``cores`` (PyMC defaults ``cores=chains``,
        but a user passing ``cores`` explicitly should not be oversubscribed).
        Memory scales with the observed-data footprint plus the in-memory
        posterior trace size (chains x draws x parameter count).
        """
        import numpy as np

        chains = sample_kwargs.get("chains") or 4
        cores = sample_kwargs.get("cores") or chains
        draws = sample_kwargs.get("draws", 1000)

        # -- CPU: max of chains and cores so neither side is starved --
        cpu = max(4, min(max(chains, cores), 32))

        # -- Memory: base + data footprint + posterior trace size --
        # PyMC 5 exposes shapes via model.eval_rv_shapes() without evaluating
        # the underlying graph. Fall back to rv.type.shape for the rare RV
        # missing from that map.
        try:
            shapes = model.eval_rv_shapes()
        except Exception:
            shapes = {}

        obs_bytes = 0
        for rv in model.observed_RVs:
            shape = shapes.get(rv.name)
            if shape is None:
                shape = tuple(d for d in (rv.type.shape or ()) if d is not None)
            n = 1
            for d in shape:
                n *= int(d) if d is not None else 1
            try:
                itemsize = np.dtype(rv.dtype).itemsize
            except TypeError:
                itemsize = 8
            obs_bytes += n * itemsize
        obs_mb = obs_bytes / (1024 * 1024)

        # Posterior trace: chains x draws x sum(prod(shape)) x 8 bytes (float64).
        n_param_scalars = 0
        for rv in model.free_RVs:
            shape = shapes.get(rv.name)
            if shape is None:
                shape = tuple(d for d in (rv.type.shape or ()) if d is not None)
            n = 1
            for d in shape:
                n *= int(d) if d is not None else 1
            n_param_scalars += n
        trace_mb = chains * draws * n_param_scalars * 8 / (1024 * 1024)

        # Base headroom + data (held by every chain) + posterior trace.
        memory_mb = 2048 + int(obs_mb * chains * 1.5) + int(trace_mb * 1.5)

        # Round up to nearest power-of-2 GB (Modal-friendly sizes)
        memory_gb = max(4, 2 ** math.ceil(math.log2(max(1, memory_mb / 1024))))
        memory_mb = min(65536, memory_gb * 1024)

        # -- GPU: provision for JAX-based samplers --
        gpu = None
        if nuts_sampler in ("numpyro", "blackjax"):
            gpu = "A10G"

        return cls(cpu=cpu, memory=memory_mb, gpu=gpu, auto_sized=True)

    def describe(self) -> str:
        """Human-readable description for progress display."""
        prefix = "auto-sized" if self.auto_sized else "preset"
        parts = [f"{self.cpu:.0f} cores", f"{self.memory / 1024:.0f}GB"]
        if self.gpu:
            parts.append(self.gpu)
        return f"{prefix}: {', '.join(parts)}"


DEFAULT_PACKAGES = [
    "pymc",
    "arviz",
    "numpy",
    "pytensor",
    "cloudpickle",
    "lz4",
    "msgpack",
    "fastapi[standard]",
]

OPTIONAL_PACKAGES = {
    "nutpie": "nutpie",
    "numpyro": "numpyro",
}
