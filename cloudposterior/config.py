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
    idle_timeout: int = 600  # seconds before idle container is torn down

    @classmethod
    def from_instance(cls, instance: str | None, model=None, sample_kwargs: dict | None = None) -> RemoteConfig:
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
            return cls._auto(model, sample_kwargs)

        return cls()

    @classmethod
    def _auto(cls, model, sample_kwargs: dict) -> RemoteConfig:
        """Estimate optimal resources from model characteristics."""
        import numpy as np

        chains = sample_kwargs.get("chains") or 4
        draws = sample_kwargs.get("draws", 1000)
        tune = sample_kwargs.get("tune", 1000)

        # -- CPU: 1 core per chain, minimum 4 --
        cpu = max(4, min(chains, 32))

        # -- Memory: base + data + parameter overhead --
        obs_bytes = 0
        for rv in model.observed_RVs:
            try:
                val = rv.tag.test_value if hasattr(rv.tag, "test_value") else None
                if val is not None:
                    obs_bytes += np.asarray(val).nbytes
            except Exception:
                pass

        obs_mb = obs_bytes / (1024 * 1024)
        n_params = len(model.free_RVs)

        # Each chain needs: observed data in memory + parameter traces + PyTensor overhead
        per_chain_mb = max(256, obs_mb * 3 + n_params * 0.01 * draws)
        memory_mb = 2048 + int(chains * per_chain_mb)

        # Round up to nearest power-of-2 GB (Modal-friendly sizes)
        memory_gb = max(4, 2 ** math.ceil(math.log2(max(1, memory_mb / 1024))))
        memory_mb = min(65536, memory_gb * 1024)

        return cls(cpu=cpu, memory=memory_mb, auto_sized=True)

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
