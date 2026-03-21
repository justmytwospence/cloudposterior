from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RemoteConfig:
    """Configuration for remote sampling."""

    cpu: float = 8.0
    memory: int = 16384  # MB
    timeout: int = 3600  # seconds
    gpu: str | None = None

    @classmethod
    def from_instance(cls, instance: str | None) -> RemoteConfig:
        """Map instance hint strings to concrete resource configs."""
        if instance is None:
            return cls()
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


DEFAULT_PACKAGES = [
    "pymc",
    "arviz",
    "numpy",
    "pytensor",
    "cloudpickle",
    "lz4",
    "msgpack",
]

OPTIONAL_PACKAGES = {
    "nutpie": "nutpie",
    "numpyro": "numpyro",
}
