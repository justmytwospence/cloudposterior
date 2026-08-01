from __future__ import annotations

import math
from dataclasses import dataclass

# Per-draw budget for the auto-sized timeout, i.e. a floor of 10 draws/sec
# across all chains. Deliberately pessimistic -- it only has to beat
# pathological slowness, and over-estimating costs nothing (Modal bills for
# time used, not for the timeout). Capped at Modal's own 24h ceiling.
_SECONDS_PER_DRAW = 0.1
_MAX_TIMEOUT = 86400


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
            raise ValueError(
                f"unknown instance preset {instance!r}; choose one of "
                f"{sorted(presets)}, or omit instance to auto-size from the model"
            )

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
        # `or` (not a get default): pm.sample accepts draws=None, and a literal
        # None here would blow up the trace-size arithmetic below.
        draws = sample_kwargs.get("draws") or 1000

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
        # Tune draws are kept in the trace when warmup is retained.
        tune = sample_kwargs.get("tune")
        if tune is None:
            tune = 400 if nuts_sampler == "nutpie" else 1000
        retained = draws
        if sample_kwargs.get("discard_tuned_samples") is False or sample_kwargs.get(
            "save_warmup"
        ):
            retained += tune
        trace_mb = chains * retained * n_param_scalars * 8 / (1024 * 1024)

        # The log_likelihood group is n_obs x chains x draws x 8 bytes and is
        # routinely the largest single consumer -- the usual cause of an OOM.
        n_obs = max(1, int(obs_bytes // 8))
        loglik_mb = 0.0
        if sample_kwargs.get("idata_kwargs", {}).get("log_likelihood") is not False:
            loglik_mb = chains * retained * n_obs * 8 / (1024 * 1024)

        # Base headroom + data (held by every chain) + posterior + log-likelihood.
        memory_mb = (
            2048
            + int(obs_mb * chains * 1.5)
            + int(trace_mb * 1.5)
            + int(loglik_mb * 1.5)
        )

        # Round *up* to a power-of-2 GB (Modal-friendly). Never below the
        # estimate: rounding down would undo the headroom just added.
        memory_gb = max(4, 2 ** math.ceil(math.log2(max(1, memory_mb / 1024))))
        memory_mb = min(65536, max(memory_mb, memory_gb * 1024))

        # -- GPU: opt-in. A JAX sampler alone is not evidence the model is big
        # enough to be worth GPU rates, and silently provisioning one billed
        # users for a three-parameter model. Ask for it with instance="gpu" or
        # RemoteConfig(gpu=...).
        gpu = None

        # -- Timeout: scale with the work, keeping the one-hour floor. A long
        # run was killed at 3600s with the partial trace lost.
        total_draws = chains * (draws + tune)
        timeout = min(_MAX_TIMEOUT, max(3600, int(total_draws * _SECONDS_PER_DRAW)))

        return cls(
            cpu=cpu, memory=memory_mb, gpu=gpu, timeout=timeout, auto_sized=True
        )

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

# numpyro is installed on demand (GPU images, or nuts_sampler="numpyro"/"blackjax").
# nutpie is installed unconditionally for CPU images (see _build_pip_specs) since it
# is the default sampler -- it is not pinned to the local version because the model is
# recompiled remotely, so only the model pickle (pymc/pytensor/numpy) must version-match.
OPTIONAL_PACKAGES = {
    "numpyro": "numpyro",
    "blackjax": "blackjax",
    # arviz 1.x metapackage components; absent under arviz 0.x.
    "arviz-base": "arviz-base",
    "arviz-stats": "arviz-stats",
    "arviz-plots": "arviz-plots",
}
