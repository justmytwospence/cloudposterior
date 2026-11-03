"""Compatibility shims for ArviZ 0.x InferenceData and 1.x DataTree.

Storage conversion belongs to pymc-cache; sampler trace operations belong to
pymc-monitor. These names remain available for existing cloudposterior users.
"""

from __future__ import annotations

from pymc_cache.io import (  # noqa: F401
    add_group,
    get_group,
    group_attrs,
    group_names,
    load_all,
    sanitize_inference_data,
)
from pymc_monitor.idata import concat_chains, to_inference_data, trim_incomplete_draws  # noqa: F401


def pymc_major() -> int:
    """Major PyMC API version for client compatibility checks."""
    try:
        import pymc

        return int(pymc.__version__.split(".")[0])
    except Exception:
        return 5


_ess_tail_fallback_warned = False


def ess_tail(arr) -> float:
    """Tail ESS with a fallback for the different ArviZ 0.x/1.x signatures."""
    import arviz as az

    try:
        return float(az.ess(arr, method="tail"))
    except (TypeError, ValueError):
        for kwargs in ({"method": "tail", "prob": (0.025, 0.975)},
                       {"method": "tail", "prob": 0.05}):
            try:
                return float(az.ess(arr, **kwargs))
            except Exception:
                continue
        global _ess_tail_fallback_warned
        if not _ess_tail_fallback_warned:
            _ess_tail_fallback_warned = True
            import warnings

            warnings.warn("arviz tail-ESS API unavailable; reporting bulk ESS in place "
                          "of tail ESS.", stacklevel=2)
        return float(az.ess(arr))
