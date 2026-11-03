"""CPU-only, installed-artifact smoke for all three composed distributions."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch


def main() -> None:
    import pymc as pm
    import pymc_cache
    import pymc_monitor

    import cloudposterior as cp
    from cloudposterior import api
    from cloudposterior.convergence import read_report

    root = Path.cwd().resolve()
    for package in (cp, pymc_monitor, pymc_cache):
        loaded = Path(package.__file__).resolve()
        if loaded.is_relative_to(root / package.__name__):
            raise AssertionError(f"{package.__name__} was imported from the source checkout")
        if loaded.is_relative_to(root / "src"):
            raise AssertionError(f"{package.__name__} was imported from local src")
    from pymc_monitor.engine import iter_sample

    assert cp.__all__ and callable(iter_sample)
    assert hasattr(pymc_cache, "CacheSession")
    import cloudposterior.remote.worker  # noqa: F401

    with pm.Model() as model:
        pm.Normal("theta", 0, 1)
    kwargs = dict(draws=15, tune=15, chains=2, cores=2, nuts_sampler="pymc",
                  random_seed=45, compute_convergence_checks=False, progressbar=False)
    with cp.cloud(model, remote=False, cache=False, progress=False,
                  until={"ess": 20, "r_hat": 1.2, "patience": 1}):
        adaptive = pm.sample(**{**kwargs, "draws": 160, "tune": 60})
    assert read_report(adaptive)["status"] == "met"
    assert adaptive.posterior.sizes["draw"] < 160
    assert adaptive.posterior.sizes["chain"] == 2
    assert adaptive.posterior.attrs["pymc_monitor_convergence"] == (
        adaptive.posterior.attrs["cloudposterior_convergence"]
    )
    with cp.cloud(model, remote=False, cache=True, progress=False):
        fixed = pm.sample(**kwargs)
        with patch.object(api, "_run_local", side_effect=AssertionError("cache hit sampled")):
            hit = pm.sample(**kwargs)
    assert fixed.posterior.sizes == hit.posterior.sizes
    print("cloudposterior extraction smoke OK", file=sys.stdout)


if __name__ == "__main__":
    main()
