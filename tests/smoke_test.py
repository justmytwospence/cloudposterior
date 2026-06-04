"""Packaging smoke test for a built cloudposterior artifact.

Run against a built wheel or sdist (NOT the source tree) to confirm the artifact
is self-contained: public API present, the remotely-executed worker module
shipped, and a tiny end-to-end LOCAL sample works. Touches no cloud / Modal.

    uv run --isolated --no-project --with dist/*.whl tests/smoke_test.py
    uv run --isolated --no-project --with dist/*.tar.gz tests/smoke_test.py
"""

from __future__ import annotations


def main() -> None:
    import cloudposterior as cp

    # Public API is importable and intact.
    for name in ("cloud", "sample", "map", "cleanup_volumes"):
        assert hasattr(cp, name), f"cloudposterior.{name} missing from the build"

    # The worker ships inside the wheel; Modal loads it remotely, so if the build
    # drops it, remote sampling breaks only in production. Fail here instead.
    import cloudposterior.remote.worker  # noqa: F401

    # End-to-end local path: trivial model, a few draws, no cloud/progress UI.
    import numpy as np
    import pymc as pm

    from cloudposterior._idata import group_names

    rng = np.random.default_rng(0)
    y = rng.normal(loc=3.0, size=50)

    with pm.Model() as model:
        mu = pm.Normal("mu", 0.0, 10.0)
        pm.Normal("y", mu=mu, sigma=1.0, observed=y)

    with cp.cloud(model, progress=False):
        idata = pm.sample(
            draws=20, tune=20, chains=1, progressbar=False, random_seed=0
        )

    assert "posterior" in group_names(idata), "no posterior group in the result"
    print("smoke test OK")


if __name__ == "__main__":
    main()
