"""Eight Schools example -- the classic PyMC test model.

Usage:
    uv run python examples/eight_schools.py
"""

import numpy as np
import pymc as pm

import cloudposterior as cp

# Eight Schools data
y = np.array([28, 8, -3, 7, -1, 1, 18, 12], dtype=np.float64)
sigma = np.array([15, 10, 16, 11, 9, 11, 10, 18], dtype=np.float64)

with pm.Model() as eight_schools:
    mu = pm.Normal("mu", 0, 5)
    tau = pm.HalfCauchy("tau", 5)
    theta = pm.Normal("theta", mu=mu, sigma=tau, shape=8)
    pm.Normal("obs", mu=theta, sigma=sigma, observed=y)

# Wrap the model -- pm.sample() inside runs on Modal with disk caching
with cp.wrap(eight_schools, remote=True, cache="disk"):
    idata = pm.sample(draws=1000, tune=1000, chains=4)

print("\nSampling complete!")
print(idata.posterior)
