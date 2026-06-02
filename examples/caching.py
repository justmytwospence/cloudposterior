import marimo

__generated_with = "0.23.8"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # cloudposterior: Caching

    cloudposterior automatically caches sampling results so you never re-run the same model twice. This is useful even without cloud execution -- just wrap your model in `cp.cloud()` and re-running a notebook cell returns the cached result instantly.

    Two caching modes:

    - **In-memory** (`cache=True`, the default) -- results are cached for the current session. Re-running a cell in the same kernel is instant.
    - **Disk** (`cache="disk"`) -- results persist across kernel restarts. Re-opening a notebook and running the same model returns the cached result without any sampling.
    """)
    return


@app.cell
def _():
    import numpy as np
    import pandas as pd
    import pymc as pm
    import arviz as az

    import cloudposterior as cp

    return az, cp, pd, pm


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Setup

    Using the Radon model from [basics.ipynb](basics.ipynb).
    """)
    return


@app.cell
def _(pd, pm):
    df = pd.read_csv(pm.get_data('radon.csv'))
    with pm.Model(name='radon_intercepts', coords={'county': df.county.unique()}) as radon:
        _mu_a = pm.Normal('mu_a', mu=0, sigma=5)
        _sigma_a = pm.HalfNormal('sigma_a', sigma=2)
        _a_raw = pm.Normal('a_raw', mu=0, sigma=1, dims='county')
        _a = pm.Deterministic('a', _mu_a + _sigma_a * _a_raw, dims='county')
        b_floor = pm.Normal('b_floor', mu=0, sigma=5)
        _mu = _a[df.county_code.values] + b_floor * df.floor.values
        _sigma_y = pm.HalfNormal('sigma_y', sigma=2)
        pm.Normal('obs', mu=_mu, sigma=_sigma_y, observed=df.log_radon.values)
    return df, radon


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Local caching (no cloud needed)

    You don't need cloud execution to use caching. Just wrap your model in `cp.cloud()` -- it intercepts `pm.sample()` and caches the result. Sampling runs locally with PyMC's normal output. The second time you run the same cell, the result is returned from cache.

    This is useful when you're iterating on analysis code downstream of sampling -- you don't want to re-sample every time you tweak a plot.
    """)
    return


@app.cell
def _(cp, pm, radon):
    # First run: samples normally (PyMC progress bar shown)
    with cp.cloud(radon):
        _idata = pm.sample(draws=2000, tune=1000, chains=4)
    return


@app.cell
def _(cp, pm, radon):
    # Re-run: instant (in-memory cache hit)
    with cp.cloud(radon):
        _idata = pm.sample(draws=2000, tune=1000, chains=4)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Disk caching

    With `cache="disk"`, results are saved to `.cloudposterior/` and survive kernel restarts. Restart your kernel and re-run this cell -- the result comes back instantly without any sampling.

    The cache key includes the model structure, observed data, and all sampling parameters. Changing any of these triggers a new sample.
    """)
    return


@app.cell
def _(cp, pm, radon):
    with cp.cloud(radon, cache='disk'):
        _idata = pm.sample(draws=2000, tune=1000, chains=4)
    return


@app.cell
def _(cp, pm, radon):
    with cp.cloud(radon, cache='disk'):
        _idata = pm.sample(draws=2000, tune=1000, chains=4)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Cache layout

    The disk cache uses human-readable directory names with a hash suffix for uniqueness:

    ```
    .cloudposterior/
    ├── radon_intercepts/
    │   └── draws2000_tune1000_chains4-a3f7b2c9.nc
    └── radon_slopes/
        └── draws2000_tune1000_chains4-7c2e5fa8.nc
    ```

    Model names come from `pm.Model(name="radon_intercepts")`. The hash suffix ensures that runs with different non-displayed parameters (like `random_seed`) get separate cache files.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Model iteration

    Caching works naturally with model iteration. Each model variant gets its own cache entry. Switching back to a previous model returns the cached result.
    """)
    return


@app.cell
def _(df, pm):
    with pm.Model(name='radon_slopes', coords={'county': df.county.unique()}) as radon_slopes:
        _mu_a = pm.Normal('mu_a', mu=0, sigma=5)
        _sigma_a = pm.HalfNormal('sigma_a', sigma=2)
        _a_raw = pm.Normal('a_raw', mu=0, sigma=1, dims='county')
        _a = pm.Deterministic('a', _mu_a + _sigma_a * _a_raw, dims='county')
        mu_b = pm.Normal('mu_b', mu=0, sigma=5)
        sigma_b = pm.HalfNormal('sigma_b', sigma=2)
        b_raw = pm.Normal('b_raw', mu=0, sigma=1, dims='county')
        b = pm.Deterministic('b', mu_b + sigma_b * b_raw, dims='county')
        _mu = _a[df.county_code.values] + b[df.county_code.values] * df.floor.values
        _sigma_y = pm.HalfNormal('sigma_y', sigma=2)
        pm.Normal('obs', mu=_mu, sigma=_sigma_y, observed=df.log_radon.values)
    return (radon_slopes,)


@app.cell
def _(cp, pm, radon_slopes):
    # New model -> samples fresh
    with cp.cloud(radon_slopes, cache="disk"):
        idata_slopes = pm.sample(draws=2000, tune=1000, chains=4)
    return (idata_slopes,)


@app.cell
def _(az, idata_slopes):
    az.summary(idata_slopes, filter_vars="like", var_names=["mu_a", "sigma_a", "mu_b", "sigma_b", "sigma_y"])
    return


if __name__ == "__main__":
    app.run()
