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
    # cloudposterior: Cloud Execution

    Run PyMC MCMC sampling on cloud VMs with one line of code. This notebook demonstrates remote execution using the classic Radon dataset from Gelman & Hill (2006).

    > Run this notebook locally (Jupyter or marimo) to watch the live progress display animate in-cell. Some outputs don't render in GitHub's notebook viewer.
    """)
    return


@app.cell
def _():
    import numpy as np
    import pandas as pd
    import pymc as pm
    import arviz as az

    return az, pd, pm


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Start fresh

    Clear any cached results and this project's Modal volume so the notebook runs
    cold and reproducibly. (Shown in marimo; hidden in the rendered notebook.)
    """)
    return


@app.cell
def _():
    import shutil
    from pathlib import Path

    import cloudposterior as cp

    # Wipe the local result cache + this project's Modal volume so the example
    # starts cold. The sampling cells below use `cp`, so marimo runs this first.
    shutil.rmtree(Path(".cloudposterior"), ignore_errors=True)
    cp.cleanup_volumes()
    return (cp,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Data

    919 household radon measurements across 85 Minnesota counties (Gelman & Hill, 2006).
    """)
    return


@app.cell
def _(pd, pm):
    df = pd.read_csv(pm.get_data("radon.csv"))

    county_names = df.county.unique()
    county_idx = df.county_code.values
    log_radon = df.log_radon.values
    floor = df.floor.values

    print(f"{len(df)} observations, {len(county_names)} counties")
    return county_idx, county_names, floor, log_radon


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Model

    Hierarchical varying-intercepts model with non-centered parameterization. Each county gets its own intercept (partial pooling), and floor level (basement vs first floor) is a fixed effect.
    """)
    return


@app.cell
def _(county_idx, county_names, floor, log_radon, pm):
    with pm.Model(name="radon_intercepts", coords={"county": county_names}) as radon:
        mu_a = pm.Normal("mu_a", mu=0, sigma=5)
        sigma_a = pm.HalfNormal("sigma_a", sigma=2)
        a_raw = pm.Normal("a_raw", mu=0, sigma=1, dims="county")
        a = pm.Deterministic("a", mu_a + sigma_a * a_raw, dims="county")
        b_floor = pm.Normal("b_floor", mu=0, sigma=5)
        mu = a[county_idx] + b_floor * floor
        sigma_y = pm.HalfNormal("sigma_y", sigma=2)
        pm.Normal("obs", mu=mu, sigma=sigma_y, observed=log_radon)
    return (radon,)


@app.cell
def _(pm, radon):
    pm.model_to_graphviz(radon)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Remote execution

    `cp.cloud()` intercepts `pm.sample()` and runs it on a cloud VM. The model is uploaded to a volume on first run. Resources (CPU cores, memory) are auto-sized to your model.
    """)
    return


@app.cell
def _(cp, pm, radon):
    with cp.cloud(radon, remote=True):
        idata = pm.sample(draws=2000, tune=1000, chains=4)
    return (idata,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Diagnostics
    """)
    return


@app.cell
def _(az, idata):
    az.summary(idata, filter_vars="like", var_names=["mu_a", "sigma_a", "b_floor", "sigma_y"])
    return


@app.cell
def _(az, idata):
    az.plot_trace(idata, filter_vars="like", var_names=["mu_a", "sigma_a", "b_floor", "sigma_y"])[0, 0].figure
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## GPU acceleration with JAX

    For models that benefit from GPU acceleration, use `nuts_sampler="numpyro"` to sample with JAX via NumPyro. cloudposterior automatically provisions a GPU container and installs `jax[cuda12]` when it detects a JAX-based sampler -- no configuration needed.
    """)
    return


@app.cell
def _(cp, pm, radon):
    with cp.cloud(radon, remote=True):
        idata_jax = pm.sample(draws=2000, tune=1000, chains=4, nuts_sampler="numpyro")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Cleanup

    Model payloads are stored in a project-scoped volume. Delete it when you're done.
    """)
    return


@app.cell
def _(cp):
    cp.cleanup_volumes()
    return


if __name__ == "__main__":
    app.run()
