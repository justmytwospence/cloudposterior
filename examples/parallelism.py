import marimo

__generated_with = "0.23.8"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # cloudposterior: Parallel model comparison

    `cp.map` fits **many models at once** on the cloud -- the canonical parallel
    Bayesian workflow. Here we fit three increasingly-pooled radon models
    concurrently on one warm container and compare them with LOO cross-validation.

    > Run this notebook locally (Jupyter or marimo). `cp.map` reports job-level
    > progress (no per-chain widget) -- the variants run in parallel.
    """)
    return


@app.cell
def _():
    import arviz as az
    import pandas as pd
    import pymc as pm

    import cloudposterior as cp

    return az, cp, pd, pm


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
    counties = df.county.unique()
    county_idx = df.county_code.values
    log_radon = df.log_radon.values
    floor = df.floor.values
    return counties, county_idx, floor, log_radon


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Three models

    - **Complete pooling** -- one intercept for all counties (ignores county).
    - **Hierarchical** -- partial pooling: county intercepts shrink toward a shared mean.
    - **Unpooled** -- an independent intercept per county (no sharing).

    All three share a floor (basement vs first-floor) fixed effect. We build them
    in one cell so each name is defined once (marimo's single-definition rule).
    """)
    return


@app.cell
def _(counties, county_idx, floor, log_radon, pm):
    with pm.Model(name="pooled") as pooled:
        a = pm.Normal("a", 0, 5)
        b = pm.Normal("b_floor", 0, 5)
        sigma = pm.HalfNormal("sigma", 2)
        pm.Normal("obs", a + b * floor, sigma, observed=log_radon)

    with pm.Model(name="hierarchical", coords={"county": counties}) as hierarchical:
        mu_a = pm.Normal("mu_a", 0, 5)
        sigma_a = pm.HalfNormal("sigma_a", 2)
        a_raw = pm.Normal("a_raw", 0, 1, dims="county")
        a = pm.Deterministic("a", mu_a + sigma_a * a_raw, dims="county")
        b = pm.Normal("b_floor", 0, 5)
        sigma = pm.HalfNormal("sigma", 2)
        pm.Normal("obs", a[county_idx] + b * floor, sigma, observed=log_radon)

    with pm.Model(name="unpooled", coords={"county": counties}) as unpooled:
        a = pm.Normal("a", 0, 5, dims="county")
        b = pm.Normal("b_floor", 0, 5)
        sigma = pm.HalfNormal("sigma", 2)
        pm.Normal("obs", a[county_idx] + b * floor, sigma, observed=log_radon)

    models = [pooled, hierarchical, unpooled]
    return (models,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Fit all three in parallel

    `cp.map` uploads each model once and runs the fits concurrently on a single
    warm Modal container. Results come back in input order.
    """)
    return


@app.cell
def _(cp, models):
    idatas = cp.map(models, {"draws": 1000, "tune": 1000, "chains": 4})
    return (idatas,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Compare with LOO

    Compute pointwise log-likelihood locally (cheap, no resampling), then rank the
    models by leave-one-out cross-validation. The hierarchical model usually wins:
    partial pooling beats both the over-confident unpooled fit and the
    over-smoothed complete-pooling fit.
    """)
    return


@app.cell
def _(az, idatas, models, pm):
    for _model, _idata in zip(models, idatas):
        with _model:
            pm.compute_log_likelihood(_idata)

    comparison = az.compare(dict(zip(["pooled", "hierarchical", "unpooled"], idatas)))
    comparison
    return (comparison,)


@app.cell
def _(az, comparison):
    az.plot_compare(comparison, figsize=(8, 3))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Cleanup

    `cp.map` provisions a project-scoped volume. Delete it when you're done.
    """)
    return


@app.cell
def _(cp):
    cp.cleanup_volumes()
    return


if __name__ == "__main__":
    app.run()
