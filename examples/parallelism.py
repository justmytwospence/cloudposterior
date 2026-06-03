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
    concurrently and compare them with LOO cross-validation.

    > Run this notebook locally (Jupyter or marimo). `cp.map` prints job-level
    > progress inline and serves a live dashboard for the per-model detail.
    """)
    return


@app.cell
def _():
    import arviz as az
    import pandas as pd
    import pymc as pm

    return az, pd, pm


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Start fresh

    Clear any cached fits and this project's Modal volume so the notebook runs
    cold and reproducibly. (Shown in marimo; hidden in the rendered notebook.)
    """)
    return


@app.cell
def _():
    import shutil
    from pathlib import Path

    import cloudposterior as cp

    # Wipe the local result cache + this project's Modal volume so the example
    # starts cold. The cp.map cells below use `cp`, so marimo runs this first.
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

    `cp.map` uploads each model once and runs the fits concurrently on the cloud.
    Results come back in input order.

    It also serves a **live dashboard** by default (printed as a link below): an
    overview of all three models that you can drill into for any single model's
    chains, convergence, and live traces -- with a global *Stop all* and a
    per-model *Stop*. Pass `dashboard=False` to opt out.
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
    ## Adaptive early-stop with `until`

    Pass `until=True` and each fit stops as soon as every scalar parameter clears
    the convergence target (R-hat <= 1.01, ESS >= 400), with `draws=` as the cap.
    The cheap pooled model converges in a few hundred draws while the richer
    models keep going -- no compute wasted oversampling the easy ones. (nutpie /
    pymc, remote.)
    """)
    return


@app.cell
def _(cp, models):
    _idatas_until = cp.map(models, {"draws": 4000, "tune": 1000, "chains": 4}, until=True)
    # draws each model kept -- early-stop => fewer than the 4000 cap
    {m.name: int(it.posterior.sizes["draw"]) for m, it in zip(models, _idatas_until)}
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Forcing a re-fit with `overwrite`

    The first `cp.map` cached each fit, so a plain re-run is an instant cache hit
    (it prints `all 3 cached`). Pass `overwrite=True` to ignore the cache, refit
    every model, and replace the stored results -- note it prints `fitting 3
    model(s)` instead.
    """)
    return


@app.cell
def _(cp, models):
    # Re-run ignoring the cache: refits all three (not a cache hit).
    _idatas_overwrite = cp.map(models, {"draws": 1000, "tune": 1000, "chains": 4}, overwrite=True)
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
