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
    # cloudposterior: Monitoring

    Monitor MCMC sampling remotely with a live dashboard or push notifications.

    - **`dashboard=True`** (default for remote) -- live web dashboard with convergence diagnostics, trace plots, and a stop button. Open on your phone or any browser via QR code.
    - **`notify=True`** -- push notifications via [ntfy](https://ntfy.sh) when sampling starts and completes.

    > Run this notebook locally (Jupyter or marimo) to see the dashboard links and QR codes, plus the in-cell live progress.
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


@app.cell
def _(pd, pm):
    df = pd.read_csv(pm.get_data("radon.csv"))

    with pm.Model(name="radon_intercepts", coords={"county": df.county.unique()}) as radon:
        mu_a = pm.Normal("mu_a", mu=0, sigma=5)
        sigma_a = pm.HalfNormal("sigma_a", sigma=2)
        a_raw = pm.Normal("a_raw", mu=0, sigma=1, dims="county")
        a = pm.Deterministic("a", mu_a + sigma_a * a_raw, dims="county")
        b_floor = pm.Normal("b_floor", mu=0, sigma=5)
        mu = a[df.county_code.values] + b_floor * df.floor.values
        sigma_y = pm.HalfNormal("sigma_y", sigma=2)
        pm.Normal("obs", mu=mu, sigma=sigma_y, observed=df.log_radon.values)
    return (radon,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Live dashboard

    The dashboard is on by default for remote runs. Scan the QR code or open the URL to see:

    - Per-chain progress bars with speed, divergences, and ETA
    - Live convergence diagnostics (R-hat, ESS) with color-coded status
    - Live trace plots and posterior KDE per parameter
    - A stop button to end sampling early if convergence looks good
    """)
    return


@app.cell
def _(cp, pm, radon):
    with cp.cloud(radon, remote=True, cache=False):
        idata = pm.sample(draws=10000, tune=1000, chains=4)
    return (idata,)


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
    ## Push notifications

    Pass `notify=True` to get push notifications when sampling starts and completes. Works for both local and remote runs. Scan the QR code with the [ntfy app](https://ntfy.sh) to subscribe.

    For private notifications, point to your own [ntfy server](https://docs.ntfy.sh/install/) with `notify={"server": "https://ntfy.example.com"}`.
    """)
    return


@app.cell
def _(cp, pm, radon):
    with cp.cloud(radon, cache=False, notify=True):
        idata_1 = pm.sample(draws=2000, tune=1000, chains=4)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Both

    Use both together to get the live dashboard AND push notifications.
    """)
    return


@app.cell
def _(cp, pm, radon):
    with cp.cloud(radon, remote=True, cache=False, notify=True):
        idata_2 = pm.sample(draws=1000, tune=1000, chains=4)
    return (idata_2,)


@app.cell
def _(az, idata_2):
    az.summary(idata_2, filter_vars='like', var_names=['mu_a', 'sigma_a', 'b_floor', 'sigma_y'])
    return


if __name__ == "__main__":
    app.run()
