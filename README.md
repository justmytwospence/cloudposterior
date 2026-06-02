# cloudposterior

**Stop waiting for MCMC. Start shipping posteriors.**

cloudposterior lets you run PyMC models on cloud VMs without changing your sampling code. One extra line gives you cloud compute, automatic caching, and phone notifications -- while `pm.sample()` stays exactly the same.

```python
import cloudposterior as cp

with cp.cloud(model, remote=True):
    idata = pm.sample(draws=5000, chains=8)  # 8 cores in the cloud, zero config
```

---

## Why?

You've built a hierarchical model. It's beautiful. But sampling takes 45 minutes on your laptop, your fans sound like a jet engine, and you can't use your machine for anything else.

cloudposterior fixes this:

- **Ship sampling to the cloud** with one line. Your model runs on a VM with as many cores and as much RAM as it needs.
- **Never re-run the same model twice.** Results are cached automatically -- re-execute a notebook cell and get your posterior back instantly.
- **Monitor from anywhere.** Get live progress notifications on your phone while your model samples.

All three features work independently. Use any combination, or just the caching.

---

## Quick start

```bash
uv add cloudposterior

# For cloud execution (optional):
uv add modal && uv run modal setup
```

```python
import pymc as pm
import cloudposterior as cp

with pm.Model() as my_model:
    mu = pm.Normal("mu", 0, 5)
    sigma = pm.HalfNormal("sigma", 5)
    pm.Normal("obs", mu, sigma, observed=data)

# This is the only line you add:
with cp.cloud(my_model, remote=True, cache="disk"):
    idata = pm.sample(draws=2000, chains=4)
```

Second time you run that cell? Instant. The result is already cached.

---

## Features

### Cloud execution

Offload MCMC to cloud VMs. No Docker, no infrastructure, no config files. [Modal](https://modal.com) handles containers, scaling, and cleanup.

```python
with cp.cloud(model, remote=True):
    idata = pm.sample(draws=5000, chains=8)
```

Your model is serialized with cloudpickle, shipped to a container with version-matched dependencies (PyMC, PyTensor, numpy -- all pinned to your exact local versions), sampled there, and the trace is compressed and sent back. The container image is built once and cached, so subsequent runs start in seconds.

### Smart resource sizing

cloudposterior inspects your model and sampling config to right-size the VM automatically:

- **CPU cores** matched to your chain count (8 chains = 8 cores)
- **Memory** scaled to your observed data size and parameter count

No guessing, no over-provisioning. A small model gets 4 cores and 4GB. A hierarchical model with large datasets gets 8+ cores and 16GB+. The progress display shows what was chosen:

```
cloudposterior -- Modal (auto-sized: 8 cores, 8GB)
```

Want explicit control? Use a preset:

```python
with cp.cloud(model, remote=True, instance="xlarge"):  # 32 cores, 64GB
    ...
```

The container is sized on the **first** `pm.sample()` call inside `with cp.cloud(...)` and stays that size for the duration of the block. If a later call uses different `chains`/`draws` that the auto-sizer would have provisioned differently, you'll get a warning -- start a new `cp.cloud()` block to resize.

### Automatic caching

Re-running a notebook cell? If the model, data, and sampling config haven't changed, cloudposterior returns the cached result instantly. No wasted compute. Caching is **on by default**.

```python
with cp.cloud(model):
    idata = pm.sample(draws=2000)  # samples normally

with cp.cloud(model):
    idata = pm.sample(draws=2000)  # instant -- cached
```

For persistence across kernel restarts, use disk caching:

```python
with cp.cloud(model, cache="disk"):
    idata = pm.sample(draws=2000)
```

Results are stored in a human-readable directory tree:

```
.cloudposterior/
├── radon_intercepts/
│   └── draws2000_tune1000_chains4-a3f7b2c9.nc
└── radon_slopes/
    └── draws2000_tune1000_chains4-7c2e5fa8.nc
```

Model names come from `pm.Model(name="radon_intercepts")`. The hash suffix ensures uniqueness when non-displayed parameters (like `random_seed`) differ.

### Monitoring

Two ways to monitor sampling:

**Live dashboard** (on by default for remote) -- convergence diagnostics, trace plots, and a stop button:

```python
with cp.cloud(model, remote=True):
    idata = pm.sample(draws=5000, chains=8)
```

Scan the QR code or open the URL on your phone. No app install needed.

**Push notifications** -- get notified when sampling starts and completes via [ntfy](https://ntfy.sh):

```python
with cp.cloud(model, notify=True):              # auto topic
with cp.cloud(model, notify="my-channel"):      # custom topic
with cp.cloud(model, remote=True, notify=True): # remote (dashboard on by default) + ntfy
```

With `remote=True`, the dashboard is on by default; `notify=True` adds ntfy push notifications on top.

### Live progress display

Both Jupyter notebooks and terminals show real-time, in-place progress for every phase:

1. Serialization
2. Upload
3. Container provisioning
4. MCMC sampling -- per-chain progress bars, divergences, step size, grad evals, speed, ETA
5. Result download

Notebooks get a live anywidget display that animates in-cell in both Jupyter and marimo. Terminals get a Rich TUI. Progress bars turn red when chains diverge, just like PyMC's native display. During a remote run a **Stop** button appears in the cell -- click it to abort early and keep the partial trace.

### Samplers

cloudposterior defaults to **nutpie** -- PyMC's recommended NUTS sampler, roughly 2x faster on CPU -- for fully continuous models, and falls back to PyMC's built-in sampler for models with discrete variables. Override per call:

```python
with cp.cloud(model, remote=True):
    idata = pm.sample()                         # nutpie (default for continuous models)
    idata = pm.sample(nuts_sampler="pymc")      # PyMC's sampler (handles discrete vars)
    idata = pm.sample(nuts_sampler="numpyro")   # JAX sampler (GPU auto-provisioned)
```

Live per-chain progress, convergence diagnostics (rank-normalized R-hat and ESS), and the stop button work with **nutpie** and **pymc**. JAX samplers (`numpyro`, `blackjax`) run entirely inside a compiled graph with no per-draw hook, so they report phase-level progress only.

Works with both **PyMC 5 and PyMC 6** (PyMC 6 ships arviz 1.x's DataTree); the versions installed in the remote container are matched to your local environment.

### Adaptive sampling

Stop as soon as the chains converge instead of guessing a draw count -- `draws` becomes the cap:

```python
with cp.cloud(model, remote=True, until={"r_hat": 1.01, "ess": 400}):
    idata = pm.sample(draws=20000)   # stops early once every parameter clears the target
```

`until=True` uses the Vehtari (2021) defaults shown above. Works with **nutpie** and **pymc** (the samplers with a per-draw hook).

### Parallel fitting

Fit many models at once on a single warm container -- ideal for model comparison and prior sensitivity:

```python
import arviz as az

idatas = cp.map([pooled, hierarchical, per_county], {"draws": 1000})
az.compare(dict(zip(["pooled", "hier", "county"], idatas)))
```

`sample_kwargs` is a shared dict or a list aligned with the models. Results return in input order. See [examples/parallelism.ipynb](examples/parallelism.ipynb).

### Predictive checks

`pm.sample_prior_predictive()` and `pm.sample_posterior_predictive()` are intercepted too, so prior/posterior predictive checks run on the same cloud container.

---

## Composable features

| Feature | Default | Control |
|---------|---------|---------|
| Caching | **on** (in-memory) | `cache=True` / `False` / `"disk"` / `Path(...)` |
| Cloud execution | off | `remote=True` / `False` |
| Live dashboard | **on** (when remote) | `dashboard=True` / `False` |
| Push notifications | off | `notify=True` / `"topic"` / `{"server": ..., "topic": ...}` |
| Adaptive early-stop | off | `until=True` / `{"r_hat": ..., "ess": ...}` (remote) |

Mix and match:

```python
with cp.cloud(model):                                          # local + memory cache
with cp.cloud(model, cache="disk"):                            # local + disk cache
with cp.cloud(model, remote=True):                             # cloud + dashboard
with cp.cloud(model, remote=True, cache="disk", notify=True):  # everything
```

---

## Configuration

### Instance presets

| Name     | CPUs | Memory |
|----------|------|--------|
| `small`  | 4    | 8 GB   |
| `medium` | 8    | 16 GB  |
| `large`  | 16   | 32 GB  |
| `xlarge` | 32   | 64 GB  |
| `gpu`    | 8    | 16 GB + A100 |

### Environment variables

| Variable | Description |
|----------|-------------|
| `CLOUDPOSTERIOR_NTFY_TOPIC` | Default ntfy topic |
| `CLOUDPOSTERIOR_NTFY_SERVER` | Custom ntfy server (default: `https://ntfy.sh`) |

---

## Cloud backend

Cloud execution currently uses [Modal](https://modal.com). Modal provides fast container spin-up, automatic dependency packaging, and a generous free tier.

```bash
uv add modal
modal setup  # one-time browser auth
```

The backend is abstracted behind a `ComputeBackend` interface. Support for additional providers (AWS, GCP, SSH to your own machines) is planned.

---

## How it works

1. **Serialize** -- The model is serialized with cloudpickle + lz4 on the first `pm.sample()` call inside `with cp.cloud(...)`. Cloudpickle bundles your observed data into the model object, so there's a single payload, not two. A version manifest captures your exact package versions.
2. **Upload once** -- The serialized payload is uploaded to a Modal Volume the first time. Subsequent calls with the same model skip the upload entirely. Old payloads from past edits are pruned automatically.
3. **Sample** -- `pm.sample()` runs remotely. The container loads the payload from the mounted Volume (fast local read) and streams per-chain progress back in real time via msgpack. Only sample kwargs and a path string are sent over the wire per call.
4. **Return** -- The InferenceData trace is compressed as NetCDF, sent back, and cached.

Containers stay warm for 20 minutes after the last run, so iterating on sampling settings is near-instant.

---

## Cleanup

Model payloads are stored in a project-scoped Modal Volume. Delete when you're done:

```python
cp.cleanup_volumes()                        # delete the current project's volume
cp.cleanup_volumes(project="my-research")   # delete a specific project's volume
```

For a one-shot teardown that also stops a warm container immediately, use the context manager's `destroy()`:

```python
session = cp.cloud(model, remote=True)
with session:
    idata = pm.sample(draws=2000)
session.destroy()  # stop the container and delete the project volume
```

---

## Explicit API

If you prefer not to use the context manager, `cp.sample()` runs a single remote sampling job (always cloud, no persistent container reuse):

```python
idata = cp.sample(model, draws=2000, chains=4)
```

For repeated sampling with the same model, the `cp.cloud()` context manager is cheaper -- it keeps the container warm and only sends kwargs after the first call.

---

## Example

Clone and run locally (Jupyter or marimo) for the full live progress display.

- [examples/basics.ipynb](examples/basics.ipynb) -- cloud execution and GPU acceleration with the Minnesota Radon dataset
- [examples/caching.ipynb](examples/caching.ipynb) -- local and disk caching, model iteration
- [examples/monitoring.ipynb](examples/monitoring.ipynb) -- live dashboard and push notifications
- [examples/parallelism.ipynb](examples/parallelism.ipynb) -- fit many models in parallel with `cp.map` and compare them with LOO

---

## Tests

The default suite is fast and free -- it covers serialization, caching, naming, kwarg validation, and runs `cp.cloud(model)` end-to-end against a local PyMC sampler:

```bash
uv run pytest tests/ -v
```

A separate suite of end-to-end tests hits real Modal infrastructure to verify the cloud path. These cost a small amount of Modal credit per run and are skipped by default. Opt in with `--run-modal`:

```bash
uv run pytest tests/test_modal_e2e.py -v --run-modal
```

Modal tests provision the smallest possible instance, sample 20 draws on a 2-RV model, and use isolated per-test project volumes that are cleaned up at teardown.

---

## Status

Early proof of concept. Works end-to-end with 75+ passing tests, but expect rough edges. Contributions and feedback welcome.

## License

MIT
