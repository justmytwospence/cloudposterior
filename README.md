# cloudposterior

**Stop waiting for MCMC. Start shipping posteriors.**

cloudposterior lets you run PyMC models on cloud VMs without changing your sampling code. One extra line gives you cloud compute, automatic caching, and phone notifications -- while `pm.sample()` stays exactly the same.

```python
import cloudposterior as cp

with cp.wrap(model, remote=True):
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
with cp.wrap(my_model, remote=True, cache="disk"):
    idata = pm.sample(draws=2000, chains=4)
```

Second time you run that cell? Instant. The result is already cached.

---

## Features

### Cloud execution

Offload MCMC to cloud VMs. No Docker, no infrastructure, no config files. [Modal](https://modal.com) handles containers, scaling, and cleanup.

```python
with cp.wrap(model, remote=True):
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
with cp.wrap(model, remote=True, instance="xlarge"):  # 32 cores, 64GB
    ...
```

### Automatic caching

Re-running a notebook cell? If the model, data, and sampling config haven't changed, cloudposterior returns the cached result instantly. No wasted compute. Caching is **on by default**.

```python
with cp.wrap(model):
    idata = pm.sample(draws=2000)  # samples normally

with cp.wrap(model):
    idata = pm.sample(draws=2000)  # instant -- cached
```

For persistence across kernel restarts, use disk caching:

```python
with cp.wrap(model, cache="disk"):
    idata = pm.sample(draws=2000)
```

Results are stored in a human-readable directory tree:

```
.cloudposterior/
    eight_schools/
        data-gentle-fox/
            draws2000_tune1000_chains4.nc
    my_regression/
        data-watchful-panda/
            draws5000_tune2000_chains8.nc
```

Model names are derived automatically from your code -- `pm.Model(name="eight_schools")`, or the variable name you used (`with pm.Model() as my_regression:`), or the names of your random variables.

### Phone notifications

Monitor sampling from your phone (or anywhere) via [ntfy](https://ntfy.sh) push notifications. Progress updates live with per-chain stats, divergence counts, and speed.

```python
with cp.wrap(model, remote=True, notify=True):
    idata = pm.sample(draws=10000, chains=8)
```

cloudposterior prints a link and QR code you can scan to subscribe:

```
Notifications: https://ntfy.sh/eight-schools-subtle-pug
[QR CODE]
```

Notifications include a live-updating markdown table:

```
[done] serializing (0.1s) | [done] uploaded (1.2s) | [done] provisioned (3.1s)

| Chain | Progress    | Draws     | Div | Step  | Speed  |
|-------|-------------|-----------|-----|-------|--------|
| 0     | ========>.. | 1200/2000 | 0   | 0.832 | 421/s  |
| 1     | =======>... | 1180/2000 | 2   | 0.741 | 398/s  |
```

Custom topic or self-hosted server:

```python
with cp.wrap(model, notify="my-channel"):                                   # custom topic
with cp.wrap(model, notify={"server": "https://ntfy.example.com"}):         # self-hosted
```

### Live progress display

Both Jupyter notebooks and terminals show real-time, in-place progress for every phase:

1. Serialization
2. Upload
3. Container provisioning
4. MCMC sampling -- per-chain progress bars, divergences, step size, grad evals, speed, ETA
5. Result download

Notebooks get an ipywidgets GUI. Terminals get a Rich TUI. Progress bars turn red when chains diverge, just like PyMC's native display.

---

## Composable features

| Feature | Default | Control |
|---------|---------|---------|
| Caching | **on** (in-memory) | `cache=True` / `False` / `"disk"` / `Path(...)` |
| Cloud execution | off | `remote=True` / `False` |
| Notifications | off | `notify=True` / `"topic"` / `{"server": ..., "topic": ...}` |

Mix and match:

```python
with cp.wrap(model):                                          # local + memory cache
with cp.wrap(model, cache="disk"):                            # local + disk cache
with cp.wrap(model, remote=True):                             # cloud + memory cache
with cp.wrap(model, remote=True, cache="disk", notify=True):  # everything
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

1. **Serialize** -- Model and observed data are serialized with cloudpickle + lz4. A version manifest captures your exact package versions.
2. **Ship** -- Payload is sent to Modal, which builds a matching container image (cached after first run). VM resources are auto-sized to your model.
3. **Sample** -- `pm.sample()` runs remotely with a callback that streams per-chain progress back in real time via msgpack.
4. **Return** -- The InferenceData trace is compressed as NetCDF, sent back, and cached.

---

## Explicit API

If you prefer not to use the context manager:

```python
idata = cp.sample(model, draws=2000, chains=4, remote=True)
```

---

## Status

This is an early proof of concept. It works end-to-end with 30 passing tests, but expect rough edges. Contributions and feedback welcome.

## License

MIT
