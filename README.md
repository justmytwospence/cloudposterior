# cloudposterior

Run PyMC models on cloud VMs with one extra line of code. Get results cached, progress streamed to your notebook, and notifications pushed to your phone.

```python
import cloudposterior as cp

with cp.wrap(model, remote=True):
    idata = pm.sample(draws=2000, chains=4)  # runs on a cloud VM
```

Your `pm.sample()` call stays exactly the same. `cloudposterior` intercepts it, ships the model to a beefy VM, streams progress back in real time, caches the result, and returns an `InferenceData` object as if nothing happened.

## Features

### Cloud execution

Offload MCMC sampling to cloud VMs. No infrastructure to manage -- [Modal](https://modal.com) handles containers, scaling, and cleanup. Resources are auto-sized based on your model and sampling config (number of chains, data size, parameter count), or you can pick a preset.

```python
# Auto-sized (default) -- picks CPU and memory based on chains + data size
with cp.wrap(model, remote=True):
    idata = pm.sample(draws=5000, chains=8)

# Manual preset
with cp.wrap(model, remote=True, instance="xlarge"):
    idata = pm.sample(draws=5000, chains=8)
```

The model is serialized with cloudpickle, shipped to a container with version-matched dependencies (PyMC, PyTensor, numpy -- all pinned to your local versions), sampled there, and the trace is compressed and sent back. The remote environment is built once and cached for instant startups on subsequent runs.

### Automatic caching

Re-running a notebook cell? If the model, data, and sampling config haven't changed, cloudposterior returns the cached result instantly. No wasted compute, no waiting. Caching is on by default.

```python
# In-memory cache (default) -- fast, session-scoped
with cp.wrap(model):
    idata = pm.sample(draws=2000)  # runs sampling

with cp.wrap(model):
    idata = pm.sample(draws=2000)  # instant cache hit

# Persistent disk cache -- survives kernel restarts
with cp.wrap(model, cache="disk"):
    idata = pm.sample(draws=2000)

# Custom cache directory
with cp.wrap(model, cache="/data/mcmc-cache"):
    idata = pm.sample(draws=2000)
```

Disk cache organizes results in a human-readable hierarchy using model names and word-based slugs:

```
.cloudposterior/
    eight_schools/
        data-gentle-fox/
            draws2000_tune1000_chains4.nc
            draws1000_tune500_chains2.nc
    my_regression/
        data-watchful-panda/
            draws5000_tune2000_chains8.nc
```

The model directory name comes from (in priority order):
1. `pm.Model(name="eight_schools")` -- explicit model name
2. The variable name you used: `with pm.Model() as my_regression:` (introspected from the call stack)
3. Free RV names: `mu_tau_theta`

### Phone notifications

Monitor long-running sampling from your phone (or anywhere) via [ntfy](https://ntfy.sh) push notifications. Progress updates live with per-chain stats, divergence counts, and speed.

```python
with cp.wrap(model, remote=True, notify=True):
    idata = pm.sample(draws=10000, chains=8)
```

When the topic is auto-generated, cloudposterior prints a link and QR code you can scan to subscribe:

```
Notifications: https://ntfy.sh/eight-schools-subtle-pug
[QR CODE]
```

Notifications include markdown-formatted chain progress tables that update as sampling runs:

```
[done] serializing (0.1s) | [done] uploaded (1.2s) | [done] provisioned (3.1s)
Sampling -- 4 chains, 1000 tune + 2000 draws

| Chain | Progress    | Draws     | Div | Step  | Speed  |
|-------|-------------|-----------|-----|-------|--------|
| 0     | ========>.. | 1200/2000 | 0   | 0.832 | 421/s  |
| 1     | =======>... | 1180/2000 | 2   | 0.741 | 398/s  |
```

You can specify a custom topic, or point to your own ntfy server:

```python
# Custom topic (no QR code -- you already know it)
with cp.wrap(model, notify="my-sampling-channel"):
    ...

# Self-hosted ntfy
with cp.wrap(model, notify={"server": "https://ntfy.example.com"}):
    ...

# Both
with cp.wrap(model, notify={"server": "https://ntfy.example.com", "topic": "mcmc"}):
    ...
```

### Live progress display

Both Jupyter notebooks and terminals get proper in-place progress displays showing every phase of the pipeline:

- Serialization (model + data packaging)
- Upload to cloud
- Container provisioning
- MCMC sampling with per-chain progress bars, divergences, step size, grad evals, speed, elapsed time, and ETA
- Result download

Notebooks use ipywidgets for a native GUI experience. Terminals use Rich for a TUI. The sampling display matches PyMC's native progress -- per-chain bars that turn red on divergences, with all the stats you expect.

## Composable features

| Feature | Default | Flag |
|---------|---------|------|
| Caching (in-memory) | on | `cache=True` / `False` / `"disk"` / `Path` |
| Remote execution | off | `remote=True` / `False` |
| Notifications | off | `notify=True` / `"topic"` / `{"server": ...}` |

All three are independent. Use any combination:

```python
# Just caching (default) -- local sampling, results cached in memory
with cp.wrap(model):
    idata = pm.sample(draws=2000)

# Cloud + disk cache
with cp.wrap(model, remote=True, cache="disk"):
    idata = pm.sample(draws=2000)

# Everything
with cp.wrap(model, remote=True, cache="disk", notify=True):
    idata = pm.sample(draws=2000)

# Local + disk cache + notifications (no cloud)
with cp.wrap(model, cache="disk", notify=True):
    idata = pm.sample(draws=2000)
```

## Installation

```bash
pip install cloudposterior
```

For cloud execution, you'll also need a [Modal](https://modal.com) account:

```bash
pip install modal
modal setup  # one-time browser auth
```

## Configuration

### Instance sizes

When `remote=True` and no `instance` is specified, resources are auto-sized based on the model:
- **CPU**: 1 core per chain (min 4, max 32)
- **Memory**: scales with observed data size and parameter count

You can override with a preset:

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
| `CLOUDPOSTERIOR_NTFY_TOPIC` | Default ntfy topic name |
| `CLOUDPOSTERIOR_NTFY_SERVER` | Custom ntfy server URL (default: `https://ntfy.sh`) |

## How it works

1. **Serialize**: The PyMC model and observed data are serialized with cloudpickle + lz4 compression. A version manifest captures your exact package versions.
2. **Ship**: The payload is sent to Modal, which builds a container image matching your local environment (cached after first run). Resources are auto-sized or use the preset you specified.
3. **Sample**: `pm.sample()` runs on the remote VM with a custom callback that streams per-chain progress back via msgpack-encoded generator yields.
4. **Return**: The InferenceData trace is compressed as NetCDF, sent back, and cached for instant re-use.

## Explicit API

If you prefer not to use the context manager, there's a direct function:

```python
import cloudposterior as cp

idata = cp.sample(model, draws=2000, chains=4)
```

## License

MIT
