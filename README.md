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

**Live dashboard** (default for remote) -- a web page with per-chain progress bars updating in real time:

```python
with cp.cloud(model, remote=True, notify=True):
    idata = pm.sample(draws=5000, chains=8)
```

Scan the QR code or open the URL on your phone. No app install needed.

**Push notifications** (default for local) -- get notified when sampling starts and completes via [ntfy](https://ntfy.sh):

```python
with cp.cloud(model, notify=True):                                           # ntfy (local)
with cp.cloud(model, notify="my-channel"):                                   # custom topic
with cp.cloud(model, notify={"server": "https://ntfy.example.com"}):         # self-hosted
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
with cp.cloud(model):                                          # local + memory cache
with cp.cloud(model, cache="disk"):                            # local + disk cache
with cp.cloud(model, remote=True):                             # cloud + memory cache
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

1. **Serialize** -- Model and observed data are serialized with cloudpickle + lz4. A version manifest captures your exact package versions.
2. **Upload once** -- The serialized payload is uploaded to a Modal Volume the first time. Subsequent calls with the same model + data skip the upload entirely.
3. **Sample** -- `pm.sample()` runs remotely. The container loads the payload from the mounted Volume (fast local read) and streams per-chain progress back in real time via msgpack. Only sample kwargs are sent over the wire per call.
4. **Return** -- The InferenceData trace is compressed as NetCDF, sent back, and cached.

Containers stay warm for 20 minutes after the last run, so iterating on sampling settings is near-instant.

---

## Cleanup

Model payloads are stored in a project-scoped Modal Volume. Delete when you're done:

```python
cp.cleanup_volumes()                        # delete default project volume
cp.cleanup_volumes(project="my-research")   # delete specific project
```

---

## Explicit API

If you prefer not to use the context manager:

```python
idata = cp.sample(model, draws=2000, chains=4, remote=True)
```

---

## Example

Clone and run locally for the full interactive progress display.

- [examples/radon.ipynb](examples/radon.ipynb) -- remote execution, caching, and model iteration with the Minnesota Radon dataset
- [examples/notifications.ipynb](examples/notifications.ipynb) -- phone notifications with ntfy

---

## Status

Early proof of concept. Works end-to-end with 43 passing tests, but expect rough edges. Contributions and feedback welcome.

## License

MIT
