# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

cloudposterior lets you run PyMC MCMC sampling on cloud VMs (currently Modal) with one line of code. It intercepts `pm.sample()` via a context manager (`cp.cloud`) and adds cloud execution, automatic caching, live progress display, and phone notifications.

## Commands

```bash
uv sync                          # install all deps (including dev group)
uv run pytest tests/ -v          # run all tests
uv run pytest tests/test_cache.py -v          # run one test file
uv run pytest tests/test_cache.py::test_name -v  # run single test
```

CI runs pytest on Python 3.11 and 3.12.

## Architecture

### Request flow

1. `cp.cloud(model)` context manager monkeypatches `pm.sample` to route through `_run_sample()` in `api.py`
2. Model + observed data are serialized separately (cloudpickle + lz4 for the model, numpy + lz4 for data) in `serialize.py`
3. Cache key is computed from the serialized bytes + sample kwargs (`cache.py`)
4. If remote: `ModalBackend` (`backends/modal_backend.py`) submits a `SamplingPayload` to Modal, which runs `remote/worker.py` in a container with version-matched dependencies
5. If local: the original `pm.sample` is called directly
6. Progress events stream back via msgpack and are rendered by display sinks (ipywidgets for notebooks, Rich for terminals) in `display.py`
7. Results are cached and returned as `az.InferenceData`

### Key abstractions

- **`ComputeBackend` / `SamplingJob`** (`backends/__init__.py`): Abstract interface for compute providers. Only Modal is implemented; designed for future providers.
- **`RemoteEnvironment`** (`backends/__init__.py`): Persistent execution environment with data pre-loaded (via Modal Volumes). Accepts multiple sampling jobs without re-uploading data. Provisioned via `ComputeBackend.provision()`.
- **`CacheBackend`** (`cache.py`): Protocol with `MemoryCache` (default, session-scoped) and `DiskCache` (persistent, human-readable directory tree under `.cloudposterior/`).
- **`ProgressEvent`** (`progress.py`): Union type of `PhaseUpdate` and `SamplingProgress` that flows through display sinks.
- **`SamplingPayload`** (`serialize.py`): Dataclass bundling serialized model, data, version manifest, and sample kwargs for transport.

### Persistent containers and volumes

When `remote=True`, containers stay warm for 20 minutes. Model payloads are stored in a project-scoped Modal Volume so only sample kwargs are sent per-call:

1. Model is serialized once in `__enter__()` and uploaded to a Volume at `{model_slug}/{data_slug}/payload-{hash}.bin`
2. A `modal.Cls`-based sampler loads the payload from the mounted Volume (fast local read)
3. Each `pm.sample()` call sends only kwargs + a path string -- no model/data bytes on the wire
4. If the model changes between calls, the new payload is uploaded to the Volume (KB, fast)
5. Volume is project-scoped (`cp-{project}`) -- cleaned up via `cp.cleanup_volumes(project=...)`

### Naming conventions (two layers)

Human-readable names for browsability, machine hashes for correctness:

| System | Human-readable (cosmetic) | Machine-correct (identity) |
|--------|--------------------------|---------------------------|
| Local disk cache | `{model_slug}/{data_slug}/{params}.nc` | `compute_cache_key()` SHA-256 |
| Remote Volume | `{model_slug}/{data_slug}/payload-{hash}.bin` | `payload_hash()` SHA-256 prefix |
| Notifications | `{model_slug}-{random_wordhash}` | N/A |

Shared utilities in `naming.py`: `model_slug()`, `data_slug()`, `payload_hash()`, `wordhash()`

### Live dashboard (`notify="dashboard"` or `notify=True` with `remote=True`)

`dashboard.py` contains `DashboardSink` (writes progress to a Modal Dict) and `DASHBOARD_HTML` (self-contained page with JS polling). Two `@modal.fastapi_endpoint` functions serve the HTML and progress JSON. The dashboard URL includes the model name for readability (e.g., `radon-intercepts-a3f7b2-dev.modal.run`).

When `notify=True` and `remote=True`, defaults to dashboard. When local, defaults to ntfy (start + complete notifications only).

### Remote worker

`remote/worker.py` runs inside Modal containers. It is never imported locally -- Modal serializes and executes it. It deserializes the model, runs sampling with a progress callback that streams per-chain stats via a queue, and returns lz4-compressed NetCDF.

### Auto-sizing

`RemoteConfig._auto()` in `config.py` inspects the model's observed data size, parameter count, and chain count to right-size VM resources (CPU cores and memory).