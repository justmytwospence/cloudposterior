# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

cloudposterior lets you run PyMC MCMC sampling on cloud VMs (currently Modal) with one line of code. It intercepts `pm.sample()` via a context manager (`cp.cloud`) and adds cloud execution, automatic caching, live progress display, and phone notifications.

## Commands

```bash
uv sync                          # install all deps (including dev group)
uv run pytest tests/ -v          # run free local tests (default)
uv run pytest tests/ -v --run-modal   # also run paid Modal e2e tests
uv run pytest tests/test_cache.py -v          # run one test file
uv run pytest tests/test_cache.py::test_name -v  # run single test
```

Tests marked `@pytest.mark.modal` (`tests/test_modal_e2e.py`) hit real Modal infrastructure and incur cloud costs. They are skipped unless `--run-modal` is passed. See `tests/conftest.py` for the marker plumbing.

CI runs pytest on Python 3.11-3.13 (free tests only -- Modal tests are not in CI).

## Example notebooks

Each example in `examples/` exists in two formats that must be kept in sync:

- `*.ipynb` -- the Jupyter version. This is the artifact GitHub renders (with embedded outputs), so it is what users see in the repo.
- `*.py` -- the marimo version (`uv run marimo edit examples/<name>.py`). This is the source you edit and pair on.

**Sync rule: whenever you change one format, update the other in the same change.** They are equivalent notebooks, not independent files -- an edit to a marimo `.py` cell must be mirrored into the matching `.ipynb`, and vice versa.

- marimo `.py` -> `.ipynb`: `uv run marimo export ipynb examples/<name>.py -o examples/<name>.ipynb`. Add `--include-outputs` to refresh the rendered outputs, but note it re-executes the notebook (real Modal sampling, so cost + time -- only do this deliberately).
- `.ipynb` -> marimo `.py`: `uvx marimo convert examples/<name>.ipynb -o examples/<name>.py`, then re-apply the marimo cleanups (drop trailing `;` output-suppression, make each cell's final expression the value to render). Do NOT add a PEP 723 script-metadata block -- `cloudposterior` is a local editable install and would fail to resolve from PyPI; these run in the project venv.
- After editing either format, run `uv run marimo check examples/<name>.py` before committing.

## Architecture

### Request flow

1. `cp.cloud(model)` context manager monkeypatches five PyMC functions to route through `api.py`: `pm.sample` (→ `_run_sample` / `_run_sample_persistent`), `pm.sample_prior_predictive` / `pm.sample_posterior_predictive` (→ `_run_predictive`), `pm.sample_smc` (→ `_run_smc`), and `pm.compute_log_likelihood` (→ `_run_idata_op`). The latter three reuse a blocking (non-streaming) remote-op template; only `pm.sample` streams per-draw progress.
2. The model is serialized as a single cloudpickle + lz4 blob in `serialize.py` (observed data is captured inside the model pickle -- no separate data payload)
3. Cache key is computed from the serialized bytes + sample kwargs (`naming.cache_key`); cache backends live in `cache.py`
4. If remote: `ModalBackend` (`backends/modal_backend.py`) submits a `SamplingPayload` to Modal, which runs `remote/worker.py` in a container with version-matched dependencies
5. If local: the original `pm.sample` is called directly
6. Progress events stream back via msgpack and are rendered by display sinks (an anywidget that animates live in both Jupyter and marimo notebooks, Rich for terminals) in `display.py`
7. Results are cached and returned as `az.InferenceData`

### Key abstractions

- **`ComputeBackend` / `SamplingJob`** (`backends/__init__.py`): Abstract interface for compute providers. Only Modal is implemented; designed for future providers.
- **`RemoteEnvironment`** (`backends/__init__.py`): Persistent execution environment with data pre-loaded (via Modal Volumes). Accepts multiple sampling jobs without re-uploading data. Provisioned via `ComputeBackend.provision()`.
- **`CacheBackend`** (`cache.py`): Protocol with `MemoryCache` (default, session-scoped) and `DiskCache` (persistent, human-readable directory tree under `.cloudposterior/`).
- **`ProgressEvent`** (`progress.py`): Union type of `PhaseUpdate` and `SamplingProgress` that flows through display sinks.
- **`SamplingPayload`** (`serialize.py`): Dataclass bundling the serialized model (data included in the pickle), version manifest, and sample kwargs for transport.

### Persistent containers and volumes

When `remote=True`, containers stay warm for 20 minutes. Model payloads are stored in a project-scoped Modal Volume so only sample kwargs are sent per-call:

1. Model is serialized once on the first `pm.sample()` call (lazy first-touch, memoized on the model) and uploaded to a Volume at `{model_slug}/payload-{hash}.bin`
2. A `modal.Cls`-based sampler loads the payload from the mounted Volume (fast local read)
3. Each `pm.sample()` call sends only kwargs + a path string -- no model/data bytes on the wire
4. If the model changes between calls, the new payload is uploaded to the Volume (KB, fast)
5. Volume is project-scoped (`cp-{project}`) -- cleaned up via `cp.cleanup_volumes(project=...)`

The provisioned env is **kept warm past the `with` block** (not torn down in `__exit__`): it's held in `api._LIVE_ENVS` keyed by `(project, model_slug)` and reused by later runs of the same model, so the dashboard stays browsable and re-runs skip cold start. It's torn down by `cp.cleanup_volumes()` / `session.destroy()` / an `atexit` hook (Modal's `scaledown_window` idles the container out ~20 min regardless). Each run clears the control `Dict["stop"]` flag so a reused env doesn't inherit a stale stop.

### Naming conventions (two layers)

Human-readable names for browsability, machine hashes for correctness:

| System | Human-readable (cosmetic) | Machine-correct (identity) |
|--------|--------------------------|---------------------------|
| Local disk cache | `{model_slug}/{params_label}-{key8}.nc` | `cache_key()` SHA-256 |
| Remote Volume | `{model_slug}/payload-{hash}.bin` | `payload_hash()` SHA-256 prefix |
| Notifications | `{model_slug}-{random_wordhash}` | N/A |

Shared utilities: `model_slug()`, `payload_hash()`, `cache_key()` in `naming.py`; `wordhash()` in `wordhash.py`

### Live dashboard (`dashboard=` kwarg; on by default for `remote=True`)

`dashboard.py` contains `DashboardSink` (writes progress to a Modal Dict) and `DASHBOARD_HTML` (self-contained page with JS polling). Two `@modal.fastapi_endpoint` functions serve the HTML and progress JSON. The dashboard URL includes the model name for readability (e.g., `radon-intercepts-a3f7b2-dev.modal.run`).

The dashboard is controlled by the separate `dashboard: bool | None` kwarg (default: on for remote runs, off -- with a warning if explicitly requested -- for local). `notify=` is ntfy-only: `True` auto-generates a topic, a string is the topic, a dict accepts `{"topic", "server"}`; sends fire on sampling start, completion, and errors.

### Remote worker

`remote/worker.py` runs inside Modal containers. It is never imported locally -- Modal serializes and executes it. It deserializes the model, runs sampling while streaming per-chain stats via a queue, and returns lz4-compressed NetCDF. `_sample_and_stream` branches three ways by sampler:

- **`pymc`**: `pm.sample(nuts_sampler="pymc", callback=...)` -- the per-draw callback fills the progress queue. The `nuts_sampler="pymc"` is explicit so PyMC 6 doesn't auto-select nutpie (which would reject the callback).
- **`nutpie`** (the default): runs nutpie's background sampler (`blocking=False`) with its native `progress_callback`; the generator loop polls the stop Dict and calls `handle.abort()` to stop early (keeping the partial trace), then `handle.wait()` for the final result.
- **`numpyro`/`blackjax`**: `pm.sample(nuts_sampler=..., progressbar=False)` with **no callback** -- PyMC's external NUTS samplers run inside JAX with no per-draw hook (and PyMC 6 raises if a callback is passed). These report phase-level progress only.

Besides streaming sampling, the worker has **blocking (non-streaming) entries** that load the model and return lz4 NetCDF directly: `run_prior_predictive` / `run_posterior_predictive`, `run_smc` (`pm.sample_smc`), and `run_compute_log_likelihood`. They mount as `@modal.method()`s on the persistent `Sampler` Cls and are driven client-side by `_run_blocking_op`.

**Custom `step=` over the wire**: a step instance pickled separately from the model has value variables in a different graph than the worker's Volume-loaded model (PyMC raises -- 5.28 words it "the following variables are not random variables in the model"). So `intercepted_sample` ships a combined `{model, step}` blob via `serialize_model_with_step`; `_unpack_model_payload` on the worker detects the dict, re-injects the step, and forces the pymc sampler (so the per-draw callback / live progress still work).

### Samplers and arviz compatibility

- **Default sampler**: `api._default_sampler()` picks nutpie for fully continuous models (PyMC's own default, ~2x faster) and the pymc sampler when there are discrete free RVs; locally it falls back to pymc when nutpie isn't installed. nutpie is always installed in CPU remote images (`_build_pip_specs`).
- **Callback constraint**: PyMC's per-draw `callback` only fires for `nuts_sampler="pymc"`; PyMC 6 *raises* if a callback is passed to an external sampler. Only attach a callback on the pymc path (worker + `_run_local`).
- **`_idata.py`**: thin shims so the codebase works on **both arviz 0.x (PyMC 5) and arviz 1.x (PyMC 6, DataTree)** -- `.groups()` vs `.groups`, removed `convert_to_inference_data`, changed `ess(method="tail")`, the dict-valued `sample_stats` attr nutpie writes, and object-dtype data vars from SMC's `sample_stats` (`beta`/`accept_rate`/`log_marginal_likelihood`, which even native `to_netcdf` rejects) -- both handled by `sanitize_inference_data`. `add_group` merges a remotely-computed group (e.g. `log_likelihood`) into a local idata in place. Always go through these helpers instead of calling arviz idata methods directly.

### Auto-sizing

`RemoteConfig._auto()` in `config.py` inspects the model's observed data size, parameter count, and chain count to right-size VM resources (CPU cores and memory).