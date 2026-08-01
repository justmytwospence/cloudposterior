# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Breaking

- **Cache keys changed.** Cache identity is now derived from the model's
  structure (its logp graph plus a content hash of its data) instead of a hash
  of the cloudpickle bytes. Rebuilding the same model in a new interpreter
  produces different pickle bytes — every random variable's shared RNG is
  re-seeded at build time — so the persistent disk cache previously missed on
  every cross-session lookup. Existing `.cloudposterior/` entries are orphaned
  by the change; `cp.cleanup_cache()` removes them.
- **The live dashboard now requires its token.** The dashboard page and its
  `/progress` feed are authenticated with the same token as `/stop`, which the
  URL cloudposterior prints carries. Previously both were open to anyone who
  reached the URL, and `/progress` serves parameter names and posterior draws.
- **`session.destroy()` is scoped to its own model** and no longer deletes the
  project-wide Modal Volume. Pass `destroy(delete_volume=True)` for the old
  behavior.
- **Auto-sizing no longer provisions a GPU** just because `nuts_sampler` is
  `numpyro` or `blackjax`. Request one with `instance="gpu"` or
  `RemoteConfig(gpu=...)`. CPU images install jax/numpyro when the sampler
  needs them, so those runs still work.
- `fastapi` and `ipywidgets` are no longer installed as client dependencies
  (neither was imported client-side).

### Added

- `cp.cleanup_cache()` — the local-disk counterpart to `cp.cleanup_volumes()`.
- `CacheBackend`, `MemoryCache` and `DiskCache` are exported from the package
  root.
- Explicit PyMC 6 support: `backend=` passes through, `tune` is resolved per
  sampler (PyMC 6 defaults nutpie to 400 rather than 1000), and
  `sample_posterior_predictive`'s changed `var_names` semantics are flagged.
- Remote timeouts scale with the work instead of a fixed one hour.

### Fixed

- `cp.cloud(...)` is no longer reentrant and restores PyMC when entering the
  model context fails. Re-entry previously left `pm.sample` recursing into
  itself, and a failed entry left PyMC patched for the whole process.
- Serialized model bytes are re-generated when observed data changes, so a
  `pm.set_data` mutation reaches the worker instead of silently reusing
  pre-mutation state under the old cache key.
- The predictive, SMC and log-likelihood interceptors honor an explicit
  `model=` argument; they previously ran the wrapped model and returned
  results for a model the caller never named.
- A remote job whose stream produced no trace fails instead of silently
  starting a second billed run.
- The Volume payload prune can no longer delete the payload just uploaded.
- `MemoryCache` returns a copy and is bounded; extending a returned trace no
  longer corrupts the cached entry.
- `DiskCache`: 64-bit filename discriminator with full-key verification,
  unique temp names (concurrent saves could interleave into one file),
  corrupt entries treated as a miss, path traversal via sample kwargs blocked,
  and entries pruned to the newest 20.
- Different `random_seed` generators no longer share a cache entry, and
  callable kwargs (`step=`, `callback=`) no longer defeat caching.
- ntfy topics carry 64 bits of entropy, are validated, and are generated once
  per `cp.cloud` block; sends moved off the progress thread.
- The dashboard vendors uPlot instead of loading it from a CDN, and reports
  render errors as errors rather than as the run having ended.
- `cp.map` failures cancel sibling containers; its warm environment records
  its config; and its runs now mark the dashboard complete instead of polling
  a billed endpoint indefinitely.
- Numerous smaller fixes: `draws=None` crashing auto-sizing, progress decoding
  raising on partial payloads, the aggregator's missing final snapshot, a
  broken widget taking down the whole run, `until=` silently doing nothing on
  models with no scalar parameters, and Modal auth errors being reported for
  unrelated failures.

## [0.6.1]

- Added the Documentation URL to the package metadata.

## [0.6.0]

- Quarto documentation site, deployed to GitHub Pages.
- CI exercises both the PyMC 5 (arviz 0.x) and PyMC 6 (arviz 1.x) stacks.

## [0.4.0]

- `cp.map` for fitting many models in parallel, with a per-model dashboard.

## [0.3.0]

- Automatic GPU provisioning for JAX samplers, `nuts_sampler` forwarded from
  `pm.sample`, cache determinism fixes, and split example notebooks.

## [0.2.0]

- Persistent containers, Volume-based model payloads, and a progress overhaul.
