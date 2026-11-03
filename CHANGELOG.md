# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.8.0] -- 2026-11-02

### Extraction and migration

- Compose independent [pymc-monitor](https://justmytwospence.github.io/pymc-monitor/) and [pymc-cache](https://justmytwospence.github.io/pymc-cache/) packages. Cloudposterior keeps PyMC interception, Modal transport, notifications and model-map fanout. Both are released on PyPI (0.1.0) and required as `pymc-monitor[notebook,dashboard]>=0.1,<0.2` and `pymc-cache>=0.1,<0.2`.
- Use the same monitoring engine for local and cloud sampling: `until=` and `init="prior"` now work locally; `dashboard=True` locally serves an authenticated loopback dashboard. The canonical report/superchain attrs have `pymc_monitor_` names; cloudposterior writes its legacy aliases too.
- Cache lookup precedes serialization, volume upload and provisioning. Versioned graph/data/environment fingerprints deliberately invalidate old `.cloudposterior/` hits without removing files. `cleanup_cache()` removes only verified owned entries and matching legacy pairs, preserving unrelated files and symlinks. Stopped, futile, unmet and indeterminate adaptive traces are not automatically cached; `cache=False` skips identity work and `overwrite=True` forces a fresh eligible result.
- Require PyMC >=5.28 and PyTensor >=2.38 (Python >=3.11); keep PyMC 5/6 compatibility. Monitor convergence presets remain heuristics, not scientific certification.

### Added

- **Convergence layer** (`cloudposterior.convergence`) behind `until=`. It
  draws on Vehtari et al. (2021), Margossian et al. (2024, nested R-hat),
  Margossian & Gelman (2024) and Vats & Knudson (2021).
  - Checks every element of every free variable, not just scalars. It uses
    vectorized rank-normalized R-hat, bulk/tail ESS and MCSE that match ArviZ
    0.x and behave the same under ArviZ 1.x.
  - Checks at geometrically spaced checkpoints and needs the target to hold at
    two in a row before stopping, so it doesn't stop on noise.
  - HMC gates: post-warmup divergences and E-BFMI.
  - The R-hat threshold scales with the chain count.
  - Stops when the target is unreachable (divergences, stalled R-hat or ESS)
    and names the likely cause: warmup too short, chains disagreeing,
    divergences or slow mixing. It reports the projected draws, with an ETA,
    to reach the target.
  - Presets `until="final"` (= `True`) and `until="draft"`; new targets
    `mcse`, `quantiles`, `var_names`, `patience`, `growth`, `futility`.
  - `until={"escalate": n}`: automatic re-runs with adjusted settings
    (`target_accept`, `tune`, nutpie low-rank adaptation, a larger draw cap),
    chosen by the diagnosed cause, inside the same container call.
  - `until={"superchains": K}`: chains start in K groups that share a starting
    point, and nested R-hat is checked (nutpie and pymc).
- Every remote result carries a convergence report in
  `posterior.attrs["cloudposterior_convergence"]`. A remote run whose report
  flags a problem warns with the cause and a hint, as PyMC does locally.
- `cp.check_convergence(idata, until=...)` produces the same report for any
  trace.
- `init="prior"` on `cp.cloud` / `cp.map` starts each chain from a prior draw.
- The dashboard shows the verdict, the run's own thresholds, the projected
  draws to reach the target, and the hint.

### Changed

- `until=` dicts are validated: unknown keys raise instead of being ignored.
  Normalized targets now carry every key, so cache keys of `until=` runs
  change.
- A model whose free variables are all vectors no longer disables `until=`.
- Remote runs resolve the chain count the way `pm.sample` does
  (`max(2, min(4, cores))`) rather than taking nutpie's default of 6.

### Fixed

- A nutpie run aborted early (Stop or `until=`) no longer returns NaN-padded
  draws for the chains that were ahead; every chain is cut to the shortest
  complete one.
- Stopping a multi-process pymc run on its convergence target no longer lets
  PyMC drop the chains that were behind.

## [0.7.0] -- 2026-08-01

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
