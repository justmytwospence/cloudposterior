#!/usr/bin/env bash
# Re-lock dependencies so the lockfile is reproducible off this machine.
#
# CI runs `uv sync --locked`, which requires uv.lock to resolve identically on a
# clean checkout. Plain `uv lock` bakes user-level settings from
# ~/.config/uv/uv.toml into the lock -- an `exclude-newer` span, and any
# `exclude-newer-package` entries -- that CI does not share, and every --locked
# job then fails with "the lockfile needs to be updated".
#
# A project-level setting cannot undo this: uv *merges* the user-level
# exclude-newer-package table into the project's rather than replacing it, so
# the entries leak in regardless. Ignoring user config entirely is the only fix.
set -euo pipefail

cd "$(dirname "$0")/.."
uv lock --no-config "$@"

if grep -q "exclude-newer" uv.lock; then
  echo "warning: uv.lock still carries exclude-newer settings; CI will reject it." >&2
  exit 1
fi
echo "uv.lock is portable."
