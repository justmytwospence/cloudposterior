#!/usr/bin/env bash
# Build the Quarto documentation site into _site/.
#
# The marimo examples/*.py files look like Jupyter percent-scripts to Quarto and
# collide with the paired examples/*.ipynb (Quarto then renders neither, and just
# copies the .ipynb as a resource). Quarto ignores .gitignore / .quartoignore and
# render-list negations here, so the only reliable fix is to move the .py files
# out of the tree for the render and restore them afterward (restored even on
# error via the trap). The tracked marimo files are left untouched.
#
# Run via uv so quartodoc (venv) and quarto (system) both resolve:
#   uv run bash scripts/build-docs.sh
set -euo pipefail

cd "$(dirname "$0")/.."

# Refuse to move files carrying uncommitted work: the restore below falls back
# to git, which can only recover what git already knows about.
if ! git diff --quiet -- examples/*.py 2>/dev/null; then
  echo "error: examples/*.py have uncommitted changes; commit or stash first." >&2
  echo "       (this script moves them aside during the render)" >&2
  exit 1
fi

stash="$(mktemp -d)"
restore() {
  mv "$stash"/*.py examples/ 2>/dev/null || true
  rm -rf "$stash"
  # Belt and braces: if the move back didn't happen (interrupted run, full
  # disk), recover the tracked originals from git rather than leaving the
  # working tree missing four files.
  git checkout -- examples/ 2>/dev/null || true
}
trap restore EXIT
mv examples/*.py "$stash"/

quartodoc build
quarto render
