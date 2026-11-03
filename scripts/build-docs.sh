#!/usr/bin/env bash
# Generate API docs and render from a temporary docs-only project. This never
# moves, rewrites, executes or restores the paired example notebooks.
set -euo pipefail

cd "$(dirname "$0")/.."
quartodoc build

stage="$(mktemp -d)"
trap 'rm -rf "$stage"' EXIT
cp _quarto.yml custom.scss *.qmd "$stage/"
mkdir -p "$stage/examples"
cp examples/*.ipynb "$stage/examples/"
cp -R reference "$stage/reference"
(
  cd "$stage"
  quarto render
)
mkdir -p _site
cp -R "$stage/_site/." _site/
