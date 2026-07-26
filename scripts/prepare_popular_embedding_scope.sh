#!/usr/bin/env bash
set -euo pipefail

lock="work/popular-embedding-scope.lock"
log_summary="outputs/popular-packages-100-scope-summary-v1.0.0.json"

mkdir -p work outputs configs
exec 9>"$lock"
if ! flock -n 9; then
  echo "Another popular-package scope job is already active." >&2
  exit 5
fi
if [[ -e "$log_summary" ]]; then
  echo "A frozen popular-package scope already exists at $log_summary." >&2
  exit 2
fi

PYTHONPATH=src work/large-model-venv/bin/python \
  scripts/prepare_popular_release_scope.py \
  --top-count 100 \
  --release-cutoff "2026-07-27T00:00:00+00:00"

echo "Popular-package release enumeration completed. Embeddings were not generated."
