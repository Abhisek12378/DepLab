#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
export PYTHONPATH=src

python3 -m deplab scope-plan \
  --plan configs/large-dataset-plan-v3.0.0.json \
  --output configs/large-scope-draft-v3.0.0.json

python3 -m deplab scope-audit \
  --input configs/large-scope-draft-v3.0.0.json \
  --output configs/large-scope-v3.0.0.json

python3 -m deplab matrix \
  --scope configs/large-scope-v3.0.0.json \
  --pairs configs/large-development-pairs-v3.0.0.json \
  --output configs/large-development-matrix-v3.0.0.json

python3 -m deplab matrix \
  --scope configs/large-scope-v3.0.0.json \
  --pairs configs/large-validation-pairs-v3.0.0.json \
  --output configs/large-validation-matrix-v3.0.0.json

python3 -m deplab matrix \
  --scope configs/large-scope-v3.0.0.json \
  --pairs configs/large-final-test-pairs-v3.0.0.json \
  --output configs/large-final-test-matrix-v3.0.0.json

python3 scripts/audit_large_dataset_scope.py

echo "Large-dataset scope, matrices, and audit completed."
