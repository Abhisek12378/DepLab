#!/usr/bin/env bash
set -euo pipefail

venv="work/large-model-venv"
lock="work/deep-mlp-experiment.lock"
output="outputs/deplab-modernbert-mlp-experiment-v3.0.0"
requirements="outputs/deplab-modernbert-mlp-experiment-v3.0.0-requirements.txt"

mkdir -p work outputs
exec 9>"$lock"
if ! flock -n 9; then
  echo "Another deep-learning experiment is already active." >&2
  exit 5
fi
if [[ ! -x "$venv/bin/python" ]]; then
  echo "The existing large-model environment is missing at $venv." >&2
  exit 2
fi
if ! "$venv/bin/python" -c 'import torch, numpy, pandas' >/dev/null 2>&1; then
  echo "The large-model environment is missing PyTorch, NumPy or pandas." >&2
  exit 2
fi
if [[ -e "$output/experiment-metrics.json" ]]; then
  echo "A completed experiment already exists at $output." >&2
  exit 2
fi

echo "Training the simple four-class ModernBERT MLP"
PYTHONPATH=src "$venv/bin/python" scripts/train_deep_mlp_experiment.py \
  --epochs 40 \
  --batch-size 512 \
  --learning-rate 0.001 \
  --hidden-one 256 \
  --hidden-two 128 \
  --dropout 0.2 \
  --development-fold 5 \
  --output-dir "$output"

uv pip freeze --python "$venv/bin/python" > "$requirements"
echo "Deep-learning experiment completed. The final-test outcomes were not used."
