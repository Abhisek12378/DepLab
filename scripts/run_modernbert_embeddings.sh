#!/usr/bin/env bash
set -euo pipefail

venv="work/modernbert-venv"
cache="work/huggingface-cache"
lock="work/modernbert-embedding.lock"
input="outputs/changelog-text-catalog-expanded-v1.0.0.jsonl"
output="outputs/modernbert-release-embeddings-v1.0.0.jsonl"
runtime="outputs/modernbert-release-embeddings-v1.0.0-runtime.json"

mkdir -p work outputs "$cache"
exec 9>"$lock"
if ! flock -n 9; then
  echo "Another ModernBERT embedding job is already running." >&2
  exit 5
fi
if [[ ! -f "$input" ]]; then
  echo "Missing changelog text catalog: $input" >&2
  exit 2
fi
if [[ ! -x "$venv/bin/python" ]]; then
  uv venv --python 3.11 "$venv"
fi
if ! "$venv/bin/python" -c 'import torch, transformers' >/dev/null 2>&1; then
  uv pip install --python "$venv/bin/python" \
    --index-url https://download.pytorch.org/whl/cpu \
    torch
  uv pip install --python "$venv/bin/python" \
    'transformers>=4.48,<5' safetensors
fi

export HF_HOME="$cache"
export TOKENIZERS_PARALLELISM=false
"$venv/bin/python" scripts/embed_changelog_modernbert.py \
  --input "$input" \
  --output "$output" \
  --runtime "$runtime" \
  --model answerdotai/ModernBERT-base \
  --maximum-tokens 512 \
  --maximum-chunks 4 \
  --batch-size 4

uv pip freeze --python "$venv/bin/python" \
  > outputs/modernbert-release-embeddings-v1.0.0-requirements.txt
echo "ModernBERT release embeddings completed."
