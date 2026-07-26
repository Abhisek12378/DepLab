#!/usr/bin/env bash
set -euo pipefail

venv="work/large-model-venv"
lock="work/openai-embedding-experiment.lock"
evidence="outputs/large-release-evidence-text-v3.0.0.jsonl"
embeddings="outputs/large-release-openai-embedding-3-large-v3.0.0.jsonl"
runtime="outputs/large-release-openai-embedding-3-large-v3.0.0-runtime.json"
requirements="outputs/large-release-openai-embedding-3-large-v3.0.0-requirements.txt"
experiment="outputs/deplab-openai-embedding-experiment-v3.0.0"

mkdir -p work outputs
exec 9>"$lock"
if ! flock -n 9; then
  echo "Another OpenAI embedding experiment is already active." >&2
  exit 5
fi
if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "OPENAI_API_KEY is not set in this shell." >&2
  exit 2
fi
if [[ ! -x "$venv/bin/python" ]]; then
  echo "The existing large-model environment is missing at $venv." >&2
  exit 2
fi
if [[ -e "$experiment/experiment-metrics.json" ]]; then
  echo "A completed experiment already exists at $experiment." >&2
  exit 2
fi

echo "Stage 1/3: installing the OpenAI embedding client in the isolated model environment"
uv pip install --python "$venv/bin/python" \
  "openai>=2,<3" \
  "tiktoken>=0.9,<1"

echo "Stage 2/3: creating resumable text-embedding-3-large release vectors"
"$venv/bin/python" scripts/embed_release_openai.py \
  --input "$evidence" \
  --output "$embeddings" \
  --runtime "$runtime" \
  --model text-embedding-3-large \
  --dimensions 3072 \
  --maximum-tokens 8000 \
  --batch-size 16
uv pip freeze --python "$venv/bin/python" > "$requirements"

echo "Stage 3/3: training the same hybrid heads and comparing validation behavior"
PYTHONPATH=src "$venv/bin/python" scripts/evaluate_openai_embedding_experiment.py \
  --embeddings "$embeddings" \
  --model text-embedding-3-large \
  --output-dir "$experiment"

echo "OpenAI embedding comparison completed. The final-test outcomes were not used."
