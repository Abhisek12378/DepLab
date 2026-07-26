#!/usr/bin/env bash
set -euo pipefail

venv="work/large-model-venv"
cache="work/large-model-huggingface-cache"
lock="work/large-model-pipeline.lock"
catalog="outputs/large-package-catalog-v3.0.0.jsonl"
features="outputs/deplab-large-features-v3.0.0"
evidence="outputs/large-release-evidence-text-v3.0.0.jsonl"
embeddings="outputs/large-release-modernbert-v3.0.0.jsonl"
embedding_runtime="outputs/large-release-modernbert-v3.0.0-runtime.json"
embedding_requirements="outputs/large-release-modernbert-v3.0.0-requirements.txt"
freeze="outputs/deplab-large-candidate-freeze-v3.0.0"

mkdir -p work outputs "$cache"
exec 9>"$lock"
if ! flock -n 9; then
  echo "Another large-model pipeline is already active." >&2
  exit 5
fi
if [[ -f "$freeze/candidate-freeze-manifest.json" ]]; then
  echo "A frozen candidate pipeline already exists at $freeze." >&2
  exit 2
fi
if [[ -d "$freeze" ]] && find "$freeze" -mindepth 1 -print -quit | grep -q .; then
  echo "A partial candidate directory exists at $freeze; preserve and inspect it before retrying." >&2
  exit 2
fi

echo "Stage 1/5: collecting resumable PyPI release metadata"
PYTHONPATH=src python3 -m deplab catalog \
  --scope configs/large-scope-v3.0.0.json \
  --output "$catalog"

echo "Stage 2/5: preparing the isolated model environment"
if [[ ! -x "$venv/bin/python" ]]; then
  uv venv --python 3.11 "$venv"
fi
uv pip install --python "$venv/bin/python" \
  "numpy==2.4.6" \
  "pandas==3.0.3" \
  "packaging==26.2"

echo "Stage 3/5: building label-safe feature tables and release evidence text"
PYTHONPATH=src "$venv/bin/python" scripts/build_large_features.py \
  --catalog "$catalog" \
  --output-dir "$features"
PYTHONPATH=src "$venv/bin/python" scripts/build_large_release_text.py \
  --catalog "$catalog" \
  --output "$evidence"

echo "Stage 4/5: creating resumable frozen ModernBERT release embeddings"
if ! "$venv/bin/python" -c 'import torch, transformers' >/dev/null 2>&1; then
  uv pip install --python "$venv/bin/python" \
    --index-url https://download.pytorch.org/whl/cpu \
    "torch==2.13.0"
  uv pip install --python "$venv/bin/python" \
    "transformers==4.57.6" \
    "safetensors==0.8.0"
fi
export HF_HOME="$cache"
export TOKENIZERS_PARALLELISM=false
"$venv/bin/python" scripts/embed_changelog_modernbert.py \
  --input "$evidence" \
  --output "$embeddings" \
  --runtime "$embedding_runtime" \
  --model answerdotai/ModernBERT-base \
  --maximum-tokens 512 \
  --maximum-chunks 4 \
  --batch-size 4
uv pip freeze --python "$venv/bin/python" > "$embedding_requirements"

echo "Stage 5/5: training and freezing blind validation candidates"
PYTHONPATH=src "$venv/bin/python" scripts/train_large_hybrid.py \
  --features "$features/development-features.csv" \
  --validation-inputs "$features/validation-inputs.csv" \
  --policy "$features/model-input-policy.json" \
  --embeddings "$embeddings" \
  --pipeline configs/large-model-pipeline-v3.0.0.json \
  --output-dir "$freeze"

echo "Large feature and candidate-model pipeline completed without reading validation outcomes."
