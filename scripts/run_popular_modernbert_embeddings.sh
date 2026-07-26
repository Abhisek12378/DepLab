#!/usr/bin/env bash
set -euo pipefail

venv="work/large-model-venv"
cache="work/large-model-huggingface-cache"
lock="work/popular-modernbert-embeddings.lock"
scope="configs/popular-packages-100-scope-v1.0.0.json"
evidence="outputs/popular-release-evidence-v1.0.0.jsonl"
evidence_runtime="outputs/popular-release-evidence-v1.0.0-runtime.json"
embeddings="outputs/popular-release-modernbert-v1.0.0.jsonl"
embedding_runtime="outputs/popular-release-modernbert-v1.0.0-runtime.json"
requirements="outputs/popular-release-modernbert-v1.0.0-requirements.txt"

mkdir -p work outputs "$cache"
exec 9>"$lock"
if ! flock -n 9; then
  echo "Another popular-release embedding job is already active." >&2
  exit 5
fi
if [[ ! -x "$venv/bin/python" ]]; then
  echo "The existing large-model environment is missing at $venv." >&2
  exit 2
fi
if [[ ! -f "$scope" ]]; then
  echo "The frozen popular-package scope is missing at $scope." >&2
  exit 2
fi

echo "Stage 1/3: collecting resumable release-specific PyPI evidence"
PYTHONPATH=src "$venv/bin/python" scripts/collect_popular_release_evidence.py \
  --scope "$scope" \
  --output "$evidence" \
  --runtime "$evidence_runtime" \
  --workers 4 \
  --batch-size 64

echo "Stage 2/3: reusing exact matching ModernBERT vectors"
PYTHONPATH=src "$venv/bin/python" scripts/seed_popular_embeddings.py \
  --evidence "$evidence" \
  --existing-embeddings outputs/large-release-modernbert-v3.0.0.jsonl \
  --output "$embeddings"

echo "Stage 3/3: generating the remaining resumable ModernBERT vectors"
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

uv pip freeze --python "$venv/bin/python" > "$requirements"

"$venv/bin/python" - "$scope" "$evidence" "$embeddings" <<'PY'
import json
import sys
from pathlib import Path

scope = json.loads(Path(sys.argv[1]).read_text())
expected_keys = {
    (name.lower().replace("_", "-").replace(".", "-"), str(version))
    for name, package in scope["packages"].items()
    for version in package["versions"]
}

def rows(path):
    return [
        json.loads(line)
        for line in Path(path).read_text().splitlines()
        if line.strip()
    ]

evidence = rows(sys.argv[2])
embeddings = rows(sys.argv[3])
evidence_keys = {
    (row["package"].lower().replace("_", "-").replace(".", "-"), str(row["version"]))
    for row in evidence
}
embedding_keys = {
    (row["package"].lower().replace("_", "-").replace(".", "-"), str(row["version"]))
    for row in embeddings
}
evidence_hashes = {
    (row["package"].lower().replace("_", "-").replace(".", "-"), str(row["version"])):
    row["selected_text_sha256"]
    for row in evidence
}
invalid_embeddings = [
    row
    for row in embeddings
    if row.get("model") != "answerdotai/ModernBERT-base"
    or row.get("selected_text_sha256") != evidence_hashes.get(
        (row["package"].lower().replace("_", "-").replace(".", "-"), str(row["version"]))
    )
]
print(f"Expected releases: {len(expected_keys)}")
print(f"Evidence rows: {len(evidence)}")
print(f"Embedding rows: {len(embeddings)}")
if (
    len(evidence) != len(evidence_keys)
    or len(embeddings) != len(embedding_keys)
    or evidence_keys != expected_keys
    or embedding_keys != expected_keys
    or invalid_embeddings
):
    raise SystemExit("Popular release embedding audit failed")
print("POPULAR MODERNBERT EMBEDDINGS COMPLETE")
PY

echo "Popular-package ModernBERT embedding pipeline completed."
