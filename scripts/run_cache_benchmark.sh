#!/usr/bin/env bash
set -euo pipefail

start_repeat="${1:-2}"
end_repeat="${2:-3}"

if ! [[ "$start_repeat" =~ ^[0-9]+$ && "$end_repeat" =~ ^[0-9]+$ ]]; then
  echo "Usage: scripts/run_cache_benchmark.sh [START_REPEAT] [END_REPEAT]" >&2
  exit 2
fi

if (( start_repeat < 1 || end_repeat < start_repeat )); then
  echo "Repeat numbers must be positive and END_REPEAT must be at least START_REPEAT." >&2
  exit 2
fi

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"

for repeat in $(seq "$start_repeat" "$end_repeat"); do
  label="$(printf '%02d' "$repeat")"
  cache_root="work/cache-study-$label"
  cold_output="outputs/cache-cold-$label.jsonl"
  warm_output="outputs/cache-warm-$label.jsonl"

  if [[ -e "$cache_root" || -e "$cold_output" || -e "$warm_output" ]]; then
    echo "Refusing to overwrite repetition $label." >&2
    echo "Existing path: $cache_root, $cold_output, or $warm_output" >&2
    exit 3
  fi

  echo "Repetition $label: running cold-cache experiments"
  PYTHONPATH=src python3 -m deplab batch \
    --manifest configs/cache-measurement.json \
    --output "$cold_output" \
    --run-root work/cache-study-runs \
    --cache-dir "$cache_root" \
    --cache-scope experiment \
    --workers 1

  echo "Repetition $label: running warm-cache experiments"
  PYTHONPATH=src python3 -m deplab batch \
    --manifest configs/cache-measurement.json \
    --output "$warm_output" \
    --run-root work/cache-study-runs \
    --cache-dir "$cache_root" \
    --cache-scope experiment \
    --workers 1
done

echo "Cache benchmark repetitions $start_repeat through $end_repeat completed."

