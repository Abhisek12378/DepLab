#!/usr/bin/env bash
set -euo pipefail

split="${1:-}"
start_shard="${2:-1}"
end_shard="${3:-}"
workers="${4:-1}"
minimum_free_gib="${DEPLAB_MIN_FREE_GIB:-20}"
maximum_attempts=3
freeze="configs/large-execution-freeze-v3.0.0.json"

case "$split" in
  development)
    shard_dir="configs/large-development-shards-v3.0.0"
    output="outputs/large-development-results-v3.0.0.jsonl"
    run_root="work/large-development-runs"
    cache_dir="work/large-development-cache"
    ;;
  validation)
    shard_dir="configs/large-validation-shards-v3.0.0"
    output="outputs/large-validation-results-v3.0.0.jsonl"
    run_root="work/large-validation-runs"
    cache_dir="work/large-validation-cache"
    ;;
  final_test)
    if [[ "${DEPLAB_FINAL_TEST_UNLOCK:-}" != "FROZEN_MODEL_AND_EVALUATION" ]]; then
      echo "Final test is locked. Freeze the model and evaluation pipeline before unlocking it." >&2
      exit 6
    fi
    shard_dir="configs/large-final-test-shards-v3.0.0"
    output="outputs/large-final-test-results-v3.0.0.jsonl"
    run_root="work/large-final-test-runs"
    cache_dir="work/large-final-test-cache"
    ;;
  *)
    echo "Usage: $0 development|validation|final_test START END WORKERS" >&2
    exit 2
    ;;
esac

if [[ ! -f "$freeze" ]]; then
  echo "Missing execution freeze: $freeze" >&2
  exit 2
fi
read -r shard_count filename_width < <(
  python3 - "$freeze" "$split" <<'PY'
import json
import sys

payload = json.load(open(sys.argv[1], encoding="utf-8"))
split = payload["splits"][sys.argv[2]]
print(split["shards"], split["filename_width"])
PY
)
end_shard="${end_shard:-$shard_count}"

if ! [[ "$start_shard" =~ ^[0-9]+$ && "$end_shard" =~ ^[0-9]+$ ]]; then
  echo "Shard numbers must be whole numbers." >&2
  exit 2
fi
if ! [[ "$workers" =~ ^[0-9]+$ ]] || (( workers < 1 || workers > 16 )); then
  echo "Worker count must be between 1 and 16." >&2
  exit 2
fi
if ! [[ "$minimum_free_gib" =~ ^[0-9]+$ ]] || (( minimum_free_gib < 1 )); then
  echo "DEPLAB_MIN_FREE_GIB must be a positive whole number." >&2
  exit 2
fi
if (( start_shard < 1 || end_shard > shard_count || start_shard > end_shard )); then
  echo "Choose a shard range between 1 and $shard_count." >&2
  exit 2
fi
if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required but was not found." >&2
  exit 2
fi

mkdir -p work outputs "$run_root" "$cache_dir"
exec 9>"work/large-dataset-run.lock"
if ! flock -n 9; then
  echo "Another large-dataset runner is already active." >&2
  exit 5
fi

minimum_free_kb=$((minimum_free_gib * 1024 * 1024))
for ((number = start_shard; number <= end_shard; number++)); do
  shard_number=$(printf "%0${filename_width}d" "$number")
  total_number=$(printf "%0${filename_width}d" "$shard_count")
  shard="$shard_dir/shard-${shard_number}-of-${total_number}.json"
  if [[ ! -f "$shard" ]]; then
    echo "Missing shard: $shard" >&2
    exit 2
  fi
  free_kb=$(df -Pk . | awk 'NR == 2 {print $4}')
  if (( free_kb < minimum_free_kb )); then
    echo "Stopping before shard $number because less than ${minimum_free_gib} GiB is free." >&2
    exit 4
  fi

  echo "Running $split shard $shard_number of $total_number"
  completed=false
  for ((attempt = 1; attempt <= maximum_attempts; attempt++)); do
    set +e
    PYTHONPATH=src python3 -m deplab batch \
      --manifest "$shard" \
      --output "$output" \
      --run-root "$run_root" \
      --cache-dir "$cache_dir" \
      --cache-scope shared \
      --cleanup-environments \
      --workers "$workers" \
      --timeout 180
    status=$?
    set -e
    if (( status == 0 )); then
      completed=true
      break
    fi
    if (( status != 3 || attempt == maximum_attempts )); then
      echo "Shard $number stopped with status $status on attempt $attempt." >&2
      exit "$status"
    fi
    echo "Repairing retryable result rows before attempt $((attempt + 1))."
    PYTHONPATH=src python3 scripts/repair_large_results.py --split "$split"
  done
  if [[ "$completed" != true ]]; then
    echo "Shard $number did not complete." >&2
    exit 3
  fi
done

PYTHONPATH=src python3 scripts/audit_large_results.py --split "$split"
echo "Large $split shards $start_shard through $end_shard completed."
