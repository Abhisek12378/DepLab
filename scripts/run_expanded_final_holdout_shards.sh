#!/usr/bin/env bash
set -euo pipefail

start_shard="${1:-1}"
end_shard="${2:-9}"
workers="${3:-6}"
minimum_free_kb=$((8 * 1024 * 1024))
lock_file="work/expanded-final-holdout-run.lock"

mkdir -p work outputs
exec 9>"$lock_file"
if ! flock -n 9; then
  echo "Another final-holdout runner is already active. No second runner was started." >&2
  exit 5
fi

if ! [[ "$start_shard" =~ ^[0-9]+$ && "$end_shard" =~ ^[0-9]+$ && "$workers" =~ ^[0-9]+$ ]]; then
  echo "Shard numbers and worker count must be whole numbers." >&2
  exit 2
fi
if (( start_shard < 1 || end_shard > 9 || start_shard > end_shard )); then
  echo "Choose a shard range between 1 and 9, with the start not greater than the end." >&2
  exit 2
fi
if (( workers < 1 )); then
  echo "Worker count must be at least 1." >&2
  exit 2
fi

for ((number = start_shard; number <= end_shard; number++)); do
  shard=$(printf "configs/expanded-final-holdout-shards/shard-%02d-of-09.json" "$number")
  if [[ ! -f "$shard" ]]; then
    echo "Missing final-holdout shard: $shard" >&2
    exit 2
  fi
  free_kb=$(df -Pk . | awk 'NR == 2 {print $4}')
  if (( free_kb < minimum_free_kb )); then
    echo "Stopping before shard $number because less than 8 GiB is free." >&2
    exit 4
  fi

  echo "Running final-holdout shard $(printf '%02d' "$number") of 09"
  PYTHONPATH=src python3 -m deplab batch \
    --manifest "$shard" \
    --output outputs/expanded-final-holdout-results.jsonl \
    --run-root work/expanded-final-holdout-runs \
    --cache-dir work/expanded-final-holdout-cache \
    --cache-scope shared \
    --cleanup-environments \
    --workers "$workers" \
    --timeout 180
done

echo "Final-holdout shards $start_shard through $end_shard completed."
