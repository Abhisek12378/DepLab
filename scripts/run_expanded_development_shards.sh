#!/usr/bin/env bash
set -euo pipefail

start_shard="${1:-1}"
end_shard="${2:-33}"
workers="${3:-1}"
minimum_free_kb=$((8 * 1024 * 1024))
lock_file="work/expanded-development-run.lock"

mkdir -p work
exec 9>"$lock_file"
if ! flock -n 9; then
  echo "Another expanded development runner is already active. No second runner was started." >&2
  exit 5
fi

if ! [[ "$start_shard" =~ ^[0-9]+$ && "$end_shard" =~ ^[0-9]+$ && "$workers" =~ ^[0-9]+$ ]]; then
  echo "Shard numbers and worker count must be whole numbers." >&2
  exit 2
fi
if (( start_shard < 1 || end_shard > 33 || start_shard > end_shard )); then
  echo "Choose a shard range between 1 and 33, with the start not greater than the end." >&2
  exit 2
fi
if (( workers < 1 )); then
  echo "Worker count must be at least 1." >&2
  exit 2
fi

for ((number = start_shard; number <= end_shard; number++)); do
  shard=$(printf "configs/expanded-development-shards/shard-%02d-of-33.json" "$number")
  if [[ ! -f "$shard" ]]; then
    echo "Missing development shard: $shard" >&2
    exit 2
  fi
  free_kb=$(df -Pk . | awk 'NR == 2 {print $4}')
  if (( free_kb < minimum_free_kb )); then
    echo "Stopping before shard $number because less than 8 GiB is free." >&2
    exit 4
  fi

  echo "Running expanded development shard $(printf '%02d' "$number") of 33"
  PYTHONPATH=src python3 -m deplab batch \
    --manifest "$shard" \
    --output outputs/expanded-development-results.jsonl \
    --run-root work/expanded-development-runs \
    --cache-dir work/expanded-development-cache \
    --cache-scope shared \
    --cleanup-environments \
    --workers "$workers" \
    --timeout 180
done

echo "Expanded development shards $start_shard through $end_shard completed."
