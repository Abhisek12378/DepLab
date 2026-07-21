#!/usr/bin/env bash
set -euo pipefail

start_shard="${1:-1}"
end_shard="${2:-13}"
minimum_free_kb=$((4 * 1024 * 1024))

if ! [[ "$start_shard" =~ ^[0-9]+$ && "$end_shard" =~ ^[0-9]+$ ]]; then
  echo "Shard numbers must be whole numbers." >&2
  exit 2
fi
if (( start_shard < 1 || end_shard > 13 || start_shard > end_shard )); then
  echo "Choose a shard range between 1 and 13, with the start not greater than the end." >&2
  exit 2
fi

for ((number = start_shard; number <= end_shard; number++)); do
  shard=$(printf "configs/systematic-shards/shard-%02d-of-13.json" "$number")
  if [[ ! -f "$shard" ]]; then
    echo "Missing shard manifest: $shard" >&2
    exit 2
  fi
  free_kb=$(df -Pk . | awk 'NR == 2 {print $4}')
  if (( free_kb < minimum_free_kb )); then
    echo "Stopping before shard $number because less than 4 GiB is free." >&2
    exit 4
  fi

  echo "Running systematic shard $(printf '%02d' "$number") of 13"
  PYTHONPATH=src python3 -m deplab batch \
    --manifest "$shard" \
    --output outputs/systematic-main.jsonl \
    --run-root work/systematic-runs \
    --cache-dir work/systematic-cache \
    --cache-scope shared \
    --cleanup-environments \
    --workers 1 \
    --timeout 180
done

echo "Systematic shards $start_shard through $end_shard completed."
