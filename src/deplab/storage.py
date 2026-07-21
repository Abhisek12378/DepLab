from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def completed_ids(path: Path) -> set[str]:
    return {
        record["experiment_id"]
        for record in read_jsonl(path)
        if "experiment_id" in record and record.get("outcome") != "infrastructure_failure"
    }
