from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


MODEL_NAME = "answerdotai/ModernBERT-base"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Reuse exact matching frozen ModernBERT release embeddings"
    )
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--existing-embeddings", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    evidence = {
        key(row): row for row in read_jsonl(args.evidence)
    }
    output_rows = read_jsonl(args.output)
    validate_output_rows(output_rows, evidence)
    completed = {key(row) for row in output_rows}
    reusable = []
    for row in read_jsonl(args.existing_embeddings):
        release_key = key(row)
        source = evidence.get(release_key)
        if source is None or release_key in completed:
            continue
        if row.get("selected_text_sha256") != source["selected_text_sha256"]:
            continue
        if row.get("model") != MODEL_NAME:
            continue
        reusable.append(row)
        completed.add(release_key)
    append_rows(args.output, reusable)
    print(
        json.dumps(
            {
                "evidence_rows": len(evidence),
                "existing_output_rows": len(output_rows),
                "reused_embeddings": len(reusable),
                "output_rows": len(output_rows) + len(reusable),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def validate_output_rows(
    rows: list[dict[str, Any]],
    evidence: dict[tuple[str, str], dict[str, Any]],
) -> None:
    keys = [key(row) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("popular embedding output contains duplicate rows")
    invalid = []
    for row in rows:
        source = evidence.get(key(row))
        if (
            source is None
            or row.get("selected_text_sha256") != source["selected_text_sha256"]
            or row.get("model") != MODEL_NAME
        ):
            invalid.append(key(row))
    if invalid:
        raise ValueError(
            "popular embedding output contains rows from different evidence "
            f"or model: {invalid[:5]}"
        )


def key(row: dict[str, Any]) -> tuple[str, str]:
    package = str(row["package"]).lower().replace("_", "-").replace(".", "-")
    return package, str(row["version"])


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def append_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as file:
        for row in rows:
            file.write(json.dumps(row, separators=(",", ":"), sort_keys=True) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
