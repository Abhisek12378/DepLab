from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict
from pathlib import Path

from deplab.dataset_execution import (
    DatasetExecutionError,
    seed_result_file,
    sha256,
)
from deplab.shards import shard_manifest


ROOT = Path(__file__).resolve().parents[1]
SCOPE_AUDIT = ROOT / "outputs/large-dataset-scope-audit-v3.0.0.json"
FREEZE = ROOT / "configs/large-execution-freeze-v3.0.0.json"
PILOT = ROOT / "configs/large-runtime-pilot-v3.0.0.json"
SHARD_SIZE = 100
SPLITS = {
    "development": {
        "matrix": ROOT / "configs/large-development-matrix-v3.0.0.json",
        "shards": ROOT / "configs/large-development-shards-v3.0.0",
        "output": ROOT / "outputs/large-development-results-v3.0.0.jsonl",
    },
    "validation": {
        "matrix": ROOT / "configs/large-validation-matrix-v3.0.0.json",
        "shards": ROOT / "configs/large-validation-shards-v3.0.0",
        "output": ROOT / "outputs/large-validation-results-v3.0.0.jsonl",
    },
    "final_test": {
        "matrix": ROOT / "configs/large-final-test-matrix-v3.0.0.json",
        "shards": ROOT / "configs/large-final-test-shards-v3.0.0",
        "output": ROOT / "outputs/large-final-test-results-v3.0.0.jsonl",
    },
}
KNOWN_SOURCES = [
    ROOT / "outputs/expanded-development-results.jsonl",
    ROOT / "outputs/expanded-final-holdout-results.jsonl",
]


def main() -> None:
    os.chdir(ROOT)
    scope_audit = read_object(SCOPE_AUDIT)
    if scope_audit.get("valid") is not True:
        raise DatasetExecutionError("large-dataset scope audit is not valid")

    split_freeze = {}
    for split, paths in SPLITS.items():
        expected_hash = scope_audit["splits"][split]["matrix_sha256"]
        actual_hash = sha256(paths["matrix"])
        if actual_hash != expected_hash:
            raise DatasetExecutionError(
                f"{split} matrix hash is {actual_hash}, expected {expected_hash}"
            )
        matrix_relative = paths["matrix"].relative_to(ROOT)
        shards_relative = paths["shards"].relative_to(ROOT)
        summary = shard_manifest(matrix_relative, shards_relative, SHARD_SIZE)
        split_freeze[split] = {
            **asdict(summary),
            "matrix_sha256": actual_hash,
            "shard_set_sha256": shard_set_sha256(paths["shards"]),
            "result_output": relative_path(paths["output"]),
        }

    pilot_summary = write_runtime_pilot(SPLITS["development"]["matrix"], PILOT)
    seed_summary = seed_result_file(
        SPLITS["development"]["matrix"],
        KNOWN_SOURCES,
        SPLITS["development"]["output"],
    )
    if seed_summary.unique_seed_rows != 4_109:
        raise DatasetExecutionError(
            f"seeded {seed_summary.unique_seed_rows} known rows, expected 4109"
        )

    payload = {
        "schema_version": "3.0.0",
        "scope_audit": relative_path(SCOPE_AUDIT),
        "scope_audit_sha256": sha256(SCOPE_AUDIT),
        "shard_size": SHARD_SIZE,
        "splits": split_freeze,
        "known_result_sources": [
            {
                "path": relative_path(path),
                "rows": line_count(path),
                "sha256": sha256(path),
            }
            for path in KNOWN_SOURCES
        ],
        "development_seed": {
            "matrix": relative_path(SPLITS["development"]["matrix"]),
            "output": relative_path(SPLITS["development"]["output"]),
            "known_rows": seed_summary.unique_seed_rows,
            "initial_remaining": (
                seed_summary.matrix_experiments - seed_summary.unique_seed_rows
            ),
        },
        "runtime_pilot": pilot_summary,
        "execution_order": [
            "development",
            "validation_after_model_pipeline_is_frozen",
            "final_test_after_model_and_evaluation_are_frozen",
        ],
        "final_test_lock": "FROZEN_MODEL_AND_EVALUATION",
    }
    FREEZE.write_bytes((json.dumps(payload, indent=2) + "\n").encode("utf-8"))
    print(json.dumps(payload, indent=2))


def write_runtime_pilot(matrix_path: Path, output_path: Path) -> dict:
    matrix = read_object(matrix_path)
    expected_versions = ["3.8", "3.9", "3.10", "3.11", "3.12", "3.13", "3.14"]
    candidates = [
        row
        for row in matrix["experiments"]
        if row["family"] == "requests-urllib3"
        and row["package_a"] == "requests==2.32.3"
        and row["package_b"] == "urllib3==2.2.3"
    ]
    by_python = {str(row["python"]): row for row in candidates}
    missing = [version for version in expected_versions if version not in by_python]
    if missing:
        raise DatasetExecutionError(
            f"runtime pilot has no eligible experiment for Python {', '.join(missing)}"
        )
    rows = [by_python[version] for version in expected_versions]
    payload = {
        "schema_version": "3.0.0",
        "description": (
            "Seven lightweight experiments used to verify CPython 3.8 through 3.14 "
            "before any large-dataset shard is started."
        ),
        "source_manifest": relative_path(matrix_path),
        "experiments": rows,
    }
    output_path.write_bytes((json.dumps(payload, indent=2) + "\n").encode("utf-8"))
    return {
        "manifest": relative_path(output_path),
        "experiments": len(rows),
        "python_versions": expected_versions,
        "sha256": sha256(output_path),
    }


def shard_set_sha256(directory: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(directory.glob("shard-*.json")):
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()


def line_count(path: Path) -> int:
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def relative_path(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def read_object(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise DatasetExecutionError(f"{path} must contain a JSON object")
    return value


if __name__ == "__main__":
    main()
