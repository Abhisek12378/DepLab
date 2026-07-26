from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from .batch import load_manifest
from .models import ExperimentSpec, PackagePin


class DatasetExecutionError(ValueError):
    pass


MEASURED_OUTCOMES = {
    "pass",
    "resolution_failure",
    "installation_failure",
    "import_failure",
    "smoke_test_failure",
    "timeout",
}


@dataclass(frozen=True)
class SeedSummary:
    matrix: str
    output: str
    matrix_experiments: int
    source_rows: int
    unique_seed_rows: int
    output_rows: int
    remaining: int


def seed_result_file(
    matrix_path: Path,
    source_paths: Iterable[Path],
    output_path: Path,
) -> SeedSummary:
    specs = load_manifest(matrix_path)
    ordered_ids = [spec.experiment_id for spec in specs]
    allowed_ids = set(ordered_ids)
    rows_by_id: dict[str, dict[str, Any]] = {}
    source_ids: set[str] = set()
    source_rows = 0

    inputs = list(source_paths)
    for path in inputs:
        for row in read_jsonl(path):
            source_rows += 1
            experiment_id = validate_result_row(row)
            source_ids.add(experiment_id)
            if experiment_id not in allowed_ids:
                raise DatasetExecutionError(
                    f"result {experiment_id} from {path} is outside {matrix_path}"
                )
            previous = rows_by_id.setdefault(experiment_id, row)
            if previous != row:
                raise DatasetExecutionError(
                    f"conflicting result rows for experiment {experiment_id}"
                )
    if output_path.exists() and output_path not in inputs:
        for row in read_jsonl(output_path):
            experiment_id = validate_result_row(row)
            if experiment_id not in allowed_ids:
                raise DatasetExecutionError(
                    f"result {experiment_id} from {output_path} is outside {matrix_path}"
                )
            previous = rows_by_id.setdefault(experiment_id, row)
            if previous != row:
                raise DatasetExecutionError(
                    f"conflicting result rows for experiment {experiment_id}"
                )

    ordered_rows = [
        rows_by_id[experiment_id]
        for experiment_id in ordered_ids
        if experiment_id in rows_by_id
    ]
    write_jsonl_atomic(output_path, ordered_rows)
    return SeedSummary(
        matrix=str(matrix_path),
        output=str(output_path),
        matrix_experiments=len(ordered_ids),
        source_rows=source_rows,
        unique_seed_rows=len(source_ids),
        output_rows=len(ordered_rows),
        remaining=len(ordered_ids) - len(ordered_rows),
    )


def audit_result_file(matrix_path: Path, result_path: Path) -> dict[str, Any]:
    matrix = read_json(matrix_path)
    specs = load_manifest(matrix_path)
    expected_ids = {spec.experiment_id for spec in specs}
    family_by_id = {
        spec.experiment_id: str(row["family"])
        for spec, row in zip(specs, matrix["experiments"])
    }
    rows = read_jsonl(result_path) if result_path.exists() else []
    counts = Counter(str(row.get("experiment_id") or "") for row in rows)
    actual_ids = set(counts)
    duplicates = sorted(key for key, count in counts.items() if count > 1)
    missing = sorted(expected_ids - actual_ids)
    extra = sorted(actual_ids - expected_ids)
    invalid_specs: list[str] = []
    infrastructure: list[str] = []
    invalid_outcomes: list[str] = []
    unmeasured: list[str] = []
    outcomes: Counter[str] = Counter()
    family_outcomes: dict[str, Counter[str]] = defaultdict(Counter)
    python_outcomes: dict[str, Counter[str]] = defaultdict(Counter)

    for row in rows:
        experiment_id = str(row.get("experiment_id") or "")
        outcome = str(row.get("outcome") or "")
        outcomes[outcome] += 1
        try:
            calculated_id = result_spec(row).experiment_id
        except (KeyError, TypeError, ValueError):
            calculated_id = ""
        if calculated_id != experiment_id:
            invalid_specs.append(experiment_id)
        if outcome == "infrastructure_failure":
            infrastructure.append(experiment_id)
        elif outcome not in MEASURED_OUTCOMES:
            invalid_outcomes.append(experiment_id)
        if row.get("measured") is not True:
            unmeasured.append(experiment_id)
        if experiment_id in family_by_id:
            family_outcomes[family_by_id[experiment_id]][outcome] += 1
        python_version = str((row.get("spec") or {}).get("python_version") or "unknown")
        python_outcomes[python_version][outcome] += 1

    structural_valid = not (
        duplicates
        or extra
        or invalid_specs
        or infrastructure
        or invalid_outcomes
        or unmeasured
    )
    complete = structural_valid and not missing and len(rows) == len(expected_ids)
    return {
        "schema_version": "3.0.0",
        "matrix": str(matrix_path),
        "matrix_sha256": sha256(matrix_path),
        "results": str(result_path),
        "results_sha256": sha256(result_path) if result_path.exists() else None,
        "structural_valid": structural_valid,
        "complete": complete,
        "expected_experiments": len(expected_ids),
        "result_rows": len(rows),
        "unique_result_ids": len(actual_ids),
        "duplicate_count": len(duplicates),
        "duplicate_ids": duplicates[:100],
        "missing_count": len(missing),
        "missing_ids_preview": missing[:100],
        "extra_count": len(extra),
        "extra_ids": extra[:100],
        "invalid_spec_count": len(invalid_specs),
        "invalid_spec_ids": sorted(invalid_specs)[:100],
        "infrastructure_failure_count": len(infrastructure),
        "infrastructure_failure_ids": sorted(infrastructure)[:100],
        "invalid_outcome_count": len(invalid_outcomes),
        "unmeasured_count": len(unmeasured),
        "outcome_counts": dict(sorted(outcomes.items())),
        "family_outcome_counts": {
            family: dict(sorted(values.items()))
            for family, values in sorted(family_outcomes.items())
        },
        "python_outcome_counts": {
            version: dict(sorted(values.items()))
            for version, values in sorted(python_outcomes.items())
        },
    }


def validate_result_row(row: dict[str, Any]) -> str:
    experiment_id = row.get("experiment_id")
    if not isinstance(experiment_id, str) or not experiment_id:
        raise DatasetExecutionError("result row has no experiment_id")
    if result_spec(row).experiment_id != experiment_id:
        raise DatasetExecutionError(f"result {experiment_id} has an invalid spec")
    outcome = row.get("outcome")
    if outcome not in MEASURED_OUTCOMES:
        raise DatasetExecutionError(
            f"result {experiment_id} has non-reusable outcome {outcome!r}"
        )
    if row.get("measured") is not True:
        raise DatasetExecutionError(f"result {experiment_id} is not marked measured")
    return experiment_id


def result_spec(row: dict[str, Any]) -> ExperimentSpec:
    spec = row["spec"]
    package_a = spec["package_a"]
    package_b = spec["package_b"]
    return ExperimentSpec(
        PackagePin(str(package_a["name"]), str(package_a["version"])),
        PackagePin(str(package_b["name"]), str(package_b["version"])),
        str(spec["python_version"]),
        str(spec.get("os", "linux")),
        str(spec.get("architecture", "x86_64")),
    )


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DatasetExecutionError(f"cannot read JSON file {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise DatasetExecutionError(f"{path} must contain a JSON object")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
        for line_number, line in enumerate(lines, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise DatasetExecutionError(f"{path} line {line_number} is not an object")
            rows.append(row)
    except (OSError, json.JSONDecodeError) as exc:
        raise DatasetExecutionError(f"cannot read JSONL file {path}: {exc}") from exc
    return rows


def write_jsonl_atomic(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".writing")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    temporary.replace(path)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def seed_summary_json(summary: SeedSummary) -> str:
    return json.dumps(asdict(summary), indent=2, sort_keys=True)
