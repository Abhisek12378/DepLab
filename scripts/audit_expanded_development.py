from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

from deplab.batch import load_manifest
from deplab.models import ExperimentSpec, PackagePin


ROOT = Path(__file__).resolve().parents[1]
MATRIX = ROOT / "configs/expanded-development-matrix.json"
RESULTS = ROOT / "outputs/expanded-development-results.jsonl"
SUMMARY = ROOT / "outputs/expanded-development-audit.json"
REPORT = ROOT / "outputs/expanded-development-audit-report.md"


def result_spec(row: dict) -> ExperimentSpec:
    spec = row["spec"]
    return ExperimentSpec(
        PackagePin(spec["package_a"]["name"], spec["package_a"]["version"]),
        PackagePin(spec["package_b"]["name"], spec["package_b"]["version"]),
        spec["python_version"],
        spec.get("os", "linux"),
        spec.get("architecture", "x86_64"),
    )


def main() -> None:
    matrix_payload = json.loads(MATRIX.read_text(encoding="utf-8"))
    specs = load_manifest(MATRIX)
    expected_ids = {spec.experiment_id for spec in specs}
    family_by_id = {
        spec.experiment_id: experiment["family"]
        for spec, experiment in zip(specs, matrix_payload["experiments"])
    }
    rows = [
        json.loads(line)
        for line in RESULTS.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    counts = Counter(row["experiment_id"] for row in rows)
    actual_ids = set(counts)
    duplicate_ids = sorted(item for item, count in counts.items() if count > 1)
    missing_ids = sorted(expected_ids - actual_ids)
    extra_ids = sorted(actual_ids - expected_ids)
    invalid_spec_ids = sorted(
        row["experiment_id"]
        for row in rows
        if result_spec(row).experiment_id != row["experiment_id"]
    )
    infrastructure_ids = sorted(
        row["experiment_id"]
        for row in rows
        if row["outcome"] == "infrastructure_failure"
    )

    outcomes = Counter(row["outcome"] for row in rows)
    family_outcomes: dict[str, Counter[str]] = defaultdict(Counter)
    python_outcomes: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        experiment_id = row["experiment_id"]
        if experiment_id in family_by_id:
            family_outcomes[family_by_id[experiment_id]][row["outcome"]] += 1
        python_outcomes[row["spec"]["python_version"]][row["outcome"]] += 1

    pass_rows = outcomes.get("pass", 0)
    failure_rows = len(rows) - pass_rows
    valid = not (
        duplicate_ids
        or missing_ids
        or extra_ids
        or invalid_spec_ids
        or infrastructure_ids
        or len(rows) != len(expected_ids)
    )
    summary = {
        "schema_version": "1.0.0",
        "valid": valid,
        "matrix": str(MATRIX.relative_to(ROOT)),
        "results": str(RESULTS.relative_to(ROOT)),
        "expected_experiments": len(expected_ids),
        "result_rows": len(rows),
        "unique_result_ids": len(actual_ids),
        "duplicate_ids": duplicate_ids,
        "missing_ids": missing_ids,
        "extra_ids": extra_ids,
        "invalid_spec_ids": invalid_spec_ids,
        "infrastructure_failure_ids": infrastructure_ids,
        "outcome_counts": dict(sorted(outcomes.items())),
        "pass_rows": pass_rows,
        "failure_rows": failure_rows,
        "failure_percentage": round(100 * failure_rows / len(rows), 2),
        "family_outcome_counts": {
            family: dict(sorted(counts.items()))
            for family, counts in sorted(family_outcomes.items())
        },
        "python_outcome_counts": {
            version: dict(sorted(counts.items()))
            for version, counts in sorted(python_outcomes.items())
        },
    }
    SUMMARY.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Expanded development dataset audit",
        "",
        f"- Valid: **{valid}**",
        f"- Expected experiments: {len(expected_ids):,}",
        f"- Result rows: {len(rows):,}",
        f"- Unique result IDs: {len(actual_ids):,}",
        f"- Duplicate IDs: {len(duplicate_ids):,}",
        f"- Missing IDs: {len(missing_ids):,}",
        f"- Extra IDs: {len(extra_ids):,}",
        f"- Infrastructure failures: {len(infrastructure_ids):,}",
        f"- Pass rows: {pass_rows:,}",
        f"- Failure rows: {failure_rows:,} ({summary['failure_percentage']:.2f}%)",
        "",
        "## Outcomes",
        "",
    ]
    lines.extend(f"- {name}: {count:,}" for name, count in sorted(outcomes.items()))
    lines.extend(["", "## Outcomes by family", ""])
    for family, counts in sorted(family_outcomes.items()):
        rendered = ", ".join(f"{name}={count}" for name, count in sorted(counts.items()))
        lines.append(f"- {family}: {rendered}")
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps(summary, indent=2))
    if not valid:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
