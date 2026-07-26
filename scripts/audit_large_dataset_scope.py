from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

from deplab.batch import load_manifest
from deplab.scope_plan import read_and_validate_plan


ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "configs/large-dataset-plan-v3.0.0.json"
SCOPE = ROOT / "configs/large-scope-v3.0.0.json"
OUTPUT = ROOT / "outputs/large-dataset-scope-audit-v3.0.0.json"
REPORT = ROOT / "outputs/large-dataset-scope-audit-v3.0.0-report.md"
SPLITS = {
    "development": (
        "configs/large-development-pairs-v3.0.0.json",
        "configs/large-development-matrix-v3.0.0.json",
    ),
    "validation": (
        "configs/large-validation-pairs-v3.0.0.json",
        "configs/large-validation-matrix-v3.0.0.json",
    ),
    "final_test": (
        "configs/large-final-test-pairs-v3.0.0.json",
        "configs/large-final-test-matrix-v3.0.0.json",
    ),
}


def main() -> None:
    plan, plan_summary = read_and_validate_plan(PLAN)
    scope = read_json(SCOPE)
    errors = validate_scope(plan, scope)
    split_rows = {}
    split_ids = {}
    total_candidates = 0
    for split, (pairs_path, matrix_path) in SPLITS.items():
        pairs = read_json(ROOT / pairs_path)
        matrix = read_json(ROOT / matrix_path)
        specs = load_manifest(ROOT / matrix_path)
        identifiers = {spec.experiment_id for spec in specs}
        expected_candidates = candidate_count(scope, pairs)
        exclusions = list(matrix.get("coverage_exclusions") or [])
        actual_candidates = len(specs) + len(exclusions)
        if expected_candidates != actual_candidates:
            errors.append(
                f"{split} candidate count is {actual_candidates}, expected {expected_candidates}"
            )
        if len(identifiers) != len(specs):
            errors.append(f"{split} matrix has duplicate experiment IDs")
        reasons = Counter()
        for row in exclusions:
            for key in ("package_a_coverage", "package_b_coverage"):
                detail = row.get(key) or {}
                if not detail.get("eligible", False):
                    reasons[str(detail.get("reason") or "unknown")] += 1
        split_ids[split] = identifiers
        split_rows[split] = {
            "pairs": pairs_path,
            "matrix": matrix_path,
            "matrix_sha256": sha256(ROOT / matrix_path),
            "families": len(pairs["families"]),
            "runnable_experiments": len(specs),
            "coverage_exclusions": len(exclusions),
            "candidate_combinations": actual_candidates,
            "coverage_reason_counts": dict(sorted(reasons.items())),
        }
        total_candidates += actual_candidates

    split_names = list(SPLITS)
    for index, left in enumerate(split_names):
        for right in split_names[index + 1 :]:
            if split_ids[left] & split_ids[right]:
                errors.append(f"{left} and {right} experiment IDs overlap")

    known_ids = result_ids(
        ROOT / "outputs/expanded-development-results.jsonl",
        ROOT / "outputs/expanded-final-holdout-results.jsonl",
    )
    if len(known_ids) != 4109:
        errors.append(f"known result count is {len(known_ids)}, expected 4109")
    if not known_ids <= split_ids["development"]:
        errors.append("not all 4,109 known results are reusable development rows")

    payload = {
        "schema_version": "3.0.0",
        "valid": not errors,
        "errors": errors,
        "plan": str(PLAN.relative_to(ROOT)),
        "scope": str(SCOPE.relative_to(ROOT)),
        "scope_sha256": sha256(SCOPE),
        "packages": plan_summary.packages,
        "selected_releases": sum(
            len(package["versions"]) for package in scope["packages"].values()
        ),
        "python_versions": plan["coverage_order"],
        "splits": split_rows,
        "known_results_reused_in_development": len(known_ids),
        "runnable_experiments": sum(
            row["runnable_experiments"] for row in split_rows.values()
        ),
        "coverage_exclusions": sum(
            row["coverage_exclusions"] for row in split_rows.values()
        ),
        "candidate_combinations": total_candidates,
    }
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    REPORT.write_text(render_report(payload), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    if errors:
        raise SystemExit("large-dataset scope audit failed")


def validate_scope(plan: dict, scope: dict) -> list[str]:
    errors = []
    if scope.get("coverage_order") != plan["coverage_order"]:
        errors.append("scope Python coverage does not match the plan")
    if set(scope.get("packages") or {}) != set(plan["packages"]):
        errors.append("scope package names do not match the plan")
    target = int(plan["target_versions_per_package"])
    for name, package in dict(scope.get("packages") or {}).items():
        definition = plan["packages"][name]
        minimum = int(
            definition.get(
                "minimum_versions",
                plan["minimum_versions_per_package"],
            )
        )
        count = len(package.get("versions") or [])
        if not minimum <= count <= target:
            errors.append(
                f"package {name} has {count} selected releases; expected {minimum}–{target}"
            )
    return errors


def candidate_count(scope: dict, pairs: dict) -> int:
    packages = scope["packages"]
    python_count = len(pairs["python_versions"])
    return sum(
        len(packages[family["package_a"]]["versions"])
        * len(packages[family["package_b"]]["versions"])
        * python_count
        for family in pairs["families"]
    )


def result_ids(*paths: Path) -> set[str]:
    identifiers = set()
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                identifiers.add(json.loads(line)["experiment_id"])
    return identifiers


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def render_report(payload: dict) -> str:
    rows = "\n".join(
        (
            f"| {name.replace('_', ' ').title()} | {split['families']} | "
            f"{split['runnable_experiments']:,} | {split['coverage_exclusions']:,} |"
        )
        for name, split in payload["splits"].items()
    )
    return f"""# DepLab large-dataset scope audit v3.0.0

- Valid: **{payload['valid']}**
- Packages: **{payload['packages']}**
- Selected releases: **{payload['selected_releases']}**
- Python targets: **{', '.join(payload['python_versions'])}**
- Runnable experiments: **{payload['runnable_experiments']:,}**
- Deterministic coverage exclusions: **{payload['coverage_exclusions']:,}**
- Total candidate combinations: **{payload['candidate_combinations']:,}**
- Known rows reused in development: **{payload['known_results_reused_in_development']:,}**

| Split | Families | Runnable | Coverage exclusions |
|---|---:|---:|---:|
{rows}

Coverage exclusions are deterministic facts and are not learned failure labels.
Final-test outcomes must remain sealed until the model and evaluation pipeline are frozen.
"""


if __name__ == "__main__":
    main()
