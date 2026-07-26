from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from deplab.scope_plan import read_and_validate_plan, write_pair_definitions


ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "configs/large-dataset-plan-v3.0.0.json"
SUMMARY = ROOT / "outputs/large-dataset-plan-v3.0.0-summary.json"
REPORT = ROOT / "outputs/large-dataset-plan-v3.0.0-report.md"


def main() -> None:
    plan, _ = read_and_validate_plan(PLAN)
    summary = write_pair_definitions(PLAN, ROOT / "configs")
    payload = {
        **asdict(summary),
        "known_measured_rows_reusable_for_development": 4109,
        "exact_experiment_count_status": (
            "Pending PyPI release selection and seven-Python wheel audit."
        ),
        "estimated_wheel_eligible_experiments": {
            "lower": round(summary.maximum_cartesian_experiments * 0.70),
            "upper": round(summary.maximum_cartesian_experiments * 0.90),
        },
        "coverage_labels": [
            "eligible",
            "requires_python_excluded",
            "wheel_unavailable",
            "incompatible_wheel_tags",
            "yanked_release",
            "all_wheels_yanked",
        ],
        "measured_outcomes": [
            "pass",
            "resolution_failure",
            "installation_failure",
            "import_failure",
            "smoke_test_failure",
            "timeout",
        ],
        "infrastructure_policy": "Infrastructure failures are repaired and retried, never model labels.",
    }
    SUMMARY.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    REPORT.write_text(_report(plan, payload), encoding="utf-8")
    print(json.dumps(payload, indent=2))


def _report(plan: dict, summary: dict) -> str:
    package_groups = {
        split: [
            name
            for name, package in plan["packages"].items()
            if package["split"] == split
        ]
        for split in ("development", "validation", "final_test")
    }
    return f"""# DepLab large-dataset plan v3.0.0

Frozen planning cutoff: **{plan['release_cutoff']}**

## Planned scale

- Packages: **{summary['packages']}**
- Package-pair families: **{summary['families']}**
- Versions per package: **{summary['target_versions_per_package']}**
- Python targets: **{', '.join(plan['coverage_order'])}**
- Platform: **{plan['target_platform']}**
- Maximum pre-audit Cartesian rows: **{summary['maximum_cartesian_experiments']:,}**
- Estimated wheel-eligible rows: **{summary['estimated_wheel_eligible_experiments']['lower']:,}–{summary['estimated_wheel_eligible_experiments']['upper']:,}**
- Existing measured rows reusable for development: **4,109**

The exact runnable count is intentionally not claimed yet. It will be known only
after the official PyPI metadata and wheel audit for all seven Python targets.

## Leakage-safe splits

- Development: **{summary['split_packages']['development']} packages / {summary['split_families']['development']} families**
  - {', '.join(package_groups['development'])}
- Validation: **{summary['split_packages']['validation']} packages / {summary['split_families']['validation']} families**
  - {', '.join(package_groups['validation'])}
- Final test: **{summary['split_packages']['final_test']} packages / {summary['split_families']['final_test']} families**
  - {', '.join(package_groups['final_test'])}

Package names do not cross splits. The final-test outcomes must remain sealed
until the feature schema, trained model, thresholds, and evaluation code are frozen.

## Deterministic coverage facts

The scope audit records why a release/Python target cannot be scheduled:
Requires-Python exclusion, no wheel, incompatible wheel tags, yanked release,
or all wheels yanked. These records are facts and are not compatibility labels.

## Measured labels

Runnable experiments may produce pass, resolution failure, installation failure,
import failure, smoke-test failure, or timeout. Infrastructure failures are
repaired and retried and never enter the model dataset.

## Next preparation command

Run the following on a Linux machine with network access:

```bash
PYTHONPATH=src python3 -m deplab scope-plan \\
  --plan configs/large-dataset-plan-v3.0.0.json \\
  --output configs/large-scope-draft-v3.0.0.json
```

This selects the exact releases using the frozen cutoff. The next command will
audit their wheels for Python 3.8 through 3.14 before any installation matrix is run.
"""


if __name__ == "__main__":
    main()
