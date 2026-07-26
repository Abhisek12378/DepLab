from __future__ import annotations

import argparse
import json
from pathlib import Path

from deplab.dataset_execution import audit_result_file


ROOT = Path(__file__).resolve().parents[1]
SPLITS = {
    "development": (
        ROOT / "configs/large-development-matrix-v3.0.0.json",
        ROOT / "outputs/large-development-results-v3.0.0.jsonl",
    ),
    "validation": (
        ROOT / "configs/large-validation-matrix-v3.0.0.json",
        ROOT / "outputs/large-validation-results-v3.0.0.jsonl",
    ),
    "final_test": (
        ROOT / "configs/large-final-test-matrix-v3.0.0.json",
        ROOT / "outputs/large-final-test-results-v3.0.0.jsonl",
    ),
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit one DepLab large-dataset split")
    parser.add_argument("--split", choices=tuple(SPLITS), required=True)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()

    matrix, results = SPLITS[args.split]
    summary = audit_result_file(matrix, results)
    output = ROOT / f"outputs/large-{args.split.replace('_', '-')}-audit-v3.0.0.json"
    report = ROOT / f"outputs/large-{args.split.replace('_', '-')}-audit-v3.0.0-report.md"
    output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    report.write_text(render_report(args.split, summary), encoding="utf-8")

    accepted = summary["complete"] if args.require_complete else summary["structural_valid"]
    print(
        json.dumps(
            {
                "split": args.split,
                "accepted": accepted,
                "complete": summary["complete"],
                "expected": summary["expected_experiments"],
                "observed": summary["result_rows"],
                "remaining": summary["missing_count"],
                "outcomes": summary["outcome_counts"],
                "infrastructure_failures": summary["infrastructure_failure_count"],
                "duplicates": summary["duplicate_count"],
                "summary": str(output.relative_to(ROOT)),
                "report": str(report.relative_to(ROOT)),
            },
            indent=2,
        )
    )
    return 0 if accepted else 4


def render_report(split: str, summary: dict) -> str:
    outcomes = "\n".join(
        f"- {name}: {count:,}" for name, count in summary["outcome_counts"].items()
    )
    return f"""# DepLab large {split.replace('_', ' ')} audit

- Structurally valid: **{summary['structural_valid']}**
- Complete: **{summary['complete']}**
- Expected experiments: **{summary['expected_experiments']:,}**
- Observed rows: **{summary['result_rows']:,}**
- Remaining experiments: **{summary['missing_count']:,}**
- Duplicate IDs: **{summary['duplicate_count']:,}**
- Infrastructure failures: **{summary['infrastructure_failure_count']:,}**
- Invalid specs: **{summary['invalid_spec_count']:,}**

## Outcomes

{outcomes or '- No measured rows yet.'}
"""


if __name__ == "__main__":
    raise SystemExit(main())
