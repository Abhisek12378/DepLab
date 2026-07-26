from __future__ import annotations

import argparse
from pathlib import Path

from deplab.result_repair import repair_result_file, summary_json


ROOT = Path(__file__).resolve().parents[1]
RESULTS = {
    "development": ROOT / "outputs/large-development-results-v3.0.0.jsonl",
    "validation": ROOT / "outputs/large-validation-results-v3.0.0.jsonl",
    "final_test": ROOT / "outputs/large-final-test-results-v3.0.0.jsonl",
}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Remove duplicate and infrastructure-only rows before retrying a split"
    )
    parser.add_argument("--split", choices=tuple(RESULTS), required=True)
    args = parser.parse_args()
    path = RESULTS[args.split]
    if not path.exists():
        print(f"No result file exists yet: {path.relative_to(ROOT)}")
        return 0
    backup = path.with_name(f"{path.stem}.retry-backup{path.suffix}")
    print(summary_json(repair_result_file(path, backup_path=backup)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
