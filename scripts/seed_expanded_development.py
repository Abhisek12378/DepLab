from __future__ import annotations

import json
from pathlib import Path

from deplab.batch import load_manifest


ROOT = Path(__file__).resolve().parents[1]
MATRIX = ROOT / "configs/expanded-development-matrix.json"
OUTPUT = ROOT / "outputs/expanded-development-results.jsonl"
SYSTEMATIC_CANDIDATES = [
    ROOT / "outputs/systematic-main-full.jsonl",
    ROOT / "outputs/systematic-main.jsonl",
]


def rows(path: Path):
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            yield json.loads(line)


def main() -> None:
    allowed = {spec.experiment_id for spec in load_manifest(MATRIX)}
    existing = {row["experiment_id"] for row in rows(OUTPUT)} if OUTPUT.exists() else set()
    systematic = next((path for path in SYSTEMATIC_CANDIDATES if path.exists()), None)
    if systematic is None:
        raise SystemExit(
            "missing systematic results: expected systematic-main-full.jsonl or systematic-main.jsonl"
        )
    sources = [systematic, ROOT / "outputs/external-test-results.jsonl"]
    missing = [str(path) for path in sources if not path.exists()]
    if missing:
        raise SystemExit(f"missing result source: {', '.join(missing)}")
    candidates = {}
    for source in sources:
        for row in rows(source):
            experiment_id = row["experiment_id"]
            if experiment_id not in allowed:
                raise SystemExit(f"existing result {experiment_id} is outside the development matrix")
            previous = candidates.setdefault(experiment_id, row)
            if previous != row:
                raise SystemExit(f"conflicting existing results for {experiment_id}")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    added = 0
    with OUTPUT.open("a", encoding="utf-8", newline="\n") as file:
        for experiment_id, row in candidates.items():
            if experiment_id not in existing:
                file.write(json.dumps(row, sort_keys=True) + "\n")
                existing.add(experiment_id)
                added += 1
    print(
        json.dumps(
            {
                "eligible_existing_results": len(candidates),
                "added": added,
                "output_completed": len(existing & allowed),
                "development_matrix": len(allowed),
                "remaining": len(allowed - existing),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
