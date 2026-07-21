from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from deplab.batch import load_manifest


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def directory_sha256(directory: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(directory.glob("*.json")):
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(sha256(path)))
    return digest.hexdigest()


def read_json(relative: str) -> dict:
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def pair_packages(relative: str) -> list[str]:
    pairs = read_json(relative)["families"]
    return sorted({row[key] for row in pairs for key in ("package_a", "package_b")})


def artifact(relative: str) -> dict[str, str]:
    return {"path": relative, "sha256": sha256(ROOT / relative)}


def result_ids(relative: str) -> set[str]:
    return {
        json.loads(line)["experiment_id"]
        for line in (ROOT / relative).read_text(encoding="utf-8").splitlines()
        if line.strip()
    }


def main() -> None:
    development_matrix = "configs/expanded-development-matrix.json"
    holdout_matrix = "configs/expanded-final-holdout-matrix.json"
    development_pairs = "configs/expanded-development-pairs.json"
    holdout_pairs = "configs/expanded-final-holdout-pairs.json"
    development_shards = ROOT / "configs/expanded-development-shards"
    holdout_shards = ROOT / "configs/expanded-final-holdout-shards"

    development_ids = {
        spec.experiment_id for spec in load_manifest(ROOT / development_matrix)
    }
    holdout_ids = {spec.experiment_id for spec in load_manifest(ROOT / holdout_matrix)}
    development_packages = pair_packages(development_pairs)
    holdout_packages = pair_packages(holdout_pairs)
    if development_ids & holdout_ids:
        raise SystemExit("development and final-holdout experiment IDs overlap")
    if set(development_packages) & set(holdout_packages):
        raise SystemExit("development and final-holdout package names overlap")

    systematic_candidates = [
        "outputs/systematic-main-full.jsonl",
        "outputs/systematic-main.jsonl",
    ]
    systematic = next(
        (path for path in systematic_candidates if (ROOT / path).exists()), None
    )
    if systematic is None:
        raise SystemExit(
            "missing systematic results: expected systematic-main-full.jsonl or systematic-main.jsonl"
        )
    reused_paths = [systematic, "outputs/external-test-results.jsonl"]
    reused_ids = set().union(*(result_ids(path) for path in reused_paths))
    if not reused_ids <= development_ids:
        raise SystemExit("an existing result is outside the expanded development matrix")

    frozen_inputs = [
        "configs/expanded-scope.json",
        development_pairs,
        holdout_pairs,
        "configs/changelog-sources.json",
        "outputs/changelog-catalog-expanded-v1.2.0.jsonl",
        "src/deplab/smoke.py",
    ]
    payload = {
        "schema_version": "1.0.0",
        "freeze_id": "deplab-expanded-matrices-v1.0.0",
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "target_platform": "linux_x86_64",
        "python_versions": ["3.10", "3.11", "3.12"],
        "execution_policy": {
            "development": "Outcomes may be inspected and used for feature and model development.",
            "final_holdout": (
                "Keep outcomes sealed. Run only after the feature list, trained model, "
                "decision threshold, and evaluation script are frozen."
            ),
            "inference_inputs": (
                "PyPI metadata, wheel metadata, version pins, Python target, and version-pinned "
                "changelog signals only. Import and smoke-test outcomes are labels, never inputs."
            ),
        },
        "splits": {
            "development": {
                **artifact(development_matrix),
                "experiments": len(development_ids),
                "families": len(read_json(development_pairs)["families"]),
                "package_names": development_packages,
                "shards": len(list(development_shards.glob("*.json"))),
                "shard_size": 100,
                "shard_directory": "configs/expanded-development-shards",
                "shard_directory_sha256": directory_sha256(development_shards),
            },
            "final_holdout": {
                **artifact(holdout_matrix),
                "experiments": len(holdout_ids),
                "families": len(read_json(holdout_pairs)["families"]),
                "package_names": holdout_packages,
                "shards": len(list(holdout_shards.glob("*.json"))),
                "shard_size": 100,
                "shard_directory": "configs/expanded-final-holdout-shards",
                "shard_directory_sha256": directory_sha256(holdout_shards),
            },
        },
        "existing_development_results": {
            "sources": [artifact(path) for path in reused_paths],
            "unique_experiments": len(reused_ids),
            "all_inside_development_matrix": True,
            "remaining_new_development_experiments": len(development_ids - reused_ids),
        },
        "frozen_inputs": [artifact(path) for path in frozen_inputs],
    }
    output = ROOT / "configs/expanded-matrices-freeze-v1.0.0.json"
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "development_experiments": len(development_ids),
                "final_holdout_experiments": len(holdout_ids),
                "total_experiments": len(development_ids) + len(holdout_ids),
                "reused_development_results": len(reused_ids),
                "new_development_runs": len(development_ids - reused_ids),
                "output": str(output.relative_to(ROOT)),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
