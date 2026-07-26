from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from train_expanded_baseline import _metrics, _to_bool
    from train_large_hybrid import CANDIDATES, select_candidate, subtype_metrics
except ImportError:
    from scripts.train_expanded_baseline import _metrics, _to_bool
    from scripts.train_large_hybrid import (
        CANDIDATES,
        select_candidate,
        subtype_metrics,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Open sealed validation labels after candidate artifacts are frozen"
    )
    parser.add_argument(
        "--freeze-dir",
        type=Path,
        default=Path("outputs/deplab-large-candidate-freeze-v3.0.0"),
    )
    parser.add_argument(
        "--validation-results",
        type=Path,
        default=Path("outputs/large-validation-results-v3.0.0.jsonl"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "outputs/deplab-large-candidate-freeze-v3.0.0/validation-evaluation"
        ),
    )
    args = parser.parse_args()

    freeze_path = args.freeze_dir / "candidate-freeze-manifest.json"
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    _validate_freeze(args.freeze_dir, freeze)
    blind_path = args.freeze_dir / "validation-blind-predictions.csv"
    blind = pd.read_csv(
        blind_path,
        dtype={"experiment_id": "string", "python_version": "string"},
    )
    rows = _read_jsonl(args.validation_results)
    result_by_id = _validate_validation_results(rows)
    outcomes = np.asarray(
        [str(result_by_id[str(value)]["outcome"]) for value in blind["experiment_id"]],
        dtype=str,
    )
    target = (outcomes != "pass").astype(int)
    candidate_metrics = {}
    for name in CANDIDATES:
        probability = pd.to_numeric(
            blind[f"{name}_probability_failure"]
        ).to_numpy(float)
        threshold = float(freeze["frozen_thresholds"][name])
        predicted = probability >= threshold
        stored = _to_bool(blind[f"{name}_predicted_failure"])
        if not np.array_equal(predicted, stored):
            raise ValueError(f"blind predictions changed for {name}")
        candidate_metrics[name] = {
            **_metrics(target, predicted, probability),
            **subtype_metrics(outcomes, predicted),
        }
    selected = select_candidate(candidate_metrics)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    selected_source = args.freeze_dir / f"candidate-{selected}.json"
    selected_target = output / "selected-development-model.json"
    shutil.copyfile(selected_source, selected_target)
    payload = {
        "schema_version": "3.0.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "validation_rows": len(rows),
        "validation_outcome_counts": {
            str(key): int(value)
            for key, value in pd.Series(outcomes).value_counts().sort_index().items()
        },
        "candidate_metrics": candidate_metrics,
        "selected_candidate": selected,
        "selection_rule": freeze["validation_selection_rule"],
        "threshold": float(freeze["frozen_thresholds"][selected]),
        "threshold_retuned_on_validation": False,
        "final_test_outcomes_used": False,
        "source_sha256": {
            "candidate_freeze_manifest": _sha256(freeze_path),
            "validation_blind_predictions": _sha256(blind_path),
            "sealed_validation_results": _sha256(args.validation_results),
            "selected_candidate_model": _sha256(selected_source),
        },
    }
    metrics_path = output / "validation-metrics.json"
    metrics_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output / "report.md").write_text(_report(payload), encoding="utf-8")
    _write_checksums(output)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _validate_freeze(directory: Path, freeze: dict[str, Any]) -> None:
    if freeze.get("validation_outcomes_used"):
        raise ValueError("candidate freeze says validation outcomes were already used")
    if tuple(freeze.get("candidates", [])) != CANDIDATES:
        raise ValueError("candidate freeze has an unexpected candidate list")
    for name, expected in dict(freeze["frozen_artifact_sha256"]).items():
        filename = _artifact_filename(name)
        path = directory / filename
        if not path.exists() or _sha256(path) != expected:
            raise ValueError(f"frozen artifact hash mismatch: {filename}")


def _artifact_filename(name: str) -> str:
    fixed = {
        "development_oof_predictions": "development-oof-predictions.csv",
        "validation_blind_predictions": "validation-blind-predictions.csv",
        "development_metrics": "development-metrics.json",
        "development_folds": "development-folds.json",
    }
    if name in fixed:
        return fixed[name]
    prefix = "model_"
    if name.startswith(prefix):
        return f"candidate-{name[len(prefix):]}.json"
    raise ValueError(f"unknown frozen artifact name: {name}")


def _validate_validation_results(
    rows: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    if len(rows) != 3432:
        raise ValueError(f"expected 3,432 validation rows; got {len(rows):,}")
    indexed = {str(row["experiment_id"]): row for row in rows}
    if len(indexed) != len(rows):
        raise ValueError("validation results contain duplicate experiment IDs")
    infrastructure = [
        key
        for key, row in indexed.items()
        if row.get("outcome") == "infrastructure_failure"
    ]
    unmeasured = [key for key, row in indexed.items() if not row.get("measured")]
    if infrastructure or unmeasured:
        raise ValueError(
            "validation results contain infrastructure failures or unmeasured rows"
        )
    return indexed


def _report(payload: dict[str, Any]) -> str:
    selected = payload["selected_candidate"]
    metrics = payload["candidate_metrics"][selected]
    return f"""# DepLab large validation evaluation

The candidate models and their decision thresholds were frozen before the validation
outcomes were opened.

- Validation rows: **{payload['validation_rows']:,}**
- Selected candidate: **{selected}**
- Accuracy: **{metrics['accuracy']:.3f}**
- Balanced accuracy: **{metrics['balanced_accuracy']:.3f}**
- Failure precision: **{metrics['failure_precision']:.3f}**
- Failure recall: **{metrics['failure_recall']:.3f}**
- Failure F1: **{metrics['failure_f1']:.3f}**
- Frozen threshold: **{payload['threshold']:.6f}**

The final-test outcomes remain sealed. The selected algorithm may now be refit on
development plus validation while keeping this decision threshold unchanged.
"""


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_checksums(output: Path) -> None:
    checksum = output / "SHA256SUMS.txt"
    lines = [
        f"{_sha256(path)}  {path.name}"
        for path in sorted(output.iterdir())
        if path.is_file() and path != checksum
    ]
    checksum.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
