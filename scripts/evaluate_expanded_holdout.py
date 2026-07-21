from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Apply the frozen DepLab v2 evaluation rule to completed final-holdout outcomes"
    )
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    predictions = pd.read_csv(
        args.predictions,
        dtype={"experiment_id": "string", "python_version": "string"},
    )
    results = _read_jsonl(args.results)
    result_by_id = {str(row["experiment_id"]): row for row in results}
    if len(predictions) != 840 or predictions["experiment_id"].nunique() != 840:
        raise ValueError("frozen prediction file must contain exactly 840 unique experiments")
    if len(results) != 840 or len(result_by_id) != 840:
        raise ValueError("final holdout results must contain exactly 840 unique experiments")
    if set(predictions["experiment_id"].astype(str)) != set(result_by_id):
        raise ValueError("prediction and result experiment IDs do not match exactly")
    infrastructure = [
        experiment_id
        for experiment_id, row in result_by_id.items()
        if row.get("outcome") == "infrastructure_failure"
    ]
    if infrastructure:
        raise ValueError(
            f"final holdout has {len(infrastructure)} infrastructure failures; repair and retry them before scoring"
        )

    scored = predictions.copy()
    scored["actual_outcome"] = [
        result_by_id[str(experiment_id)]["outcome"]
        for experiment_id in scored["experiment_id"]
    ]
    scored["actual_failure"] = scored["actual_outcome"] != "pass"
    predicted = _to_bool(scored["predicted_failure"])
    actual = scored["actual_failure"].to_numpy(bool)
    probabilities = pd.to_numeric(
        scored["predicted_probability_failure"], errors="raise"
    ).to_numpy(float)
    scored["prediction_correct"] = predicted == actual

    overall = _metrics(actual.astype(int), predicted, probabilities)
    family_metrics = []
    for family, family_frame in scored.groupby("family", sort=True):
        indexes = family_frame.index.to_numpy()
        family_metrics.append(
            {
                "family": str(family),
                "rows": len(indexes),
                "failures": int(actual[indexes].sum()),
                **_metrics(actual[indexes].astype(int), predicted[indexes], probabilities[indexes]),
            }
        )

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    scored.to_csv(output / "scored-predictions.csv", index=False)
    payload = {
        "evaluation_id": "deplab-expanded-final-holdout-v2.0.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "evaluation_design": "840 experiments from three package families whose six package names are absent from development",
        "rows": len(scored),
        "threshold_changed_after_freeze": False,
        "outcome_counts": dict(sorted(Counter(scored["actual_outcome"]).items())),
        "overall": overall,
        "families": family_metrics,
        "source_sha256": {
            "frozen_predictions": _sha256(args.predictions),
            "final_results": _sha256(args.results),
        },
    }
    (output / "metrics.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "report.md").write_text(_report(payload), encoding="utf-8")
    _write_checksums(output)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _metrics(actual: np.ndarray, predicted: np.ndarray, probability: np.ndarray) -> dict[str, float | int]:
    actual = actual.astype(int)
    predicted = predicted.astype(int)
    true_failure = int(np.sum((actual == 1) & (predicted == 1)))
    true_pass = int(np.sum((actual == 0) & (predicted == 0)))
    false_failure = int(np.sum((actual == 0) & (predicted == 1)))
    missed_failure = int(np.sum((actual == 1) & (predicted == 0)))
    failure_precision = _safe_divide(true_failure, true_failure + false_failure)
    failure_recall = _safe_divide(true_failure, true_failure + missed_failure)
    pass_recall = _safe_divide(true_pass, true_pass + false_failure)
    return {
        "accuracy": float(np.mean(actual == predicted)),
        "balanced_accuracy": (failure_recall + pass_recall) / 2,
        "failure_precision": failure_precision,
        "failure_recall": failure_recall,
        "failure_f1": _safe_divide(2 * failure_precision * failure_recall, failure_precision + failure_recall),
        "pass_recall": pass_recall,
        "roc_auc": _roc_auc(actual, probability),
        "failure_average_precision": _average_precision(actual, probability),
        "true_failure": true_failure,
        "true_pass": true_pass,
        "false_failure": false_failure,
        "missed_failure": missed_failure,
    }


def _to_bool(series: pd.Series) -> np.ndarray:
    if series.dtype == bool:
        return series.to_numpy(bool)
    values = series.astype(str).str.lower()
    if not values.isin({"true", "false"}).all():
        raise ValueError("predicted_failure values must be true or false")
    return (values == "true").to_numpy(bool)


def _roc_auc(actual: np.ndarray, probability: np.ndarray) -> float:
    positive = int(actual.sum())
    negative = len(actual) - positive
    if not positive or not negative:
        return float("nan")
    ranks = pd.Series(probability).rank(method="average").to_numpy()
    return float((ranks[actual == 1].sum() - positive * (positive + 1) / 2) / (positive * negative))


def _average_precision(actual: np.ndarray, probability: np.ndarray) -> float:
    positive = int(actual.sum())
    if not positive:
        return float("nan")
    order = np.argsort(-probability, kind="stable")
    sorted_actual = actual[order]
    precision_at_rank = np.cumsum(sorted_actual) / (np.arange(len(actual)) + 1)
    return float(np.sum(precision_at_rank * sorted_actual) / positive)


def _safe_divide(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_checksums(output: Path) -> None:
    checksum = output / "SHA256SUMS.txt"
    lines = [f"{_sha256(path)}  {path.name}" for path in sorted(output.iterdir()) if path.is_file() and path != checksum]
    checksum.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _report(payload: dict[str, Any]) -> str:
    metric = payload["overall"]
    return f"""# DepLab final holdout evaluation

This is the frozen evaluation of {payload['rows']} experiments from package families and package names absent from model development.

- Accuracy: {metric['accuracy']:.3f}
- Balanced accuracy: {metric['balanced_accuracy']:.3f}
- Failure recall: {metric['failure_recall']:.3f}
- Failure precision: {metric['failure_precision']:.3f}
- Failure F1: {metric['failure_f1']:.3f}
- Failure average precision: {metric['failure_average_precision']:.3f}
- ROC AUC: {metric['roc_auc']:.3f}
- Missed failures: {metric['missed_failure']}
"""


if __name__ == "__main__":
    raise SystemExit(main())
