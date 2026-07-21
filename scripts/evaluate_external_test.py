from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from train_baseline import _metrics


def main() -> int:
    parser = argparse.ArgumentParser(description="Score blind predictions after outcomes exist")
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    predictions = pd.read_csv(
        args.predictions,
        dtype={
            "experiment_id": "string",
            "python_version": "string",
            "package_a_version": "string",
            "package_b_version": "string",
        },
    )
    results = _read_jsonl(args.results)
    by_id = {str(row["experiment_id"]): row for row in results}
    missing = sorted(set(predictions["experiment_id"].astype(str)) - set(by_id))
    if missing:
        raise ValueError(f"results are missing {len(missing)} predicted experiments")
    extra = sorted(set(by_id) - set(predictions["experiment_id"].astype(str)))
    if extra:
        raise ValueError(f"results contain {len(extra)} experiments without blind predictions")

    ordered_results = [by_id[str(value)] for value in predictions["experiment_id"]]
    outcomes = [str(row["outcome"]) for row in ordered_results]
    invalid = [outcome for outcome in outcomes if outcome in {"infrastructure_failure", "wheel_unavailable"}]
    if invalid:
        raise ValueError(
            "external test contains non-evaluable outcomes: "
            + ", ".join(f"{key}={value}" for key, value in Counter(invalid).items())
        )

    actual = np.asarray([outcome == "pass" for outcome in outcomes], dtype=int)
    probability = predictions["predicted_probability_compatible"].to_numpy(float)
    predicted = _to_bool(predictions["predicted_is_compatible"]).astype(int)
    scored = predictions.copy()
    scored["actual_outcome"] = outcomes
    scored["actual_is_compatible"] = actual.astype(bool)
    scored["prediction_correct"] = predicted == actual

    overall = _metrics(actual, predicted, probability)
    accuracy_interval = _wilson_interval(
        int(np.sum(predicted == actual)), len(actual)
    )
    all_compatible_accuracy = float(actual.mean())
    folds = []
    for family in sorted(scored["family"].unique()):
        mask = scored["family"] == family
        family_actual = actual[mask]
        family_metrics = _metrics(
            family_actual, predicted[mask], probability[mask]
        )
        if family_actual.sum() in {0, len(family_actual)}:
            family_metrics["balanced_accuracy"] = None
            family_metrics["roc_auc"] = None
        folds.append({"family": family, "rows": int(mask.sum()), **family_metrics})

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    scored.to_csv(output / "scored-predictions.csv", index=False)
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "evaluation": "blind external test on held-out versions of known packages",
        "rows": len(scored),
        "outcome_counts": dict(sorted(Counter(outcomes).items())),
        "overall": overall,
        "accuracy_95_percent_confidence_interval": {
            "lower": accuracy_interval[0],
            "upper": accuracy_interval[1],
        },
        "all_compatible_baseline": {
            "accuracy": all_compatible_accuracy,
            "balanced_accuracy": 0.5,
        },
        "accuracy_improvement_over_all_compatible": (
            overall["accuracy"] - all_compatible_accuracy
        ),
        "families": folds,
    }
    (output / "metrics.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "report.md").write_text(_report(payload), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _to_bool(series: pd.Series) -> np.ndarray:
    if series.dtype == bool:
        return series.to_numpy(bool)
    values = series.astype(str).str.lower()
    if not values.isin({"true", "false"}).all():
        raise ValueError("prediction labels must be true or false")
    return (values == "true").to_numpy(bool)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    proportion = successes / total
    denominator = 1 + z**2 / total
    centre = proportion + z**2 / (2 * total)
    margin = z * (
        (proportion * (1 - proportion) / total + z**2 / (4 * total**2)) ** 0.5
    )
    return ((centre - margin) / denominator, (centre + margin) / denominator)


def _report(payload: dict[str, Any]) -> str:
    metric = payload["overall"]
    lines = [
        "# DepLab blind external-version evaluation",
        "",
        "The predictions were saved before experiment outcomes were collected. Every package version in this test was absent from the 646-row training dataset.",
        "",
        "## Overall results",
        "",
        f"- Rows: {payload['rows']}",
        f"- Accuracy: {metric['accuracy']:.3f}",
        f"- 95% confidence interval for accuracy: {payload['accuracy_95_percent_confidence_interval']['lower']:.3f} to {payload['accuracy_95_percent_confidence_interval']['upper']:.3f}",
        f"- Always-compatible baseline accuracy: {payload['all_compatible_baseline']['accuracy']:.3f}",
        f"- Accuracy improvement over that baseline: {payload['accuracy_improvement_over_all_compatible']:+.3f}",
        f"- Balanced accuracy: {metric['balanced_accuracy']:.3f}",
        f"- ROC AUC: {metric['roc_auc']:.3f}",
        f"- Compatible cases correctly found: {metric['true_positive']}",
        f"- Incompatible cases correctly found: {metric['true_negative']}",
        "",
        "## Results by package family",
        "",
        "| Family | Rows | Accuracy | Balanced accuracy | ROC AUC |",
        "|---|---:|---:|---:|---:|",
    ]
    for family in payload["families"]:
        lines.append(
            f"| {family['family']} | {family['rows']} | {family['accuracy']:.3f} | "
            f"{_format_metric(family['balanced_accuracy'])} | {_format_metric(family['roc_auc'])} |"
        )
    lines.append("")
    return "\n".join(lines)


def _format_metric(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.3f}"


if __name__ == "__main__":
    raise SystemExit(main())
