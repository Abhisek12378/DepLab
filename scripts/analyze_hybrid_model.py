from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from train_expanded_baseline import _metrics, _select_threshold, _to_bool
except ImportError:
    from scripts.train_expanded_baseline import _metrics, _select_threshold, _to_bool


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate the logistic plus post-install-risk hybrid")
    parser.add_argument("--advanced-dir", type=Path, default=Path("outputs/deplab-advanced-model-comparison-v3.0.0"))
    parser.add_argument("--baseline-dir", type=Path, default=Path("outputs/deplab-expanded-weighted-logistic-v2.0.0"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/deplab-hybrid-validation-v3.0.0"))
    args = parser.parse_args()

    advanced_oof = pd.read_csv(args.advanced_dir / "development-oof-predictions.csv", dtype={"experiment_id": "string"})
    logistic_oof = pd.read_csv(args.baseline_dir / "development-oof-predictions.csv", dtype={"experiment_id": "string"})[
        ["experiment_id", "predicted_probability_failure", "predicted_failure"]
    ].rename(columns={"predicted_probability_failure": "logistic_probability", "predicted_failure": "logistic_prediction"})
    development = advanced_oof.merge(logistic_oof, on="experiment_id", validate="one_to_one")
    development_target = (development["outcome"] != "pass").astype(int).to_numpy()
    development_post = development["gbdt_two_head_probability_post_install"].to_numpy(float)
    development_logistic = development["logistic_probability"].to_numpy(float)
    development_hybrid_probability = 1.0 - (1.0 - development_logistic) * (1.0 - development_post)
    development_threshold, _ = _select_threshold(development_target, development_hybrid_probability)
    development_prediction = development_hybrid_probability >= development_threshold
    development_metrics = {
        **_metrics(development_target, development_prediction, development_hybrid_probability),
        "import_smoke_recall": float(
            development_prediction[np.isin(development["outcome"], ["import_failure", "smoke_test_failure"])].mean()
        ),
    }

    advanced_known = pd.read_csv(args.advanced_dir / "known-840-benchmark-predictions.csv", dtype={"experiment_id": "string"})
    logistic_known = pd.read_csv(args.baseline_dir / "final-holdout-evaluation/scored-predictions.csv", dtype={"experiment_id": "string"})[
        ["experiment_id", "predicted_probability_failure", "predicted_failure"]
    ].rename(columns={"predicted_probability_failure": "logistic_probability", "predicted_failure": "logistic_prediction"})
    known = advanced_known.merge(logistic_known, on="experiment_id", validate="one_to_one")
    known_target = (known["actual_outcome"] != "pass").astype(int).to_numpy()
    known_logistic_prediction = _to_bool(known["logistic_prediction"])
    known_logistic_probability = known["logistic_probability"].to_numpy(float)
    known_post = known["predicted_probability_post_install"].to_numpy(float)
    known_hybrid_probability = 1.0 - (1.0 - known_logistic_probability) * (1.0 - known_post)

    development_selected_prediction = known_hybrid_probability >= development_threshold
    development_selected_known_metrics = {
        **_metrics(known_target, development_selected_prediction, known_hybrid_probability),
        "import_failure_recall": float(
            development_selected_prediction[known["actual_outcome"].eq("import_failure")].mean()
        ),
    }

    post_threshold, validation_metrics = _select_validation_operating_point(
        known_target,
        known_logistic_prediction,
        known_logistic_probability,
        known_post,
        known["actual_outcome"].to_numpy(str),
        minimum_precision=0.90,
    )
    validation_prediction = known_logistic_prediction | (known_post >= post_threshold)
    known["hybrid_predicted_failure"] = validation_prediction
    known["hybrid_prediction_correct"] = validation_prediction == known_target.astype(bool)
    known["hybrid_probability_for_ranking"] = np.maximum(known_logistic_probability, known_post)

    baseline_metrics = _metrics(known_target, known_logistic_prediction, known_logistic_probability)
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "method": "weighted logistic resolution/general risk plus a two-head boosted-tree post-install warning",
        "development_only_selection": {
            "combined_probability": "1 - (1 - p_logistic) * (1 - p_post_install)",
            "threshold": development_threshold,
            "development_metrics": development_metrics,
            "known_840_metrics": development_selected_known_metrics,
        },
        "known_840_validation_selected_operating_point": {
            "status": "validation result, not an untouched test",
            "rule": "logistic failure at its frozen threshold OR post-install head probability at or above the validation-selected threshold",
            "minimum_failure_precision_constraint": 0.90,
            "post_install_threshold": post_threshold,
            "metrics": validation_metrics,
        },
        "previous_logistic_known_840": baseline_metrics,
        "validation_change_from_logistic": {
            key: validation_metrics[key] - baseline_metrics[key]
            for key in ("accuracy", "balanced_accuracy", "failure_precision", "failure_recall", "failure_f1")
        },
        "interpretation": [
            "The standalone boosted models do not replace logistic regression for unseen packages.",
            "The hybrid can catch import failures that logistic regression misses, at the cost of more false warnings.",
            "The 840 rows are now validation data and may be included in v3 training.",
            "A newly frozen set of package families is required for the next official evaluation.",
        ],
    }
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    known.to_csv(output / "known-840-hybrid-validation-predictions.csv", index=False)
    (output / "metrics.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output / "report.md").write_text(_report(payload), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _select_validation_operating_point(
    target: np.ndarray,
    logistic_prediction: np.ndarray,
    logistic_probability: np.ndarray,
    post_probability: np.ndarray,
    outcomes: np.ndarray,
    minimum_precision: float,
) -> tuple[float, dict[str, float | int]]:
    candidates = np.unique(np.concatenate(([0.0, 1.0], post_probability)))
    best: tuple[tuple[float, ...], float, dict[str, float | int]] | None = None
    import_mask = outcomes == "import_failure"
    for threshold in candidates:
        predicted = logistic_prediction | (post_probability >= threshold)
        probability = np.maximum(logistic_probability, post_probability)
        metrics = _metrics(target, predicted, probability)
        if metrics["failure_precision"] + 1e-12 < minimum_precision:
            continue
        import_recall = float(predicted[import_mask].mean()) if import_mask.any() else 0.0
        metrics = {**metrics, "import_failure_recall": import_recall}
        key = (
            metrics["failure_recall"],
            metrics["failure_f1"],
            metrics["balanced_accuracy"],
            -float(threshold),
        )
        if best is None or key > best[0]:
            best = (key, float(threshold), metrics)
    if best is None:
        raise ValueError("no validation threshold satisfies the precision constraint")
    return best[1], best[2]


def _report(payload: dict) -> str:
    baseline = payload["previous_logistic_known_840"]
    validation = payload["known_840_validation_selected_operating_point"]
    hybrid = validation["metrics"]
    return f"""# DepLab hybrid model validation

The boosted-tree candidates improved development import-failure learning but did not generalize well enough to replace logistic regression. The recommended v3 direction is therefore a hybrid: preserve the logistic signal and add a conservative post-install failure warning head.

## Known 840-row validation comparison

These rows are now validation data, not a second untouched test.

| Metric | Logistic v2 | Hybrid validation rule | Change |
|---|---:|---:|---:|
| Accuracy | {baseline['accuracy']:.3f} | {hybrid['accuracy']:.3f} | {hybrid['accuracy'] - baseline['accuracy']:+.3f} |
| Balanced accuracy | {baseline['balanced_accuracy']:.3f} | {hybrid['balanced_accuracy']:.3f} | {hybrid['balanced_accuracy'] - baseline['balanced_accuracy']:+.3f} |
| Failure recall | {baseline['failure_recall']:.3f} | {hybrid['failure_recall']:.3f} | {hybrid['failure_recall'] - baseline['failure_recall']:+.3f} |
| Failure precision | {baseline['failure_precision']:.3f} | {hybrid['failure_precision']:.3f} | {hybrid['failure_precision'] - baseline['failure_precision']:+.3f} |
| Failure F1 | {baseline['failure_f1']:.3f} | {hybrid['failure_f1']:.3f} | {hybrid['failure_f1'] - baseline['failure_f1']:+.3f} |
| Import-failure recall | 0.000 | {hybrid['import_failure_recall']:.3f} | {hybrid['import_failure_recall']:+.3f} |

The validation-selected post-install threshold is {validation['post_install_threshold']:.6f}, with a minimum failure-precision requirement of {validation['minimum_failure_precision_constraint']:.0%}.

The next official evaluation must use new package families selected before their outcomes are collected.
"""


if __name__ == "__main__":
    raise SystemExit(main())
