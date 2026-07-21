from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from advanced_models import (
        HistogramBinner,
        HistogramGradientBoosting,
        TwoHeadBoosting,
        balanced_binary_weights,
    )
    from train_expanded_baseline import (
        Preprocessor,
        _metrics,
        _select_threshold,
        _to_bool,
        assign_family_folds,
    )
except ImportError:
    from scripts.advanced_models import (
        HistogramBinner,
        HistogramGradientBoosting,
        TwoHeadBoosting,
        balanced_binary_weights,
    )
    from scripts.train_expanded_baseline import (
        Preprocessor,
        _metrics,
        _select_threshold,
        _to_bool,
        assign_family_folds,
    )


MODEL_ID = "deplab-advanced-model-comparison-v3.0.0"
MODEL_OPTIONS = {
    "estimators": 90,
    "learning_rate": 0.07,
    "maximum_depth": 3,
    "minimum_leaf": 20,
    "l2": 2.0,
    "feature_fraction": 0.6,
    "random_seed": 20260719,
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare advanced leakage-safe DepLab models")
    parser.add_argument("--features", type=Path, default=Path("outputs/deplab-expanded-development-v2.0.0/features.csv"))
    parser.add_argument("--holdout-inputs", type=Path, default=Path("outputs/deplab-expanded-development-v2.0.0/final-holdout-inputs.csv"))
    parser.add_argument("--holdout-results", type=Path, default=Path("outputs/expanded-final-holdout-results.jsonl"))
    parser.add_argument("--policy", type=Path, default=Path("outputs/deplab-expanded-development-v2.0.0/model-input-policy.json"))
    parser.add_argument("--baseline-oof", type=Path, default=Path("outputs/deplab-expanded-weighted-logistic-v2.0.0/development-oof-predictions.csv"))
    parser.add_argument("--baseline-holdout-metrics", type=Path, default=Path("outputs/deplab-expanded-weighted-logistic-v2.0.0/final-holdout-evaluation/metrics.json"))
    parser.add_argument("--output-dir", type=Path, default=Path(f"outputs/{MODEL_ID}"))
    args = parser.parse_args()

    dtype = {
        "experiment_id": "string",
        "family": "string",
        "python_version": "string",
        "package_a_version": "string",
        "package_b_version": "string",
    }
    frame = pd.read_csv(args.features, dtype=dtype)
    holdout = pd.read_csv(args.holdout_inputs, dtype=dtype)
    policy = json.loads(args.policy.read_text(encoding="utf-8"))
    numeric = list(policy["numeric_columns"])
    categorical = list(policy["categorical_columns"])
    target = _to_bool(frame["is_failure"]).astype(int)
    outcomes = frame["outcome"].astype(str).to_numpy()
    folds = assign_family_folds(frame["family"].astype(str), 5)

    candidate_probabilities: dict[str, np.ndarray] = {}
    component_probabilities: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for candidate in ("gbdt_binary", "gbdt_two_head"):
        probabilities = np.full(len(frame), np.nan, dtype=float)
        resolution_component = np.full(len(frame), np.nan, dtype=float)
        post_component = np.full(len(frame), np.nan, dtype=float)
        for fold in sorted(set(folds)):
            train_mask = folds != fold
            test_mask = folds == fold
            preprocessor = Preprocessor.fit(frame.loc[train_mask], numeric, categorical)
            x_train = preprocessor.transform(frame.loc[train_mask])
            x_test = preprocessor.transform(frame.loc[test_mask])
            binner = HistogramBinner.fit(x_train)
            train_bins = binner.transform(x_train)
            test_bins = binner.transform(x_test)
            options = {**MODEL_OPTIONS, "random_seed": MODEL_OPTIONS["random_seed"] + int(fold) * 100}
            if candidate == "gbdt_binary":
                model = HistogramGradientBoosting(**options).fit(
                    train_bins,
                    target[train_mask],
                    balanced_binary_weights(target[train_mask]),
                )
                probabilities[test_mask] = model.predict_proba(test_bins)
            else:
                model = TwoHeadBoosting.fit(train_bins, outcomes[train_mask], **options)
                resolution, post = model.predict_components(test_bins)
                resolution_component[test_mask] = resolution
                post_component[test_mask] = post
                probabilities[test_mask] = 1.0 - (1.0 - resolution) * (1.0 - post)
        if np.isnan(probabilities).any():
            raise ValueError(f"{candidate} did not score every development row")
        candidate_probabilities[candidate] = probabilities
        if candidate == "gbdt_two_head":
            component_probabilities[candidate] = (resolution_component, post_component)

    baseline = pd.read_csv(args.baseline_oof, dtype={"experiment_id": "string"})
    baseline = baseline.set_index("experiment_id").loc[frame["experiment_id"].astype(str)].reset_index()
    baseline_probability = pd.to_numeric(baseline["predicted_probability_failure"]).to_numpy(float)
    baseline_prediction = _to_bool(baseline["predicted_failure"])
    comparison_rows = [
        {
            "candidate": "weighted_logistic_v2",
            "threshold": 0.3439009058187318,
            **_metrics(target, baseline_prediction, baseline_probability),
            **_subtype_metrics(outcomes, baseline_prediction),
        }
    ]
    selected_thresholds: dict[str, float] = {}
    candidate_metrics: dict[str, dict[str, Any]] = {}
    for name, probabilities in candidate_probabilities.items():
        threshold, _ = _select_threshold(target, probabilities)
        predicted = probabilities >= threshold
        metrics = {
            **_metrics(target, predicted, probabilities),
            **_subtype_metrics(outcomes, predicted),
        }
        selected_thresholds[name] = threshold
        candidate_metrics[name] = metrics
        comparison_rows.append({"candidate": name, "threshold": threshold, **metrics})

    selected_name = max(
        candidate_metrics,
        key=lambda name: (
            candidate_metrics[name]["balanced_accuracy"],
            candidate_metrics[name]["post_install_failure_recall"],
            candidate_metrics[name]["failure_recall"],
        ),
    )
    selected_probability = candidate_probabilities[selected_name]
    selected_threshold = selected_thresholds[selected_name]
    selected_prediction = selected_probability >= selected_threshold

    # Fit the selected candidate on all 3,269 development rows, then use the already-known
    # 840 rows only as a transparent benchmark. It is not called an untouched test again.
    preprocessor = Preprocessor.fit(frame, numeric, categorical)
    x_train = preprocessor.transform(frame)
    x_holdout = preprocessor.transform(holdout)
    binner = HistogramBinner.fit(x_train)
    train_bins = binner.transform(x_train)
    holdout_bins = binner.transform(x_holdout)
    if selected_name == "gbdt_binary":
        final_model: HistogramGradientBoosting | TwoHeadBoosting = HistogramGradientBoosting(**MODEL_OPTIONS).fit(
            train_bins, target, balanced_binary_weights(target)
        )
        holdout_probability = final_model.predict_proba(holdout_bins)
        holdout_resolution = holdout_post = None
    else:
        final_model = TwoHeadBoosting.fit(train_bins, outcomes, **MODEL_OPTIONS)
        holdout_resolution, holdout_post = final_model.predict_components(holdout_bins)
        holdout_probability = 1.0 - (1.0 - holdout_resolution) * (1.0 - holdout_post)
    holdout_prediction = holdout_probability >= selected_threshold
    result_by_id = {
        str(row["experiment_id"]): row
        for row in _read_jsonl(args.holdout_results)
    }
    holdout_outcomes = np.asarray(
        [result_by_id[str(experiment_id)]["outcome"] for experiment_id in holdout["experiment_id"]],
        dtype=str,
    )
    holdout_target = (holdout_outcomes != "pass").astype(int)
    holdout_metrics = {
        **_metrics(holdout_target, holdout_prediction, holdout_probability),
        **_subtype_metrics(holdout_outcomes, holdout_prediction),
    }
    baseline_holdout = json.loads(args.baseline_holdout_metrics.read_text(encoding="utf-8"))["overall"]

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(comparison_rows).to_csv(output / "development-model-comparison.csv", index=False)
    oof = frame.loc[:, ["matrix_order", "experiment_id", "family", "outcome", "is_failure"]].copy()
    oof["fold"] = folds
    for name, probabilities in candidate_probabilities.items():
        oof[f"{name}_probability_failure"] = probabilities
        oof[f"{name}_predicted_failure"] = probabilities >= selected_thresholds[name]
    if "gbdt_two_head" in component_probabilities:
        resolution, post = component_probabilities["gbdt_two_head"]
        oof["gbdt_two_head_probability_resolution"] = resolution
        oof["gbdt_two_head_probability_post_install"] = post
    oof.to_csv(output / "development-oof-predictions.csv", index=False)

    benchmark = holdout.loc[:, ["matrix_order", "experiment_id", "family", "package_a_name", "package_a_version", "package_b_name", "package_b_version", "python_version"]].copy()
    benchmark["actual_outcome"] = holdout_outcomes
    benchmark["actual_failure"] = holdout_target.astype(bool)
    benchmark["predicted_probability_failure"] = holdout_probability
    benchmark["predicted_failure"] = holdout_prediction
    if holdout_resolution is not None and holdout_post is not None:
        benchmark["predicted_probability_resolution"] = holdout_resolution
        benchmark["predicted_probability_post_install"] = holdout_post
    benchmark["prediction_correct"] = holdout_prediction == holdout_target.astype(bool)
    benchmark.to_csv(output / "known-840-benchmark-predictions.csv", index=False)

    payload = {
        "comparison_id": MODEL_ID,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "development_evaluation": "five family-separated folds; model and threshold selection use development rows only",
        "development_rows": len(frame),
        "candidate_metrics": {
            row["candidate"]: {key: value for key, value in row.items() if key != "candidate"}
            for row in comparison_rows
        },
        "selected_candidate": selected_name,
        "selected_threshold": selected_threshold,
        "selection_rule": "highest development balanced accuracy, then post-install failure recall, then overall failure recall",
        "known_840_benchmark": {
            "status": "not untouched because its outcomes were examined before v3 model development",
            "rows": len(holdout),
            "advanced_model": holdout_metrics,
            "previous_logistic": baseline_holdout,
            "improvement": {
                key: holdout_metrics[key] - baseline_holdout[key]
                for key in ("accuracy", "balanced_accuracy", "failure_precision", "failure_recall", "failure_f1", "roc_auc", "failure_average_precision")
            },
        },
        "model_options": MODEL_OPTIONS,
        "next_evaluation_rule": "Retrain on the combined 4,109 rows only after this comparison, then create new package families with no outcomes observed before freezing v3 predictions.",
        "source_sha256": {
            "development_features": _sha256(args.features),
            "holdout_inputs": _sha256(args.holdout_inputs),
            "holdout_results": _sha256(args.holdout_results),
            "input_policy": _sha256(args.policy),
        },
    }
    (output / "metrics.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    model_payload = {
        "model_id": MODEL_ID,
        "training_rows": len(frame),
        "selected_candidate": selected_name,
        "threshold": selected_threshold,
        "preprocessor": preprocessor.to_dict(),
        "binner": binner.to_dict(),
        "model": final_model.to_dict(),
    }
    (output / "model.json").write_text(json.dumps(model_payload, separators=(",", ":")) + "\n", encoding="utf-8")
    (output / "report.md").write_text(_report(payload), encoding="utf-8")
    _write_checksums(output)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _subtype_metrics(outcomes: np.ndarray, predicted: np.ndarray) -> dict[str, float | int]:
    resolution = outcomes == "resolution_failure"
    post_install = np.isin(outcomes, ["import_failure", "smoke_test_failure"])
    return {
        "resolution_failure_rows": int(resolution.sum()),
        "resolution_failure_recall": float(predicted[resolution].mean()) if resolution.any() else float("nan"),
        "post_install_failure_rows": int(post_install.sum()),
        "post_install_failure_recall": float(predicted[post_install].mean()) if post_install.any() else float("nan"),
    }


def _report(payload: dict[str, Any]) -> str:
    selected = payload["candidate_metrics"][payload["selected_candidate"]]
    benchmark = payload["known_840_benchmark"]
    advanced = benchmark["advanced_model"]
    previous = benchmark["previous_logistic"]
    return f"""# DepLab advanced model comparison v3

## Development selection

The advanced candidates were compared on the same five package-family-separated development folds. The selected model is **{payload['selected_candidate']}**.

- Development balanced accuracy: {selected['balanced_accuracy']:.3f}
- Development failure recall: {selected['failure_recall']:.3f}
- Development post-install failure recall: {selected['post_install_failure_recall']:.3f}
- Frozen development-selected threshold: {payload['selected_threshold']:.6f}

## Known 840-row benchmark

The 840 outcomes were already examined before this v3 work, so this is an honest comparison benchmark, not a second untouched test.

| Metric | Previous logistic | Advanced model | Change |
|---|---:|---:|---:|
| Accuracy | {previous['accuracy']:.3f} | {advanced['accuracy']:.3f} | {advanced['accuracy'] - previous['accuracy']:+.3f} |
| Balanced accuracy | {previous['balanced_accuracy']:.3f} | {advanced['balanced_accuracy']:.3f} | {advanced['balanced_accuracy'] - previous['balanced_accuracy']:+.3f} |
| Failure recall | {previous['failure_recall']:.3f} | {advanced['failure_recall']:.3f} | {advanced['failure_recall'] - previous['failure_recall']:+.3f} |
| Failure precision | {previous['failure_precision']:.3f} | {advanced['failure_precision']:.3f} | {advanced['failure_precision'] - previous['failure_precision']:+.3f} |
| Failure F1 | {previous['failure_f1']:.3f} | {advanced['failure_f1']:.3f} | {advanced['failure_f1'] - previous['failure_f1']:+.3f} |
| ROC AUC | {previous['roc_auc']:.3f} | {advanced['roc_auc']:.3f} | {advanced['roc_auc'] - previous['roc_auc']:+.3f} |

The next official evaluation must use newly selected package families whose outcomes remain unknown until v3 features, model and predictions are frozen.
"""


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


if __name__ == "__main__":
    raise SystemExit(main())
