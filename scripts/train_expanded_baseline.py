from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


MODEL_ID = "deplab-expanded-weighted-logistic-v2.0.0"


@dataclass
class Preprocessor:
    numeric_columns: list[str]
    categorical_columns: list[str]
    numeric_medians: dict[str, float]
    numeric_means: dict[str, float]
    numeric_scales: dict[str, float]
    category_levels: dict[str, list[str]]
    transformed_names: list[str]

    @classmethod
    def fit(cls, frame: pd.DataFrame, numeric: list[str], categorical: list[str]) -> "Preprocessor":
        medians: dict[str, float] = {}
        means: dict[str, float] = {}
        scales: dict[str, float] = {}
        transformed = []
        for column in numeric:
            values = pd.to_numeric(frame[column], errors="coerce")
            median = float(values.median()) if values.notna().any() else 0.0
            filled = values.fillna(median).astype(float)
            mean = float(filled.mean())
            scale = float(filled.std(ddof=0))
            medians[column] = median
            means[column] = mean
            scales[column] = scale if scale > 1e-12 else 1.0
            transformed.append(f"numeric::{column}")
        levels: dict[str, list[str]] = {}
        for column in categorical:
            values = frame[column].fillna("<missing>").astype(str)
            levels[column] = sorted(values.unique().tolist())
            transformed.extend(f"category::{column}=={level}" for level in levels[column])
        return cls(numeric, categorical, medians, means, scales, levels, transformed)

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        blocks = []
        for column in self.numeric_columns:
            values = pd.to_numeric(frame[column], errors="coerce").fillna(self.numeric_medians[column])
            blocks.append(((values.to_numpy(float) - self.numeric_means[column]) / self.numeric_scales[column])[:, None])
        for column in self.categorical_columns:
            values = frame[column].fillna("<missing>").astype(str).to_numpy()
            levels = self.category_levels[column]
            blocks.append(np.column_stack([values == level for level in levels]).astype(float))
        return np.column_stack(blocks)

    def to_dict(self) -> dict[str, Any]:
        return {
            "numeric_columns": self.numeric_columns,
            "categorical_columns": self.categorical_columns,
            "numeric_medians": self.numeric_medians,
            "numeric_means": self.numeric_means,
            "numeric_scales": self.numeric_scales,
            "category_levels": self.category_levels,
            "transformed_names": self.transformed_names,
        }


@dataclass
class LogisticModel:
    weights: np.ndarray
    intercept: float
    iterations: int
    final_loss: float

    def predict_proba(self, matrix: np.ndarray) -> np.ndarray:
        return _sigmoid(matrix @ self.weights + self.intercept)


def fit_weighted_logistic(
    matrix: np.ndarray,
    target: np.ndarray,
    learning_rate: float = 0.08,
    l2: float = 0.1,
    maximum_iterations: int = 1600,
    tolerance: float = 1e-7,
) -> LogisticModel:
    positive = int(target.sum())
    negative = len(target) - positive
    if positive == 0 or negative == 0:
        raise ValueError("weighted logistic regression needs both classes")
    class_weights = np.where(target == 1, len(target) / (2 * positive), len(target) / (2 * negative))
    weight_total = float(class_weights.sum())
    weights = np.zeros(matrix.shape[1], dtype=float)
    intercept = 0.0
    previous_loss = math.inf
    final_loss = math.inf
    for iteration in range(1, maximum_iterations + 1):
        logits = matrix @ weights + intercept
        probabilities = _sigmoid(logits)
        weighted_error = class_weights * (probabilities - target)
        gradient = matrix.T @ weighted_error / weight_total + l2 * weights / len(target)
        intercept_gradient = float(weighted_error.sum() / weight_total)
        weights -= learning_rate * gradient
        intercept -= learning_rate * intercept_gradient
        if iteration % 20 == 0 or iteration == maximum_iterations:
            final_loss = float(
                np.sum(class_weights * (np.logaddexp(0.0, logits) - target * logits)) / weight_total
                + 0.5 * l2 * np.dot(weights, weights) / len(target)
            )
            if abs(previous_loss - final_loss) < tolerance:
                break
            previous_loss = final_loss
    return LogisticModel(weights, intercept, iteration, final_loss)


def main() -> int:
    parser = argparse.ArgumentParser(description="Train and freeze the expanded leakage-safe baseline")
    parser.add_argument("--features", type=Path, default=Path("outputs/deplab-expanded-development-v2.0.0/features.csv"))
    parser.add_argument("--holdout-inputs", type=Path, default=Path("outputs/deplab-expanded-development-v2.0.0/final-holdout-inputs.csv"))
    parser.add_argument("--policy", type=Path, default=Path("outputs/deplab-expanded-development-v2.0.0/model-input-policy.json"))
    parser.add_argument("--dictionary", type=Path, default=Path("outputs/deplab-expanded-development-v2.0.0/feature-dictionary.csv"))
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
    dictionary = pd.read_csv(args.dictionary)
    numeric_all = list(policy["numeric_columns"])
    categorical = list(policy["categorical_columns"])
    _validate_inputs(frame, holdout, dictionary, numeric_all, categorical)

    target = _to_bool(frame["is_failure"]).astype(int)
    folds = assign_family_folds(frame["family"].astype(str), fold_count=5)
    candidate_sets = {
        "metadata_only": [column for column in numeric_all if "changelog" not in column],
        "metadata_plus_changelog": numeric_all,
    }
    candidates: dict[str, dict[str, Any]] = {}
    candidate_rows = []
    for name, numeric in candidate_sets.items():
        probabilities, fit_rows = _out_of_fold_probabilities(frame, target, folds, numeric, categorical)
        threshold, threshold_metrics = _select_threshold(target, probabilities)
        metrics = _metrics(target, probabilities >= threshold, probabilities)
        candidates[name] = {
            "numeric_columns": numeric,
            "probabilities": probabilities,
            "threshold": threshold,
            "metrics": metrics,
            "fits": fit_rows,
        }
        candidate_rows.append({"candidate": name, "threshold": threshold, **metrics})

    selected_name = max(
        candidates,
        key=lambda name: (
            candidates[name]["metrics"]["balanced_accuracy"],
            candidates[name]["metrics"]["failure_recall"],
            -abs(candidates[name]["threshold"] - 0.5),
        ),
    )
    selected = candidates[selected_name]
    probabilities = selected["probabilities"]
    threshold = float(selected["threshold"])
    predictions = probabilities >= threshold
    always_pass_probability = np.full(len(target), float(target.mean()))
    always_pass = np.zeros(len(target), dtype=bool)
    majority_metrics = _metrics(target, always_pass, always_pass_probability)

    fold_metrics = []
    for fold in sorted(set(folds)):
        mask = folds == fold
        fold_metrics.append(
            {
                "fold": int(fold),
                "held_out_families": ", ".join(sorted(frame.loc[mask, "family"].unique())),
                "rows": int(mask.sum()),
                "failures": int(target[mask].sum()),
                **_metrics(target[mask], predictions[mask], probabilities[mask]),
            }
        )

    preprocessor = Preprocessor.fit(frame, selected["numeric_columns"], categorical)
    matrix = preprocessor.transform(frame)
    final_model = fit_weighted_logistic(matrix, target)
    holdout_matrix = preprocessor.transform(holdout)
    holdout_probability = final_model.predict_proba(holdout_matrix)
    holdout_prediction = holdout_probability >= threshold
    unknown_categories = _unknown_categories(holdout, preprocessor)

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    oof = frame.loc[:, ["matrix_order", "experiment_id", "family", "package_a_name", "package_a_version", "package_b_name", "package_b_version", "python_version", "outcome", "is_failure"]].copy()
    oof["fold"] = folds
    oof["predicted_probability_failure"] = probabilities
    oof["predicted_failure"] = predictions
    oof["prediction_correct"] = predictions == target.astype(bool)
    oof.to_csv(output / "development-oof-predictions.csv", index=False, quoting=csv.QUOTE_MINIMAL)
    pd.DataFrame(fold_metrics).to_csv(output / "fold-metrics.csv", index=False)
    pd.DataFrame(candidate_rows).to_csv(output / "candidate-metrics.csv", index=False)

    coefficients = pd.DataFrame(
        {
            "transformed_feature": preprocessor.transformed_names,
            "coefficient_toward_failure": final_model.weights,
            "absolute_coefficient": np.abs(final_model.weights),
        }
    ).sort_values("absolute_coefficient", ascending=False)
    coefficients.to_csv(output / "coefficients.csv", index=False)

    holdout_predictions = holdout.loc[:, ["matrix_order", "experiment_id", "family", "package_a_name", "package_a_version", "package_b_name", "package_b_version", "python_version"]].copy()
    holdout_predictions["predicted_probability_failure"] = holdout_probability
    holdout_predictions["predicted_failure"] = holdout_prediction
    holdout_predictions.to_csv(output / "final-holdout-blind-predictions.csv", index=False)

    final_metrics = selected["metrics"]
    metrics_payload = {
        "model_id": MODEL_ID,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "task": "predict any compatibility failure before installation",
        "positive_class": "failure",
        "development_rows": len(frame),
        "development_families": int(frame["family"].nunique()),
        "class_counts": {"pass": int((target == 0).sum()), "failure": int(target.sum())},
        "evaluation": "five-fold package-family-separated out-of-fold evaluation",
        "fold_assignment": "families sorted alphabetically, then assigned round-robin to five folds",
        "selected_candidate": selected_name,
        "selected_threshold": threshold,
        "selection_rule": "highest development OOF balanced accuracy; then failure recall; then threshold nearest 0.5",
        "weighted_logistic": final_metrics,
        "always_pass_baseline": majority_metrics,
        "candidate_models": {
            name: {"threshold": value["threshold"], "metrics": value["metrics"]}
            for name, value in candidates.items()
        },
        "folds": fold_metrics,
        "final_fit": {
            "rows": len(frame),
            "numeric_columns": len(selected["numeric_columns"]),
            "categorical_columns": len(categorical),
            "transformed_features": len(preprocessor.transformed_names),
            "iterations": final_model.iterations,
            "final_loss": final_model.final_loss,
            "class_weight": "balanced",
            "l2": 0.1,
        },
        "sealed_holdout": {
            "rows": len(holdout),
            "outcomes_used": False,
            "blind_predictions_created": True,
            "predicted_failures": int(holdout_prediction.sum()),
            "predicted_passes": int((~holdout_prediction).sum()),
            "unknown_category_values": unknown_categories,
        },
    }
    (output / "metrics.json").write_text(json.dumps(metrics_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    model_payload = {
        "model_id": MODEL_ID,
        "model_type": "balanced NumPy logistic regression",
        "target": "is_failure",
        "positive_class": "failure",
        "threshold": threshold,
        "preprocessor": preprocessor.to_dict(),
        "weights": final_model.weights.tolist(),
        "intercept": final_model.intercept,
        "training": metrics_payload["final_fit"],
    }
    (output / "model.json").write_text(json.dumps(model_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    family_order = sorted(frame["family"].unique())
    split_manifest = {
        "schema_version": "1.0.0",
        "fold_count": 5,
        "group_column": "family",
        "algorithm": metrics_payload["fold_assignment"],
        "assignments": [
            {"family": family, "fold": index % 5 + 1}
            for index, family in enumerate(family_order)
        ],
    }
    (output / "split-manifest.json").write_text(json.dumps(split_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output / "report.md").write_text(_report(metrics_payload), encoding="utf-8")
    evaluator_source = Path(__file__).with_name("evaluate_expanded_holdout.py")
    evaluator_target = output / "evaluate_final_holdout.py"
    shutil.copyfile(evaluator_source, evaluator_target)

    freeze_files = [
        "model.json",
        "metrics.json",
        "split-manifest.json",
        "final-holdout-blind-predictions.csv",
        "evaluate_final_holdout.py",
    ]
    freeze = {
        "freeze_version": "1.0.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "holdout_outcomes_available_at_freeze": False,
        "decision_threshold": threshold,
        "input_sha256": {
            "development_features": _sha256(args.features),
            "holdout_inputs": _sha256(args.holdout_inputs),
            "model_input_policy": _sha256(args.policy),
            "feature_dictionary": _sha256(args.dictionary),
        },
        "frozen_artifact_sha256": {name: _sha256(output / name) for name in freeze_files},
        "evaluation_rule": "After collecting all 840 outcomes, join only by experiment_id and calculate the predeclared binary metrics at the frozen threshold.",
    }
    (output / "freeze-manifest.json").write_text(json.dumps(freeze, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_checksums(output)
    print(json.dumps(metrics_payload, indent=2, sort_keys=True))
    return 0


def assign_family_folds(families: pd.Series, fold_count: int = 5) -> np.ndarray:
    unique = sorted(set(families.astype(str)))
    mapping = {family: index % fold_count + 1 for index, family in enumerate(unique)}
    return families.astype(str).map(mapping).to_numpy(int)


def _out_of_fold_probabilities(
    frame: pd.DataFrame,
    target: np.ndarray,
    folds: np.ndarray,
    numeric: list[str],
    categorical: list[str],
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    probabilities = np.full(len(frame), np.nan, dtype=float)
    fits = []
    for fold in sorted(set(folds)):
        train_mask = folds != fold
        test_mask = folds == fold
        preprocessor = Preprocessor.fit(frame.loc[train_mask], numeric, categorical)
        model = fit_weighted_logistic(preprocessor.transform(frame.loc[train_mask]), target[train_mask])
        probabilities[test_mask] = model.predict_proba(preprocessor.transform(frame.loc[test_mask]))
        fits.append({"fold": int(fold), "train_rows": int(train_mask.sum()), "test_rows": int(test_mask.sum()), "iterations": model.iterations, "final_loss": model.final_loss})
    if np.isnan(probabilities).any():
        raise ValueError("some development rows did not receive an out-of-fold prediction")
    return probabilities, fits


def _select_threshold(target: np.ndarray, probabilities: np.ndarray) -> tuple[float, dict[str, Any]]:
    candidates = np.unique(np.concatenate(([0.0, 0.5, 1.0], probabilities)))
    best_threshold = 0.5
    best_metrics = _metrics(target, probabilities >= best_threshold, probabilities)
    best_key = (best_metrics["balanced_accuracy"], best_metrics["failure_recall"], -abs(best_threshold - 0.5))
    for threshold in candidates:
        metrics = _metrics(target, probabilities >= threshold, probabilities)
        key = (metrics["balanced_accuracy"], metrics["failure_recall"], -abs(float(threshold) - 0.5))
        if key > best_key:
            best_threshold = float(threshold)
            best_metrics = metrics
            best_key = key
    return best_threshold, best_metrics


def _metrics(actual: np.ndarray, predicted: np.ndarray, probability: np.ndarray) -> dict[str, float | int]:
    actual = actual.astype(int)
    predicted = predicted.astype(int)
    true_failure = int(np.sum((actual == 1) & (predicted == 1)))
    true_pass = int(np.sum((actual == 0) & (predicted == 0)))
    false_failure = int(np.sum((actual == 0) & (predicted == 1)))
    missed_failure = int(np.sum((actual == 1) & (predicted == 0)))
    precision = _safe_divide(true_failure, true_failure + false_failure)
    recall = _safe_divide(true_failure, true_failure + missed_failure)
    pass_recall = _safe_divide(true_pass, true_pass + false_failure)
    clipped = np.clip(probability, 1e-12, 1 - 1e-12)
    return {
        "accuracy": float(np.mean(actual == predicted)),
        "balanced_accuracy": (recall + pass_recall) / 2,
        "failure_precision": precision,
        "failure_recall": recall,
        "failure_f1": _safe_divide(2 * precision * recall, precision + recall),
        "pass_recall": pass_recall,
        "roc_auc": _roc_auc(actual, probability),
        "failure_average_precision": _average_precision(actual, probability),
        "log_loss": float(-np.mean(actual * np.log(clipped) + (1 - actual) * np.log(1 - clipped))),
        "brier_score": float(np.mean((probability - actual) ** 2)),
        "true_failure": true_failure,
        "true_pass": true_pass,
        "false_failure": false_failure,
        "missed_failure": missed_failure,
    }


def _validate_inputs(frame: pd.DataFrame, holdout: pd.DataFrame, dictionary: pd.DataFrame, numeric: list[str], categorical: list[str]) -> None:
    if len(frame) != 3269 or len(holdout) != 840:
        raise ValueError("unexpected expanded development or final holdout row count")
    if "is_failure" in holdout.columns or "outcome" in holdout.columns:
        raise ValueError("holdout inputs contain forbidden outcome labels")
    allowed = set(dictionary.loc[dictionary["role"] == "inference_safe_input", "column"])
    requested = set(numeric) | set(categorical)
    if requested != allowed:
        raise ValueError("model policy and feature dictionary disagree")
    forbidden_fragments = ("outcome", "import", "smoke", "installed", "duration", "cache", "network", "error", "family", "package_a_name", "package_b_name")
    leaked = sorted(column for column in requested if any(fragment in column for fragment in forbidden_fragments))
    if leaked:
        raise ValueError(f"forbidden post-run or identity inputs requested: {leaked}")


def _to_bool(series: pd.Series) -> np.ndarray:
    if series.dtype == bool:
        return series.to_numpy(bool)
    values = series.astype(str).str.lower()
    if not values.isin({"true", "false"}).all():
        raise ValueError("boolean labels must be true or false")
    return (values == "true").to_numpy(bool)


def _unknown_categories(frame: pd.DataFrame, preprocessor: Preprocessor) -> dict[str, list[str]]:
    result = {}
    for column, levels in preprocessor.category_levels.items():
        unknown = sorted(set(frame[column].fillna("<missing>").astype(str)) - set(levels))
        if unknown:
            result[column] = unknown
    return result


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-np.clip(values, -35, 35)))


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


def _report(payload: dict[str, Any]) -> str:
    model = payload["weighted_logistic"]
    baseline = payload["always_pass_baseline"]
    return f"""# DepLab expanded baseline evaluation

## What was tested

The model predicts whether a package pair will fail before installation. Five grouped folds were used: every package family is placed wholly in one fold, so related rows from that family are never split between training and validation in the same fold.

## Development out-of-fold result

- Selected inputs: {payload['selected_candidate']}
- Rows: {payload['development_rows']}
- Failure rows: {payload['class_counts']['failure']}
- Frozen failure threshold: {payload['selected_threshold']:.6f}
- Accuracy: {model['accuracy']:.3f}
- Balanced accuracy: {model['balanced_accuracy']:.3f}
- Failure recall: {model['failure_recall']:.3f}
- Failure precision: {model['failure_precision']:.3f}
- Failure F1: {model['failure_f1']:.3f}
- Failure average precision: {model['failure_average_precision']:.3f}
- ROC AUC: {model['roc_auc']:.3f}
- Missed failures: {model['missed_failure']} of {payload['class_counts']['failure']}

The always-pass baseline has {baseline['accuracy']:.3f} accuracy and {baseline['balanced_accuracy']:.3f} balanced accuracy. Accuracy alone is therefore not enough; failure recall and balanced accuracy are the main checks.

## Final holdout status

Predictions for all {payload['sealed_holdout']['rows']} final experiments are frozen, but their outcomes have not been run or viewed. The EC2 can be restarted only after these model artifacts and hashes are accepted as final.
"""


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
