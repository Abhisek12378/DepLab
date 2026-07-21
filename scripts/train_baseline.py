from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


MODEL_VERSION = "1.0.0"

NUMERIC_COLUMNS = [
    "package_a_version_major",
    "package_a_version_minor",
    "package_a_version_patch",
    "package_b_version_major",
    "package_b_version_minor",
    "package_b_version_patch",
    "python_major",
    "python_minor",
    "release_date_distance_days",
    "package_a_requires_dist_count",
    "package_b_requires_dist_count",
    "package_a_eligible_wheel_count",
    "package_b_eligible_wheel_count",
    "top_level_wheel_bytes",
    "package_a_release_ordinal",
    "package_b_release_ordinal",
    "package_a_declares_package_b",
    "package_b_declares_package_a",
    "package_a_has_native_extensions",
    "package_b_has_native_extensions",
    "either_top_level_has_native_extensions",
    "package_a_requirement_has_upper_bound",
    "package_b_requirement_has_upper_bound",
    "package_a_requirement_has_lower_bound",
    "package_b_requirement_has_lower_bound",
]

CATEGORICAL_COLUMNS = [
    "python_version",
    "package_a_requires_python",
    "package_b_requires_python",
    "package_a_wheel_python_tag",
    "package_b_wheel_python_tag",
    "package_a_wheel_abi_tag",
    "package_b_wheel_abi_tag",
    "package_a_wheel_platform_tag",
    "package_b_wheel_platform_tag",
]


@dataclass
class Preprocessor:
    numeric_medians: dict[str, float]
    numeric_means: dict[str, float]
    numeric_scales: dict[str, float]
    category_levels: dict[str, list[str]]
    feature_names: list[str]

    @classmethod
    def fit(cls, frame: pd.DataFrame) -> "Preprocessor":
        medians: dict[str, float] = {}
        means: dict[str, float] = {}
        scales: dict[str, float] = {}
        names = []
        for column in NUMERIC_COLUMNS:
            values = pd.to_numeric(frame[column], errors="coerce")
            median = float(values.median()) if values.notna().any() else 0.0
            filled = values.fillna(median).astype(float)
            mean = float(filled.mean())
            scale = float(filled.std(ddof=0))
            medians[column] = median
            means[column] = mean
            scales[column] = scale if scale > 1e-12 else 1.0
            names.append(f"numeric::{column}")
        levels: dict[str, list[str]] = {}
        for column in CATEGORICAL_COLUMNS:
            values = frame[column].fillna("<missing>").astype(str)
            levels[column] = sorted(values.unique().tolist())
            names.extend(f"category::{column}=={level}" for level in levels[column])
        return cls(medians, means, scales, levels, names)

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        blocks = []
        for column in NUMERIC_COLUMNS:
            values = pd.to_numeric(frame[column], errors="coerce").fillna(
                self.numeric_medians[column]
            )
            blocks.append(
                ((values.to_numpy(dtype=float) - self.numeric_means[column]) / self.numeric_scales[column])[:, None]
            )
        for column in CATEGORICAL_COLUMNS:
            values = frame[column].fillna("<missing>").astype(str).to_numpy()
            levels = self.category_levels[column]
            blocks.append(np.column_stack([values == level for level in levels]).astype(float))
        return np.column_stack(blocks)

    def to_dict(self) -> dict[str, Any]:
        return {
            "numeric_columns": NUMERIC_COLUMNS,
            "categorical_columns": CATEGORICAL_COLUMNS,
            "numeric_medians": self.numeric_medians,
            "numeric_means": self.numeric_means,
            "numeric_scales": self.numeric_scales,
            "category_levels": self.category_levels,
            "feature_names": self.feature_names,
        }


@dataclass
class LogisticModel:
    weights: np.ndarray
    intercept: float
    iterations: int
    final_loss: float

    def predict_proba(self, matrix: np.ndarray) -> np.ndarray:
        return _sigmoid(matrix @ self.weights + self.intercept)


def fit_logistic(
    matrix: np.ndarray,
    target: np.ndarray,
    learning_rate: float = 0.08,
    l2: float = 0.05,
    maximum_iterations: int = 5000,
    tolerance: float = 1e-8,
) -> LogisticModel:
    weights = np.zeros(matrix.shape[1], dtype=float)
    intercept = 0.0
    previous_loss = math.inf
    final_loss = math.inf
    for iteration in range(1, maximum_iterations + 1):
        logits = matrix @ weights + intercept
        probabilities = _sigmoid(logits)
        error = probabilities - target
        gradient = (matrix.T @ error) / len(target) + l2 * weights / len(target)
        intercept_gradient = float(error.mean())
        weights -= learning_rate * gradient
        intercept -= learning_rate * intercept_gradient
        if iteration % 25 == 0 or iteration == maximum_iterations:
            final_loss = float(
                np.mean(np.logaddexp(0.0, logits) - target * logits)
                + 0.5 * l2 * np.dot(weights, weights) / len(target)
            )
            if abs(previous_loss - final_loss) < tolerance:
                break
            previous_loss = final_loss
    return LogisticModel(weights, intercept, iteration, final_loss)


def main() -> int:
    parser = argparse.ArgumentParser(description="Train and evaluate the first DepLab baseline")
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--dictionary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--zip", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    dictionary = pd.read_csv(args.dictionary)
    allowed_input_columns = set(dictionary.loc[dictionary["role"] == "input_feature", "column"])
    required = set(NUMERIC_COLUMNS) | set(CATEGORICAL_COLUMNS)
    derived = {
        "package_a_release_ordinal",
        "package_b_release_ordinal",
        "package_a_requirement_has_upper_bound",
        "package_b_requirement_has_upper_bound",
        "package_a_requirement_has_lower_bound",
        "package_b_requirement_has_lower_bound",
    }
    if not (required - derived).issubset(allowed_input_columns):
        raise ValueError("baseline requests columns not marked as safe input features")

    frame = pd.read_csv(
        args.features,
        dtype={
            "experiment_id": "string",
            "family": "string",
            "python_version": "string",
            "package_a_version": "string",
            "package_b_version": "string",
        },
    )
    frame = _add_derived_inputs(frame)
    target = frame["is_compatible"].astype(bool).astype(int).to_numpy()
    groups = frame["family"].astype(str).to_numpy()
    families = sorted(set(groups))
    if len(frame) != 646 or len(families) != 6:
        raise ValueError("unexpected dataset shape for baseline v1")

    prediction_rows = []
    fold_rows = []
    for family in families:
        train_mask = groups != family
        test_mask = groups == family
        train_frame = frame.loc[train_mask]
        test_frame = frame.loc[test_mask]
        y_train = target[train_mask]
        y_test = target[test_mask]
        preprocessor = Preprocessor.fit(train_frame)
        x_train = preprocessor.transform(train_frame)
        x_test = preprocessor.transform(test_frame)
        model = fit_logistic(x_train, y_train)
        probabilities = model.predict_proba(x_test)
        predictions = (probabilities >= 0.5).astype(int)
        majority_class = int(y_train.mean() >= 0.5)
        majority_probability = float(y_train.mean())
        majority_predictions = np.full(len(y_test), majority_class, dtype=int)
        metrics = _metrics(y_test, predictions, probabilities)
        majority_metrics = _metrics(
            y_test,
            majority_predictions,
            np.full(len(y_test), majority_probability),
        )
        fold_rows.append(
            {
                "held_out_family": family,
                "train_rows": int(train_mask.sum()),
                "test_rows": int(test_mask.sum()),
                "test_compatible": int(y_test.sum()),
                "test_incompatible": int((1 - y_test).sum()),
                **{f"logistic_{key}": value for key, value in metrics.items()},
                **{f"majority_{key}": value for key, value in majority_metrics.items()},
                "model_iterations": model.iterations,
                "model_final_loss": model.final_loss,
            }
        )
        for position, (_, row) in enumerate(test_frame.iterrows()):
            prediction_rows.append(
                {
                    "experiment_id": row["experiment_id"],
                    "held_out_family": family,
                    "package_a": f"{row['package_a_name']}=={row['package_a_version']}",
                    "package_b": f"{row['package_b_name']}=={row['package_b_version']}",
                    "python_version": row["python_version"],
                    "actual_outcome": row["outcome"],
                    "actual_is_compatible": bool(y_test[position]),
                    "predicted_probability_compatible": float(probabilities[position]),
                    "predicted_is_compatible": bool(predictions[position]),
                    "prediction_correct": bool(predictions[position] == y_test[position]),
                    "majority_probability_compatible": majority_probability,
                    "majority_predicted_is_compatible": bool(majority_predictions[position]),
                    "majority_correct": bool(majority_predictions[position] == y_test[position]),
                }
            )

    predictions_frame = pd.DataFrame(prediction_rows)
    predictions_frame = predictions_frame.set_index("experiment_id").loc[
        frame["experiment_id"].astype(str)
    ].reset_index()
    actual = predictions_frame["actual_is_compatible"].astype(int).to_numpy()
    logistic_prediction = predictions_frame["predicted_is_compatible"].astype(int).to_numpy()
    logistic_probability = predictions_frame["predicted_probability_compatible"].to_numpy(float)
    majority_prediction = predictions_frame["majority_predicted_is_compatible"].astype(int).to_numpy()
    majority_probability = predictions_frame["majority_probability_compatible"].to_numpy(float)
    overall = _metrics(actual, logistic_prediction, logistic_probability)
    majority_overall = _metrics(actual, majority_prediction, majority_probability)

    final_preprocessor = Preprocessor.fit(frame)
    final_matrix = final_preprocessor.transform(frame)
    final_model = fit_logistic(final_matrix, target)
    coefficients = pd.DataFrame(
        {
            "transformed_feature": final_preprocessor.feature_names,
            "coefficient": final_model.weights,
            "absolute_coefficient": np.abs(final_model.weights),
        }
    ).sort_values("absolute_coefficient", ascending=False)

    predictions_path = output / "predictions.csv"
    folds_path = output / "fold-metrics.csv"
    coefficients_path = output / "coefficients.csv"
    predictions_frame.to_csv(predictions_path, index=False, quoting=csv.QUOTE_MINIMAL)
    pd.DataFrame(fold_rows).to_csv(folds_path, index=False)
    coefficients.to_csv(coefficients_path, index=False)

    metrics_payload = {
        "model_id": "deplab-logistic-baseline-v1.0.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "task": "binary compatibility prediction",
        "positive_class": "compatible",
        "decision_threshold": 0.5,
        "evaluation": "leave-one-package-family-out cross-validation",
        "rows": len(frame),
        "families": families,
        "class_counts": {
            "compatible": int(target.sum()),
            "incompatible": int((1 - target).sum()),
        },
        "input_policy": {
            "numeric_columns": NUMERIC_COLUMNS,
            "categorical_columns": CATEGORICAL_COLUMNS,
            "excluded": (
                "package names, family identity, exact version strings, wheel filenames, labels, "
                "diagnostics, post-run evidence, resource measurements, runtime identity, and hashes"
            ),
        },
        "logistic": overall,
        "majority": majority_overall,
        "improvement": {
            key: overall[key] - majority_overall[key]
            for key in ("accuracy", "balanced_accuracy", "f1", "roc_auc", "average_precision")
        },
        "folds": fold_rows,
        "final_fit": {
            "rows": len(frame),
            "transformed_features": len(final_preprocessor.feature_names),
            "iterations": final_model.iterations,
            "final_loss": final_model.final_loss,
            "l2": 0.05,
            "learning_rate": 0.08,
        },
        "limitations": [
            "Only six package-pair families are available, so each held-out fold is a large domain shift.",
            "The final fitted model is a demonstration artifact and is not validated for arbitrary PyPI packages.",
            "Probabilities are not calibrated on an independent dataset.",
            "The 646 Cartesian rows are related; fold-level family results are more informative than random row metrics.",
        ],
    }
    (output / "metrics.json").write_text(
        json.dumps(metrics_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    model_payload = {
        "model_id": metrics_payload["model_id"],
        "model_version": MODEL_VERSION,
        "model_type": "NumPy logistic regression",
        "target": "is_compatible",
        "threshold": 0.5,
        "preprocessor": final_preprocessor.to_dict(),
        "weights": final_model.weights.tolist(),
        "intercept": final_model.intercept,
        "training": metrics_payload["final_fit"],
    }
    (output / "model.json").write_text(
        json.dumps(model_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "report.md").write_text(
        _report(metrics_payload), encoding="utf-8"
    )
    (output / "README.md").write_text(_readme(), encoding="utf-8")
    _write_checksums(output)

    args.zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in sorted(output.iterdir()):
            if path.is_file():
                archive.write(path, arcname=f"deplab-logistic-baseline-v1.0.0/{path.name}")
    zip_hash = _sha256(args.zip)
    args.zip.with_suffix(args.zip.suffix + ".sha256").write_text(
        f"{zip_hash}  {args.zip.name}\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "logistic": overall,
                "majority": majority_overall,
                "output_dir": str(output),
                "zip": str(args.zip),
                "zip_sha256": zip_hash,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _add_derived_inputs(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    for side in ("a", "b"):
        date_column = f"package_{side}_release_date"
        parsed = pd.to_datetime(frame[date_column], utc=True, errors="coerce")
        frame[f"package_{side}_release_ordinal"] = parsed.map(
            lambda value: value.toordinal() if pd.notna(value) else np.nan
        )
        requirement = frame[f"package_{side}_requirement_on_{'b' if side == 'a' else 'a'}"].fillna("").astype(str)
        frame[f"package_{side}_requirement_has_upper_bound"] = requirement.str.contains("<", regex=False)
        frame[f"package_{side}_requirement_has_lower_bound"] = requirement.str.contains(">", regex=False)
    return frame


def _sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, -35.0, 35.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def _metrics(actual: np.ndarray, predicted: np.ndarray, probability: np.ndarray) -> dict[str, float | int]:
    actual = actual.astype(int)
    predicted = predicted.astype(int)
    true_positive = int(np.sum((actual == 1) & (predicted == 1)))
    true_negative = int(np.sum((actual == 0) & (predicted == 0)))
    false_positive = int(np.sum((actual == 0) & (predicted == 1)))
    false_negative = int(np.sum((actual == 1) & (predicted == 0)))
    precision = _safe_divide(true_positive, true_positive + false_positive)
    recall = _safe_divide(true_positive, true_positive + false_negative)
    specificity = _safe_divide(true_negative, true_negative + false_positive)
    clipped = np.clip(probability, 1e-12, 1 - 1e-12)
    return {
        "accuracy": float(np.mean(actual == predicted)),
        "balanced_accuracy": (recall + specificity) / 2.0,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": _safe_divide(2 * precision * recall, precision + recall),
        "roc_auc": _roc_auc(actual, probability),
        "average_precision": _average_precision(actual, probability),
        "log_loss": float(-np.mean(actual * np.log(clipped) + (1 - actual) * np.log(1 - clipped))),
        "brier_score": float(np.mean((probability - actual) ** 2)),
        "true_positive": true_positive,
        "true_negative": true_negative,
        "false_positive": false_positive,
        "false_negative": false_negative,
    }


def _roc_auc(actual: np.ndarray, probability: np.ndarray) -> float:
    positive = int(actual.sum())
    negative = len(actual) - positive
    if positive == 0 or negative == 0:
        return float("nan")
    ranks = pd.Series(probability).rank(method="average").to_numpy()
    return float((ranks[actual == 1].sum() - positive * (positive + 1) / 2) / (positive * negative))


def _average_precision(actual: np.ndarray, probability: np.ndarray) -> float:
    positive = int(actual.sum())
    if positive == 0:
        return float("nan")
    order = np.argsort(-probability, kind="stable")
    sorted_actual = actual[order]
    cumulative = np.cumsum(sorted_actual)
    precision_at_rank = cumulative / (np.arange(len(actual)) + 1)
    return float(np.sum(precision_at_rank * sorted_actual) / positive)


def _safe_divide(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _report(metrics: dict[str, Any]) -> str:
    logistic = metrics["logistic"]
    majority = metrics["majority"]
    lines = [
        "# DepLab baseline model evaluation",
        "",
        "## Evaluation design",
        "",
        "The model is evaluated with leave-one-package-family-out cross-validation. Each fold trains on five families and tests on the sixth, which prevents nearby rows from the same package family appearing in both training and test data.",
        "",
        "## Overall out-of-family results",
        "",
        "| Metric | Logistic regression | Majority baseline | Difference |",
        "|---|---:|---:|---:|",
    ]
    for key in ("accuracy", "balanced_accuracy", "precision", "recall", "specificity", "f1", "roc_auc", "average_precision", "log_loss", "brier_score"):
        difference = logistic[key] - majority[key]
        lines.append(f"| {key} | {logistic[key]:.3f} | {majority[key]:.3f} | {difference:+.3f} |")
    lines.extend(
        [
            "",
            "Confusion matrix for logistic regression:",
            "",
            f"- True compatible predicted compatible: {logistic['true_positive']}",
            f"- True incompatible predicted incompatible: {logistic['true_negative']}",
            f"- Incompatible incorrectly predicted compatible: {logistic['false_positive']}",
            f"- Compatible incorrectly predicted incompatible: {logistic['false_negative']}",
            "",
            "## Results by held-out family",
            "",
            "| Held-out family | Rows | Compatible | Accuracy | Balanced accuracy | ROC AUC | Majority accuracy |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for fold in metrics["folds"]:
        lines.append(
            f"| {fold['held_out_family']} | {fold['test_rows']} | {fold['test_compatible']} | "
            f"{fold['logistic_accuracy']:.3f} | {fold['logistic_balanced_accuracy']:.3f} | "
            f"{fold['logistic_roc_auc']:.3f} | {fold['majority_accuracy']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "This is a strict transfer test, not a random row split. Strong performance would indicate that generic release, dependency, Python, and wheel metadata transfer to an unseen package family. Weak or uneven family performance is still useful: it demonstrates that six families are insufficient for broad generalization and identifies where the next dataset expansion should focus.",
            "",
            "The final `model.json` is fitted on all 646 rows only as a reproducible demonstration artifact. It must not be presented as a validated general PyPI compatibility model.",
            "",
        ]
    )
    return "\n".join(lines)


def _readme() -> str:
    return """# DepLab logistic baseline v1.0.0

This package contains the first leakage-safe binary compatibility baseline for the 646-row DepLab systematic dataset.

Files:

- `metrics.json`: overall and leave-one-family-out metrics.
- `fold-metrics.csv`: one evaluation row per held-out package family.
- `predictions.csv`: out-of-family prediction for every experiment.
- `coefficients.csv`: final all-data fit coefficients for inspection.
- `model.json`: portable preprocessing state and logistic weights fitted on all rows.
- `report.md`: readable methodology, results, and limitations.
- `SHA256SUMS.txt`: file integrity checksums.

The evaluation excludes labels, diagnostics, installed-environment evidence, resource measurements, hashes, family identity, package names, exact version strings, and wheel filenames. The model uses generic pre-run version, release, declared dependency, Python, ABI, platform, native-wheel, and size features.
"""


def _write_checksums(output: Path) -> None:
    checksum = output / "SHA256SUMS.txt"
    lines = [
        f"{_sha256(path)}  {path.name}"
        for path in sorted(output.iterdir())
        if path.is_file() and path.name != checksum.name
    ]
    checksum.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
