from __future__ import annotations

import argparse
import csv
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
        StageAwareBoosting,
        TwoHeadBoosting,
    )
    from train_expanded_baseline import (
        Preprocessor,
        _metrics,
        _select_threshold,
        _to_bool,
        fit_weighted_logistic,
    )
    from train_modernbert_hybrid import (
        MatrixScaler,
        ReleaseProjector,
        frame_release_keys,
        release_key,
        text_interaction_matrix,
    )
except ImportError:
    from scripts.advanced_models import (
        HistogramBinner,
        StageAwareBoosting,
        TwoHeadBoosting,
    )
    from scripts.train_expanded_baseline import (
        Preprocessor,
        _metrics,
        _select_threshold,
        _to_bool,
        fit_weighted_logistic,
    )
    from scripts.train_modernbert_hybrid import (
        MatrixScaler,
        ReleaseProjector,
        frame_release_keys,
        release_key,
        text_interaction_matrix,
    )


PIPELINE_ID = "deplab-large-hybrid-v3.0.0"
CANDIDATES = (
    "structured_weighted_logistic",
    "modernbert_two_head_hybrid",
    "modernbert_stage_aware_hybrid",
)
MODEL_OPTIONS = {
    "estimators": 70,
    "learning_rate": 0.07,
    "maximum_depth": 3,
    "minimum_leaf": 30,
    "l2": 2.0,
    "feature_fraction": 0.65,
    "random_seed": 20260726,
}
PCA_DIMENSIONS = 32


def main() -> int:
    args = _arguments()
    frame, validation = _load_frames(args.features, args.validation_inputs)
    policy = json.loads(args.policy.read_text(encoding="utf-8"))
    pipeline = json.loads(args.pipeline.read_text(encoding="utf-8"))
    _validate_pipeline(pipeline)
    numeric = list(policy["numeric_columns"])
    categorical = list(policy["categorical_columns"])
    _validate_model_inputs(frame, validation, numeric, categorical)
    embeddings = load_embeddings(args.embeddings)
    _validate_embedding_coverage(frame, validation, embeddings)

    target = _to_bool(frame["is_failure"]).astype(int)
    outcomes = frame["outcome"].astype(str).to_numpy()
    folds, assignments = assign_balanced_family_folds(
        frame["family"].astype(str),
        target,
        fold_count=5,
    )
    probabilities = out_of_fold_predictions(
        frame,
        target,
        outcomes,
        folds,
        numeric,
        categorical,
        embeddings,
    )
    thresholds, candidate_metrics = evaluate_candidates(
        target, outcomes, probabilities
    )
    development_preference = select_candidate(candidate_metrics)

    model_payloads, validation_probabilities = fit_final_candidates(
        frame,
        validation,
        target,
        outcomes,
        numeric,
        categorical,
        embeddings,
        thresholds,
    )
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    model_paths = _write_candidate_models(output, model_payloads)
    oof_path = _write_oof_predictions(
        output, frame, folds, probabilities, thresholds
    )
    blind_path = _write_blind_predictions(
        output, validation, validation_probabilities, thresholds
    )
    metrics_path = _write_development_metrics(
        output,
        frame,
        target,
        candidate_metrics,
        thresholds,
        development_preference,
        assignments,
    )
    split_path = output / "development-folds.json"
    split_path.write_text(
        json.dumps(
            {
                "schema_version": "3.0.0",
                "method": "deterministic greedy row-and-failure-balanced family groups",
                "fold_count": 5,
                "assignments": assignments,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    _freeze(
        args,
        output,
        model_paths,
        oof_path,
        blind_path,
        metrics_path,
        split_path,
        thresholds,
        candidate_metrics,
    )
    _write_checksums(output)
    print(metrics_path.read_text(encoding="utf-8"))
    return 0


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train and freeze DepLab candidates without reading validation outcomes"
        )
    )
    parser.add_argument(
        "--features",
        type=Path,
        default=Path(
            "outputs/deplab-large-features-v3.0.0/development-features.csv"
        ),
    )
    parser.add_argument(
        "--validation-inputs",
        type=Path,
        default=Path(
            "outputs/deplab-large-features-v3.0.0/validation-inputs.csv"
        ),
    )
    parser.add_argument(
        "--policy",
        type=Path,
        default=Path(
            "outputs/deplab-large-features-v3.0.0/model-input-policy.json"
        ),
    )
    parser.add_argument(
        "--embeddings",
        type=Path,
        default=Path("outputs/large-release-modernbert-v3.0.0.jsonl"),
    )
    parser.add_argument(
        "--pipeline",
        type=Path,
        default=Path("configs/large-model-pipeline-v3.0.0.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/deplab-large-candidate-freeze-v3.0.0"),
    )
    return parser.parse_args()


def _load_frames(
    features: Path, validation_inputs: Path
) -> tuple[pd.DataFrame, pd.DataFrame]:
    dtype = {
        "experiment_id": "string",
        "family": "string",
        "python_version": "string",
        "package_a_version": "string",
        "package_b_version": "string",
    }
    return (
        pd.read_csv(features, dtype=dtype),
        pd.read_csv(validation_inputs, dtype=dtype),
    )


def _validate_pipeline(pipeline: dict[str, Any]) -> None:
    if pipeline.get("pipeline_id") != PIPELINE_ID:
        raise ValueError("unexpected model pipeline configuration")
    if tuple(pipeline.get("candidates", [])) != CANDIDATES:
        raise ValueError("pipeline candidate list differs from training code")
    if dict(pipeline.get("boosting", {})) != MODEL_OPTIONS:
        raise ValueError("pipeline boosting options differ from training code")


def _validate_model_inputs(
    frame: pd.DataFrame,
    validation: pd.DataFrame,
    numeric: list[str],
    categorical: list[str],
) -> None:
    if len(frame) != 21490 or len(validation) != 3432:
        raise ValueError(
            f"expected 21,490 development and 3,432 validation rows; "
            f"got {len(frame):,} and {len(validation):,}"
        )
    leaked = {"outcome", "is_failure", "failure_stage"} & set(validation.columns)
    if leaked:
        raise ValueError(
            f"sealed validation inputs contain outcome labels: {sorted(leaked)}"
        )
    requested = set(numeric) | set(categorical)
    missing = sorted(requested - set(frame.columns))
    if missing:
        raise ValueError(f"model policy references missing columns: {missing}")
    forbidden = (
        "outcome",
        "installed",
        "duration",
        "cache",
        "network",
        "error",
        "experiment_id",
        "family",
        "package_a_name",
        "package_b_name",
    )
    unsafe = sorted(
        column
        for column in requested
        if any(fragment in column for fragment in forbidden)
    )
    if unsafe:
        raise ValueError(f"forbidden model inputs requested: {unsafe}")


def load_embeddings(path: Path) -> dict[tuple[str, str], np.ndarray]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    result = {
        release_key(str(row["package"]), str(row["version"])): np.asarray(
            row["embedding"], dtype=float
        )
        for row in rows
    }
    if len(result) != len(rows):
        raise ValueError("release embeddings contain duplicate package versions")
    dimensions = {len(value) for value in result.values()}
    if len(dimensions) != 1 or next(iter(dimensions), 0) < 1:
        raise ValueError(f"unexpected embedding dimensions: {sorted(dimensions)}")
    return result


def _validate_embedding_coverage(
    frame: pd.DataFrame,
    validation: pd.DataFrame,
    embeddings: dict[tuple[str, str], np.ndarray],
) -> None:
    required = frame_release_keys(frame) | frame_release_keys(validation)
    missing = sorted(required - set(embeddings))
    if missing:
        examples = ", ".join(f"{name}=={version}" for name, version in missing[:10])
        raise ValueError(
            f"missing {len(missing)} development/validation embeddings: {examples}"
        )


def assign_balanced_family_folds(
    families: pd.Series,
    target: np.ndarray,
    fold_count: int = 5,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    if fold_count < 2:
        raise ValueError("at least two folds are required")
    table = pd.DataFrame(
        {"family": families.astype(str).to_numpy(), "failure": target.astype(int)}
    )
    grouped = [
        {
            "family": str(family),
            "rows": int(len(group)),
            "failures": int(group["failure"].sum()),
        }
        for family, group in table.groupby("family", sort=True)
    ]
    if len(grouped) < fold_count:
        raise ValueError("fewer package families than requested folds")
    grouped.sort(key=lambda item: (-item["rows"], -item["failures"], item["family"]))
    target_rows = len(table) / fold_count
    target_failures = max(1.0, float(target.sum()) / fold_count)
    fold_rows = [0] * fold_count
    fold_failures = [0] * fold_count
    fold_families: list[list[str]] = [[] for _ in range(fold_count)]
    for group in grouped:
        selected = min(
            range(fold_count),
            key=lambda fold: _fold_score(
                fold_rows[fold] + group["rows"],
                fold_failures[fold] + group["failures"],
                len(fold_families[fold]) + 1,
                target_rows,
                target_failures,
                fold,
            ),
        )
        fold_rows[selected] += group["rows"]
        fold_failures[selected] += group["failures"]
        fold_families[selected].append(group["family"])
    mapping = {
        family: fold + 1
        for fold, fold_values in enumerate(fold_families)
        for family in fold_values
    }
    folds = families.astype(str).map(mapping).to_numpy(int)
    assignments = [
        {
            "fold": fold + 1,
            "rows": fold_rows[fold],
            "failures": fold_failures[fold],
            "families": sorted(fold_families[fold]),
        }
        for fold in range(fold_count)
    ]
    return folds, assignments


def _fold_score(
    rows: int,
    failures: int,
    family_count: int,
    target_rows: float,
    target_failures: float,
    fold: int,
) -> tuple[float, int, int]:
    normalized = (rows / target_rows) ** 2 + (failures / target_failures) ** 2
    return normalized, family_count, fold


def out_of_fold_predictions(
    frame: pd.DataFrame,
    target: np.ndarray,
    outcomes: np.ndarray,
    folds: np.ndarray,
    numeric: list[str],
    categorical: list[str],
    embeddings: dict[tuple[str, str], np.ndarray],
) -> dict[str, np.ndarray]:
    probabilities = {
        name: np.full(len(frame), np.nan, dtype=float) for name in CANDIDATES
    }
    for fold in sorted(set(folds)):
        train_mask = folds != fold
        test_mask = folds == fold
        train = frame.loc[train_mask]
        test = frame.loc[test_mask]
        fold_probabilities = _fit_fold(
            train,
            test,
            target[train_mask],
            outcomes[train_mask],
            numeric,
            categorical,
            embeddings,
            int(fold),
        )
        for name, values in fold_probabilities.items():
            probabilities[name][test_mask] = values
    incomplete = [
        name for name, values in probabilities.items() if np.isnan(values).any()
    ]
    if incomplete:
        raise ValueError(f"candidates did not score every development row: {incomplete}")
    return probabilities


def _fit_fold(
    train: pd.DataFrame,
    test: pd.DataFrame,
    target: np.ndarray,
    outcomes: np.ndarray,
    numeric: list[str],
    categorical: list[str],
    embeddings: dict[tuple[str, str], np.ndarray],
    fold: int,
) -> dict[str, np.ndarray]:
    preprocessor = Preprocessor.fit(train, numeric, categorical)
    structured_train = preprocessor.transform(train)
    structured_test = preprocessor.transform(test)
    logistic = fit_weighted_logistic(
        structured_train,
        target,
        maximum_iterations=600,
    )
    hybrid_train, hybrid_test = _hybrid_matrices(
        train,
        test,
        structured_train,
        structured_test,
        embeddings,
    )
    binner = HistogramBinner.fit(hybrid_train)
    train_bins = binner.transform(hybrid_train)
    test_bins = binner.transform(hybrid_test)
    options = {
        **MODEL_OPTIONS,
        "random_seed": MODEL_OPTIONS["random_seed"] + fold * 100,
    }
    two_head = TwoHeadBoosting.fit(train_bins, outcomes, **options)
    stage_aware = StageAwareBoosting.fit(train_bins, outcomes, **options)
    return {
        "structured_weighted_logistic": logistic.predict_proba(structured_test),
        "modernbert_two_head_hybrid": two_head.predict_proba(test_bins),
        "modernbert_stage_aware_hybrid": stage_aware.predict_proba(test_bins),
    }


def _hybrid_matrices(
    train: pd.DataFrame,
    other: pd.DataFrame,
    structured_train: np.ndarray,
    structured_other: np.ndarray,
    embeddings: dict[tuple[str, str], np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    projector = ReleaseProjector.fit(
        embeddings,
        frame_release_keys(train),
        dimensions=PCA_DIMENSIONS,
    )
    text_train = text_interaction_matrix(train, embeddings, projector)
    scaler = MatrixScaler.fit(text_train)
    return (
        np.column_stack([structured_train, scaler.transform(text_train)]),
        np.column_stack(
            [
                structured_other,
                scaler.transform(
                    text_interaction_matrix(other, embeddings, projector)
                ),
            ]
        ),
    )


def evaluate_candidates(
    target: np.ndarray,
    outcomes: np.ndarray,
    probabilities: dict[str, np.ndarray],
) -> tuple[dict[str, float], dict[str, dict[str, Any]]]:
    thresholds: dict[str, float] = {}
    metrics: dict[str, dict[str, Any]] = {}
    for name in CANDIDATES:
        threshold, _ = _select_threshold(target, probabilities[name])
        predicted = probabilities[name] >= threshold
        thresholds[name] = threshold
        metrics[name] = {
            **_metrics(target, predicted, probabilities[name]),
            **subtype_metrics(outcomes, predicted),
        }
    return thresholds, metrics


def subtype_metrics(
    outcomes: np.ndarray, predicted: np.ndarray
) -> dict[str, float | int]:
    result: dict[str, float | int] = {}
    recalls = []
    for outcome in (
        "resolution_failure",
        "import_failure",
        "smoke_test_failure",
    ):
        mask = outcomes == outcome
        recall = float(predicted[mask].mean()) if mask.any() else float("nan")
        result[f"{outcome}_rows"] = int(mask.sum())
        result[f"{outcome}_recall"] = recall
        if mask.any():
            recalls.append(recall)
    result["failure_subtype_macro_recall"] = float(np.mean(recalls))
    return result


def select_candidate(metrics: dict[str, dict[str, Any]]) -> str:
    return max(
        CANDIDATES,
        key=lambda name: (
            metrics[name]["balanced_accuracy"],
            metrics[name]["failure_subtype_macro_recall"],
            metrics[name]["failure_recall"],
            -metrics[name]["brier_score"],
        ),
    )


def fit_final_candidates(
    frame: pd.DataFrame,
    validation: pd.DataFrame,
    target: np.ndarray,
    outcomes: np.ndarray,
    numeric: list[str],
    categorical: list[str],
    embeddings: dict[tuple[str, str], np.ndarray],
    thresholds: dict[str, float],
) -> tuple[dict[str, dict[str, Any]], dict[str, np.ndarray]]:
    preprocessor = Preprocessor.fit(frame, numeric, categorical)
    structured_train = preprocessor.transform(frame)
    structured_validation = preprocessor.transform(validation)
    logistic = fit_weighted_logistic(
        structured_train,
        target,
        maximum_iterations=600,
    )
    projector = ReleaseProjector.fit(
        embeddings,
        frame_release_keys(frame),
        dimensions=PCA_DIMENSIONS,
    )
    text_train = text_interaction_matrix(frame, embeddings, projector)
    scaler = MatrixScaler.fit(text_train)
    hybrid_train = np.column_stack(
        [structured_train, scaler.transform(text_train)]
    )
    hybrid_validation = np.column_stack(
        [
            structured_validation,
            scaler.transform(
                text_interaction_matrix(validation, embeddings, projector)
            ),
        ]
    )
    binner = HistogramBinner.fit(hybrid_train)
    train_bins = binner.transform(hybrid_train)
    validation_bins = binner.transform(hybrid_validation)
    two_head = TwoHeadBoosting.fit(train_bins, outcomes, **MODEL_OPTIONS)
    stage_aware = StageAwareBoosting.fit(train_bins, outcomes, **MODEL_OPTIONS)
    shared = {
        "pipeline_id": PIPELINE_ID,
        "training_rows": len(frame),
        "preprocessor": preprocessor.to_dict(),
    }
    payloads = {
        "structured_weighted_logistic": {
            **shared,
            "candidate": "structured_weighted_logistic",
            "threshold": thresholds["structured_weighted_logistic"],
            "text_features": None,
            "model": {
                "type": "weighted_logistic",
                "weights": logistic.weights.tolist(),
                "intercept": logistic.intercept,
                "iterations": logistic.iterations,
                "final_loss": logistic.final_loss,
            },
        },
        "modernbert_two_head_hybrid": _hybrid_payload(
            shared,
            "modernbert_two_head_hybrid",
            thresholds,
            projector,
            scaler,
            binner,
            two_head.to_dict(),
        ),
        "modernbert_stage_aware_hybrid": _hybrid_payload(
            shared,
            "modernbert_stage_aware_hybrid",
            thresholds,
            projector,
            scaler,
            binner,
            stage_aware.to_dict(),
        ),
    }
    probabilities = {
        "structured_weighted_logistic": logistic.predict_proba(
            structured_validation
        ),
        "modernbert_two_head_hybrid": two_head.predict_proba(validation_bins),
        "modernbert_stage_aware_hybrid": stage_aware.predict_proba(
            validation_bins
        ),
    }
    return payloads, probabilities


def _hybrid_payload(
    shared: dict[str, Any],
    name: str,
    thresholds: dict[str, float],
    projector: ReleaseProjector,
    scaler: MatrixScaler,
    binner: HistogramBinner,
    model: dict[str, Any],
) -> dict[str, Any]:
    return {
        **shared,
        "candidate": name,
        "threshold": thresholds[name],
        "text_features": {
            "encoder": "answerdotai/ModernBERT-base",
            "encoder_frozen": True,
            "release_projector": projector.to_dict(),
            "text_scaler": scaler.to_dict(),
            "interaction": "A, B, absolute difference, elementwise product",
        },
        "binner": binner.to_dict(),
        "model": model,
    }


def _write_candidate_models(
    output: Path, payloads: dict[str, dict[str, Any]]
) -> dict[str, Path]:
    paths = {}
    for name, payload in payloads.items():
        path = output / f"candidate-{name}.json"
        path.write_text(
            json.dumps(payload, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
        paths[name] = path
    return paths


def _write_oof_predictions(
    output: Path,
    frame: pd.DataFrame,
    folds: np.ndarray,
    probabilities: dict[str, np.ndarray],
    thresholds: dict[str, float],
) -> Path:
    columns = [
        "matrix_order",
        "experiment_id",
        "family",
        "outcome",
        "is_failure",
    ]
    result = frame.loc[:, columns].copy()
    result["fold"] = folds
    for name in CANDIDATES:
        result[f"{name}_probability_failure"] = probabilities[name]
        result[f"{name}_predicted_failure"] = (
            probabilities[name] >= thresholds[name]
        )
    path = output / "development-oof-predictions.csv"
    result.to_csv(path, index=False, quoting=csv.QUOTE_MINIMAL)
    return path


def _write_blind_predictions(
    output: Path,
    validation: pd.DataFrame,
    probabilities: dict[str, np.ndarray],
    thresholds: dict[str, float],
) -> Path:
    columns = [
        "matrix_order",
        "experiment_id",
        "family",
        "package_a_name",
        "package_a_version",
        "package_b_name",
        "package_b_version",
        "python_version",
    ]
    result = validation.loc[:, columns].copy()
    for name in CANDIDATES:
        result[f"{name}_probability_failure"] = probabilities[name]
        result[f"{name}_predicted_failure"] = (
            probabilities[name] >= thresholds[name]
        )
    path = output / "validation-blind-predictions.csv"
    result.to_csv(path, index=False, quoting=csv.QUOTE_MINIMAL)
    return path


def _write_development_metrics(
    output: Path,
    frame: pd.DataFrame,
    target: np.ndarray,
    metrics: dict[str, dict[str, Any]],
    thresholds: dict[str, float],
    preference: str,
    assignments: list[dict[str, Any]],
) -> Path:
    payload = {
        "pipeline_id": PIPELINE_ID,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "development_rows": len(frame),
        "development_families": int(frame["family"].nunique()),
        "class_counts": {
            "pass": int((target == 0).sum()),
            "failure": int(target.sum()),
        },
        "evaluation": "five family-grouped out-of-fold splits",
        "candidate_metrics": metrics,
        "frozen_thresholds": thresholds,
        "development_preference": preference,
        "validation_selection_pending": True,
        "validation_outcomes_used": False,
        "folds": assignments,
    }
    path = output / "development-metrics.json"
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def _freeze(
    args: argparse.Namespace,
    output: Path,
    model_paths: dict[str, Path],
    oof_path: Path,
    blind_path: Path,
    metrics_path: Path,
    split_path: Path,
    thresholds: dict[str, float],
    candidate_metrics: dict[str, dict[str, Any]],
) -> None:
    frozen_paths = {
        **{f"model_{name}": path for name, path in model_paths.items()},
        "development_oof_predictions": oof_path,
        "validation_blind_predictions": blind_path,
        "development_metrics": metrics_path,
        "development_folds": split_path,
    }
    payload = {
        "schema_version": "3.0.0",
        "pipeline_id": PIPELINE_ID,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "validation_outcomes_available_to_training_code": False,
        "validation_outcomes_used": False,
        "candidates": list(CANDIDATES),
        "frozen_thresholds": thresholds,
        "development_candidate_metrics": candidate_metrics,
        "validation_selection_rule": (
            "highest balanced accuracy, then mean recall across resolution/import/"
            "smoke failures, then failure recall, then lower Brier score"
        ),
        "thresholds_may_be_retuned_on_validation": False,
        "input_sha256": {
            "development_features": _sha256(args.features),
            "validation_inputs_without_labels": _sha256(args.validation_inputs),
            "model_input_policy": _sha256(args.policy),
            "release_embeddings": _sha256(args.embeddings),
            "pipeline_configuration": _sha256(args.pipeline),
        },
        "frozen_artifact_sha256": {
            name: _sha256(path) for name, path in frozen_paths.items()
        },
    }
    (output / "candidate-freeze-manifest.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


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
