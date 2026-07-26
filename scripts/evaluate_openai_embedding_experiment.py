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
    from evaluate_large_validation import (
        _read_jsonl,
        _validate_validation_results,
    )
    from train_expanded_baseline import _metrics, _to_bool
    from train_large_hybrid import (
        _load_frames,
        _validate_embedding_coverage,
        _validate_model_inputs,
        assign_balanced_family_folds,
        evaluate_candidates,
        fit_final_candidates,
        load_embeddings,
        out_of_fold_predictions,
        subtype_metrics,
    )
except ImportError:
    from scripts.evaluate_large_validation import (
        _read_jsonl,
        _validate_validation_results,
    )
    from scripts.train_expanded_baseline import _metrics, _to_bool
    from scripts.train_large_hybrid import (
        _load_frames,
        _validate_embedding_coverage,
        _validate_model_inputs,
        assign_balanced_family_folds,
        evaluate_candidates,
        fit_final_candidates,
        load_embeddings,
        out_of_fold_predictions,
        subtype_metrics,
    )


SOURCE_NAMES = {
    "modernbert_two_head_hybrid": "openai_embedding_3_large_two_head",
    "modernbert_stage_aware_hybrid": "openai_embedding_3_large_stage_aware",
}


def main() -> int:
    args = _arguments()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(
            f"{output} already exists; preserve it or choose another output directory"
        )

    frame, validation = _load_frames(args.features, args.validation_inputs)
    policy = json.loads(args.policy.read_text(encoding="utf-8"))
    numeric = list(policy["numeric_columns"])
    categorical = list(policy["categorical_columns"])
    _validate_model_inputs(frame, validation, numeric, categorical)
    embeddings = load_embeddings(args.embeddings)
    _validate_embedding_coverage(frame, validation, embeddings)

    target = _to_bool(frame["is_failure"]).astype(int)
    outcomes = frame["outcome"].astype(str).to_numpy()
    folds, assignments = assign_balanced_family_folds(
        frame["family"].astype(str), target, fold_count=5
    )
    raw_oof = out_of_fold_predictions(
        frame,
        target,
        outcomes,
        folds,
        numeric,
        categorical,
        embeddings,
    )
    raw_thresholds, raw_development_metrics = evaluate_candidates(
        target, outcomes, raw_oof
    )
    _, raw_validation_probabilities = fit_final_candidates(
        frame,
        validation,
        target,
        outcomes,
        numeric,
        categorical,
        embeddings,
        raw_thresholds,
    )
    validation_outcomes = _validation_outcomes(
        validation, args.validation_results
    )
    validation_target = (validation_outcomes != "pass").astype(int)

    thresholds = _rename(raw_thresholds)
    development_metrics = _rename(raw_development_metrics)
    validation_metrics = _validation_metrics(
        validation_target,
        validation_outcomes,
        raw_validation_probabilities,
        raw_thresholds,
    )
    output.mkdir(parents=True)
    oof_path = _write_predictions(
        output / "development-oof-predictions.csv",
        frame,
        folds,
        raw_oof,
        raw_thresholds,
    )
    validation_path = _write_predictions(
        output / "validation-predictions.csv",
        validation.assign(outcome=validation_outcomes),
        None,
        raw_validation_probabilities,
        raw_thresholds,
    )
    baseline = _load_baseline(args.baseline_validation_metrics)
    payload = {
        "schema_version": "3.0.0",
        "experiment": "OpenAI embedding comparison after validation diagnosis",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "embedding_model": args.model,
        "embedding_dimensions": _embedding_dimensions(embeddings),
        "development_rows": len(frame),
        "validation_rows": len(validation),
        "development_evaluation": "five family-grouped out-of-fold splits",
        "validation_policy": (
            "development thresholds reused without validation retuning"
        ),
        "development_metrics": development_metrics,
        "validation_metrics": validation_metrics,
        "baseline_validation_metrics": baseline,
        "frozen_thresholds": thresholds,
        "development_folds": assignments,
        "final_test_outcomes_used": False,
        "selection_warning": (
            "This experiment was motivated by validation results, so validation "
            "metrics are tuning evidence, not a final unbiased evaluation."
        ),
        "source_sha256": {
            "development_features": _sha256(args.features),
            "validation_inputs": _sha256(args.validation_inputs),
            "validation_results": _sha256(args.validation_results),
            "release_embeddings": _sha256(args.embeddings),
            "development_predictions": _sha256(oof_path),
            "validation_predictions": _sha256(validation_path),
        },
    }
    metrics_path = output / "experiment-metrics.json"
    metrics_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output / "report.md").write_text(_report(payload), encoding="utf-8")
    _write_checksums(output)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare OpenAI release embeddings without using final-test labels"
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
        "--validation-results",
        type=Path,
        default=Path("outputs/large-validation-results-v3.0.0.jsonl"),
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
        default=Path(
            "outputs/large-release-openai-embedding-3-large-v3.0.0.jsonl"
        ),
    )
    parser.add_argument(
        "--baseline-validation-metrics",
        type=Path,
        default=Path(
            "outputs/deplab-large-candidate-freeze-v3.0.0/"
            "validation-evaluation/validation-metrics.json"
        ),
    )
    parser.add_argument("--model", default="text-embedding-3-large")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/deplab-openai-embedding-experiment-v3.0.0"),
    )
    return parser.parse_args()


def _validation_outcomes(
    validation: pd.DataFrame, results_path: Path
) -> np.ndarray:
    rows = _read_jsonl(results_path)
    indexed = _validate_validation_results(rows)
    missing = [
        str(value)
        for value in validation["experiment_id"]
        if str(value) not in indexed
    ]
    if missing:
        raise ValueError(f"validation results miss {len(missing)} input IDs")
    return np.asarray(
        [indexed[str(value)]["outcome"] for value in validation["experiment_id"]],
        dtype=str,
    )


def _validation_metrics(
    target: np.ndarray,
    outcomes: np.ndarray,
    probabilities: dict[str, np.ndarray],
    thresholds: dict[str, float],
) -> dict[str, dict[str, Any]]:
    result = {}
    for source, name in SOURCE_NAMES.items():
        probability = probabilities[source]
        predicted = probability >= thresholds[source]
        result[name] = _with_correct_counts(
            {
                **_metrics(target, predicted, probability),
                **subtype_metrics(outcomes, predicted),
            }
        )
    return result


def _with_correct_counts(metrics: dict[str, Any]) -> dict[str, Any]:
    result = dict(metrics)
    for outcome in (
        "resolution_failure",
        "import_failure",
        "smoke_test_failure",
    ):
        rows = int(result[f"{outcome}_rows"])
        recall = float(result[f"{outcome}_recall"])
        result[f"{outcome}_correct"] = int(round(rows * recall))
    return result


def _rename(values: dict[str, Any]) -> dict[str, Any]:
    return {
        name: _with_correct_counts(values[source])
        if isinstance(values[source], dict)
        else values[source]
        for source, name in SOURCE_NAMES.items()
    }


def _write_predictions(
    path: Path,
    frame: pd.DataFrame,
    folds: np.ndarray | None,
    probabilities: dict[str, np.ndarray],
    thresholds: dict[str, float],
) -> Path:
    columns = ["experiment_id", "family", "outcome"]
    result = frame.loc[:, columns].copy()
    if folds is not None:
        result["fold"] = folds
    for source, name in SOURCE_NAMES.items():
        result[f"{name}_probability_failure"] = probabilities[source]
        result[f"{name}_predicted_failure"] = (
            probabilities[source] >= thresholds[source]
        )
    result.to_csv(path, index=False, quoting=csv.QUOTE_MINIMAL)
    return path


def _load_baseline(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    selected = str(payload["selected_candidate"])
    candidates = payload["candidate_metrics"]
    return {
        "selected_candidate": selected,
        "selected_metrics": _with_correct_counts(candidates[selected]),
        "all_candidates": {
            name: _with_correct_counts(metrics)
            for name, metrics in candidates.items()
        },
    }


def _embedding_dimensions(
    embeddings: dict[tuple[str, str], np.ndarray]
) -> int:
    dimensions = {len(value) for value in embeddings.values()}
    if len(dimensions) != 1:
        raise ValueError("release embeddings have inconsistent dimensions")
    return dimensions.pop()


def _report(payload: dict[str, Any]) -> str:
    lines = [
        "# DepLab OpenAI embedding experiment",
        "",
        "The same structured features, family-grouped folds, boosting settings, "
        "and development-selected thresholds were retained. Only the frozen "
        "release-text embedding source changed.",
        "",
        "| Candidate | Validation accuracy | Balanced accuracy | Import caught | "
        "Smoke caught | False warnings |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, metrics in payload["validation_metrics"].items():
        lines.append(
            f"| {name} | {metrics['accuracy']:.3f} | "
            f"{metrics['balanced_accuracy']:.3f} | "
            f"{metrics['import_failure_correct']}/"
            f"{metrics['import_failure_rows']} | "
            f"{metrics['smoke_test_failure_correct']}/"
            f"{metrics['smoke_test_failure_rows']} | "
            f"{metrics['false_failure']} |"
        )
    lines.extend(
        [
            "",
            "The 3,158-row final test was not read or used.",
            "",
            f"Important: {payload['selection_warning']}",
            "",
        ]
    )
    return "\n".join(lines)


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
