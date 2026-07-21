from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from advanced_models import HistogramBinner, TwoHeadBoosting
    from train_expanded_baseline import (
        Preprocessor,
        _metrics,
        _select_threshold,
        _to_bool,
        assign_family_folds,
        fit_weighted_logistic,
    )
except ImportError:
    from scripts.advanced_models import HistogramBinner, TwoHeadBoosting
    from scripts.train_expanded_baseline import (
        Preprocessor,
        _metrics,
        _select_threshold,
        _to_bool,
        assign_family_folds,
        fit_weighted_logistic,
    )


MODEL_ID = "deplab-modernbert-two-head-v4.0.0"
PCA_DIMENSIONS = 32
MODEL_OPTIONS = {
    "estimators": 90,
    "learning_rate": 0.07,
    "maximum_depth": 3,
    "minimum_leaf": 20,
    "l2": 2.0,
    "feature_fraction": 0.6,
    "random_seed": 20260719,
}


def canonical(name: str) -> str:
    return name.lower().replace("_", "-").replace(".", "-")


def release_key(name: str, version: str) -> tuple[str, str]:
    return canonical(str(name)), str(version)


@dataclass
class ReleaseProjector:
    mean: np.ndarray
    components: np.ndarray

    @classmethod
    def fit(
        cls,
        embeddings: dict[tuple[str, str], np.ndarray],
        keys: set[tuple[str, str]],
        dimensions: int = PCA_DIMENSIONS,
    ) -> "ReleaseProjector":
        missing = sorted(keys - set(embeddings))
        if missing:
            raise ValueError(f"missing release embeddings: {missing[:5]}")
        matrix = np.vstack([embeddings[key] for key in sorted(keys)]).astype(float)
        mean = matrix.mean(axis=0)
        _, _, right = np.linalg.svd(matrix - mean, full_matrices=False)
        count = min(dimensions, len(right))
        return cls(mean=mean, components=right[:count])

    def transform(self, vectors: np.ndarray) -> np.ndarray:
        return (vectors - self.mean) @ self.components.T

    def to_dict(self) -> dict[str, Any]:
        return {"mean": self.mean.tolist(), "components": self.components.tolist()}


@dataclass
class MatrixScaler:
    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, matrix: np.ndarray) -> "MatrixScaler":
        mean = matrix.mean(axis=0)
        scale = matrix.std(axis=0)
        scale[scale < 1e-12] = 1.0
        return cls(mean, scale)

    def transform(self, matrix: np.ndarray) -> np.ndarray:
        return (matrix - self.mean) / self.scale

    def to_dict(self) -> dict[str, Any]:
        return {"mean": self.mean.tolist(), "scale": self.scale.tolist()}


def frame_release_keys(frame: pd.DataFrame) -> set[tuple[str, str]]:
    return {
        release_key(row[f"package_{side}_name"], row[f"package_{side}_version"])
        for _, row in frame.iterrows()
        for side in ("a", "b")
    }


def text_interaction_matrix(
    frame: pd.DataFrame,
    embeddings: dict[tuple[str, str], np.ndarray],
    projector: ReleaseProjector,
) -> np.ndarray:
    sides = []
    for side in ("a", "b"):
        vectors = []
        for _, row in frame.iterrows():
            key = release_key(row[f"package_{side}_name"], row[f"package_{side}_version"])
            if key not in embeddings:
                raise ValueError(f"missing release embedding for {key[0]}=={key[1]}")
            vectors.append(embeddings[key])
        sides.append(projector.transform(np.vstack(vectors)))
    left, right = sides
    return np.column_stack([left, right, np.abs(left - right), left * right])


def load_embeddings(path: Path) -> dict[tuple[str, str], np.ndarray]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    result = {
        release_key(row["package"], row["version"]): np.asarray(row["embedding"], dtype=float)
        for row in rows
    }
    if len(rows) != 200 or len(result) != 200:
        raise ValueError("expected 200 unique ModernBERT release embeddings")
    dimensions = {len(value) for value in result.values()}
    if dimensions != {768}:
        raise ValueError(f"unexpected embedding dimensions: {sorted(dimensions)}")
    return result


def subtype_metrics(outcomes: np.ndarray, predicted: np.ndarray) -> dict[str, float | int]:
    resolution = outcomes == "resolution_failure"
    post = np.isin(outcomes, ["import_failure", "smoke_test_failure"])
    imported = outcomes == "import_failure"
    return {
        "resolution_failure_rows": int(resolution.sum()),
        "resolution_failure_recall": float(predicted[resolution].mean()) if resolution.any() else float("nan"),
        "post_install_failure_rows": int(post.sum()),
        "post_install_failure_recall": float(predicted[post].mean()) if post.any() else float("nan"),
        "import_failure_rows": int(imported.sum()),
        "import_failure_recall": float(predicted[imported].mean()) if imported.any() else float("nan"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Train the frozen-ModernBERT DepLab hybrid")
    parser.add_argument("--features", type=Path, default=Path("outputs/deplab-expanded-development-v2.0.0/features.csv"))
    parser.add_argument("--holdout-inputs", type=Path, default=Path("outputs/deplab-expanded-development-v2.0.0/final-holdout-inputs.csv"))
    parser.add_argument("--holdout-results", type=Path, default=Path("outputs/expanded-final-holdout-results.jsonl"))
    parser.add_argument("--policy", type=Path, default=Path("outputs/deplab-expanded-development-v2.0.0/model-input-policy.json"))
    parser.add_argument("--embeddings", type=Path, default=Path("outputs/modernbert-release-embeddings-v1.0.0.jsonl"))
    parser.add_argument("--baseline-oof", type=Path, default=Path("outputs/deplab-expanded-weighted-logistic-v2.0.0/development-oof-predictions.csv"))
    parser.add_argument("--baseline-holdout-metrics", type=Path, default=Path("outputs/deplab-expanded-weighted-logistic-v2.0.0/final-holdout-evaluation/metrics.json"))
    parser.add_argument("--output-dir", type=Path, default=Path(f"outputs/{MODEL_ID}"))
    args = parser.parse_args()

    dtype = {"experiment_id": "string", "family": "string", "python_version": "string", "package_a_version": "string", "package_b_version": "string"}
    frame = pd.read_csv(args.features, dtype=dtype)
    holdout = pd.read_csv(args.holdout_inputs, dtype=dtype)
    if len(frame) != 3269 or len(holdout) != 840:
        raise ValueError(f"expected 3,269 development and 840 benchmark rows; got {len(frame)} and {len(holdout)}")
    embeddings = load_embeddings(args.embeddings)
    policy = json.loads(args.policy.read_text(encoding="utf-8"))
    numeric, categorical = list(policy["numeric_columns"]), list(policy["categorical_columns"])
    target = _to_bool(frame["is_failure"]).astype(int)
    outcomes = frame["outcome"].astype(str).to_numpy()
    folds = assign_family_folds(frame["family"].astype(str), 5)
    candidate_names = ("modernbert_weighted_logistic", "modernbert_two_head_hybrid")
    probabilities = {name: np.full(len(frame), np.nan, dtype=float) for name in candidate_names}
    components = {
        "resolution": np.full(len(frame), np.nan, dtype=float),
        "post_install": np.full(len(frame), np.nan, dtype=float),
    }

    for fold in sorted(set(folds)):
        train_mask, test_mask = folds != fold, folds == fold
        train_frame, test_frame = frame.loc[train_mask], frame.loc[test_mask]
        preprocessor = Preprocessor.fit(train_frame, numeric, categorical)
        projector = ReleaseProjector.fit(embeddings, frame_release_keys(train_frame))
        train_text_raw = text_interaction_matrix(train_frame, embeddings, projector)
        scaler = MatrixScaler.fit(train_text_raw)
        x_train = np.column_stack([preprocessor.transform(train_frame), scaler.transform(train_text_raw)])
        x_test = np.column_stack([
            preprocessor.transform(test_frame),
            scaler.transform(text_interaction_matrix(test_frame, embeddings, projector)),
        ])
        logistic = fit_weighted_logistic(x_train, target[train_mask])
        probabilities["modernbert_weighted_logistic"][test_mask] = logistic.predict_proba(x_test)
        binner = HistogramBinner.fit(x_train)
        options = {**MODEL_OPTIONS, "random_seed": MODEL_OPTIONS["random_seed"] + int(fold) * 100}
        two_head = TwoHeadBoosting.fit(binner.transform(x_train), outcomes[train_mask], **options)
        resolution, post = two_head.predict_components(binner.transform(x_test))
        components["resolution"][test_mask] = resolution
        components["post_install"][test_mask] = post
        probabilities["modernbert_two_head_hybrid"][test_mask] = 1.0 - (1.0 - resolution) * (1.0 - post)

    candidate_metrics: dict[str, dict[str, Any]] = {}
    thresholds: dict[str, float] = {}
    comparison = []
    for name in candidate_names:
        if np.isnan(probabilities[name]).any():
            raise ValueError(f"{name} did not score all development rows")
        threshold, _ = _select_threshold(target, probabilities[name])
        predicted = probabilities[name] >= threshold
        metrics = {**_metrics(target, predicted, probabilities[name]), **subtype_metrics(outcomes, predicted)}
        thresholds[name] = threshold
        candidate_metrics[name] = metrics
        comparison.append({"candidate": name, "threshold": threshold, **metrics})

    # This selection is made only from the 3,269 development rows.
    selected_name = max(candidate_names, key=lambda name: (
        candidate_metrics[name]["balanced_accuracy"],
        candidate_metrics[name]["post_install_failure_recall"],
        candidate_metrics[name]["failure_recall"],
    ))
    threshold = thresholds[selected_name]

    preprocessor = Preprocessor.fit(frame, numeric, categorical)
    projector = ReleaseProjector.fit(embeddings, frame_release_keys(frame))
    train_text_raw = text_interaction_matrix(frame, embeddings, projector)
    scaler = MatrixScaler.fit(train_text_raw)
    x_train = np.column_stack([preprocessor.transform(frame), scaler.transform(train_text_raw)])
    x_holdout = np.column_stack([
        preprocessor.transform(holdout),
        scaler.transform(text_interaction_matrix(holdout, embeddings, projector)),
    ])
    holdout_resolution = holdout_post = None
    if selected_name == "modernbert_weighted_logistic":
        final_model: Any = fit_weighted_logistic(x_train, target)
        holdout_probability = final_model.predict_proba(x_holdout)
        final_model_payload = {"type": "weighted_logistic", "weights": final_model.weights.tolist(), "intercept": final_model.intercept}
        binner = None
    else:
        binner = HistogramBinner.fit(x_train)
        final_model = TwoHeadBoosting.fit(binner.transform(x_train), outcomes, **MODEL_OPTIONS)
        holdout_resolution, holdout_post = final_model.predict_components(binner.transform(x_holdout))
        holdout_probability = 1.0 - (1.0 - holdout_resolution) * (1.0 - holdout_post)
        final_model_payload = final_model.to_dict()
    holdout_prediction = holdout_probability >= threshold

    # Outcomes are opened only now, after model, candidate and threshold are frozen above.
    result_rows = [json.loads(line) for line in args.holdout_results.read_text(encoding="utf-8").splitlines() if line.strip()]
    result_by_id = {str(row["experiment_id"]): row for row in result_rows}
    holdout_outcomes = np.asarray([result_by_id[str(value)]["outcome"] for value in holdout["experiment_id"]], dtype=str)
    holdout_target = (holdout_outcomes != "pass").astype(int)
    benchmark_metrics = {
        **_metrics(holdout_target, holdout_prediction, holdout_probability),
        **subtype_metrics(holdout_outcomes, holdout_prediction),
    }

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(comparison).to_csv(output / "development-model-comparison.csv", index=False)
    oof = frame.loc[:, ["matrix_order", "experiment_id", "family", "outcome", "is_failure"]].copy()
    oof["fold"] = folds
    for name in candidate_names:
        oof[f"{name}_probability_failure"] = probabilities[name]
        oof[f"{name}_predicted_failure"] = probabilities[name] >= thresholds[name]
    oof["two_head_probability_resolution"] = components["resolution"]
    oof["two_head_probability_post_install"] = components["post_install"]
    oof.to_csv(output / "development-oof-predictions.csv", index=False)
    benchmark = holdout.loc[:, ["matrix_order", "experiment_id", "family", "package_a_name", "package_a_version", "package_b_name", "package_b_version", "python_version"]].copy()
    benchmark["actual_outcome"] = holdout_outcomes
    benchmark["actual_failure"] = holdout_target.astype(bool)
    benchmark["predicted_probability_failure"] = holdout_probability
    benchmark["predicted_failure"] = holdout_prediction
    if holdout_resolution is not None:
        benchmark["predicted_probability_resolution"] = holdout_resolution
        benchmark["predicted_probability_post_install"] = holdout_post
    benchmark["prediction_correct"] = holdout_prediction == holdout_target.astype(bool)
    benchmark.to_csv(output / "known-840-benchmark-predictions.csv", index=False)

    baseline_dev = pd.read_csv(args.baseline_oof, dtype={"experiment_id": "string"})
    baseline_dev = baseline_dev.set_index("experiment_id").loc[frame["experiment_id"].astype(str)]
    baseline_dev_probability = pd.to_numeric(baseline_dev["predicted_probability_failure"]).to_numpy(float)
    baseline_dev_prediction = _to_bool(baseline_dev["predicted_failure"])
    baseline_dev_metrics = {**_metrics(target, baseline_dev_prediction, baseline_dev_probability), **subtype_metrics(outcomes, baseline_dev_prediction)}
    baseline_holdout = json.loads(args.baseline_holdout_metrics.read_text(encoding="utf-8"))["overall"]
    payload = {
        "model_id": MODEL_ID,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "training_rows": 3269,
        "training_rule": "model and threshold selection use only five family-separated development folds",
        "embedding_model": "answerdotai/ModernBERT-base",
        "embedding_policy": "frozen encoder; release text embedded once; no labels used by encoder",
        "text_features": f"{PCA_DIMENSIONS}-dimension release projections for A, B, absolute difference and product",
        "candidates": candidate_metrics,
        "selected_candidate": selected_name,
        "selected_threshold": threshold,
        "selection_rule": "highest development balanced accuracy, then post-install failure recall, then failure recall",
        "structured_baseline_development": baseline_dev_metrics,
        "known_840_benchmark": {
            "rows": 840,
            "status": "unseen package families, but outcomes were known before this v4 design; benchmark rather than a newly untouched final test",
            "modernbert": benchmark_metrics,
            "structured_baseline": baseline_holdout,
        },
        "source_sha256": {
            "development_features": sha256(args.features),
            "holdout_inputs": sha256(args.holdout_inputs),
            "holdout_results": sha256(args.holdout_results),
            "embeddings": sha256(args.embeddings),
        },
    }
    (output / "metrics.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    model_payload = {
        "model_id": MODEL_ID,
        "training_rows": 3269,
        "selected_candidate": selected_name,
        "threshold": threshold,
        "preprocessor": preprocessor.to_dict(),
        "release_projector": projector.to_dict(),
        "text_scaler": scaler.to_dict(),
        "binner": binner.to_dict() if binner is not None else None,
        "model": final_model_payload,
    }
    (output / "model.json").write_text(json.dumps(model_payload, separators=(",", ":")) + "\n", encoding="utf-8")
    (output / "report.md").write_text(report(payload), encoding="utf-8")
    write_checksums(output)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def report(payload: dict[str, Any]) -> str:
    dev = payload["candidates"][payload["selected_candidate"]]
    test = payload["known_840_benchmark"]["modernbert"]
    return f"""# DepLab ModernBERT hybrid v4

The model was selected using only the 3,269 development rows. ModernBERT is frozen: it converts changelog text to numeric release vectors, while the two prediction heads learn resolution failures and post-install failures separately.

## Development selection

- Selected model: **{payload['selected_candidate']}**
- Balanced accuracy: {dev['balanced_accuracy']:.3f}
- Failure recall: {dev['failure_recall']:.3f}
- Import-failure recall: {dev['import_failure_recall']:.3f}
- Frozen threshold: {payload['selected_threshold']:.6f}

## 840-row package-family benchmark

- Accuracy: {test['accuracy']:.3f}
- Balanced accuracy: {test['balanced_accuracy']:.3f}
- Failure precision: {test['failure_precision']:.3f}
- Failure recall: {test['failure_recall']:.3f}
- Failure F1: {test['failure_f1']:.3f}
- Import-failure recall: {test['import_failure_recall']:.3f}

These 840 rows use unseen package families, but their outcomes had already been examined before v4 was designed. A later final round must retrain on all 4,109 rows and evaluate on newly collected families.
"""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_checksums(output: Path) -> None:
    checksum = output / "SHA256SUMS.txt"
    lines = [f"{sha256(path)}  {path.name}" for path in sorted(output.iterdir()) if path.is_file() and path != checksum]
    checksum.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
