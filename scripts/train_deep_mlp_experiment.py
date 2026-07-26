from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import random
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
    from train_expanded_baseline import Preprocessor, _metrics
    from train_large_hybrid import (
        _hybrid_matrices,
        _load_frames,
        _validate_embedding_coverage,
        _validate_model_inputs,
        assign_balanced_family_folds,
        load_embeddings,
    )
except ImportError:
    from scripts.evaluate_large_validation import (
        _read_jsonl,
        _validate_validation_results,
    )
    from scripts.train_expanded_baseline import Preprocessor, _metrics
    from scripts.train_large_hybrid import (
        _hybrid_matrices,
        _load_frames,
        _validate_embedding_coverage,
        _validate_model_inputs,
        assign_balanced_family_folds,
        load_embeddings,
    )


SCHEMA_VERSION = "3.0.0"
MODEL_ID = "deplab-modernbert-mlp-experiment-v3.0.0"
OUTCOMES = (
    "pass",
    "resolution_failure",
    "import_failure",
    "smoke_test_failure",
)
OUTCOME_TO_CLASS = {name: index for index, name in enumerate(OUTCOMES)}
DEFAULT_SEED = 20260727


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
    labels = encode_outcomes(frame["outcome"].astype(str).to_numpy())
    binary = (labels != OUTCOME_TO_CLASS["pass"]).astype(int)
    folds, assignments = assign_balanced_family_folds(
        frame["family"].astype(str), binary, fold_count=5
    )

    development_mask = folds == args.development_fold
    development_train = frame.loc[~development_mask]
    development_test = frame.loc[development_mask]
    development_train_labels = labels[~development_mask]
    development_test_labels = labels[development_mask]
    dev_train_matrix, dev_test_matrix, _ = prepare_matrices(
        development_train,
        development_test,
        numeric,
        categorical,
        embeddings,
    )
    development_model, development_history = train_mlp(
        dev_train_matrix,
        development_train_labels,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        hidden_sizes=(args.hidden_one, args.hidden_two),
        dropout=args.dropout,
        seed=args.seed,
    )
    development_probabilities = predict_probabilities(
        development_model, dev_test_matrix, args.batch_size
    )
    development_metrics = classification_metrics(
        development_test_labels, development_probabilities
    )

    final_train_matrix, validation_matrix, transforms = prepare_matrices(
        frame,
        validation,
        numeric,
        categorical,
        embeddings,
    )
    final_model, final_history = train_mlp(
        final_train_matrix,
        labels,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        hidden_sizes=(args.hidden_one, args.hidden_two),
        dropout=args.dropout,
        seed=args.seed,
    )
    validation_outcomes = load_validation_outcomes(
        validation, args.validation_results
    )
    validation_labels = encode_outcomes(validation_outcomes)
    validation_probabilities = predict_probabilities(
        final_model, validation_matrix, args.batch_size
    )
    validation_metrics = classification_metrics(
        validation_labels, validation_probabilities
    )

    output.mkdir(parents=True)
    model_path = output / "model-state.pt"
    _save_model(final_model, model_path)
    transform_path = output / "input-transforms.json"
    transform_path.write_text(
        json.dumps(transforms, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    development_path = write_predictions(
        output / "development-held-family-predictions.csv",
        development_test,
        development_test_labels,
        development_probabilities,
    )
    validation_path = write_predictions(
        output / "validation-predictions.csv",
        validation.assign(outcome=validation_outcomes),
        validation_labels,
        validation_probabilities,
    )
    baseline = load_baseline(args.baseline_validation_metrics)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "model_id": MODEL_ID,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "architecture": {
            "type": "four-class feed-forward neural network",
            "input_size": int(final_train_matrix.shape[1]),
            "hidden_sizes": [args.hidden_one, args.hidden_two],
            "activation": "GELU",
            "dropout": args.dropout,
            "output_classes": list(OUTCOMES),
        },
        "training": {
            "rows": len(frame),
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "seed": args.seed,
            "class_weighting": "square-root inverse frequency",
            "final_history": final_history,
        },
        "development_check": {
            "held_out_fold": args.development_fold,
            "held_out_families": sorted(
                development_test["family"].astype(str).unique().tolist()
            ),
            "rows": len(development_test),
            "metrics": development_metrics,
            "history": development_history,
        },
        "validation": {
            "rows": len(validation),
            "metrics": validation_metrics,
            "baseline": baseline,
            "status": "tuning evidence because validation was previously opened",
        },
        "development_folds": assignments,
        "embedding_model": "answerdotai/ModernBERT-base",
        "final_test_outcomes_used": False,
        "source_sha256": {
            "development_features": _sha256(args.features),
            "validation_inputs": _sha256(args.validation_inputs),
            "validation_results": _sha256(args.validation_results),
            "release_embeddings": _sha256(args.embeddings),
            "model_state": _sha256(model_path),
            "input_transforms": _sha256(transform_path),
            "development_predictions": _sha256(development_path),
            "validation_predictions": _sha256(validation_path),
        },
        "runtime": runtime_versions(),
    }
    metrics_path = output / "experiment-metrics.json"
    metrics_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output / "report.md").write_text(report(payload), encoding="utf-8")
    write_checksums(output)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a simple four-class MLP without reading final-test outcomes"
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
        default=Path("outputs/large-release-modernbert-v3.0.0.jsonl"),
    )
    parser.add_argument(
        "--baseline-validation-metrics",
        type=Path,
        default=Path(
            "outputs/deplab-large-candidate-freeze-v3.0.0/"
            "validation-evaluation/validation-metrics.json"
        ),
    )
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--hidden-one", type=int, default=256)
    parser.add_argument("--hidden-two", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--development-fold", type=int, default=5)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(f"outputs/{MODEL_ID}"),
    )
    args = parser.parse_args()
    _validate_arguments(args)
    return args


def _validate_arguments(args: argparse.Namespace) -> None:
    positive = (
        args.epochs,
        args.batch_size,
        args.learning_rate,
        args.hidden_one,
        args.hidden_two,
    )
    if any(value <= 0 for value in positive):
        raise ValueError("training sizes, epochs and learning rate must be positive")
    if not 0 <= args.dropout < 1:
        raise ValueError("dropout must be at least zero and below one")
    if args.development_fold not in range(1, 6):
        raise ValueError("development fold must be between 1 and 5")


def encode_outcomes(outcomes: np.ndarray) -> np.ndarray:
    unknown = sorted(set(map(str, outcomes)) - set(OUTCOME_TO_CLASS))
    if unknown:
        raise ValueError(f"unknown outcomes: {unknown}")
    return np.asarray([OUTCOME_TO_CLASS[str(value)] for value in outcomes], dtype=int)


def class_weights(labels: np.ndarray, class_count: int = 4) -> np.ndarray:
    counts = np.bincount(labels.astype(int), minlength=class_count).astype(float)
    if len(counts) != class_count or np.any(counts <= 0):
        raise ValueError("every output class must have at least one training row")
    weights = np.sqrt(len(labels) / (class_count * counts))
    return weights / weights.mean()


def prepare_matrices(
    train: pd.DataFrame,
    other: pd.DataFrame,
    numeric: list[str],
    categorical: list[str],
    embeddings: dict[tuple[str, str], np.ndarray],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    preprocessor = Preprocessor.fit(train, numeric, categorical)
    structured_train = preprocessor.transform(train)
    structured_other = preprocessor.transform(other)
    train_matrix, other_matrix = _hybrid_matrices(
        train,
        other,
        structured_train,
        structured_other,
        embeddings,
    )
    return (
        train_matrix.astype(np.float32),
        other_matrix.astype(np.float32),
        {
            "preprocessor": preprocessor.to_dict(),
            "text_projection": (
                "32-dimensional training-only PCA for each release, followed by "
                "A, B, absolute-difference and product interactions"
            ),
        },
    )


def train_mlp(
    matrix: np.ndarray,
    labels: np.ndarray,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    hidden_sizes: tuple[int, int],
    dropout: float,
    seed: int,
) -> tuple[Any, list[dict[str, float]]]:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    configure_torch(torch, seed)
    model = make_model(nn, matrix.shape[1], hidden_sizes, dropout)
    features = torch.from_numpy(matrix)
    targets = torch.from_numpy(labels.astype(np.int64))
    dataset = TensorDataset(features, targets)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
    )
    weights = torch.tensor(class_weights(labels), dtype=torch.float32)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=0.0001
    )
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        seen = 0
        for batch_features, batch_targets in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(batch_features), batch_targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            count = len(batch_targets)
            total_loss += float(loss.detach()) * count
            seen += count
        average = total_loss / max(1, seen)
        history.append({"epoch": epoch, "training_loss": average})
        if epoch == 1 or epoch % 5 == 0 or epoch == epochs:
            print(f"Epoch {epoch:03d}/{epochs:03d}: loss={average:.6f}", flush=True)
    model.eval()
    return model, history


def configure_torch(torch: Any, seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))
    torch.use_deterministic_algorithms(True)


def make_model(
    nn: Any,
    input_size: int,
    hidden_sizes: tuple[int, int],
    dropout: float,
) -> Any:
    return nn.Sequential(
        nn.Linear(input_size, hidden_sizes[0]),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_sizes[0], hidden_sizes[1]),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_sizes[1], len(OUTCOMES)),
    )


def predict_probabilities(
    model: Any, matrix: np.ndarray, batch_size: int
) -> np.ndarray:
    import torch

    blocks = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(matrix), batch_size):
            logits = model(torch.from_numpy(matrix[start : start + batch_size]))
            blocks.append(torch.softmax(logits, dim=1).cpu().numpy())
    return np.vstack(blocks).astype(float)


def classification_metrics(
    labels: np.ndarray, probabilities: np.ndarray
) -> dict[str, Any]:
    if probabilities.shape != (len(labels), len(OUTCOMES)):
        raise ValueError("probability matrix has an unexpected shape")
    failure_probability = 1.0 - probabilities[:, OUTCOME_TO_CLASS["pass"]]
    predicted_failure = failure_probability >= 0.5
    actual_failure = labels != OUTCOME_TO_CLASS["pass"]
    predicted_class = probabilities.argmax(axis=1)
    metrics: dict[str, Any] = {
        **_metrics(actual_failure.astype(int), predicted_failure, failure_probability),
        "exact_stage_accuracy": float(np.mean(predicted_class == labels)),
    }
    subtype_recalls = []
    for name in OUTCOMES[1:]:
        class_id = OUTCOME_TO_CLASS[name]
        mask = labels == class_id
        detected = int(np.sum(predicted_failure[mask]))
        exact = int(np.sum(predicted_class[mask] == class_id))
        rows = int(mask.sum())
        metrics[f"{name}_rows"] = rows
        metrics[f"{name}_detected"] = detected
        metrics[f"{name}_detection_recall"] = detected / rows if rows else 0.0
        metrics[f"{name}_correct_stage"] = exact
        metrics[f"{name}_stage_recall"] = exact / rows if rows else 0.0
        if rows:
            subtype_recalls.append(detected / rows)
    metrics["failure_subtype_macro_detection_recall"] = float(
        np.mean(subtype_recalls)
    )
    return metrics


def load_validation_outcomes(
    validation: pd.DataFrame, path: Path
) -> np.ndarray:
    indexed = _validate_validation_results(_read_jsonl(path))
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


def write_predictions(
    path: Path,
    frame: pd.DataFrame,
    labels: np.ndarray,
    probabilities: np.ndarray,
) -> Path:
    predicted_class = probabilities.argmax(axis=1)
    result = frame.loc[:, ["experiment_id", "family", "outcome"]].copy()
    result["actual_class"] = [OUTCOMES[value] for value in labels]
    result["predicted_class"] = [OUTCOMES[value] for value in predicted_class]
    result["probability_failure"] = 1.0 - probabilities[:, 0]
    result["predicted_failure"] = result["probability_failure"] >= 0.5
    for index, name in enumerate(OUTCOMES):
        result[f"probability_{name}"] = probabilities[:, index]
    result.to_csv(path, index=False, quoting=csv.QUOTE_MINIMAL)
    return path


def load_baseline(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    selected = str(payload["selected_candidate"])
    return {
        "selected_candidate": selected,
        "metrics": payload["candidate_metrics"][selected],
    }


def _save_model(model: Any, path: Path) -> None:
    import torch

    torch.save(model.state_dict(), path)


def runtime_versions() -> dict[str, str]:
    import torch

    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
    }


def report(payload: dict[str, Any]) -> str:
    metrics = payload["validation"]["metrics"]
    return f"""# DepLab simple deep-learning experiment

The model is a small four-class feed-forward neural network using the same
structured inputs and frozen ModernBERT release embeddings as the existing
hybrid candidates.

- Validation rows: **{payload['validation']['rows']:,}**
- Accuracy: **{metrics['accuracy']:.3f}**
- Balanced accuracy: **{metrics['balanced_accuracy']:.3f}**
- Failure precision: **{metrics['failure_precision']:.3f}**
- Failure recall: **{metrics['failure_recall']:.3f}**
- Import failures detected: **{metrics['import_failure_detected']}/{metrics['import_failure_rows']}**
- Smoke failures detected: **{metrics['smoke_test_failure_detected']}/{metrics['smoke_test_failure_rows']}**
- False warnings: **{metrics['false_failure']}**

The final-test outcomes were not read. Validation was already opened before this
experiment, so these numbers are tuning evidence rather than a final unbiased
evaluation.
"""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_checksums(output: Path) -> None:
    checksum = output / "SHA256SUMS.txt"
    lines = [
        f"{_sha256(path)}  {path.name}"
        for path in sorted(output.iterdir())
        if path.is_file() and path != checksum
    ]
    checksum.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
