from __future__ import annotations

import bisect
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .model_runtime import FrozenPreprocessor, FrozenTreeHead
from .validation import canonical_name


EXPECTED_PIPELINE_ID = "deplab-large-hybrid-v3.0.0"
EXPECTED_EMBEDDING_MODEL = "answerdotai/ModernBERT-base"


def _sigmoid(value: float) -> float:
    value = max(-35.0, min(35.0, value))
    return 1.0 / (1.0 + math.exp(-value))


@dataclass(frozen=True)
class StructuredRisk:
    probability: float
    predicted_failure: bool


class StructuredRankingModel:
    """Fast ranking surrogate. A resolver remains authoritative."""

    def __init__(self, model_path: Path) -> None:
        payload = _read_model(
            model_path,
            expected_candidate="structured_weighted_logistic",
        )
        model = payload["model"]
        if model.get("type") != "weighted_logistic":
            raise ValueError("unexpected structured model type")
        self.model_id = str(payload["candidate"])
        self.preprocessor = FrozenPreprocessor(payload["preprocessor"])
        self.weights = [float(value) for value in model["weights"]]
        self.intercept = float(model["intercept"])
        self.threshold = float(payload["threshold"])

    def score(self, row: dict[str, Any]) -> StructuredRisk:
        features = self.preprocessor.transform(row)
        if len(features) != len(self.weights):
            raise ValueError("structured feature width differs from frozen weights")
        probability = _sigmoid(
            self.intercept
            + sum(weight * value for weight, value in zip(self.weights, features))
        )
        return StructuredRisk(
            probability=probability,
            predicted_failure=probability >= self.threshold,
        )


@dataclass(frozen=True)
class PostInstallRisk:
    import_probability: float
    smoke_probability: float
    combined_probability: float
    predicted_import_failure: bool
    predicted_smoke_failure: bool
    risk_detected: bool


class ReleaseEmbeddingStore:
    def __init__(self, path: Path) -> None:
        self.vectors: dict[tuple[str, str], list[float]] = {}
        dimensions: set[int] = set()
        models: set[str] = set()
        with path.open(encoding="utf-8") as file:
            for line in file:
                if not line.strip():
                    continue
                row = json.loads(line)
                key = (canonical_name(row["package"]), str(row["version"]))
                if key in self.vectors:
                    raise ValueError(f"duplicate release embedding: {key}")
                vector = [float(value) for value in row["embedding"]]
                dimensions.add(len(vector))
                models.add(str(row["model"]))
                self.vectors[key] = vector
        if dimensions != {768}:
            raise ValueError(f"unexpected embedding dimensions: {sorted(dimensions)}")
        if models != {EXPECTED_EMBEDDING_MODEL}:
            raise ValueError(f"unexpected embedding models: {sorted(models)}")

    def get(self, package: str, version: str) -> list[float] | None:
        return self.vectors.get((canonical_name(package), str(version)))


class FrozenProjection:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.mean = [float(value) for value in payload["mean"]]
        self.components = [
            [float(value) for value in component]
            for component in payload["components"]
        ]
        if not self.components or any(
            len(component) != len(self.mean) for component in self.components
        ):
            raise ValueError("invalid frozen release projection")

    def transform(self, vector: list[float]) -> list[float]:
        if len(vector) != len(self.mean):
            raise ValueError("embedding width differs from frozen projection")
        centered = [
            value - mean for value, mean in zip(vector, self.mean)
        ]
        return [
            sum(value * weight for value, weight in zip(centered, component))
            for component in self.components
        ]


class FrozenScaler:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.mean = [float(value) for value in payload["mean"]]
        self.scale = [float(value) for value in payload["scale"]]
        if len(self.mean) != len(self.scale) or any(
            value <= 0.0 for value in self.scale
        ):
            raise ValueError("invalid frozen text scaler")

    def transform(self, values: list[float]) -> list[float]:
        if len(values) != len(self.mean):
            raise ValueError("text feature width differs from frozen scaler")
        return [
            (value - mean) / scale
            for value, mean, scale in zip(values, self.mean, self.scale)
        ]


class ModernBertPostInstallModel:
    """Runs frozen release embeddings and stage heads without the encoder."""

    def __init__(
        self,
        model_path: Path,
        embeddings_path: Path,
        post_install_threshold: float,
    ) -> None:
        payload = _read_model(
            model_path,
            expected_candidate="modernbert_stage_aware_hybrid",
        )
        model = payload["model"]
        if model.get("type") != "stage_aware_histogram_gradient_boosting":
            raise ValueError("unexpected ModernBERT stage-aware model type")
        self.model_id = str(payload["candidate"])
        self.preprocessor = FrozenPreprocessor(payload["preprocessor"])
        text = payload["text_features"]
        if (
            text.get("encoder") != EXPECTED_EMBEDDING_MODEL
            or text.get("encoder_frozen") is not True
        ):
            raise ValueError("unexpected frozen embedding configuration")
        self.projector = FrozenProjection(text["release_projector"])
        self.scaler = FrozenScaler(text["text_scaler"])
        self.bin_thresholds = [
            [float(value) for value in cuts]
            for cuts in payload["binner"]["thresholds"]
        ]
        self.import_head = FrozenTreeHead(model["import_failure_head"])
        self.smoke_head = FrozenTreeHead(model["smoke_failure_head"])
        self.embeddings = ReleaseEmbeddingStore(embeddings_path)
        self._projected: dict[tuple[str, str], list[float]] = {}
        self.post_install_threshold = _probability(
            post_install_threshold,
            "post-install threshold",
        )

    def has_releases(self, row: dict[str, Any]) -> bool:
        return all(
            self.embeddings.get(
                str(row[f"package_{side}_name"]),
                str(row[f"package_{side}_version"]),
            )
            is not None
            for side in ("a", "b")
        )

    def score(self, row: dict[str, Any]) -> PostInstallRisk | None:
        left = self._projection(
            str(row["package_a_name"]), str(row["package_a_version"])
        )
        right = self._projection(
            str(row["package_b_name"]), str(row["package_b_version"])
        )
        if left is None or right is None:
            return None
        interactions = _interactions(left, right)
        features = (
            self.preprocessor.transform(row)
            + self.scaler.transform(interactions)
        )
        bins = _bin_features(features, self.bin_thresholds)
        import_probability = self.import_head.probability(bins)
        smoke_probability = self.smoke_head.probability(bins)
        combined = 1.0 - (1.0 - import_probability) * (1.0 - smoke_probability)
        risk_detected = combined >= self.post_install_threshold
        import_failure = risk_detected and import_probability >= smoke_probability
        smoke_failure = risk_detected and smoke_probability > import_probability
        return PostInstallRisk(
            import_probability=import_probability,
            smoke_probability=smoke_probability,
            combined_probability=combined,
            predicted_import_failure=import_failure,
            predicted_smoke_failure=smoke_failure,
            risk_detected=risk_detected,
        )

    def _projection(
        self, package: str, version: str
    ) -> list[float] | None:
        key = (canonical_name(package), str(version))
        existing = self._projected.get(key)
        if existing is not None:
            return existing
        vector = self.embeddings.get(package, version)
        if vector is None:
            return None
        projected = self.projector.transform(vector)
        self._projected[key] = projected
        return projected


def _interactions(left: list[float], right: list[float]) -> list[float]:
    if len(left) != len(right):
        raise ValueError("projected release vectors have different widths")
    return (
        left
        + right
        + [abs(a - b) for a, b in zip(left, right)]
        + [a * b for a, b in zip(left, right)]
    )


def _bin_features(
    features: list[float],
    thresholds: list[list[float]],
) -> list[int]:
    if len(features) != len(thresholds):
        raise ValueError("hybrid feature width differs from frozen binner")
    return [
        bisect.bisect_right(cuts, value)
        for cuts, value in zip(thresholds, features)
    ]


def _read_model(path: Path, expected_candidate: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("pipeline_id") != EXPECTED_PIPELINE_ID:
        raise ValueError("unexpected production model pipeline")
    if payload.get("candidate") != expected_candidate:
        raise ValueError(f"unexpected candidate in {path}")
    if int(payload.get("training_rows", 0)) != 21_490:
        raise ValueError("unexpected production model training-row count")
    return payload


def _probability(value: float, label: str) -> float:
    result = float(value)
    if not 0.0 < result < 1.0:
        raise ValueError(f"{label} must be between zero and one")
    return result
