from __future__ import annotations

import bisect
import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .validation import canonical_name


def _number(value: Any, fallback: float) -> float:
    if value is None or value == "":
        return fallback
    if isinstance(value, bool):
        return float(value)
    text = str(value).strip().lower()
    if text == "true":
        return 1.0
    if text == "false":
        return 0.0
    try:
        return float(text)
    except ValueError:
        return fallback


def _sigmoid(value: float) -> float:
    value = max(-35.0, min(35.0, value))
    return 1.0 / (1.0 + math.exp(-value))


def _truth(value: Any) -> bool:
    return str(value).strip().lower() == "true"


class FrozenPreprocessor:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.numeric_columns = payload["numeric_columns"]
        self.categorical_columns = payload["categorical_columns"]
        self.numeric_medians = payload["numeric_medians"]
        self.numeric_means = payload["numeric_means"]
        self.numeric_scales = payload["numeric_scales"]
        self.category_levels = payload["category_levels"]

    def transform(self, row: dict[str, Any]) -> list[float]:
        result: list[float] = []
        for column in self.numeric_columns:
            value = _number(row.get(column), float(self.numeric_medians[column]))
            result.append(
                (value - float(self.numeric_means[column]))
                / float(self.numeric_scales[column])
            )
        for column in self.categorical_columns:
            value = "<missing>" if row.get(column) in (None, "") else str(row[column])
            result.extend(float(value == level) for level in self.category_levels[column])
        return result


class FrozenTreeHead:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.base_logit = float(payload["base_logit"])
        self.learning_rate = float(payload["learning_rate"])
        self.trees = payload["trees"]

    @staticmethod
    def _leaf(nodes: list[dict[str, Any]], bins: list[int]) -> float:
        index = 0
        while int(nodes[index]["feature"]) >= 0:
            node = nodes[index]
            index = int(node["left"] if bins[int(node["feature"])] <= int(node["split_bin"]) else node["right"])
        return float(nodes[index]["value"])

    def probability(self, bins: list[int]) -> float:
        logit = self.base_logit
        for tree in self.trees:
            logit += self.learning_rate * self._leaf(tree["nodes"], bins)
        return _sigmoid(logit)


@dataclass(frozen=True)
class ModelScore:
    risk_score: float
    predicted_failure: bool
    logistic_probability: float
    resolution_probability: float
    post_install_probability: float
    likely_stage: str


class HybridModel:
    """Loads the frozen real DepLab models without retraining them in the app."""

    def __init__(self, logistic_path: Path, advanced_path: Path, hybrid_metrics_path: Path) -> None:
        logistic = json.loads(logistic_path.read_text(encoding="utf-8"))
        advanced = json.loads(advanced_path.read_text(encoding="utf-8"))
        metrics = json.loads(hybrid_metrics_path.read_text(encoding="utf-8"))
        self.model_id = "deplab-hybrid-validation-v3.0.0"
        self.preprocessor = FrozenPreprocessor(logistic["preprocessor"])
        self.weights = [float(value) for value in logistic["weights"]]
        self.intercept = float(logistic["intercept"])
        self.logistic_threshold = float(logistic["threshold"])
        self.bin_thresholds = advanced["binner"]["thresholds"]
        self.resolution_head = FrozenTreeHead(advanced["model"]["resolution_head"])
        self.post_install_head = FrozenTreeHead(advanced["model"]["post_install_head"])
        self.post_threshold = float(
            metrics["known_840_validation_selected_operating_point"]["post_install_threshold"]
        )

    def score(self, row: dict[str, Any]) -> ModelScore:
        features = self.preprocessor.transform(row)
        logistic = _sigmoid(
            self.intercept + sum(weight * value for weight, value in zip(self.weights, features))
        )
        bins = [
            bisect.bisect_right([float(cut) for cut in cuts], value)
            for cuts, value in zip(self.bin_thresholds, features)
        ]
        resolution = self.resolution_head.probability(bins)
        post_install = self.post_install_head.probability(bins)
        predicted = logistic >= self.logistic_threshold or post_install >= self.post_threshold
        likely_stage = "post-install/import" if post_install >= resolution else "dependency resolution"
        return ModelScore(
            risk_score=max(logistic, post_install),
            predicted_failure=predicted,
            logistic_probability=logistic,
            resolution_probability=resolution,
            post_install_probability=post_install,
            likely_stage=likely_stage,
        )


class FeatureTable:
    """Exact frozen feature rows; replaceable later with a live feature provider."""

    def __init__(self, paths: Iterable[Path]) -> None:
        self.rows: list[dict[str, str]] = []
        self.index: dict[tuple[str, str, str, str, str], dict[str, str]] = {}
        for path in paths:
            with path.open(encoding="utf-8", newline="") as file:
                for row in csv.DictReader(file):
                    key = (
                        canonical_name(row["package_a_name"]),
                        row["package_a_version"],
                        canonical_name(row["package_b_name"]),
                        row["package_b_version"],
                        row["python_version"],
                    )
                    self.index.setdefault(key, row)
                    self.rows.append(row)

    def find(
        self,
        package_one: str,
        version_one: str,
        package_two: str,
        version_two: str,
        python_version: str,
    ) -> dict[str, str] | None:
        one, two = canonical_name(package_one), canonical_name(package_two)
        direct = self.index.get((one, version_one, two, version_two, python_version))
        if direct:
            return direct
        return self.index.get((two, version_two, one, version_one, python_version))

    def target_versions(
        self,
        target: str,
        related: str,
        related_version: str,
        python_version: str,
    ) -> set[str]:
        target, related = canonical_name(target), canonical_name(related)
        versions: set[str] = set()
        for row in self.rows:
            if row["python_version"] != python_version:
                continue
            a, b = canonical_name(row["package_a_name"]), canonical_name(row["package_b_name"])
            if a == target and b == related and row["package_b_version"] == related_version:
                versions.add(row["package_a_version"])
            elif b == target and a == related and row["package_a_version"] == related_version:
                versions.add(row["package_b_version"])
        return versions

    def pair_rows(
        self,
        target: str,
        target_version: str | None,
        related: str,
        python_version: str,
    ) -> list[tuple[dict[str, str], str, str]]:
        """Return (row, target version, related version) in a stable orientation."""
        target, related = canonical_name(target), canonical_name(related)
        result: list[tuple[dict[str, str], str, str]] = []
        for row in self.rows:
            if row["python_version"] != python_version:
                continue
            a, b = canonical_name(row["package_a_name"]), canonical_name(row["package_b_name"])
            if a == target and b == related:
                candidate_target = row["package_a_version"]
                candidate_related = row["package_b_version"]
            elif b == target and a == related:
                candidate_target = row["package_b_version"]
                candidate_related = row["package_a_version"]
            else:
                continue
            if target_version is None or candidate_target == target_version:
                result.append((row, candidate_target, candidate_related))
        return result

    def has_family(self, target: str, related: str, python_version: str) -> bool:
        return bool(self.pair_rows(target, None, related, python_version))

    @staticmethod
    def published_constraints_allow(row: dict[str, str]) -> bool:
        """Check direct A/B declarations recorded from published package metadata."""
        a_allows = not _truth(row.get("package_a_declares_package_b")) or _truth(
            row.get("package_a_requirement_allows_b")
        )
        b_allows = not _truth(row.get("package_b_declares_package_a")) or _truth(
            row.get("package_b_requirement_allows_a")
        )
        return a_allows and b_allows

    @staticmethod
    def constraint_conflict_records(row: dict[str, str]) -> list[dict[str, Any]]:
        """Return deterministic direct-constraint evidence in package orientation."""
        records: list[dict[str, Any]] = []
        directions = (
            (
                "a",
                "b",
                "package_a_declares_package_b",
                "package_a_requirement_allows_b",
                "package_a_requirement_on_b",
            ),
            (
                "b",
                "a",
                "package_b_declares_package_a",
                "package_b_requirement_allows_a",
                "package_b_requirement_on_a",
            ),
        )
        for declaring, dependency, declares_key, allows_key, requirement_key in directions:
            if not _truth(row.get(declares_key)) or _truth(row.get(allows_key)):
                continue
            requirement = str(row.get(requirement_key) or "").split(";", 1)[0].strip()
            dependency_version = row[f"package_{dependency}_version"]
            records.append(
                {
                    "declaring_package": canonical_name(row[f"package_{declaring}_name"]),
                    "declaring_version": row[f"package_{declaring}_version"],
                    "dependency_package": canonical_name(row[f"package_{dependency}_name"]),
                    "dependency_version": dependency_version,
                    "requirement": requirement,
                    "blocking_specifiers": FeatureTable._blocking_specifiers(
                        requirement, dependency_version
                    ),
                }
            )
        return records

    @staticmethod
    def _blocking_specifiers(requirement: str, version: str) -> list[str]:
        """Identify the simple published clauses that reject an exact numeric release."""
        candidate = FeatureTable._numeric_version(version)
        match = re.match(r"^[A-Za-z0-9._-]+(?:\[[^]]+\])?\s*(.*)$", requirement)
        specifier_text = match.group(1).strip() if match else ""
        blocking: list[str] = []
        for clause in (item.strip() for item in specifier_text.split(",")):
            comparison = re.fullmatch(r"(===|==|!=|<=|>=|<|>)\s*([A-Za-z0-9.*+!_-]+)", clause)
            if not comparison:
                continue
            operator, boundary_text = comparison.groups()
            if "*" in boundary_text or not FeatureTable._simple_clause_allows(
                candidate, operator, FeatureTable._numeric_version(boundary_text)
            ):
                blocking.append(clause)
        return blocking

    @staticmethod
    def _numeric_version(version: str) -> tuple[int, ...]:
        parts = [int(value) for value in re.findall(r"\d+", version)[:4]]
        return tuple((parts + [0, 0, 0, 0])[:4])

    @staticmethod
    def _simple_clause_allows(
        candidate: tuple[int, ...], operator: str, boundary: tuple[int, ...]
    ) -> bool:
        return {
            "<": candidate < boundary,
            "<=": candidate <= boundary,
            ">": candidate > boundary,
            ">=": candidate >= boundary,
            "==": candidate == boundary,
            "===": candidate == boundary,
            "!=": candidate != boundary,
        }[operator]
