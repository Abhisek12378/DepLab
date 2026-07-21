from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


def sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(values, -35.0, 35.0)))


def balanced_binary_weights(target: np.ndarray) -> np.ndarray:
    target = target.astype(int)
    positive = int(target.sum())
    negative = len(target) - positive
    if positive == 0 or negative == 0:
        raise ValueError("both classes are required")
    return np.where(
        target == 1,
        len(target) / (2.0 * positive),
        len(target) / (2.0 * negative),
    )


@dataclass
class HistogramBinner:
    thresholds: list[np.ndarray]

    @classmethod
    def fit(cls, matrix: np.ndarray, maximum_bins: int = 24) -> "HistogramBinner":
        quantiles = np.linspace(0.0, 1.0, maximum_bins + 1)[1:-1]
        thresholds = []
        for column in range(matrix.shape[1]):
            values = matrix[:, column]
            cuts = np.unique(np.quantile(values, quantiles))
            minimum = float(values.min())
            maximum = float(values.max())
            cuts = cuts[(cuts > minimum) & (cuts < maximum)]
            thresholds.append(cuts.astype(float))
        return cls(thresholds)

    def transform(self, matrix: np.ndarray) -> np.ndarray:
        result = np.empty(matrix.shape, dtype=np.uint8)
        for column, cuts in enumerate(self.thresholds):
            result[:, column] = np.searchsorted(cuts, matrix[:, column], side="right")
        return result

    def to_dict(self) -> dict[str, Any]:
        return {"thresholds": [cuts.tolist() for cuts in self.thresholds]}


@dataclass
class BoostNode:
    value: float
    feature: int = -1
    split_bin: int = -1
    left: int = -1
    right: int = -1

    def to_dict(self) -> dict[str, Any]:
        return {
            "value": self.value,
            "feature": self.feature,
            "split_bin": self.split_bin,
            "left": self.left,
            "right": self.right,
        }


class NewtonHistogramTree:
    def __init__(
        self,
        maximum_depth: int = 3,
        minimum_leaf: int = 20,
        l2: float = 2.0,
        minimum_gain: float = 1e-7,
        feature_fraction: float = 0.6,
        random_seed: int = 0,
    ) -> None:
        self.maximum_depth = maximum_depth
        self.minimum_leaf = minimum_leaf
        self.l2 = l2
        self.minimum_gain = minimum_gain
        self.feature_fraction = feature_fraction
        self.random_seed = random_seed
        self.nodes: list[BoostNode] = []

    def fit(self, matrix: np.ndarray, gradient: np.ndarray, hessian: np.ndarray) -> "NewtonHistogramTree":
        self.nodes = []
        self._rng = np.random.default_rng(self.random_seed)
        self._matrix = matrix
        self._gradient = gradient
        self._hessian = hessian
        self._grow(np.arange(len(matrix), dtype=int), 0)
        del self._matrix, self._gradient, self._hessian, self._rng
        return self

    def _grow(self, indexes: np.ndarray, depth: int) -> int:
        gradient_sum = float(self._gradient[indexes].sum())
        hessian_sum = float(self._hessian[indexes].sum())
        value = gradient_sum / (hessian_sum + self.l2)
        node_index = len(self.nodes)
        self.nodes.append(BoostNode(value=float(np.clip(value, -5.0, 5.0))))
        if depth >= self.maximum_depth or len(indexes) < 2 * self.minimum_leaf:
            return node_index

        feature_count = self._matrix.shape[1]
        selected_count = max(1, int(round(feature_count * self.feature_fraction)))
        selected = self._rng.choice(feature_count, selected_count, replace=False)
        parent_score = gradient_sum**2 / (hessian_sum + self.l2)
        best_gain = self.minimum_gain
        best_feature = -1
        best_split = -1
        best_mask: np.ndarray | None = None
        for feature in selected:
            bins = self._matrix[indexes, feature].astype(int)
            bin_count = int(bins.max()) + 1
            if bin_count <= 1:
                continue
            counts = np.bincount(bins, minlength=bin_count)
            gradients = np.bincount(
                bins, weights=self._gradient[indexes], minlength=bin_count
            )
            hessians = np.bincount(
                bins, weights=self._hessian[indexes], minlength=bin_count
            )
            left_count = np.cumsum(counts)[:-1]
            right_count = len(indexes) - left_count
            valid = (left_count >= self.minimum_leaf) & (right_count >= self.minimum_leaf)
            if not valid.any():
                continue
            left_gradient = np.cumsum(gradients)[:-1]
            left_hessian = np.cumsum(hessians)[:-1]
            right_gradient = gradient_sum - left_gradient
            right_hessian = hessian_sum - left_hessian
            gains = (
                left_gradient**2 / (left_hessian + self.l2)
                + right_gradient**2 / (right_hessian + self.l2)
                - parent_score
            )
            gains[~valid] = -np.inf
            split = int(np.argmax(gains))
            gain = float(gains[split])
            if gain > best_gain:
                best_gain = gain
                best_feature = int(feature)
                best_split = split
                best_mask = bins <= split
        if best_mask is None:
            return node_index

        left_indexes = indexes[best_mask]
        right_indexes = indexes[~best_mask]
        left = self._grow(left_indexes, depth + 1)
        right = self._grow(right_indexes, depth + 1)
        self.nodes[node_index].feature = best_feature
        self.nodes[node_index].split_bin = best_split
        self.nodes[node_index].left = left
        self.nodes[node_index].right = right
        return node_index

    def predict(self, matrix: np.ndarray) -> np.ndarray:
        result = np.empty(len(matrix), dtype=float)
        self._predict_node(0, np.arange(len(matrix), dtype=int), matrix, result)
        return result

    def _predict_node(
        self,
        node_index: int,
        indexes: np.ndarray,
        matrix: np.ndarray,
        result: np.ndarray,
    ) -> None:
        if not len(indexes):
            return
        node = self.nodes[node_index]
        if node.feature < 0:
            result[indexes] = node.value
            return
        mask = matrix[indexes, node.feature] <= node.split_bin
        self._predict_node(node.left, indexes[mask], matrix, result)
        self._predict_node(node.right, indexes[~mask], matrix, result)

    def to_dict(self) -> dict[str, Any]:
        return {"nodes": [node.to_dict() for node in self.nodes]}


class HistogramGradientBoosting:
    def __init__(
        self,
        estimators: int = 90,
        learning_rate: float = 0.07,
        maximum_depth: int = 3,
        minimum_leaf: int = 20,
        l2: float = 2.0,
        feature_fraction: float = 0.6,
        random_seed: int = 20260719,
    ) -> None:
        self.estimators = estimators
        self.learning_rate = learning_rate
        self.maximum_depth = maximum_depth
        self.minimum_leaf = minimum_leaf
        self.l2 = l2
        self.feature_fraction = feature_fraction
        self.random_seed = random_seed
        self.trees: list[NewtonHistogramTree] = []
        self.base_logit = 0.0

    def fit(
        self,
        matrix: np.ndarray,
        target: np.ndarray,
        sample_weight: np.ndarray | None = None,
    ) -> "HistogramGradientBoosting":
        target = target.astype(float)
        sample_weight = (
            np.ones(len(target), dtype=float)
            if sample_weight is None
            else sample_weight.astype(float)
        )
        weighted_rate = float(np.sum(sample_weight * target) / sample_weight.sum())
        weighted_rate = float(np.clip(weighted_rate, 1e-6, 1 - 1e-6))
        self.base_logit = float(np.log(weighted_rate / (1 - weighted_rate)))
        logits = np.full(len(target), self.base_logit, dtype=float)
        self.trees = []
        for number in range(self.estimators):
            probability = sigmoid(logits)
            gradient = sample_weight * (target - probability)
            hessian = sample_weight * probability * (1 - probability)
            tree = NewtonHistogramTree(
                maximum_depth=self.maximum_depth,
                minimum_leaf=self.minimum_leaf,
                l2=self.l2,
                feature_fraction=self.feature_fraction,
                random_seed=self.random_seed + number,
            ).fit(matrix, gradient, hessian)
            logits += self.learning_rate * tree.predict(matrix)
            self.trees.append(tree)
        return self

    def predict_proba(self, matrix: np.ndarray) -> np.ndarray:
        logits = np.full(len(matrix), self.base_logit, dtype=float)
        for tree in self.trees:
            logits += self.learning_rate * tree.predict(matrix)
        return sigmoid(logits)

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "histogram_gradient_boosting",
            "base_logit": self.base_logit,
            "learning_rate": self.learning_rate,
            "estimators": self.estimators,
            "maximum_depth": self.maximum_depth,
            "minimum_leaf": self.minimum_leaf,
            "l2": self.l2,
            "feature_fraction": self.feature_fraction,
            "random_seed": self.random_seed,
            "trees": [tree.to_dict() for tree in self.trees],
        }


@dataclass
class TwoHeadBoosting:
    resolution: HistogramGradientBoosting
    post_install: HistogramGradientBoosting

    @classmethod
    def fit(
        cls,
        matrix: np.ndarray,
        outcomes: np.ndarray,
        **model_options: Any,
    ) -> "TwoHeadBoosting":
        resolution_target = (outcomes == "resolution_failure").astype(int)
        post_mask = outcomes != "resolution_failure"
        post_target = (outcomes[post_mask] != "pass").astype(int)
        resolution = HistogramGradientBoosting(**model_options).fit(
            matrix,
            resolution_target,
            balanced_binary_weights(resolution_target),
        )
        post_options = dict(model_options)
        post_options["random_seed"] = int(post_options.get("random_seed", 20260719)) + 10000
        post_install = HistogramGradientBoosting(**post_options).fit(
            matrix[post_mask],
            post_target,
            balanced_binary_weights(post_target),
        )
        return cls(resolution, post_install)

    def predict_components(self, matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return self.resolution.predict_proba(matrix), self.post_install.predict_proba(matrix)

    def predict_proba(self, matrix: np.ndarray) -> np.ndarray:
        resolution, post_install = self.predict_components(matrix)
        return 1.0 - (1.0 - resolution) * (1.0 - post_install)

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "two_head_histogram_gradient_boosting",
            "resolution_head": self.resolution.to_dict(),
            "post_install_head": self.post_install.to_dict(),
            "combination": "1 - (1 - p_resolution) * (1 - p_post_install)",
        }
