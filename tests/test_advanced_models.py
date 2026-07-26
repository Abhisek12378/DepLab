from __future__ import annotations

import unittest

import numpy as np

from scripts.advanced_models import (
    HistogramBinner,
    HistogramGradientBoosting,
    StageAwareBoosting,
    TwoHeadBoosting,
    balanced_binary_weights,
)


class AdvancedModelTests(unittest.TestCase):
    def test_histogram_boosting_learns_nonlinear_interaction(self) -> None:
        matrix = np.array(
            [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]] * 40,
            dtype=float,
        )
        target = np.array([0, 0, 0, 1] * 40)
        binner = HistogramBinner.fit(matrix, maximum_bins=4)
        bins = binner.transform(matrix)
        model = HistogramGradientBoosting(
            estimators=50,
            learning_rate=0.12,
            maximum_depth=2,
            minimum_leaf=10,
            feature_fraction=1.0,
            random_seed=3,
        ).fit(bins, target, balanced_binary_weights(target))
        predicted = model.predict_proba(bins) >= 0.5
        self.assertGreater(float((predicted == target).mean()), 0.99)

    def test_two_head_model_returns_valid_probabilities(self) -> None:
        rng = np.random.default_rng(4)
        matrix = rng.normal(size=(180, 4))
        outcomes = np.asarray(
            ["resolution_failure"] * 60 + ["import_failure"] * 60 + ["pass"] * 60
        )
        binner = HistogramBinner.fit(matrix, maximum_bins=8)
        bins = binner.transform(matrix)
        model = TwoHeadBoosting.fit(
            bins,
            outcomes,
            estimators=5,
            maximum_depth=2,
            minimum_leaf=10,
            feature_fraction=1.0,
            random_seed=5,
        )
        probability = model.predict_proba(bins)
        self.assertEqual(probability.shape, (180,))
        self.assertTrue(np.all((probability >= 0) & (probability <= 1)))

    def test_stage_aware_model_scores_each_failure_stage(self) -> None:
        rng = np.random.default_rng(8)
        matrix = rng.normal(size=(240, 5))
        outcomes = np.asarray(
            ["resolution_failure"] * 60
            + ["import_failure"] * 60
            + ["smoke_test_failure"] * 60
            + ["pass"] * 60
        )
        binner = HistogramBinner.fit(matrix, maximum_bins=8)
        bins = binner.transform(matrix)
        model = StageAwareBoosting.fit(
            bins,
            outcomes,
            estimators=4,
            maximum_depth=2,
            minimum_leaf=10,
            feature_fraction=1.0,
            random_seed=9,
        )
        components = model.predict_components(bins)
        self.assertEqual(len(components), 3)
        for values in components:
            self.assertEqual(values.shape, (240,))
            self.assertTrue(np.all((values >= 0) & (values <= 1)))
        probability = model.predict_proba(bins)
        self.assertTrue(np.all((probability >= 0) & (probability <= 1)))


if __name__ == "__main__":
    unittest.main()
