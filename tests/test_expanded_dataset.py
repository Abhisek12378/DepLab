from __future__ import annotations

import json
import unittest
from pathlib import Path

from scripts.build_expanded_dataset import (
    BASE_NUMERIC_COLUMNS,
    CATEGORICAL_COLUMNS,
    _read_jsonl,
    _requirement_allows,
    build_features,
)
from scripts.train_expanded_baseline import assign_family_folds
from scripts.evaluate_expanded_holdout import _metrics


ROOT = Path(__file__).resolve().parents[1]


class ExpandedDatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.catalog = _read_jsonl(ROOT / "outputs/expanded-package-catalog.jsonl")
        cls.changelogs = _read_jsonl(ROOT / "outputs/changelog-catalog-expanded-v1.2.0.jsonl")
        cls.development_matrix = json.loads((ROOT / "configs/expanded-development-matrix.json").read_text(encoding="utf-8"))
        cls.holdout_matrix = json.loads((ROOT / "configs/expanded-final-holdout-matrix.json").read_text(encoding="utf-8"))
        cls.results = _read_jsonl(ROOT / "outputs/expanded-development-results.jsonl")

    def test_requirement_compatibility_is_available_before_installation(self) -> None:
        self.assertFalse(_requirement_allows("Werkzeug (<2.0,>=0.15)", "2.0.0"))
        self.assertTrue(_requirement_allows("Werkzeug (<2.0,>=0.15)", "1.0.1"))
        self.assertIsNone(_requirement_allows(None, "1.0.1"))

    def test_development_and_holdout_features_are_disjoint_and_label_safe(self) -> None:
        development = build_features(self.development_matrix, self.catalog, self.changelogs, self.results)
        holdout = build_features(self.holdout_matrix, self.catalog, self.changelogs, None)
        self.assertEqual(len(development), 3269)
        self.assertEqual(len(holdout), 840)
        self.assertEqual(int(development["is_failure"].sum()), 1044)
        self.assertFalse(set(development["experiment_id"]) & set(holdout["experiment_id"]))
        self.assertNotIn("outcome", holdout.columns)
        self.assertNotIn("is_failure", holdout.columns)

    def test_model_inputs_are_inference_safe(self) -> None:
        forbidden = ("outcome", "import", "smoke", "installed", "duration", "cache", "network", "error")
        inputs = BASE_NUMERIC_COLUMNS + CATEGORICAL_COLUMNS
        self.assertFalse([column for column in inputs if any(term in column for term in forbidden)])

    def test_family_folds_keep_each_family_in_one_fold(self) -> None:
        import pandas as pd

        families = pd.Series(["a", "a", "b", "c", "c", "d", "e"])
        folds = assign_family_folds(families, 3)
        for family in families.unique():
            self.assertEqual(len(set(folds[families == family])), 1)

    def test_final_metrics_treat_failure_as_positive_class(self) -> None:
        import numpy as np

        actual = np.array([1, 1, 0, 0])
        predicted = np.array([1, 0, 1, 0])
        probability = np.array([0.9, 0.4, 0.6, 0.1])
        metrics = _metrics(actual, predicted, probability)
        self.assertEqual(metrics["true_failure"], 1)
        self.assertEqual(metrics["missed_failure"], 1)
        self.assertEqual(metrics["false_failure"], 1)
        self.assertEqual(metrics["true_pass"], 1)
        self.assertAlmostEqual(metrics["balanced_accuracy"], 0.5)


if __name__ == "__main__":
    unittest.main()
