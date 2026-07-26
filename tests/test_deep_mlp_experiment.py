from __future__ import annotations

import unittest

import numpy as np

from scripts.train_deep_mlp_experiment import (
    OUTCOME_TO_CLASS,
    class_weights,
    classification_metrics,
    encode_outcomes,
)


class DeepMLPExperimentTests(unittest.TestCase):
    def test_rare_failure_classes_receive_larger_weights(self) -> None:
        labels = np.asarray([0] * 12 + [1] * 6 + [2] * 3 + [3])
        weights = class_weights(labels)
        self.assertLess(weights[0], weights[1])
        self.assertLess(weights[1], weights[2])
        self.assertLess(weights[2], weights[3])
        self.assertAlmostEqual(float(weights.mean()), 1.0)

    def test_metrics_distinguish_failure_detection_from_exact_stage(self) -> None:
        outcomes = np.asarray(
            ["pass", "resolution_failure", "import_failure", "smoke_test_failure"]
        )
        labels = encode_outcomes(outcomes)
        probabilities = np.asarray(
            [
                [0.8, 0.1, 0.05, 0.05],
                [0.1, 0.7, 0.1, 0.1],
                [0.1, 0.6, 0.2, 0.1],
                [0.2, 0.1, 0.1, 0.6],
            ]
        )
        metrics = classification_metrics(labels, probabilities)
        self.assertEqual(metrics["import_failure_detected"], 1)
        self.assertEqual(metrics["import_failure_correct_stage"], 0)
        self.assertEqual(metrics["smoke_test_failure_correct_stage"], 1)
        self.assertEqual(metrics["false_failure"], 0)

    def test_unknown_outcome_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown outcomes"):
            encode_outcomes(np.asarray(["pass", "infrastructure_failure"]))

    def test_outcome_ids_are_stable(self) -> None:
        self.assertEqual(OUTCOME_TO_CLASS["pass"], 0)
        self.assertEqual(OUTCOME_TO_CLASS["import_failure"], 2)


if __name__ == "__main__":
    unittest.main()
