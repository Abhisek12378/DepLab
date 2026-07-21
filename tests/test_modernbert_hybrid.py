from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from scripts.train_modernbert_hybrid import (
    ReleaseProjector,
    frame_release_keys,
    release_key,
    text_interaction_matrix,
)


class ModernBERTHybridTests(unittest.TestCase):
    def test_release_projection_and_pair_interactions_are_deterministic(self) -> None:
        frame = pd.DataFrame(
            {
                "package_a_name": ["Alpha", "Alpha"],
                "package_a_version": ["1.0", "2.0"],
                "package_b_name": ["Beta", "Beta"],
                "package_b_version": ["1.0", "1.0"],
            }
        )
        embeddings = {
            release_key("alpha", "1.0"): np.asarray([1.0, 0.0, 0.0]),
            release_key("alpha", "2.0"): np.asarray([0.0, 1.0, 0.0]),
            release_key("beta", "1.0"): np.asarray([0.0, 0.0, 1.0]),
        }
        keys = frame_release_keys(frame)
        projector = ReleaseProjector.fit(embeddings, keys, dimensions=2)
        first = text_interaction_matrix(frame, embeddings, projector)
        second = text_interaction_matrix(frame, embeddings, projector)
        self.assertEqual(first.shape, (2, 8))
        np.testing.assert_allclose(first, second)

    def test_missing_training_release_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "missing release embeddings"):
            ReleaseProjector.fit(
                {release_key("alpha", "1"): np.ones(3)},
                {release_key("alpha", "1"), release_key("beta", "1")},
                dimensions=2,
            )


if __name__ == "__main__":
    unittest.main()
