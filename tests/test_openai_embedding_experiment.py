from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.embed_release_openai import (
    _validate_existing,
    _validate_resume_source,
    normalized_vector,
    prepare_inputs,
)
from scripts.train_large_hybrid import load_embeddings


class FakeEncoding:
    def encode(self, text: str) -> list[int]:
        return list(range(len(text)))


class OpenAIEmbeddingExperimentTests(unittest.TestCase):
    def test_prepares_resumable_truncated_public_release_inputs(self) -> None:
        rows = [
            {
                "package": "Alpha",
                "version": "1.0",
                "selected_text": "123456",
                "selected_text_sha256": "abc",
            },
            {
                "package": "Beta",
                "version": "2.0",
                "selected_text": "xyz",
                "selected_text_sha256": "def",
            },
        ]
        prepared = prepare_inputs(
            rows,
            {("alpha", "1.0")},
            FakeEncoding(),
            maximum_tokens=2,
        )
        self.assertEqual(len(prepared), 1)
        self.assertEqual(prepared[0]["token_ids"], [0, 1])
        self.assertTrue(prepared[0]["source_truncated"])

    def test_normalizes_vectors_and_rejects_zero_vector(self) -> None:
        vector = normalized_vector([3.0, 4.0])
        np.testing.assert_allclose(vector, [0.6, 0.8])
        with self.assertRaisesRegex(ValueError, "non-zero"):
            normalized_vector([0.0, 0.0])

    def test_existing_resume_must_use_same_model_and_dimensions(self) -> None:
        args = argparse.Namespace(model="text-embedding-3-large", dimensions=3)
        rows = [
            {
                "package": "Alpha",
                "version": "1.0",
                "model": args.model,
                "dimensions": 3,
                "embedding": [1.0, 0.0, 0.0],
            }
        ]
        self.assertEqual(_validate_existing(rows, args), {("alpha", "1.0")})
        rows[0]["dimensions"] = 2
        with self.assertRaisesRegex(ValueError, "dimensions"):
            _validate_existing(rows, args)

    def test_large_hybrid_loader_accepts_one_consistent_dimension(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "embeddings.jsonl"
            rows = [
                {
                    "package": "alpha",
                    "version": "1",
                    "embedding": [1.0, 0.0, 0.0],
                },
                {
                    "package": "beta",
                    "version": "1",
                    "embedding": [0.0, 1.0, 0.0],
                },
            ]
            path.write_text(
                "\n".join(json.dumps(row) for row in rows) + "\n",
                encoding="utf-8",
            )
            loaded = load_embeddings(path)
            self.assertEqual({len(value) for value in loaded.values()}, {3})

    def test_resume_rejects_changed_release_text_before_api_use(self) -> None:
        source = [
            {
                "package": "alpha",
                "version": "1",
                "selected_text_sha256": "new",
            }
        ]
        existing = [
            {
                "package": "alpha",
                "version": "1",
                "selected_text_sha256": "old",
            }
        ]
        with self.assertRaisesRegex(ValueError, "source text changed"):
            _validate_resume_source(source, existing)


if __name__ == "__main__":
    unittest.main()
