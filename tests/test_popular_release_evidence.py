from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.collect_popular_release_evidence import (
    evidence_row,
    selected_releases,
    validate_complete,
)
from scripts.seed_popular_embeddings import (
    MODEL_NAME,
    main as seed_main,
    validate_output_rows,
)


class PopularReleaseEvidenceTests(unittest.TestCase):
    def test_builds_inference_safe_release_document(self) -> None:
        payload = {
            "info": {
                "name": "Example",
                "version": "1.0.0",
                "requires_python": ">=3.9",
                "requires_dist": ["provider<2"],
                "classifiers": ["Programming Language :: Python :: 3"],
            },
            "urls": [
                {"upload_time_iso_8601": "2025-01-01T00:00:00Z"}
            ],
        }
        row = evidence_row("example", "1.0.0", payload, None)
        self.assertIn("provider<2", row["selected_text"])
        self.assertFalse(row["changelog_available"])
        self.assertEqual(len(row["selected_text_sha256"]), 64)

    def test_scope_and_complete_evidence_have_identical_keys(self) -> None:
        scope = {
            "packages": {
                "alpha": {"versions": ["1.0", "2.0"]},
                "beta": {"versions": ["1.0"]},
            }
        }
        tasks = selected_releases(scope)
        rows = [
            {
                "package": package,
                "version": version,
                "selected_text": f"{package}-{version}",
                "selected_text_sha256": __import__("hashlib")
                .sha256(f"{package}-{version}".encode())
                .hexdigest(),
            }
            for package, version in tasks
        ]
        validate_complete(tasks, rows)

    def test_seed_script_reuses_only_matching_text_hash(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            evidence = root / "evidence.jsonl"
            existing = root / "existing.jsonl"
            output = root / "output.jsonl"
            evidence.write_text(
                json.dumps(
                    {
                        "package": "alpha",
                        "version": "1",
                        "selected_text_sha256": "same",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            existing.write_text(
                json.dumps(
                    {
                        "package": "alpha",
                        "version": "1",
                        "selected_text_sha256": "same",
                        "model": MODEL_NAME,
                        "embedding": [1.0],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            import sys
            from unittest.mock import patch

            arguments = [
                "seed",
                "--evidence",
                str(evidence),
                "--existing-embeddings",
                str(existing),
                "--output",
                str(output),
            ]
            with patch.object(sys, "argv", arguments):
                self.assertEqual(seed_main(), 0)
            rows = [
                json.loads(line)
                for line in output.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(len(rows), 1)

    def test_rejects_resumed_embedding_from_different_model(self) -> None:
        evidence = {
            ("alpha", "1"): {
                "package": "alpha",
                "version": "1",
                "selected_text_sha256": "same",
            }
        }
        rows = [
            {
                "package": "alpha",
                "version": "1",
                "selected_text_sha256": "same",
                "model": "different-model",
            }
        ]
        with self.assertRaisesRegex(ValueError, "different evidence or model"):
            validate_output_rows(rows, evidence)


if __name__ == "__main__":
    unittest.main()
