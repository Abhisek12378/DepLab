from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from scripts.build_large_features import (
    _release_catalog,
    _release_ranks,
    build_features,
)
from scripts.build_large_release_text import release_evidence_text
from scripts.train_large_hybrid import assign_balanced_family_folds


def catalog_row(
    name: str,
    version: str,
    python: str,
    requires_dist: list[str],
) -> dict:
    return {
        "target": {"python_version": python},
        "release": {
            "name": name,
            "version": version,
            "requires_python": ">=3.8",
            "requires_dist": requires_dist,
            "extras": [],
            "classifiers": ["Programming Language :: Python :: 3"],
            "project_urls": {},
            "release_date": "2024-01-01T00:00:00Z",
            "yanked": False,
            "wheels": [
                {
                    "filename": f"{name}-{version}-py3-none-any.whl",
                    "url": "https://example.test/wheel",
                    "size": 100,
                    "sha256": "abc",
                    "python_tag": "py3",
                    "abi_tag": "none",
                    "platform_tag": "any",
                    "has_native_extensions": False,
                    "uploaded_at": "2024-01-01T00:00:00Z",
                    "yanked": False,
                    "compatible": True,
                    "compatibility_reason": "compatible",
                }
            ],
        },
    }


class LargeFeatureTests(unittest.TestCase):
    def setUp(self) -> None:
        rows = [
            catalog_row("consumer", "1.0.0", "3.11", ["provider<2"]),
            catalog_row("provider", "2.0.0", "3.11", []),
            catalog_row("provider", "1.0.0", "3.11", []),
        ]
        self.catalog = _release_catalog(rows)
        self.ranks = _release_ranks(self.catalog)
        self.matrix = {
            "experiments": [
                {
                    "family": "consumer-provider",
                    "package_a": "consumer==1.0.0",
                    "package_b": "provider==2.0.0",
                    "python": "3.11",
                    "platform": "linux_x86_64",
                }
            ]
        }

    def test_builds_pre_install_constraint_fact_without_outcome_leakage(self) -> None:
        frame = build_features(
            self.matrix,
            self.catalog,
            self.ranks,
            changelog_rows=[],
            results=None,
        )
        self.assertEqual(len(frame), 1)
        self.assertTrue(bool(frame.loc[0, "published_constraint_blocked"]))
        self.assertFalse(bool(frame.loc[0, "package_a_changelog_available"]))
        self.assertNotIn("outcome", frame.columns)
        self.assertNotIn("is_failure", frame.columns)

    def test_measured_result_adds_label_but_never_error_text(self) -> None:
        inputs = build_features(
            self.matrix,
            self.catalog,
            self.ranks,
            changelog_rows=[],
            results=None,
        )
        experiment_id = str(inputs.loc[0, "experiment_id"])
        results = [
            {
                "experiment_id": experiment_id,
                "outcome": "resolution_failure",
                "measured": True,
                "normalized_error": "must never become a feature",
            }
        ]
        frame = build_features(
            self.matrix,
            self.catalog,
            self.ranks,
            changelog_rows=[],
            results=results,
        )
        self.assertEqual(frame.loc[0, "outcome"], "resolution_failure")
        self.assertTrue(bool(frame.loc[0, "is_failure"]))
        self.assertNotIn("normalized_error", frame.columns)

    def test_release_evidence_uses_metadata_when_changelog_is_missing(self) -> None:
        release = self.catalog[("consumer", "1.0.0", "3.11")]
        text = release_evidence_text(release, None)
        self.assertIn("Requires Python: >=3.8", text)
        self.assertIn("provider<2", text)
        self.assertIn("No version-pinned changelog text", text)


class LargeModelSplitTests(unittest.TestCase):
    def test_balanced_folds_are_deterministic_and_keep_families_together(self) -> None:
        families = pd.Series(
            ["a"] * 8 + ["b"] * 7 + ["c"] * 6 + ["d"] * 5 + ["e"] * 4
        )
        target = np.asarray(
            [1, 0] * 4
            + [1, 1, 0, 0, 0, 1, 0]
            + [0, 0, 1, 1, 0, 0]
            + [1, 0, 1, 0, 0]
            + [0, 1, 0, 1]
        )
        first, first_manifest = assign_balanced_family_folds(
            families, target, fold_count=3
        )
        second, second_manifest = assign_balanced_family_folds(
            families, target, fold_count=3
        )
        np.testing.assert_array_equal(first, second)
        self.assertEqual(first_manifest, second_manifest)
        for family in families.unique():
            self.assertEqual(len(set(first[families == family])), 1)


if __name__ == "__main__":
    unittest.main()
