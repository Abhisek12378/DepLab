from __future__ import annotations

import unittest

from scripts.prepare_popular_release_scope import (
    canonical,
    eligible_versions,
    parse_datetime,
    ranked_projects,
)


def release_file(
    uploaded: str = "2025-01-01T00:00:00Z",
    yanked: bool = False,
) -> dict:
    return {
        "upload_time_iso_8601": uploaded,
        "yanked": yanked,
        "filename": "example.whl",
    }


class PopularReleaseScopeTests(unittest.TestCase):
    def test_selects_all_stable_non_yanked_releases_before_cutoff(self) -> None:
        project = {
            "releases": {
                "1.0.0": [release_file()],
                "1.1.0.post1": [release_file()],
                "2.0.0rc1": [release_file()],
                "2.0.0.dev1": [release_file()],
                "3.0.0": [release_file(yanked=True)],
                "4.0.0": [release_file("2027-01-01T00:00:00Z")],
                "not-a-version": [release_file()],
            }
        }
        selected = eligible_versions(
            project, parse_datetime("2026-07-27T00:00:00Z")
        )
        self.assertEqual(selected, ["1.0.0", "1.1.0.post1"])

    def test_ranking_is_canonicalized_and_deduplicated(self) -> None:
        payload = {
            "rows": [
                {"project": "My_Package", "download_count": 20},
                {"project": "my.package", "download_count": 10},
                {"project": "Other", "download_count": 5},
            ]
        }
        ranked = ranked_projects(payload)
        self.assertEqual([row["name"] for row in ranked], ["my-package", "other"])
        self.assertEqual([row["rank"] for row in ranked], [1, 2])

    def test_canonical_name_uses_pep_503_style(self) -> None:
        self.assertEqual(canonical("zope.interface"), "zope-interface")


if __name__ == "__main__":
    unittest.main()
