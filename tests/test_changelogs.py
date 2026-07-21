import json
import tempfile
import unittest
from pathlib import Path

from deplab.changelogs import (
    ChangelogClient,
    _source_urls,
    collect_changelogs,
    extract_release_changelog,
)


HISTORY = """Version 3.0.3
-------------

- Fix documentation.

Version 3.0.0
-------------

- Remove previously deprecated API functions required by demo-client.
- This is a breaking compatibility change.

Version 2.3.0
-------------

- This older line must not enter the 3.0 release-series signals.
"""


class ChangelogTests(unittest.TestCase):
    def test_history_selection_includes_release_series_but_not_older_series(self) -> None:
        client = ChangelogClient(lambda url: HISTORY)
        record = extract_release_changelog(
            "demo",
            "3.0.3",
            {"kind": "history", "url_template": "https://example.test/{version}.rst"},
            ["demo", "demo-client"],
            client,
        )
        signals = record["signals"]
        self.assertTrue(record["version_section_found"])
        self.assertTrue(signals["breaking_flag"])
        self.assertTrue(signals["removal_flag"])
        self.assertTrue(signals["removed_deprecated_flag"])
        self.assertTrue(signals["api_removal_flag"])
        self.assertFalse(signals["abi_break_flag"])
        self.assertEqual(signals["package_mentions"], {"demo-client": 1})
        self.assertNotIn("older line", " ".join(signals["evidence_lines"]).lower())

    def test_release_note_urls_include_series_and_major_context(self) -> None:
        urls = _source_urls(
            "2.1.3",
            {
                "kind": "release_note",
                "tag_template": "v{version}",
                "url_template": "https://example.test/{tag}/{note_version}.rst",
                "include_series_baseline": True,
                "include_major_baseline_from": 2,
            },
        )
        self.assertEqual(
            urls,
            [
                ("exact", "https://example.test/v2.1.3/2.1.3.rst"),
                ("series_baseline", "https://example.test/v2.1.3/2.1.0.rst"),
                ("major_baseline", "https://example.test/v2.1.3/2.0.0.rst"),
            ],
        )

    def test_historical_release_note_path_is_selected_by_version(self) -> None:
        urls = _source_urls(
            "1.10.1",
            {
                "kind": "release_note",
                "url_template": "https://example.test/new/{note_version}.rst",
                "historical_before": "1.11.0",
                "historical_url_template": "https://example.test/old/{note_version}.rst",
            },
        )
        self.assertEqual(urls, [("exact", "https://example.test/old/1.10.1.rst")])

    def test_markdown_version_headings_are_selected(self) -> None:
        history = """# Changelog

## 2.1.1 (2025-01-02)

- Fix a dependency compatibility issue.

## 2.1.0 (2025-01-01)

- Remove a deprecated method.

## 2.0.0 (2024-01-01)

- This older series is excluded.
"""
        record = extract_release_changelog(
            "demo",
            "2.1.1",
            {"kind": "history", "url_template": "https://example.test/{version}.md"},
            ["demo"],
            ChangelogClient(lambda url: history),
        )
        self.assertTrue(record["version_section_found"])
        self.assertTrue(record["signals"]["dependency_flag"])
        self.assertTrue(record["signals"]["api_removal_flag"])

    def test_calendar_release_tag_can_zero_pad_minor_version(self) -> None:
        urls = _source_urls(
            "2025.1.2",
            {
                "kind": "history",
                "tag_template": "v{padded_version}",
                "url_template": "https://example.test/{tag}/changes.rst",
            },
        )
        self.assertEqual(
            urls,
            [("exact", "https://example.test/v2025.01.2/changes.rst")],
        )

    def test_collection_is_release_level_and_resumable(self) -> None:
        calls = []

        def fetch(url):
            calls.append(url)
            return HISTORY

        scope = {"packages": {"demo": {"versions": [{"version": "3.0.3"}]}}}
        sources = {
            "packages": {
                "demo": {
                    "kind": "history",
                    "url_template": "https://example.test/{version}.rst",
                }
            }
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scope_path = root / "scope.json"
            sources_path = root / "sources.json"
            output = root / "changelogs.jsonl"
            scope_path.write_text(json.dumps(scope), encoding="utf-8")
            sources_path.write_text(json.dumps(sources), encoding="utf-8")
            client = ChangelogClient(fetch)
            first = collect_changelogs(scope_path, sources_path, output, client)
            second = collect_changelogs(scope_path, sources_path, output, client)
            rows = output.read_text(encoding="utf-8").splitlines()
        self.assertEqual(first.collected, 1)
        self.assertEqual(second.collected, 0)
        self.assertEqual(len(calls), 1)
        self.assertEqual(len(rows), 1)


if __name__ == "__main__":
    unittest.main()
