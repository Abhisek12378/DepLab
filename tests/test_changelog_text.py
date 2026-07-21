from __future__ import annotations

import hashlib
import unittest

from deplab.changelogs import ChangelogClient, extract_signals
from scripts.collect_changelog_text import build_text_record


class ChangelogTextTests(unittest.TestCase):
    def test_reproduces_signal_catalog_before_storing_text(self) -> None:
        text = """1.2.0
=====
- Removed deprecated API used by dependency consumers.

1.1.0
=====
- Older change.
"""
        selected = "1.2.0\n=====\n- Removed deprecated API used by dependency consumers."
        expected_signals = extract_signals(selected, ["alpha", "beta"], "alpha")
        expected = {
            "changelog_id": "example",
            "schema_version": "1.1.0",
            "sources": [
                {
                    "url": "https://example.test/history",
                    "sha256": hashlib.sha256(text.encode()).hexdigest(),
                    "version_section_found": True,
                }
            ],
            "signals": expected_signals,
        }
        record = build_text_record(
            "alpha",
            "1.2.0",
            {"kind": "history", "url_template": "https://example.test/history"},
            expected,
            ["alpha", "beta"],
            ChangelogClient(lambda _: text),
        )
        self.assertEqual(record["selected_text"], selected)
        self.assertTrue(record["signal_reproduction_verified"])


if __name__ == "__main__":
    unittest.main()
