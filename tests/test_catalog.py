import json
import tempfile
import unittest
from pathlib import Path

from deplab.catalog import collect_catalog
from deplab.pypi import PyPIClient


PAYLOAD = {
    "info": {
        "name": "demo",
        "version": "1.0.0",
        "requires_python": ">=3.10",
        "requires_dist": [],
        "provides_extra": [],
        "classifiers": [],
        "project_urls": {},
        "yanked": False,
    },
    "urls": [
        {
            "packagetype": "bdist_wheel",
            "filename": "demo-1.0.0-py3-none-any.whl",
            "url": "https://files.example.test/demo.whl",
            "size": 10,
            "digests": {"sha256": "abc"},
            "upload_time_iso_8601": "2024-01-01T00:00:00Z",
            "yanked": False,
        }
    ],
}


class CatalogTests(unittest.TestCase):
    def test_collects_three_targets_and_resumes_with_one_metadata_fetch(self) -> None:
        calls = []

        def fetch(url):
            calls.append(url)
            return PAYLOAD

        scope = {
            "coverage_order": ["3.10", "3.11", "3.12"],
            "packages": {"demo": {"versions": [{"version": "1.0.0"}]}},
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scope_path = root / "scope.json"
            output = root / "catalog.jsonl"
            scope_path.write_text(json.dumps(scope), encoding="utf-8")
            client = PyPIClient(fetch)
            first = collect_catalog(scope_path, output, client)
            second = collect_catalog(scope_path, output, client)
            rows = output.read_text(encoding="utf-8").splitlines()
        self.assertEqual(first.collected, 3)
        self.assertEqual(second.collected, 0)
        self.assertEqual(second.skipped_existing, 3)
        self.assertEqual(len(rows), 3)
        self.assertEqual(len(calls), 1)

    def test_collects_python_38_through_314_from_string_version_entry(self) -> None:
        calls = []

        def fetch(url):
            calls.append(url)
            return PAYLOAD

        scope = {
            "coverage_order": [
                "3.8",
                "3.9",
                "3.10",
                "3.11",
                "3.12",
                "3.13",
                "3.14",
            ],
            "packages": {"demo": {"versions": ["1.0.0"]}},
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scope_path = root / "scope.json"
            output = root / "catalog.jsonl"
            scope_path.write_text(json.dumps(scope), encoding="utf-8")
            summary = collect_catalog(scope_path, output, PyPIClient(fetch))
            rows = [
                json.loads(line)
                for line in output.read_text(encoding="utf-8").splitlines()
            ]
        self.assertEqual(summary.requested, 7)
        self.assertEqual(summary.collected, 7)
        self.assertEqual(
            [row["target"]["python_version"] for row in rows],
            scope["coverage_order"],
        )
        self.assertEqual(len(calls), 1)


if __name__ == "__main__":
    unittest.main()
