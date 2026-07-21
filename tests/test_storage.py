import tempfile
import unittest
from pathlib import Path

from deplab.storage import append_jsonl, completed_ids


class StorageTests(unittest.TestCase):
    def test_completed_ids_enable_resume(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "results.jsonl"
            append_jsonl(path, {"experiment_id": "one", "outcome": "pass"})
            append_jsonl(path, {"experiment_id": "two", "outcome": "import_failure"})
            append_jsonl(path, {"experiment_id": "retry-me", "outcome": "infrastructure_failure"})
            self.assertEqual(completed_ids(path), {"one", "two"})


if __name__ == "__main__":
    unittest.main()
