import json
import tempfile
import unittest
from pathlib import Path

from deplab.result_repair import repair_result_file


class ResultRepairTests(unittest.TestCase):
    def test_removes_all_duplicate_and_infrastructure_only_ids_for_retry(self) -> None:
        rows = [
            {"experiment_id": "good", "outcome": "pass"},
            {"experiment_id": "duplicate", "outcome": "pass"},
            {"experiment_id": "duplicate", "outcome": "infrastructure_failure"},
            {"experiment_id": "retry", "outcome": "infrastructure_failure"},
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "results.jsonl"
            path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
            summary = repair_result_file(path)
            repaired = [json.loads(line) for line in path.read_text().splitlines()]
            backup_rows = [
                json.loads(line)
                for line in Path(summary.backup).read_text().splitlines()
            ]
        self.assertEqual(summary.input_rows, 4)
        self.assertEqual(summary.duplicate_ids_removed_for_retry, 1)
        self.assertEqual(summary.infrastructure_only_ids_removed_for_retry, 1)
        self.assertEqual(repaired, [{"experiment_id": "good", "outcome": "pass"}])
        self.assertEqual(backup_rows, rows)


if __name__ == "__main__":
    unittest.main()
