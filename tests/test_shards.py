import json
import tempfile
import unittest
from pathlib import Path

from deplab.batch import load_manifest
from deplab.shards import ShardError, shard_manifest


ROOT = Path(__file__).resolve().parents[1]


class ShardTests(unittest.TestCase):
    def test_splits_matrix_without_loss_or_duplicates(self) -> None:
        source_path = ROOT / "configs/systematic-matrix.json"
        source_ids = [spec.experiment_id for spec in load_manifest(source_path)]
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory) / "shards"
            summary = shard_manifest(source_path, output_dir, shard_size=50)
            paths = sorted(output_dir.glob("shard-*.json"))
            combined_ids = [
                spec.experiment_id for path in paths for spec in load_manifest(path)
            ]
            last_payload = json.loads(paths[-1].read_text(encoding="utf-8"))

        self.assertEqual(summary.shards, 13)
        self.assertEqual(summary.first_shard_size, 50)
        self.assertEqual(summary.last_shard_size, 46)
        self.assertEqual(combined_ids, source_ids)
        self.assertEqual(len(set(combined_ids)), 646)
        self.assertEqual(last_payload["shard_index"], 13)
        self.assertEqual(last_payload["shard_count"], 13)

    def test_rejects_invalid_size(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ShardError):
                shard_manifest(
                    ROOT / "configs/systematic-matrix.json", Path(directory), shard_size=0
                )


if __name__ == "__main__":
    unittest.main()
