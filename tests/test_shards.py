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

    def test_uses_consistent_three_digit_names_for_large_shard_sets(self) -> None:
        experiments = [
            {
                "package_a": f"alpha==1.0.{index}",
                "package_b": "beta==2.0",
                "python": "3.11",
                "platform": "linux_x86_64",
            }
            for index in range(101)
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "manifest.json"
            output = root / "shards"
            manifest.write_text(
                json.dumps({"experiments": experiments}),
                encoding="utf-8",
            )
            summary = shard_manifest(manifest, output, shard_size=1)
            names = sorted(path.name for path in output.glob("shard-*.json"))
        self.assertEqual(summary.shards, 101)
        self.assertEqual(summary.filename_width, 3)
        self.assertEqual(names[0], "shard-001-of-101.json")
        self.assertEqual(names[-1], "shard-101-of-101.json")


if __name__ == "__main__":
    unittest.main()
