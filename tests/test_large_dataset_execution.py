import json
import tempfile
import unittest
from pathlib import Path

from deplab.batch import load_manifest
from deplab.dataset_execution import (
    DatasetExecutionError,
    audit_result_file,
    seed_result_file,
)
from deplab.models import ExperimentSpec, PackagePin
from scripts.prepare_large_dataset_execution import shard_set_sha256


ROOT = Path(__file__).resolve().parents[1]


def manifest_row(version: str, python_version: str = "3.11") -> tuple[dict, ExperimentSpec]:
    spec = ExperimentSpec(
        PackagePin("alpha", version),
        PackagePin("beta", "2.0"),
        python_version,
    )
    return (
        {
            "family": "alpha-beta",
            "package_a": f"alpha=={version}",
            "package_b": "beta==2.0",
            "python": python_version,
            "platform": "linux_x86_64",
        },
        spec,
    )


def result_row(spec: ExperimentSpec, outcome: str = "pass") -> dict:
    return {
        "schema_version": "1.3.0",
        "experiment_id": spec.experiment_id,
        "spec": {
            "package_a": {
                "name": spec.package_a.name,
                "version": spec.package_a.version,
            },
            "package_b": {
                "name": spec.package_b.name,
                "version": spec.package_b.version,
            },
            "python_version": spec.python_version,
            "os": spec.os,
            "architecture": spec.architecture,
        },
        "outcome": outcome,
        "measured": True,
    }


class LargeDatasetExecutionTests(unittest.TestCase):
    def test_frozen_shards_have_no_loss_or_duplicates(self) -> None:
        freeze = json.loads(
            (ROOT / "configs/large-execution-freeze-v3.0.0.json").read_text(
                encoding="utf-8"
            )
        )
        for split, details in freeze["splits"].items():
            matrix = ROOT / details["manifest"]
            shard_directory = ROOT / details["output_dir"]
            paths = sorted(shard_directory.glob("shard-*.json"))
            matrix_ids = [spec.experiment_id for spec in load_manifest(matrix)]
            shard_ids = [
                spec.experiment_id
                for path in paths
                for spec in load_manifest(path)
            ]
            self.assertEqual(len(paths), details["shards"], split)
            self.assertEqual(shard_ids, matrix_ids, split)
            self.assertEqual(len(set(shard_ids)), len(matrix_ids), split)
            self.assertEqual(
                shard_set_sha256(shard_directory),
                details["shard_set_sha256"],
                split,
            )

    def test_runtime_pilot_covers_every_python_version_once(self) -> None:
        pilot = json.loads(
            (ROOT / "configs/large-runtime-pilot-v3.0.0.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(
            [row["python"] for row in pilot["experiments"]],
            ["3.8", "3.9", "3.10", "3.11", "3.12", "3.13", "3.14"],
        )
        self.assertTrue(
            all(row["family"] == "requests-urllib3" for row in pilot["experiments"])
        )

    def test_seed_is_deterministic_and_audit_is_complete(self) -> None:
        first_manifest, first_spec = manifest_row("1.0")
        second_manifest, second_spec = manifest_row("1.1")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            matrix = root / "matrix.json"
            source_a = root / "source-a.jsonl"
            source_b = root / "source-b.jsonl"
            output = root / "results.jsonl"
            matrix.write_text(
                json.dumps({"experiments": [first_manifest, second_manifest]}),
                encoding="utf-8",
            )
            source_a.write_text(
                json.dumps(result_row(second_spec)) + "\n",
                encoding="utf-8",
            )
            source_b.write_text(
                json.dumps(result_row(first_spec, "import_failure")) + "\n",
                encoding="utf-8",
            )
            summary = seed_result_file(matrix, [source_a, source_b], output)
            repeated = seed_result_file(matrix, [source_a, source_b], output)
            audit = audit_result_file(matrix, output)
            rows = [
                json.loads(line)
                for line in output.read_text(encoding="utf-8").splitlines()
            ]
        self.assertEqual(summary.output_rows, 2)
        self.assertEqual(summary.remaining, 0)
        self.assertEqual(repeated.source_rows, 2)
        self.assertEqual(repeated.unique_seed_rows, 2)
        self.assertEqual(repeated.output_rows, 2)
        self.assertEqual(
            [row["experiment_id"] for row in rows],
            [first_spec.experiment_id, second_spec.experiment_id],
        )
        self.assertTrue(audit["structural_valid"])
        self.assertTrue(audit["complete"])
        self.assertEqual(audit["outcome_counts"], {"import_failure": 1, "pass": 1})

    def test_seed_rejects_infrastructure_failure(self) -> None:
        manifest, spec = manifest_row("1.0")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            matrix = root / "matrix.json"
            source = root / "source.jsonl"
            matrix.write_text(
                json.dumps({"experiments": [manifest]}),
                encoding="utf-8",
            )
            source.write_text(
                json.dumps(result_row(spec, "infrastructure_failure")) + "\n",
                encoding="utf-8",
            )
            with self.assertRaises(DatasetExecutionError):
                seed_result_file(matrix, [source], root / "output.jsonl")

    def test_progress_audit_detects_duplicates_and_infrastructure(self) -> None:
        manifest, spec = manifest_row("1.0")
        row = result_row(spec, "infrastructure_failure")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            matrix = root / "matrix.json"
            results = root / "results.jsonl"
            matrix.write_text(
                json.dumps({"experiments": [manifest]}),
                encoding="utf-8",
            )
            results.write_text(
                json.dumps(row) + "\n" + json.dumps(row) + "\n",
                encoding="utf-8",
            )
            audit = audit_result_file(matrix, results)
        self.assertFalse(audit["structural_valid"])
        self.assertFalse(audit["complete"])
        self.assertEqual(audit["duplicate_count"], 1)
        self.assertEqual(audit["infrastructure_failure_count"], 2)


if __name__ == "__main__":
    unittest.main()
