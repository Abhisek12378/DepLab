import json
import tempfile
import unittest
from pathlib import Path

from deplab.batch import ManifestError, load_manifest, run_batch
from deplab.models import ExperimentResult, PackageRelease, utc_now


class StubClient:
    def release(self, package, version, python_version):
        return PackageRelease(
            name=package,
            version=version,
            requires_python=None,
            requires_dist=[],
            extras=[],
            classifiers=[],
            project_urls={},
            release_date=None,
            yanked=False,
            wheels=[],
            source="test_fixture",
        )


class StubRunner:
    def __init__(self):
        self.seen = []

    def run(self, spec, release_a, release_b):
        self.seen.append(spec.experiment_id)
        return ExperimentResult(
            schema_version="1.1.0",
            experiment_id=spec.experiment_id,
            spec=spec,
            outcome="pass",
            started_at=utc_now(),
            duration_seconds=0.01,
            measured=True,
        )


def write_manifest(path: Path, experiments) -> None:
    path.write_text(json.dumps({"experiments": experiments}), encoding="utf-8")


class BatchTests(unittest.TestCase):
    def test_loads_supported_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            write_manifest(
                path,
                [
                    {
                        "package_a": "alpha==1.0",
                        "package_b": "beta==2.0",
                        "python": "3.11",
                        "platform": "linux_x86_64",
                    }
                ],
            )
            specs = load_manifest(path)
        self.assertEqual(specs[0].package_a.requirement, "alpha==1.0")
        self.assertEqual(specs[0].python_version, "3.11")

    def test_loads_full_large_dataset_python_range(self) -> None:
        experiments = [
            {
                "package_a": "alpha==1.0",
                "package_b": "beta==2.0",
                "python": version,
                "platform": "linux_x86_64",
            }
            for version in ("3.8", "3.9", "3.10", "3.11", "3.12", "3.13", "3.14")
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            write_manifest(path, experiments)
            specs = load_manifest(path)
        self.assertEqual(
            [spec.python_version for spec in specs],
            ["3.8", "3.9", "3.10", "3.11", "3.12", "3.13", "3.14"],
        )

    def test_rejects_duplicate_experiment(self) -> None:
        row = {
            "package_a": "alpha==1.0",
            "package_b": "beta==2.0",
            "python": "3.11",
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            write_manifest(path, [row, row])
            with self.assertRaises(ManifestError):
                load_manifest(path)

    def test_batch_appends_and_resumes(self) -> None:
        rows = [
            {"package_a": "alpha==1.0", "package_b": "beta==2.0", "python": "3.10"},
            {"package_a": "alpha==1.0", "package_b": "beta==2.0", "python": "3.11"},
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "manifest.json"
            output = root / "results.jsonl"
            write_manifest(manifest, rows)
            runner = StubRunner()
            first = run_batch(manifest, output, runner, client=StubClient(), workers=2)
            second = run_batch(manifest, output, runner, client=StubClient(), workers=2)
            records = output.read_text(encoding="utf-8").splitlines()
        self.assertEqual(first.scheduled, 2)
        self.assertEqual(first.outcome_counts, {"pass": 2})
        self.assertEqual(second.scheduled, 0)
        self.assertEqual(second.skipped_completed, 2)
        self.assertEqual(len(records), 2)


if __name__ == "__main__":
    unittest.main()
