import json
import tempfile
import unittest
from pathlib import Path

from deplab.batch import load_manifest
from deplab.matrix import generate_matrix


ROOT = Path(__file__).resolve().parents[1]


class ExternalTestManifestTests(unittest.TestCase):
    def test_external_versions_are_disjoint_and_generate_61_experiments(self):
        training = json.loads((ROOT / "configs/package-scope.json").read_text())
        external = json.loads((ROOT / "configs/external-test-scope.json").read_text())
        for package, external_package in external["packages"].items():
            training_versions = {
                row["version"] for row in training["packages"][package]["versions"]
            }
            external_versions = {row["version"] for row in external_package["versions"]}
            self.assertEqual(len(external_versions), 2)
            self.assertTrue(training_versions.isdisjoint(external_versions))

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "matrix.json"
            summary = generate_matrix(
                ROOT / "configs/external-test-scope.json",
                ROOT / "configs/pair-families.json",
                output,
            )
            self.assertEqual(summary.experiments, 61)
            specs = load_manifest(output)
            self.assertEqual(len(specs), 61)
            self.assertEqual(len({spec.experiment_id for spec in specs}), 61)


if __name__ == "__main__":
    unittest.main()
