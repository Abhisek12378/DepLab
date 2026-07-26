import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from deplab.batch import load_manifest
from deplab.matrix import generate_matrix


ROOT = Path(__file__).resolve().parents[1]


class MatrixTests(unittest.TestCase):
    def test_generates_complete_deterministic_wheel_eligible_matrix(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "first.json"
            second = Path(directory) / "second.json"
            summary = generate_matrix(
                ROOT / "configs/package-scope.json",
                ROOT / "configs/pair-families.json",
                first,
            )
            generate_matrix(
                ROOT / "configs/package-scope.json",
                ROOT / "configs/pair-families.json",
                second,
            )
            payload = json.loads(first.read_text(encoding="utf-8"))
            specs = load_manifest(first)
            first_contents = first.read_bytes()
            second_contents = second.read_bytes()

        self.assertEqual(first_contents, second_contents)
        self.assertEqual(summary.families, 6)
        self.assertEqual(summary.experiments + summary.excluded_for_wheel_coverage, 882)
        self.assertEqual(
            len(payload["coverage_exclusions"]),
            summary.excluded_for_wheel_coverage,
        )
        self.assertTrue(
            all(
                row["selection_method"] == "deterministic_coverage_exclusion"
                for row in payload["coverage_exclusions"]
            )
        )
        self.assertEqual(len(specs), summary.experiments)
        self.assertEqual(len({spec.experiment_id for spec in specs}), summary.experiments)
        self.assertTrue(
            all(row["selection_method"] == "systematic_cartesian_wheel_eligible" for row in payload["experiments"])
        )
        self.assertEqual(
            Counter(row["family"] for row in payload["experiments"]),
            {
                "numpy-pandas": 80,
                "numpy-scipy": 75,
                "numpy-scikit-learn": 85,
                "requests-urllib3": 147,
                "flask-werkzeug": 147,
                "jinja2-markupsafe": 112,
            },
        )


if __name__ == "__main__":
    unittest.main()
