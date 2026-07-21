import json
import unittest
from collections import Counter
from pathlib import Path

from deplab.batch import load_manifest
from deplab.smoke import PAIR_TESTS, build_smoke_script


ROOT = Path(__file__).resolve().parents[1]


class PilotManifestTests(unittest.TestCase):
    def test_scope_has_ten_packages_and_seven_versions_each(self) -> None:
        scope = json.loads((ROOT / "configs/package-scope.json").read_text(encoding="utf-8"))
        self.assertEqual(len(scope["packages"]), 10)
        self.assertTrue(
            all(len(package["versions"]) == 7 for package in scope["packages"].values())
        )

    def test_pilot_has_50_unique_wheel_eligible_experiments(self) -> None:
        path = ROOT / "configs/pilot-50.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        specs = load_manifest(path)
        scope = json.loads((ROOT / "configs/package-scope.json").read_text(encoding="utf-8"))
        coverage_order = scope["coverage_order"]
        package_scope = scope["packages"]
        selected_packages = set()

        self.assertEqual(len(specs), 50)
        self.assertEqual(len({spec.experiment_id for spec in specs}), 50)
        for spec in specs:
            for pin in (spec.package_a, spec.package_b):
                selected_packages.add(pin.name)
                entries = {item["version"]: item for item in package_scope[pin.name]["versions"]}
                self.assertIn(pin.version, entries)
                python_index = coverage_order.index(spec.python_version)
                self.assertTrue(entries[pin.version]["coverage"][python_index])
            self.assertIn(
                frozenset((spec.package_a.name, spec.package_b.name)),
                PAIR_TESTS,
            )
        self.assertEqual(len(selected_packages), 10)
        self.assertEqual(
            Counter(row["family"] for row in payload["experiments"]),
            {
                "numpy-pandas": 12,
                "numpy-scipy": 10,
                "numpy-scikit-learn": 10,
                "requests-urllib3": 6,
                "flask-werkzeug": 6,
                "jinja2-markupsafe": 6,
            },
        )

    def test_new_web_pair_smokes_are_interoperability_tests(self) -> None:
        self.assertIn('"strength": "interoperability"', build_smoke_script("flask", "werkzeug"))
        self.assertIn(
            '"strength": "interoperability"', build_smoke_script("jinja2", "markupsafe")
        )


if __name__ == "__main__":
    unittest.main()

