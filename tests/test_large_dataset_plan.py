import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from deplab.scope_plan import (
    build_scope_draft,
    eligible_project_versions,
    read_and_validate_plan,
    select_versions,
    write_pair_definitions,
)
from deplab.smoke_v3 import build_smoke_script


ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "configs/large-dataset-plan-v3.0.0.json"


def project_payload(package: str, versions: list[str]) -> dict:
    return {
        "releases": {
            version: [
                {
                    "filename": f"{package}-{version}-py3-none-any.whl",
                    "packagetype": "bdist_wheel",
                    "requires_python": ">=3.8",
                    "upload_time_iso_8601": f"2025-{index + 1:02d}-01T00:00:00Z",
                    "yanked": False,
                }
            ]
            for index, version in enumerate(versions)
        }
    }


class FakeProjectClient:
    def project(self, package: str) -> dict:
        return project_payload(package, ["1.0.0", "1.1.0", "2.0.0"])


class LargeDatasetPlanTests(unittest.TestCase):
    def test_plan_has_expected_scale_and_disjoint_splits(self) -> None:
        plan, summary = read_and_validate_plan(PLAN)
        self.assertEqual(summary.packages, 50)
        self.assertEqual(summary.families, 45)
        self.assertEqual(summary.python_versions, 7)
        self.assertEqual(summary.target_versions_per_package, 12)
        self.assertEqual(summary.maximum_cartesian_experiments, 45_360)
        self.assertEqual(
            summary.split_packages,
            {"development": 35, "validation": 6, "final_test": 9},
        )
        self.assertEqual(
            summary.split_families,
            {"development": 36, "validation": 4, "final_test": 5},
        )
        self.assertEqual(plan["minimum_versions_per_package"], 8)
        self.assertEqual(plan["packages"]["sniffio"]["minimum_versions"], 5)
        for name, package in plan["packages"].items():
            minimum = package.get(
                "minimum_versions",
                plan["minimum_versions_per_package"],
            )
            self.assertGreaterEqual(minimum, 5, name)
            self.assertLessEqual(minimum, summary.target_versions_per_package, name)
        for split in ("development", "validation", "final_test"):
            names = {
                name
                for name, package in plan["packages"].items()
                if package["split"] == split
            }
            for family in plan["families"]:
                if family["split"] == split:
                    self.assertIn(family["package_a"], names)
                    self.assertIn(family["package_b"], names)

    def test_every_family_has_an_interoperability_smoke_test(self) -> None:
        plan, _ = read_and_validate_plan(PLAN)
        for family in plan["families"]:
            script = build_smoke_script(family["package_a"], family["package_b"])
            self.assertIn('"strength": "interoperability"', script, family["name"])

    def test_pair_definitions_are_generated_by_split(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            summary = write_pair_definitions(PLAN, output)
            development = json.loads(
                (output / "large-development-pairs-v3.0.0.json").read_text()
            )
            validation = json.loads(
                (output / "large-validation-pairs-v3.0.0.json").read_text()
            )
            final_test = json.loads(
                (output / "large-final-test-pairs-v3.0.0.json").read_text()
            )
        self.assertEqual(summary.families, 45)
        self.assertEqual(len(development["families"]), 36)
        self.assertEqual(len(validation["families"]), 4)
        self.assertEqual(len(final_test["families"]), 5)
        self.assertEqual(development["python_versions"][0], "3.8")
        self.assertEqual(development["python_versions"][-1], "3.14")

    def test_release_selection_excludes_prerelease_yanked_and_future_files(self) -> None:
        payload = project_payload("demo", ["1.0.0", "1.1.0", "2.0.0"])
        payload["releases"]["2.0.0rc1"] = [
            {
                "filename": "demo-2.0.0rc1-py3-none-any.whl",
                "upload_time_iso_8601": "2025-01-01T00:00:00Z",
                "yanked": False,
            }
        ]
        payload["releases"]["3.0.0"] = [
            {
                "filename": "demo-3.0.0-py3-none-any.whl",
                "upload_time_iso_8601": "2027-01-01T00:00:00Z",
                "yanked": False,
            }
        ]
        payload["releases"]["4.0.0"] = [
            {
                "filename": "demo-4.0.0-py3-none-any.whl",
                "upload_time_iso_8601": "2025-01-01T00:00:00Z",
                "yanked": True,
            }
        ]
        selected = eligible_project_versions(
            payload,
            ["3.8", "3.14"],
            datetime(2026, 7, 25, tzinfo=timezone.utc),
            history_limit=60,
        )
        self.assertEqual(selected, ["1.0.0", "1.1.0", "2.0.0"])
        self.assertEqual(
            select_versions(selected, target=2),
            ["1.0.0", "2.0.0"],
        )
        self.assertEqual(
            select_versions(selected, target=2, preserved=["1.1.0"]),
            ["1.1.0", "2.0.0"],
        )

    def test_scope_draft_can_be_built_from_project_metadata(self) -> None:
        plan = {
            "schema_version": "test",
            "plan_id": "test-plan",
            "release_cutoff": "2026-07-25T00:00:00Z",
            "target_platform": "linux_x86_64",
            "coverage_order": ["3.8", "3.9", "3.10", "3.11", "3.12", "3.13", "3.14"],
            "target_versions_per_package": 2,
            "minimum_versions_per_package": 2,
            "version_selection": {"candidate_history_limit": 60},
            "packages": {
                "demo-a": {"split": "development", "role": "A", "ecosystem": "test"},
                "demo-b": {"split": "development", "role": "B", "ecosystem": "test"},
            },
            "families": [
                {
                    "name": "demo-a-demo-b",
                    "package_a": "demo-a",
                    "package_b": "demo-b",
                    "split": "development",
                }
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan_path = root / "plan.json"
            output_path = root / "scope.json"
            plan_path.write_text(json.dumps(plan), encoding="utf-8")
            summary = build_scope_draft(
                plan_path,
                output_path,
                client=FakeProjectClient(),
                root=root,
            )
            output = json.loads(output_path.read_text(encoding="utf-8"))
        self.assertEqual(summary.packages, 2)
        self.assertEqual(summary.releases, 4)
        self.assertEqual(output["packages"]["demo-a"]["versions"], ["1.0.0", "2.0.0"])


if __name__ == "__main__":
    unittest.main()
