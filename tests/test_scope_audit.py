import json
import tempfile
import unittest
from pathlib import Path

from deplab.models import PackageRelease, WheelArtifact
from deplab.scope_audit import audit_scope


class FakeClient:
    def release(self, package, version, python_version):
        compatible = python_version != "3.12"
        return PackageRelease(
            name=package,
            version=version,
            requires_python=">=3.10",
            requires_dist=[],
            extras=[],
            classifiers=[],
            project_urls={},
            release_date="2024-01-01T00:00:00Z",
            yanked=False,
            wheels=[
                WheelArtifact(
                    filename=f"{package}-{version}-py3-none-any.whl",
                    url="https://example.test/wheel.whl",
                    size=10,
                    sha256="abc",
                    python_tag="py3",
                    abi_tag="none",
                    platform_tag="any",
                    compatible=compatible,
                )
            ],
        )


class ScopeAuditTests(unittest.TestCase):
    def test_adds_three_target_coverage_and_preserves_role(self):
        draft = {
            "coverage_order": ["3.10", "3.11", "3.12"],
            "packages": {
                "demo": {"role": "fixture", "versions": ["1.0.0", "2.0.0"]}
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "draft.json"
            output_path = root / "audited.json"
            input_path.write_text(json.dumps(draft), encoding="utf-8")
            summary = audit_scope(input_path, output_path, FakeClient())
            output = json.loads(output_path.read_text(encoding="utf-8"))
        self.assertEqual(summary.releases, 2)
        self.assertEqual(summary.eligible_release_targets, 4)
        self.assertEqual(summary.excluded_release_targets, 2)
        self.assertEqual(output["packages"]["demo"]["role"], "fixture")
        self.assertEqual(
            output["packages"]["demo"]["versions"][0]["coverage"],
            [True, True, False],
        )
        self.assertEqual(
            output["packages"]["demo"]["versions"][0]["coverage_details"][2]["reason"],
            "incompatible_wheel_tags",
        )
        self.assertEqual(summary.exclusion_counts, {"incompatible_wheel_tags": 2})

    def test_accepts_python_38_through_314(self):
        draft = {
            "coverage_order": ["3.8", "3.9", "3.10", "3.11", "3.12", "3.13", "3.14"],
            "packages": {"demo": {"role": "fixture", "versions": ["1.0.0"]}},
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "draft.json"
            output_path = root / "audited.json"
            input_path.write_text(json.dumps(draft), encoding="utf-8")
            summary = audit_scope(input_path, output_path, FakeClient())
            output = json.loads(output_path.read_text(encoding="utf-8"))
        self.assertEqual(summary.python_targets, 7)
        self.assertEqual(len(output["packages"]["demo"]["versions"][0]["coverage"]), 7)


if __name__ == "__main__":
    unittest.main()
