import hashlib
import json
import unittest
from pathlib import Path

from deplab.batch import load_manifest
from deplab.smoke import build_smoke_script


ROOT = Path(__file__).resolve().parents[1]


def read_json(relative: str) -> dict:
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def directory_sha256(directory: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(directory.glob("*.json")):
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(sha256(path)))
    return digest.hexdigest()


class ExpandedMatrixTests(unittest.TestCase):
    def test_scope_has_twenty_packages_and_ten_versions_each(self) -> None:
        scope = read_json("configs/expanded-scope.json")
        self.assertEqual(len(scope["packages"]), 20)
        self.assertTrue(
            all(len(package["versions"]) == 10 for package in scope["packages"].values())
        )

    def test_development_and_final_holdout_are_disjoint_and_complete(self) -> None:
        development = load_manifest(ROOT / "configs/expanded-development-matrix.json")
        holdout = load_manifest(ROOT / "configs/expanded-final-holdout-matrix.json")
        development_ids = {spec.experiment_id for spec in development}
        holdout_ids = {spec.experiment_id for spec in holdout}
        development_pairs = read_json("configs/expanded-development-pairs.json")["families"]
        holdout_pairs = read_json("configs/expanded-final-holdout-pairs.json")["families"]
        development_packages = {
            row[key] for row in development_pairs for key in ("package_a", "package_b")
        }
        holdout_packages = {
            row[key] for row in holdout_pairs for key in ("package_a", "package_b")
        }
        self.assertEqual(len(development), 3269)
        self.assertEqual(len(holdout), 840)
        self.assertEqual(len(development) + len(holdout), 4109)
        self.assertEqual(len(development_pairs), 15)
        self.assertEqual(len(holdout_pairs), 3)
        self.assertFalse(development_ids & holdout_ids)
        self.assertFalse(development_packages & holdout_packages)

    def test_every_family_has_an_interoperability_smoke_test(self) -> None:
        families = (
            read_json("configs/expanded-development-pairs.json")["families"]
            + read_json("configs/expanded-final-holdout-pairs.json")["families"]
        )
        for family in families:
            script = build_smoke_script(family["package_a"], family["package_b"])
            self.assertIn('"strength": "interoperability"', script, family["name"])

    def test_every_selected_release_has_a_version_pinned_changelog_record(self) -> None:
        rows = [
            json.loads(line)
            for line in (
                ROOT / "outputs/changelog-catalog-expanded-v1.2.0.jsonl"
            ).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        self.assertEqual(len(rows), 200)
        self.assertEqual(len({row["changelog_id"] for row in rows}), 200)
        self.assertTrue(all(row["version_section_found"] for row in rows))

    def test_all_previous_results_are_reusable_development_rows(self) -> None:
        development_ids = {
            spec.experiment_id
            for spec in load_manifest(ROOT / "configs/expanded-development-matrix.json")
        }
        previous = set()
        systematic_candidates = (
            ROOT / "outputs/systematic-main-full.jsonl",
            ROOT / "outputs/systematic-main.jsonl",
        )
        systematic = next((path for path in systematic_candidates if path.exists()), None)
        self.assertIsNotNone(systematic, "systematic result file is missing")
        for path in (systematic, ROOT / "outputs/external-test-results.jsonl"):
            for line in path.read_text(encoding="utf-8").splitlines():
                previous.add(json.loads(line)["experiment_id"])
        self.assertEqual(len(previous), 707)
        self.assertTrue(previous <= development_ids)
        self.assertEqual(len(development_ids - previous), 2562)

    def test_frozen_hashes_and_shards_match(self) -> None:
        freeze = read_json("configs/expanded-matrices-freeze-v1.0.0.json")
        for split in freeze["splits"].values():
            self.assertEqual(sha256(ROOT / split["path"]), split["sha256"])
            directory = ROOT / split["shard_directory"]
            self.assertEqual(len(list(directory.glob("*.json"))), split["shards"])
            self.assertEqual(directory_sha256(directory), split["shard_directory_sha256"])
        for artifact in freeze["frozen_inputs"]:
            self.assertEqual(sha256(ROOT / artifact["path"]), artifact["sha256"])


if __name__ == "__main__":
    unittest.main()
