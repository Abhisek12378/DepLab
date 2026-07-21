import tempfile
import unittest
from pathlib import Path

from deplab.artifacts import ArtifactLockError, load_installed_wheels
from deplab.models import ExperimentSpec, PackagePin


PYLOCK = """\
lock-version = "1.0"
environments = ["python_full_version == '3.11.15'"]

[[packages]]
name = "alpha"
version = "1.0.0"
[[packages.wheels]]
name = "alpha-1.0.0-py3-none-any.whl"
url = "https://files.example/alpha-1.0.0-py3-none-any.whl"
size = 100
hashes = { sha256 = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa" }

[[packages]]
name = "helper"
version = "2.0.0"
[[packages.wheels]]
name = "helper-2.0.0-cp310-abi3-manylinux_2_17_x86_64.whl"
url = "https://files.example/helper-2.0.0-cp310-abi3-manylinux_2_17_x86_64.whl"
size = 200
hashes = { sha256 = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb" }
[[packages.wheels]]
name = "helper-2.0.0-cp311-cp311-win_amd64.whl"
url = "https://files.example/helper-2.0.0-cp311-cp311-win_amd64.whl"
size = 201
hashes = { sha256 = "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc" }

[[packages]]
name = "beta"
version = "1.0.0"
[packages.archive]
url = "https://files.example/beta-1.0.0-py3-none-any.whl"
size = 300
hashes = { sha256 = "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd" }
"""


class ArtifactLockTests(unittest.TestCase):
    def setUp(self) -> None:
        self.spec = ExperimentSpec(
            PackagePin("alpha", "1.0.0"), PackagePin("beta", "1.0.0"), "3.11"
        )

    def test_selects_exact_compatible_transitive_wheels(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pylock.toml"
            path.write_text(PYLOCK, encoding="utf-8")
            artifacts = load_installed_wheels(path, self.spec)
        self.assertEqual([item.package for item in artifacts], ["alpha", "beta", "helper"])
        self.assertEqual(sum(item.top_level for item in artifacts), 2)
        helper = next(item for item in artifacts if item.package == "helper")
        self.assertIn("manylinux_2_17_x86_64", helper.filename)
        self.assertEqual(helper.sha256, "b" * 64)

    def test_rejects_artifact_without_hash(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pylock.toml"
            path.write_text(PYLOCK.replace('hashes = { sha256 = "' + "b" * 64 + '" }', "hashes = {}"), encoding="utf-8")
            with self.assertRaises(ArtifactLockError):
                load_installed_wheels(path, self.spec)


if __name__ == "__main__":
    unittest.main()
