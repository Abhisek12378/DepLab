import unittest

from deplab.wheels import parse_wheel_tags, requires_python_allows, wheel_is_compatible


class WheelTests(unittest.TestCase):
    def test_parses_platform_with_hyphenated_distribution(self) -> None:
        tags = parse_wheel_tags("demo_pkg-1.2.0-cp311-cp311-manylinux_2_17_x86_64.whl")
        self.assertEqual(tags.python, "cp311")
        self.assertEqual(tags.abi, "cp311")
        self.assertEqual(tags.platform, "manylinux_2_17_x86_64")

    def test_accepts_matching_manylinux_wheel(self) -> None:
        compatible, _ = wheel_is_compatible(
            "numpy-1.26.4-cp311-cp311-manylinux_2_17_x86_64.whl", "3.11"
        )
        self.assertTrue(compatible)

    def test_accepts_older_stable_abi_wheel(self) -> None:
        compatible, reason = wheel_is_compatible(
            "cryptography-42.0.0-cp37-abi3-manylinux_2_17_x86_64.whl", "3.12"
        )
        self.assertTrue(compatible)
        self.assertIn("stable ABI", reason)

    def test_rejects_wrong_python_and_platform(self) -> None:
        wrong_python, _ = wheel_is_compatible(
            "numpy-1.26.4-cp310-cp310-manylinux_2_17_x86_64.whl", "3.11"
        )
        wrong_platform, _ = wheel_is_compatible(
            "numpy-1.26.4-cp311-cp311-win_amd64.whl", "3.11"
        )
        musl_platform, _ = wheel_is_compatible(
            "numpy-1.26.4-cp311-cp311-musllinux_1_2_x86_64.whl", "3.11"
        )
        self.assertFalse(wrong_python)
        self.assertFalse(wrong_platform)
        self.assertFalse(musl_platform)

    def test_requires_python_common_constraints(self) -> None:
        self.assertTrue(requires_python_allows(">=3.9,<3.12", "3.11"))
        self.assertFalse(requires_python_allows(">=3.9,<3.12", "3.12"))
        self.assertTrue(requires_python_allows("~=3.11.1", "3.11"))
        self.assertFalse(requires_python_allows("~=3.11.1", "3.12"))


if __name__ == "__main__":
    unittest.main()
