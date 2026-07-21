import unittest

from deplab.pypi import PyPIClient


PAYLOAD = {
    "info": {
        "name": "demo",
        "version": "1.0.0",
        "requires_python": ">=3.10,<3.12",
        "requires_dist": ["helper>=2"],
        "provides_extra": ["test"],
        "classifiers": ["Programming Language :: Python :: 3"],
        "project_urls": {"Source": "https://example.test/source"},
        "yanked": False,
    },
    "urls": [
        {
            "packagetype": "bdist_wheel",
            "filename": "demo-1.0.0-py3-none-any.whl",
            "url": "https://files.example.test/demo.whl",
            "size": 123,
            "digests": {"sha256": "abc"},
            "upload_time_iso_8601": "2024-01-02T03:04:05Z",
            "yanked": False,
        },
        {
            "packagetype": "sdist",
            "filename": "demo-1.0.0.tar.gz",
            "url": "https://files.example.test/demo.tar.gz",
        },
    ],
}


class PyPIClientTests(unittest.TestCase):
    def test_collects_metadata_and_marks_eligible_wheel(self) -> None:
        client = PyPIClient(fetcher=lambda _: PAYLOAD)
        release = client.release("demo", "1.0.0", "3.11")
        self.assertEqual(release.requires_dist, ["helper>=2"])
        self.assertEqual(len(release.wheels), 1)
        self.assertTrue(release.wheels[0].compatible)
        self.assertEqual(release.wheels[0].sha256, "abc")
        self.assertFalse(release.wheels[0].has_native_extensions)

    def test_requires_python_exclusion_is_coverage_not_pair_failure(self) -> None:
        client = PyPIClient(fetcher=lambda _: PAYLOAD)
        release = client.release("demo", "1.0.0", "3.12")
        self.assertFalse(release.wheels[0].compatible)
        self.assertIn("Requires-Python", release.wheels[0].compatibility_reason)


if __name__ == "__main__":
    unittest.main()
