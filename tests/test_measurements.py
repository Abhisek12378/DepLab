import tempfile
import unittest
from pathlib import Path

from deplab.measurements import capture_cache, finish_resource_metrics, start_resource_metrics
from deplab.models import StageResult


class MeasurementTests(unittest.TestCase):
    def test_cache_snapshot_and_changes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache = root / "cache"
            metrics = start_resource_metrics(root / "runs", cache)
            self.assertEqual(metrics.cache_state_before, "empty")
            cache.mkdir()
            (cache / "first.whl").write_bytes(b"1234")
            nested = cache / "nested"
            nested.mkdir()
            (nested / "metadata").write_bytes(b"abc")
            stages = [StageResult("install", [], 0, 0.1, peak_rss_bytes=12345)]
            finish_resource_metrics(metrics, root / "runs", cache, stages)
        self.assertEqual(metrics.cache_after.file_count, 2)
        self.assertEqual(metrics.cache_after.size_bytes, 7)
        self.assertEqual(metrics.cache_size_change_bytes, 7)
        self.assertEqual(metrics.peak_stage_rss_bytes, 12345)

    def test_existing_cache_is_populated(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory)
            (cache / "entry").write_bytes(b"data")
            snapshot = capture_cache(cache)
            self.assertTrue(snapshot.exists)
            self.assertEqual(snapshot.file_count, 1)
            self.assertEqual(snapshot.size_bytes, 4)

    def test_parallel_shared_cache_snapshot_omits_expensive_content_scan(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache = root / "cache"
            cache.mkdir()
            (cache / "entry").write_bytes(b"data")
            metrics = start_resource_metrics(
                root / "runs", cache, measure_cache_contents=False
            )
            finish_resource_metrics(
                metrics,
                root / "runs",
                cache,
                [],
                measure_cache_contents=False,
            )
        self.assertEqual(metrics.cache_state_before, "populated")
        self.assertIsNone(metrics.cache_before.file_count)
        self.assertIsNone(metrics.cache_before.size_bytes)
        self.assertIsNone(metrics.cache_after.file_count)
        self.assertIsNone(metrics.cache_after.size_bytes)
        self.assertIsNone(metrics.cache_size_change_bytes)
        self.assertIn("omitted", metrics.measurement_scope)


if __name__ == "__main__":
    unittest.main()
