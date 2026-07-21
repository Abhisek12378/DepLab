import tempfile
import unittest
from pathlib import Path

from deplab.models import (
    ExperimentSpec,
    PackagePin,
    PackageRelease,
    StageResult,
    WheelArtifact,
)
from deplab.runner import ExperimentRunner


def release(name: str, compatible: bool = True) -> PackageRelease:
    wheel = WheelArtifact(
        filename=f"{name}-1.0.0-py3-none-any.whl",
        url=f"https://files.example.test/{name}.whl",
        size=100,
        sha256="a" * 64,
        python_tag="py3",
        abi_tag="none",
        platform_tag="any",
        compatible=compatible,
        compatibility_reason="fixture",
    )
    return PackageRelease(
        name=name,
        version="1.0.0",
        requires_python=">=3.10",
        requires_dist=[],
        extras=[],
        classifiers=[],
        project_urls={},
        release_date="2024-01-01T00:00:00Z",
        yanked=False,
        wheels=[wheel],
        source="test_fixture",
    )


def stage(
    exit_code: int = 0,
    stdout: str = "",
    stderr: str = "",
    peak_rss_bytes: int | None = None,
) -> StageResult:
    return StageResult(
        "", [], exit_code, 0.01, stdout, stderr, peak_rss_bytes=peak_rss_bytes
    )


RUNTIME_JSON = (
    '{"python_version":"3.11.15","python_implementation":"CPython",'
    '"os":"linux","kernel":"6.8.0","architecture":"x86_64","libc":"glibc 2.39"}\n'
)

PYLOCK = """\
lock-version = "1.0"
[[packages]]
name = "alpha"
version = "1.0.0"
[[packages.wheels]]
name = "alpha-1.0.0-py3-none-any.whl"
url = "https://files.example.test/alpha.whl"
size = 100
hashes = { sha256 = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa" }
[[packages]]
name = "beta"
version = "1.0.0"
[[packages.wheels]]
name = "beta-1.0.0-py3-none-any.whl"
url = "https://files.example.test/beta.whl"
size = 100
hashes = { sha256 = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa" }
[[packages]]
name = "helper"
version = "2.0.0"
[[packages.wheels]]
name = "helper-2.0.0-py3-none-any.whl"
url = "https://files.example.test/helper.whl"
size = 100
hashes = { sha256 = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb" }
"""


class SequenceRunner:
    def __init__(self, responses: list[StageResult]) -> None:
        self.responses = iter(responses)
        self.commands: list[list[str]] = []

    def __call__(self, command, timeout):
        self.commands.append(list(command))
        response = next(self.responses)
        if "venv" in command and response.exit_code == 0:
            Path(command[-1]).mkdir(parents=True, exist_ok=True)
        if "compile" in command and response.exit_code == 0:
            output_index = command.index("--output-file") + 1
            Path(command[output_index]).write_text(PYLOCK, encoding="utf-8")
        return response


class ExperimentRunnerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.spec = ExperimentSpec(PackagePin("alpha", "1.0.0"), PackagePin("beta", "1.0.0"), "3.11")

    def test_wheel_unavailable_is_not_compatibility_failure(self) -> None:
        fake = SequenceRunner([])
        with tempfile.TemporaryDirectory() as directory:
            result = ExperimentRunner(
                Path(directory), command_runner=fake, enforce_linux_host=False
            ).run(self.spec, release("alpha", compatible=False), release("beta"))
        self.assertEqual(result.outcome, "wheel_unavailable")
        self.assertFalse(result.measured)
        self.assertEqual(fake.commands, [])

    def test_success_records_full_environment(self) -> None:
        fake = SequenceRunner(
            [
                stage(peak_rss_bytes=10_000),
                stage(stdout=RUNTIME_JSON),
                stage(stdout="uv 0.11.29\n"),
                stage(peak_rss_bytes=50_000),
                stage(peak_rss_bytes=40_000),
                stage(stdout='[["alpha", "1.0.0"], ["beta", "1.0.0"], ["helper", "2.0.0"]]\n'),
                stage(stdout='{"deplab_stage": "imports_passed"}\n{"deplab_smoke": "pass"}\n'),
            ]
        )
        with tempfile.TemporaryDirectory() as directory:
            result = ExperimentRunner(
                Path(directory),
                command_runner=fake,
                enforce_linux_host=False,
                uv_cache_dir=Path(directory) / "measurement-cache",
                cache_scope="experiment",
            ).run(self.spec, release("alpha"), release("beta"))
        self.assertEqual(result.outcome, "pass")
        self.assertTrue(result.measured)
        self.assertEqual(result.runtime.python_version, "3.11.15")
        self.assertEqual(result.runtime.uv_version, "uv 0.11.29")
        self.assertEqual(result.resources.cache_state_before, "empty")
        self.assertEqual(result.resources.peak_stage_rss_bytes, 50_000)
        self.assertEqual(result.installed_environment[-1], "helper==2.0.0")
        self.assertEqual(len(result.installed_wheel_artifacts), 3)
        self.assertTrue(all(item.size == 100 for item in result.installed_wheel_artifacts))
        self.assertEqual(result.artifact_lock_format, "pylock.toml (PEP 751)")
        self.assertEqual(len(result.artifact_lock_sha256), 64)
        self.assertIn("--clear", fake.commands[0])
        self.assertIn("--cache-dir", fake.commands[0])
        self.assertIn(self.spec.experiment_id, fake.commands[0][2])
        self.assertIn("--generate-hashes", fake.commands[3])
        self.assertIn("--only-binary=:all:", fake.commands[4])
        self.assertIn("--no-deps", fake.commands[4])
        self.assertTrue(all("#sha256=" in target for target in fake.commands[4][-3:]))

    def test_resolution_and_import_failures_are_distinct(self) -> None:
        resolution = SequenceRunner(
            [stage(), stage(stdout=RUNTIME_JSON), stage(stdout="uv 0.11.29\n"), stage(1, stderr="No solution found: version conflict")]
        )
        with tempfile.TemporaryDirectory() as directory:
            result = ExperimentRunner(
                Path(directory), command_runner=resolution, enforce_linux_host=False
            ).run(self.spec, release("alpha"), release("beta"))
        self.assertEqual(result.outcome, "resolution_failure")

        imports = SequenceRunner(
            [
                stage(),
                stage(stdout=RUNTIME_JSON),
                stage(stdout="uv 0.11.29\n"),
                stage(),
                stage(),
                stage(stdout='[["alpha", "1.0.0"], ["beta", "1.0.0"], ["helper", "2.0.0"]]\n'),
                stage(1, stderr="ModuleNotFoundError: No module named 'alpha'"),
            ]
        )
        with tempfile.TemporaryDirectory() as directory:
            result = ExperimentRunner(
                Path(directory), command_runner=imports, enforce_linux_host=False
            ).run(self.spec, release("alpha"), release("beta"))
        self.assertEqual(result.outcome, "import_failure")
        self.assertEqual(result.exception_type, "ModuleNotFoundError")

    def test_failure_after_imports_is_smoke_test_failure(self) -> None:
        fake = SequenceRunner(
            [
                stage(),
                stage(stdout=RUNTIME_JSON),
                stage(stdout="uv 0.11.29\n"),
                stage(),
                stage(),
                stage(stdout='[["alpha", "1.0.0"], ["beta", "1.0.0"], ["helper", "2.0.0"]]\n'),
                stage(1, stdout='{"deplab_stage": "imports_passed"}\n', stderr="AssertionError"),
            ]
        )
        with tempfile.TemporaryDirectory() as directory:
            result = ExperimentRunner(
                Path(directory), command_runner=fake, enforce_linux_host=False
            ).run(self.spec, release("alpha"), release("beta"))
        self.assertEqual(result.outcome, "smoke_test_failure")
        self.assertEqual(result.exception_type, "AssertionError")

    def test_cleanup_removes_only_the_temporary_environment(self) -> None:
        fake = SequenceRunner(
            [
                stage(),
                stage(stdout=RUNTIME_JSON),
                stage(stdout="uv 0.11.29\n"),
                stage(),
                stage(),
                stage(stdout='[["alpha", "1.0.0"], ["beta", "1.0.0"], ["helper", "2.0.0"]]\n'),
                stage(stdout='{"deplab_stage": "imports_passed"}\n{"deplab_smoke": "pass"}\n'),
            ]
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "runs"
            result = ExperimentRunner(
                root,
                command_runner=fake,
                enforce_linux_host=False,
                cleanup_environments=True,
            ).run(self.spec, release("alpha"), release("beta"))
            experiment_dir = root / self.spec.experiment_id
            self.assertFalse((experiment_dir / ".venv").exists())
            self.assertTrue((experiment_dir / "pylock.toml").exists())
        self.assertEqual(result.outcome, "pass")
        self.assertEqual(result.stages[-1].stage, "cleanup_environment")
        self.assertEqual(result.stages[-1].exit_code, 0)


if __name__ == "__main__":
    unittest.main()
