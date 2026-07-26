from __future__ import annotations

import json
import os
import re
import signal
import shutil
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Callable, Sequence

from .artifacts import ArtifactLockError, load_installed_wheels, lock_sha256
from .models import (
    ExperimentResult,
    ExperimentSpec,
    PackageRelease,
    RuntimeEnvironment,
    StageResult,
    utc_now,
)
from .measurements import finish_resource_metrics, process_tree_rss_bytes, start_resource_metrics
from .smoke_v3 import build_smoke_script


CommandRunner = Callable[[Sequence[str], float], StageResult]


def subprocess_runner(command: Sequence[str], timeout: float) -> StageResult:
    started = time.monotonic()
    process = subprocess.Popen(
        list(command),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=os.name == "posix",
    )
    deadline = started + timeout
    peak_rss: int | None = None
    timed_out = False
    stdout = ""
    stderr = ""
    while True:
        current_rss = process_tree_rss_bytes(process.pid)
        if current_rss is not None:
            peak_rss = max(peak_rss or 0, current_rss)
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            timed_out = True
            _kill_process_tree(process)
            stdout, stderr = process.communicate()
            break
        try:
            stdout, stderr = process.communicate(timeout=min(0.05, remaining))
            break
        except subprocess.TimeoutExpired:
            continue
    return StageResult(
        stage="",
        command=list(command),
        exit_code=None if timed_out else process.returncode,
        duration_seconds=time.monotonic() - started,
        stdout=_to_text(stdout),
        stderr=_to_text(stderr),
        timed_out=timed_out,
        peak_rss_bytes=peak_rss,
    )


def _kill_process_tree(process: subprocess.Popen[str]) -> None:
    try:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGKILL)
        else:
            process.kill()
    except (OSError, ProcessLookupError):
        pass


def _to_text(value: str | bytes | None) -> str:
    if isinstance(value, bytes):
        return value.decode(errors="replace")
    return value or ""


class ExperimentRunner:
    def __init__(
        self,
        run_root: Path,
        uv_command: str = "uv",
        timeout_seconds: float = 180.0,
        command_runner: CommandRunner = subprocess_runner,
        enforce_linux_host: bool = True,
        uv_cache_dir: Path | None = None,
        cache_scope: str = "shared",
        cleanup_environments: bool = False,
        measure_cache_contents: bool = True,
    ) -> None:
        self.run_root = run_root
        self.uv_command = uv_command
        self.timeout_seconds = timeout_seconds
        self._execute = command_runner
        self.enforce_linux_host = enforce_linux_host
        self.uv_cache_dir = uv_cache_dir.resolve() if uv_cache_dir is not None else None
        if cache_scope not in {"shared", "experiment"}:
            raise ValueError("cache_scope must be 'shared' or 'experiment'")
        self.cache_scope = cache_scope
        self.cleanup_environments = cleanup_environments
        self.measure_cache_contents = measure_cache_contents

    def run(
        self,
        spec: ExperimentSpec,
        release_a: PackageRelease,
        release_b: PackageRelease,
    ) -> ExperimentResult:
        started_clock = time.monotonic()
        result = ExperimentResult(
            schema_version="1.3.0",
            experiment_id=spec.experiment_id,
            spec=spec,
            outcome="infrastructure_failure",
            started_at=utc_now(),
            duration_seconds=0.0,
        )
        selected = []
        for release in (release_a, release_b):
            candidates = sorted(
                (wheel for wheel in release.wheels if wheel.compatible and not wheel.yanked),
                key=lambda wheel: wheel.filename,
            )
            if not candidates:
                result.outcome = "wheel_unavailable"
                result.normalized_error = (
                    f"No compatible non-yanked wheel for {release.name}=={release.version} "
                    f"on Python {spec.python_version} {spec.os}/{spec.architecture}"
                )
                return self._finish(result, started_clock)
            selected.append(candidates[0])
        result.wheel_artifacts = selected

        if self.enforce_linux_host and not sys.platform.startswith("linux"):
            result.normalized_error = "Experiments for the fixed Linux platform must execute on a Linux host"
            return self._finish(result, started_clock)
        if shutil.which(self.uv_command) is None and self._execute is subprocess_runner:
            result.normalized_error = f"uv executable not found: {self.uv_command}"
            return self._finish(result, started_clock)

        result.measured = True
        cache_dir = self._cache_dir(spec)
        result.resources = start_resource_metrics(
            self.run_root,
            cache_dir,
            measure_cache_contents=self.measure_cache_contents,
        )
        experiment_dir = self.run_root / spec.experiment_id
        environment_dir = experiment_dir / ".venv"
        experiment_dir.mkdir(parents=True, exist_ok=True)
        smoke_path = experiment_dir / "smoke.py"
        smoke_path.write_text(
            build_smoke_script(spec.package_a.name, spec.package_b.name), encoding="utf-8"
        )
        python_path = environment_dir / "bin" / "python"
        requirements_path = experiment_dir / "requirements.in"
        lock_path = experiment_dir / "pylock.toml"

        create = self._stage(
            "create_environment",
            [
                *self._uv("venv", cache_dir),
                "--clear",
                "--python",
                spec.python_version,
                str(environment_dir),
            ],
        )
        result.stages.append(create)
        if create.timed_out:
            result.outcome = "timeout"
            result.normalized_error = "Environment creation timed out"
            return self._finish(result, started_clock)
        if create.exit_code != 0:
            result.normalized_error = _normalize_error(create.stderr or create.stdout)
            return self._finish(result, started_clock)

        runtime = self._stage(
            "capture_runtime",
            [
                str(python_path),
                "-c",
                (
                    "import json,platform;"
                    "print(json.dumps({'python_version':platform.python_version(),"
                    "'python_implementation':platform.python_implementation(),"
                    "'os':platform.system().lower(),'kernel':platform.release(),"
                    "'architecture':platform.machine(),"
                    "'libc':' '.join(x for x in platform.libc_ver() if x)}))"
                ),
            ],
        )
        result.stages.append(runtime)
        toolchain = self._stage("capture_toolchain", [self.uv_command, "--version"])
        result.stages.append(toolchain)
        if runtime.timed_out or toolchain.timed_out:
            result.outcome = "timeout"
            result.normalized_error = "Runtime identity capture timed out"
            return self._finish(result, started_clock)
        if runtime.exit_code != 0 or toolchain.exit_code != 0:
            result.normalized_error = _normalize_error(
                runtime.stderr or toolchain.stderr or runtime.stdout or toolchain.stdout
            )
            return self._finish(result, started_clock)
        try:
            identity = json.loads(runtime.stdout)
            result.runtime = RuntimeEnvironment(
                python_version=str(identity["python_version"]),
                python_implementation=str(identity["python_implementation"]),
                os=str(identity["os"]),
                kernel=str(identity["kernel"]),
                architecture=str(identity["architecture"]),
                libc=str(identity["libc"]),
                uv_version=toolchain.stdout.strip(),
            )
            if not result.runtime.python_version.startswith(f"{spec.python_version}."):
                raise ValueError(
                    f"requested Python {spec.python_version}, got {result.runtime.python_version}"
                )
            if result.runtime.os != "linux" or result.runtime.architecture not in {
                "x86_64",
                "amd64",
            }:
                raise ValueError(
                    f"runtime is {result.runtime.os}/{result.runtime.architecture}, expected linux/x86_64"
                )
            if "glibc" not in result.runtime.libc.lower():
                raise ValueError(f"runtime libc is {result.runtime.libc!r}, expected glibc")
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            result.normalized_error = f"Invalid runtime identity output: {exc}"
            return self._finish(result, started_clock)

        top_level_targets = [
            f"{wheel.url}#sha256={wheel.sha256}" if wheel.sha256 else wheel.url for wheel in selected
        ]
        requirements_path.write_text("\n".join(top_level_targets) + "\n", encoding="utf-8")
        lock_path.unlink(missing_ok=True)
        resolve = self._stage(
            "resolve_artifacts",
            [
                *self._uv("pip", cache_dir),
                "compile",
                str(requirements_path),
                "--output-file",
                str(lock_path),
                "--format",
                "pylock.toml",
                "--python",
                str(python_path),
                "--only-binary",
                ":all:",
                "--generate-hashes",
            ],
        )
        result.stages.append(resolve)
        if resolve.timed_out:
            result.outcome = "timeout"
            result.normalized_error = "Artifact resolution timed out"
            return self._finish(result, started_clock)
        if resolve.exit_code != 0:
            combined = f"{resolve.stdout}\n{resolve.stderr}"
            result.outcome = (
                "resolution_failure"
                if re.search(r"no solution|unsatisfiable|resolution|conflict", combined, re.I)
                else "infrastructure_failure"
            )
            result.normalized_error = _normalize_error(combined)
            result.exception_type = _exception_type(combined)
            return self._finish(result, started_clock)

        try:
            artifacts = load_installed_wheels(lock_path, spec)
            artifacts = _verify_and_enrich_top_level_artifacts(artifacts, selected, spec)
            result.artifact_lock_format = "pylock.toml (PEP 751)"
            result.artifact_lock_sha256 = lock_sha256(lock_path)
        except (ArtifactLockError, OSError, ValueError) as exc:
            result.normalized_error = f"Invalid artifact lock: {exc}"
            result.exception_type = type(exc).__name__
            return self._finish(result, started_clock)

        exact_targets = [_hashed_url(item.url, item.sha256) for item in artifacts]
        install = self._stage(
            "install_exact_artifacts",
            [
                *self._uv("pip", cache_dir),
                "install",
                "--python",
                str(python_path),
                "--only-binary=:all:",
                "--no-deps",
                *exact_targets,
            ],
        )
        result.stages.append(install)
        if install.timed_out:
            result.outcome = "timeout"
            result.normalized_error = "Exact artifact installation timed out"
            return self._finish(result, started_clock)
        if install.exit_code != 0:
            combined = f"{install.stdout}\n{install.stderr}"
            result.outcome = "installation_failure"
            result.normalized_error = _normalize_error(combined)
            result.exception_type = _exception_type(combined)
            return self._finish(result, started_clock)

        environment = self._stage(
            "capture_environment",
            [
                str(python_path),
                "-c",
                (
                    "import importlib.metadata as m,json;"
                    "print(json.dumps(sorted([[d.metadata['Name'],d.version] "
                    "for d in m.distributions()])))"
                ),
            ],
        )
        result.stages.append(environment)
        if environment.timed_out:
            result.outcome = "timeout"
            result.normalized_error = "Installed environment capture timed out"
            return self._finish(result, started_clock)
        try:
            installed = json.loads(environment.stdout) if environment.exit_code == 0 else None
            result.installed_environment = _validate_installed_environment(installed, artifacts)
            result.installed_wheel_artifacts = artifacts
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            result.normalized_error = f"Invalid installed environment: {exc}"
            result.exception_type = type(exc).__name__
            return self._finish(result, started_clock)

        smoke = self._stage("smoke_test", [str(python_path), str(smoke_path)])
        result.stages.append(smoke)
        if smoke.timed_out:
            result.outcome = "timeout"
            result.normalized_error = "Smoke test timed out"
            return self._finish(result, started_clock)
        if smoke.exit_code != 0:
            combined = f"{smoke.stdout}\n{smoke.stderr}"
            result.outcome = (
                "smoke_test_failure" if '"deplab_stage": "imports_passed"' in smoke.stdout else "import_failure"
            )
            result.normalized_error = _normalize_error(combined)
            result.exception_type = _exception_type(combined)
            return self._finish(result, started_clock)

        result.outcome = "pass"
        return self._finish(result, started_clock)

    def _stage(self, name: str, command: list[str]) -> StageResult:
        return replace(self._execute(command, self.timeout_seconds), stage=name)

    def _uv(self, subcommand: str, cache_dir: Path) -> list[str]:
        command = [self.uv_command]
        if self.uv_cache_dir is not None or self.cache_scope == "experiment":
            command.extend(("--cache-dir", str(cache_dir)))
        command.append(subcommand)
        return command

    def _cache_dir(self, spec: ExperimentSpec) -> Path:
        if self.uv_cache_dir is not None:
            base = self.uv_cache_dir
        elif os.environ.get("UV_CACHE_DIR"):
            base = Path(os.environ["UV_CACHE_DIR"]).resolve()
        elif os.environ.get("XDG_CACHE_HOME"):
            base = (Path(os.environ["XDG_CACHE_HOME"]) / "uv").resolve()
        else:
            base = Path.home() / ".cache" / "uv"
        return base / spec.experiment_id if self.cache_scope == "experiment" else base

    def _finish(self, result: ExperimentResult, started: float) -> ExperimentResult:
        result.duration_seconds = time.monotonic() - started
        if result.resources is not None:
            finish_resource_metrics(
                result.resources,
                self.run_root,
                Path(result.resources.cache_before.directory),
                result.stages,
                measure_cache_contents=self.measure_cache_contents,
            )
        if self.cleanup_environments:
            cleanup_started = time.monotonic()
            environment_dir = (self.run_root / result.experiment_id / ".venv").resolve()
            expected_parent = (self.run_root / result.experiment_id).resolve()
            try:
                if environment_dir.parent != expected_parent:
                    raise ValueError("refusing to clean an environment outside its experiment directory")
                if environment_dir.exists():
                    shutil.rmtree(environment_dir)
                result.stages.append(
                    StageResult(
                        stage="cleanup_environment",
                        command=["internal", "remove_environment", str(environment_dir)],
                        exit_code=0,
                        duration_seconds=time.monotonic() - cleanup_started,
                    )
                )
            except (OSError, ValueError) as exc:
                result.stages.append(
                    StageResult(
                        stage="cleanup_environment",
                        command=["internal", "remove_environment", str(environment_dir)],
                        exit_code=1,
                        duration_seconds=time.monotonic() - cleanup_started,
                        stderr=str(exc),
                    )
                )
                if result.outcome == "pass":
                    result.outcome = "infrastructure_failure"
                    result.normalized_error = f"Environment cleanup failed: {exc}"
        return result


def _normalize_error(output: str, max_length: int = 2000) -> str:
    normalized = re.sub(r"\s+", " ", output).strip()
    normalized = re.sub(r"/[^ ]*/\.venv", "<venv>", normalized)
    return normalized[:max_length]


def _exception_type(output: str) -> str | None:
    matches = re.findall(r"^([A-Za-z_][\w.]*(?:Error|Exception))(?::|$)", output, re.MULTILINE)
    return matches[-1] if matches else None


def _hashed_url(url: str, sha256: str) -> str:
    separator = "&" if "#" in url else "#"
    return f"{url}{separator}sha256={sha256}"


def _canonical_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _verify_and_enrich_top_level_artifacts(artifacts, selected, spec: ExperimentSpec):
    expected_wheels = {
        _canonical_name(pin.name): wheel
        for pin, wheel in zip((spec.package_a, spec.package_b), selected)
    }
    expected = {
        name: (wheel.filename, wheel.sha256) for name, wheel in expected_wheels.items()
    }
    actual = {
        _canonical_name(item.package): (item.filename, item.sha256)
        for item in artifacts
        if item.top_level
    }
    if actual != expected:
        raise ArtifactLockError(
            f"top-level lock artifacts differ from audited wheels: expected {expected}, got {actual}"
        )
    enriched = [
        replace(
            item,
            size=item.size
            if item.size is not None
            else expected_wheels[_canonical_name(item.package)].size,
        )
        if item.top_level
        else item
        for item in artifacts
    ]
    if any(item.size is None for item in enriched):
        raise ArtifactLockError("one or more selected wheel artifacts have no file size")
    return enriched


def _validate_installed_environment(installed, artifacts) -> list[str]:
    if not isinstance(installed, list):
        raise ValueError("capture output is not a list")
    actual: dict[str, tuple[str, str]] = {}
    for row in installed:
        if not isinstance(row, list) or len(row) != 2:
            raise ValueError("capture output contains an invalid package row")
        name, version = str(row[0]), str(row[1])
        actual[_canonical_name(name)] = (name, version)
    expected = {
        _canonical_name(item.package): (item.package, item.version) for item in artifacts
    }
    actual_versions = {name: value[1] for name, value in actual.items()}
    expected_versions = {name: value[1] for name, value in expected.items()}
    if actual_versions != expected_versions:
        raise ValueError(
            f"installed packages differ from artifact lock: expected {expected_versions}, got {actual_versions}"
        )
    return [f"{actual[name][0]}=={actual[name][1]}" for name in sorted(actual)]
