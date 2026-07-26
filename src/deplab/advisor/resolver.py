from __future__ import annotations

import os
import re
import subprocess
import tempfile
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Protocol

from packaging.version import InvalidVersion, Version

from .validation import canonical_name


_PACKAGE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
_CONFLICT_PATTERN = re.compile(
    r"no solution|unsatisfiable|conflicting requirements|dependency conflict",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ResolverResult:
    status: str
    resolvable: bool | None
    duration_seconds: float
    cache_hit: bool = False
    explanation: str = ""


class ResolverVerifier(Protocol):
    def verify(
        self,
        requirements: Mapping[str, str],
        python_version: str,
        platform: str,
    ) -> ResolverResult: ...


class UVCompileVerifier:
    """Checks resolution only. It never installs or executes package code."""

    def __init__(
        self,
        uv_command: str = "uv",
        timeout_seconds: float = 15.0,
        cache_entries: int = 2_048,
        cache_ttl_seconds: float = 900.0,
        maximum_concurrency: int = 2,
        uv_cache_dir: Path | None = None,
    ) -> None:
        if timeout_seconds <= 0 or cache_entries < 1 or maximum_concurrency < 1:
            raise ValueError("resolver limits must be positive")
        self.uv_command = uv_command
        self.timeout_seconds = timeout_seconds
        self.cache_entries = cache_entries
        self.cache_ttl_seconds = cache_ttl_seconds
        self.uv_cache_dir = uv_cache_dir
        self._semaphore = threading.BoundedSemaphore(maximum_concurrency)
        self._lock = threading.Lock()
        self._cache: OrderedDict[
            tuple[str, str, tuple[tuple[str, str], ...]],
            tuple[float, ResolverResult],
        ] = OrderedDict()

    def verify(
        self,
        requirements: Mapping[str, str],
        python_version: str,
        platform: str,
    ) -> ResolverResult:
        pins = _validated_pins(requirements)
        key = (python_version, platform, tuple(sorted(pins.items())))
        cached = self._cached(key)
        if cached is not None:
            return cached
        if platform != "linux-x86_64":
            return ResolverResult(
                status="unsupported_platform",
                resolvable=None,
                duration_seconds=0.0,
                explanation="The resolver currently supports Linux x86_64 only.",
            )
        started = time.monotonic()
        with self._semaphore:
            result = self._execute(pins, python_version, started)
        self._store(key, result)
        return result

    def _execute(
        self,
        pins: dict[str, str],
        python_version: str,
        started: float,
    ) -> ResolverResult:
        with tempfile.TemporaryDirectory(prefix="deplab-resolve-") as directory:
            root = Path(directory)
            source = root / "requirements.in"
            output = root / "requirements.lock"
            source.write_text(
                "".join(
                    f"{name}=={version}\n"
                    for name, version in sorted(pins.items())
                ),
                encoding="utf-8",
            )
            command = [
                self.uv_command,
                "pip",
                "compile",
                str(source),
                "--output-file",
                str(output),
                "--python-version",
                python_version,
                "--python-platform",
                "x86_64-unknown-linux-gnu",
                "--only-binary",
                ":all:",
                "--generate-hashes",
                "--no-header",
                "--no-annotate",
                "--color",
                "never",
            ]
            environment = os.environ.copy()
            if self.uv_cache_dir is not None:
                environment["UV_CACHE_DIR"] = str(self.uv_cache_dir)
            try:
                completed = subprocess.run(
                    command,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout_seconds,
                    check=False,
                    cwd=root,
                    env=environment,
                )
            except subprocess.TimeoutExpired:
                return ResolverResult(
                    status="timeout",
                    resolvable=None,
                    duration_seconds=time.monotonic() - started,
                    explanation="The resolver timed out; no compatibility conclusion was made.",
                )
            except OSError:
                return ResolverResult(
                    status="unavailable",
                    resolvable=None,
                    duration_seconds=time.monotonic() - started,
                    explanation="The uv resolver is temporarily unavailable.",
                )
            return _completed_result(completed, started)

    def _cached(
        self,
        key: tuple[str, str, tuple[tuple[str, str], ...]],
    ) -> ResolverResult | None:
        now = time.monotonic()
        with self._lock:
            item = self._cache.get(key)
            if item is None:
                return None
            created, result = item
            if now - created > self.cache_ttl_seconds:
                self._cache.pop(key, None)
                return None
            self._cache.move_to_end(key)
            return ResolverResult(
                **{
                    **result.__dict__,
                    "cache_hit": True,
                }
            )

    def _store(
        self,
        key: tuple[str, str, tuple[tuple[str, str], ...]],
        result: ResolverResult,
    ) -> None:
        with self._lock:
            self._cache[key] = (time.monotonic(), result)
            self._cache.move_to_end(key)
            while len(self._cache) > self.cache_entries:
                self._cache.popitem(last=False)


def _validated_pins(requirements: Mapping[str, str]) -> dict[str, str]:
    if not requirements or len(requirements) > 100:
        raise ValueError("resolver requires between 1 and 100 exact package pins")
    result: dict[str, str] = {}
    for raw_name, raw_version in requirements.items():
        name = str(raw_name).strip()
        version = str(raw_version).strip()
        if not _PACKAGE_NAME.fullmatch(name):
            raise ValueError(f"invalid package name for resolver: {name!r}")
        try:
            Version(version)
        except InvalidVersion as exc:
            raise ValueError(
                f"invalid exact version for resolver: {name}=={version}"
            ) from exc
        canonical = canonical_name(name)
        if canonical in result:
            raise ValueError(f"duplicate package pin for resolver: {canonical}")
        result[canonical] = version
    return result


def _completed_result(
    completed: subprocess.CompletedProcess[str],
    started: float,
) -> ResolverResult:
    duration = time.monotonic() - started
    if completed.returncode == 0:
        return ResolverResult(
            status="resolved",
            resolvable=True,
            duration_seconds=duration,
            explanation="uv produced a complete dependency lock without installing packages.",
        )
    combined = f"{completed.stdout}\n{completed.stderr}"
    if _CONFLICT_PATTERN.search(combined):
        return ResolverResult(
            status="unresolvable",
            resolvable=False,
            duration_seconds=duration,
            explanation="uv could not find a dependency set satisfying all exact pins.",
        )
    return ResolverResult(
        status="unavailable",
        resolvable=None,
        duration_seconds=duration,
        explanation="The resolver failed for a non-compatibility reason; no conclusion was made.",
    )
