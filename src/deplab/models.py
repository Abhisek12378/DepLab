from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal


Outcome = Literal[
    "pass",
    "resolution_failure",
    "wheel_unavailable",
    "installation_failure",
    "import_failure",
    "smoke_test_failure",
    "timeout",
    "infrastructure_failure",
]


@dataclass(frozen=True)
class PackagePin:
    name: str
    version: str

    @property
    def requirement(self) -> str:
        return f"{self.name}=={self.version}"


@dataclass(frozen=True)
class WheelArtifact:
    filename: str
    url: str
    size: int | None
    sha256: str | None
    python_tag: str
    abi_tag: str
    platform_tag: str
    has_native_extensions: bool | None = None
    uploaded_at: str | None = None
    yanked: bool = False
    compatible: bool = False
    compatibility_reason: str = ""


@dataclass(frozen=True)
class InstalledWheelArtifact:
    package: str
    version: str
    filename: str
    url: str
    size: int | None
    sha256: str
    python_tag: str
    abi_tag: str
    platform_tag: str
    top_level: bool = False


@dataclass(frozen=True)
class PackageRelease:
    name: str
    version: str
    requires_python: str | None
    requires_dist: list[str]
    extras: list[str]
    classifiers: list[str]
    project_urls: dict[str, str]
    release_date: str | None
    yanked: bool
    wheels: list[WheelArtifact]
    source: str = "pypi"


@dataclass(frozen=True)
class ExperimentSpec:
    package_a: PackagePin
    package_b: PackagePin
    python_version: str
    os: str = "linux"
    architecture: str = "x86_64"

    @property
    def experiment_id(self) -> str:
        raw = "|".join(
            (
                self.package_a.name.lower(),
                self.package_a.version,
                self.package_b.name.lower(),
                self.package_b.version,
                self.python_version,
                self.os,
                self.architecture,
            )
        )
        import hashlib

        return hashlib.sha256(raw.encode()).hexdigest()[:20]


@dataclass
class StageResult:
    stage: str
    command: list[str]
    exit_code: int | None
    duration_seconds: float
    stdout: str = ""
    stderr: str = ""
    timed_out: bool = False
    peak_rss_bytes: int | None = None


@dataclass(frozen=True)
class RuntimeEnvironment:
    python_version: str
    python_implementation: str
    os: str
    kernel: str
    architecture: str
    libc: str
    uv_version: str


@dataclass(frozen=True)
class MachineSnapshot:
    captured_at: str
    cpu_count: int | None
    load_1m: float | None
    memory_total_bytes: int | None
    memory_available_bytes: int | None
    disk_total_bytes: int
    disk_free_bytes: int
    network_received_bytes: int | None
    network_transmitted_bytes: int | None


@dataclass(frozen=True)
class CacheSnapshot:
    directory: str
    exists: bool
    file_count: int | None
    size_bytes: int | None


@dataclass
class ResourceMetrics:
    measurement_scope: str
    machine_before: MachineSnapshot
    cache_before: CacheSnapshot
    machine_after: MachineSnapshot | None = None
    cache_after: CacheSnapshot | None = None
    cache_state_before: str = "unknown"
    cache_size_change_bytes: int | None = None
    disk_free_change_bytes: int | None = None
    host_network_received_change_bytes: int | None = None
    host_network_transmitted_change_bytes: int | None = None
    peak_stage_rss_bytes: int | None = None


@dataclass
class ExperimentResult:
    schema_version: str
    experiment_id: str
    spec: ExperimentSpec
    outcome: Outcome
    started_at: str
    duration_seconds: float
    stages: list[StageResult] = field(default_factory=list)
    runtime: RuntimeEnvironment | None = None
    resources: ResourceMetrics | None = None
    installed_environment: list[str] = field(default_factory=list)
    wheel_artifacts: list[WheelArtifact] = field(default_factory=list)
    installed_wheel_artifacts: list[InstalledWheelArtifact] = field(default_factory=list)
    artifact_lock_format: str | None = None
    artifact_lock_sha256: str | None = None
    exception_type: str | None = None
    normalized_error: str | None = None
    retry_count: int = 0
    measured: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
