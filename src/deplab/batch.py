from __future__ import annotations

import json
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .models import ExperimentResult, ExperimentSpec, PackagePin, utc_now
from .pypi import PyPIClient
from .runner import ExperimentRunner
from .storage import append_jsonl, completed_ids


class ManifestError(ValueError):
    pass


SUPPORTED_PYTHON_VERSIONS = {
    "3.8",
    "3.9",
    "3.10",
    "3.11",
    "3.12",
    "3.13",
    "3.14",
}


@dataclass(frozen=True)
class BatchSummary:
    manifest: str
    requested: int
    scheduled: int
    skipped_completed: int
    outcome_counts: dict[str, int]
    duration_seconds: float
    workers: int


def load_manifest(path: Path) -> list[ExperimentSpec]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ManifestError(f"cannot read manifest {path}: {exc}") from exc
    experiments = payload.get("experiments") if isinstance(payload, dict) else None
    if not isinstance(experiments, list) or not experiments:
        raise ManifestError("manifest must contain a non-empty 'experiments' list")

    specs: list[ExperimentSpec] = []
    seen: set[str] = set()
    for index, row in enumerate(experiments):
        if not isinstance(row, dict):
            raise ManifestError(f"experiment {index} must be an object")
        try:
            package_a = _manifest_pin(row["package_a"])
            package_b = _manifest_pin(row["package_b"])
            python_version = str(row["python"])
        except KeyError as exc:
            raise ManifestError(f"experiment {index} is missing {exc.args[0]!r}") from exc
        if python_version not in SUPPORTED_PYTHON_VERSIONS:
            raise ManifestError(f"experiment {index} has unsupported Python {python_version!r}")
        if row.get("platform", "linux_x86_64") != "linux_x86_64":
            raise ManifestError(f"experiment {index} must target linux_x86_64")
        if package_a.name.lower() == package_b.name.lower():
            raise ManifestError(f"experiment {index} must contain two different packages")
        spec = ExperimentSpec(package_a, package_b, python_version)
        if spec.experiment_id in seen:
            raise ManifestError(f"experiment {index} duplicates experiment ID {spec.experiment_id}")
        seen.add(spec.experiment_id)
        specs.append(spec)
    return specs


def run_batch(
    manifest_path: Path,
    output_path: Path,
    runner: ExperimentRunner,
    client: PyPIClient | None = None,
    workers: int = 1,
) -> BatchSummary:
    if workers < 1:
        raise ValueError("workers must be at least 1")
    specs = load_manifest(manifest_path)
    already_done = completed_ids(output_path)
    pending = [spec for spec in specs if spec.experiment_id not in already_done]
    client = client or PyPIClient()
    started = time.monotonic()
    outcomes: Counter[str] = Counter()

    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="deplab") as pool:
        future_to_spec = {
            pool.submit(_run_one, spec, runner, client): spec for spec in pending
        }
        for future in as_completed(future_to_spec):
            spec = future_to_spec[future]
            try:
                result = future.result()
            except Exception as exc:  # Preserve the batch and make unexpected worker failures auditable.
                result = ExperimentResult(
                    schema_version="1.3.0",
                    experiment_id=spec.experiment_id,
                    spec=spec,
                    outcome="infrastructure_failure",
                    started_at=utc_now(),
                    duration_seconds=0.0,
                    exception_type=type(exc).__name__,
                    normalized_error=str(exc)[:2000],
                    measured=False,
                )
            append_jsonl(output_path, result.to_dict())
            outcomes[result.outcome] += 1

    return BatchSummary(
        manifest=str(manifest_path),
        requested=len(specs),
        scheduled=len(pending),
        skipped_completed=len(specs) - len(pending),
        outcome_counts=dict(sorted(outcomes.items())),
        duration_seconds=time.monotonic() - started,
        workers=workers,
    )


def _run_one(
    spec: ExperimentSpec, runner: ExperimentRunner, client: PyPIClient
) -> ExperimentResult:
    release_a = client.release(
        spec.package_a.name, spec.package_a.version, spec.python_version
    )
    release_b = client.release(
        spec.package_b.name, spec.package_b.version, spec.python_version
    )
    result = runner.run(spec, release_a, release_b)
    result.schema_version = "1.3.0"
    return result


def _manifest_pin(value: Any) -> PackagePin:
    if not isinstance(value, str) or "==" not in value:
        raise ManifestError(f"package pin must use NAME==VERSION, got {value!r}")
    name, version = value.split("==", 1)
    if not name or not version:
        raise ManifestError(f"package pin must use NAME==VERSION, got {value!r}")
    return PackagePin(name=name, version=version)
