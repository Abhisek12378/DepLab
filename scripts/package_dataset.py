from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


DATASET_VERSION = "1.0.0"
DATASET_ID = f"deplab-systematic-v{DATASET_VERSION}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Package the audited DepLab dataset")
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--zip", type=Path, required=True)
    args = parser.parse_args()
    root = args.project_root.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    sources = {
        "experiments.jsonl": root / "outputs/systematic-main-full.jsonl",
        "package-catalog.jsonl": root / "outputs/package-catalog.jsonl",
        "matrix.json": root / "configs/systematic-matrix.json",
        "package-scope.json": root / "configs/package-scope.json",
        "pair-families.json": root / "configs/pair-families.json",
        "audit-summary.json": root / "outputs/systematic-main-audit-summary.json",
        "audit-report.md": root / "outputs/systematic-main-audit-report.md",
    }
    for destination, source in sources.items():
        if not source.exists():
            raise FileNotFoundError(f"required package input is missing: {source}")
        shutil.copyfile(source, output / destination)

    results = _read_jsonl(sources["experiments.jsonl"])
    catalog_rows = _read_jsonl(sources["package-catalog.jsonl"])
    matrix = _read_json(sources["matrix.json"])
    audit = _read_json(sources["audit-summary.json"])
    if audit.get("audit_status") != "pass" or audit.get("validation_errors"):
        raise ValueError("refusing to package a dataset that did not pass its audit")

    matrix_rows = matrix.get("experiments")
    if not isinstance(matrix_rows, list):
        raise ValueError("matrix has no experiments list")
    result_by_id = {str(row["experiment_id"]): row for row in results}
    catalog = {
        (
            _canonical_name(str(row["release"]["name"])),
            str(row["release"]["version"]),
            str(row["target"]["python_version"]),
        ): row["release"]
        for row in catalog_rows
    }

    features = []
    for order, matrix_row in enumerate(matrix_rows, 1):
        experiment_id = _experiment_id(matrix_row)
        result = result_by_id.get(experiment_id)
        if result is None:
            raise ValueError(f"audited result is missing for {experiment_id}")
        features.append(_feature_row(order, matrix_row, result, catalog))
    if len(features) != 646 or len({row["experiment_id"] for row in features}) != 646:
        raise ValueError("feature table is not a one-to-one transformation of the matrix")

    feature_path = output / "features.csv"
    frame = pd.DataFrame(features)
    frame.to_csv(feature_path, index=False, encoding="utf-8", lineterminator="\n")
    parquet_path = output / "features.parquet"
    frame.to_parquet(parquet_path, engine="pyarrow", compression="zstd", index=False)

    dictionary = _feature_dictionary(frame)
    dictionary_path = output / "feature-dictionary.csv"
    pd.DataFrame(dictionary).to_csv(
        dictionary_path, index=False, encoding="utf-8", lineterminator="\n"
    )

    manifest = _manifest(output, frame, audit, matrix, catalog_rows, dictionary)
    manifest_path = output / "dataset-manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    readme_path = output / "README.md"
    readme_path.write_text(_package_readme(manifest), encoding="utf-8")
    _write_checksums(output)

    args.zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in sorted(output.iterdir()):
            if path.is_file():
                archive.write(path, arcname=f"{DATASET_ID}/{path.name}")
    zip_hash = _sha256(args.zip)
    args.zip.with_suffix(args.zip.suffix + ".sha256").write_text(
        f"{zip_hash}  {args.zip.name}\n", encoding="utf-8"
    )

    print(
        json.dumps(
            {
                "dataset_id": DATASET_ID,
                "rows": len(frame),
                "feature_columns": len(frame.columns),
                "package_dir": str(output),
                "parquet_bytes": parquet_path.stat().st_size,
                "zip": str(args.zip),
                "zip_sha256": zip_hash,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _feature_row(
    order: int,
    matrix_row: dict[str, Any],
    result: dict[str, Any],
    catalog: dict[tuple[str, str, str], dict[str, Any]],
) -> dict[str, Any]:
    spec = result["spec"]
    pin_a = spec["package_a"]
    pin_b = spec["package_b"]
    python_version = str(spec["python_version"])
    release_a = catalog[(_canonical_name(pin_a["name"]), str(pin_a["version"]), python_version)]
    release_b = catalog[(_canonical_name(pin_b["name"]), str(pin_b["version"]), python_version)]
    wheel_a, wheel_b = result["wheel_artifacts"]
    version_a = _version_parts(str(pin_a["version"]))
    version_b = _version_parts(str(pin_b["version"]))
    python_parts = _version_parts(python_version)
    requirement_a_on_b = _direct_requirement(release_a.get("requires_dist", []), pin_b["name"])
    requirement_b_on_a = _direct_requirement(release_b.get("requires_dist", []), pin_a["name"])
    release_date_a = _parse_date(release_a.get("release_date"))
    release_date_b = _parse_date(release_b.get("release_date"))
    artifacts = result.get("installed_wheel_artifacts", [])
    resources = result.get("resources") or {}
    machine_before = resources.get("machine_before") or {}
    runtime = result.get("runtime") or {}
    outcome = str(result["outcome"])
    imports_succeeded = outcome in {"pass", "smoke_test_failure"}
    resolution_succeeded = outcome != "resolution_failure"

    return {
        "dataset_version": DATASET_VERSION,
        "matrix_order": order,
        "experiment_id": result["experiment_id"],
        "family": matrix_row["family"],
        "package_a_name": pin_a["name"],
        "package_a_version": pin_a["version"],
        "package_a_version_major": version_a[0],
        "package_a_version_minor": version_a[1],
        "package_a_version_patch": version_a[2],
        "package_b_name": pin_b["name"],
        "package_b_version": pin_b["version"],
        "package_b_version_major": version_b[0],
        "package_b_version_minor": version_b[1],
        "package_b_version_patch": version_b[2],
        "python_version": python_version,
        "python_major": python_parts[0],
        "python_minor": python_parts[1],
        "os": spec["os"],
        "architecture": spec["architecture"],
        "package_a_requires_python": release_a.get("requires_python"),
        "package_b_requires_python": release_b.get("requires_python"),
        "package_a_release_date": release_a.get("release_date"),
        "package_b_release_date": release_b.get("release_date"),
        "release_date_distance_days": abs((release_date_a - release_date_b).days)
        if release_date_a and release_date_b
        else None,
        "package_a_requires_dist_count": len(release_a.get("requires_dist", [])),
        "package_b_requires_dist_count": len(release_b.get("requires_dist", [])),
        "package_a_declares_package_b": bool(requirement_a_on_b),
        "package_b_declares_package_a": bool(requirement_b_on_a),
        "package_a_requirement_on_b": requirement_a_on_b,
        "package_b_requirement_on_a": requirement_b_on_a,
        "package_a_eligible_wheel_count": sum(
            bool(wheel.get("compatible")) for wheel in release_a.get("wheels", [])
        ),
        "package_b_eligible_wheel_count": sum(
            bool(wheel.get("compatible")) for wheel in release_b.get("wheels", [])
        ),
        "package_a_wheel_filename": wheel_a["filename"],
        "package_b_wheel_filename": wheel_b["filename"],
        "package_a_wheel_python_tag": wheel_a["python_tag"],
        "package_b_wheel_python_tag": wheel_b["python_tag"],
        "package_a_wheel_abi_tag": wheel_a["abi_tag"],
        "package_b_wheel_abi_tag": wheel_b["abi_tag"],
        "package_a_wheel_platform_tag": wheel_a["platform_tag"],
        "package_b_wheel_platform_tag": wheel_b["platform_tag"],
        "package_a_has_native_extensions": bool(wheel_a.get("has_native_extensions")),
        "package_b_has_native_extensions": bool(wheel_b.get("has_native_extensions")),
        "either_top_level_has_native_extensions": bool(
            wheel_a.get("has_native_extensions") or wheel_b.get("has_native_extensions")
        ),
        "top_level_wheel_bytes": int(wheel_a["size"] or 0) + int(wheel_b["size"] or 0),
        "outcome": outcome,
        "compatibility_label": "compatible" if outcome == "pass" else "incompatible",
        "is_compatible": outcome == "pass",
        "resolution_succeeded": resolution_succeeded,
        "imports_succeeded": imports_succeeded,
        "interoperability_smoke_succeeded": outcome == "pass",
        "failure_category": _failure_category(result),
        "exception_type": result.get("exception_type"),
        "installed_distribution_count": len(result.get("installed_environment", [])),
        "installed_wheel_artifact_count": len(artifacts),
        "transitive_distribution_count": max(0, len(artifacts) - 2),
        "installed_wheel_bytes": sum(int(artifact["size"]) for artifact in artifacts),
        "installed_native_wheel_count": sum(
            artifact.get("abi_tag") != "none" or artifact.get("platform_tag") != "any"
            for artifact in artifacts
        ),
        "duration_seconds": result.get("duration_seconds"),
        "peak_stage_rss_bytes": resources.get("peak_stage_rss_bytes"),
        "cache_state_before": resources.get("cache_state_before"),
        "cache_size_before_bytes": (resources.get("cache_before") or {}).get("size_bytes"),
        "cache_size_after_bytes": (resources.get("cache_after") or {}).get("size_bytes"),
        "host_network_received_change_bytes": resources.get("host_network_received_change_bytes"),
        "host_network_transmitted_change_bytes": resources.get("host_network_transmitted_change_bytes"),
        "host_cpu_count": machine_before.get("cpu_count"),
        "host_memory_total_bytes": machine_before.get("memory_total_bytes"),
        "runtime_python_version": runtime.get("python_version"),
        "runtime_kernel": runtime.get("kernel"),
        "runtime_libc": runtime.get("libc"),
        "runtime_uv_version": runtime.get("uv_version"),
        "artifact_lock_sha256": result.get("artifact_lock_sha256"),
        "normalized_error": result.get("normalized_error"),
    }


def _feature_dictionary(frame: pd.DataFrame) -> list[dict[str, str]]:
    labels = {
        "outcome",
        "compatibility_label",
        "is_compatible",
        "resolution_succeeded",
        "imports_succeeded",
        "interoperability_smoke_succeeded",
    }
    diagnostic = {"failure_category", "exception_type", "normalized_error"}
    evidence = {
        "installed_distribution_count",
        "installed_wheel_artifact_count",
        "transitive_distribution_count",
        "installed_wheel_bytes",
        "installed_native_wheel_count",
        "artifact_lock_sha256",
    }
    measurements = {
        "duration_seconds",
        "peak_stage_rss_bytes",
        "cache_state_before",
        "cache_size_before_bytes",
        "cache_size_after_bytes",
        "host_network_received_change_bytes",
        "host_network_transmitted_change_bytes",
        "host_cpu_count",
        "host_memory_total_bytes",
    }
    identifiers = {"dataset_version", "matrix_order", "experiment_id"}
    runtime_context = {
        "runtime_python_version",
        "runtime_kernel",
        "runtime_libc",
        "runtime_uv_version",
    }
    descriptions = {
        "family": "Registered package-pair interoperability family.",
        "outcome": "Measured final outcome; primary multiclass label.",
        "compatibility_label": "Binary label: compatible only when the full smoke test passed.",
        "is_compatible": "Boolean form of compatibility_label.",
        "resolution_succeeded": "Whether dependency resolution produced an installable lock.",
        "imports_succeeded": "Whether both imports completed before the smoke test.",
        "interoperability_smoke_succeeded": "Whether the registered pair-specific smoke test passed.",
        "failure_category": "Human-readable grouping derived from the measured error; exclude from training.",
        "normalized_error": "Normalized diagnostic text; exclude from training to prevent label leakage.",
        "artifact_lock_sha256": "Reproducibility checksum; identifier evidence, not a model feature.",
        "release_date_distance_days": "Absolute distance between the two top-level release dates.",
        "either_top_level_has_native_extensions": "True when either selected top-level wheel is platform/ABI specific.",
    }
    rows = []
    for column in frame.columns:
        if column in identifiers:
            role = "identifier"
        elif column in labels:
            role = "label"
        elif column in diagnostic:
            role = "diagnostic_do_not_train"
        elif column in evidence:
            role = "post_run_evidence_do_not_train"
        elif column in measurements:
            role = "measurement_do_not_train"
        elif column in runtime_context:
            role = "runtime_context"
        else:
            role = "input_feature"
        rows.append(
            {
                "column": column,
                "data_type": str(frame[column].dtype),
                "role": role,
                "description": descriptions.get(column, _default_description(column)),
            }
        )
    return rows


def _manifest(
    output: Path,
    frame: pd.DataFrame,
    audit: dict[str, Any],
    matrix: dict[str, Any],
    catalog_rows: list[dict[str, Any]],
    dictionary: list[dict[str, str]],
) -> dict[str, Any]:
    files = {}
    for name in (
        "experiments.jsonl",
        "package-catalog.jsonl",
        "matrix.json",
        "package-scope.json",
        "pair-families.json",
        "audit-summary.json",
        "audit-report.md",
        "features.csv",
        "features.parquet",
        "feature-dictionary.csv",
    ):
        path = output / name
        files[name] = {"bytes": path.stat().st_size, "sha256": _sha256(path)}
    roles = Counter(item["role"] for item in dictionary)
    return {
        "dataset_id": DATASET_ID,
        "version": DATASET_VERSION,
        "title": "DepLab systematic Python dependency compatibility dataset",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "description": (
            "Measured wheel-only compatibility outcomes for exact Python package-version pairs "
            "on glibc Linux x86_64 across Python 3.10, 3.11, and 3.12."
        ),
        "source_result_schema_version": "1.3.0",
        "audit_status": audit["audit_status"],
        "scope": {
            "experiments": len(frame),
            "families": int(frame["family"].nunique()),
            "packages": int(len(set(frame["package_a_name"]) | set(frame["package_b_name"]))),
            "python_versions": sorted(frame["python_version"].unique().tolist()),
            "platform": "linux_x86_64_glibc",
            "catalog_rows": len(catalog_rows),
            "wheel_eligible_matrix_rows": len(matrix["experiments"]),
        },
        "outcomes": {str(key): int(value) for key, value in frame["outcome"].value_counts().sort_index().items()},
        "labels": {
            "primary_multiclass": "outcome",
            "binary": "is_compatible",
            "compatible_definition": "Both imports and the registered interoperability smoke test passed.",
            "wheel_unavailable_policy": "Excluded during coverage filtering; never labeled incompatible.",
        },
        "feature_table": {
            "rows": len(frame),
            "columns": len(frame.columns),
            "role_counts": dict(sorted(roles.items())),
            "recommended_training_columns": [
                item["column"] for item in dictionary if item["role"] == "input_feature"
            ],
            "group_split_column": "family",
            "warning": (
                "Use family-grouped or version-blocked splits. Do not train on labels, diagnostics, "
                "post-run evidence, timings, resource measurements, hashes, or unique identifiers."
            ),
        },
        "runtime_identities": audit["runtime_identities"],
        "known_limitations": [
            "Only six package-pair families are represented; broad ecosystem generalization is not established.",
            "The matrix contains related Cartesian rows, so random row splits would leak nearby version patterns.",
            "Only glibc Linux x86_64 and CPython 3.10-3.12 are measured.",
            "Host-level network, cache, disk, and memory counters may include unrelated machine activity.",
            "Source distributions and source builds are intentionally excluded.",
        ],
        "files": files,
        "license_note": (
            "DepLab code is MIT licensed. This package redistributes measured metadata and URLs, "
            "not third-party wheel contents; upstream package licenses still apply."
        ),
    }


def _package_readme(manifest: dict[str, Any]) -> str:
    outcomes = manifest["outcomes"]
    return f"""# DepLab systematic dataset v{DATASET_VERSION}

This package contains {manifest['scope']['experiments']} measured compatibility experiments across {manifest['scope']['families']} package-pair families and CPython 3.10-3.12 on glibc Linux x86_64.

## Outcome counts

- pass: {outcomes.get('pass', 0)}
- resolution_failure: {outcomes.get('resolution_failure', 0)}
- import_failure: {outcomes.get('import_failure', 0)}
- smoke_test_failure: {outcomes.get('smoke_test_failure', 0)}

## Files

- `experiments.jsonl`: immutable, full-fidelity schema 1.3 experimental evidence.
- `features.csv`: flat processed table for inspection and modeling.
- `features.parquet`: typed, compressed equivalent of `features.csv`.
- `feature-dictionary.csv`: column roles, types, and descriptions.
- `package-catalog.jsonl`: audited PyPI release and wheel metadata.
- `matrix.json`, `package-scope.json`, `pair-families.json`: selection provenance.
- `audit-summary.json`, `audit-report.md`: validation evidence.
- `dataset-manifest.json`: version, labels, feature policy, limitations, and file hashes.
- `SHA256SUMS.txt`: integrity checksums for every packaged file except itself.

## Label policy

`pass` means both imports and the registered interoperability smoke test passed. Resolution, import, and smoke-test failures are measured negative outcomes. Wheel-unavailable rows were filtered as missing coverage and were not labeled incompatible.

## Modeling warning

Use `family` for grouped evaluation. Random row splitting is not reliable because nearby package-version combinations are strongly related. Train only with columns marked `input_feature` in `feature-dictionary.csv`; other roles contain labels, diagnostics, post-run evidence, measurements, runtime context, or identifiers.

Prefer Parquet for typed analysis. When reading CSV, explicitly load `python_version`, both package-version columns, and `experiment_id` as strings; otherwise some CSV readers may interpret Python `3.10` as the number `3.1`.

## Reproducibility

Verify this package with `SHA256SUMS.txt`. Each successful installation also records its PEP 751 lock hash and every exact installed wheel filename, URL, size, and SHA-256 inside `experiments.jsonl`.
"""


def _write_checksums(output: Path) -> None:
    checksum_path = output / "SHA256SUMS.txt"
    lines = [
        f"{_sha256(path)}  {path.name}"
        for path in sorted(output.iterdir())
        if path.is_file() and path.name != checksum_path.name
    ]
    checksum_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _failure_category(result: dict[str, Any]) -> str | None:
    outcome = result["outcome"]
    error = str(result.get("normalized_error") or "")
    if outcome == "pass":
        return None
    if outcome == "resolution_failure":
        return "declared_dependency_resolution_conflict"
    if "numpy.dtype size changed" in error:
        return "numpy_binary_abi_incompatibility"
    if "soft_unicode" in error:
        return "removed_markupsafe_soft_unicode_api"
    if "url_quote" in error:
        return "removed_werkzeug_url_quote_api"
    if outcome == "smoke_test_failure":
        return "interoperability_behavior_failure"
    return "other_failure"


def _direct_requirement(requirements: list[str], target: str) -> str | None:
    target_name = _canonical_name(str(target))
    for requirement in requirements:
        match = re.match(r"\s*([A-Za-z0-9_.-]+)", str(requirement))
        if match and _canonical_name(match.group(1)) == target_name:
            return str(requirement)
    return None


def _version_parts(version: str) -> tuple[int | None, int | None, int | None]:
    numbers = [int(value) for value in re.findall(r"\d+", version)[:3]]
    values: list[int | None] = numbers + [None, None, None]
    return values[0], values[1], values[2]


def _parse_date(value: Any) -> datetime | None:
    if not value:
        return None
    return datetime.fromisoformat(str(value).replace("Z", "+00:00"))


def _experiment_id(row: dict[str, Any]) -> str:
    package_a, version_a = str(row["package_a"]).split("==", 1)
    package_b, version_b = str(row["package_b"]).split("==", 1)
    raw = "|".join(
        (
            package_a.lower(),
            version_a,
            package_b.lower(),
            version_b,
            str(row["python"]),
            "linux",
            "x86_64",
        )
    )
    return hashlib.sha256(raw.encode()).hexdigest()[:20]


def _canonical_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _default_description(column: str) -> str:
    return column.replace("_", " ").capitalize() + "."


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path} contains a non-object row")
            rows.append(value)
    return rows


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
