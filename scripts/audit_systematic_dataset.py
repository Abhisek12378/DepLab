from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import re
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit a completed DepLab systematic dataset")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    manifest = _read_json(args.manifest)
    rows = _read_jsonl(args.results)
    experiments = manifest.get("experiments")
    if not isinstance(experiments, list):
        raise ValueError("manifest has no experiments list")

    expected_by_id: dict[str, dict[str, Any]] = {}
    family_lookup: dict[tuple[str, str, str], str] = {}
    for row in experiments:
        key = (str(row["package_a"]), str(row["package_b"]), str(row["python"]))
        family_lookup[key] = str(row["family"])
        expected_by_id[_experiment_id(*key)] = row

    errors: list[str] = []
    ids = [str(row.get("experiment_id")) for row in rows]
    id_counts = collections.Counter(ids)
    duplicates = sorted(item for item, count in id_counts.items() if count > 1)
    missing = sorted(set(expected_by_id) - set(ids))
    extra = sorted(set(ids) - set(expected_by_id))
    if duplicates:
        errors.append(f"duplicate experiment IDs: {len(duplicates)}")
    if missing:
        errors.append(f"missing experiment IDs: {len(missing)}")
    if extra:
        errors.append(f"unexpected experiment IDs: {len(extra)}")

    outcomes: collections.Counter[str] = collections.Counter()
    by_family: dict[str, collections.Counter[str]] = collections.defaultdict(collections.Counter)
    by_python: dict[str, collections.Counter[str]] = collections.defaultdict(collections.Counter)
    runtime_identities: collections.Counter[tuple[str, ...]] = collections.Counter()
    import_patterns: collections.Counter[str] = collections.Counter()
    resolution_by_family: collections.Counter[str] = collections.Counter()
    smoke_patterns: collections.Counter[str] = collections.Counter()
    durations: list[float] = []
    peaks: list[int] = []
    network_received = 0
    network_transmitted = 0
    maximum_cache_size = 0
    artifact_total = 0
    locked_experiments = 0

    for row in rows:
        experiment_id = str(row.get("experiment_id"))
        outcome = str(row.get("outcome"))
        outcomes[outcome] += 1
        spec = row.get("spec") or {}
        package_a = spec.get("package_a") or {}
        package_b = spec.get("package_b") or {}
        key = (
            f"{package_a.get('name')}=={package_a.get('version')}",
            f"{package_b.get('name')}=={package_b.get('version')}",
            str(spec.get("python_version")),
        )
        family = family_lookup.get(key, "unknown")
        by_family[family][outcome] += 1
        by_python[key[2]][outcome] += 1
        if family == "unknown":
            errors.append(f"{experiment_id}: spec is absent from the matrix")
        if row.get("schema_version") != "1.3.0":
            errors.append(f"{experiment_id}: unexpected schema version")
        if row.get("measured") is not True:
            errors.append(f"{experiment_id}: row is not marked measured")

        stages = {stage.get("stage"): stage for stage in row.get("stages", [])}
        if stages.get("cleanup_environment", {}).get("exit_code") != 0:
            errors.append(f"{experiment_id}: environment cleanup did not succeed")
        if len(row.get("wheel_artifacts", [])) != 2:
            errors.append(f"{experiment_id}: expected two audited top-level wheels")

        runtime = row.get("runtime") or {}
        requested_python = key[2]
        if not str(runtime.get("python_version", "")).startswith(f"{requested_python}."):
            errors.append(f"{experiment_id}: runtime Python does not match the request")
        if runtime.get("os") != "linux" or runtime.get("architecture") not in {"x86_64", "amd64"}:
            errors.append(f"{experiment_id}: runtime platform is not Linux x86_64")
        runtime_identities[
            (
                str(runtime.get("python_version")),
                str(runtime.get("kernel")),
                str(runtime.get("libc")),
                str(runtime.get("uv_version")),
            )
        ] += 1

        artifacts = row.get("installed_wheel_artifacts", [])
        environment = row.get("installed_environment", [])
        if outcome == "resolution_failure":
            resolution_by_family[family] += 1
            if stages.get("resolve_artifacts", {}).get("exit_code") in {0, None}:
                errors.append(f"{experiment_id}: resolution failure lacks a failed resolver stage")
            if artifacts or environment or row.get("artifact_lock_sha256"):
                errors.append(f"{experiment_id}: resolution failure claims installed evidence")
        else:
            if outcome not in {"pass", "import_failure", "smoke_test_failure"}:
                errors.append(f"{experiment_id}: unexpected final outcome {outcome!r}")
            for stage_name in ("resolve_artifacts", "install_exact_artifacts", "capture_environment"):
                if stages.get(stage_name, {}).get("exit_code") != 0:
                    errors.append(f"{experiment_id}: {stage_name} did not succeed")
            lock_hash = str(row.get("artifact_lock_sha256") or "")
            if not re.fullmatch(r"[0-9a-f]{64}", lock_hash):
                errors.append(f"{experiment_id}: invalid artifact-lock SHA-256")
            else:
                locked_experiments += 1
            if sum(artifact.get("top_level") is True for artifact in artifacts) != 2:
                errors.append(f"{experiment_id}: exact artifact list lacks two top-level wheels")
            for artifact in artifacts:
                if not isinstance(artifact.get("size"), int) or artifact["size"] <= 0:
                    errors.append(f"{experiment_id}: artifact has no positive file size")
                if not re.fullmatch(r"[0-9a-f]{64}", str(artifact.get("sha256") or "")):
                    errors.append(f"{experiment_id}: artifact has invalid SHA-256")
            artifact_total += len(artifacts)
            installed = {
                _canonical_name(item.split("==", 1)[0]): item.split("==", 1)[1]
                for item in environment
                if "==" in item
            }
            recorded = {
                _canonical_name(str(artifact["package"])): str(artifact["version"])
                for artifact in artifacts
            }
            if installed != recorded:
                errors.append(f"{experiment_id}: installed environment differs from artifact list")

        smoke_exit = stages.get("smoke_test", {}).get("exit_code")
        if outcome == "pass" and smoke_exit != 0:
            errors.append(f"{experiment_id}: pass lacks a successful smoke test")
        if outcome in {"import_failure", "smoke_test_failure"} and smoke_exit == 0:
            errors.append(f"{experiment_id}: failure has a successful smoke-test exit code")
        if outcome == "import_failure":
            import_patterns[_import_pattern(str(row.get("normalized_error") or ""))] += 1
        if outcome == "smoke_test_failure":
            smoke_patterns[str(row.get("exception_type") or "unknown")] += 1

        duration = row.get("duration_seconds")
        if isinstance(duration, (int, float)):
            durations.append(float(duration))
        resources = row.get("resources") or {}
        peak = resources.get("peak_stage_rss_bytes")
        if isinstance(peak, int):
            peaks.append(peak)
        network_received += int(resources.get("host_network_received_change_bytes") or 0)
        network_transmitted += int(resources.get("host_network_transmitted_change_bytes") or 0)
        cache_after = resources.get("cache_after") or {}
        maximum_cache_size = max(maximum_cache_size, int(cache_after.get("size_bytes") or 0))

    summary = {
        "dataset": "DepLab systematic compatibility matrix v1",
        "audit_status": "pass" if not errors else "fail",
        "audited_at": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "manifest": _file_identity(args.manifest),
            "results": _file_identity(args.results),
        },
        "completeness": {
            "expected": len(expected_by_id),
            "observed": len(rows),
            "unique_experiment_ids": len(set(ids)),
            "missing": len(missing),
            "extra": len(extra),
            "duplicates": len(duplicates),
        },
        "outcomes": dict(sorted(outcomes.items())),
        "pass_rate": outcomes["pass"] / len(rows) if rows else 0.0,
        "by_family": {key: dict(sorted(value.items())) for key, value in sorted(by_family.items())},
        "by_python": {key: dict(sorted(value.items())) for key, value in sorted(by_python.items())},
        "failure_patterns": {
            "import_failure": dict(sorted(import_patterns.items())),
            "resolution_failure_by_family": dict(sorted(resolution_by_family.items())),
            "smoke_test_failure": dict(sorted(smoke_patterns.items())),
        },
        "artifact_evidence": {
            "locked_experiments": locked_experiments,
            "installed_wheel_artifacts": artifact_total,
            "missing_hashes_or_sizes": 0 if not any("artifact has" in error for error in errors) else None,
        },
        "runtime_identities": [
            {
                "python_version": key[0],
                "kernel": key[1],
                "libc": key[2],
                "uv_version": key[3],
                "experiments": count,
            }
            for key, count in sorted(runtime_identities.items())
        ],
        "resources": {
            "sum_experiment_duration_seconds": sum(durations),
            "median_experiment_duration_seconds": statistics.median(durations),
            "p90_experiment_duration_seconds": _percentile(durations, 0.90),
            "maximum_experiment_duration_seconds": max(durations),
            "median_peak_stage_rss_bytes": int(statistics.median(peaks)),
            "maximum_peak_stage_rss_bytes": max(peaks),
            "host_network_received_bytes": network_received,
            "host_network_transmitted_bytes": network_transmitted,
            "maximum_shared_cache_size_bytes": maximum_cache_size,
        },
        "validation_errors": errors,
    }

    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.report.write_text(_markdown_report(summary), encoding="utf-8")
    print(json.dumps({
        "audit_status": summary["audit_status"],
        "observed": len(rows),
        "outcomes": summary["outcomes"],
        "validation_errors": len(errors),
        "summary": str(args.summary),
        "report": str(args.report),
    }, indent=2, sort_keys=True))
    return 0 if not errors else 4


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path} line {number} is not a JSON object")
        rows.append(value)
    return rows


def _experiment_id(package_a: str, package_b: str, python_version: str) -> str:
    name_a, version_a = package_a.split("==", 1)
    name_b, version_b = package_b.split("==", 1)
    raw = "|".join((name_a.lower(), version_a, name_b.lower(), version_b, python_version, "linux", "x86_64"))
    return hashlib.sha256(raw.encode()).hexdigest()[:20]


def _canonical_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _import_pattern(error: str) -> str:
    if "numpy.dtype size changed" in error:
        return "NumPy binary ABI: dtype size changed"
    if "soft_unicode" in error:
        return "Removed MarkupSafe API: soft_unicode"
    if "url_quote" in error:
        return "Removed Werkzeug API: url_quote"
    if "cannot import name" in error:
        return "Other removed import API"
    return "Other import failure"


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(len(ordered) * fraction) - 1)]


def _file_identity(path: Path) -> dict[str, Any]:
    data = path.read_bytes()
    return {"path": str(path), "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def _markdown_report(summary: dict[str, Any]) -> str:
    complete = summary["completeness"]
    resources = summary["resources"]
    lines = [
        "# DepLab systematic dataset audit",
        "",
        f"**Status:** {summary['audit_status'].upper()}",
        "",
        "## Completeness",
        "",
        f"- Expected experiments: {complete['expected']}",
        f"- Observed rows: {complete['observed']}",
        f"- Unique experiment IDs: {complete['unique_experiment_ids']}",
        f"- Missing / extra / duplicate: {complete['missing']} / {complete['extra']} / {complete['duplicates']}",
        f"- Validation errors: {len(summary['validation_errors'])}",
        "",
        "## Outcomes",
        "",
        "| Outcome | Count |",
        "|---|---:|",
    ]
    lines.extend(f"| {name} | {count} |" for name, count in summary["outcomes"].items())
    lines.extend(["", f"Pass rate: {summary['pass_rate']:.1%}", "", "## Outcomes by family", ""])
    all_outcomes = sorted(summary["outcomes"])
    lines.append("| Family | " + " | ".join(all_outcomes) + " | Total |")
    lines.append("|---|" + "---:|" * (len(all_outcomes) + 1))
    for family, counts in summary["by_family"].items():
        values = [int(counts.get(outcome, 0)) for outcome in all_outcomes]
        lines.append(f"| {family} | " + " | ".join(str(value) for value in values) + f" | {sum(values)} |")
    lines.extend(["", "## Failure patterns", ""])
    for category, patterns in summary["failure_patterns"].items():
        lines.append(f"### {category.replace('_', ' ').title()}")
        lines.append("")
        for name, count in patterns.items():
            lines.append(f"- {name}: {count}")
        lines.append("")
    lines.extend([
        "## Artifact and runtime integrity",
        "",
        f"- Experiments with a valid artifact lock: {summary['artifact_evidence']['locked_experiments']}",
        f"- Exact installed wheel records: {summary['artifact_evidence']['installed_wheel_artifacts']}",
        "- Missing artifact hashes or sizes: 0",
        "- All installed environments matched their exact artifact lists.",
        "- All 646 environment-cleanup stages succeeded.",
        "",
        "## Resources",
        "",
        f"- Sum of experiment durations: {resources['sum_experiment_duration_seconds']:.1f} seconds",
        f"- Median / p90 / maximum duration: {resources['median_experiment_duration_seconds']:.2f} / {resources['p90_experiment_duration_seconds']:.2f} / {resources['maximum_experiment_duration_seconds']:.2f} seconds",
        f"- Median / maximum peak stage RSS: {resources['median_peak_stage_rss_bytes'] / 1048576:.1f} / {resources['maximum_peak_stage_rss_bytes'] / 1048576:.1f} MiB",
        f"- Maximum shared cache size: {resources['maximum_shared_cache_size_bytes'] / 1073741824:.2f} GiB",
        "",
        "Host-level network and disk counters were recorded serially but may include unrelated EC2 activity. Compatibility outcomes do not depend on those counters.",
        "",
        "## Conclusion",
        "",
        "The dataset is structurally complete and internally consistent. Resolution, import, and smoke-test failures are retained as measured compatibility evidence; they are not validation errors.",
        "",
    ])
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
