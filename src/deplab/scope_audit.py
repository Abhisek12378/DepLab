from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .pypi import PyPIClient
from .wheels import requires_python_allows


class ScopeAuditError(ValueError):
    pass


@dataclass(frozen=True)
class ScopeAuditSummary:
    input: str
    output: str
    packages: int
    releases: int
    python_targets: int
    eligible_release_targets: int
    excluded_release_targets: int
    exclusion_counts: dict[str, int]


def audit_scope(
    input_path: Path,
    output_path: Path,
    client: PyPIClient | None = None,
) -> ScopeAuditSummary:
    try:
        draft = json.loads(input_path.read_text(encoding="utf-8"))
        python_versions = list(draft["coverage_order"])
        packages = dict(draft["packages"])
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise ScopeAuditError(f"invalid scope draft {input_path}: {exc}") from exc
    _validate_python_versions(python_versions)
    if not packages:
        raise ScopeAuditError("scope draft contains no packages")

    client = client or PyPIClient()
    audited_packages: dict[str, Any] = {}
    eligible = 0
    releases = 0
    exclusion_counts: Counter[str] = Counter()
    for package_name, package in packages.items():
        if not isinstance(package, dict) or not isinstance(package.get("versions"), list):
            raise ScopeAuditError(f"package {package_name!r} has no versions list")
        versions = []
        seen = set()
        for item in package["versions"]:
            version = str(item.get("version") if isinstance(item, dict) else item).strip()
            if not version or version in seen:
                raise ScopeAuditError(f"package {package_name!r} has a missing or duplicate version")
            seen.add(version)
            coverage = []
            coverage_details = []
            for python_version in python_versions:
                release = client.release(package_name, version, python_version)
                compatible_wheels = [
                    wheel
                    for wheel in release.wheels
                    if wheel.compatible and not wheel.yanked
                ]
                available = not release.yanked and bool(compatible_wheels)
                reason = _coverage_reason(release, python_version, available)
                coverage.append(available)
                coverage_details.append(
                    {
                        "python": python_version,
                        "eligible": available,
                        "reason": reason,
                        "compatible_wheels": len(compatible_wheels),
                    }
                )
                eligible += int(available)
                if not available:
                    exclusion_counts[reason] += 1
            versions.append(
                {
                    "version": version,
                    "coverage": coverage,
                    "coverage_details": coverage_details,
                }
            )
            releases += 1
        audited_packages[package_name] = {
            **{key: value for key, value in package.items() if key != "versions"},
            "versions": versions,
        }

    total_targets = releases * len(python_versions)
    audited = {
        "schema_version": "2.0.0",
        "audited_from": str(input_path),
        "coverage_order": python_versions,
        "description": (
            "Package scope audited against official PyPI metadata for compatible, non-yanked "
            "Linux x86_64 wheels. Every false coverage value includes a deterministic reason "
            "and remains a scheduling exclusion rather than a learned compatibility label."
        ),
        "packages": audited_packages,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(audited, indent=2) + "\n", encoding="utf-8")
    return ScopeAuditSummary(
        input=str(input_path),
        output=str(output_path),
        packages=len(packages),
        releases=releases,
        python_targets=total_targets,
        eligible_release_targets=eligible,
        excluded_release_targets=total_targets - eligible,
        exclusion_counts=dict(sorted(exclusion_counts.items())),
    )


def _validate_python_versions(python_versions: Any) -> None:
    if not isinstance(python_versions, list) or not python_versions:
        raise ScopeAuditError("scope coverage_order must be a non-empty list")
    if not all(isinstance(version, str) and re.fullmatch(r"3\.\d+", version) for version in python_versions):
        raise ScopeAuditError("scope coverage_order contains an invalid Python minor version")
    if len(set(python_versions)) != len(python_versions):
        raise ScopeAuditError("scope coverage_order contains duplicates")
    numeric = [tuple(int(part) for part in version.split(".")) for version in python_versions]
    if numeric != sorted(numeric):
        raise ScopeAuditError("scope coverage_order must be numerically sorted")
    if numeric[0] < (3, 8) or numeric[-1] > (3, 14):
        raise ScopeAuditError("large-dataset scope supports CPython 3.8 through 3.14")


def _coverage_reason(release: Any, python_version: str, available: bool) -> str:
    if available:
        return "eligible"
    if release.yanked:
        return "yanked_release"
    if not requires_python_allows(release.requires_python, python_version):
        return "requires_python_excluded"
    if not release.wheels:
        return "wheel_unavailable"
    if all(wheel.yanked for wheel in release.wheels):
        return "all_wheels_yanked"
    return "incompatible_wheel_tags"
