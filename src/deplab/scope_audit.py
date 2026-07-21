from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .pypi import PyPIClient


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
    if python_versions != ["3.10", "3.11", "3.12"]:
        raise ScopeAuditError("scope coverage_order must be Python 3.10, 3.11 and 3.12")
    if not packages:
        raise ScopeAuditError("scope draft contains no packages")

    client = client or PyPIClient()
    audited_packages: dict[str, Any] = {}
    eligible = 0
    releases = 0
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
            for python_version in python_versions:
                release = client.release(package_name, version, python_version)
                available = not release.yanked and any(
                    wheel.compatible and not wheel.yanked for wheel in release.wheels
                )
                coverage.append(available)
                eligible += int(available)
            versions.append({"version": version, "coverage": coverage})
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
            "Expanded package scope audited against official PyPI metadata for compatible, "
            "non-yanked Linux x86_64 wheels. False coverage is a scheduling exclusion, not "
            "a compatibility label."
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
    )
