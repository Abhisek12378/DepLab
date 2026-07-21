from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .batch import load_manifest


class MatrixError(ValueError):
    pass


@dataclass(frozen=True)
class MatrixSummary:
    scope: str
    pairs: str
    output: str
    families: int
    experiments: int
    excluded_for_wheel_coverage: int


def generate_matrix(scope_path: Path, pairs_path: Path, output_path: Path) -> MatrixSummary:
    scope = _read_object(scope_path, "scope")
    pairs = _read_object(pairs_path, "pair definition")
    coverage_order = scope.get("coverage_order")
    packages = scope.get("packages")
    families = pairs.get("families")
    python_versions = pairs.get("python_versions", coverage_order)
    platform = str(pairs.get("platform", "linux_x86_64"))
    if not isinstance(coverage_order, list) or not all(
        isinstance(item, str) for item in coverage_order
    ):
        raise MatrixError("scope coverage_order must be a list of Python versions")
    if not isinstance(packages, dict) or not packages:
        raise MatrixError("scope must contain packages")
    if not isinstance(families, list) or not families:
        raise MatrixError("pair definition must contain families")
    if not isinstance(python_versions, list) or not python_versions:
        raise MatrixError("pair definition must contain Python versions")
    if any(version not in coverage_order for version in python_versions):
        raise MatrixError("pair definition includes Python outside the audited scope")
    if platform != "linux_x86_64":
        raise MatrixError("systematic matrix currently supports only linux_x86_64")

    experiments: list[dict[str, str]] = []
    excluded = 0
    seen_family_names: set[str] = set()
    for index, family in enumerate(families):
        if not isinstance(family, dict):
            raise MatrixError(f"family {index} must be an object")
        family_name = str(family.get("name") or "").strip()
        package_a = str(family.get("package_a") or "").strip()
        package_b = str(family.get("package_b") or "").strip()
        if not family_name or not package_a or not package_b:
            raise MatrixError(f"family {index} lacks name, package_a, or package_b")
        if family_name in seen_family_names:
            raise MatrixError(f"duplicate family name {family_name!r}")
        seen_family_names.add(family_name)
        if package_a == package_b:
            raise MatrixError(f"family {family_name!r} repeats the same package")
        if package_a not in packages or package_b not in packages:
            raise MatrixError(f"family {family_name!r} refers to a package outside the scope")

        versions_a = _versions(packages[package_a], package_a, len(coverage_order))
        versions_b = _versions(packages[package_b], package_b, len(coverage_order))
        for python_version in python_versions:
            python_index = coverage_order.index(python_version)
            for version_a, coverage_a in versions_a:
                for version_b, coverage_b in versions_b:
                    if not coverage_a[python_index] or not coverage_b[python_index]:
                        excluded += 1
                        continue
                    experiments.append(
                        {
                            "family": family_name,
                            "package_a": f"{package_a}=={version_a}",
                            "package_b": f"{package_b}=={version_b}",
                            "python": python_version,
                            "platform": platform,
                            "selection_method": "systematic_cartesian_wheel_eligible",
                        }
                    )

    payload = {
        "schema_version": "1.0.0",
        "description": (
            "Systematic Cartesian matrix across every audited package-pair version and "
            "Python combination where both top-level wheels are available."
        ),
        "scope": str(scope_path),
        "pair_definitions": str(pairs_path),
        "experiments": experiments,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    # Reuse the production manifest validator before reporting success.
    load_manifest(output_path)
    return MatrixSummary(
        scope=str(scope_path),
        pairs=str(pairs_path),
        output=str(output_path),
        families=len(families),
        experiments=len(experiments),
        excluded_for_wheel_coverage=excluded,
    )


def _read_object(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MatrixError(f"cannot read {label} {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise MatrixError(f"{label} must be a JSON object")
    return payload


def _versions(package: Any, name: str, coverage_length: int) -> list[tuple[str, list[bool]]]:
    if not isinstance(package, dict) or not isinstance(package.get("versions"), list):
        raise MatrixError(f"scope package {name!r} has no versions list")
    result = []
    seen: set[str] = set()
    for item in package["versions"]:
        if not isinstance(item, dict):
            raise MatrixError(f"scope package {name!r} has an invalid version row")
        version = str(item.get("version") or "").strip()
        coverage = item.get("coverage")
        if not version or version in seen:
            raise MatrixError(f"scope package {name!r} has a missing or duplicate version")
        if not isinstance(coverage, list) or len(coverage) != coverage_length or not all(
            isinstance(value, bool) for value in coverage
        ):
            raise MatrixError(f"scope package {name!r} version {version} has invalid coverage")
        seen.add(version)
        result.append((version, coverage))
    return result
