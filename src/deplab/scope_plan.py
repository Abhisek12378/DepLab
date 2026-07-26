from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from .pypi import PyPIClient
from .wheels import requires_python_allows, wheel_is_compatible


class ScopePlanError(ValueError):
    pass


SPLITS = ("development", "validation", "final_test")
_STABLE_VERSION = re.compile(
    r"^(?P<release>[0-9]+(?:\.[0-9]+){1,3})(?:\.post(?P<post>[0-9]+))?$"
)


@dataclass(frozen=True)
class ScopePlanSummary:
    plan: str
    output: str
    packages: int
    families: int
    python_versions: int
    target_versions_per_package: int
    maximum_cartesian_experiments: int
    split_packages: dict[str, int]
    split_families: dict[str, int]


@dataclass(frozen=True)
class ScopeDraftSummary:
    plan: str
    output: str
    packages: int
    releases: int
    reused_releases: int
    discovered_releases: int
    python_versions: int


def read_and_validate_plan(plan_path: Path) -> tuple[dict[str, Any], ScopePlanSummary]:
    plan = _read_object(plan_path, "large-dataset plan")
    coverage_order = plan.get("coverage_order")
    packages = plan.get("packages")
    families = plan.get("families")
    target_versions = plan.get("target_versions_per_package")
    default_minimum = plan.get("minimum_versions_per_package")
    if coverage_order != ["3.8", "3.9", "3.10", "3.11", "3.12", "3.13", "3.14"]:
        raise ScopePlanError("coverage_order must contain Python 3.8 through 3.14")
    if not isinstance(packages, dict) or not packages:
        raise ScopePlanError("plan must contain packages")
    if not isinstance(families, list) or not families:
        raise ScopePlanError("plan must contain families")
    if not isinstance(target_versions, int) or target_versions < 2:
        raise ScopePlanError("target_versions_per_package must be an integer of at least 2")
    if (
        not isinstance(default_minimum, int)
        or default_minimum < 2
        or default_minimum > target_versions
    ):
        raise ScopePlanError(
            "minimum_versions_per_package must be between 2 and "
            "target_versions_per_package"
        )

    package_splits = _validate_packages(packages, default_minimum, target_versions)
    family_counts = _validate_families(families, package_splits)
    used_packages = {
        family[key] for family in families for key in ("package_a", "package_b")
    }
    unused = sorted(set(packages) - used_packages)
    if unused:
        raise ScopePlanError(f"packages are not used by a family: {', '.join(unused)}")

    package_counts = {
        split: sum(value == split for value in package_splits.values()) for split in SPLITS
    }
    maximum = len(families) * target_versions * target_versions * len(coverage_order)
    summary = ScopePlanSummary(
        plan=str(plan_path),
        output="",
        packages=len(packages),
        families=len(families),
        python_versions=len(coverage_order),
        target_versions_per_package=target_versions,
        maximum_cartesian_experiments=maximum,
        split_packages=package_counts,
        split_families=family_counts,
    )
    return plan, summary


def write_pair_definitions(plan_path: Path, output_directory: Path) -> ScopePlanSummary:
    plan, summary = read_and_validate_plan(plan_path)
    output_directory.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        payload = {
            "schema_version": "3.0.0",
            "description": _split_description(split),
            "platform": plan["target_platform"],
            "python_versions": plan["coverage_order"],
            "families": [
                {
                    "name": family["name"],
                    "package_a": family["package_a"],
                    "package_b": family["package_b"],
                }
                for family in plan["families"]
                if family["split"] == split
            ],
        }
        path = output_directory / f"large-{split.replace('_', '-')}-pairs-v3.0.0.json"
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return ScopePlanSummary(
        **{**asdict(summary), "output": str(output_directory)}
    )


def build_scope_draft(
    plan_path: Path,
    output_path: Path,
    client: PyPIClient | None = None,
    root: Path | None = None,
) -> ScopeDraftSummary:
    plan, _ = read_and_validate_plan(plan_path)
    root = root or plan_path.resolve().parents[1]
    existing = _load_reused_versions(plan, root)
    client = client or PyPIClient()
    target = int(plan["target_versions_per_package"])
    history_limit = int(plan["version_selection"]["candidate_history_limit"])
    cutoff = _parse_datetime(str(plan["release_cutoff"]))
    packages: dict[str, Any] = {}
    reused_count = 0
    discovered_count = 0

    for name, definition in plan["packages"].items():
        minimum = int(
            definition.get(
                "minimum_versions",
                plan["minimum_versions_per_package"],
            )
        )
        project = client.project(name)
        all_candidates = eligible_project_versions(
            project,
            plan["coverage_order"],
            cutoff,
            history_limit=10_000,
        )
        reused = [
            version for version in existing.get(name, []) if version in all_candidates
        ]
        recent_candidates = all_candidates[-history_limit:]
        selection_pool = sorted(
            set(recent_candidates + reused),
            key=_version_key,
        )
        selected = select_versions(selection_pool, target, reused)
        if len(selected) < minimum:
            raise ScopePlanError(
                f"package {name!r} has only {len(selected)} eligible stable releases; "
                f"minimum is {minimum}"
            )
        reused_count += len(set(selected) & set(reused))
        discovered_count += len(set(selected) - set(reused))
        packages[name] = {
            **definition,
            "versions": selected,
            "selection": {
                "eligible_candidates": len(all_candidates),
                "reused_versions": len(set(selected) & set(reused)),
                "discovered_versions": len(set(selected) - set(reused)),
            },
        }

    draft = {
        "schema_version": "3.0.0-draft",
        "plan_id": plan["plan_id"],
        "planned_from": str(plan_path),
        "release_cutoff": plan["release_cutoff"],
        "coverage_order": plan["coverage_order"],
        "description": (
            "Candidate releases for the DepLab large dataset. Exact wheel coverage "
            "and deterministic exclusion reasons must be added by scope-audit before "
            "matrix generation."
        ),
        "packages": packages,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(draft, indent=2) + "\n", encoding="utf-8")
    return ScopeDraftSummary(
        plan=str(plan_path),
        output=str(output_path),
        packages=len(packages),
        releases=sum(len(package["versions"]) for package in packages.values()),
        reused_releases=reused_count,
        discovered_releases=discovered_count,
        python_versions=len(plan["coverage_order"]),
    )


def eligible_project_versions(
    project: dict[str, Any],
    python_versions: list[str],
    release_cutoff: datetime,
    history_limit: int,
) -> list[str]:
    releases = project.get("releases")
    if not isinstance(releases, dict):
        raise ScopePlanError("PyPI project payload has no releases object")
    candidates = []
    for version, files in releases.items():
        if not _STABLE_VERSION.fullmatch(str(version)) or not isinstance(files, list):
            continue
        if _release_has_target_wheel(files, python_versions, release_cutoff):
            candidates.append(str(version))
    ordered = sorted(set(candidates), key=_version_key)
    return ordered[-history_limit:]


def select_versions(
    candidates: list[str],
    target: int,
    preserved: list[str] | None = None,
) -> list[str]:
    ordered = sorted(set(candidates), key=_version_key)
    preserved_ordered = [
        version for version in sorted(set(preserved or []), key=_version_key) if version in ordered
    ]
    if len(preserved_ordered) > target:
        raise ScopePlanError("preserved releases exceed target_versions_per_package")
    if len(preserved_ordered) == target:
        return preserved_ordered

    remaining = [version for version in ordered if version not in preserved_ordered]
    needed = target - len(preserved_ordered)
    if preserved_ordered:
        additions = list(reversed(remaining))[:needed]
    else:
        additions = _spread(remaining, needed)
    return sorted(set(preserved_ordered + additions), key=_version_key)


def _validate_packages(
    packages: dict[str, Any],
    default_minimum: int,
    target_versions: int,
) -> dict[str, str]:
    result = {}
    for name, definition in packages.items():
        if not isinstance(definition, dict):
            raise ScopePlanError(f"package {name!r} must be an object")
        split = definition.get("split")
        if split not in SPLITS:
            raise ScopePlanError(f"package {name!r} has invalid split {split!r}")
        if not str(definition.get("role") or "").strip():
            raise ScopePlanError(f"package {name!r} has no role")
        minimum = definition.get("minimum_versions", default_minimum)
        if (
            not isinstance(minimum, int)
            or minimum < 2
            or minimum > target_versions
        ):
            raise ScopePlanError(
                f"package {name!r} minimum_versions must be between 2 and "
                "target_versions_per_package"
            )
        result[name] = str(split)
    return result


def _validate_families(
    families: list[Any],
    package_splits: dict[str, str],
) -> dict[str, int]:
    names: set[str] = set()
    pairs: set[frozenset[str]] = set()
    counts = {split: 0 for split in SPLITS}
    for index, family in enumerate(families):
        if not isinstance(family, dict):
            raise ScopePlanError(f"family {index} must be an object")
        name = str(family.get("name") or "").strip()
        package_a = str(family.get("package_a") or "").strip()
        package_b = str(family.get("package_b") or "").strip()
        split = family.get("split")
        if not name or name in names:
            raise ScopePlanError(f"family {index} has a missing or duplicate name")
        if package_a not in package_splits or package_b not in package_splits:
            raise ScopePlanError(f"family {name!r} refers to a package outside the plan")
        if package_a == package_b:
            raise ScopePlanError(f"family {name!r} repeats the same package")
        pair = frozenset((package_a, package_b))
        if pair in pairs:
            raise ScopePlanError(f"family {name!r} duplicates an unordered package pair")
        if split not in SPLITS:
            raise ScopePlanError(f"family {name!r} has invalid split {split!r}")
        if package_splits[package_a] != split or package_splits[package_b] != split:
            raise ScopePlanError(f"family {name!r} crosses package splits")
        names.add(name)
        pairs.add(pair)
        counts[str(split)] += 1
    return counts


def _load_reused_versions(plan: dict[str, Any], root: Path) -> dict[str, list[str]]:
    relative = plan.get("reuse_versions_from")
    if not relative:
        return {}
    payload = _read_object(root / str(relative), "reused scope")
    result = {}
    for name, package in dict(payload.get("packages") or {}).items():
        versions = package.get("versions") if isinstance(package, dict) else None
        if not isinstance(versions, list):
            continue
        result[name] = [
            str(item["version"])
            for item in versions
            if isinstance(item, dict) and item.get("version")
        ]
    return result


def _release_has_target_wheel(
    files: list[Any],
    python_versions: list[str],
    release_cutoff: datetime,
) -> bool:
    for file in files:
        if not isinstance(file, dict) or file.get("yanked"):
            continue
        filename = str(file.get("filename") or "")
        if not filename.endswith(".whl") or not _before_cutoff(file, release_cutoff):
            continue
        requires_python = file.get("requires_python")
        for python_version in python_versions:
            compatible, _ = wheel_is_compatible(filename, python_version)
            if compatible and requires_python_allows(requires_python, python_version):
                return True
    return False


def _before_cutoff(file: dict[str, Any], cutoff: datetime) -> bool:
    value = file.get("upload_time_iso_8601") or file.get("upload_time")
    if not value:
        return False
    try:
        return _parse_datetime(str(value)) <= cutoff
    except ValueError:
        return False


def _spread(ordered: list[str], count: int) -> list[str]:
    if count <= 0 or not ordered:
        return []
    if count >= len(ordered):
        return ordered
    if count == 1:
        return [ordered[-1]]
    indexes = {
        round(index * (len(ordered) - 1) / (count - 1)) for index in range(count)
    }
    return [ordered[index] for index in sorted(indexes)]


def _version_key(version: str) -> tuple[tuple[int, ...], int]:
    match = _STABLE_VERSION.fullmatch(version)
    if not match:
        raise ScopePlanError(f"unsupported stable version format {version!r}")
    release = tuple(int(part) for part in match.group("release").split("."))
    return release, int(match.group("post") or -1)


def _parse_datetime(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _read_object(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ScopePlanError(f"cannot read {label} {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ScopePlanError(f"{label} must be a JSON object")
    return payload


def _split_description(split: str) -> str:
    descriptions = {
        "development": "Known and new development families whose outcomes may be inspected.",
        "validation": "Package-disjoint validation families used only after development is frozen.",
        "final_test": "Package-disjoint final-test families whose outcomes must remain sealed.",
    }
    return descriptions[split]
