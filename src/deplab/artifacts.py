from __future__ import annotations

import hashlib
import re
import urllib.parse
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - exercised only on Python 3.10
    import tomli as tomllib  # type: ignore[no-redef]

from .models import ExperimentSpec, InstalledWheelArtifact
from .wheels import parse_wheel_tags, wheel_is_compatible


class ArtifactLockError(ValueError):
    pass


def lock_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_installed_wheels(
    path: Path, spec: ExperimentSpec
) -> list[InstalledWheelArtifact]:
    """Choose one exact compatible wheel for every package in a PEP 751 lock."""
    try:
        with path.open("rb") as file:
            payload = tomllib.load(file)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise ArtifactLockError(f"cannot read artifact lock {path}: {exc}") from exc

    packages = payload.get("packages")
    if not isinstance(packages, list) or not packages:
        raise ArtifactLockError("artifact lock has no packages")

    top_level = {
        _canonical_name(spec.package_a.name),
        _canonical_name(spec.package_b.name),
    }
    selected: list[InstalledWheelArtifact] = []
    seen: set[str] = set()
    for index, package in enumerate(packages):
        if not isinstance(package, dict):
            raise ArtifactLockError(f"artifact lock package {index} is not a table")
        name = str(package.get("name") or "").strip()
        version = str(package.get("version") or "").strip()
        canonical_name = _canonical_name(name)
        if not name or not version:
            raise ArtifactLockError(f"artifact lock package {index} lacks name or version")
        if canonical_name in seen:
            raise ArtifactLockError(f"artifact lock repeats package {name!r}")
        seen.add(canonical_name)

        candidates = _wheel_candidates(package)
        compatible = []
        for wheel in candidates:
            filename = _wheel_filename(wheel)
            is_compatible, _ = wheel_is_compatible(
                filename,
                spec.python_version,
                spec.os,
                spec.architecture,
            )
            if is_compatible:
                compatible.append(wheel)
        if not compatible:
            raise ArtifactLockError(
                f"artifact lock has no compatible wheel for {name}=={version}"
            )

        wheel = max(compatible, key=lambda item: _wheel_rank(_wheel_filename(item), spec))
        filename = _wheel_filename(wheel)
        url = str(wheel.get("url") or "")
        hashes = wheel.get("hashes")
        sha256 = str(hashes.get("sha256") or "") if isinstance(hashes, dict) else ""
        if not url:
            raise ArtifactLockError(f"selected wheel {filename} has no URL")
        if not re.fullmatch(r"[0-9a-fA-F]{64}", sha256):
            raise ArtifactLockError(f"selected wheel {filename} has no valid SHA-256")
        size_value = wheel.get("size")
        size = int(size_value) if isinstance(size_value, int) else None
        tags = parse_wheel_tags(filename)
        selected.append(
            InstalledWheelArtifact(
                package=name,
                version=version,
                filename=filename,
                url=url,
                size=size,
                sha256=sha256.lower(),
                python_tag=tags.python,
                abi_tag=tags.abi,
                platform_tag=tags.platform,
                top_level=canonical_name in top_level,
            )
        )

    if {_canonical_name(item.package) for item in selected if item.top_level} != top_level:
        raise ArtifactLockError("artifact lock does not contain both top-level packages")
    return sorted(selected, key=lambda item: _canonical_name(item.package))


def _wheel_candidates(package: dict[str, Any]) -> list[dict[str, Any]]:
    candidates = [item for item in package.get("wheels", []) if isinstance(item, dict)]
    archive = package.get("archive")
    if isinstance(archive, dict):
        try:
            if _wheel_filename(archive).endswith(".whl"):
                candidates.append(archive)
        except ArtifactLockError:
            pass
    return candidates


def _wheel_filename(wheel: dict[str, Any]) -> str:
    name = str(wheel.get("name") or "").strip()
    if name:
        return name
    url = str(wheel.get("url") or "").strip()
    if url:
        return urllib.parse.unquote(Path(urllib.parse.urlsplit(url).path).name)
    path = str(wheel.get("path") or "").strip()
    if path:
        return Path(path).name
    raise ArtifactLockError("wheel entry has no name, URL, or path")


def _wheel_rank(filename: str, spec: ExperimentSpec) -> tuple[int, int, int, str]:
    tags = parse_wheel_tags(filename)
    compact = spec.python_version.replace(".", "")
    python_tags = tags.python.split(".")
    abi_tags = tags.abi.split(".")
    platforms = tags.platform.split(".")
    python_rank = 3 if f"cp{compact}" in python_tags else 2 if "abi3" in abi_tags else 1
    abi_rank = 3 if f"cp{compact}" in abi_tags else 2 if "abi3" in abi_tags else 1
    platform_rank = 1 if platforms == ["any"] else 2
    return platform_rank, python_rank, abi_rank, filename


def _canonical_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()
