from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class WheelTags:
    python: str
    abi: str
    platform: str


def parse_wheel_tags(filename: str) -> WheelTags:
    """Parse the three compatibility fields at the end of a wheel filename."""
    if not filename.endswith(".whl"):
        raise ValueError(f"not a wheel filename: {filename}")
    parts = filename[:-4].rsplit("-", 3)
    if len(parts) != 4:
        raise ValueError(f"invalid wheel filename: {filename}")
    return WheelTags(python=parts[1], abi=parts[2], platform=parts[3])


def wheel_is_compatible(
    filename: str,
    python_version: str,
    os_name: str = "linux",
    architecture: str = "x86_64",
) -> tuple[bool, str]:
    """Conservative compatibility check for DepLab's fixed MVP platform.

    A positive answer means at least one tag combination looks usable. The final
    authority remains uv/pip inside the target interpreter.
    """
    try:
        tags = parse_wheel_tags(filename)
    except ValueError as exc:
        return False, str(exc)

    if os_name != "linux" or architecture != "x86_64":
        return False, "MVP supports only linux x86_64"

    platforms = tags.platform.split(".")
    platform_ok = any(
        tag == "any"
        or tag == "linux_x86_64"
        or (tag.startswith("manylinux") and tag.endswith("_x86_64"))
        for tag in platforms
    )
    if not platform_ok:
        return False, f"platform tag {tags.platform!r} is not glibc Linux x86_64"

    major, minor = (int(part) for part in python_version.split(".")[:2])
    compact = f"{major}{minor}"
    python_tags = tags.python.split(".")
    abi_tags = tags.abi.split(".")

    for py_tag in python_tags:
        if py_tag in {f"py{major}", f"py{compact}"} and "none" in abi_tags:
            return True, f"compatible universal tag {py_tag}-none"
        if py_tag == f"cp{compact}" and (
            f"cp{compact}" in abi_tags or "abi3" in abi_tags or "none" in abi_tags
        ):
            return True, f"compatible CPython {python_version} tag"
        match = re.fullmatch(r"cp(\d)(\d+)", py_tag)
        if match and "abi3" in abi_tags:
            built_for = (int(match.group(1)), int(match.group(2)))
            if built_for <= (major, minor):
                return True, f"compatible stable ABI from CPython {built_for[0]}.{built_for[1]}"

    return False, f"Python/ABI tags {tags.python}-{tags.abi} do not match {python_version}"


_SPECIFIER = re.compile(r"^(===|==|!=|~=|>=|<=|>|<)\s*([0-9]+(?:\.[0-9]+){0,2})(\.\*)?$")


def requires_python_allows(specifier: str | None, python_version: str) -> bool:
    """Evaluate common Requires-Python clauses without importing packaging.

    Unknown clauses are kept eligible and delegated to uv, avoiding false
    coverage gaps. PyPI's common comma-separated comparison forms are handled.
    """
    if not specifier:
        return True
    # A matrix target such as "3.11" means the current patch interpreter uv
    # selects in that series, rather than the historical 3.11.0 specifically.
    target = _version_tuple(python_version, missing_patch=999)
    for raw_clause in specifier.split(","):
        clause = raw_clause.strip()
        match = _SPECIFIER.match(clause)
        if not match:
            continue
        op, raw_version, wildcard = match.groups()
        expected = _version_tuple(raw_version)
        if wildcard:
            prefix = tuple(int(part) for part in raw_version.split("."))
            matches_prefix = target[: len(prefix)] == prefix
            if op in {"==", "==="} and not matches_prefix:
                return False
            if op == "!=" and matches_prefix:
                return False
            continue
        if op in {"==", "==="} and target != expected:
            return False
        if op == "!=" and target == expected:
            return False
        if op == ">=" and target < expected:
            return False
        if op == ">" and target <= expected:
            return False
        if op == "<=" and target > expected:
            return False
        if op == "<" and target >= expected:
            return False
        if op == "~=":
            upper = (expected[0] + 1, 0, 0) if expected[2] == 0 else (expected[0], expected[1] + 1, 0)
            if not (target >= expected and target < upper):
                return False
    return True


def _version_tuple(value: str, missing_patch: int = 0) -> tuple[int, int, int]:
    parts = [int(part) for part in value.split(".")]
    if len(parts) == 2:
        parts.append(missing_patch)
    return tuple((parts + [0, 0, 0])[:3])  # type: ignore[return-value]
