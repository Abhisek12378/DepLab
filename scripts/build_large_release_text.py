from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "3.0.0"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build one inference-safe evidence document per selected release"
    )
    parser.add_argument(
        "--scope",
        type=Path,
        default=Path("configs/large-scope-v3.0.0.json"),
    )
    parser.add_argument(
        "--catalog",
        type=Path,
        default=Path("outputs/large-package-catalog-v3.0.0.jsonl"),
    )
    parser.add_argument(
        "--changelog-text",
        type=Path,
        default=Path("outputs/changelog-text-catalog-expanded-v1.0.0.jsonl"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/large-release-evidence-text-v3.0.0.jsonl"),
    )
    args = parser.parse_args()

    scope = _read_json(args.scope)
    catalog_rows = _read_jsonl(args.catalog)
    changelog_rows = (
        _read_jsonl(args.changelog_text) if args.changelog_text.exists() else []
    )
    releases = _release_index(catalog_rows)
    changelogs = {
        (_canonical(str(row["package"])), str(row["version"])): row
        for row in changelog_rows
    }
    selected = _selected_releases(scope)
    missing = sorted(selected - set(releases))
    if missing:
        examples = ", ".join(f"{name}=={version}" for name, version in missing[:10])
        raise ValueError(f"catalog is missing {len(missing)} selected releases: {examples}")

    rows = []
    for package, version in sorted(selected):
        release = releases[(package, version)]
        changelog = changelogs.get((package, version))
        text = release_evidence_text(release, changelog)
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "package": str(release["name"]),
                "version": version,
                "text_sources": (
                    ["pypi_release_metadata", "frozen_changelog_text"]
                    if changelog
                    else ["pypi_release_metadata"]
                ),
                "changelog_available": changelog is not None,
                "selected_text": text,
                "selected_text_sha256": hashlib.sha256(
                    text.encode("utf-8")
                ).hexdigest(),
                "selected_characters": len(text),
            }
        )
    _write_jsonl(args.output, rows)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "scope": str(args.scope),
        "catalog": str(args.catalog),
        "output": str(args.output),
        "rows": len(rows),
        "changelog_rows": sum(bool(row["changelog_available"]) for row in rows),
        "metadata_only_rows": sum(
            not bool(row["changelog_available"]) for row in rows
        ),
        "output_sha256": _sha256(args.output),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def release_evidence_text(
    release: dict[str, Any],
    changelog: dict[str, Any] | None,
) -> str:
    requirements = sorted(str(value) for value in release.get("requires_dist", []))
    classifiers = sorted(
        str(value)
        for value in release.get("classifiers", [])
        if str(value).startswith(
            (
                "Programming Language :: Python",
                "Operating System",
                "Topic",
            )
        )
    )
    lines = [
        f"Package: {release['name']}",
        f"Release version: {release['version']}",
        f"Requires Python: {release.get('requires_python') or 'not declared'}",
        f"Release date: {release.get('release_date') or 'unknown'}",
        "Published dependencies:",
        *(requirements or ["none declared"]),
        "Relevant classifiers:",
        *(classifiers or ["none"]),
    ]
    if changelog:
        lines.extend(
            [
                "Frozen release notes:",
                str(changelog.get("selected_text") or ""),
            ]
        )
    else:
        lines.extend(
            [
                "Frozen release notes:",
                "No version-pinned changelog text is available; rely on published metadata.",
            ]
        )
    return "\n".join(lines).strip() + "\n"


def _release_index(
    rows: list[dict[str, Any]],
) -> dict[tuple[str, str], dict[str, Any]]:
    result: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        release = dict(row["release"])
        key = (_canonical(str(release["name"])), str(release["version"]))
        if key not in result:
            result[key] = release
            continue
        _validate_release_metadata(result[key], release, key)
    return result


def _validate_release_metadata(
    left: dict[str, Any],
    right: dict[str, Any],
    key: tuple[str, str],
) -> None:
    stable_fields = (
        "name",
        "version",
        "requires_python",
        "requires_dist",
        "classifiers",
        "release_date",
        "yanked",
    )
    different = [field for field in stable_fields if left.get(field) != right.get(field)]
    if different:
        raise ValueError(
            f"catalog metadata differs across Python targets for {key}: {different}"
        )


def _selected_releases(scope: dict[str, Any]) -> set[tuple[str, str]]:
    selected = set()
    for package, payload in dict(scope["packages"]).items():
        for version_entry in payload["versions"]:
            version = (
                str(version_entry["version"])
                if isinstance(version_entry, dict)
                else str(version_entry)
            )
            selected.add((_canonical(package), version))
    return selected


def _canonical(value: str) -> str:
    return value.lower().replace("_", "-").replace(".", "-")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as file:
        for row in rows:
            file.write(
                json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
            )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
