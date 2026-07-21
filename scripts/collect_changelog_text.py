from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from deplab.changelogs import (
    ChangelogClient,
    _select_release_series,
    _source_urls,
    extract_signals,
)
from deplab.models import utc_now
from deplab.storage import append_jsonl, read_jsonl


SCHEMA_VERSION = "1.0.0"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Collect the exact release-note text used for frozen text embeddings"
    )
    parser.add_argument("--scope", type=Path, default=Path("configs/expanded-scope.json"))
    parser.add_argument(
        "--development-matrix",
        type=Path,
        default=Path("configs/expanded-development-matrix.json"),
        help="Defines the package vocabulary used by the frozen signal catalog",
    )
    parser.add_argument("--sources", type=Path, default=Path("configs/changelog-sources.json"))
    parser.add_argument("--signal-catalog", type=Path, default=Path("outputs/changelog-catalog-expanded-v1.2.0.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("outputs/changelog-text-catalog-expanded-v1.0.0.jsonl"))
    args = parser.parse_args()

    scope = json.loads(args.scope.read_text(encoding="utf-8"))
    source_registry = json.loads(args.sources.read_text(encoding="utf-8"))["packages"]
    signal_rows = read_jsonl(args.signal_catalog)
    signal_by_release = {
        (_canonical(row["package"]), str(row["version"])): row for row in signal_rows
    }
    existing = {
        (_canonical(row["package"]), str(row["version"]))
        for row in read_jsonl(args.output)
    }
    development = json.loads(args.development_matrix.read_text(encoding="utf-8"))
    known_packages = sorted(
        {
            experiment[field].split("==", 1)[0]
            for experiment in development["experiments"]
            for field in ("package_a", "package_b")
        }
    )
    client = ChangelogClient()
    collected = 0
    for package, package_scope in scope["packages"].items():
        source = source_registry[package]
        for version_row in package_scope["versions"]:
            version = str(version_row["version"])
            key = (_canonical(package), version)
            if key in existing:
                continue
            expected = signal_by_release[key]
            record = build_text_record(
                package,
                version,
                source,
                expected,
                known_packages,
                client,
            )
            append_jsonl(
                args.output,
                {
                    "schema_version": SCHEMA_VERSION,
                    "collected_at": utc_now(),
                    **record,
                },
            )
            existing.add(key)
            collected += 1
    summary = {
        "requested": len(signal_rows),
        "collected": collected,
        "skipped_existing": len(signal_rows) - collected,
        "fetched_urls": client.fetched_urls,
        "output": str(args.output),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def build_text_record(
    package: str,
    version: str,
    source: dict[str, Any],
    expected: dict[str, Any],
    known_packages: list[str],
    client: ChangelogClient,
) -> dict[str, Any]:
    kind = str(source["kind"])
    expected_sources = {item["url"]: item for item in expected["sources"]}
    sections = []
    source_rows = []
    for role, url in _source_urls(version, source):
        text = client.get(url)
        source_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        expected_source = expected_sources.get(url)
        if expected_source is None:
            raise ValueError(f"unexpected changelog URL for {package}=={version}: {url}")
        if source_hash != expected_source["sha256"]:
            raise ValueError(f"source content changed for {package}=={version}: {url}")
        if kind in {"history", "series_history"}:
            selected, found = _select_release_series(text, version)
        else:
            selected, found = text, True
        if bool(found) != bool(expected_source["version_section_found"]):
            raise ValueError(f"section-selection result changed for {package}=={version}")
        sections.append(selected)
        source_rows.append(
            {
                "role": role,
                "url": url,
                "source_sha256": source_hash,
                "selected_sha256": hashlib.sha256(selected.encode("utf-8")).hexdigest(),
                "selected_characters": len(selected),
                "version_section_found": bool(found),
            }
        )
    combined = "\n\n".join(sections)
    recalculated = extract_signals(combined, known_packages, package)
    expected_signals = expected["signals"]
    stable_keys = sorted((set(recalculated) | set(expected_signals)) - {"package_mentions"})
    stable_differences = {
        key: {"expected": expected_signals.get(key), "actual": recalculated.get(key)}
        for key in stable_keys
        if recalculated.get(key) != expected_signals.get(key)
    }
    if stable_differences:
        differences = stable_differences
        raise ValueError(
            f"structured signals no longer reproduce for {package}=={version}: "
            f"{json.dumps(differences, sort_keys=True)}"
        )
    return {
        "package": package,
        "version": version,
        "source_kind": kind,
        "sources": source_rows,
        "selected_text": combined,
        "selected_text_sha256": hashlib.sha256(combined.encode("utf-8")).hexdigest(),
        "selected_characters": len(combined),
        "signal_catalog_id": expected["changelog_id"],
        "signal_catalog_schema_version": expected["schema_version"],
        "signal_reproduction_verified": True,
        "package_mentions": expected_signals.get("package_mentions", {}),
    }


def _canonical(value: str) -> str:
    return value.lower().replace("_", "-").replace(".", "-")


if __name__ == "__main__":
    raise SystemExit(main())
