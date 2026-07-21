from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from .models import utc_now
from .pypi import PyPIClient
from .storage import append_jsonl, read_jsonl


class ScopeError(ValueError):
    pass


@dataclass(frozen=True)
class CatalogSummary:
    scope: str
    requested: int
    collected: int
    skipped_existing: int
    output: str


def collect_catalog(
    scope_path: Path,
    output_path: Path,
    client: PyPIClient | None = None,
) -> CatalogSummary:
    try:
        scope = json.loads(scope_path.read_text(encoding="utf-8"))
        python_versions = list(scope["coverage_order"])
        packages = dict(scope["packages"])
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise ScopeError(f"invalid package scope {scope_path}: {exc}") from exc
    if python_versions != ["3.10", "3.11", "3.12"]:
        raise ScopeError("scope coverage_order must be Python 3.10, 3.11 and 3.12")

    requested = sum(len(package.get("versions", [])) for package in packages.values()) * len(
        python_versions
    )
    existing = {
        row["catalog_id"] for row in read_jsonl(output_path) if "catalog_id" in row
    }
    collected = 0
    client = client or PyPIClient()
    for package_name, package in packages.items():
        for version_entry in package.get("versions", []):
            version = str(version_entry["version"])
            for python_version in python_versions:
                catalog_id = _catalog_id(package_name, version, python_version)
                if catalog_id in existing:
                    continue
                release = client.release(package_name, version, python_version)
                append_jsonl(
                    output_path,
                    {
                        "schema_version": "1.0.0",
                        "catalog_id": catalog_id,
                        "collected_at": utc_now(),
                        "target": {
                            "python_version": python_version,
                            "os": "linux",
                            "architecture": "x86_64",
                            "libc": "glibc",
                        },
                        "release": asdict(release),
                    },
                )
                existing.add(catalog_id)
                collected += 1
    return CatalogSummary(
        scope=str(scope_path),
        requested=requested,
        collected=collected,
        skipped_existing=requested - collected,
        output=str(output_path),
    )


def _catalog_id(package: str, version: str, python_version: str) -> str:
    raw = f"{package.lower()}|{version}|{python_version}|linux|x86_64|glibc"
    return hashlib.sha256(raw.encode()).hexdigest()[:20]

