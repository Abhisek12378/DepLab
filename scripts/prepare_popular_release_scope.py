from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from packaging.version import InvalidVersion, Version


SCHEMA_VERSION = "1.0.0"
DEFAULT_RANKING_URL = (
    "https://hugovk.github.io/top-pypi-packages/"
    "top-pypi-packages-30-days.min.json"
)
USER_AGENT = "DepLab/1.0 popular-release-scope (+https://github.com/Abhisek12378/DepLab)"
_CANONICAL = re.compile(r"[-_.]+")


def main() -> int:
    args = _arguments()
    if args.scope.exists() or args.ranking_snapshot.exists():
        raise FileExistsError(
            "popular-package scope or ranking snapshot already exists; "
            "preserve it or choose new output paths"
        )
    cutoff = parse_datetime(args.release_cutoff)
    ranking_payload = fetch_json(args.ranking_url, timeout=args.timeout)
    ranked = ranked_projects(ranking_payload)
    selected, exclusions = collect_projects(
        ranked,
        top_count=args.top_count,
        cutoff=cutoff,
        cache_directory=args.cache_directory,
        timeout=args.timeout,
    )
    ranking_snapshot = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "ranking_method": "project downloads during the source's rolling 30-day window",
        "source_url": args.ranking_url,
        "source_sha256": sha256_json(ranking_payload),
        "download_count_is_quality_score": False,
        "packages": [
            {
                "rank": item["rank"],
                "name": item["name"],
                "download_count": item["download_count"],
            }
            for item in selected
        ],
    }
    scope = {
        "schema_version": SCHEMA_VERSION,
        "scope_id": "deplab-popular-100-all-stable-releases-v1.0.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "release_cutoff": cutoff.isoformat(),
        "coverage_order": ["3.11"],
        "target_platform": {
            "os": "linux",
            "architecture": "x86_64",
            "libc": "glibc",
        },
        "selection_policy": {
            "packages": "top 100 by frozen rolling-30-day download ranking",
            "versions": (
                "every PEP 440 stable, non-yanked release with at least one "
                "file uploaded on or before the frozen cutoff"
            ),
            "excluded": [
                "pre-releases",
                "development releases",
                "versions with no files",
                "releases whose files are all yanked",
                "files uploaded after the frozen cutoff",
                "non-PEP-440 versions",
            ],
        },
        "packages": {
            item["name"]: {
                "popularity_rank": item["rank"],
                "download_count_30_days": item["download_count"],
                "versions": item["versions"],
            }
            for item in selected
        },
    }
    args.ranking_snapshot.parent.mkdir(parents=True, exist_ok=True)
    args.scope.parent.mkdir(parents=True, exist_ok=True)
    args.ranking_snapshot.write_text(
        json.dumps(ranking_snapshot, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    args.scope.write_text(
        json.dumps(scope, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    summary = scope_summary(
        scope,
        args.ranking_snapshot,
        args.scope,
        exclusions,
    )
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    args.report.write_text(scope_report(summary), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Freeze the top PyPI packages and all eligible stable releases"
    )
    parser.add_argument("--ranking-url", default=DEFAULT_RANKING_URL)
    parser.add_argument("--top-count", type=int, default=100)
    parser.add_argument(
        "--release-cutoff",
        default="2026-07-27T00:00:00+00:00",
    )
    parser.add_argument(
        "--cache-directory",
        type=Path,
        default=Path("work/popular-package-project-cache-v1.0.0"),
    )
    parser.add_argument(
        "--ranking-snapshot",
        type=Path,
        default=Path("outputs/popular-packages-100-ranking-v1.0.0.json"),
    )
    parser.add_argument(
        "--scope",
        type=Path,
        default=Path("configs/popular-packages-100-scope-v1.0.0.json"),
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("outputs/popular-packages-100-scope-summary-v1.0.0.json"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("outputs/popular-packages-100-scope-report-v1.0.0.md"),
    )
    parser.add_argument("--timeout", type=float, default=60.0)
    args = parser.parse_args()
    if args.top_count < 1 or args.timeout <= 0:
        raise ValueError("top count and timeout must be positive")
    return args


def ranked_projects(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows = payload.get("rows")
    if not isinstance(rows, list):
        raise ValueError("ranking payload has no rows list")
    result = []
    seen: set[str] = set()
    for row in rows:
        if not isinstance(row, dict):
            continue
        raw_name = row.get("project")
        raw_count = row.get("download_count")
        if not raw_name or not isinstance(raw_count, int) or raw_count < 0:
            continue
        name = canonical(str(raw_name))
        if name in seen:
            continue
        seen.add(name)
        result.append(
            {
                "rank": len(result) + 1,
                "name": name,
                "download_count": raw_count,
            }
        )
    if not result:
        raise ValueError("ranking payload contains no valid projects")
    return result


def collect_projects(
    ranked: list[dict[str, Any]],
    top_count: int,
    cutoff: datetime,
    cache_directory: Path,
    timeout: float,
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    selected = []
    exclusions = []
    for ranked_item in ranked:
        if len(selected) >= top_count:
            break
        name = ranked_item["name"]
        try:
            project = cached_project(name, cache_directory, timeout)
            versions = eligible_versions(project, cutoff)
            if not versions:
                raise ValueError("no eligible stable releases")
        except (OSError, ValueError, urllib.error.URLError) as exc:
            exclusions.append({"name": name, "reason": str(exc)})
            continue
        selected.append({**ranked_item, "versions": versions})
        print(
            f"Selected {len(selected):03d}/{top_count:03d}: "
            f"{name} ({len(versions)} releases)",
            flush=True,
        )
    if len(selected) != top_count:
        raise RuntimeError(
            f"only {len(selected)} valid packages were available for top {top_count}"
        )
    return selected, exclusions


def cached_project(name: str, directory: Path, timeout: float) -> dict[str, Any]:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{canonical(name)}.json"
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    quoted = urllib.parse.quote(name, safe="")
    payload = fetch_json(f"https://pypi.org/pypi/{quoted}/json", timeout)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(payload, separators=(",", ":"), sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(path)
    return payload


def eligible_versions(project: dict[str, Any], cutoff: datetime) -> list[str]:
    releases = project.get("releases")
    if not isinstance(releases, dict):
        raise ValueError("project payload has no releases object")
    selected = []
    for raw_version, raw_files in releases.items():
        try:
            parsed = Version(str(raw_version))
        except InvalidVersion:
            continue
        if parsed.is_prerelease or parsed.is_devrelease:
            continue
        if release_has_eligible_file(raw_files, cutoff):
            selected.append(str(raw_version))
    return sorted(set(selected), key=Version)


def release_has_eligible_file(files: Any, cutoff: datetime) -> bool:
    if not isinstance(files, list):
        return False
    for file in files:
        if not isinstance(file, dict) or bool(file.get("yanked")):
            continue
        value = file.get("upload_time_iso_8601") or file.get("upload_time")
        if not value:
            continue
        try:
            uploaded = parse_datetime(str(value))
        except ValueError:
            continue
        if uploaded <= cutoff:
            return True
    return False


def fetch_json(url: str, timeout: float) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        headers={"Accept": "application/json", "User-Agent": USER_AGENT},
    )
    last_error: Exception | None = None
    for attempt in range(5):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                payload = json.load(response)
            if not isinstance(payload, dict):
                raise ValueError(f"{url} did not return a JSON object")
            return payload
        except (OSError, ValueError, urllib.error.URLError) as exc:
            last_error = exc
            if attempt < 4:
                time.sleep(2**attempt)
    raise OSError(f"failed to retrieve {url}: {last_error}")


def scope_summary(
    scope: dict[str, Any],
    ranking_path: Path,
    scope_path: Path,
    exclusions: list[dict[str, str]],
) -> dict[str, Any]:
    packages = dict(scope["packages"])
    counts = {
        name: len(payload["versions"]) for name, payload in packages.items()
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "scope_id": scope["scope_id"],
        "packages": len(packages),
        "releases": sum(counts.values()),
        "minimum_releases_per_package": min(counts.values()),
        "maximum_releases_per_package": max(counts.values()),
        "median_releases_per_package": float(
            sorted(counts.values())[(len(counts) - 1) // 2]
        ),
        "largest_release_histories": [
            {"name": name, "releases": count}
            for name, count in sorted(
                counts.items(), key=lambda item: (-item[1], item[0])
            )[:10]
        ],
        "ranking_snapshot": str(ranking_path),
        "ranking_snapshot_sha256": sha256_file(ranking_path),
        "scope": str(scope_path),
        "scope_sha256": sha256_file(scope_path),
        "ranking_candidates_skipped": exclusions,
        "embeddings_generated": False,
    }


def scope_report(summary: dict[str, Any]) -> str:
    largest = "\n".join(
        f"- {row['name']}: {row['releases']:,} releases"
        for row in summary["largest_release_histories"]
    )
    return f"""# DepLab popular-package release scope

- Packages: **{summary['packages']:,}**
- Eligible stable releases: **{summary['releases']:,}**
- Minimum releases for one package: **{summary['minimum_releases_per_package']:,}**
- Maximum releases for one package: **{summary['maximum_releases_per_package']:,}**

## Largest release histories

{largest}

Embeddings have not been generated yet. Review this frozen scope before starting
the CPU-intensive ModernBERT stage.
"""


def parse_datetime(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def canonical(value: str) -> str:
    return _CANONICAL.sub("-", value).lower()


def sha256_json(payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()
    return hashlib.sha256(raw).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
