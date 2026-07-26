from __future__ import annotations

import argparse
import hashlib
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

try:
    from build_large_release_text import release_evidence_text
except ImportError:
    from scripts.build_large_release_text import release_evidence_text


SCHEMA_VERSION = "1.0.0"
USER_AGENT = "DepLab/1.0 release-evidence (+https://github.com/Abhisek12378/DepLab)"
FetchJson = Callable[[str], dict[str, Any]]


def main() -> int:
    args = _arguments()
    scope = read_json(args.scope)
    tasks = selected_releases(scope)
    changelogs = changelog_index(args.changelog_text)
    existing_rows = read_jsonl(args.output)
    completed = validate_existing(existing_rows)
    pending = [task for task in tasks if task not in completed]
    collected = collect_pending(
        pending,
        changelogs,
        args.output,
        workers=args.workers,
        batch_size=args.batch_size,
        timeout=args.timeout,
    )
    final_rows = read_jsonl(args.output)
    validate_complete(tasks, final_rows)
    runtime = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scope": str(args.scope),
        "scope_sha256": sha256_file(args.scope),
        "output": str(args.output),
        "output_sha256": sha256_file(args.output),
        "rows": len(final_rows),
        "collected_this_run": collected,
        "workers": args.workers,
        "batch_size": args.batch_size,
        "changelog_rows": sum(bool(row["changelog_available"]) for row in final_rows),
        "metadata_only_rows": sum(
            not bool(row["changelog_available"]) for row in final_rows
        ),
    }
    args.runtime.parent.mkdir(parents=True, exist_ok=True)
    args.runtime.write_text(
        json.dumps(runtime, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(runtime, indent=2, sort_keys=True))
    return 0


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect resumable release evidence for the frozen popular scope"
    )
    parser.add_argument(
        "--scope",
        type=Path,
        default=Path("configs/popular-packages-100-scope-v1.0.0.json"),
    )
    parser.add_argument(
        "--changelog-text",
        type=Path,
        default=Path("outputs/changelog-text-catalog-expanded-v1.0.0.jsonl"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/popular-release-evidence-v1.0.0.jsonl"),
    )
    parser.add_argument(
        "--runtime",
        type=Path,
        default=Path("outputs/popular-release-evidence-v1.0.0-runtime.json"),
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--timeout", type=float, default=60.0)
    args = parser.parse_args()
    if args.workers < 1 or args.batch_size < 1 or args.timeout <= 0:
        raise ValueError("workers, batch size and timeout must be positive")
    return args


def selected_releases(scope: dict[str, Any]) -> list[tuple[str, str]]:
    tasks = []
    for package, payload in dict(scope["packages"]).items():
        for version in payload["versions"]:
            tasks.append((canonical(package), str(version)))
    if len(tasks) != len(set(tasks)):
        raise ValueError("popular scope contains duplicate package releases")
    return sorted(tasks)


def changelog_index(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    if not path.is_file():
        return {}
    return {
        (canonical(row["package"]), str(row["version"])): row
        for row in read_jsonl(path)
    }


def collect_pending(
    pending: list[tuple[str, str]],
    changelogs: dict[tuple[str, str], dict[str, Any]],
    output: Path,
    workers: int,
    batch_size: int,
    timeout: float,
    fetcher: FetchJson | None = None,
) -> int:
    fetcher = fetcher or (lambda url: fetch_json(url, timeout))
    collected = 0
    for start in range(0, len(pending), batch_size):
        batch = pending[start : start + batch_size]
        with ThreadPoolExecutor(max_workers=workers) as executor:
            rows = list(
                executor.map(
                    lambda key: evidence_row(
                        key[0],
                        key[1],
                        fetch_release(key[0], key[1], fetcher),
                        changelogs.get(key),
                    ),
                    batch,
                )
            )
        append_rows(output, rows)
        collected += len(rows)
        print(
            f"Evidence {collected:05d}/{len(pending):05d} pending releases",
            flush=True,
        )
    return collected


def fetch_release(package: str, version: str, fetcher: FetchJson) -> dict[str, Any]:
    quoted_package = urllib.parse.quote(package, safe="")
    quoted_version = urllib.parse.quote(version, safe="")
    return fetcher(
        f"https://pypi.org/pypi/{quoted_package}/{quoted_version}/json"
    )


def evidence_row(
    package: str,
    version: str,
    payload: dict[str, Any],
    changelog: dict[str, Any] | None,
) -> dict[str, Any]:
    info = dict(payload.get("info") or {})
    urls = list(payload.get("urls") or [])
    dates = sorted(
        str(file.get("upload_time_iso_8601") or file.get("upload_time"))
        for file in urls
        if file.get("upload_time_iso_8601") or file.get("upload_time")
    )
    release = {
        "name": str(info.get("name") or package),
        "version": str(info.get("version") or version),
        "requires_python": info.get("requires_python"),
        "requires_dist": list(info.get("requires_dist") or []),
        "classifiers": list(info.get("classifiers") or []),
        "release_date": dates[0] if dates else None,
        "yanked": bool(info.get("yanked", False)),
    }
    if canonical(release["name"]) != canonical(package):
        raise ValueError(
            f"release name mismatch for {package}=={version}: {release['name']}"
        )
    if release["version"] != version:
        raise ValueError(
            f"release version mismatch for {package}=={version}: "
            f"{release['version']}"
        )
    text = release_evidence_text(release, changelog)
    return {
        "schema_version": SCHEMA_VERSION,
        "package": release["name"],
        "version": version,
        "text_sources": (
            ["pypi_release_metadata", "frozen_changelog_text"]
            if changelog
            else ["pypi_release_metadata"]
        ),
        "changelog_available": changelog is not None,
        "selected_text": text,
        "selected_text_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "selected_characters": len(text),
    }


def fetch_json(url: str, timeout: float) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        headers={"Accept": "application/json", "User-Agent": USER_AGENT},
    )
    last_error: Exception | None = None
    for attempt in range(7):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                payload = json.load(response)
            if not isinstance(payload, dict):
                raise ValueError(f"{url} returned a non-object JSON value")
            return payload
        except (
            OSError,
            ValueError,
            urllib.error.HTTPError,
            urllib.error.URLError,
        ) as exc:
            last_error = exc
            if attempt < 6:
                time.sleep(min(30, 2**attempt))
    raise OSError(f"failed to retrieve {url}: {last_error}")


def validate_existing(rows: list[dict[str, Any]]) -> set[tuple[str, str]]:
    keys = [(canonical(row["package"]), str(row["version"])) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("release-evidence output contains duplicate rows")
    return set(keys)


def validate_complete(
    tasks: list[tuple[str, str]], rows: list[dict[str, Any]]
) -> None:
    result = validate_existing(rows)
    expected = set(tasks)
    if result != expected:
        raise ValueError(
            f"release evidence is incomplete: expected {len(expected)}, "
            f"observed {len(result)}"
        )
    invalid = [
        row
        for row in rows
        if hashlib.sha256(row["selected_text"].encode()).hexdigest()
        != row["selected_text_sha256"]
    ]
    if invalid:
        raise ValueError("release evidence contains an invalid text hash")


def append_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as file:
        for row in rows:
            file.write(json.dumps(row, separators=(",", ":"), sort_keys=True) + "\n")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def canonical(value: Any) -> str:
    return str(value).lower().replace("_", "-").replace(".", "-")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
