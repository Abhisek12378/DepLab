from __future__ import annotations

import hashlib
import json
import re
import urllib.error
import urllib.request
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

from .models import utc_now
from .storage import append_jsonl, read_jsonl


class ChangelogError(ValueError):
    pass


FetchText = Callable[[str], str]

SIGNAL_PATTERNS = {
    "breaking": (r"\bbreaking\b", r"backwards? incompatible", r"\bincompatib"),
    "removal": (r"\bremove[ds]?\b", r"\bno longer\b", r"\bdrop(?:ped)?\b"),
    "deprecation": (r"\bdeprecat" ,),
    "api": (r"\bapi\b", r"\bfunction\b", r"\bmethod\b", r"\bmodule\b", r"\battribute\b"),
    "abi": (r"\babi\b", r"binary incompat", r"compiled against", r"\brebuild"),
    "dependency": (r"\bdependenc", r"\brequires?\b", r"\brequirement", r"\bpin(?:ned|ning)?\b", r"upper bound"),
    "python_support": (r"\bpython\b", r"\bcpython\b", r"\bpypy\b"),
    "wheel_build": (r"\bwheel\b", r"\bbuild", r"\bcython\b", r"\bcompiler\b", r"\bmanylinux\b"),
}

HIGH_SIGNAL_GROUPS = {
    "removed_deprecated": (
        (r"\bremove[ds]?\b", r"\bno longer\b"),
        (r"\bdeprecat",),
    ),
    "api_removal": (
        (r"\bremove[ds]?\b", r"\bno longer\b"),
        (r"\bapi\b", r"\bfunction\b", r"\bmethod\b", r"\bmodule\b", r"\battribute\b", r"\bdeprecat"),
    ),
    "abi_break": (
        (r"\babi\b", r"binary incompat", r"compiled against"),
        (r"\bbreak", r"\bincompatib", r"\brebuild", r"\bchange[ds]?\b"),
    ),
    "dependency_compatibility": (
        (r"\bdependenc", r"\brequires?\b", r"\bpin(?:ned|ning)?\b"),
        (r"\bbreak", r"\bincompatib", r"\bsupport", r"upper bound", r"lower bound"),
    ),
    "support_drop": (
        (r"\bdrop(?:ped)?\b", r"\bno longer\b"),
        (r"\bsupport", r"\bpython\b", r"\bdependency"),
    ),
}

VERSION_PATTERN = re.compile(r"(?<!\d)(\d+\.\d+(?:\.\d+)?)(?!\d)")


@dataclass(frozen=True)
class ChangelogSummary:
    scope: str
    sources: str
    requested: int
    collected: int
    skipped_existing: int
    fetched_urls: int
    output: str


class ChangelogClient:
    def __init__(self, fetcher: FetchText | None = None) -> None:
        self._fetcher = fetcher or fetch_text
        self._cache: dict[str, str] = {}

    @property
    def fetched_urls(self) -> int:
        return len(self._cache)

    def get(self, url: str) -> str:
        if url not in self._cache:
            self._cache[url] = self._fetcher(url)
        return self._cache[url]


def fetch_text(url: str, timeout: float = 30.0) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": "DepLab/0.1"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.read().decode("utf-8", errors="replace")
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise ChangelogError(f"failed to fetch changelog {url}: {exc}") from exc


def collect_changelogs(
    scope_path: Path,
    sources_path: Path,
    output_path: Path,
    client: ChangelogClient | None = None,
) -> ChangelogSummary:
    scope = _read_object(scope_path, "scope")
    source_registry = _read_object(sources_path, "changelog sources")
    packages = scope.get("packages")
    sources = source_registry.get("packages")
    if not isinstance(packages, dict) or not packages:
        raise ChangelogError("scope must contain packages")
    if not isinstance(sources, dict):
        raise ChangelogError("changelog source registry must contain packages")
    missing_sources = sorted(set(packages) - set(sources))
    if missing_sources:
        raise ChangelogError(f"missing changelog sources for: {', '.join(missing_sources)}")

    requested = sum(len(package.get("versions", [])) for package in packages.values())
    existing = {
        str(row["changelog_id"])
        for row in read_jsonl(output_path)
        if "changelog_id" in row
    }
    client = client or ChangelogClient()
    known_packages = sorted(packages)
    collected = 0
    for package_name, package in packages.items():
        version_rows = package.get("versions")
        if not isinstance(version_rows, list):
            raise ChangelogError(f"scope package {package_name!r} has no versions")
        source = sources[package_name]
        for version_row in version_rows:
            version = str(version_row["version"])
            changelog_id = _changelog_id(package_name, version)
            if changelog_id in existing:
                continue
            record = extract_release_changelog(
                package_name,
                version,
                source,
                known_packages,
                client,
            )
            append_jsonl(
                output_path,
                {
                    "schema_version": "1.1.0",
                    "changelog_id": changelog_id,
                    "collected_at": utc_now(),
                    **record,
                },
            )
            existing.add(changelog_id)
            collected += 1
    return ChangelogSummary(
        scope=str(scope_path),
        sources=str(sources_path),
        requested=requested,
        collected=collected,
        skipped_existing=requested - collected,
        fetched_urls=client.fetched_urls,
        output=str(output_path),
    )


def extract_release_changelog(
    package: str,
    version: str,
    source: dict[str, Any],
    known_packages: list[str],
    client: ChangelogClient,
) -> dict[str, Any]:
    if not isinstance(source, dict):
        raise ChangelogError(f"invalid source definition for {package}")
    kind = str(source.get("kind") or "")
    if kind not in {"history", "series_history", "release_note"}:
        raise ChangelogError(f"unsupported changelog source kind {kind!r} for {package}")
    urls = _source_urls(version, source)
    sections = []
    source_rows = []
    exact_section_found = True
    for role, url in urls:
        text = client.get(url)
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if kind in {"history", "series_history"}:
            selected, found = _select_release_series(text, version)
            exact_section_found = exact_section_found and found
        else:
            selected, found = text, True
        sections.append(selected)
        source_rows.append(
            {
                "role": role,
                "url": url,
                "sha256": digest,
                "source_characters": len(text),
                "selected_characters": len(selected),
                "version_section_found": found,
            }
        )
    combined = "\n\n".join(sections)
    return {
        "package": package,
        "version": version,
        "source_kind": kind,
        "sources": source_rows,
        "version_section_found": exact_section_found,
        "signals": extract_signals(combined, known_packages, package),
    }


def extract_signals(text: str, known_packages: list[str], current_package: str) -> dict[str, Any]:
    normalized = re.sub(r"\s+", " ", text).strip()
    lower = normalized.lower()
    counts = {
        name: sum(len(re.findall(pattern, lower)) for pattern in patterns)
        for name, patterns in SIGNAL_PATTERNS.items()
    }
    meaningful_lines = [line.lower() for line in text.splitlines() if line.strip()]
    high_counts = {
        name: sum(
            any(re.search(pattern, line) for pattern in first_group)
            and any(re.search(pattern, line) for pattern in second_group)
            for line in meaningful_lines
        )
        for name, (first_group, second_group) in HIGH_SIGNAL_GROUPS.items()
    }
    mentions = {
        package: len(re.findall(rf"(?<![\w-]){re.escape(package.lower())}(?![\w-])", lower))
        for package in known_packages
        if package != current_package
    }
    mentions = {name: count for name, count in mentions.items() if count}
    evidence_lines = []
    keywords = tuple(pattern for patterns in SIGNAL_PATTERNS.values() for pattern in patterns)
    for raw_line in text.splitlines():
        line = re.sub(r"\s+", " ", raw_line).strip(" -*#`\t")
        if len(line) < 12:
            continue
        if any(re.search(pattern, line.lower()) for pattern in keywords):
            evidence_lines.append(line[:180])
        if len(evidence_lines) == 8:
            break
    return {
        "selected_characters": len(normalized),
        "selected_lines": len([line for line in text.splitlines() if line.strip()]),
        **{f"{name}_count": count for name, count in counts.items()},
        **{f"{name}_flag": count > 0 for name, count in counts.items()},
        **{f"{name}_count": count for name, count in high_counts.items()},
        **{f"{name}_flag": count > 0 for name, count in high_counts.items()},
        "package_mentions": mentions,
        "evidence_lines": evidence_lines,
    }


def _source_urls(version: str, source: dict[str, Any]) -> list[tuple[str, str]]:
    parts = _version_tuple(version)
    exact = _format_url(source, version, version)
    urls = [("exact", exact)]
    if source.get("include_series_baseline") and parts[2] != 0:
        baseline = f"{parts[0]}.{parts[1]}.0"
        url = _format_url(source, version, baseline)
        if url != exact:
            urls.append(("series_baseline", url))
    major_from = source.get("include_major_baseline_from")
    if isinstance(major_from, int) and parts[0] >= major_from and (parts[1], parts[2]) != (0, 0):
        baseline = f"{parts[0]}.0.0"
        url = _format_url(source, version, baseline)
        if url not in {item[1] for item in urls}:
            urls.append(("major_baseline", url))
    return urls


def _format_url(source: dict[str, Any], release_version: str, note_version: str) -> str:
    template = str(source.get("url_template") or "")
    historical_before = source.get("historical_before")
    historical_template = source.get("historical_url_template")
    if (
        isinstance(historical_before, str)
        and isinstance(historical_template, str)
        and _version_tuple(release_version) < _version_tuple(historical_before)
    ):
        template = historical_template
    if not template:
        raise ChangelogError("changelog source has no URL template")
    major, minor, _ = _version_tuple(release_version)
    padded_version = f"{major}.{minor:02d}.{_version_tuple(release_version)[2]}"
    tag_template = str(source.get("tag_template") or "{version}")
    tag = tag_template.format(version=release_version, padded_version=padded_version)
    return template.format(
        version=release_version,
        padded_version=padded_version,
        note_version=note_version,
        tag=tag,
        major_minor=f"{major}.{minor}",
        series_version=f"{major}.{minor}.0",
    )


def _select_release_series(text: str, version: str) -> tuple[str, bool]:
    target = _version_tuple(version)
    headings = []
    lines = text.splitlines()
    for index, line in enumerate(lines):
        underlined = (
            index + 1 < len(lines)
            and re.fullmatch(r"\s*[-=~^`:#*+]{3,}\s*", lines[index + 1]) is not None
        )
        atx = re.match(r"^\s*#{1,6}\s+(.+?)\s*#*\s*$", line)
        if not underlined and not atx:
            continue
        heading = atx.group(1) if atx else line
        match = VERSION_PATTERN.search(heading)
        if match:
            headings.append((index, _version_tuple(match.group(1))))
    start_position = next((position for position, (_, item) in enumerate(headings) if item == target), None)
    if start_position is None:
        return "", False
    start = headings[start_position][0]
    end = len(lines)
    for line_index, item in headings[start_position + 1 :]:
        if item[:2] != target[:2]:
            end = line_index
            break
    return "\n".join(lines[start:end]).strip(), True


def _version_tuple(version: str) -> tuple[int, int, int]:
    match = re.match(r"^(\d+)\.(\d+)(?:\.(\d+))?", version)
    if not match:
        raise ChangelogError(f"unsupported release version {version!r}")
    return int(match.group(1)), int(match.group(2)), int(match.group(3) or 0)


def _changelog_id(package: str, version: str) -> str:
    raw = f"{package.lower()}|{version}|changelog-v1.1"
    return hashlib.sha256(raw.encode()).hexdigest()[:20]


def _read_object(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ChangelogError(f"cannot read {label} {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ChangelogError(f"{label} must be a JSON object")
    return payload
