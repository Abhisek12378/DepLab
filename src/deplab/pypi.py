from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Callable

from .models import PackageRelease, WheelArtifact
from .wheels import parse_wheel_tags, requires_python_allows, wheel_is_compatible


class PyPIError(RuntimeError):
    pass


FetchJson = Callable[[str], dict[str, Any]]


def fetch_json(url: str, timeout: float = 30.0) -> dict[str, Any]:
    request = urllib.request.Request(url, headers={"User-Agent": "DepLab/0.1"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.load(response)
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise PyPIError(f"failed to fetch {url}: {exc}") from exc


class PyPIClient:
    def __init__(self, fetcher: FetchJson = fetch_json) -> None:
        self._fetch = fetcher
        self._payload_cache: dict[str, dict[str, Any]] = {}

    def _get(self, url: str) -> dict[str, Any]:
        if url not in self._payload_cache:
            self._payload_cache[url] = self._fetch(url)
        return self._payload_cache[url]

    def project(self, package: str) -> dict[str, Any]:
        """Return the PyPI project payload used for deterministic release selection."""
        safe_package = urllib.parse.quote(package, safe="")
        return self._get(f"https://pypi.org/pypi/{safe_package}/json")

    def release(
        self,
        package: str,
        version: str,
        python_version: str,
        os_name: str = "linux",
        architecture: str = "x86_64",
    ) -> PackageRelease:
        safe_package = urllib.parse.quote(package, safe="")
        safe_version = urllib.parse.quote(version, safe="")
        payload = self._get(f"https://pypi.org/pypi/{safe_package}/{safe_version}/json")
        info = payload.get("info", {})
        urls = payload.get("urls", [])
        requires_python = info.get("requires_python")
        python_allowed = requires_python_allows(requires_python, python_version)

        wheels: list[WheelArtifact] = []
        for file in urls:
            if file.get("packagetype") != "bdist_wheel" or not str(file.get("filename", "")).endswith(".whl"):
                continue
            filename = str(file["filename"])
            tags = parse_wheel_tags(filename)
            compatible, reason = wheel_is_compatible(filename, python_version, os_name, architecture)
            if not python_allowed:
                compatible = False
                reason = f"Requires-Python {requires_python!r} excludes {python_version}"
            wheels.append(
                WheelArtifact(
                    filename=filename,
                    url=str(file.get("url", "")),
                    size=file.get("size"),
                    sha256=(file.get("digests") or {}).get("sha256"),
                    python_tag=tags.python,
                    abi_tag=tags.abi,
                    platform_tag=tags.platform,
                    has_native_extensions=tags.abi != "none" or tags.platform != "any",
                    uploaded_at=file.get("upload_time_iso_8601") or file.get("upload_time"),
                    yanked=bool(file.get("yanked", False)),
                    compatible=compatible and not bool(file.get("yanked", False)),
                    compatibility_reason=reason,
                )
            )

        dates = [
            file.get("upload_time_iso_8601") or file.get("upload_time")
            for file in urls
            if file.get("upload_time_iso_8601") or file.get("upload_time")
        ]
        return PackageRelease(
            name=str(info.get("name") or package),
            version=str(info.get("version") or version),
            requires_python=requires_python,
            requires_dist=list(info.get("requires_dist") or []),
            extras=list(info.get("provides_extra") or []),
            classifiers=list(info.get("classifiers") or []),
            project_urls=dict(info.get("project_urls") or {}),
            release_date=min(dates) if dates else None,
            yanked=bool(info.get("yanked", False)),
            wheels=wheels,
        )
