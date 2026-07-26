from __future__ import annotations

import re

from .contracts import AnalysisRequest, ParsedIntent


_NAME_NORMALIZER = re.compile(r"[-_.]+")


def canonical_name(value: str) -> str:
    return _NAME_NORMALIZER.sub("-", value).lower().strip()


def validate_intent(request: AnalysisRequest, intent: ParsedIntent) -> list[str]:
    errors: list[str] = []
    if not re.fullmatch(r"3\.(8|9|10|11|12|13|14)", request.python_version.strip()):
        errors.append("The current model supports Python 3.8 through 3.14 only.")
    if not intent.requirements:
        errors.append("No installable package requirement was found.")
        return errors

    source_lines = {
        line.strip()
        for line in request.requirements_text.splitlines()
        if line.strip() and not line.lstrip().startswith(("#", "-"))
    }
    for item in intent.requirements:
        if item.raw.strip() not in source_lines:
            errors.append(
                f"The parsed requirement '{item.raw}' was not found verbatim in requirements.txt."
            )
            continue
        name_match = re.match(r"\s*([A-Za-z0-9][A-Za-z0-9._-]*)", item.raw)
        if not name_match or canonical_name(name_match.group(1)) != canonical_name(item.name):
            errors.append(f"The parsed package name does not match '{item.raw}'.")
        pin_match = re.search(r"(?<![!<>~])==\s*([^\s;,]+)", item.raw)
        source_version = pin_match.group(1) if pin_match else None
        if source_version != item.version:
            errors.append(
                f"The parsed exact version for '{item.name}' does not match requirements.txt."
            )

    names = [canonical_name(item.name) for item in intent.requirements]
    if len(names) != len(set(names)):
        errors.append("The requirements contain the same package more than once.")
    if canonical_name(intent.target_package) not in set(names):
        errors.append(
            f"The requested package '{intent.target_package}' is not present in requirements.txt."
        )
    if not intent.requested_version.strip():
        errors.append("The requested new version is missing.")
    return errors
