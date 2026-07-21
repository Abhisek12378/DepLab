from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from packaging.requirements import InvalidRequirement, Requirement
from packaging.version import InvalidVersion, Version

try:
    from build_changelog_features import augment as augment_changelogs
except ImportError:  # Imported as scripts.build_expanded_dataset in tests.
    from scripts.build_changelog_features import augment as augment_changelogs


DATASET_VERSION = "2.0.0"
DATASET_ID = f"deplab-expanded-development-v{DATASET_VERSION}"

BASE_NUMERIC_COLUMNS = [
    "package_a_version_major",
    "package_a_version_minor",
    "package_a_version_patch",
    "package_b_version_major",
    "package_b_version_minor",
    "package_b_version_patch",
    "python_major",
    "python_minor",
    "release_date_distance_days",
    "package_a_release_ordinal",
    "package_b_release_ordinal",
    "package_a_requires_dist_count",
    "package_b_requires_dist_count",
    "package_a_declares_package_b",
    "package_b_declares_package_a",
    "package_a_requirement_has_upper_bound",
    "package_b_requirement_has_upper_bound",
    "package_a_requirement_has_lower_bound",
    "package_b_requirement_has_lower_bound",
    "package_a_requirement_has_exact_pin",
    "package_b_requirement_has_exact_pin",
    "package_a_requirement_allows_b",
    "package_b_requirement_allows_a",
    "package_a_eligible_wheel_count",
    "package_b_eligible_wheel_count",
    "package_a_wheel_bytes",
    "package_b_wheel_bytes",
    "top_level_wheel_bytes",
    "package_a_has_native_extensions",
    "package_b_has_native_extensions",
    "either_top_level_has_native_extensions",
]

CATEGORICAL_COLUMNS = [
    "python_version",
    "package_a_requires_python",
    "package_b_requires_python",
    "package_a_wheel_python_tag",
    "package_b_wheel_python_tag",
    "package_a_wheel_abi_tag",
    "package_b_wheel_abi_tag",
    "package_a_wheel_platform_tag",
    "package_b_wheel_platform_tag",
]

IDENTIFIER_COLUMNS = {
    "dataset_version",
    "matrix_order",
    "experiment_id",
    "family",
    "package_a_name",
    "package_a_version",
    "package_b_name",
    "package_b_version",
}
LABEL_COLUMNS = {"outcome", "compatibility_label", "is_compatible", "is_failure"}
CONTEXT_COLUMNS = {
    "package_a_release_date",
    "package_b_release_date",
    "package_a_requirement_on_b",
    "package_b_requirement_on_a",
    "package_a_wheel_filename",
    "package_b_wheel_filename",
}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build leakage-safe expanded development and sealed-holdout features"
    )
    parser.add_argument("--development-matrix", type=Path, default=Path("configs/expanded-development-matrix.json"))
    parser.add_argument("--holdout-matrix", type=Path, default=Path("configs/expanded-final-holdout-matrix.json"))
    parser.add_argument("--results", type=Path, default=Path("outputs/expanded-development-results.jsonl"))
    parser.add_argument("--audit", type=Path, default=Path("outputs/expanded-development-audit.json"))
    parser.add_argument("--catalog", type=Path, default=Path("outputs/expanded-package-catalog.jsonl"))
    parser.add_argument("--changelogs", type=Path, default=Path("outputs/changelog-catalog-expanded-v1.2.0.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path(f"outputs/{DATASET_ID}"))
    args = parser.parse_args()

    audit = _read_json(args.audit)
    if not audit.get("valid"):
        raise ValueError("refusing to build features from development results that failed audit")
    development_matrix = _read_json(args.development_matrix)
    holdout_matrix = _read_json(args.holdout_matrix)
    results = _read_jsonl(args.results)
    catalog_rows = _read_jsonl(args.catalog)
    changelog_rows = _read_jsonl(args.changelogs)

    development = build_features(
        development_matrix, catalog_rows, changelog_rows, results=results
    )
    holdout = build_features(
        holdout_matrix, catalog_rows, changelog_rows, results=None
    )
    _validate_frames(development, holdout, audit)

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    development_path = output / "features.csv"
    holdout_path = output / "final-holdout-inputs.csv"
    development.to_csv(
        development_path, index=False, encoding="utf-8", lineterminator="\n", quoting=csv.QUOTE_MINIMAL
    )
    holdout.to_csv(
        holdout_path, index=False, encoding="utf-8", lineterminator="\n", quoting=csv.QUOTE_MINIMAL
    )

    numeric_columns = BASE_NUMERIC_COLUMNS + _changelog_model_columns(development)
    model_columns = numeric_columns + CATEGORICAL_COLUMNS
    dictionary = _feature_dictionary(development, set(model_columns))
    dictionary_path = output / "feature-dictionary.csv"
    pd.DataFrame(dictionary).to_csv(
        dictionary_path, index=False, encoding="utf-8", lineterminator="\n"
    )
    policy = {
        "schema_version": "1.0.0",
        "target": "is_failure",
        "positive_class": "compatibility_failure",
        "numeric_columns": numeric_columns,
        "categorical_columns": CATEGORICAL_COLUMNS,
        "group_column": "family",
        "forbidden_input_classes": [
            "outcomes and derived labels",
            "import and smoke-test results",
            "resolution output and installed environment",
            "runtime duration, cache, network and machine measurements",
            "errors, logs, hashes and unique identifiers",
            "package names, family identity and exact version strings",
        ],
    }
    policy_path = output / "model-input-policy.json"
    policy_path.write_text(json.dumps(policy, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    source_paths = {
        "development_matrix": args.development_matrix,
        "final_holdout_matrix": args.holdout_matrix,
        "development_results": args.results,
        "development_audit": args.audit,
        "pypi_catalog": args.catalog,
        "changelog_catalog": args.changelogs,
    }
    manifest = {
        "dataset_id": DATASET_ID,
        "schema_version": "1.0.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "development_rows": len(development),
        "sealed_final_holdout_rows": len(holdout),
        "development_families": sorted(development["family"].unique().tolist()),
        "final_holdout_families": sorted(holdout["family"].unique().tolist()),
        "development_outcomes": {
            str(key): int(value)
            for key, value in development["outcome"].value_counts().sort_index().items()
        },
        "failure_rows": int(development["is_failure"].sum()),
        "failure_percentage": round(float(development["is_failure"].mean() * 100), 2),
        "model_input_columns": len(model_columns),
        "holdout_outcomes_present": False,
        "source_sha256": {name: _sha256(path) for name, path in source_paths.items()},
        "files": {},
        "notes": [
            "The development table contains measured labels but only pre-run columns are eligible model inputs.",
            "The final holdout table contains prediction inputs only; no holdout experiment outcomes have been collected.",
            "CSV preserves exact version strings when they are explicitly read as strings.",
        ],
    }
    for path in (development_path, holdout_path, dictionary_path, policy_path):
        manifest["files"][path.name] = {
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
    (output / "dataset-manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "README.md").write_text(_readme(manifest), encoding="utf-8")
    _write_checksums(output)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


def build_features(
    matrix: dict[str, Any],
    catalog_rows: list[dict[str, Any]],
    changelog_rows: list[dict[str, Any]],
    results: list[dict[str, Any]] | None,
) -> pd.DataFrame:
    experiments = matrix.get("experiments")
    if not isinstance(experiments, list) or not experiments:
        raise ValueError("matrix has no experiments")
    catalog = {
        (
            _canonical(row["release"]["name"]),
            str(row["release"]["version"]),
            str(row["target"]["python_version"]),
        ): row["release"]
        for row in catalog_rows
    }
    result_by_id = None
    if results is not None:
        result_by_id = {str(row["experiment_id"]): row for row in results}
        if len(result_by_id) != len(results):
            raise ValueError("development results contain duplicate experiment IDs")

    rows = []
    for order, experiment in enumerate(experiments, 1):
        name_a, version_a = _pin(experiment["package_a"])
        name_b, version_b = _pin(experiment["package_b"])
        python_version = str(experiment["python"])
        experiment_id = _experiment_id(experiment)
        release_a = catalog[(_canonical(name_a), version_a, python_version)]
        release_b = catalog[(_canonical(name_b), version_b, python_version)]
        outcome = None
        if result_by_id is not None:
            result = result_by_id.get(experiment_id)
            if result is None:
                raise ValueError(f"missing audited result {experiment_id}")
            outcome = str(result["outcome"])
        rows.append(
            _feature_row(
                order,
                experiment_id,
                str(experiment["family"]),
                name_a,
                version_a,
                name_b,
                version_b,
                python_version,
                release_a,
                release_b,
                outcome,
            )
        )
    frame, _ = augment_changelogs(pd.DataFrame(rows), changelog_rows)
    return frame


def _feature_row(
    order: int,
    experiment_id: str,
    family: str,
    name_a: str,
    version_a: str,
    name_b: str,
    version_b: str,
    python_version: str,
    release_a: dict[str, Any],
    release_b: dict[str, Any],
    outcome: str | None,
) -> dict[str, Any]:
    wheel_a = _selected_wheel(release_a)
    wheel_b = _selected_wheel(release_b)
    requirement_a = _direct_requirement(release_a.get("requires_dist", []), name_b)
    requirement_b = _direct_requirement(release_b.get("requires_dist", []), name_a)
    date_a = _parse_date(release_a.get("release_date"))
    date_b = _parse_date(release_b.get("release_date"))
    parts_a = _version_parts(version_a)
    parts_b = _version_parts(version_b)
    python_parts = _version_parts(python_version)
    row = {
        "dataset_version": DATASET_VERSION,
        "matrix_order": order,
        "experiment_id": experiment_id,
        "family": family,
        "package_a_name": name_a,
        "package_a_version": version_a,
        "package_a_version_major": parts_a[0],
        "package_a_version_minor": parts_a[1],
        "package_a_version_patch": parts_a[2],
        "package_b_name": name_b,
        "package_b_version": version_b,
        "package_b_version_major": parts_b[0],
        "package_b_version_minor": parts_b[1],
        "package_b_version_patch": parts_b[2],
        "python_version": python_version,
        "python_major": python_parts[0],
        "python_minor": python_parts[1],
        "package_a_requires_python": release_a.get("requires_python"),
        "package_b_requires_python": release_b.get("requires_python"),
        "package_a_release_date": release_a.get("release_date"),
        "package_b_release_date": release_b.get("release_date"),
        "release_date_distance_days": abs((date_a - date_b).days) if date_a and date_b else None,
        "package_a_release_ordinal": date_a.toordinal() if date_a else None,
        "package_b_release_ordinal": date_b.toordinal() if date_b else None,
        "package_a_requires_dist_count": len(release_a.get("requires_dist", [])),
        "package_b_requires_dist_count": len(release_b.get("requires_dist", [])),
        "package_a_declares_package_b": requirement_a is not None,
        "package_b_declares_package_a": requirement_b is not None,
        "package_a_requirement_on_b": requirement_a,
        "package_b_requirement_on_a": requirement_b,
        "package_a_requirement_has_upper_bound": _has_operator(requirement_a, "<"),
        "package_b_requirement_has_upper_bound": _has_operator(requirement_b, "<"),
        "package_a_requirement_has_lower_bound": _has_operator(requirement_a, ">"),
        "package_b_requirement_has_lower_bound": _has_operator(requirement_b, ">"),
        "package_a_requirement_has_exact_pin": _has_operator(requirement_a, "=="),
        "package_b_requirement_has_exact_pin": _has_operator(requirement_b, "=="),
        "package_a_requirement_allows_b": _requirement_allows(requirement_a, version_b),
        "package_b_requirement_allows_a": _requirement_allows(requirement_b, version_a),
        "package_a_eligible_wheel_count": _eligible_wheel_count(release_a),
        "package_b_eligible_wheel_count": _eligible_wheel_count(release_b),
        "package_a_wheel_filename": wheel_a["filename"],
        "package_b_wheel_filename": wheel_b["filename"],
        "package_a_wheel_python_tag": wheel_a["python_tag"],
        "package_b_wheel_python_tag": wheel_b["python_tag"],
        "package_a_wheel_abi_tag": wheel_a["abi_tag"],
        "package_b_wheel_abi_tag": wheel_b["abi_tag"],
        "package_a_wheel_platform_tag": wheel_a["platform_tag"],
        "package_b_wheel_platform_tag": wheel_b["platform_tag"],
        "package_a_wheel_bytes": int(wheel_a.get("size") or 0),
        "package_b_wheel_bytes": int(wheel_b.get("size") or 0),
        "top_level_wheel_bytes": int(wheel_a.get("size") or 0) + int(wheel_b.get("size") or 0),
        "package_a_has_native_extensions": bool(wheel_a.get("has_native_extensions")),
        "package_b_has_native_extensions": bool(wheel_b.get("has_native_extensions")),
        "either_top_level_has_native_extensions": bool(
            wheel_a.get("has_native_extensions") or wheel_b.get("has_native_extensions")
        ),
    }
    if outcome is not None:
        row.update(
            {
                "outcome": outcome,
                "compatibility_label": "compatible" if outcome == "pass" else "incompatible",
                "is_compatible": outcome == "pass",
                "is_failure": outcome != "pass",
            }
        )
    return row


def _validate_frames(development: pd.DataFrame, holdout: pd.DataFrame, audit: dict[str, Any]) -> None:
    expected = int(audit["expected_experiments"])
    if len(development) != expected or development["experiment_id"].nunique() != expected:
        raise ValueError("development features are not one-to-one with the audited matrix")
    if len(holdout) != 840 or holdout["experiment_id"].nunique() != 840:
        raise ValueError("final holdout features are not one-to-one with the frozen 840-row matrix")
    if set(development["experiment_id"]) & set(holdout["experiment_id"]):
        raise ValueError("development and final holdout experiment IDs overlap")
    if LABEL_COLUMNS & set(holdout.columns):
        raise ValueError("sealed holdout input table must not contain outcome labels")
    if int(development["is_failure"].sum()) != 1044:
        raise ValueError("development failure count does not match the trusted audit")


def _changelog_model_columns(frame: pd.DataFrame) -> list[str]:
    return sorted(
        column
        for column in frame.columns
        if column.startswith("package_a_changelog_")
        or column.startswith("package_b_changelog_")
        or column.startswith("changelog_")
    )


def _feature_dictionary(frame: pd.DataFrame, model_columns: set[str]) -> list[dict[str, str]]:
    rows = []
    for column in frame.columns:
        if column in IDENTIFIER_COLUMNS:
            role = "identifier_do_not_train"
        elif column in LABEL_COLUMNS:
            role = "label_do_not_train"
        elif column in CONTEXT_COLUMNS:
            role = "raw_context_do_not_train"
        elif column in model_columns:
            role = "inference_safe_input"
        else:
            role = "context_do_not_train"
        rows.append(
            {
                "column": column,
                "data_type": str(frame[column].dtype),
                "role": role,
                "description": _description(column, role),
            }
        )
    return rows


def _description(column: str, role: str) -> str:
    special = {
        "is_failure": "Binary target: true for resolution, import, or interoperability smoke-test failure.",
        "family": "Package-pair family used only for grouped evaluation, never as a model input.",
        "package_a_requirement_allows_b": "Whether package A's published direct requirement accepts the selected B version.",
        "package_b_requirement_allows_a": "Whether package B's published direct requirement accepts the selected A version.",
    }
    return special.get(column, f"{column.replace('_', ' ').capitalize()} ({role}).")


def _requirement_allows(requirement: str | None, selected_version: str) -> bool | None:
    if not requirement:
        return None
    try:
        parsed = Requirement(requirement)
        return parsed.specifier.contains(Version(selected_version), prereleases=True)
    except (InvalidRequirement, InvalidVersion):
        return None


def _has_operator(requirement: str | None, operator: str) -> bool:
    if not requirement:
        return False
    try:
        return any(spec.operator == operator or spec.operator.startswith(operator) for spec in Requirement(requirement).specifier)
    except InvalidRequirement:
        return operator in requirement


def _direct_requirement(requirements: list[str], target: str) -> str | None:
    target_name = _canonical(target)
    for requirement in requirements:
        match = re.match(r"\s*([A-Za-z0-9_.-]+)", str(requirement))
        if match and _canonical(match.group(1)) == target_name:
            return str(requirement)
    return None


def _selected_wheel(release: dict[str, Any]) -> dict[str, Any]:
    candidates = sorted(
        (
            wheel
            for wheel in release.get("wheels", [])
            if wheel.get("compatible") and not wheel.get("yanked")
        ),
        key=lambda wheel: str(wheel["filename"]),
    )
    if not candidates:
        raise ValueError(f"no eligible wheel for {release.get('name')}=={release.get('version')}")
    return candidates[0]


def _eligible_wheel_count(release: dict[str, Any]) -> int:
    return sum(bool(wheel.get("compatible")) and not bool(wheel.get("yanked")) for wheel in release.get("wheels", []))


def _experiment_id(row: dict[str, Any]) -> str:
    name_a, version_a = _pin(row["package_a"])
    name_b, version_b = _pin(row["package_b"])
    raw = "|".join((name_a.lower(), version_a, name_b.lower(), version_b, str(row["python"]), "linux", "x86_64"))
    return hashlib.sha256(raw.encode()).hexdigest()[:20]


def _pin(value: str) -> tuple[str, str]:
    name, version = str(value).split("==", 1)
    return name, version


def _version_parts(version: str) -> tuple[int | None, int | None, int | None]:
    numbers = [int(value) for value in re.findall(r"\d+", version)[:3]]
    padded: list[int | None] = numbers + [None, None, None]
    return padded[0], padded[1], padded[2]


def _parse_date(value: Any) -> datetime | None:
    if not value:
        return None
    return datetime.fromisoformat(str(value).replace("Z", "+00:00"))


def _canonical(value: str) -> str:
    return re.sub(r"[-_.]+", "-", str(value)).lower()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain an object")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_checksums(output: Path) -> None:
    checksum = output / "SHA256SUMS.txt"
    lines = [
        f"{_sha256(path)}  {path.name}"
        for path in sorted(output.iterdir())
        if path.is_file() and path != checksum
    ]
    checksum.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _readme(manifest: dict[str, Any]) -> str:
    return f"""# DepLab expanded development dataset v{DATASET_VERSION}

This processed dataset contains {manifest['development_rows']} audited development experiments and {manifest['sealed_final_holdout_rows']} label-free final-holdout inputs.

## Important files

- `features.csv`: development inputs plus measured labels.
- `final-holdout-inputs.csv`: the frozen 840 prediction inputs, with no outcomes.
- `feature-dictionary.csv`: the role and type of every column.
- `model-input-policy.json`: the exact inference-safe columns permitted for training.
- `dataset-manifest.json`: source hashes, counts and provenance.
- `SHA256SUMS.txt`: integrity checksums.

The model policy excludes import results, smoke-test results, installed packages, resolution output, errors, timings, resource measurements, identifiers, package names, family names, and exact version strings. Changelog and PyPI fields are allowed because they are available before installation.

When loading CSV, read `experiment_id`, `python_version`, and both exact package-version columns as strings.
"""


if __name__ == "__main__":
    raise SystemExit(main())
