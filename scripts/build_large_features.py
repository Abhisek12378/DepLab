from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from packaging.version import InvalidVersion, Version

try:
    from build_changelog_features import SIGNAL_NAMES
    from build_expanded_dataset import (
        BASE_NUMERIC_COLUMNS,
        CATEGORICAL_COLUMNS,
        _experiment_id,
        _feature_row,
        _read_jsonl,
    )
except ImportError:
    from scripts.build_changelog_features import SIGNAL_NAMES
    from scripts.build_expanded_dataset import (
        BASE_NUMERIC_COLUMNS,
        CATEGORICAL_COLUMNS,
        _experiment_id,
        _feature_row,
        _read_jsonl,
    )


DATASET_VERSION = "3.0.0"
DATASET_ID = f"deplab-large-features-v{DATASET_VERSION}"
RELEASE_CUTOFF = pd.Timestamp("2026-07-25T00:00:00Z")

DERIVED_NUMERIC_COLUMNS = [
    "package_a_selected_version_rank",
    "package_b_selected_version_rank",
    "package_a_selected_version_percentile",
    "package_b_selected_version_percentile",
    "package_a_release_age_days",
    "package_b_release_age_days",
    "package_a_is_latest_selected_version",
    "package_b_is_latest_selected_version",
    "published_constraint_blocked",
    "published_constraint_allows",
    "published_constraint_unknown",
    "direct_dependency_edges",
    "both_top_level_native",
    "exactly_one_top_level_native",
    "wheel_python_tags_match",
    "wheel_abi_tags_match",
    "wheel_platform_tags_match",
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
LABEL_COLUMNS = {
    "outcome",
    "compatibility_label",
    "is_compatible",
    "is_failure",
    "failure_stage",
}
RAW_CONTEXT_COLUMNS = {
    "package_a_release_date",
    "package_b_release_date",
    "package_a_requirement_on_b",
    "package_b_requirement_on_a",
    "package_a_wheel_filename",
    "package_b_wheel_filename",
}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build inference-safe features for the DepLab large dataset"
    )
    parser.add_argument(
        "--development-matrix",
        type=Path,
        default=Path("configs/large-development-matrix-v3.0.0.json"),
    )
    parser.add_argument(
        "--validation-matrix",
        type=Path,
        default=Path("configs/large-validation-matrix-v3.0.0.json"),
    )
    parser.add_argument(
        "--final-test-matrix",
        type=Path,
        default=Path("configs/large-final-test-matrix-v3.0.0.json"),
    )
    parser.add_argument(
        "--development-results",
        type=Path,
        default=Path("outputs/large-development-results-v3.0.0.jsonl"),
    )
    parser.add_argument(
        "--development-audit",
        type=Path,
        default=Path("outputs/large-development-audit-v3.0.0.json"),
    )
    parser.add_argument(
        "--catalog",
        type=Path,
        default=Path("outputs/large-package-catalog-v3.0.0.jsonl"),
    )
    parser.add_argument(
        "--changelogs",
        type=Path,
        default=Path("outputs/changelog-catalog-expanded-v1.2.0.jsonl"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(f"outputs/{DATASET_ID}"),
    )
    args = parser.parse_args()

    audit = _read_json(args.development_audit)
    _validate_development_audit(audit)
    catalog_rows = _read_jsonl(args.catalog)
    changelog_rows = _read_jsonl(args.changelogs) if args.changelogs.exists() else []
    results = _read_jsonl(args.development_results)
    matrices = {
        "development": _read_json(args.development_matrix),
        "validation": _read_json(args.validation_matrix),
        "final_test": _read_json(args.final_test_matrix),
    }

    release_catalog = _release_catalog(catalog_rows)
    release_ranks = _release_ranks(release_catalog)
    development = build_features(
        matrices["development"],
        release_catalog,
        release_ranks,
        changelog_rows,
        results,
    )
    validation = build_features(
        matrices["validation"],
        release_catalog,
        release_ranks,
        changelog_rows,
        None,
    )
    final_test = build_features(
        matrices["final_test"],
        release_catalog,
        release_ranks,
        changelog_rows,
        None,
    )
    _validate_frames(development, validation, final_test, audit)

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    paths = {
        "development_features": output / "development-features.csv",
        "validation_inputs": output / "validation-inputs.csv",
        "final_test_inputs": output / "final-test-inputs.csv",
    }
    _write_frame(development, paths["development_features"])
    _write_frame(validation, paths["validation_inputs"])
    _write_frame(final_test, paths["final_test_inputs"])

    model_columns = _model_columns(development)
    policy_path = output / "model-input-policy.json"
    policy = _model_input_policy(model_columns)
    policy_path.write_text(
        json.dumps(policy, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    dictionary_path = output / "feature-dictionary.csv"
    _feature_dictionary(development, set(model_columns)).to_csv(
        dictionary_path,
        index=False,
        encoding="utf-8",
        lineterminator="\n",
    )

    source_paths = {
        "development_matrix": args.development_matrix,
        "validation_matrix": args.validation_matrix,
        "final_test_matrix": args.final_test_matrix,
        "development_results": args.development_results,
        "development_audit": args.development_audit,
        "pypi_catalog": args.catalog,
    }
    if args.changelogs.exists():
        source_paths["optional_changelog_catalog"] = args.changelogs
    manifest = _manifest(
        development,
        validation,
        final_test,
        paths,
        policy_path,
        dictionary_path,
        source_paths,
    )
    manifest_path = output / "dataset-manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output / "README.md").write_text(_readme(manifest), encoding="utf-8")
    _write_checksums(output)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


def build_features(
    matrix: dict[str, Any],
    release_catalog: dict[tuple[str, str, str], dict[str, Any]],
    release_ranks: dict[tuple[str, str], tuple[int, int]],
    changelog_rows: list[dict[str, Any]],
    results: list[dict[str, Any]] | None,
) -> pd.DataFrame:
    experiments = matrix.get("experiments")
    if not isinstance(experiments, list) or not experiments:
        raise ValueError("matrix has no experiments")
    result_by_id = _result_index(results)
    rows: list[dict[str, Any]] = []
    for order, experiment in enumerate(experiments, 1):
        name_a, version_a = _pin(str(experiment["package_a"]))
        name_b, version_b = _pin(str(experiment["package_b"]))
        python_version = str(experiment["python"])
        experiment_id = _experiment_id(experiment)
        release_a = _catalog_release(
            release_catalog, name_a, version_a, python_version
        )
        release_b = _catalog_release(
            release_catalog, name_b, version_b, python_version
        )
        outcome = _outcome(result_by_id, experiment_id)
        row = _feature_row(
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
        row["dataset_version"] = DATASET_VERSION
        if outcome is not None:
            row["failure_stage"] = _failure_stage(outcome)
        rows.append(row)
    frame = pd.DataFrame(rows)
    frame = _add_release_rank_features(frame, release_ranks)
    frame = _add_pair_features(frame)
    frame = _add_optional_changelog_features(frame, changelog_rows)
    return frame


def _result_index(
    results: list[dict[str, Any]] | None,
) -> dict[str, dict[str, Any]] | None:
    if results is None:
        return None
    indexed = {str(row["experiment_id"]): row for row in results}
    if len(indexed) != len(results):
        raise ValueError("development results contain duplicate experiment IDs")
    return indexed


def _outcome(
    result_by_id: dict[str, dict[str, Any]] | None,
    experiment_id: str,
) -> str | None:
    if result_by_id is None:
        return None
    result = result_by_id.get(experiment_id)
    if result is None:
        raise ValueError(f"missing audited development result {experiment_id}")
    if not bool(result.get("measured")):
        raise ValueError(f"development result {experiment_id} is not measured")
    return str(result["outcome"])


def _release_catalog(
    rows: list[dict[str, Any]],
) -> dict[tuple[str, str, str], dict[str, Any]]:
    catalog: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in rows:
        release = dict(row["release"])
        key = (
            _canonical(str(release["name"])),
            str(release["version"]),
            str(row["target"]["python_version"]),
        )
        previous = catalog.setdefault(key, release)
        if previous != release:
            raise ValueError(f"conflicting catalog rows for {key}")
    if not catalog:
        raise ValueError("PyPI catalog is empty")
    return catalog


def _catalog_release(
    catalog: dict[tuple[str, str, str], dict[str, Any]],
    name: str,
    version: str,
    python_version: str,
) -> dict[str, Any]:
    key = (_canonical(name), version, python_version)
    if key not in catalog:
        raise ValueError(
            f"missing PyPI catalog row for {name}=={version} on Python {python_version}"
        )
    return catalog[key]


def _release_ranks(
    catalog: dict[tuple[str, str, str], dict[str, Any]],
) -> dict[tuple[str, str], tuple[int, int]]:
    versions_by_package: dict[str, set[str]] = {}
    for package, version, _ in catalog:
        versions_by_package.setdefault(package, set()).add(version)
    result: dict[tuple[str, str], tuple[int, int]] = {}
    for package, versions in versions_by_package.items():
        ordered = sorted(versions, key=_version_sort_key)
        for rank, version in enumerate(ordered):
            result[(package, version)] = (rank, len(ordered))
    return result


def _version_sort_key(value: str) -> tuple[int, Any]:
    try:
        return 0, Version(value)
    except InvalidVersion:
        return 1, value


def _add_release_rank_features(
    frame: pd.DataFrame,
    ranks: dict[tuple[str, str], tuple[int, int]],
) -> pd.DataFrame:
    added: dict[str, list[Any]] = {}
    for side in ("a", "b"):
        selected_ranks: list[int] = []
        percentiles: list[float] = []
        latest: list[bool] = []
        ages: list[float] = []
        for name, version, released in zip(
            frame[f"package_{side}_name"].astype(str),
            frame[f"package_{side}_version"].astype(str),
            frame[f"package_{side}_release_date"],
        ):
            rank, count = ranks[(_canonical(name), version)]
            selected_ranks.append(rank)
            percentiles.append(rank / max(1, count - 1))
            latest.append(rank == count - 1)
            timestamp = pd.to_datetime(released, utc=True, errors="coerce")
            ages.append(
                max(0.0, float((RELEASE_CUTOFF - timestamp).days))
                if not pd.isna(timestamp)
                else np.nan
            )
        added[f"package_{side}_selected_version_rank"] = selected_ranks
        added[f"package_{side}_selected_version_percentile"] = percentiles
        added[f"package_{side}_is_latest_selected_version"] = latest
        added[f"package_{side}_release_age_days"] = ages
    return pd.concat([frame, pd.DataFrame(added, index=frame.index)], axis=1)


def _add_pair_features(frame: pd.DataFrame) -> pd.DataFrame:
    allows_a = frame["package_a_requirement_allows_b"]
    allows_b = frame["package_b_requirement_allows_a"]
    declares_a = frame["package_a_declares_package_b"].astype(bool)
    declares_b = frame["package_b_declares_package_a"].astype(bool)
    declared_allowances = pd.concat(
        [allows_a.where(declares_a), allows_b.where(declares_b)],
        axis=1,
    )
    known = declared_allowances.notna()
    blocked = (declared_allowances.eq(False) & known).any(axis=1)
    allows = known.any(axis=1) & ~blocked
    native_a = frame["package_a_has_native_extensions"].astype(bool)
    native_b = frame["package_b_has_native_extensions"].astype(bool)
    added = pd.DataFrame(
        {
            "published_constraint_blocked": blocked,
            "published_constraint_allows": allows,
            "published_constraint_unknown": ~known.any(axis=1),
            "direct_dependency_edges": declares_a.astype(int)
            + declares_b.astype(int),
            "both_top_level_native": native_a & native_b,
            "exactly_one_top_level_native": native_a ^ native_b,
            "wheel_python_tags_match": frame["package_a_wheel_python_tag"].astype(str)
            == frame["package_b_wheel_python_tag"].astype(str),
            "wheel_abi_tags_match": frame["package_a_wheel_abi_tag"].astype(str)
            == frame["package_b_wheel_abi_tag"].astype(str),
            "wheel_platform_tags_match": frame[
                "package_a_wheel_platform_tag"
            ].astype(str)
            == frame["package_b_wheel_platform_tag"].astype(str),
        },
        index=frame.index,
    )
    return pd.concat([frame, added], axis=1)


def _add_optional_changelog_features(
    frame: pd.DataFrame,
    rows: list[dict[str, Any]],
) -> pd.DataFrame:
    catalog = {
        (_canonical(str(row["package"])), str(row["version"])): row for row in rows
    }
    added: dict[str, list[Any]] = {}
    records_by_side: dict[str, list[dict[str, Any] | None]] = {}
    for side in ("a", "b"):
        records = [
            catalog.get((_canonical(name), version))
            for name, version in zip(
                frame[f"package_{side}_name"].astype(str),
                frame[f"package_{side}_version"].astype(str),
            )
        ]
        records_by_side[side] = records
        added[f"package_{side}_changelog_available"] = [
            record is not None for record in records
        ]
        added[f"package_{side}_changelog_selected_characters"] = [
            int((record or {}).get("signals", {}).get("selected_characters", 0))
            for record in records
        ]
        added[f"package_{side}_changelog_version_section_found"] = [
            bool((record or {}).get("version_section_found", False))
            for record in records
        ]
        for signal in SIGNAL_NAMES:
            added[f"package_{side}_changelog_{signal}_count"] = [
                int((record or {}).get("signals", {}).get(f"{signal}_count", 0))
                for record in records
            ]
            added[f"package_{side}_changelog_{signal}_flag"] = [
                bool((record or {}).get("signals", {}).get(f"{signal}_flag", False))
                for record in records
            ]

    mentions_a: list[int] = []
    mentions_b: list[int] = []
    for index, row in frame.iterrows():
        record_a = records_by_side["a"][index]
        record_b = records_by_side["b"][index]
        mentions_a.append(
            _mention_count(record_a, str(row["package_b_name"]))
        )
        mentions_b.append(
            _mention_count(record_b, str(row["package_a_name"]))
        )
    added["package_a_changelog_mentions_b_count"] = mentions_a
    added["package_b_changelog_mentions_a_count"] = mentions_b
    result = pd.concat([frame, pd.DataFrame(added, index=frame.index)], axis=1)
    for signal in ("breaking", "removal", "deprecation", "api", "abi"):
        result[f"changelog_either_{signal}_flag"] = (
            result[f"package_a_changelog_{signal}_flag"].astype(bool)
            | result[f"package_b_changelog_{signal}_flag"].astype(bool)
        )
    result["changelog_pair_mentions"] = (
        result["package_a_changelog_mentions_b_count"]
        + result["package_b_changelog_mentions_a_count"]
    )
    return result


def _mention_count(record: dict[str, Any] | None, target: str) -> int:
    if record is None:
        return 0
    mentions = record.get("signals", {}).get("package_mentions", {})
    return int(mentions.get(_canonical(target), 0))


def _model_columns(frame: pd.DataFrame) -> list[str]:
    changelog = sorted(
        column
        for column in frame.columns
        if column.startswith("package_a_changelog_")
        or column.startswith("package_b_changelog_")
        or column.startswith("changelog_")
    )
    return list(dict.fromkeys(BASE_NUMERIC_COLUMNS + DERIVED_NUMERIC_COLUMNS + changelog))


def _model_input_policy(numeric_columns: list[str]) -> dict[str, Any]:
    return {
        "schema_version": "3.0.0",
        "target": "is_failure",
        "outcome_subtype_target": "outcome",
        "positive_class": "compatibility_failure",
        "numeric_columns": numeric_columns,
        "categorical_columns": CATEGORICAL_COLUMNS,
        "group_column": "family",
        "inference_contract": (
            "Every model input is available from pins, target Python, PyPI release "
            "metadata, wheel metadata, or frozen changelog evidence before installation."
        ),
        "forbidden_input_classes": [
            "outcomes and derived labels",
            "resolution, installation, import and smoke-test results",
            "installed or resolved environments",
            "runtime duration and machine, cache or network measurements",
            "errors, logs, hashes and unique experiment identifiers",
            "package names, family identity and exact version strings",
        ],
        "deterministic_fact_columns": [
            "published_constraint_blocked",
            "published_constraint_allows",
            "published_constraint_unknown",
        ],
    }


def _feature_dictionary(
    frame: pd.DataFrame, model_columns: set[str]
) -> pd.DataFrame:
    rows = []
    for column in frame.columns:
        if column in IDENTIFIER_COLUMNS:
            role = "identifier_do_not_train"
        elif column in LABEL_COLUMNS:
            role = "label_do_not_train"
        elif column in RAW_CONTEXT_COLUMNS:
            role = "raw_context_do_not_train"
        elif column in model_columns or column in CATEGORICAL_COLUMNS:
            role = "inference_safe_input"
        else:
            role = "context_do_not_train"
        rows.append(
            {
                "column": column,
                "data_type": str(frame[column].dtype),
                "role": role,
            }
        )
    return pd.DataFrame(rows)


def _validate_development_audit(audit: dict[str, Any]) -> None:
    checks = {
        "structural_valid": bool(audit.get("structural_valid")),
        "complete": bool(audit.get("complete")),
        "no missing rows": int(audit.get("missing_count", -1)) == 0,
        "no duplicate rows": int(audit.get("duplicate_count", -1)) == 0,
        "no infrastructure failures": int(
            audit.get("infrastructure_failure_count", -1)
        )
        == 0,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError(
            "refusing to build development features: " + ", ".join(failed)
        )


def _validate_frames(
    development: pd.DataFrame,
    validation: pd.DataFrame,
    final_test: pd.DataFrame,
    audit: dict[str, Any],
) -> None:
    expected = int(audit["expected_experiments"])
    _validate_frame_identity(development, expected, "development")
    _validate_frame_identity(validation, 3432, "validation")
    _validate_frame_identity(final_test, 3158, "final test")
    id_sets = [
        set(development["experiment_id"]),
        set(validation["experiment_id"]),
        set(final_test["experiment_id"]),
    ]
    if id_sets[0] & id_sets[1] or id_sets[0] & id_sets[2] or id_sets[1] & id_sets[2]:
        raise ValueError("development, validation and final-test experiment IDs overlap")
    for name, frame in (("validation", validation), ("final test", final_test)):
        leaked = LABEL_COLUMNS & set(frame.columns)
        if leaked:
            raise ValueError(f"{name} inputs contain leaked labels: {sorted(leaked)}")
    expected_outcomes = {
        str(key): int(value)
        for key, value in dict(audit["outcome_counts"]).items()
    }
    actual_outcomes = {
        str(key): int(value)
        for key, value in development["outcome"].value_counts().items()
    }
    if actual_outcomes != expected_outcomes:
        raise ValueError("development outcome counts differ from trusted audit")


def _validate_frame_identity(
    frame: pd.DataFrame, expected: int, name: str
) -> None:
    if len(frame) != expected or frame["experiment_id"].nunique() != expected:
        raise ValueError(f"{name} features are not one-to-one with the matrix")


def _manifest(
    development: pd.DataFrame,
    validation: pd.DataFrame,
    final_test: pd.DataFrame,
    paths: dict[str, Path],
    policy_path: Path,
    dictionary_path: Path,
    source_paths: dict[str, Path],
) -> dict[str, Any]:
    files = {**paths, "model_input_policy": policy_path, "feature_dictionary": dictionary_path}
    return {
        "dataset_id": DATASET_ID,
        "schema_version": "3.0.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "target_platform": "linux_x86_64_glibc",
        "python_versions": ["3.8", "3.9", "3.10", "3.11", "3.12", "3.13", "3.14"],
        "development_rows": len(development),
        "validation_input_rows": len(validation),
        "final_test_input_rows": len(final_test),
        "development_outcomes": {
            str(key): int(value)
            for key, value in development["outcome"].value_counts().sort_index().items()
        },
        "development_failure_percentage": round(
            float(development["is_failure"].mean() * 100), 2
        ),
        "validation_outcomes_used": False,
        "final_test_outcomes_used": False,
        "optional_changelog_coverage": {
            "development_a": int(
                development["package_a_changelog_available"].sum()
            ),
            "development_b": int(
                development["package_b_changelog_available"].sum()
            ),
        },
        "source_sha256": {
            name: _sha256(path) for name, path in source_paths.items()
        },
        "files": {
            path.name: {
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for path in files.values()
        },
    }


def _readme(manifest: dict[str, Any]) -> str:
    return f"""# DepLab large feature dataset v3.0.0

This directory contains inference-safe inputs for the large Linux x86_64 dataset.

- Development rows with labels: **{manifest['development_rows']:,}**
- Sealed validation inputs without labels: **{manifest['validation_input_rows']:,}**
- Final-test inputs without labels: **{manifest['final_test_input_rows']:,}**
- Python targets: **3.8 through 3.14**

Only pins, target Python, PyPI release metadata, wheel metadata and frozen release-note
signals are eligible inputs. Installation results, imports, smoke tests, resolved
environments, errors and measurements remain labels or audit context.
"""


def _write_frame(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(
        path,
        index=False,
        encoding="utf-8",
        lineterminator="\n",
        quoting=csv.QUOTE_MINIMAL,
    )


def _failure_stage(outcome: str) -> str:
    if outcome == "pass":
        return "pass"
    if outcome == "resolution_failure":
        return "resolution"
    return "post_install"


def _pin(value: str) -> tuple[str, str]:
    if "==" not in value:
        raise ValueError(f"package pin must use NAME==VERSION: {value!r}")
    return tuple(value.split("==", 1))  # type: ignore[return-value]


def _canonical(value: str) -> str:
    return value.lower().replace("_", "-").replace(".", "-")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_checksums(directory: Path) -> None:
    checksum_path = directory / "SHA256SUMS.txt"
    lines = [
        f"{_sha256(path)}  {path.name}"
        for path in sorted(directory.iterdir())
        if path.is_file() and path != checksum_path
    ]
    checksum_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
