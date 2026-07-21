from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


SIGNAL_NAMES = [
    "breaking",
    "removal",
    "deprecation",
    "api",
    "abi",
    "dependency",
    "python_support",
    "wheel_build",
    "removed_deprecated",
    "api_removal",
    "abi_break",
    "dependency_compatibility",
    "support_drop",
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Join release-level changelog signals to features")
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--changelogs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()

    frame = pd.read_csv(
        args.features,
        dtype={
            "experiment_id": "string",
            "python_version": "string",
            "package_a_version": "string",
            "package_b_version": "string",
        },
    )
    records = _read_jsonl(args.changelogs)
    augmented, added_columns = augment(frame, records)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    augmented.to_csv(args.output, index=False, encoding="utf-8", lineterminator="\n")
    summary = {
        "rows": len(augmented),
        "original_columns": len(frame.columns),
        "added_columns": added_columns,
        "added_column_count": len(added_columns),
        "total_columns": len(augmented.columns),
        "unique_changelog_releases": len(records),
        "missing_release_records": 0,
        "input": str(args.features),
        "changelogs": str(args.changelogs),
        "output": str(args.output),
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def augment(
    frame: pd.DataFrame, records: list[dict[str, Any]]
) -> tuple[pd.DataFrame, list[str]]:
    result = frame.copy()
    catalog = {
        (_canonical(str(row["package"])), str(row["version"])): row
        for row in records
    }
    missing = set()
    for side in ("a", "b"):
        for name, version in zip(
            result[f"package_{side}_name"].astype(str),
            result[f"package_{side}_version"].astype(str),
        ):
            key = (_canonical(name), version)
            if key not in catalog:
                missing.add(key)
    if missing:
        examples = ", ".join(f"{name}=={version}" for name, version in sorted(missing)[:10])
        raise ValueError(f"missing {len(missing)} changelog release records: {examples}")

    added = []
    side_signals: dict[str, dict[str, list[Any]]] = {}
    for side in ("a", "b"):
        values: dict[str, list[Any]] = {
            "selected_characters": [],
            "version_section_found": [],
            **{f"{name}_count": [] for name in SIGNAL_NAMES},
            **{f"{name}_flag": [] for name in SIGNAL_NAMES},
        }
        for name, version in zip(
            result[f"package_{side}_name"].astype(str),
            result[f"package_{side}_version"].astype(str),
        ):
            record = catalog[(_canonical(name), version)]
            signals = record["signals"]
            values["selected_characters"].append(signals["selected_characters"])
            values["version_section_found"].append(bool(record["version_section_found"]))
            for signal in SIGNAL_NAMES:
                values[f"{signal}_count"].append(int(signals[f"{signal}_count"]))
                values[f"{signal}_flag"].append(bool(signals[f"{signal}_flag"]))
        side_signals[side] = values
        for key, column_values in values.items():
            column = f"package_{side}_changelog_{key}"
            result[column] = column_values
            added.append(column)

    a_mentions_b = []
    b_mentions_a = []
    for _, row in result.iterrows():
        record_a = catalog[(_canonical(str(row["package_a_name"])), str(row["package_a_version"]))]
        record_b = catalog[(_canonical(str(row["package_b_name"])), str(row["package_b_version"]))]
        a_mentions_b.append(
            int(record_a["signals"]["package_mentions"].get(_canonical(str(row["package_b_name"])), 0))
        )
        b_mentions_a.append(
            int(record_b["signals"]["package_mentions"].get(_canonical(str(row["package_a_name"])), 0))
        )
    result["package_a_changelog_mentions_b_count"] = a_mentions_b
    result["package_b_changelog_mentions_a_count"] = b_mentions_a
    added.extend(
        ["package_a_changelog_mentions_b_count", "package_b_changelog_mentions_a_count"]
    )

    for signal in ("breaking", "removal", "deprecation", "abi"):
        column = f"changelog_either_{signal}_flag"
        result[column] = (
            result[f"package_a_changelog_{signal}_flag"].astype(bool)
            | result[f"package_b_changelog_{signal}_flag"].astype(bool)
        )
        added.append(column)
    native_a = _boolean_series(result.get("package_a_has_native_extensions", False), len(result))
    native_b = _boolean_series(result.get("package_b_has_native_extensions", False), len(result))
    result["changelog_native_abi_risk"] = (
        result["package_a_changelog_abi_flag"].astype(bool) & native_b
    ) | (result["package_b_changelog_abi_flag"].astype(bool) & native_a)
    added.append("changelog_native_abi_risk")
    result["changelog_native_abi_break_risk"] = (
        result["package_a_changelog_abi_break_flag"].astype(bool) & native_b
    ) | (result["package_b_changelog_abi_break_flag"].astype(bool) & native_a)
    added.append("changelog_native_abi_break_risk")

    if {"package_a_release_date", "package_b_release_date"}.issubset(result.columns):
        date_a = pd.to_datetime(result["package_a_release_date"], utc=True, errors="coerce")
        date_b = pd.to_datetime(result["package_b_release_date"], utc=True, errors="coerce")
    elif {"package_a_release_ordinal", "package_b_release_ordinal"}.issubset(result.columns):
        date_a = pd.to_numeric(result["package_a_release_ordinal"], errors="coerce")
        date_b = pd.to_numeric(result["package_b_release_ordinal"], errors="coerce")
    else:
        raise ValueError("features need release dates or release ordinals for temporal changelog risk")
    breaking_a = (
        result["package_a_changelog_breaking_flag"].astype(bool)
        | result["package_a_changelog_removal_flag"].astype(bool)
    )
    breaking_b = (
        result["package_b_changelog_breaking_flag"].astype(bool)
        | result["package_b_changelog_removal_flag"].astype(bool)
    )
    result["changelog_newer_provider_break_risk"] = (
        (date_a > date_b) & breaking_a
    ) | ((date_b > date_a) & breaking_b)
    added.append("changelog_newer_provider_break_risk")
    result["changelog_newer_api_removal_risk"] = (
        (date_a > date_b) & result["package_a_changelog_api_removal_flag"].astype(bool)
    ) | (
        (date_b > date_a) & result["package_b_changelog_api_removal_flag"].astype(bool)
    )
    added.append("changelog_newer_api_removal_risk")
    return result, added


def _boolean_series(value: Any, length: int) -> pd.Series:
    if isinstance(value, pd.Series):
        if value.dtype == bool:
            return value
        return value.astype(str).str.lower().isin({"true", "1"})
    return pd.Series([bool(value)] * length)


def _canonical(value: str) -> str:
    return value.lower().replace("_", "-").replace(".", "-")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


if __name__ == "__main__":
    raise SystemExit(main())
