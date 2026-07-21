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

from deplab.models import ExperimentSpec, PackagePin
from package_dataset import _canonical_name, _direct_requirement, _parse_date, _version_parts
from train_baseline import LogisticModel, Preprocessor, _add_derived_inputs


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create blind predictions for held-out package versions"
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--training-features", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    experiments = manifest.get("experiments")
    if not isinstance(experiments, list) or not experiments:
        raise ValueError("external-test manifest has no experiments")
    catalog_rows = _read_jsonl(args.catalog)
    catalog = {
        (
            _canonical_name(str(row["release"]["name"])),
            str(row["release"]["version"]),
            str(row["target"]["python_version"]),
        ): row["release"]
        for row in catalog_rows
    }
    training = pd.read_csv(
        args.training_features,
        dtype={"package_a_version": "string", "package_b_version": "string"},
    )
    training_versions = _training_versions(training)

    feature_rows = []
    for order, row in enumerate(experiments, 1):
        package_a = _pin(row["package_a"])
        package_b = _pin(row["package_b"])
        python_version = str(row["python"])
        _assert_unseen(training_versions, package_a)
        _assert_unseen(training_versions, package_b)
        spec = ExperimentSpec(package_a, package_b, python_version)
        release_a = catalog[
            (_canonical_name(package_a.name), package_a.version, python_version)
        ]
        release_b = catalog[
            (_canonical_name(package_b.name), package_b.version, python_version)
        ]
        feature_rows.append(
            _prediction_input(order, str(row["family"]), spec, release_a, release_b)
        )

    frame = _add_derived_inputs(pd.DataFrame(feature_rows))
    if len(frame) != len({str(value) for value in frame["experiment_id"]}):
        raise ValueError("external-test manifest contains duplicate experiment IDs")

    model_payload = json.loads(args.model.read_text(encoding="utf-8"))
    preprocessor = _load_preprocessor(model_payload["preprocessor"])
    model = LogisticModel(
        weights=np.asarray(model_payload["weights"], dtype=float),
        intercept=float(model_payload["intercept"]),
        iterations=0,
        final_loss=float("nan"),
    )
    matrix = preprocessor.transform(frame)
    probabilities = model.predict_proba(matrix)
    threshold = float(model_payload["threshold"])
    predictions = probabilities >= threshold

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    input_columns = [
        "matrix_order",
        "experiment_id",
        "family",
        "package_a_name",
        "package_a_version",
        "package_b_name",
        "package_b_version",
        *preprocessor.numeric_medians.keys(),
        *preprocessor.category_levels.keys(),
    ]
    input_columns = list(dict.fromkeys(input_columns))
    frame.loc[:, input_columns].to_csv(
        output / "prediction-inputs.csv", index=False, quoting=csv.QUOTE_MINIMAL
    )
    prediction_frame = frame.loc[
        :,
        [
            "matrix_order",
            "experiment_id",
            "family",
            "package_a_name",
            "package_a_version",
            "package_b_name",
            "package_b_version",
            "python_version",
        ],
    ].copy()
    prediction_frame["predicted_probability_compatible"] = probabilities
    prediction_frame["predicted_is_compatible"] = predictions
    prediction_frame.to_csv(
        output / "blind-predictions.csv", index=False, quoting=csv.QUOTE_MINIMAL
    )

    provenance = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "evaluation_design": "same package names with versions absent from training",
        "predictions_created_before_outcomes": True,
        "rows": len(frame),
        "families": sorted(frame["family"].unique().tolist()),
        "decision_threshold": threshold,
        "compatible_predictions": int(predictions.sum()),
        "incompatible_predictions": int((~predictions).sum()),
        "unknown_category_values": _unknown_categories(frame, preprocessor),
        "source_sha256": {
            "manifest": _sha256(args.manifest),
            "catalog": _sha256(args.catalog),
            "training_features": _sha256(args.training_features),
            "model": _sha256(args.model),
        },
    }
    (output / "prediction-provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_checksums(output)
    print(json.dumps(provenance, indent=2, sort_keys=True))
    return 0


def _prediction_input(
    order: int,
    family: str,
    spec: ExperimentSpec,
    release_a: dict[str, Any],
    release_b: dict[str, Any],
) -> dict[str, Any]:
    wheel_a = _selected_wheel(release_a)
    wheel_b = _selected_wheel(release_b)
    version_a = _version_parts(spec.package_a.version)
    version_b = _version_parts(spec.package_b.version)
    python_parts = _version_parts(spec.python_version)
    requirement_a = _direct_requirement(
        release_a.get("requires_dist", []), spec.package_b.name
    )
    requirement_b = _direct_requirement(
        release_b.get("requires_dist", []), spec.package_a.name
    )
    date_a = _parse_date(release_a.get("release_date"))
    date_b = _parse_date(release_b.get("release_date"))
    return {
        "matrix_order": order,
        "experiment_id": spec.experiment_id,
        "family": family,
        "package_a_name": spec.package_a.name,
        "package_a_version": spec.package_a.version,
        "package_a_version_major": version_a[0],
        "package_a_version_minor": version_a[1],
        "package_a_version_patch": version_a[2],
        "package_b_name": spec.package_b.name,
        "package_b_version": spec.package_b.version,
        "package_b_version_major": version_b[0],
        "package_b_version_minor": version_b[1],
        "package_b_version_patch": version_b[2],
        "python_version": spec.python_version,
        "python_major": python_parts[0],
        "python_minor": python_parts[1],
        "package_a_requires_python": release_a.get("requires_python"),
        "package_b_requires_python": release_b.get("requires_python"),
        "package_a_release_date": release_a.get("release_date"),
        "package_b_release_date": release_b.get("release_date"),
        "release_date_distance_days": abs((date_a - date_b).days)
        if date_a and date_b
        else None,
        "package_a_requires_dist_count": len(release_a.get("requires_dist", [])),
        "package_b_requires_dist_count": len(release_b.get("requires_dist", [])),
        "package_a_declares_package_b": bool(requirement_a),
        "package_b_declares_package_a": bool(requirement_b),
        "package_a_requirement_on_b": requirement_a,
        "package_b_requirement_on_a": requirement_b,
        "package_a_eligible_wheel_count": sum(
            bool(wheel.get("compatible")) and not bool(wheel.get("yanked"))
            for wheel in release_a.get("wheels", [])
        ),
        "package_b_eligible_wheel_count": sum(
            bool(wheel.get("compatible")) and not bool(wheel.get("yanked"))
            for wheel in release_b.get("wheels", [])
        ),
        "package_a_wheel_python_tag": wheel_a["python_tag"],
        "package_b_wheel_python_tag": wheel_b["python_tag"],
        "package_a_wheel_abi_tag": wheel_a["abi_tag"],
        "package_b_wheel_abi_tag": wheel_b["abi_tag"],
        "package_a_wheel_platform_tag": wheel_a["platform_tag"],
        "package_b_wheel_platform_tag": wheel_b["platform_tag"],
        "package_a_has_native_extensions": bool(wheel_a["has_native_extensions"]),
        "package_b_has_native_extensions": bool(wheel_b["has_native_extensions"]),
        "either_top_level_has_native_extensions": bool(
            wheel_a["has_native_extensions"] or wheel_b["has_native_extensions"]
        ),
        "top_level_wheel_bytes": int(wheel_a.get("size") or 0)
        + int(wheel_b.get("size") or 0),
    }


def _load_preprocessor(payload: dict[str, Any]) -> Preprocessor:
    return Preprocessor(
        numeric_medians={key: float(value) for key, value in payload["numeric_medians"].items()},
        numeric_means={key: float(value) for key, value in payload["numeric_means"].items()},
        numeric_scales={key: float(value) for key, value in payload["numeric_scales"].items()},
        category_levels={key: list(value) for key, value in payload["category_levels"].items()},
        feature_names=list(payload["feature_names"]),
    )


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
        raise ValueError(
            f"no compatible wheel for {release.get('name')}=={release.get('version')}"
        )
    return candidates[0]


def _pin(value: str) -> PackagePin:
    name, version = str(value).split("==", 1)
    return PackagePin(name=name, version=version)


def _training_versions(frame: pd.DataFrame) -> dict[str, set[str]]:
    result: dict[str, set[str]] = {}
    for side in ("a", "b"):
        for name, version in zip(
            frame[f"package_{side}_name"].astype(str),
            frame[f"package_{side}_version"].astype(str),
        ):
            result.setdefault(_canonical_name(name), set()).add(version)
    return result


def _assert_unseen(training_versions: dict[str, set[str]], pin: PackagePin) -> None:
    if pin.version in training_versions.get(_canonical_name(pin.name), set()):
        raise ValueError(f"external version was used in training: {pin.requirement}")


def _unknown_categories(frame: pd.DataFrame, preprocessor: Preprocessor) -> dict[str, list[str]]:
    result = {}
    for column, levels in preprocessor.category_levels.items():
        values = set(frame[column].fillna("<missing>").astype(str))
        unknown = sorted(values - set(levels))
        if unknown:
            result[column] = unknown
    return result


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_checksums(output: Path) -> None:
    lines = [
        f"{_sha256(path)}  {path.name}"
        for path in sorted(output.iterdir())
        if path.is_file() and path.name != "SHA256SUMS.txt"
    ]
    (output / "SHA256SUMS.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
