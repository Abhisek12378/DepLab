from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import zipfile
from pathlib import Path


PACKAGE_ID = "deplab-external-test-v1.0.0"


def main() -> int:
    parser = argparse.ArgumentParser(description="Package the completed blind external test")
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--zip", type=Path, required=True)
    args = parser.parse_args()
    root = args.project_root.resolve()
    output = args.output_dir.resolve()
    required = {
        "external-test-scope.json": root / "configs/external-test-scope.json",
        "external-test-matrix.json": root / "configs/external-test-matrix.json",
        "external-test-catalog.jsonl": root / "outputs/external-test-catalog.jsonl",
        "external-test-results.jsonl": root / "outputs/external-test-results.jsonl",
    }
    for destination, source in required.items():
        if not source.exists():
            raise FileNotFoundError(source)
        shutil.copyfile(source, output / destination)
    metrics = json.loads((output / "evaluation/metrics.json").read_text(encoding="utf-8"))
    provenance = json.loads((output / "prediction-provenance.json").read_text(encoding="utf-8"))
    if not provenance.get("predictions_created_before_outcomes") or metrics.get("rows") != 61:
        raise ValueError("external test has not passed its completion checks")
    (output / "README.md").write_text(_readme(metrics), encoding="utf-8")
    _write_checksums(output)

    args.zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in sorted(output.rglob("*")):
            if path.is_file():
                archive.write(path, arcname=f"{PACKAGE_ID}/{path.relative_to(output).as_posix()}")
    archive_hash = _sha256(args.zip)
    args.zip.with_suffix(args.zip.suffix + ".sha256").write_text(
        f"{archive_hash}  {args.zip.name}\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "package_id": PACKAGE_ID,
                "rows": metrics["rows"],
                "accuracy": metrics["overall"]["accuracy"],
                "zip": str(args.zip),
                "zip_sha256": archive_hash,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _readme(metrics: dict) -> str:
    overall = metrics["overall"]
    return f"""# DepLab blind external-version test v1.0.0

This package contains a genuine blind evaluation of the logistic baseline on 61 experiments. All 20 package versions were absent from the 646-row training dataset, and predictions were frozen before EC2 outcomes were collected.

## Main result

- Accuracy: {overall['accuracy']:.3f}
- Balanced accuracy: {overall['balanced_accuracy']:.3f}
- ROC AUC: {overall['roc_auc']:.3f}
- Correct predictions: {overall['true_positive'] + overall['true_negative']} of 61
- False compatible predictions: {overall['false_positive']}
- False incompatible predictions: {overall['false_negative']}

See `evaluation/report.md` for the readable report and `evaluation/scored-predictions.csv` for every prediction and outcome.
"""


def _write_checksums(output: Path) -> None:
    checksum = output / "SHA256SUMS.txt"
    lines = []
    for path in sorted(output.rglob("*")):
        if path.is_file() and path != checksum:
            lines.append(f"{_sha256(path)}  {path.relative_to(output).as_posix()}")
    checksum.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
