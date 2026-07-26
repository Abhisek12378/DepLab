from __future__ import annotations

import json
import shutil
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path


class ResultRepairError(ValueError):
    pass


@dataclass(frozen=True)
class ResultRepairSummary:
    input_rows: int
    input_unique_ids: int
    duplicate_ids_removed_for_retry: int
    infrastructure_only_ids_removed_for_retry: int
    output_rows: int
    backup: str
    output: str


def repair_result_file(
    path: Path,
    backup_path: Path | None = None,
) -> ResultRepairSummary:
    rows = []
    try:
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            row = json.loads(line)
            experiment_id = row.get("experiment_id")
            outcome = row.get("outcome")
            if not isinstance(experiment_id, str) or not experiment_id:
                raise ResultRepairError(f"row {line_number} has no experiment_id")
            if not isinstance(outcome, str) or not outcome:
                raise ResultRepairError(f"row {line_number} has no outcome")
            rows.append(row)
    except (OSError, json.JSONDecodeError) as exc:
        raise ResultRepairError(f"cannot read result file {path}: {exc}") from exc

    counts = Counter(row["experiment_id"] for row in rows)
    duplicate_ids = {experiment_id for experiment_id, count in counts.items() if count > 1}
    infrastructure_only_ids = {
        row["experiment_id"]
        for row in rows
        if row["outcome"] == "infrastructure_failure"
        and counts[row["experiment_id"]] == 1
    }
    retry_ids = duplicate_ids | infrastructure_only_ids
    repaired = [row for row in rows if row["experiment_id"] not in retry_ids]

    repaired_ids = [row["experiment_id"] for row in repaired]
    if len(repaired_ids) != len(set(repaired_ids)):
        raise ResultRepairError("repair did not produce unique experiment IDs")
    if any(row["outcome"] == "infrastructure_failure" for row in repaired):
        raise ResultRepairError("repair left an infrastructure failure in the output")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup = backup_path or path.with_name(
        f"{path.stem}.backup-{timestamp}{path.suffix}"
    )
    if backup.resolve() == path.resolve():
        raise ResultRepairError("backup path must differ from the result path")
    backup.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".repairing")
    shutil.copy2(path, backup)
    with temporary.open("w", encoding="utf-8", newline="\n") as file:
        for row in repaired:
            file.write(json.dumps(row, sort_keys=True) + "\n")
    temporary.replace(path)

    return ResultRepairSummary(
        input_rows=len(rows),
        input_unique_ids=len(counts),
        duplicate_ids_removed_for_retry=len(duplicate_ids),
        infrastructure_only_ids_removed_for_retry=len(infrastructure_only_ids),
        output_rows=len(repaired),
        backup=str(backup),
        output=str(path),
    )


def summary_json(summary: ResultRepairSummary) -> str:
    return json.dumps(asdict(summary), indent=2, sort_keys=True)
