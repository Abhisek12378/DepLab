from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .batch import load_manifest


class ShardError(ValueError):
    pass


@dataclass(frozen=True)
class ShardSummary:
    manifest: str
    output_dir: str
    experiments: int
    shard_size: int
    shards: int
    filename_width: int
    first_shard_size: int
    last_shard_size: int


def shard_manifest(manifest_path: Path, output_dir: Path, shard_size: int = 50) -> ShardSummary:
    if shard_size < 1:
        raise ShardError("shard size must be at least 1")
    specs = load_manifest(manifest_path)
    try:
        source = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ShardError(f"cannot read manifest {manifest_path}: {exc}") from exc
    experiments = source.get("experiments") if isinstance(source, dict) else None
    if not isinstance(experiments, list) or len(experiments) != len(specs):
        raise ShardError("manifest experiments changed during validation")

    shard_count = math.ceil(len(experiments) / shard_size)
    filename_width = max(2, len(str(shard_count)))
    output_dir.mkdir(parents=True, exist_ok=True)
    for stale_path in output_dir.glob("shard-*.json"):
        stale_path.unlink()
    sizes = []
    for index in range(shard_count):
        rows = experiments[index * shard_size : (index + 1) * shard_size]
        sizes.append(len(rows))
        payload: dict[str, Any] = {
            "schema_version": "1.0.0",
            "description": "A deterministic execution shard of the systematic matrix.",
            "source_manifest": manifest_path.as_posix(),
            "shard_index": index + 1,
            "shard_count": shard_count,
            "experiments": rows,
        }
        path = output_dir / (
            f"shard-{index + 1:0{filename_width}d}-"
            f"of-{shard_count:0{filename_width}d}.json"
        )
        path.write_bytes((json.dumps(payload, indent=2) + "\n").encode("utf-8"))
        load_manifest(path)

    return ShardSummary(
        manifest=manifest_path.as_posix(),
        output_dir=output_dir.as_posix(),
        experiments=len(experiments),
        shard_size=shard_size,
        shards=shard_count,
        filename_width=filename_width,
        first_shard_size=sizes[0],
        last_shard_size=sizes[-1],
    )
