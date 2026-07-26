from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
from pathlib import Path
from typing import Any, Iterable


SCHEMA_VERSION = "3.0.0"
DEFAULT_MODEL = "text-embedding-3-large"
DEFAULT_DIMENSIONS = 3072


def main() -> int:
    args = _arguments()
    _validate_arguments(args)
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Enter it securely in the EC2 shell "
            "before starting this script."
        )

    from openai import OpenAI, __version__ as openai_version
    import tiktoken

    source_rows = _read_jsonl(args.input)
    existing_rows = _read_jsonl(args.output)
    completed = _validate_existing(existing_rows, args)
    _validate_resume_source(source_rows, existing_rows)
    encoding = _encoding_for_model(tiktoken, args.model)
    pending = prepare_inputs(
        source_rows,
        completed,
        encoding,
        maximum_tokens=args.maximum_tokens,
    )
    client = OpenAI(max_retries=args.maximum_retries, timeout=args.timeout)
    collected = _embed_pending(client, pending, args)

    final_rows = _read_jsonl(args.output)
    _validate_complete(source_rows, final_rows, args)
    runtime = {
        "schema_version": SCHEMA_VERSION,
        "model": args.model,
        "dimensions": args.dimensions,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "openai": openai_version,
        "tiktoken": getattr(tiktoken, "__version__", "unknown"),
        "input": str(args.input),
        "input_sha256": _sha256(args.input),
        "output": str(args.output),
        "output_sha256": _sha256(args.output),
        "rows": len(final_rows),
        "collected_this_run": collected,
        "maximum_tokens_per_release": args.maximum_tokens,
        "truncated_release_count": sum(
            bool(row["source_truncated"]) for row in final_rows
        ),
        "embedded_input_tokens": sum(
            int(row["embedded_token_count"]) for row in final_rows
        ),
        "data_policy": (
            "public package release metadata and public release-note text only"
        ),
    }
    args.runtime.parent.mkdir(parents=True, exist_ok=True)
    args.runtime.write_text(
        json.dumps(runtime, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(runtime, indent=2, sort_keys=True))
    return 0


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create resumable OpenAI embeddings for public release evidence"
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--dimensions", type=int, default=DEFAULT_DIMENSIONS)
    parser.add_argument("--maximum-tokens", type=int, default=8000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--maximum-retries", type=int, default=6)
    parser.add_argument("--timeout", type=float, default=180.0)
    return parser.parse_args()


def _validate_arguments(args: argparse.Namespace) -> None:
    positive = (
        args.dimensions,
        args.maximum_tokens,
        args.batch_size,
        args.maximum_retries,
        args.timeout,
    )
    if any(value <= 0 for value in positive):
        raise ValueError("dimensions, limits, retries and timeout must be positive")
    if not args.input.is_file():
        raise FileNotFoundError(args.input)


def prepare_inputs(
    rows: list[dict[str, Any]],
    completed: set[tuple[str, str]],
    encoding: Any,
    maximum_tokens: int,
) -> list[dict[str, Any]]:
    prepared = []
    for row in rows:
        key = _release_key(row["package"], row["version"])
        if key in completed:
            continue
        token_ids = encoding.encode(str(row["selected_text"]))
        embedded = token_ids[:maximum_tokens]
        if not embedded:
            raise ValueError(f"empty release evidence for {key[0]}=={key[1]}")
        prepared.append(
            {
                "source": row,
                "token_ids": embedded,
                "source_token_count": len(token_ids),
                "source_truncated": len(token_ids) > len(embedded),
            }
        )
    return prepared


def _embed_pending(
    client: Any,
    pending: list[dict[str, Any]],
    args: argparse.Namespace,
) -> int:
    collected = 0
    for batch_number, batch in enumerate(_batches(pending, args.batch_size), 1):
        response = client.embeddings.create(
            model=args.model,
            input=[item["token_ids"] for item in batch],
            dimensions=args.dimensions,
            encoding_format="float",
        )
        ordered = sorted(response.data, key=lambda item: item.index)
        if len(ordered) != len(batch):
            raise RuntimeError("OpenAI returned a different number of embeddings")
        for item, result in zip(batch, ordered, strict=True):
            _append_embedding(args.output, item, result.embedding, args)
            collected += 1
        print(
            f"Embedded {collected:03d}/{len(pending):03d} pending releases "
            f"(batch {batch_number})",
            flush=True,
        )
    return collected


def _append_embedding(
    path: Path,
    prepared: dict[str, Any],
    embedding: list[float],
    args: argparse.Namespace,
) -> None:
    source = prepared["source"]
    vector = normalized_vector(embedding)
    if len(vector) != args.dimensions:
        raise RuntimeError(
            f"expected {args.dimensions} embedding values; got {len(vector)}"
        )
    row = {
        "schema_version": SCHEMA_VERSION,
        "package": source["package"],
        "version": source["version"],
        "selected_text_sha256": source["selected_text_sha256"],
        "model": args.model,
        "dimensions": args.dimensions,
        "pooling": "OpenAI embedding, then local L2 normalization",
        "source_token_count": prepared["source_token_count"],
        "embedded_token_count": len(prepared["token_ids"]),
        "source_truncated": prepared["source_truncated"],
        "embedding": vector,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as file:
        file.write(json.dumps(row, separators=(",", ":"), sort_keys=True) + "\n")


def normalized_vector(values: Iterable[float]) -> list[float]:
    vector = [float(value) for value in values]
    norm = math.sqrt(sum(value * value for value in vector))
    if not math.isfinite(norm) or norm <= 0:
        raise ValueError("embedding must contain finite, non-zero values")
    normalized = [value / norm for value in vector]
    if not all(math.isfinite(value) for value in normalized):
        raise ValueError("embedding contains non-finite values")
    return normalized


def _validate_existing(
    rows: list[dict[str, Any]], args: argparse.Namespace
) -> set[tuple[str, str]]:
    completed: set[tuple[str, str]] = set()
    for row in rows:
        key = _release_key(row["package"], row["version"])
        if key in completed:
            raise ValueError(f"duplicate existing embedding for {key[0]}=={key[1]}")
        if row.get("model") != args.model:
            raise ValueError("existing output uses a different embedding model")
        if int(row.get("dimensions", 0)) != args.dimensions:
            raise ValueError("existing output uses different embedding dimensions")
        if len(row.get("embedding", [])) != args.dimensions:
            raise ValueError("existing output contains an invalid vector")
        completed.add(key)
    return completed


def _validate_resume_source(
    source_rows: list[dict[str, Any]],
    existing_rows: list[dict[str, Any]],
) -> None:
    source = {
        _release_key(row["package"], row["version"]): row for row in source_rows
    }
    for row in existing_rows:
        key = _release_key(row["package"], row["version"])
        if key not in source:
            raise ValueError(
                f"existing embedding is absent from source: {key[0]}=={key[1]}"
            )
        if row.get("selected_text_sha256") != source[key]["selected_text_sha256"]:
            raise ValueError(f"source text changed for {key[0]}=={key[1]}")


def _validate_complete(
    source_rows: list[dict[str, Any]],
    embedded_rows: list[dict[str, Any]],
    args: argparse.Namespace,
) -> None:
    source = {
        _release_key(row["package"], row["version"]): row for row in source_rows
    }
    embedded = {
        _release_key(row["package"], row["version"]): row for row in embedded_rows
    }
    if len(source) != len(source_rows) or len(embedded) != len(embedded_rows):
        raise ValueError("source or embedding output contains duplicate releases")
    if set(source) != set(embedded):
        raise ValueError("embedding output does not cover every source release")
    for key, row in embedded.items():
        if row["selected_text_sha256"] != source[key]["selected_text_sha256"]:
            raise ValueError(f"source text changed for {key[0]}=={key[1]}")
        if row["model"] != args.model or int(row["dimensions"]) != args.dimensions:
            raise ValueError("embedding configuration changed during collection")


def _encoding_for_model(tiktoken: Any, model: str) -> Any:
    try:
        return tiktoken.encoding_for_model(model)
    except KeyError:
        return tiktoken.get_encoding("cl100k_base")


def _batches(values: list[Any], size: int) -> Iterable[list[Any]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def _release_key(package: Any, version: Any) -> tuple[str, str]:
    canonical = str(package).lower().replace("_", "-").replace(".", "-")
    return canonical, str(version)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
