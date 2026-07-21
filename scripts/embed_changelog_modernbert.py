from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
from pathlib import Path
from typing import Any


MODEL_NAME = "answerdotai/ModernBERT-base"
SCHEMA_VERSION = "1.0.0"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create frozen ModernBERT embeddings for release changelog text"
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--maximum-tokens", type=int, default=512)
    parser.add_argument("--maximum-chunks", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()

    import torch
    import transformers
    from transformers import AutoModel, AutoTokenizer

    if args.maximum_tokens < 32 or args.maximum_chunks < 1 or args.batch_size < 1:
        raise ValueError("token, chunk and batch limits must be positive")
    rows = _read_jsonl(args.input)
    existing = {
        (_canonical(row["package"]), str(row["version"]))
        for row in _read_jsonl(args.output)
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_grad_enabled(False)
    if device.type == "cpu":
        torch.set_num_threads(max(1, min(8, (os.cpu_count() or 1))))
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model, use_safetensors=True).to(device)
    model.eval()
    collected = 0
    for position, row in enumerate(rows, 1):
        key = (_canonical(row["package"]), str(row["version"]))
        if key in existing:
            continue
        text = (
            f"Package: {row['package']}\n"
            f"Release version: {row['version']}\n"
            f"Release notes:\n{row['selected_text']}"
        )
        embedding, chunk_count, token_count = embed_text(
            text,
            tokenizer,
            model,
            device,
            args.maximum_tokens,
            args.maximum_chunks,
            args.batch_size,
        )
        _append_jsonl(
            args.output,
            {
                "schema_version": SCHEMA_VERSION,
                "package": row["package"],
                "version": row["version"],
                "selected_text_sha256": row["selected_text_sha256"],
                "model": args.model,
                "pooling": "attention-mask mean pooling per chunk, then mean across chunks, then L2 normalization",
                "maximum_tokens": args.maximum_tokens,
                "maximum_chunks": args.maximum_chunks,
                "source_token_count": token_count,
                "embedded_chunk_count": chunk_count,
                "embedding_dimension": len(embedding),
                "embedding": embedding,
            },
        )
        existing.add(key)
        collected += 1
        print(f"Embedded {position:03d}/{len(rows):03d}: {row['package']}=={row['version']}", flush=True)

    runtime = {
        "schema_version": "1.0.0",
        "model": args.model,
        "device": str(device),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "input": str(args.input),
        "input_sha256": _sha256(args.input),
        "output": str(args.output),
        "output_sha256": _sha256(args.output),
        "rows": len(_read_jsonl(args.output)),
        "collected_this_run": collected,
        "maximum_tokens": args.maximum_tokens,
        "maximum_chunks": args.maximum_chunks,
        "batch_size": args.batch_size,
    }
    args.runtime.parent.mkdir(parents=True, exist_ok=True)
    args.runtime.write_text(json.dumps(runtime, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(runtime, indent=2, sort_keys=True))
    return 0


def embed_text(
    text: str,
    tokenizer: Any,
    model: Any,
    device: Any,
    maximum_tokens: int,
    maximum_chunks: int,
    batch_size: int,
) -> tuple[list[float], int, int]:
    import torch

    token_ids = tokenizer.encode(text, add_special_tokens=False)
    payload_size = maximum_tokens - tokenizer.num_special_tokens_to_add(pair=False)
    if payload_size < 1:
        raise ValueError("maximum token limit leaves no room for text")
    if len(token_ids) <= payload_size:
        starts = [0]
    else:
        possible = list(range(0, len(token_ids), payload_size))
        if len(possible) <= maximum_chunks:
            starts = possible
        else:
            last_start = max(0, len(token_ids) - payload_size)
            starts = sorted(
                {
                    int(round(index * last_start / (maximum_chunks - 1)))
                    for index in range(maximum_chunks)
                }
            )
    prepared = [
        tokenizer.prepare_for_model(
            token_ids[start : start + payload_size],
            add_special_tokens=True,
            truncation=True,
            max_length=maximum_tokens,
            return_attention_mask=True,
        )
        for start in starts
    ]
    vectors = []
    with torch.inference_mode():
        for offset in range(0, len(prepared), batch_size):
            batch = tokenizer.pad(
                prepared[offset : offset + batch_size],
                padding=True,
                return_tensors="pt",
            )
            batch = {key: value.to(device) for key, value in batch.items()}
            hidden = model(**batch).last_hidden_state
            mask = batch["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            vectors.append(pooled.cpu())
    vector = torch.cat(vectors, dim=0).mean(dim=0)
    vector = vector / vector.norm(p=2).clamp(min=1e-12)
    return vector.to(torch.float32).tolist(), len(starts), len(token_ids)


def _canonical(value: str) -> str:
    return value.lower().replace("_", "-").replace(".", "-")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as file:
        file.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
