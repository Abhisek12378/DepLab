from __future__ import annotations

import subprocess
from pathlib import Path, PurePosixPath


MAXIMUM_FILE_BYTES = 20 * 1024 * 1024
FORBIDDEN_DIRECTORIES = {
    ".agents",
    ".codex",
    ".venv",
    "checkpoints",
    "models",
    "node_modules",
    "outputs",
    "work",
}
FORBIDDEN_FILES = {".env"}
FORBIDDEN_SUFFIXES = {
    ".joblib",
    ".key",
    ".npy",
    ".npz",
    ".onnx",
    ".p12",
    ".parquet",
    ".pem",
    ".pfx",
    ".pickle",
    ".pkl",
    ".pt",
    ".pth",
    ".safetensors",
}


def tracked_files() -> list[PurePosixPath]:
    result = subprocess.run(
        ["git", "ls-files", "-z"],
        check=True,
        capture_output=True,
    )
    return [
        PurePosixPath(value.decode("utf-8"))
        for value in result.stdout.split(b"\0")
        if value
    ]


def rejection_reason(path: PurePosixPath) -> str | None:
    if path.name in FORBIDDEN_FILES:
        return "local secret file"
    if path.suffix.lower() in FORBIDDEN_SUFFIXES:
        return "model, credential, or generated dataset extension"
    blocked_parts = FORBIDDEN_DIRECTORIES.intersection(path.parts)
    if blocked_parts and path.name != ".gitkeep":
        return f"private/generated directory: {sorted(blocked_parts)[0]}"
    local_path = Path(path)
    if local_path.stat().st_size > MAXIMUM_FILE_BYTES:
        return "file is larger than the public repository limit of 20 MiB"
    return None


def main() -> int:
    rejected = [
        (str(path), reason)
        for path in tracked_files()
        if (reason := rejection_reason(path)) is not None
    ]
    if rejected:
        print("Public repository safety check failed:")
        for path, reason in rejected:
            print(f"- {path}: {reason}")
        return 1
    print("Public repository safety check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
