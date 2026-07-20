"""Split a large GPU pack into GitHub Release-compatible verified parts."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from gpu_runtime import GPU_PACK_PARTS_KIND, GPU_PACK_PARTS_SCHEMA


DEFAULT_PART_SIZE = 1_800_000_000


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def split_pack(
    input_path: Path,
    part_size: int = DEFAULT_PART_SIZE,
    *,
    remove_input: bool = False,
) -> Path:
    input_path = input_path.resolve()
    if part_size <= 0 or part_size >= 2 * 1024**3:
        raise ValueError("part size must be greater than zero and below 2 GiB")
    base_name = input_path.name.removesuffix(".zip")
    records = []
    with input_path.open("rb") as source:
        index = 1
        while True:
            first_chunk = source.read(min(8 * 1024 * 1024, part_size))
            if not first_chunk:
                break
            part_path = input_path.with_name(f"{base_name}.part{index:02d}")
            digest = hashlib.sha256()
            written = 0
            with part_path.open("wb") as output:
                chunk = first_chunk
                while chunk:
                    output.write(chunk)
                    digest.update(chunk)
                    written += len(chunk)
                    remaining = part_size - written
                    if remaining <= 0:
                        break
                    chunk = source.read(min(8 * 1024 * 1024, remaining))
            records.append(
                {
                    "filename": part_path.name,
                    "size": written,
                    "sha256": digest.hexdigest(),
                }
            )
            index += 1
    manifest_path = input_path.with_name(f"{base_name}.parts.json")
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": GPU_PACK_PARTS_SCHEMA,
                "kind": GPU_PACK_PARTS_KIND,
                "archive_filename": input_path.name,
                "archive_size": input_path.stat().st_size,
                "archive_sha256": file_hash(input_path),
                "parts": records,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    if remove_input:
        input_path.unlink()
        Path(f"{input_path}.sha256").unlink(missing_ok=True)
    return manifest_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--part-size", type=int, default=DEFAULT_PART_SIZE)
    parser.add_argument(
        "--keep-input",
        action="store_true",
        help="분할 검증 후에도 원본 ZIP을 보관합니다.",
    )
    args = parser.parse_args()
    print(
        split_pack(
            Path(args.input),
            args.part_size,
            remove_input=not args.keep_input,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
