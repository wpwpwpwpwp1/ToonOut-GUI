"""PyInstaller onedir 결과를 검증 가능한 ToonOut GPU 팩으로 묶는다."""

import argparse
import hashlib
import json
import sys
import zipfile
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from gpu_runtime import (
    GENERIC_GPU_RUNTIME_KIND,
    GPU_RUNTIME_SCHEMA,
    GPU_WORKER_PROTOCOL,
)


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--runtime-version", default="2")
    parser.add_argument("--vendor", choices=("nvidia", "amd"), default="nvidia")
    parser.add_argument("--backend", choices=("cuda", "rocm"), default="cuda")
    parser.add_argument("--gfx-target", action="append", default=[])
    args = parser.parse_args()

    source = Path(args.source).resolve()
    output = Path(args.output).resolve()
    worker = source / "ToonOutGpuWorker.exe"
    if not worker.is_file():
        raise SystemExit(f"GPU worker가 없습니다: {worker}")
    if args.backend == "cuda" and torch.version.cuda != "12.8":
        raise SystemExit(
            f"CUDA 12.8 환경에서 빌드해야 합니다. 현재 값: {torch.version.cuda}"
        )
    hip_version = getattr(torch.version, "hip", None)
    if args.backend == "rocm" and not hip_version:
        raise SystemExit("ROCm PyTorch 환경에서 AMD GPU 팩을 빌드해야 합니다.")
    if (args.vendor, args.backend) not in {("nvidia", "cuda"), ("amd", "rocm")}:
        raise SystemExit("지원하지 않는 GPU vendor/backend 조합입니다.")

    records = []
    files = sorted(path for path in source.rglob("*") if path.is_file())
    for path in files:
        relative = path.relative_to(source).as_posix()
        records.append(
            {
                "path": relative,
                "size": path.stat().st_size,
                "sha256": file_hash(path),
            }
        )

    manifest = {
        "schema_version": GPU_RUNTIME_SCHEMA,
        "kind": GENERIC_GPU_RUNTIME_KIND,
        "runtime_version": args.runtime_version,
        "worker_protocol": GPU_WORKER_PROTOCOL,
        "torch_version": torch.__version__,
        "vendor": args.vendor,
        "backend": args.backend,
        "compute_runtime": hip_version if args.backend == "rocm" else torch.version.cuda,
        "cuda_runtime": torch.version.cuda or "",
        "gfx_targets": sorted(set(args.gfx_target)),
        "worker": "ToonOutGpuWorker.exe",
        "files": records,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(
        output,
        "w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=6,
        allowZip64=True,
    ) as archive:
        archive.writestr(
            "manifest.json",
            json.dumps(manifest, ensure_ascii=False, indent=2),
        )
        for path in files:
            archive.write(path, path.relative_to(source).as_posix())

    pack_hash = file_hash(output)
    checksum_path = Path(f"{output}.sha256")
    checksum_path.write_text(
        f"{pack_hash}  {output.name}\n",
        encoding="ascii",
    )
    print(output)
    print(checksum_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
