"""Verify that the pinned GitHub Release contains the complete GPU pack."""

import argparse
import hashlib
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from gpu_download import (
    GPU_PACK_ARCHIVE_SIZE,
    GPU_PACK_MANIFEST_NAME,
    GPU_PACK_MANIFEST_SHA256,
    GPU_PACK_MANIFEST_SIZE,
)
from gpu_runtime import load_gpu_pack_parts


def expected_gpu_assets(manifest_path: str | Path) -> dict[str, tuple[int, str]]:
    path = Path(manifest_path)
    document = path.read_bytes()
    manifest_hash = hashlib.sha256(document).hexdigest()
    if len(document) != GPU_PACK_MANIFEST_SIZE:
        raise RuntimeError("GPU 팩 manifest 크기가 앱의 고정값과 다릅니다.")
    if manifest_hash != GPU_PACK_MANIFEST_SHA256:
        raise RuntimeError("GPU 팩 manifest SHA-256이 앱의 고정값과 다릅니다.")

    pack = load_gpu_pack_parts(path)
    if pack.archive_size != GPU_PACK_ARCHIVE_SIZE:
        raise RuntimeError("GPU 팩 전체 크기가 앱의 고정값과 다릅니다.")

    expected = {
        GPU_PACK_MANIFEST_NAME: (len(document), manifest_hash),
    }
    for part in pack.parts:
        if part.path.name in expected:
            raise RuntimeError(f"중복된 GPU 팩 자산 이름입니다: {part.path.name}")
        expected[part.path.name] = (part.size, part.sha256)
    return expected


def verify_release_assets(
    manifest_path: str | Path,
    assets_document: dict,
) -> tuple[str, ...]:
    expected = expected_gpu_assets(manifest_path)
    assets = assets_document.get("assets")
    if not isinstance(assets, list):
        raise RuntimeError("GitHub Release 자산 목록 형식이 올바르지 않습니다.")

    by_name = {}
    for asset in assets:
        if not isinstance(asset, dict) or not isinstance(asset.get("name"), str):
            raise RuntimeError("GitHub Release 자산 정보가 올바르지 않습니다.")
        name = asset["name"]
        if name in by_name:
            raise RuntimeError(f"중복된 GitHub Release 자산입니다: {name}")
        by_name[name] = asset

    for name, (expected_size, expected_hash) in expected.items():
        asset = by_name.get(name)
        if asset is None:
            raise RuntimeError(f"GitHub Release에 GPU 팩 자산이 없습니다: {name}")
        if asset.get("state") != "uploaded":
            raise RuntimeError(f"GPU 팩 자산 업로드가 완료되지 않았습니다: {name}")
        if int(asset.get("size", -1)) != expected_size:
            raise RuntimeError(f"GPU 팩 자산 크기가 다릅니다: {name}")
        if str(asset.get("digest", "")).lower() != f"sha256:{expected_hash}":
            raise RuntimeError(f"GPU 팩 자산 SHA-256이 다릅니다: {name}")

    return tuple(expected)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--assets-json", type=Path, required=True)
    args = parser.parse_args()

    assets_document = json.loads(args.assets_json.read_text(encoding="utf-8-sig"))
    verified = verify_release_assets(args.manifest, assets_document)
    for name in verified:
        print(f"verified GPU release asset: {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
