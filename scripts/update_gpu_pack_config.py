"""빌드한 분할 GPU 팩의 고정 검증값을 앱 소스에 반영한다."""

import argparse
import hashlib
import json
import re
from pathlib import Path


def update_config(manifest_path: Path, config_path: Path) -> None:
    document = manifest_path.read_bytes()
    payload = json.loads(document.decode("utf-8"))
    archive_size = int(payload["archive_size"])
    manifest_hash = hashlib.sha256(document).hexdigest()
    source = config_path.read_text(encoding="utf-8")
    replacements = {
        r"GPU_PACK_MANIFEST_SIZE = \d+": (
            f"GPU_PACK_MANIFEST_SIZE = {len(document)}"
        ),
        r'GPU_PACK_MANIFEST_SHA256 = \(\n    "[0-9a-f]{64}"\n\)': (
            'GPU_PACK_MANIFEST_SHA256 = (\n'
            f'    "{manifest_hash}"\n'
            ")"
        ),
        r"GPU_PACK_ARCHIVE_SIZE = [\d_]+": (
            f"GPU_PACK_ARCHIVE_SIZE = {archive_size:_}"
        ),
    }
    for pattern, replacement in replacements.items():
        source, count = re.subn(pattern, replacement, source)
        if count != 1:
            raise RuntimeError(
                f"gpu_download.py 설정 항목을 정확히 하나 찾지 못했습니다: {pattern}"
            )
    config_path.write_text(source, encoding="utf-8")


def main() -> int:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=project_root / "dist" / "ToonOut-NVIDIA-GPU-Pack.parts.json",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=project_root / "gpu_download.py",
    )
    args = parser.parse_args()
    update_config(args.manifest.resolve(), args.config.resolve())
    print(args.config.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
