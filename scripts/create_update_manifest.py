"""GitHub Release 설치 파일을 가리키는 서명된 update.json을 만든다."""

import argparse
import base64
import hashlib
import json
import os
import sys
from pathlib import Path

from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from gpu_runtime import GPU_WORKER_PROTOCOL
from update_service import canonical_payload_bytes, parse_version


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def create_manifest(
    installer: Path,
    version: str,
    minimum_supported_version: str,
    repository: str,
    tag: str,
    private_key_b64: str,
) -> dict[str, object]:
    parse_version(version)
    parse_version(minimum_supported_version)
    if repository.count("/") != 1:
        raise ValueError("repository는 owner/name 형식이어야 합니다")
    try:
        private_key_bytes = base64.b64decode(
            private_key_b64,
            validate=True,
        )
    except (TypeError, ValueError) as error:
        raise ValueError("개인키 Base64 형식이 올바르지 않습니다") from error
    if len(private_key_bytes) != 32:
        raise ValueError("Ed25519 개인키 길이가 올바르지 않습니다")

    asset_url = (
        f"https://github.com/{repository}/releases/download/"
        f"{tag}/{installer.name}"
    )
    payload: dict[str, object] = {
        "schema_version": 1,
        "version": version,
        "minimum_supported_version": minimum_supported_version,
        "installer_url": asset_url,
        "installer_name": installer.name,
        "installer_size": installer.stat().st_size,
        "installer_sha256": file_sha256(installer),
        "release_notes_url": f"https://github.com/{repository}/releases/tag/{tag}",
        "gpu_worker_protocol": GPU_WORKER_PROTOCOL,
    }
    private_key = Ed25519PrivateKey.from_private_bytes(private_key_bytes)
    signature = private_key.sign(canonical_payload_bytes(payload))
    return {
        "payload": payload,
        "signature": base64.b64encode(signature).decode("ascii"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--installer", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--version", required=True)
    parser.add_argument("--minimum-supported-version", default="0.1.0")
    parser.add_argument("--repository", required=True)
    parser.add_argument("--tag", required=True)
    args = parser.parse_args()

    private_key = os.environ.get("UPDATE_SIGNING_PRIVATE_KEY_B64", "")
    if not private_key:
        raise SystemExit(
            "GitHub Secret UPDATE_SIGNING_PRIVATE_KEY_B64가 필요합니다"
        )
    installer = args.installer.resolve()
    if not installer.is_file():
        raise SystemExit(f"설치 파일이 없습니다: {installer}")

    manifest = create_manifest(
        installer,
        args.version,
        args.minimum_supported_version,
        args.repository,
        args.tag,
        private_key,
    )
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
