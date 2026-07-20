"""자동 업데이트 Ed25519 키를 한 번 생성한다.

개인키는 `.secrets`에 두고, 공개키만 앱에 포함되는 `update_config.py`에 쓴다.
"""

import argparse
import base64
from pathlib import Path

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--private-output",
        type=Path,
        default=PROJECT_ROOT / ".secrets" / "update-private-key.txt",
    )
    parser.add_argument(
        "--public-module",
        type=Path,
        default=PROJECT_ROOT / "update_config.py",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    private_output = args.private_output.resolve()
    public_module = args.public_module.resolve()
    if private_output.exists() and not args.force:
        raise SystemExit(
            f"개인키가 이미 있습니다. 덮어쓰지 않았습니다: {private_output}"
        )

    private_key = Ed25519PrivateKey.generate()
    private_bytes = private_key.private_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PrivateFormat.Raw,
        encryption_algorithm=serialization.NoEncryption(),
    )
    public_bytes = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    private_value = base64.b64encode(private_bytes).decode("ascii")
    public_value = base64.b64encode(public_bytes).decode("ascii")

    private_output.parent.mkdir(parents=True, exist_ok=True)
    private_output.write_text(f"{private_value}\n", encoding="ascii")
    public_module.write_text(
        '"""빌드에 포함되는 Ed25519 업데이트 검증 공개키."""\n\n'
        f'UPDATE_PUBLIC_KEY_B64 = "{public_value}"\n',
        encoding="utf-8",
    )

    print(f"개인키 생성: {private_output}")
    print(f"공개키 모듈 갱신: {public_module}")
    print(
        "개인키 파일의 한 줄 전체를 GitHub Secret "
        "UPDATE_SIGNING_PRIVATE_KEY_B64에 저장하세요."
    )
    print("개인키 파일은 커밋하거나 다른 사람에게 보내면 안 됩니다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
