"""UI와 독립적인 ToonOut 업데이트 확인·검증·다운로드 경계."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Callable
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PublicKey,
)


VERSION_PATTERN = re.compile(r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)$")
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
MAX_INSTALLER_BYTES = 5 * 1024**3


class UpdateError(RuntimeError):
    """사용자에게 안전하게 요약할 수 있는 업데이트 실패."""


class UpdateConfigurationError(UpdateError):
    """공개키 또는 배포 설정이 아직 준비되지 않음."""


class UpdateSecurityError(UpdateError):
    """서명, 해시 또는 manifest 검증 실패."""


class UpdateCancelled(UpdateError):
    """앱 종료 등으로 다운로드를 중단함."""


@dataclass(frozen=True)
class UpdateRelease:
    version: str
    minimum_supported_version: str
    installer_url: str
    installer_name: str
    installer_size: int
    installer_sha256: str
    release_notes_url: str
    gpu_worker_protocol: int


def parse_version(value: str) -> tuple[int, int, int]:
    match = VERSION_PATTERN.fullmatch(value)
    if match is None:
        raise UpdateSecurityError(f"올바르지 않은 버전 형식: {value}")
    return tuple(int(part) for part in match.groups())


def is_newer_version(candidate: str, current: str) -> bool:
    return parse_version(candidate) > parse_version(current)


def canonical_payload_bytes(payload: dict[str, object]) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _decode_base64(value: str, label: str) -> bytes:
    try:
        return base64.b64decode(value, validate=True)
    except (ValueError, TypeError) as error:
        raise UpdateSecurityError(f"{label} 형식이 올바르지 않습니다") from error


def _required_string(payload: dict[str, object], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise UpdateSecurityError(f"manifest에 {key} 값이 없습니다")
    return value


def _https_url(payload: dict[str, object], key: str) -> str:
    value = _required_string(payload, key)
    parsed = urlparse(value)
    if parsed.scheme != "https" or not parsed.netloc:
        raise UpdateSecurityError(f"{key}는 HTTPS 주소여야 합니다")
    return value


def verify_signed_manifest(
    document: bytes,
    public_key_b64: str,
) -> UpdateRelease:
    if not public_key_b64:
        raise UpdateConfigurationError(
            "자동 업데이트 공개키가 구성되지 않았습니다"
        )
    try:
        envelope = json.loads(document.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise UpdateSecurityError("업데이트 manifest를 읽을 수 없습니다") from error
    if not isinstance(envelope, dict):
        raise UpdateSecurityError("업데이트 manifest 구조가 올바르지 않습니다")

    payload = envelope.get("payload")
    signature_value = envelope.get("signature")
    if not isinstance(payload, dict) or not isinstance(signature_value, str):
        raise UpdateSecurityError("업데이트 manifest 서명 정보가 없습니다")

    public_key_bytes = _decode_base64(public_key_b64, "공개키")
    signature = _decode_base64(signature_value, "서명")
    if len(public_key_bytes) != 32:
        raise UpdateConfigurationError("Ed25519 공개키 길이가 올바르지 않습니다")
    try:
        Ed25519PublicKey.from_public_bytes(public_key_bytes).verify(
            signature,
            canonical_payload_bytes(payload),
        )
    except InvalidSignature as error:
        raise UpdateSecurityError(
            "업데이트 정보의 디지털 서명이 일치하지 않습니다"
        ) from error

    if payload.get("schema_version") != 1:
        raise UpdateSecurityError("지원하지 않는 업데이트 manifest 버전입니다")
    version = _required_string(payload, "version")
    minimum_supported_version = _required_string(
        payload,
        "minimum_supported_version",
    )
    parse_version(version)
    parse_version(minimum_supported_version)

    installer_name = _required_string(payload, "installer_name")
    if (
        Path(installer_name).name != installer_name
        or not installer_name.casefold().endswith(".exe")
    ):
        raise UpdateSecurityError("설치 파일 이름이 안전하지 않습니다")

    installer_size = payload.get("installer_size")
    if (
        not isinstance(installer_size, int)
        or isinstance(installer_size, bool)
        or not 0 < installer_size <= MAX_INSTALLER_BYTES
    ):
        raise UpdateSecurityError("설치 파일 크기가 올바르지 않습니다")

    installer_sha256 = _required_string(
        payload,
        "installer_sha256",
    ).casefold()
    if SHA256_PATTERN.fullmatch(installer_sha256) is None:
        raise UpdateSecurityError("설치 파일 SHA-256이 올바르지 않습니다")

    gpu_worker_protocol = payload.get("gpu_worker_protocol")
    if (
        not isinstance(gpu_worker_protocol, int)
        or isinstance(gpu_worker_protocol, bool)
        or gpu_worker_protocol < 1
    ):
        raise UpdateSecurityError("GPU worker protocol 값이 올바르지 않습니다")

    return UpdateRelease(
        version=version,
        minimum_supported_version=minimum_supported_version,
        installer_url=_https_url(payload, "installer_url"),
        installer_name=installer_name,
        installer_size=installer_size,
        installer_sha256=installer_sha256,
        release_notes_url=_https_url(payload, "release_notes_url"),
        gpu_worker_protocol=gpu_worker_protocol,
    )


def fetch_update_release(
    manifest_url: str,
    public_key_b64: str,
    *,
    timeout: float = 8.0,
) -> UpdateRelease:
    request = Request(
        manifest_url,
        headers={"User-Agent": "ToonOut-Update/1"},
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            document = response.read(1024 * 1024 + 1)
    except OSError as error:
        raise UpdateError("업데이트 서버에 연결할 수 없습니다") from error
    if len(document) > 1024 * 1024:
        raise UpdateSecurityError("업데이트 manifest가 너무 큽니다")
    return verify_signed_manifest(document, public_key_b64)


def verify_installer_file(release: UpdateRelease, path: str | Path) -> Path:
    installer = Path(path)
    try:
        if installer.stat().st_size != release.installer_size:
            raise UpdateSecurityError("설치 파일 크기가 일치하지 않습니다")
        digest = hashlib.sha256()
        with installer.open("rb") as file:
            for chunk in iter(lambda: file.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise UpdateError("다운로드한 설치 파일을 읽을 수 없습니다") from error
    if digest.hexdigest() != release.installer_sha256:
        raise UpdateSecurityError("설치 파일 SHA-256이 일치하지 않습니다")
    return installer


def prune_update_cache(
    update_root: str | Path,
    keep_version: str | None,
) -> bool:
    """앱 버전 폴더를 지우고 모두 사라졌는지 반환한다."""
    if keep_version is not None:
        parse_version(keep_version)
    root = Path(update_root)
    if not root.is_dir():
        return True
    try:
        children = list(root.iterdir())
    except OSError:
        return False
    for child in children:
        if (
            (keep_version is not None and child.name == keep_version)
            or VERSION_PATTERN.fullmatch(child.name) is None
        ):
            continue
        try:
            if child.is_symlink():
                child.unlink()
            elif child.is_dir():
                shutil.rmtree(child)
        except OSError:
            # 캐시 정리 실패는 새 업데이트 다운로드를 막지 않는다.
            continue

    try:
        remaining_version_directory = any(
            VERSION_PATTERN.fullmatch(child.name) is not None
            for child in root.iterdir()
        )
    except OSError:
        return False
    if not remaining_version_directory:
        try:
            root.rmdir()
        except OSError:
            # 알 수 없는 사용자 파일이 있으면 업데이트 루트는 보존한다.
            pass
    return not remaining_version_directory


def download_installer(
    release: UpdateRelease,
    destination_directory: str | Path,
    *,
    progress: Callable[[int, int], None] | None = None,
    cancelled: Callable[[], bool] | None = None,
    timeout: float = 15.0,
) -> Path:
    destination = Path(destination_directory)
    destination.mkdir(parents=True, exist_ok=True)
    final_path = destination / release.installer_name
    partial_path = destination / f"{release.installer_name}.part"

    if final_path.is_file():
        try:
            return verify_installer_file(release, final_path)
        except UpdateError:
            final_path.unlink(missing_ok=True)

    request = Request(
        release.installer_url,
        headers={"User-Agent": "ToonOut-Update/1"},
    )
    received = 0
    digest = hashlib.sha256()
    try:
        with urlopen(request, timeout=timeout) as response, partial_path.open(
            "wb"
        ) as output:
            while True:
                if cancelled is not None and cancelled():
                    raise UpdateCancelled("업데이트 다운로드를 중단했습니다")
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                received += len(chunk)
                if received > release.installer_size:
                    raise UpdateSecurityError(
                        "설치 파일이 manifest 크기보다 큽니다"
                    )
                digest.update(chunk)
                output.write(chunk)
                if progress is not None:
                    progress(received, release.installer_size)

        if received != release.installer_size:
            raise UpdateSecurityError("설치 파일 크기가 일치하지 않습니다")
        if digest.hexdigest() != release.installer_sha256:
            raise UpdateSecurityError("설치 파일 SHA-256이 일치하지 않습니다")
        os.replace(partial_path, final_path)
        return final_path
    except UpdateError:
        partial_path.unlink(missing_ok=True)
        raise
    except OSError as error:
        partial_path.unlink(missing_ok=True)
        raise UpdateError("업데이트 설치 파일을 다운로드하지 못했습니다") from error
