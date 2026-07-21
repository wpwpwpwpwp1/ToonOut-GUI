"""현재 ToonOut에 맞는 NVIDIA GPU 팩을 안전하게 내려받아 설치한다."""

from __future__ import annotations

import hashlib
import os
import platform
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urljoin, urlparse
from urllib.request import Request, urlopen

from gpu_runtime import (
    GPU_WORKER_PROTOCOL,
    GpuRuntimeCancelled,
    GpuRuntimeError,
    GpuRuntimeManifest,
    install_gpu_runtime,
    load_gpu_pack_parts,
)


GPU_PACK_MANIFEST_NAME = "ToonOut-NVIDIA-GPU-Pack.parts.json"
# 이 값들은 배포할 분할 manifest 자체를 신뢰하기 위한 앱 내 고정값이다.
# GPU 팩을 다시 빌드하면 scripts/update_gpu_pack_config.py로 갱신한다.
GPU_PACK_MANIFEST_SIZE = 622
GPU_PACK_MANIFEST_SHA256 = (
    "e69198871ac71286a1b6d6b6e5ab1958237efe4010c9d39506c4c9567b7b7384"
)
GPU_PACK_ARCHIVE_SIZE = 3_265_060_719
GPU_INSTALL_REQUIRED_FREE_BYTES = 10_000_000_000
GPU_DOWNLOAD_PROGRESS_END = 40
DOWNLOAD_CHUNK_SIZE = 1024 * 1024


@dataclass(frozen=True)
class GpuPackRelease:
    app_version: str
    manifest_url: str
    manifest_name: str
    manifest_size: int
    manifest_sha256: str
    archive_size: int
    worker_protocol: int


def current_gpu_pack_release(
    app_version: str,
    *,
    system: str | None = None,
    machine: str | None = None,
) -> GpuPackRelease:
    """현재 Windows x64 앱 버전에 고정된 NVIDIA 팩을 선택한다."""

    current_system = (system or sys.platform).casefold()
    current_machine = (machine or platform.machine()).casefold()
    if current_system not in {"win32", "windows"}:
        raise GpuRuntimeError("GPU 가속 팩 자동 설치는 Windows에서만 지원합니다.")
    if current_machine not in {"amd64", "x86_64"}:
        raise GpuRuntimeError(
            "이 PC의 Windows 아키텍처에 맞는 GPU 가속 팩이 아직 없습니다."
        )

    manifest_url = (
        "https://github.com/wpwpwpwpwp1/ToonOut-GUI/releases/download/"
        f"v{app_version}/{quote(GPU_PACK_MANIFEST_NAME)}"
    )
    return GpuPackRelease(
        app_version=app_version,
        manifest_url=manifest_url,
        manifest_name=GPU_PACK_MANIFEST_NAME,
        manifest_size=GPU_PACK_MANIFEST_SIZE,
        manifest_sha256=GPU_PACK_MANIFEST_SHA256,
        archive_size=GPU_PACK_ARCHIVE_SIZE,
        worker_protocol=GPU_WORKER_PROTOCOL,
    )


def _raise_if_cancelled(cancelled: Callable[[], bool] | None) -> None:
    if cancelled is not None and cancelled():
        raise GpuRuntimeCancelled("GPU 가속 팩 설치를 취소했습니다.")


def _download_verified_file(
    url: str,
    destination: Path,
    expected_size: int,
    expected_sha256: str,
    *,
    progress: Callable[[int], None] | None = None,
    cancelled: Callable[[], bool] | None = None,
    timeout: float = 30.0,
) -> Path:
    parsed = urlparse(url)
    if parsed.scheme != "https" or not parsed.netloc:
        raise GpuRuntimeError("GPU 가속 팩 다운로드 주소가 안전하지 않습니다.")

    partial_path = destination.with_name(f"{destination.name}.part")
    request = Request(url, headers={"User-Agent": "ToonOut-GPU-Pack/1"})
    received = 0
    digest = hashlib.sha256()
    try:
        _raise_if_cancelled(cancelled)
        with urlopen(request, timeout=timeout) as response, partial_path.open(
            "wb"
        ) as output:
            while True:
                _raise_if_cancelled(cancelled)
                chunk = response.read(DOWNLOAD_CHUNK_SIZE)
                if not chunk:
                    break
                received += len(chunk)
                if received > expected_size:
                    raise GpuRuntimeError(
                        "GPU 가속 팩 파일이 배포 정보의 크기보다 큽니다."
                    )
                digest.update(chunk)
                output.write(chunk)
                if progress is not None:
                    progress(received)

        if received != expected_size:
            raise GpuRuntimeError("GPU 가속 팩 파일 크기가 일치하지 않습니다.")
        if digest.hexdigest() != expected_sha256:
            raise GpuRuntimeError("GPU 가속 팩 SHA-256 검증에 실패했습니다.")
        os.replace(partial_path, destination)
        return destination
    except GpuRuntimeCancelled:
        raise
    except HTTPError as error:
        if error.code == 404:
            raise GpuRuntimeError(
                "현재 ToonOut 버전용 GPU 가속 팩을 배포 서버에서 찾지 "
                "못했습니다. 앱과 GPU 팩 배포가 모두 완료되었는지 확인하세요."
            ) from error
        if error.code in {403, 429}:
            raise GpuRuntimeError(
                "GPU 가속 팩 다운로드가 일시적으로 제한되었습니다. 잠시 후 "
                "다시 시도하세요."
            ) from error
        raise GpuRuntimeError(
            f"GPU 가속 팩 다운로드 서버가 오류를 반환했습니다 ({error.code})."
        ) from error
    except (TimeoutError, URLError) as error:
        raise GpuRuntimeError(
            "GPU 가속 팩을 내려받지 못했습니다. 인터넷 연결과 프록시·방화벽을 "
            "확인한 뒤 다시 시도하세요."
        ) from error
    except OSError as error:
        raise GpuRuntimeError(
            "GPU 가속 팩 임시 파일을 저장하지 못했습니다. 저장 공간과 사용자 "
            "폴더 권한을 확인하세요."
        ) from error
    finally:
        partial_path.unlink(missing_ok=True)


def download_gpu_pack(
    release: GpuPackRelease,
    destination_directory: str | Path,
    *,
    report_progress: Callable[[str, int], None] | None = None,
    cancelled: Callable[[], bool] | None = None,
) -> Path:
    """서명된 앱에 고정된 manifest와 모든 조각을 검증하며 다운로드한다."""

    if release.worker_protocol != GPU_WORKER_PROTOCOL:
        raise GpuRuntimeError(
            "현재 ToonOut과 호환되는 GPU 가속 팩 배포 정보가 아닙니다."
        )
    destination = Path(destination_directory)
    destination.mkdir(parents=True, exist_ok=True)

    def emit(status: str, percent: int) -> None:
        if report_progress is not None:
            report_progress(status, max(0, min(GPU_DOWNLOAD_PROGRESS_END, percent)))

    emit("GPU 가속 팩 배포 정보 확인 중", 0)
    manifest_path = destination / release.manifest_name
    _download_verified_file(
        release.manifest_url,
        manifest_path,
        release.manifest_size,
        release.manifest_sha256,
        progress=lambda _received: emit("GPU 가속 팩 배포 정보 확인 중", 1),
        cancelled=cancelled,
    )
    pack = load_gpu_pack_parts(manifest_path)
    if pack.archive_size != release.archive_size:
        raise GpuRuntimeError(
            "GPU 가속 팩 전체 크기가 현재 ToonOut의 배포 정보와 다릅니다."
        )

    downloaded = 0
    base_url = release.manifest_url.rsplit("/", 1)[0] + "/"
    for index, part in enumerate(pack.parts, start=1):
        _raise_if_cancelled(cancelled)
        status = f"GPU 가속 팩 다운로드 중 · {index} / {len(pack.parts)}"
        part_url = urljoin(base_url, quote(part.path.name))

        def update_part(received: int, *, completed: int = downloaded) -> None:
            emit(
                status,
                1
                + round(
                    (completed + received)
                    * (GPU_DOWNLOAD_PROGRESS_END - 1)
                    / pack.archive_size
                ),
            )

        _download_verified_file(
            part_url,
            part.path,
            part.size,
            part.sha256,
            progress=update_part,
            cancelled=cancelled,
        )
        downloaded += part.size

    emit("GPU 가속 팩 다운로드 완료", GPU_DOWNLOAD_PROGRESS_END)
    return manifest_path


def download_and_install_gpu_runtime(
    release: GpuPackRelease,
    destination: str | Path,
    *,
    report: Callable[[str], None] | None = None,
    report_progress: Callable[[str, int], None] | None = None,
    cancelled: Callable[[], bool] | None = None,
) -> GpuRuntimeManifest:
    """다운로드 임시 파일을 항상 정리하고 검증된 팩만 원자적으로 적용한다."""

    runtime_directory = Path(destination)
    runtime_root = runtime_directory.parent
    try:
        runtime_root.mkdir(parents=True, exist_ok=True)
        free_bytes = shutil.disk_usage(runtime_root).free
    except OSError as error:
        raise GpuRuntimeError(
            "GPU 가속 팩 설치 위치를 준비하지 못했습니다. 사용자 폴더 권한을 "
            "확인하세요."
        ) from error
    if free_bytes < GPU_INSTALL_REQUIRED_FREE_BYTES:
        raise GpuRuntimeError(
            "GPU 가속 팩 설치를 시작하려면 설치 드라이브에 10GB 이상의 "
            "여유 공간이 필요합니다."
        )

    download_directory = Path(
        tempfile.mkdtemp(prefix=".nvidia-gpu-download-", dir=runtime_root)
    )
    try:
        pack_path = download_gpu_pack(
            release,
            download_directory,
            report_progress=report_progress,
            cancelled=cancelled,
        )

        def install_progress(status: str, percent: int) -> None:
            if report_progress is not None:
                report_progress(
                    status,
                    GPU_DOWNLOAD_PROGRESS_END
                    + round(percent * (100 - GPU_DOWNLOAD_PROGRESS_END) / 100),
                )

        return install_gpu_runtime(
            pack_path,
            runtime_directory,
            report=report,
            should_cancel=cancelled,
            report_progress=install_progress,
        )
    finally:
        shutil.rmtree(download_directory, ignore_errors=True)
