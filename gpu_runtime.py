"""선택형 NVIDIA GPU worker 팩을 검증하고 사용자 폴더에 관리한다."""

import hashlib
import json
import os
import shutil
import sys
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Callable


GPU_RUNTIME_KIND = "toonout-nvidia-gpu-runtime"
GPU_RUNTIME_SCHEMA = 1
GPU_WORKER_PROTOCOL = 2
GPU_RUNTIME_SETTING = "acceleration/use_gpu_runtime"
GPU_PACK_PATTERN = "ToonOut-NVIDIA-GPU-Pack*.zip"
GPU_PACK_PARTS_PATTERN = "ToonOut-NVIDIA-GPU-Pack*.parts.json"
GPU_PACK_PARTS_KIND = "toonout-nvidia-gpu-pack-parts"
GPU_PACK_PARTS_SCHEMA = 1
MAX_PACK_FILES = 100_000
MAX_UNCOMPRESSED_BYTES = 16 * 1024**3


class GpuRuntimeError(RuntimeError):
    pass


@dataclass(frozen=True)
class GpuRuntimeManifest:
    runtime_version: str
    torch_version: str
    cuda_runtime: str
    worker_path: Path
    files: tuple[dict, ...]
    worker_protocol: int = 1


def gpu_runtime_is_compatible(manifest: GpuRuntimeManifest) -> bool:
    return manifest.worker_protocol == GPU_WORKER_PROTOCOL


def default_gpu_runtime_directory(local_app_data: str | None = None) -> Path:
    base_value = local_app_data or os.environ.get("LOCALAPPDATA")
    if base_value:
        base = Path(base_value)
    else:
        base = Path.home() / "AppData" / "Local"
    return base / "ToonOut" / "runtimes" / "nvidia-gpu"


def _safe_relative_path(value: str) -> Path:
    pure_path = PurePosixPath(value.replace("\\", "/"))
    if (
        pure_path.is_absolute()
        or not pure_path.parts
        or any(part in {"", ".", ".."} for part in pure_path.parts)
        or ":" in pure_path.parts[0]
    ):
        raise GpuRuntimeError(f"안전하지 않은 가속 팩 경로입니다: {value}")
    return Path(*pure_path.parts)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_gpu_runtime_manifest(
    directory: str | Path,
    *,
    verify_files: bool = True,
) -> GpuRuntimeManifest:
    runtime_directory = Path(directory)
    manifest_path = runtime_directory / "manifest.json"
    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise GpuRuntimeError("GPU 가속 팩의 manifest.json을 읽지 못했습니다.") from error

    if data.get("schema_version") != GPU_RUNTIME_SCHEMA:
        raise GpuRuntimeError("지원하지 않는 GPU 가속 팩 형식입니다.")
    if data.get("kind") != GPU_RUNTIME_KIND:
        raise GpuRuntimeError("ToonOut용 NVIDIA GPU 가속 팩이 아닙니다.")

    try:
        worker_relative = _safe_relative_path(str(data["worker"]))
        file_records = tuple(data["files"])
        manifest = GpuRuntimeManifest(
            runtime_version=str(data["runtime_version"]),
            torch_version=str(data["torch_version"]),
            cuda_runtime=str(data["cuda_runtime"]),
            worker_path=runtime_directory / worker_relative,
            files=file_records,
            worker_protocol=int(data.get("worker_protocol", 1)),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise GpuRuntimeError("GPU 가속 팩 정보가 올바르지 않습니다.") from error

    if not manifest.worker_path.is_file():
        raise GpuRuntimeError("GPU worker 실행 파일이 없습니다.")
    if not verify_files:
        return manifest

    for record in manifest.files:
        try:
            relative = _safe_relative_path(str(record["path"]))
            expected_size = int(record["size"])
            expected_hash = str(record["sha256"]).lower()
        except (KeyError, TypeError, ValueError) as error:
            raise GpuRuntimeError("GPU 가속 팩 파일 목록이 올바르지 않습니다.") from error

        file_path = runtime_directory / relative
        if not file_path.is_file() or file_path.stat().st_size != expected_size:
            raise GpuRuntimeError(f"GPU 가속 팩 파일이 없거나 손상되었습니다: {relative}")
        if _sha256(file_path) != expected_hash:
            raise GpuRuntimeError(f"GPU 가속 팩 파일 검증에 실패했습니다: {relative}")

    return manifest


def gpu_runtime_is_installed(directory: str | Path | None = None) -> bool:
    try:
        load_gpu_runtime_manifest(
            directory or default_gpu_runtime_directory(),
            verify_files=False,
        )
        return True
    except GpuRuntimeError:
        return False


def gpu_runtime_size(directory: str | Path | None = None) -> int:
    runtime_directory = Path(directory or default_gpu_runtime_directory())
    try:
        return sum(
            path.stat().st_size
            for path in runtime_directory.rglob("*")
            if path.is_file()
        )
    except OSError:
        return 0


def find_adjacent_gpu_pack(application_path: str | Path | None = None) -> Path | None:
    if application_path is None:
        application_directory = (
            Path(sys.executable).resolve().parent
            if getattr(sys, "frozen", False)
            else Path(__file__).resolve().parent
        )
    else:
        path = Path(application_path).resolve()
        application_directory = path if path.is_dir() else path.parent
    search_directories = [application_directory]
    if not getattr(sys, "frozen", False):
        search_directories.append(application_directory / "dist")
    for directory in search_directories:
        match = next(iter(sorted(directory.glob(GPU_PACK_PARTS_PATTERN))), None)
        if match is not None:
            return match
        match = next(iter(sorted(directory.glob(GPU_PACK_PATTERN))), None)
        if match is not None:
            return match
    return None


def _assemble_split_pack(manifest_path: Path, output_path: Path) -> Path:
    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise GpuRuntimeError("분할 GPU 팩의 manifest를 읽지 못했습니다.") from error
    if (
        data.get("schema_version") != GPU_PACK_PARTS_SCHEMA
        or data.get("kind") != GPU_PACK_PARTS_KIND
    ):
        raise GpuRuntimeError("지원하지 않는 분할 GPU 가속 팩입니다.")
    try:
        archive_size = int(data["archive_size"])
        archive_hash = str(data["archive_sha256"]).lower()
        parts = tuple(data["parts"])
    except (KeyError, TypeError, ValueError) as error:
        raise GpuRuntimeError("분할 GPU 가속 팩 정보가 올바르지 않습니다.") from error
    if archive_size <= 0 or archive_size > MAX_UNCOMPRESSED_BYTES:
        raise GpuRuntimeError("분할 GPU 가속 팩의 전체 크기가 올바르지 않습니다.")
    if not parts or len(parts) > 100:
        raise GpuRuntimeError("분할 GPU 가속 팩의 조각 수가 올바르지 않습니다.")
    if shutil.disk_usage(output_path.parent).free < archive_size:
        required_gb = archive_size / 1_000_000_000
        raise GpuRuntimeError(
            f"GPU 팩을 조립할 디스크 공간이 부족합니다. 최소 {required_gb:.1f}GB의 여유 공간이 필요합니다."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with output_path.open("wb") as output:
        for index, record in enumerate(parts, start=1):
            try:
                filename = str(record["filename"])
                expected_size = int(record["size"])
                expected_hash = str(record["sha256"]).lower()
            except (KeyError, TypeError, ValueError) as error:
                raise GpuRuntimeError("분할 GPU 가속 팩 목록이 올바르지 않습니다.") from error
            relative = _safe_relative_path(filename)
            if len(relative.parts) != 1:
                raise GpuRuntimeError("GPU 팩 조각은 manifest와 같은 폴더에 있어야 합니다.")
            part_path = manifest_path.parent / relative
            if not part_path.is_file() or part_path.stat().st_size != expected_size:
                raise GpuRuntimeError(
                    f"GPU 팩 조각 {index}이 없거나 크기가 다릅니다: {filename}"
                )
            if _sha256(part_path) != expected_hash:
                raise GpuRuntimeError(f"GPU 팩 조각 {index}의 검증에 실패했습니다.")
            with part_path.open("rb") as part:
                shutil.copyfileobj(part, output, length=1024 * 1024)
            written += expected_size
    if written != archive_size or _sha256(output_path) != archive_hash:
        raise GpuRuntimeError("조립된 GPU 가속 팩의 검증에 실패했습니다.")
    return output_path


def install_gpu_runtime(
    pack_path: str | Path,
    destination: str | Path | None = None,
    report: Callable[[str], None] | None = None,
) -> GpuRuntimeManifest:
    pack = Path(pack_path)
    runtime_directory = Path(destination or default_gpu_runtime_directory())
    runtime_root = runtime_directory.parent
    runtime_root.mkdir(parents=True, exist_ok=True)

    def update(message: str):
        if report is not None:
            report(message)

    update("GPU 가속 팩의 파일 목록을 확인하는 중")
    staging_directory = Path(
        tempfile.mkdtemp(prefix=".nvidia-gpu-install-", dir=runtime_root)
    )
    assembled_pack: Path | None = None
    backup_directory: Path | None = None
    try:
        if pack.name.endswith(".parts.json"):
            update("분할 GPU 가속 팩을 검증하고 조립하는 중")
            assembled_pack = runtime_root / f".nvidia-gpu-pack-{os.getpid()}.zip"
            pack = _assemble_split_pack(pack, assembled_pack)
        with zipfile.ZipFile(pack) as archive:
            members = archive.infolist()
            if len(members) > MAX_PACK_FILES:
                raise GpuRuntimeError("GPU 가속 팩에 파일이 너무 많습니다.")
            total_size = sum(member.file_size for member in members)
            if total_size > MAX_UNCOMPRESSED_BYTES:
                raise GpuRuntimeError("GPU 가속 팩의 압축 해제 크기가 너무 큽니다.")
            if shutil.disk_usage(runtime_root).free < total_size:
                required_gb = total_size / 1_000_000_000
                raise GpuRuntimeError(
                    "GPU 팩을 설치할 디스크 공간이 부족합니다. "
                    f"최소 {required_gb:.1f}GB의 여유 공간이 필요합니다."
                )

            update("GPU 가속 파일을 설치하는 중")
            for member in members:
                relative = _safe_relative_path(member.filename)
                target = staging_directory / relative
                if member.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(member) as source, target.open("wb") as output:
                    shutil.copyfileobj(source, output, length=1024 * 1024)

        update("설치된 GPU 가속 파일을 검증하는 중")
        manifest = load_gpu_runtime_manifest(staging_directory)
        if not gpu_runtime_is_compatible(manifest):
            raise GpuRuntimeError(
                "이 GPU 가속 팩은 현재 ToonOut과 호환되지 않습니다. "
                "최신 GPU 가속 팩으로 업데이트하세요."
            )

        if runtime_directory.exists():
            backup_directory = runtime_root / (
                f".{runtime_directory.name}-previous-{os.getpid()}"
            )
            if backup_directory.exists():
                shutil.rmtree(backup_directory)
            runtime_directory.replace(backup_directory)
        staging_directory.replace(runtime_directory)
        staging_directory = runtime_directory
        if backup_directory is not None:
            shutil.rmtree(backup_directory, ignore_errors=True)
        update("GPU 가속 팩 설치가 완료되었습니다")
        return load_gpu_runtime_manifest(runtime_directory, verify_files=False)
    except (OSError, zipfile.BadZipFile) as error:
        if backup_directory is not None and not runtime_directory.exists():
            backup_directory.replace(runtime_directory)
        raise GpuRuntimeError(f"GPU 가속 팩을 설치하지 못했습니다: {error}") from error
    finally:
        if assembled_pack is not None:
            assembled_pack.unlink(missing_ok=True)
        if staging_directory.exists() and staging_directory != runtime_directory:
            shutil.rmtree(staging_directory, ignore_errors=True)


def delete_gpu_runtime(directory: str | Path | None = None) -> None:
    runtime_directory = Path(directory or default_gpu_runtime_directory()).resolve()
    expected = default_gpu_runtime_directory().resolve()
    if directory is None and runtime_directory != expected:
        raise GpuRuntimeError("GPU 가속 팩 삭제 위치가 올바르지 않습니다.")
    if runtime_directory.exists():
        load_gpu_runtime_manifest(runtime_directory, verify_files=False)
        shutil.rmtree(runtime_directory)
