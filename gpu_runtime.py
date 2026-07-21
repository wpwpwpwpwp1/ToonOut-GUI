"""선택형 NVIDIA GPU worker 팩을 검증하고 사용자 폴더에 관리한다."""

import bisect
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
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


class GpuRuntimeCancelled(GpuRuntimeError):
    pass


@dataclass(frozen=True)
class GpuRuntimeManifest:
    runtime_version: str
    torch_version: str
    cuda_runtime: str
    worker_path: Path
    files: tuple[dict, ...]
    worker_protocol: int = 1


@dataclass(frozen=True)
class _SplitPackPart:
    path: Path
    size: int
    sha256: str


@dataclass(frozen=True)
class _SplitPack:
    archive_size: int
    archive_sha256: str
    parts: tuple[_SplitPackPart, ...]


class _SplitPackReader(io.RawIOBase):
    """여러 분할 파일을 복사 없이 하나의 seek 가능한 ZIP처럼 읽는다."""

    def __init__(self, pack: _SplitPack):
        super().__init__()
        self._parts = pack.parts
        self._size = pack.archive_size
        self._ends: list[int] = []
        end = 0
        for part in self._parts:
            end += part.size
            self._ends.append(end)
        self._position = 0
        self._open_index: int | None = None
        self._open_file = None

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self._position

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            position = offset
        elif whence == io.SEEK_CUR:
            position = self._position + offset
        elif whence == io.SEEK_END:
            position = self._size + offset
        else:
            raise ValueError(f"지원하지 않는 seek 기준입니다: {whence}")
        if position < 0:
            raise OSError("분할 GPU 팩의 시작보다 앞을 읽을 수 없습니다.")
        self._position = position
        return position

    def read(self, size: int = -1) -> bytes:
        if self.closed:
            raise ValueError("닫힌 분할 GPU 팩을 읽을 수 없습니다.")
        if self._position >= self._size or size == 0:
            return b""
        if size is None or size < 0:
            size = self._size - self._position
        remaining = min(size, self._size - self._position)
        result = bytearray()
        while remaining > 0:
            part_index = bisect.bisect_right(self._ends, self._position)
            part_start = 0 if part_index == 0 else self._ends[part_index - 1]
            part_end = self._ends[part_index]
            if self._open_index != part_index:
                if self._open_file is not None:
                    self._open_file.close()
                self._open_file = self._parts[part_index].path.open("rb")
                self._open_index = part_index
            self._open_file.seek(self._position - part_start)
            chunk = self._open_file.read(min(remaining, part_end - self._position))
            if not chunk:
                raise OSError("분할 GPU 팩을 끝까지 읽지 못했습니다.")
            result.extend(chunk)
            self._position += len(chunk)
            remaining -= len(chunk)
        return bytes(result)

    def close(self) -> None:
        if self._open_file is not None:
            self._open_file.close()
            self._open_file = None
        super().close()


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


def _raise_if_cancelled(
    should_cancel: Callable[[], bool] | None,
) -> None:
    if should_cancel is not None and should_cancel():
        raise GpuRuntimeCancelled("GPU 가속 팩 설치를 취소했습니다.")


def _sha256(
    path: Path,
    should_cancel: Callable[[], bool] | None = None,
) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            _raise_if_cancelled(should_cancel)
            digest.update(chunk)
    return digest.hexdigest()


def load_gpu_runtime_manifest(
    directory: str | Path,
    *,
    verify_files: bool = True,
    should_cancel: Callable[[], bool] | None = None,
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
        _raise_if_cancelled(should_cancel)
        try:
            relative = _safe_relative_path(str(record["path"]))
            expected_size = int(record["size"])
            expected_hash = str(record["sha256"]).lower()
        except (KeyError, TypeError, ValueError) as error:
            raise GpuRuntimeError("GPU 가속 팩 파일 목록이 올바르지 않습니다.") from error

        file_path = runtime_directory / relative
        if not file_path.is_file() or file_path.stat().st_size != expected_size:
            raise GpuRuntimeError(f"GPU 가속 팩 파일이 없거나 손상되었습니다: {relative}")
        if _sha256(file_path, should_cancel) != expected_hash:
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
            _file_storage_size(path)
            for path in runtime_directory.rglob("*")
            if path.is_file()
        )
    except OSError:
        return 0


def _file_storage_size(path: Path) -> int:
    """압축 파일은 논리 크기가 아니라 실제 디스크 점유량을 반환한다."""
    logical_size = path.stat().st_size
    if os.name != "nt":
        return logical_size
    try:
        import ctypes
        from ctypes import wintypes

        get_compressed_size = ctypes.WinDLL(
            "kernel32",
            use_last_error=True,
        ).GetCompressedFileSizeW
        get_compressed_size.argtypes = [
            wintypes.LPCWSTR,
            ctypes.POINTER(wintypes.DWORD),
        ]
        get_compressed_size.restype = wintypes.DWORD
        high = wintypes.DWORD()
        ctypes.set_last_error(0)
        low = get_compressed_size(str(path), ctypes.byref(high))
        if low == 0xFFFFFFFF and ctypes.get_last_error() != 0:
            return logical_size
        return (high.value << 32) | low
    except (AttributeError, OSError, ValueError):
        return logical_size


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


def _load_split_pack(manifest_path: Path) -> _SplitPack:
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
    pack_parts = []
    for record in parts:
        try:
            filename = str(record["filename"])
            expected_size = int(record["size"])
            expected_hash = str(record["sha256"]).lower()
        except (KeyError, TypeError, ValueError) as error:
            raise GpuRuntimeError("분할 GPU 가속 팩 목록이 올바르지 않습니다.") from error
        relative = _safe_relative_path(filename)
        if len(relative.parts) != 1:
            raise GpuRuntimeError("GPU 팩 조각은 manifest와 같은 폴더에 있어야 합니다.")
        pack_parts.append(
            _SplitPackPart(
                manifest_path.parent / relative,
                expected_size,
                expected_hash,
            )
        )
    return _SplitPack(archive_size, archive_hash, tuple(pack_parts))


def _verify_split_pack(
    pack: _SplitPack,
    should_cancel: Callable[[], bool] | None = None,
) -> None:
    archive_digest = hashlib.sha256()
    total_size = 0
    for index, part in enumerate(pack.parts, start=1):
        _raise_if_cancelled(should_cancel)
        if not part.path.is_file() or part.path.stat().st_size != part.size:
            raise GpuRuntimeError(
                f"GPU 팩 조각 {index}이 없거나 크기가 다릅니다: {part.path.name}"
            )
        part_digest = hashlib.sha256()
        read_size = 0
        with part.path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                _raise_if_cancelled(should_cancel)
                part_digest.update(chunk)
                archive_digest.update(chunk)
                read_size += len(chunk)
        if read_size != part.size or part_digest.hexdigest() != part.sha256:
            raise GpuRuntimeError(f"GPU 팩 조각 {index}의 검증에 실패했습니다.")
        total_size += read_size
    if (
        total_size != pack.archive_size
        or archive_digest.hexdigest() != pack.archive_sha256
    ):
        raise GpuRuntimeError("분할 GPU 가속 팩 전체 검증에 실패했습니다.")


def _compress_runtime_directory(
    directory: Path,
    should_cancel: Callable[[], bool] | None = None,
) -> bool:
    """Windows가 지원하면 실행 가능한 파일을 LZX로 투명 압축한다."""
    if os.name != "nt":
        return False
    compact = shutil.which("compact.exe")
    if compact is None:
        return False
    creation_flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    try:
        process = subprocess.Popen(
            [compact, "/C", "/S", "/I", "/Q", "/EXE:LZX", "*"],
            cwd=directory,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=creation_flags,
        )
        while process.poll() is None:
            if should_cancel is not None and should_cancel():
                process.terminate()
                try:
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                raise GpuRuntimeCancelled(
                    "GPU 가속 팩 설치를 취소했습니다."
                )
            time.sleep(0.1)
    except GpuRuntimeCancelled:
        raise
    except OSError:
        return False
    return process.returncode == 0


def install_gpu_runtime(
    pack_path: str | Path,
    destination: str | Path | None = None,
    report: Callable[[str], None] | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> GpuRuntimeManifest:
    pack = Path(pack_path)
    runtime_directory = Path(destination or default_gpu_runtime_directory())
    runtime_root = runtime_directory.parent
    runtime_root.mkdir(parents=True, exist_ok=True)

    def update(message: str):
        _raise_if_cancelled(should_cancel)
        if report is not None:
            report(message)

    update("GPU 가속 팩의 파일 목록을 확인하는 중")
    staging_directory = Path(
        tempfile.mkdtemp(prefix=".nvidia-gpu-install-", dir=runtime_root)
    )
    split_reader: _SplitPackReader | None = None
    backup_directory: Path | None = None
    try:
        if pack.name.endswith(".parts.json"):
            update("분할 GPU 가속 팩을 검증하는 중")
            split_pack = _load_split_pack(pack)
            _verify_split_pack(split_pack, should_cancel)
            split_reader = _SplitPackReader(split_pack)
            archive_source = split_reader
        else:
            archive_source = pack
        with zipfile.ZipFile(archive_source) as archive:
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
                    while chunk := source.read(1024 * 1024):
                        _raise_if_cancelled(should_cancel)
                        output.write(chunk)

        update("설치된 GPU 가속 파일을 검증하는 중")
        manifest = load_gpu_runtime_manifest(
            staging_directory,
            should_cancel=should_cancel,
        )
        if not gpu_runtime_is_compatible(manifest):
            raise GpuRuntimeError(
                "이 GPU 가속 팩은 현재 ToonOut과 호환되지 않습니다. "
                "최신 GPU 가속 팩으로 업데이트하세요."
            )

        update("GPU 가속 팩의 디스크 사용량을 줄이는 중")
        if not _compress_runtime_directory(staging_directory, should_cancel):
            update("디스크 압축을 지원하지 않아 일반 방식으로 설치합니다")
        _raise_if_cancelled(should_cancel)

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
        if report is not None:
            report("GPU 가속 팩 설치가 완료되었습니다")
        return load_gpu_runtime_manifest(runtime_directory, verify_files=False)
    except GpuRuntimeCancelled:
        if backup_directory is not None and not runtime_directory.exists():
            backup_directory.replace(runtime_directory)
        raise
    except (OSError, zipfile.BadZipFile) as error:
        if backup_directory is not None and not runtime_directory.exists():
            backup_directory.replace(runtime_directory)
        raise GpuRuntimeError(f"GPU 가속 팩을 설치하지 못했습니다: {error}") from error
    finally:
        if split_reader is not None:
            split_reader.close()
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
