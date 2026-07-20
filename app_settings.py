"""ToonOut 사용자 설정에서 사용하는 모델 저장 경로 도우미."""

import os
import shutil
import tempfile
from pathlib import Path


MODEL_DIRECTORY_SETTING = "model/storage_directory"
# 이전 버전에서 출력 폴더를 영구 저장하던 키. 현재는 삭제 용도로만 쓴다.
LEGACY_OUTPUT_DIRECTORY_SETTING = "output/storage_directory"
OUTPUT_NAMING_PREFIX_SETTING = "output/name_prefix"
OUTPUT_NAMING_SUFFIX_SETTING = "output/name_suffix"


def default_model_directory(local_app_data: str | None = None) -> Path:
    """Windows 사용자별 앱 데이터 폴더 아래의 기본 모델 경로를 반환한다."""
    base_value = local_app_data or os.environ.get("LOCALAPPDATA")
    if base_value:
        base_directory = Path(base_value)
    else:
        base_directory = Path.home() / "AppData" / "Local"
    return base_directory / "ToonOut" / "models"


def default_update_directory(local_app_data: str | None = None) -> Path:
    """검증된 설치 파일을 둘 사용자별 임시 업데이트 폴더."""
    base_value = local_app_data or os.environ.get("LOCALAPPDATA")
    if base_value:
        base_directory = Path(base_value)
    else:
        base_directory = Path.home() / "AppData" / "Local"
    return base_directory / "ToonOut" / "updates"


def ensure_writable_model_directory(directory: str | Path) -> Path:
    """폴더를 만들고 작은 임시 파일로 실제 쓰기 가능 여부를 확인한다."""
    model_directory = Path(directory).expanduser()
    model_directory.mkdir(parents=True, exist_ok=True)

    probe_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=".toonout-write-test-",
            dir=model_directory,
            delete=False,
        ) as probe:
            probe_path = Path(probe.name)
            probe.write(b"ToonOut")
    finally:
        if probe_path is not None:
            probe_path.unlink(missing_ok=True)

    return model_directory


def ensure_writable_directory(directory: str | Path) -> Path:
    """일반 출력 폴더를 만들고 실제로 쓸 수 있는지 확인한다."""
    return ensure_writable_model_directory(directory)


def configure_huggingface_environment(directory: str | Path) -> None:
    """Hub 캐시와 원격 모델 코드 캐시를 모두 앱이 선택한 위치로 묶는다."""
    model_directory = Path(directory)
    os.environ["HF_HOME"] = str(model_directory)
    os.environ["HF_HUB_CACHE"] = str(model_directory)
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(model_directory)
    os.environ["HF_MODULES_CACHE"] = str(model_directory / "modules")


def is_permission_error(error: OSError | Exception) -> bool:
    message = str(error).lower()
    return (
        isinstance(error, PermissionError)
        or "permission" in message
        or "access is denied" in message
        or "액세스가 거부" in message
    )


def available_storage_bytes(directory: str | Path) -> int | None:
    """아직 생성되지 않은 경로도 가장 가까운 상위 폴더 기준으로 계산한다."""
    candidate = Path(directory).expanduser()
    try:
        while not candidate.exists() and candidate != candidate.parent:
            candidate = candidate.parent
        return shutil.disk_usage(candidate).free
    except OSError:
        return None


def format_storage_size(byte_count: int) -> str:
    value = float(byte_count)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024 or unit == "TB":
            digits = 0 if unit in {"B", "KB"} else 1
            return f"{value:.{digits}f} {unit}"
        value /= 1024
    raise AssertionError("도달할 수 없는 저장 공간 단위")
