"""ToonOut이 소유한 Hugging Face 모델 캐시의 상태와 파일 작업."""

import shutil
import uuid
from pathlib import Path
from typing import Callable

from inference import (
    BASE_MODEL_REPOSITORY,
    BASE_MODEL_REVISION,
    TOONOUT_REPOSITORY,
    TOONOUT_REVISION,
    TOONOUT_WEIGHTS,
)


StatusReporter = Callable[[str], None]
MODEL_REPOSITORIES = (BASE_MODEL_REPOSITORY, TOONOUT_REPOSITORY)
BASE_MODEL_CODE_FILES = (
    "config.json",
    "birefnet.py",
    "BiRefNet_config.py",
)


def repository_cache_name(repository: str) -> str:
    return f"models--{repository.replace('/', '--')}"


def repository_directory(cache_directory: str | Path, repository: str) -> Path:
    return Path(cache_directory) / repository_cache_name(repository)


def snapshot_directory(
    cache_directory: str | Path,
    repository: str,
    revision: str,
) -> Path:
    return repository_directory(cache_directory, repository) / "snapshots" / revision


def model_is_installed(cache_directory: str | Path) -> bool:
    base_snapshot = snapshot_directory(
        cache_directory,
        BASE_MODEL_REPOSITORY,
        BASE_MODEL_REVISION,
    )
    toonout_snapshot = snapshot_directory(
        cache_directory,
        TOONOUT_REPOSITORY,
        TOONOUT_REVISION,
    )

    model_code_ready = all(
        (base_snapshot / filename).is_file()
        for filename in BASE_MODEL_CODE_FILES
    )
    toonout_weights_ready = (toonout_snapshot / TOONOUT_WEIGHTS).is_file()
    return model_code_ready and toonout_weights_ready


def model_storage_size(cache_directory: str | Path) -> int:
    total = 0
    for repository in MODEL_REPOSITORIES:
        repository_path = repository_directory(cache_directory, repository)
        if not repository_path.exists():
            continue
        for path in repository_path.rglob("*"):
            try:
                if path.is_file() and not path.is_symlink():
                    total += path.stat().st_size
            except OSError:
                continue
    return total


def _assert_managed_child(root: Path, target: Path) -> None:
    resolved_root = root.resolve()
    resolved_target = target.resolve()
    if resolved_target == resolved_root or not resolved_target.is_relative_to(resolved_root):
        raise ValueError("모델 캐시 밖의 경로는 변경할 수 없습니다.")


def managed_model_directories(cache_directory: str | Path) -> list[Path]:
    root = Path(cache_directory)
    targets = [
        repository_directory(root, repository)
        for repository in MODEL_REPOSITORIES
    ]
    targets.extend(
        root / ".locks" / repository_cache_name(repository)
        for repository in MODEL_REPOSITORIES
    )
    targets.append(
        root / "modules" / "transformers_modules" / "ZhengPeng7" / "BiRefNet"
    )
    for target in targets:
        _assert_managed_child(root, target)
    return targets


def delete_model_files(
    cache_directory: str | Path,
    report_status: StatusReporter | None = None,
) -> None:
    root = Path(cache_directory)
    if report_status is not None:
        report_status("모델 파일을 삭제하는 중")

    for target in managed_model_directories(root):
        if target.is_dir() and not target.is_symlink():
            shutil.rmtree(target)
        elif target.exists() or target.is_symlink():
            target.unlink()


def move_model_files(
    source_directory: str | Path,
    destination_directory: str | Path,
    report_status: StatusReporter | None = None,
) -> bool:
    """새 위치를 검증한 다음 원본을 지운다. 반환값은 원본 정리 성공 여부다."""
    source = Path(source_directory)
    destination = Path(destination_directory)
    source_resolved = source.resolve()
    destination_resolved = destination.resolve()
    if source_resolved == destination_resolved:
        raise ValueError("현재 모델 저장 위치와 같은 폴더입니다.")
    if destination_resolved.is_relative_to(source_resolved):
        raise ValueError("현재 모델 저장 폴더 안쪽으로는 모델을 옮길 수 없습니다.")
    if not model_is_installed(source):
        raise FileNotFoundError("이동할 모델 설치를 찾지 못했습니다.")

    destination.mkdir(parents=True, exist_ok=True)
    destination_targets = [
        repository_directory(destination, repository)
        for repository in MODEL_REPOSITORIES
    ]
    if any(target.exists() for target in destination_targets):
        raise FileExistsError(
            "새 위치에 같은 모델 캐시가 이미 있습니다. 다른 폴더를 선택하세요."
        )

    staging = destination / f".toonout-transfer-{uuid.uuid4().hex}"
    _assert_managed_child(destination, staging)
    staging.mkdir()

    try:
        for index, repository in enumerate(MODEL_REPOSITORIES, start=1):
            if report_status is not None:
                report_status(f"모델 파일 복사 중 · {index} / {len(MODEL_REPOSITORIES)}")
            shutil.copytree(
                repository_directory(source, repository),
                repository_directory(staging, repository),
                symlinks=False,
            )

        if not model_is_installed(staging):
            raise OSError("복사한 모델 파일의 검증에 실패했습니다.")

        if report_status is not None:
            report_status("새 위치의 모델 파일을 확인하는 중")
        for repository in MODEL_REPOSITORIES:
            shutil.move(
                str(repository_directory(staging, repository)),
                str(repository_directory(destination, repository)),
            )

        if not model_is_installed(destination):
            raise OSError("새 위치에서 모델 설치를 확인하지 못했습니다.")
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        for target in destination_targets:
            if target.exists():
                shutil.rmtree(target, ignore_errors=True)
        raise
    finally:
        shutil.rmtree(staging, ignore_errors=True)

    if report_status is not None:
        report_status("기존 위치를 정리하는 중")
    try:
        delete_model_files(source)
    except OSError:
        return False
    return True
