"""ToonOut이 소유한 Hugging Face 모델 캐시의 상태와 파일 작업."""

import os
import shutil
import stat
import uuid
from pathlib import Path
from typing import Callable

from model_files import (
    BASE_MODEL_CODE_FILES,
    BASE_MODEL_REPOSITORY,
    BASE_MODEL_REVISION,
    LINK_FREE_LAYOUT_MARKER,
    TOONOUT_REPOSITORY,
    TOONOUT_REVISION,
    TOONOUT_WEIGHTS,
    has_link_free_layout,
    is_materialized_model_file,
    mark_link_free_layout,
    model_path_entry_exists,
    repository_cache_name,
    repository_directory,
    snapshot_directory,
)


StatusReporter = Callable[[str], None]
MODEL_REPOSITORIES = (BASE_MODEL_REPOSITORY, TOONOUT_REPOSITORY)


def _required_model_paths(
    cache_directory: str | Path,
) -> tuple[tuple[str, tuple[Path, ...]], ...]:
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
    return (
        (
            BASE_MODEL_REPOSITORY,
            tuple(base_snapshot / filename for filename in BASE_MODEL_CODE_FILES),
        ),
        (
            TOONOUT_REPOSITORY,
            (toonout_snapshot / TOONOUT_WEIGHTS,),
        ),
    )


def model_is_installed(cache_directory: str | Path) -> bool:
    # ``stat(..., follow_symlinks=False)`` avoids traversing legacy Hugging
    # Face snapshot links. A cache made of links is deliberately considered an
    # old installation so the installer can replace it with regular files.
    return all(
        is_materialized_model_file(path)
        for _repository, required_paths in _required_model_paths(cache_directory)
        for path in required_paths
    )


def model_storage_size(cache_directory: str | Path) -> int:
    total = 0
    for repository in MODEL_REPOSITORIES:
        repository_path = repository_directory(cache_directory, repository)
        try:
            repository_exists = repository_path.is_dir()
        except OSError:
            repository_exists = False
        if not repository_exists:
            continue
        for path in repository_path.rglob("*"):
            try:
                if path.is_file() and not path.is_symlink():
                    total += path.stat().st_size
            except OSError:
                continue
    return total


def _assert_managed_child(root: Path, target: Path) -> None:
    # Do not call Path.resolve() here: resolving the exact legacy links that we
    # need to delete can itself raise WinError 448 on Windows.
    absolute_root = Path(os.path.abspath(root))
    absolute_target = Path(os.path.abspath(target))
    if (
        absolute_target == absolute_root
        or not absolute_target.is_relative_to(absolute_root)
    ):
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
    targets.append(
        root / "modules" / "transformers_modules" / BASE_MODEL_REVISION
    )
    targets.append(root / LINK_FREE_LAYOUT_MARKER)
    for target in targets:
        _assert_managed_child(root, target)
    return targets


def _managed_path_exists(path: Path) -> bool:
    return model_path_entry_exists(path)


def _remove_managed_path(path: Path) -> None:
    try:
        path_status = path.stat(follow_symlinks=False)
    except FileNotFoundError:
        return
    if stat.S_ISDIR(path_status.st_mode):
        shutil.rmtree(path)
    else:
        # Files and directory links are removed as entries, never followed.
        path.unlink()


def delete_model_files(
    cache_directory: str | Path,
    report_status: StatusReporter | None = None,
) -> None:
    root = Path(cache_directory)
    if report_status is not None:
        report_status("모델 파일을 삭제하는 중")

    for target in managed_model_directories(root):
        _remove_managed_path(target)


def prepare_model_cache_for_install(
    cache_directory: str | Path,
    report_status: StatusReporter | None = None,
) -> bool:
    """완료 설치와 안전한 부분 다운로드는 보존하고 구형 캐시는 정리한다."""

    root = Path(cache_directory)
    if model_is_installed(root):
        return False

    # A partial install created by this version contains only regular files and
    # Hugging Face local_dir metadata. Preserve it so an interrupted large
    # weight download can resume. Legacy caches have no marker and are removed.
    if has_link_free_layout(root):
        unsafe_required_entry = any(
            _managed_path_exists(path) and not is_materialized_model_file(path)
            for _repository, required_paths in _required_model_paths(root)
            for path in required_paths
        )
        if not unsafe_required_entry:
            if report_status is not None:
                report_status("이전 모델 다운로드를 이어받는 중")
            return False

    cleanup_needed = False
    for target in managed_model_directories(root):
        if _managed_path_exists(target):
            cleanup_needed = True
            break

    if cleanup_needed:
        if report_status is not None:
            report_status("완료되지 않은 이전 모델 파일을 정리하는 중")
        delete_model_files(cache_directory)
        return True
    return False


def move_model_files(
    source_directory: str | Path,
    destination_directory: str | Path,
    report_status: StatusReporter | None = None,
) -> bool:
    """새 위치를 검증한 다음 원본을 지운다. 반환값은 원본 정리 성공 여부다."""
    source = Path(source_directory)
    destination = Path(destination_directory)
    source_absolute = Path(os.path.abspath(source))
    destination_absolute = Path(os.path.abspath(destination))
    if source_absolute == destination_absolute:
        raise ValueError("현재 모델 저장 위치와 같은 폴더입니다.")
    if destination_absolute.is_relative_to(source_absolute):
        raise ValueError("현재 모델 저장 폴더 안쪽으로는 모델을 옮길 수 없습니다.")
    if not model_is_installed(source):
        raise FileNotFoundError("이동할 모델 설치를 찾지 못했습니다.")

    destination.mkdir(parents=True, exist_ok=True)
    destination_targets = [
        repository_directory(destination, repository)
        for repository in MODEL_REPOSITORIES
    ]
    if any(_managed_path_exists(target) for target in destination_targets):
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
        mark_link_free_layout(destination)

        if not model_is_installed(destination):
            raise OSError("새 위치에서 모델 설치를 확인하지 못했습니다.")
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        for target in destination_targets:
            try:
                _remove_managed_path(target)
            except OSError:
                pass
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
