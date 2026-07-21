"""Pinned ToonOut model sources and their link-free local file layout."""

import stat
from pathlib import Path


BASE_MODEL_REPOSITORY = "ZhengPeng7/BiRefNet"
BASE_MODEL_REVISION = "e2bf8e4460fc8fa32bba5ea4d94b3233d367b0e4"
BASE_MODEL_CODE_FILES = (
    "config.json",
    "birefnet.py",
    "BiRefNet_config.py",
)
TOONOUT_REPOSITORY = "joelseytre/toonout"
TOONOUT_REVISION = "cbf720eca394edcde66b861a8a8c20fbabe9c748"
TOONOUT_WEIGHTS = "birefnet_finetuned_toonout.pth"

# v0.1.8 and earlier used the Hugging Face snapshot cache, whose files can be
# symbolic links. Windows may reject those links with WinError 448 when the
# process trust level changes. New installs use local_dir and regular files.
LINK_FREE_LAYOUT_MARKER = ".toonout-link-free-models-v1"


def repository_cache_name(repository: str) -> str:
    return f"models--{repository.replace('/', '--')}"


def repository_directory(
    model_directory: str | Path,
    repository: str,
) -> Path:
    return Path(model_directory) / repository_cache_name(repository)


def snapshot_directory(
    model_directory: str | Path,
    repository: str,
    revision: str,
) -> Path:
    return repository_directory(model_directory, repository) / "snapshots" / revision


def link_free_layout_marker(model_directory: str | Path) -> Path:
    return Path(model_directory) / LINK_FREE_LAYOUT_MARKER


def is_materialized_model_file(path: str | Path) -> bool:
    """Return true only for a non-empty regular file without following links."""

    try:
        file_status = Path(path).stat(follow_symlinks=False)
    except OSError:
        return False
    return stat.S_ISREG(file_status.st_mode) and file_status.st_size > 0


def model_path_entry_exists(path: str | Path) -> bool:
    """Check an entry itself without traversing a link or mount point."""

    try:
        Path(path).stat(follow_symlinks=False)
    except FileNotFoundError:
        return False
    except OSError:
        # Refusing to inspect an entry is different from it not existing.
        return True
    return True


def has_link_free_layout(model_directory: str | Path) -> bool:
    return is_materialized_model_file(link_free_layout_marker(model_directory))


def mark_link_free_layout(model_directory: str | Path) -> None:
    root = Path(model_directory)
    root.mkdir(parents=True, exist_ok=True)
    link_free_layout_marker(root).write_text("1\n", encoding="ascii")
