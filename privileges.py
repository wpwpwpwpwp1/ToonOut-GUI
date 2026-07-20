"""Windows에서 필요한 경우에만 앱을 UAC 관리자 권한으로 다시 실행한다."""

import ctypes
import subprocess
import sys
from pathlib import Path


def is_running_as_admin() -> bool:
    if sys.platform != "win32":
        return False
    try:
        return bool(ctypes.windll.shell32.IsUserAnAdmin())
    except OSError:
        return False


def elevated_launch_arguments(
    model_directory: str | Path,
    image_paths: list[str] | None = None,
) -> tuple[str, str, str]:
    model_args = [
        "--model-directory",
        str(model_directory),
        "--open-model-installer",
    ]
    for image_path in image_paths or []:
        model_args.extend(["--restore-image", image_path])
    if getattr(sys, "frozen", False):
        executable = sys.executable
        arguments = model_args
    else:
        executable = sys.executable
        arguments = [str(Path(__file__).with_name("main.py").resolve()), *model_args]
    return executable, subprocess.list2cmdline(arguments), str(Path.cwd())


def request_elevated_restart(
    model_directory: str | Path,
    image_paths: list[str] | None = None,
) -> bool:
    if sys.platform != "win32":
        return False
    executable, parameters, working_directory = elevated_launch_arguments(
        model_directory,
        image_paths,
    )
    result = ctypes.windll.shell32.ShellExecuteW(
        None,
        "runas",
        executable,
        parameters,
        working_directory,
        1,
    )
    return result > 32
