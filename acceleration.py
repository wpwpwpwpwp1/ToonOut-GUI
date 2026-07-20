"""NVIDIA 장치와 선택형 GPU 가속 팩 상태를 감지한다."""

import shutil
import subprocess
import sys
from dataclasses import dataclass
from enum import Enum
from typing import Callable

from gpu_runtime import GpuRuntimeManifest, gpu_runtime_is_compatible


class AccelerationMode(str, Enum):
    GPU_ACTIVE = "gpu_active"
    GPU_PACK_INSTALLED = "gpu_pack_installed"
    GPU_PACK_AVAILABLE = "gpu_pack_available"
    GPU_UNAVAILABLE = "gpu_unavailable"
    CPU_ONLY = "cpu_only"


@dataclass(frozen=True)
class NvidiaDevice:
    name: str
    driver_version: str | None = None
    memory_gib: float | None = None


@dataclass(frozen=True)
class AccelerationInfo:
    mode: AccelerationMode
    device: NvidiaDevice | None = None
    runtime: GpuRuntimeManifest | None = None
    runtime_size: int = 0
    runtime_directory: str | None = None

    @property
    def status_text(self) -> str:
        if self.runtime is not None and not gpu_runtime_is_compatible(
            self.runtime
        ):
            return "! GPU 팩 업데이트 필요"
        return {
            AccelerationMode.GPU_ACTIVE: "● GPU 가속 선택됨",
            AccelerationMode.GPU_PACK_INSTALLED: "GPU 가속 꺼짐",
            AccelerationMode.GPU_PACK_AVAILABLE: "◆ GPU 가속 설치 가능",
            AccelerationMode.GPU_UNAVAILABLE: "! GPU 가속 확인 필요",
            AccelerationMode.CPU_ONLY: "CPU 처리",
        }[self.mode]


def detect_nvidia_device() -> NvidiaDevice | None:
    executable = shutil.which("nvidia-smi")
    if executable is None:
        return None
    creation_flags = (
        getattr(subprocess, "CREATE_NO_WINDOW", 0)
        if sys.platform == "win32"
        else 0
    )
    try:
        result = subprocess.run(
            [
                executable,
                "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
            creationflags=creation_flags,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0 or not result.stdout.strip():
        return None
    parts = [part.strip() for part in result.stdout.splitlines()[0].split(",")]
    if not parts:
        return None
    memory_gib = None
    if len(parts) >= 3:
        try:
            memory_gib = round(float(parts[2]) / 1024, 1)
        except ValueError:
            pass
    return NvidiaDevice(
        name=parts[0],
        driver_version=parts[1] if len(parts) >= 2 else None,
        memory_gib=memory_gib,
    )


def detect_acceleration(
    runtime: GpuRuntimeManifest | None,
    gpu_enabled: bool,
    *,
    runtime_size: int = 0,
    runtime_directory: str | None = None,
    nvidia_probe: Callable[[], NvidiaDevice | None] = detect_nvidia_device,
) -> AccelerationInfo:
    device = nvidia_probe()
    if runtime is not None and not gpu_runtime_is_compatible(runtime):
        gpu_enabled = False
    if runtime is not None and gpu_enabled and device is not None:
        mode = AccelerationMode.GPU_ACTIVE
    elif runtime is not None and gpu_enabled:
        mode = AccelerationMode.GPU_UNAVAILABLE
    elif runtime is not None:
        mode = AccelerationMode.GPU_PACK_INSTALLED
    elif device is not None:
        mode = AccelerationMode.GPU_PACK_AVAILABLE
    else:
        mode = AccelerationMode.CPU_ONLY
    return AccelerationInfo(
        mode=mode,
        device=device,
        runtime=runtime,
        runtime_size=runtime_size,
        runtime_directory=runtime_directory,
    )
