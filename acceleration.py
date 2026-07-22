"""지원 GPU와 선택형 가속 팩 상태를 감지한다."""

import json
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
class GpuDevice:
    name: str
    driver_version: str | None = None
    memory_gib: float | None = None
    vendor: str = "nvidia"

    @property
    def vendor_label(self) -> str:
        return {"amd": "AMD", "nvidia": "NVIDIA"}.get(
            self.vendor.lower(), self.vendor.upper()
        )


# Source compatibility for integrations importing the old public name.
NvidiaDevice = GpuDevice


@dataclass(frozen=True)
class AccelerationInfo:
    mode: AccelerationMode
    device: GpuDevice | None = None
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


def detect_nvidia_device() -> GpuDevice | None:
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
    return GpuDevice(
        name=parts[0],
        vendor="nvidia",
        driver_version=parts[1] if len(parts) >= 2 else None,
        memory_gib=memory_gib,
    )


def detect_amd_device() -> GpuDevice | None:
    """Windows video controller inventory에서 첫 AMD GPU를 찾는다."""

    if sys.platform != "win32":
        return None
    powershell = shutil.which("powershell.exe") or shutil.which("powershell")
    if powershell is None:
        return None
    script = (
        "Get-CimInstance Win32_VideoController | "
        "Select-Object Name,DriverVersion,AdapterRAM,PNPDeviceID | "
        "ConvertTo-Json -Compress"
    )
    try:
        result = subprocess.run(
            [powershell, "-NoProfile", "-NonInteractive", "-Command", script],
            capture_output=True,
            text=True,
            timeout=8,
            check=False,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        payload = json.loads(result.stdout) if result.returncode == 0 else []
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError):
        return None
    controllers = payload if isinstance(payload, list) else [payload]
    for controller in controllers:
        if not isinstance(controller, dict):
            continue
        name = str(controller.get("Name") or "").strip()
        pnp_id = str(controller.get("PNPDeviceID") or "").upper()
        if "AMD" not in name.upper() and "RADEON" not in name.upper() and "VEN_1002" not in pnp_id:
            continue
        memory_gib = None
        try:
            memory_bytes = int(controller.get("AdapterRAM") or 0)
            # Win32_VideoController.AdapterRAM is a 32-bit field and saturates
            # on modern cards; omit that misleading value instead of showing 4 GB.
            if 0 < memory_bytes < 4_000_000_000:
                memory_gib = round(memory_bytes / 1024**3, 1)
        except (TypeError, ValueError):
            pass
        return GpuDevice(
            name=name or "AMD Radeon GPU",
            vendor="amd",
            driver_version=str(controller.get("DriverVersion") or "") or None,
            memory_gib=memory_gib,
        )
    return None


def detect_acceleration(
    runtime: GpuRuntimeManifest | None,
    gpu_enabled: bool,
    *,
    runtime_size: int = 0,
    runtime_directory: str | None = None,
    nvidia_probe: Callable[[], GpuDevice | None] = detect_nvidia_device,
    amd_probe: Callable[[], GpuDevice | None] | None = None,
) -> AccelerationInfo:
    amd_device = (
        amd_probe()
        if amd_probe is not None
        else detect_amd_device() if nvidia_probe is detect_nvidia_device else None
    )
    devices = tuple(
        device for device in (nvidia_probe(), amd_device) if device is not None
    )
    runtime_vendor = getattr(runtime, "vendor", None)
    if runtime is not None and runtime_vendor:
        device = next(
            (item for item in devices if item.vendor == runtime_vendor),
            None,
        )
    else:
        device = devices[0] if devices else None
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
