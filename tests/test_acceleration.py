import unittest
from pathlib import Path

from acceleration import (
    AccelerationMode,
    GpuDevice,
    NvidiaDevice,
    detect_acceleration,
)
from gpu_runtime import GPU_WORKER_PROTOCOL, GpuRuntimeManifest


RUNTIME = GpuRuntimeManifest(
    runtime_version="test",
    torch_version="2.7.1+cu128",
    cuda_runtime="12.8",
    worker_path=Path("ToonOutGpuWorker.exe"),
    files=(),
    worker_protocol=GPU_WORKER_PROTOCOL,
)
DEVICE = NvidiaDevice("NVIDIA Test GPU", "999.1", 16.0)
AMD_DEVICE = GpuDevice("AMD Radeon Test GPU", "1.2.3", 16.0, "amd")
AMD_RUNTIME = GpuRuntimeManifest(
    runtime_version="test",
    torch_version="2.10.0+rocm7.14.0",
    cuda_runtime="",
    worker_path=Path("ToonOutGpuWorker.exe"),
    files=(),
    worker_protocol=GPU_WORKER_PROTOCOL,
    vendor="amd",
    backend="rocm",
    compute_runtime="7.14.0",
)


class AccelerationTests(unittest.TestCase):
    def test_nvidia_without_pack_offers_installation(self):
        info = detect_acceleration(
            None,
            False,
            nvidia_probe=lambda: DEVICE,
        )
        self.assertEqual(info.mode, AccelerationMode.GPU_PACK_AVAILABLE)

    def test_installed_pack_can_be_enabled_or_disabled(self):
        active = detect_acceleration(
            RUNTIME,
            True,
            nvidia_probe=lambda: DEVICE,
        )
        disabled = detect_acceleration(
            RUNTIME,
            False,
            nvidia_probe=lambda: DEVICE,
        )
        self.assertEqual(active.mode, AccelerationMode.GPU_ACTIVE)
        self.assertEqual(disabled.mode, AccelerationMode.GPU_PACK_INSTALLED)

    def test_enabled_pack_without_nvidia_warns_and_cpu_only_needs_no_pack(self):
        unavailable = detect_acceleration(
            RUNTIME,
            True,
            nvidia_probe=lambda: None,
        )
        cpu = detect_acceleration(
            None,
            False,
            nvidia_probe=lambda: None,
        )
        self.assertEqual(unavailable.mode, AccelerationMode.GPU_UNAVAILABLE)
        self.assertEqual(cpu.mode, AccelerationMode.CPU_ONLY)

    def test_amd_runtime_selects_matching_amd_device(self):
        info = detect_acceleration(
            AMD_RUNTIME,
            True,
            nvidia_probe=lambda: DEVICE,
            amd_probe=lambda: AMD_DEVICE,
        )
        self.assertEqual(info.mode, AccelerationMode.GPU_ACTIVE)
        self.assertEqual(info.device, AMD_DEVICE)
        self.assertEqual(info.runtime.runtime_label, "ROCm 7.14.0")

    def test_amd_runtime_does_not_activate_on_nvidia_device(self):
        info = detect_acceleration(
            AMD_RUNTIME,
            True,
            nvidia_probe=lambda: DEVICE,
            amd_probe=lambda: None,
        )
        self.assertEqual(info.mode, AccelerationMode.GPU_UNAVAILABLE)
        self.assertIsNone(info.device)


if __name__ == "__main__":
    unittest.main()
