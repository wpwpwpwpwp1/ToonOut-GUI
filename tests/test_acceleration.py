import unittest
from pathlib import Path

from acceleration import (
    AccelerationMode,
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


if __name__ == "__main__":
    unittest.main()
