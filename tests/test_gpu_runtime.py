import hashlib
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from gpu_runtime import (
    GPU_WORKER_PROTOCOL,
    GPU_RUNTIME_KIND,
    GPU_RUNTIME_SCHEMA,
    GpuRuntimeCancelled,
    GpuRuntimeError,
    delete_gpu_runtime,
    find_adjacent_gpu_pack,
    gpu_runtime_is_installed,
    gpu_runtime_size,
    install_gpu_runtime,
    load_gpu_runtime_manifest,
)
from scripts.split_gpu_pack import split_pack


def write_fake_pack(
    path: Path,
    worker_content: bytes = b"fake worker",
    worker_protocol: int = GPU_WORKER_PROTOCOL,
):
    worker_hash = hashlib.sha256(worker_content).hexdigest()
    manifest = {
        "schema_version": GPU_RUNTIME_SCHEMA,
        "kind": GPU_RUNTIME_KIND,
        "runtime_version": "test-1",
        "worker_protocol": worker_protocol,
        "torch_version": "2.7.1+cu128",
        "cuda_runtime": "12.8",
        "worker": "ToonOutGpuWorker.exe",
        "files": [
            {
                "path": "ToonOutGpuWorker.exe",
                "size": len(worker_content),
                "sha256": worker_hash,
            }
        ],
    }
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("manifest.json", json.dumps(manifest))
        archive.writestr("ToonOutGpuWorker.exe", worker_content)


class GpuRuntimeTests(unittest.TestCase):
    def test_old_worker_protocol_is_rejected_before_installation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pack = root / "old-pack.zip"
            destination = root / "installed"
            write_fake_pack(pack, worker_protocol=1)

            with self.assertRaises(GpuRuntimeError):
                install_gpu_runtime(pack, destination)

            self.assertFalse(destination.exists())

    def test_adjacent_pack_is_found_for_same_app_installation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pack = root / "ToonOut-NVIDIA-GPU-Pack.zip"
            pack.touch()

            self.assertEqual(find_adjacent_gpu_pack(root), pack.resolve())

    def test_pack_is_verified_installed_and_removed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pack = root / "ToonOut-NVIDIA-GPU-Pack.zip"
            destination = root / "runtime"
            write_fake_pack(pack)

            manifest = install_gpu_runtime(pack, destination)

            self.assertEqual(manifest.cuda_runtime, "12.8")
            self.assertTrue(gpu_runtime_is_installed(destination))
            self.assertEqual(
                load_gpu_runtime_manifest(destination).torch_version,
                "2.7.1+cu128",
            )

            delete_gpu_runtime(destination)
            self.assertFalse(destination.exists())

    def test_split_pack_is_verified_and_installed_without_reassembly(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pack = root / "ToonOut-NVIDIA-GPU-Pack.zip"
            destination = root / "runtime"
            write_fake_pack(pack)
            parts_manifest = split_pack(pack, part_size=40)
            pack.unlink()

            manifest = install_gpu_runtime(parts_manifest, destination)

            self.assertEqual(manifest.worker_protocol, GPU_WORKER_PROTOCOL)
            self.assertTrue(gpu_runtime_is_installed(destination))
            logical_size = sum(
                path.stat().st_size
                for path in destination.rglob("*")
                if path.is_file()
            )
            self.assertGreater(gpu_runtime_size(destination), 0)
            self.assertLessEqual(gpu_runtime_size(destination), logical_size)

    def test_split_pack_can_remove_redundant_source_archive(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pack = root / "ToonOut-NVIDIA-GPU-Pack.zip"
            checksum = Path(f"{pack}.sha256")
            write_fake_pack(pack)
            checksum.write_text("generated checksum", encoding="ascii")

            parts_manifest = split_pack(
                pack,
                part_size=40,
                remove_input=True,
            )

            self.assertTrue(parts_manifest.is_file())
            self.assertFalse(pack.exists())
            self.assertFalse(checksum.exists())

    def test_missing_split_pack_part_is_actionable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pack = root / "ToonOut-NVIDIA-GPU-Pack.zip"
            write_fake_pack(pack)
            parts_manifest = split_pack(pack, part_size=40)
            (root / "ToonOut-NVIDIA-GPU-Pack.part01").unlink()

            with self.assertRaisesRegex(GpuRuntimeError, "조각 1"):
                install_gpu_runtime(parts_manifest, root / "runtime")

    def test_cancelled_install_removes_staging_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pack = root / "pack.zip"
            destination = root / "runtime"
            write_fake_pack(pack)
            checks = 0

            def should_cancel():
                nonlocal checks
                checks += 1
                return checks >= 3

            with self.assertRaises(GpuRuntimeCancelled):
                install_gpu_runtime(
                    pack,
                    destination,
                    should_cancel=should_cancel,
                )

            self.assertFalse(destination.exists())
            self.assertEqual(
                list(root.glob(".nvidia-gpu-install-*")),
                [],
            )

    def test_modified_worker_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pack = root / "pack.zip"
            destination = root / "runtime"
            write_fake_pack(pack)
            install_gpu_runtime(pack, destination)
            (destination / "ToonOutGpuWorker.exe").write_bytes(b"changed")

            with self.assertRaises(GpuRuntimeError):
                load_gpu_runtime_manifest(destination)

    def test_path_traversal_archive_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pack = root / "unsafe.zip"
            with zipfile.ZipFile(pack, "w") as archive:
                archive.writestr("../outside.txt", "unsafe")

            with self.assertRaises(GpuRuntimeError):
                install_gpu_runtime(pack, root / "runtime")

            self.assertFalse((root / "outside.txt").exists())


if __name__ == "__main__":
    unittest.main()
