import hashlib
import json
import os
import tempfile
import unittest
import zipfile
from pathlib import Path

from gpu_runtime import (
    GPU_WORKER_PROTOCOL,
    GPU_RUNTIME_KIND,
    GENERIC_GPU_RUNTIME_KIND,
    GPU_RUNTIME_SCHEMA,
    GpuRuntimeCancelled,
    GpuRuntimeError,
    cleanup_gpu_runtime_artifacts,
    delete_gpu_runtime,
    find_adjacent_gpu_pack,
    gpu_runtime_is_installed,
    gpu_runtime_size,
    install_gpu_runtime,
    load_gpu_runtime_manifest,
    move_gpu_runtime,
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
    def test_finds_adjacent_amd_rocm_pack(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pack = root / "ToonOut-AMD-ROCm-GPU-Pack.parts.json"
            pack.write_text("{}", encoding="utf-8")

            self.assertEqual(find_adjacent_gpu_pack(root), pack.resolve())

    def test_loads_generic_amd_rocm_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            worker = root / "ToonOutGpuWorker.exe"
            worker.write_bytes(b"worker")
            (root / "manifest.json").write_text(
                json.dumps(
                    {
                        "schema_version": GPU_RUNTIME_SCHEMA,
                        "kind": GENERIC_GPU_RUNTIME_KIND,
                        "runtime_version": "amd-test",
                        "worker_protocol": GPU_WORKER_PROTOCOL,
                        "torch_version": "2.10.0+rocm7.14.0",
                        "vendor": "amd",
                        "backend": "rocm",
                        "compute_runtime": "7.14.0",
                        "worker": worker.name,
                        "files": [],
                    }
                ),
                encoding="utf-8",
            )

            manifest = load_gpu_runtime_manifest(root)

            self.assertEqual(manifest.vendor, "amd")
            self.assertEqual(manifest.backend, "rocm")
            self.assertEqual(manifest.runtime_label, "ROCm 7.14.0")

    def test_abandoned_runtime_artifacts_are_removed_without_touching_active_one(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runtime = root / "nvidia-gpu"
            abandoned = root / ".nvidia-gpu-install-2147483647-dead"
            active = root / f".nvidia-gpu-download-{os.getpid()}-active"
            abandoned.mkdir()
            active.mkdir()
            (abandoned / "large.part").write_bytes(b"unused")

            removed = cleanup_gpu_runtime_artifacts(runtime)

            self.assertEqual(removed, 1)
            self.assertFalse(abandoned.exists())
            self.assertTrue(active.exists())

    def test_failed_runtime_swap_restores_previous_installation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            old_pack = root / "old.zip"
            new_pack = root / "new.zip"
            destination = root / "runtime"
            write_fake_pack(old_pack, b"old worker")
            write_fake_pack(new_pack, b"new worker")
            install_gpu_runtime(old_pack, destination)

            from gpu_runtime import load_gpu_runtime_manifest as real_load
            destination_checks = 0

            def fail_after_swap(path, **kwargs):
                nonlocal destination_checks
                if (
                    Path(path) == destination
                    and kwargs.get("verify_files") is False
                ):
                    destination_checks += 1
                    if destination_checks >= 2:
                        raise GpuRuntimeError("applied runtime check failed")
                return real_load(path, **kwargs)

            from unittest.mock import patch

            with patch(
                "gpu_runtime.load_gpu_runtime_manifest",
                side_effect=fail_after_swap,
            ):
                with self.assertRaisesRegex(GpuRuntimeError, "applied runtime"):
                    install_gpu_runtime(new_pack, destination)

            self.assertEqual(
                (destination / "ToonOutGpuWorker.exe").read_bytes(),
                b"old worker",
            )
            self.assertEqual(list(root.glob(".runtime-previous-*")), [])

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

    def test_install_does_not_replace_unrelated_custom_folder_contents(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pack = root / "pack.zip"
            destination = root / "custom-runtime"
            destination.mkdir()
            unrelated = destination / "keep.txt"
            unrelated.write_text("keep", encoding="utf-8")
            write_fake_pack(pack)

            with self.assertRaisesRegex(GpuRuntimeError, "다른 파일"):
                install_gpu_runtime(pack, destination)

            self.assertEqual(unrelated.read_text(encoding="utf-8"), "keep")

    def test_installed_pack_moves_to_an_empty_selected_folder(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pack = root / "pack.zip"
            source = root / "source-runtime"
            destination = root / "destination-runtime"
            destination.mkdir()
            write_fake_pack(pack)
            install_gpu_runtime(pack, source)

            source_removed = move_gpu_runtime(source, destination)

            self.assertTrue(source_removed)
            self.assertFalse(source.exists())
            self.assertTrue(gpu_runtime_is_installed(destination))
            self.assertEqual(
                (destination / "ToonOutGpuWorker.exe").read_bytes(),
                b"fake worker",
            )

    def test_pack_move_rejects_nonempty_destination_without_touching_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pack = root / "pack.zip"
            source = root / "source-runtime"
            destination = root / "destination-runtime"
            destination.mkdir()
            unrelated = destination / "keep.txt"
            unrelated.write_text("keep", encoding="utf-8")
            write_fake_pack(pack)
            install_gpu_runtime(pack, source)

            with self.assertRaisesRegex(GpuRuntimeError, "비어 있지"):
                move_gpu_runtime(source, destination)

            self.assertTrue(gpu_runtime_is_installed(source))
            self.assertEqual(unrelated.read_text(encoding="utf-8"), "keep")

    def test_cross_drive_pack_move_copies_verifies_then_removes_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pack = root / "pack.zip"
            source = root / "source-runtime"
            destination = root / "destination-runtime"
            destination.mkdir()
            write_fake_pack(pack)
            install_gpu_runtime(pack, source)
            real_replace = Path.replace

            def replace_with_cross_drive_failure(path, target):
                if Path(path) == source:
                    raise OSError("simulated cross-drive move")
                return real_replace(path, target)

            from unittest.mock import patch

            with patch.object(Path, "replace", replace_with_cross_drive_failure):
                source_removed = move_gpu_runtime(source, destination)

            self.assertTrue(source_removed)
            self.assertFalse(source.exists())
            self.assertTrue(gpu_runtime_is_installed(destination))

    def test_install_reports_monotonic_percent_progress(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pack = root / "ToonOut-NVIDIA-GPU-Pack.zip"
            destination = root / "runtime"
            write_fake_pack(pack)
            events = []

            install_gpu_runtime(
                pack,
                destination,
                report_progress=lambda status, percent: events.append(
                    (status, percent)
                ),
            )

            percentages = [percent for _status, percent in events]
            self.assertEqual(percentages[0], 0)
            self.assertEqual(percentages[-1], 100)
            self.assertEqual(percentages, sorted(percentages))
            self.assertIn("설치 완료", events[-1][0])

    def test_split_pack_is_verified_and_installed_without_reassembly(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pack = root / "ToonOut-NVIDIA-GPU-Pack.zip"
            destination = root / "runtime"
            write_fake_pack(pack)
            parts_manifest = split_pack(pack, part_size=40)
            pack.unlink()
            events = []

            manifest = install_gpu_runtime(
                parts_manifest,
                destination,
                report_progress=lambda status, percent: events.append(
                    (status, percent)
                ),
            )

            self.assertEqual(manifest.worker_protocol, GPU_WORKER_PROTOCOL)
            self.assertTrue(gpu_runtime_is_installed(destination))
            self.assertEqual(events[-1][1], 100)
            self.assertTrue(
                any("분할 GPU" in status and percent > 5 for status, percent in events)
            )
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
