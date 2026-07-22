import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from gpu_download import (
    GPU_DOWNLOAD_PROGRESS_END,
    GPU_PACK_RELEASE_TAG,
    GpuPackRelease,
    current_gpu_pack_release,
    download_and_install_gpu_runtime,
    download_gpu_pack,
)
from gpu_runtime import GpuRuntimeCancelled, GpuRuntimeError
from scripts.update_gpu_pack_config import update_config


class FakeResponse:
    def __init__(self, content: bytes):
        self._content = content
        self._offset = 0

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def read(self, size: int = -1) -> bytes:
        if size < 0:
            size = len(self._content) - self._offset
        chunk = self._content[self._offset : self._offset + size]
        self._offset += len(chunk)
        return chunk


def split_release(part_contents: list[bytes]):
    records = []
    for index, content in enumerate(part_contents, start=1):
        records.append(
            {
                "filename": f"ToonOut-NVIDIA-GPU-Pack.part{index:02d}",
                "size": len(content),
                "sha256": hashlib.sha256(content).hexdigest(),
            }
        )
    archive = b"".join(part_contents)
    document = json.dumps(
        {
            "schema_version": 1,
            "kind": "toonout-nvidia-gpu-pack-parts",
            "archive_filename": "ToonOut-NVIDIA-GPU-Pack.zip",
            "archive_size": len(archive),
            "archive_sha256": hashlib.sha256(archive).hexdigest(),
            "parts": records,
        }
    ).encode("utf-8")
    release = GpuPackRelease(
        release_tag="v0.2.0",
        manifest_url=(
            "https://github.com/example/ToonOut/releases/download/v0.2.0/"
            "ToonOut-NVIDIA-GPU-Pack.parts.json"
        ),
        manifest_name="ToonOut-NVIDIA-GPU-Pack.parts.json",
        manifest_size=len(document),
        manifest_sha256=hashlib.sha256(document).hexdigest(),
        archive_size=len(archive),
        worker_protocol=2,
    )
    return release, document


class GpuPackDownloadTests(unittest.TestCase):
    def test_current_release_uses_pinned_gpu_pack_tag(self):
        release = current_gpu_pack_release(
            system="windows",
            machine="AMD64",
        )

        self.assertEqual(release.release_tag, GPU_PACK_RELEASE_TAG)
        self.assertIn(f"/{GPU_PACK_RELEASE_TAG}/", release.manifest_url)
        self.assertEqual(release.worker_protocol, 2)

    def test_unsupported_architecture_is_rejected_before_network(self):
        with self.assertRaisesRegex(GpuRuntimeError, "아키텍처"):
            current_gpu_pack_release(
                system="windows",
                machine="ARM64",
            )

    def test_manifest_and_all_parts_are_downloaded_and_verified(self):
        parts = [b"first verified part", b"second verified part"]
        release, document = split_release(parts)
        progress = []
        responses = [FakeResponse(document), *(FakeResponse(part) for part in parts)]

        with (
            tempfile.TemporaryDirectory() as directory,
            patch("gpu_download.urlopen", side_effect=responses),
        ):
            manifest = download_gpu_pack(
                release,
                directory,
                report_progress=lambda status, percent: progress.append(
                    (status, percent)
                ),
            )

            self.assertEqual(manifest.read_bytes(), document)
            for index, content in enumerate(parts, start=1):
                self.assertEqual(
                    (
                        Path(directory)
                        / f"ToonOut-NVIDIA-GPU-Pack.part{index:02d}"
                    ).read_bytes(),
                    content,
                )
            self.assertEqual(progress[-1][1], GPU_DOWNLOAD_PROGRESS_END)
            self.assertIn("다운로드 완료", progress[-1][0])

    def test_corrupt_part_is_removed_with_partial_file(self):
        expected_parts = [b"expected part"]
        release, document = split_release(expected_parts)

        with (
            tempfile.TemporaryDirectory() as directory,
            patch(
                "gpu_download.urlopen",
                side_effect=[FakeResponse(document), FakeResponse(b"corrupt part!")],
            ),
        ):
            with self.assertRaises(GpuRuntimeError):
                download_gpu_pack(release, directory)

            files = {path.name for path in Path(directory).iterdir()}
            self.assertEqual(files, {release.manifest_name})

    def test_cancelled_install_removes_download_directory(self):
        release, _document = split_release([b"part"])
        with (
            tempfile.TemporaryDirectory() as directory,
            patch("gpu_download.shutil.disk_usage") as disk_usage,
            patch(
                "gpu_download.download_gpu_pack",
                side_effect=GpuRuntimeCancelled("cancelled"),
            ),
        ):
            disk_usage.return_value.free = 20_000_000_000
            runtime = Path(directory) / "nvidia-gpu"

            with self.assertRaises(GpuRuntimeCancelled):
                download_and_install_gpu_runtime(release, runtime)

            self.assertEqual(
                list(Path(directory).glob(".nvidia-gpu-download-*")),
                [],
            )

    def test_low_disk_space_stops_before_download(self):
        release, _document = split_release([b"part"])
        with (
            tempfile.TemporaryDirectory() as directory,
            patch("gpu_download.shutil.disk_usage") as disk_usage,
            patch("gpu_download.download_gpu_pack") as download,
        ):
            disk_usage.return_value.free = release.archive_size - 1

            with self.assertRaisesRegex(GpuRuntimeError, "다운로드 공간"):
                download_and_install_gpu_runtime(
                    release,
                    Path(directory) / "nvidia-gpu",
                )

            download.assert_not_called()

    def test_combined_progress_maps_download_and_install_to_one_bar(self):
        release, _document = split_release([b"part"])
        progress = []
        installed_manifest = object()

        def fake_install(_pack, _runtime, **kwargs):
            kwargs["report_progress"]("분할 GPU 가속 팩 검증 중", 0)
            kwargs["report_progress"]("GPU 가속 팩 설치 완료", 100)
            return installed_manifest

        with (
            tempfile.TemporaryDirectory() as directory,
            patch("gpu_download.shutil.disk_usage") as disk_usage,
            patch("gpu_download.download_gpu_pack") as download,
            patch("gpu_download.install_gpu_runtime", side_effect=fake_install),
        ):
            disk_usage.return_value.free = 20_000_000_000

            def fake_download(_release, destination, **kwargs):
                manifest = Path(destination) / release.manifest_name
                manifest.write_text("fixture", encoding="utf-8")
                kwargs["report_progress"](
                    "GPU 가속 팩 다운로드 완료",
                    GPU_DOWNLOAD_PROGRESS_END,
                )
                return manifest

            download.side_effect = fake_download
            result = download_and_install_gpu_runtime(
                release,
                Path(directory) / "nvidia-gpu",
                report_progress=lambda status, percent: progress.append(
                    (status, percent)
                ),
            )

            self.assertIs(result, installed_manifest)
            self.assertEqual([percent for _status, percent in progress], [40, 40, 100])
            self.assertEqual(
                list(Path(directory).glob(".nvidia-gpu-download-*")),
                [],
            )

    def test_pack_build_metadata_updater_records_exact_manifest_bytes(self):
        release, document = split_release([b"part one", b"part two"])
        config_source = (
            "GPU_PACK_MANIFEST_SIZE = 1\n"
            "GPU_PACK_MANIFEST_SHA256 = (\n"
            f'    "{"0" * 64}"\n'
            ")\n"
            "GPU_PACK_ARCHIVE_SIZE = 1\n"
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / release.manifest_name
            config = root / "gpu_download.py"
            manifest.write_bytes(document)
            config.write_text(config_source, encoding="utf-8")

            update_config(manifest, config)

            updated = config.read_text(encoding="utf-8")
            self.assertIn(f"GPU_PACK_MANIFEST_SIZE = {len(document)}", updated)
            self.assertIn(hashlib.sha256(document).hexdigest(), updated)
            self.assertIn(
                f"GPU_PACK_ARCHIVE_SIZE = {release.archive_size:_}",
                updated,
            )


if __name__ == "__main__":
    unittest.main()
