import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from update_service import UpdateError, UpdateRelease
from update_workers import UpdateCheckThread, UpdateDownloadThread


def release(version: str = "0.2.0") -> UpdateRelease:
    return UpdateRelease(
        version=version,
        minimum_supported_version="0.1.0",
        installer_url="https://github.com/example/setup.exe",
        installer_name=f"ToonOut-Setup-{version}.exe",
        installer_size=4,
        installer_sha256="0" * 64,
        release_notes_url="https://github.com/example/release",
        gpu_worker_protocol=2,
    )


class UpdateWorkerTests(unittest.TestCase):
    def test_check_thread_reports_only_newer_release(self):
        thread = UpdateCheckThread("https://example/update.json", "key", "0.1.0")
        available = []
        thread.available.connect(available.append)

        with patch("update_workers.fetch_update_release", return_value=release()):
            thread.run()

        self.assertEqual(available[0].version, "0.2.0")

    def test_check_thread_cleans_installed_update_cache_before_network(self):
        with tempfile.TemporaryDirectory() as directory:
            old_update = Path(directory) / "0.1.6"
            old_update.mkdir()
            (old_update / "setup.exe").write_bytes(b"old installer")
            thread = UpdateCheckThread(
                "https://example/update.json",
                "key",
                "0.1.6",
                directory,
            )

            with patch(
                "update_workers.fetch_update_release",
                return_value=release("0.1.6"),
            ):
                thread.run()

            self.assertFalse(old_update.exists())

    def test_check_thread_maps_safe_failure_text(self):
        thread = UpdateCheckThread("https://example/update.json", "key", "0.1.0")
        failures = []
        thread.failed.connect(failures.append)

        with patch(
            "update_workers.fetch_update_release",
            side_effect=UpdateError("업데이트 서버에 연결할 수 없습니다"),
        ):
            thread.run()

        self.assertEqual(failures, ["업데이트 서버에 연결할 수 없습니다"])

    def test_download_thread_reports_verified_installer(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "0.2.0"
            destination.mkdir()
            installer = destination / "ToonOut-Setup-0.2.0.exe"
            installer.write_bytes(b"test")
            thread = UpdateDownloadThread(release(), destination)
            ready = []
            thread.ready.connect(ready.append)

            with patch(
                "update_workers.download_installer",
                return_value=installer,
            ):
                thread.run()

            self.assertEqual(ready, [str(installer)])


if __name__ == "__main__":
    unittest.main()
