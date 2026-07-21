import hashlib
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from main import MainWindow
from update_service import UpdateRelease


def release_for(content: bytes = b"installer") -> UpdateRelease:
    return UpdateRelease(
        version="0.2.0",
        minimum_supported_version="0.1.0",
        installer_url="https://github.com/example/setup.exe",
        installer_name="ToonOut-Setup-0.2.0.exe",
        installer_size=len(content),
        installer_sha256=hashlib.sha256(content).hexdigest(),
        release_notes_url="https://github.com/example/release",
        gpu_worker_protocol=2,
    )


class UpdateUiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_development_build_keeps_update_button_visible_but_disabled(self):
        with (
            patch.object(MainWindow, "_refresh_acceleration_status"),
            patch("main.UPDATE_PUBLIC_KEY_B64", ""),
        ):
            window = MainWindow()

            window.check_for_updates()

            self.assertFalse(window.update_status_button.isHidden())
            self.assertFalse(window.update_status_button.isEnabled())
            self.assertEqual(window.update_status_button.text(), "업데이트 확인")
            self.assertIsNone(window._update_check_thread)
            window.close()

    def test_up_to_date_result_restores_manual_check_button(self):
        with patch.object(MainWindow, "_refresh_acceleration_status"):
            window = MainWindow()
            window.update_status_button.hide()

            window._on_update_up_to_date()

            self.assertFalse(window.update_status_button.isHidden())
            self.assertTrue(window.update_status_button.isEnabled())
            self.assertEqual(window.update_status_button.text(), "업데이트 확인")
            self.assertIn("최신 버전", window.update_status_button.toolTip())
            window.close()

    def test_available_release_starts_automatic_download(self):
        with (
            patch.object(MainWindow, "_refresh_acceleration_status"),
            patch.object(MainWindow, "_start_update_download") as download,
        ):
            window = MainWindow()
            candidate = release_for()

            window._on_update_available(candidate)

            self.assertIs(window._update_release, candidate)
            download.assert_called_once_with()
            window.close()

    def test_ready_update_waits_until_processing_finishes(self):
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(MainWindow, "_refresh_acceleration_status"),
        ):
            window = MainWindow()
            candidate = release_for()
            installer = Path(directory) / candidate.installer_name
            installer.write_bytes(b"installer")
            window._update_release = candidate
            window._update_installer_path = installer

            window._processing = True
            window._refresh_update_button()
            self.assertEqual(
                window.update_status_button.text(),
                "작업 후 업데이트",
            )
            self.assertFalse(window.update_status_button.isEnabled())

            window._processing = False
            window._refresh_update_button()
            self.assertEqual(
                window.update_status_button.text(),
                "업데이트 설치 · 0.2.0",
            )
            self.assertTrue(window.update_status_button.isEnabled())
            window.close()

    def test_one_click_install_reverifies_and_launches_silent_setup(self):
        content = b"installer"
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(MainWindow, "_refresh_acceleration_status"),
            patch("main.QProcess.startDetached", return_value=(True, 42)) as start,
            patch("main.QApplication.instance", return_value=None),
        ):
            window = MainWindow()
            candidate = release_for(content)
            installer = Path(directory) / candidate.installer_name
            installer.write_bytes(content)
            window._update_release = candidate
            window._update_installer_path = installer

            window.install_ready_update()

            arguments = start.call_args.args[1]
            self.assertIn("/SILENT", arguments)
            self.assertIn("/CLOSEAPPLICATIONS", arguments)
            self.assertEqual(start.call_args.args[0], str(installer))
            self.assertEqual(
                window.update_status_button.text(),
                "업데이트 시작 중",
            )
            window.close()


if __name__ == "__main__":
    unittest.main()
