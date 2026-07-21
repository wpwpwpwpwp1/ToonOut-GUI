import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import (
    QEvent,
    QEventLoop,
    QMimeData,
    QPoint,
    QPointF,
    QSize,
    Qt,
    QTimer,
    QUrl,
)
from PySide6.QtGui import QImage
from PySide6.QtTest import QTest
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QFileDialog,
    QLabel,
    QListView,
    QListWidgetItem,
    QPushButton,
)

from acceleration import AccelerationInfo, AccelerationMode, NvidiaDevice
from app_core import BatchItem, ItemState, OutputNamingPolicy
from main import BRAND_MASCOT_SIZE, MainWindow
from mascot import TsunaoState, TsunaoWidget
from processing import ModelInstallProcess
from widgets import (
    AccelerationDialog,
    ModelInstallDialog,
    ModelManagementDialog,
    ModelOperationDialog,
    OutputNamingDialog,
    ToonOutComboBox,
    ImageListWidget,
)


class UiSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_main_window_shows_model_installation_status(self):
        with (
            patch("main.model_is_installed", return_value=False),
            patch.object(MainWindow, "_refresh_acceleration_status"),
        ):
            window = MainWindow()

            self.assertEqual(
                window.model_status_button.text(),
                "● 모델 설치 필요",
            )

            window.close()

    def test_app_opens_directly_to_empty_workspace(self):
        with patch.object(MainWindow, "_refresh_acceleration_status"):
            window = MainWindow()

            self.assertEqual(window.file_list.count(), 0)
            self.assertFalse(window.workspace_page.isHidden())
            self.assertFalse(window.action_bar.isHidden())
            self.assertFalse(window.header_add_button.isHidden())
            self.assertIn("창 어디에나", window.status_label.text())
            self.assertEqual(window.tsunao.state, TsunaoState.SLEEPING)
            self.assertEqual(
                window.tsunao.accessibleName(),
                "잠든 츠나오",
            )
            self.assertIsNone(window._output_directory)
            self.assertEqual(
                window.output_folder_button.text(),
                "저장 위치 지정",
            )
            self.assertTrue(
                window.output_naming_button.text().startswith("이름 형식")
            )
            self.assertEqual(
                window.tsunao_hide_checkbox.text(),
                "츠나오 끄기",
            )
            self.assertFalse(window.tsunao_hide_checkbox.isChecked())
            self.assertFalse(hasattr(window, "empty_page"))
            window.close()

    def test_mascot_toggle_hides_and_restores_only_interactive_tsunao(self):
        with patch.object(MainWindow, "_refresh_acceleration_status"):
            window = MainWindow()
            window.show()
            self.app.processEvents()

            self.assertFalse(window.tsunao.isHidden())
            self.assertFalse(window.brand_mascot.isHidden())

            window.tsunao_hide_checkbox.setChecked(True)
            self.app.processEvents()
            self.assertTrue(window.tsunao.isHidden())
            self.assertFalse(window.brand_mascot.isHidden())

            window.tsunao_hide_checkbox.setChecked(False)
            self.app.processEvents()
            self.assertFalse(window.tsunao.isHidden())
            window.close()

    def test_mascot_wakes_only_when_an_image_is_loaded_while_sleeping(self):
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(MainWindow, "_refresh_acceleration_status"),
        ):
            first = Path(directory) / "first.png"
            second = Path(directory) / "second.png"
            for path in (first, second):
                image = QImage(24, 24, QImage.Format.Format_RGB32)
                image.fill(0xFFE9EAFF)
                self.assertTrue(image.save(str(path)))

            window = MainWindow()
            window.add_paths([str(first)])
            self.assertEqual(window.tsunao.state, TsunaoState.AWAKE)

            window.tsunao.advance_time(5_000)
            self.assertEqual(window.tsunao.state, TsunaoState.STANDING)

            window.add_paths([str(second)])
            self.assertEqual(window.tsunao.state, TsunaoState.STANDING)
            window.close()

    def test_mascot_can_be_picked_through_the_window_event_filter(self):
        with patch.object(MainWindow, "_refresh_acceleration_status"):
            window = MainWindow()
            window.show()
            self.app.processEvents()
            mascot_center = window.tsunao.position.toPoint()
            self.assertTrue(window.tsunao.hit_test(QPointF(mascot_center)))

            QTest.mousePress(
                window.centralWidget(),
                Qt.MouseButton.LeftButton,
                Qt.KeyboardModifier.NoModifier,
                mascot_center,
            )
            self.app.processEvents()
            self.assertEqual(window.tsunao.state, TsunaoState.PICKED)

            QTest.mouseRelease(
                window.centralWidget(),
                Qt.MouseButton.LeftButton,
                Qt.KeyboardModifier.NoModifier,
                mascot_center,
            )
            self.app.processEvents()
            self.assertEqual(window.tsunao.state, TsunaoState.ANGRY)
            window.close()

    def test_queue_thumbnail_size_follows_splitter_width(self):
        with patch.object(MainWindow, "_refresh_acceleration_status"):
            window = MainWindow()

            self.assertEqual(window.size().width(), 1560)
            self.assertEqual(window.size().height(), 960)
            self.assertEqual(window.brand_mascot.width(), BRAND_MASCOT_SIZE)
            self.assertEqual(window.brand_mascot.height(), BRAND_MASCOT_SIZE)
            self.assertEqual(window.tsunao.DISPLAY_SIZE, TsunaoWidget.DISPLAY_SIZE)
            self.assertEqual(window.tsunao.DISPLAY_SIZE, 188)
            self.assertFalse(hasattr(window, "view_selector"))
            self.assertEqual(
                window.workspace_splitter_handle.accessibleName(),
                "대기열과 미리보기 크기 조절",
            )

            window.show()
            self.app.processEvents()

            app_title = window.findChild(QLabel, "appTitle")
            self.assertIsNotNone(app_title)
            self.assertIsNone(window.findChild(QLabel, "subtitle"))
            self.assertGreater(window.brand_mascot.x(), app_title.x())

            window.workspace_splitter.setSizes([360, 980])
            window._sync_queue_layout()
            self.assertEqual(
                window.file_list.viewMode(),
                QListView.ViewMode.ListMode,
            )
            self.assertEqual(window.file_list.iconSize().width(), 42)

            window.workspace_splitter.setSizes([500, 840])
            window._sync_queue_layout()
            self.assertEqual(
                window.file_list.viewMode(),
                QListView.ViewMode.IconMode,
            )
            self.assertEqual(window.file_list.iconSize().width(), 80)
            self.assertFalse(window.preview_panel.isHidden())

            window.workspace_splitter.setSizes([760, 580])
            window._sync_queue_layout()
            self.assertEqual(
                window.file_list.viewMode(),
                QListView.ViewMode.IconMode,
            )
            self.assertEqual(window.file_list.iconSize().width(), 160)
            self.assertFalse(window.preview_panel.isHidden())
            window.close()

    def test_added_images_are_selected_and_only_selection_is_processed(self):
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(MainWindow, "_refresh_acceleration_status"),
        ):
            paths = []
            for index in range(3):
                image_path = Path(directory) / f"character-{index}.png"
                image = QImage(24, 24, QImage.Format.Format_RGB32)
                image.fill(0xFFE9EAFF)
                self.assertTrue(image.save(str(image_path)))
                paths.append(str(image_path))

            window = MainWindow()
            window.add_paths(paths)

            self.assertEqual(len(window.selected_items()), 3)
            self.assertIn("3개 선택", window.batch_count_label.text())
            self.assertEqual(
                window.primary_button.text(),
                "배경 제거 시작 (3)",
            )

            window.file_list.item(1).setSelected(False)
            with patch.object(window, "start_processing") as start:
                window.handle_primary_action()

            processed = start.call_args.args[0]
            self.assertEqual(
                [item.filename for item in processed],
                ["character-0.png", "character-2.png"],
            )

            window.file_list.clearSelection()
            self.assertFalse(window.primary_button.isEnabled())
            self.assertEqual(
                window.primary_button.text(),
                "처리할 이미지 선택",
            )
            window.select_all_button.click()
            self.assertEqual(len(window.selected_items()), 3)
            window.close()

    def test_ctrl_click_toggles_extended_selection(self):
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(MainWindow, "_refresh_acceleration_status"),
        ):
            paths = []
            for index in range(3):
                image_path = Path(directory) / f"select-{index}.png"
                image = QImage(24, 24, QImage.Format.Format_RGB32)
                image.fill(0xFFE9EAFF)
                self.assertTrue(image.save(str(image_path)))
                paths.append(str(image_path))

            window = MainWindow()
            window.show()
            window.add_paths(paths)
            window.file_list.clearSelection()
            self.app.processEvents()

            self.assertEqual(
                window.file_list.selectionMode(),
                QListView.SelectionMode.ExtendedSelection,
            )
            self.assertTrue(window.file_list.isSelectionRectVisible())

            first = window.file_list.visualItemRect(
                window.file_list.item(0)
            ).center()
            second = window.file_list.visualItemRect(
                window.file_list.item(1)
            ).center()
            QTest.mouseClick(
                window.file_list.viewport(),
                Qt.MouseButton.LeftButton,
                Qt.KeyboardModifier.NoModifier,
                first,
            )
            QTest.mouseClick(
                window.file_list.viewport(),
                Qt.MouseButton.LeftButton,
                Qt.KeyboardModifier.ControlModifier,
                second,
            )
            self.assertEqual(len(window.selected_items()), 2)

            QTest.mouseClick(
                window.file_list.viewport(),
                Qt.MouseButton.LeftButton,
                Qt.KeyboardModifier.ControlModifier,
                first,
            )
            self.assertEqual(len(window.selected_items()), 1)
            self.assertEqual(
                window.selected_items()[0].filename,
                "select-1.png",
            )
            window.close()

    def test_mouse_wheel_uses_fast_accumulated_pixel_scroll(self):
        class FakeWheelEvent:
            accepted = False

            @staticmethod
            def pixelDelta():
                return QPoint()

            @staticmethod
            def angleDelta():
                return QPoint(0, -120)

            @staticmethod
            def inverted():
                return False

            def accept(self):
                self.accepted = True

        file_list = ImageListWidget()
        file_list.resize(260, 180)
        for index in range(60):
            file_list.addItem(QListWidgetItem(f"image-{index}.png"))
        file_list.show()
        self.app.processEvents()
        event = FakeWheelEvent()

        file_list.wheelEvent(event)

        self.assertTrue(event.accepted)
        self.assertGreater(file_list._scroll_target, 120)
        file_list.close()

    def test_drag_rectangle_selects_and_ctrl_drag_toggles_items(self):
        file_list = ImageListWidget()
        file_list.setSelectionMode(
            QListView.SelectionMode.ExtendedSelection
        )
        file_list.setSelectionRectVisible(True)
        file_list.setViewMode(QListView.ViewMode.IconMode)
        file_list.setGridSize(QSize(100, 100))
        file_list.setWrapping(True)
        file_list.resize(420, 300)
        for index in range(6):
            file_list.addItem(QListWidgetItem(f"image-{index}.png"))
        file_list.show()
        self.app.processEvents()

        viewport = file_list.viewport()
        QTest.mousePress(
            viewport,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
            QPoint(380, 270),
        )
        QTest.mouseMove(viewport, QPoint(2, 2), 20)
        QTest.mouseRelease(
            viewport,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
            QPoint(2, 2),
        )
        self.assertEqual(len(file_list.selectedItems()), 6)

        QTest.mousePress(
            viewport,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.ControlModifier,
            QPoint(280, 230),
        )
        QTest.mouseMove(viewport, QPoint(2, 102), 20)
        QTest.mouseRelease(
            viewport,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.ControlModifier,
            QPoint(2, 102),
        )
        self.assertLess(len(file_list.selectedItems()), 6)
        self.assertGreater(len(file_list.selectedItems()), 0)
        file_list.close()

    def test_output_location_is_discarded_when_app_closes(self):
        class FakeSettings:
            def __init__(self):
                self.values = {
                    "output/storage_directory": "C:/old-output",
                }
                self.sync_count = 0

            def value(self, key, default=None, **kwargs):
                return self.values.get(key, default)

            def setValue(self, key, value):
                self.values[key] = value

            def remove(self, key):
                self.values.pop(key, None)

            def sync(self):
                self.sync_count += 1

        settings = FakeSettings()
        with (
            patch("main.QSettings", return_value=settings),
            patch.object(MainWindow, "_refresh_acceleration_status"),
        ):
            window = MainWindow()

            self.assertIsNone(window._output_directory)
            self.assertNotIn("output/storage_directory", settings.values)
            window._output_directory = Path("C:/current-output")
            settings.values["output/storage_directory"] = "C:/stale-output"

            window.close()

            self.assertIsNone(window._output_directory)
            self.assertNotIn("output/storage_directory", settings.values)
            self.assertGreaterEqual(settings.sync_count, 2)

    def test_drop_anywhere_adds_to_queue_while_processing(self):
        class FakeDropEvent:
            def __init__(self, path: Path):
                self._mime = QMimeData()
                self._mime.setUrls([QUrl.fromLocalFile(str(path))])
                self.accepted = False

            @staticmethod
            def type():
                return QEvent.Type.Drop

            def mimeData(self):
                return self._mime

            def acceptProposedAction(self):
                self.accepted = True

        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(MainWindow, "_refresh_acceleration_status"),
        ):
            image_path = Path(directory) / "dropped.png"
            image = QImage(32, 32, QImage.Format.Format_RGB32)
            image.fill(0xFFE9EAFF)
            self.assertTrue(image.save(str(image_path)))
            window = MainWindow()
            window._processing = True
            event = FakeDropEvent(image_path)

            handled = window.eventFilter(window.preview, event)

            self.assertTrue(handled)
            self.assertTrue(event.accepted)
            self.assertEqual(len(window._items), 1)
            self.assertEqual(
                Path(window._items[0].source_path).resolve(), image_path.resolve()
            )
            self.assertIn("현재 작업 후", window.status_label.text())
            window._processing = False
            window.close()

    def test_add_buttons_stay_enabled_while_processing(self):
        with patch.object(MainWindow, "_refresh_acceleration_status"):
            window = MainWindow()

            window._processing = True
            window._set_input_controls_enabled(False)

            self.assertTrue(window.header_add_button.isEnabled())
            self.assertTrue(window.header_folder_button.isEnabled())
            self.assertFalse(window.model_status_button.isEnabled())
            self.assertFalse(window.output_naming_button.isEnabled())
            self.assertFalse(window.performance_combo.isEnabled())
            window._processing = False
            window.close()

    def test_output_naming_dialog_previews_and_validates_affixes(self):
        dialog = OutputNamingDialog(OutputNamingPolicy())

        self.assertIn("character.png", dialog.preview_label.text())
        self.assertIn("character_no_bg.png", dialog.collision_label.text())

        dialog.prefix_field.setText("toonout_")
        dialog.suffix_field.setText("_cutout")
        self.assertIn(
            "toonout_character_cutout.png",
            dialog.preview_label.text(),
        )
        self.assertTrue(dialog.save_button.isEnabled())

        dialog.suffix_field.setText("bad:name")
        self.assertFalse(dialog.save_button.isEnabled())
        self.assertFalse(dialog.error_label.isHidden())
        dialog.close()

    def test_output_naming_choice_is_saved_and_reflected_in_button(self):
        with patch.object(MainWindow, "_refresh_acceleration_status"):
            window = MainWindow()
            window._settings = MagicMock()
            policy = OutputNamingPolicy("toonout_", "_cutout")

            with patch("main.OutputNamingDialog") as dialog_class:
                dialog = dialog_class.return_value
                dialog.exec.return_value = QDialog.DialogCode.Accepted
                dialog.policy = policy
                window.open_output_naming_dialog()

            self.assertEqual(window._output_naming_policy, policy)
            self.assertEqual(
                window.output_naming_button.text(),
                "이름 형식 · 사용자 지정",
            )
            window._settings.setValue.assert_any_call(
                "output/name_prefix",
                "toonout_",
            )
            window._settings.setValue.assert_any_call(
                "output/name_suffix",
                "_cutout",
            )
            window.close()

    def test_styled_combo_keeps_current_text_and_wide_popup(self):
        combo = ToonOutComboBox()
        combo.addItems(["절약", "균형", "최대 성능"])
        combo.setCurrentIndex(2)
        combo.setMinimumWidth(190)
        combo.show()
        self.app.processEvents()

        combo.showPopup()
        self.app.processEvents()

        self.assertEqual(combo.currentText(), "최대 성능")
        self.assertGreaterEqual(combo.view().minimumWidth(), combo.width())
        self.assertEqual(combo.view().objectName(), "comboPopup")
        combo.hidePopup()
        combo.close()

    def test_acceleration_dialog_offers_pack_install_in_same_app(self):
        info = AccelerationInfo(
            mode=AccelerationMode.GPU_PACK_AVAILABLE,
            device=NvidiaDevice("NVIDIA Test GPU", "999.1", 16.0),
        )
        dialog = AccelerationDialog(info)

        buttons = [button.text() for button in dialog.findChildren(QPushButton)]
        self.assertEqual(dialog.windowTitle(), "처리 장치와 GPU 가속")
        self.assertIn("GPU 가속 팩 설치", buttons)
        self.assertIsNotNone(
            dialog.findChild(QLabel, "accelerationAvailableLabel")
        )
        dialog.close()

    def test_model_install_dialog_can_cancel_or_hide_during_installation(self):
        with tempfile.TemporaryDirectory() as directory:
            dialog = ModelInstallDialog(
                directory,
                allow_elevation=True,
            )
            cancellations = []
            dialog.cancel_requested.connect(lambda: cancellations.append(True))
            dialog.show()
            dialog.resize(640, 371)
            self.app.processEvents()

            self.assertEqual(dialog.path_field.text(), directory)
            self.assertFalse(
                dialog.path_field.geometry().intersects(
                    dialog.space_label.geometry()
                )
            )
            self.assertEqual(dialog.install_button.text(), "모델 설치")
            dialog.install_button.click()

            self.assertEqual(dialog.cancel_button.text(), "설치 취소")
            self.assertTrue(dialog.cancel_button.isEnabled())
            self.assertEqual(dialog.progress_bar.minimum(), 0)
            self.assertEqual(dialog.progress_bar.maximum(), 100)
            self.assertEqual(dialog.progress_bar.value(), 0)
            self.assertEqual(dialog.status_label.text(), "[설치 준비 중] 0%")

            dialog.set_progress("ToonOut 가중치 다운로드 중", 47)
            self.assertEqual(dialog.progress_bar.value(), 47)
            self.assertEqual(
                dialog.status_label.text(),
                "[ToonOut 가중치 다운로드 중] 47%",
            )
            dialog.cancel_button.click()
            self.assertEqual(cancellations, [True])

            dialog.close()
            self.app.processEvents()
            self.assertFalse(dialog.isVisible())

    def test_gpu_install_dialog_shows_status_and_percent(self):
        dialog = ModelOperationDialog(
            "GPU 가속 팩 설치",
            "GPU 가속 팩 파일 목록 확인 중",
            determinate=True,
        )

        self.assertEqual(
            dialog.status_label.text(),
            "[GPU 가속 팩 파일 목록 확인 중] 0%",
        )
        dialog.set_progress("GPU 가속 파일 설치 중", 63)

        self.assertEqual(dialog.progress_bar.value(), 63)
        self.assertEqual(
            dialog.status_label.text(),
            "[GPU 가속 파일 설치 중] 63%",
        )
        dialog.close()

    def test_model_install_process_forwards_percent_event(self):
        process = ModelInstallProcess("C:/models")
        events = []
        process.progress_changed.connect(
            lambda status, percent: events.append((status, percent))
        )
        process._status_path.write_text(
            '{"event":"progress","status":"가중치 다운로드 중","percent":42}\n',
            encoding="utf-8",
        )

        process._read_status()

        self.assertEqual(events, [("가중치 다운로드 중", 42)])
        process._remove_status_file()
        process.deleteLater()

    def test_failed_model_install_starts_cleanup_before_reporting_failure(self):
        process = ModelInstallProcess("C:/models")
        process._last_error = "download failed"

        with patch.object(process, "_start_cleanup_worker") as start_cleanup:
            process._on_finished(1, None)

        start_cleanup.assert_called_once_with()
        self.assertFalse(process._settled)
        process._remove_status_file()
        process.deleteLater()

    def test_failed_model_install_process_finishes_automatic_cleanup(self):
        with tempfile.TemporaryDirectory() as directory:
            invalid_model_directory = Path(directory) / "not-a-directory"
            invalid_model_directory.write_text("fixture", encoding="utf-8")
            process = ModelInstallProcess(str(invalid_model_directory))
            statuses = []
            failures = []
            loop = QEventLoop()
            process.status_changed.connect(statuses.append)

            def handle_failure(error: str):
                failures.append(error)
                loop.quit()

            process.installation_failed.connect(handle_failure)
            QTimer.singleShot(15_000, loop.quit)
            process.start_installation()
            loop.exec()

            self.assertEqual(len(failures), 1)
            self.assertIn("남은 모델 파일을 정리하는 중", statuses)
            self.assertTrue(invalid_model_directory.is_file())
            process.deleteLater()

    def test_model_status_reopens_background_installation(self):
        with (
            patch("main.model_is_installed", return_value=False),
            patch.object(MainWindow, "_refresh_acceleration_status"),
        ):
            window = MainWindow()
            dialog = MagicMock()
            dialog.exec.return_value = QDialog.DialogCode.Rejected
            window._model_operation = "install"
            window._model_install_dialog = dialog
            window._refresh_model_status()

            window.open_model_status()

            self.assertEqual(window.model_status_button.text(), "● 모델 설치 중")
            dialog.exec.assert_called_once_with()
            window._model_operation = "idle"
            window._model_install_dialog = None
            window.close()

    def test_acceleration_status_reopens_background_gpu_installation(self):
        with patch.object(MainWindow, "_refresh_acceleration_status"):
            window = MainWindow()
            window._gpu_runtime_thread = MagicMock()
            window._gpu_runtime_dialog = MagicMock()

            window.open_acceleration_status()

            window._gpu_runtime_dialog.exec.assert_called_once_with()
            window._gpu_runtime_thread = None
            window._gpu_runtime_dialog = None
            window.close()

    def test_model_management_dialog_shows_installed_state(self):
        dialog = ModelManagementDialog("C:/ToonOut/models", "2.0 GB")

        self.assertEqual(dialog.windowTitle(), "설치된 모델 관리")
        dialog.close()

    def test_preselected_output_uses_final_paths_and_marks_result_saved(self):
        class FakeWorker:
            job_count = 0

            def set_performance_policy(self, policy):
                self.policy = policy

            def set_jobs(self, jobs):
                self.jobs = jobs
                self.job_count = len(jobs)

            def start(self):
                self.started = True

            @staticmethod
            def isRunning():
                return False

        with (
            tempfile.TemporaryDirectory() as directory,
            patch("main.model_is_installed", return_value=True),
            patch.object(MainWindow, "_refresh_acceleration_status"),
        ):
            window = MainWindow()
            worker = FakeWorker()
            window._worker = worker
            window._output_directory = Path(directory)
            window._output_naming_policy = OutputNamingPolicy()
            item = BatchItem("item-1", "C:/images/character.jpg")
            window._items = [item]
            window._items_by_id = {item.item_id: item}

            window.start_processing([item])

            output_path = Path(worker.jobs[0][2])
            self.assertEqual(output_path.parent, Path(directory))
            self.assertEqual(output_path.name, "character.png")
            self.assertEqual(window.tsunao.state, TsunaoState.DRAWING)
            window._on_item_completed(item.item_id, str(output_path))
            self.assertTrue(item.result_saved)
            window._on_batch_finished(False)
            self.assertEqual(window.tsunao.state, TsunaoState.COMPLETE)
            window.close()

    def test_processing_without_output_prompts_for_folder_and_uses_it(self):
        class FakeWorker:
            job_count = 0

            def set_performance_policy(self, policy):
                self.policy = policy

            def set_jobs(self, jobs):
                self.jobs = jobs
                self.job_count = len(jobs)

            def start(self):
                self.started = True

            @staticmethod
            def isRunning():
                return False

        with (
            tempfile.TemporaryDirectory() as directory,
            patch("main.model_is_installed", return_value=True),
            patch.object(MainWindow, "_refresh_acceleration_status"),
            patch.object(
                QFileDialog,
                "getExistingDirectory",
                return_value=directory,
            ) as folder_dialog,
        ):
            window = MainWindow()
            worker = FakeWorker()
            window._worker = worker
            item = BatchItem("item-1", "C:/images/character.jpg")
            window._items = [item]
            window._items_by_id = {item.item_id: item}

            window.start_processing([item])

            folder_dialog.assert_called_once()
            self.assertEqual(window._output_directory, Path(directory))
            self.assertEqual(Path(worker.jobs[0][2]).parent, Path(directory))
            self.assertTrue(worker.started)
            window._processing = False
            window.close()

    def test_cancelling_output_prompt_keeps_item_queued(self):
        class FakeWorker:
            job_count = 0

            @staticmethod
            def isRunning():
                return False

            def start(self):
                raise AssertionError("저장 위치 없이 작업이 시작되면 안 됩니다")

        with (
            patch.object(MainWindow, "_refresh_acceleration_status"),
            patch.object(QFileDialog, "getExistingDirectory", return_value=""),
        ):
            window = MainWindow()
            window._worker = FakeWorker()
            item = BatchItem("item-1", "C:/images/character.jpg")
            window._items = [item]
            window._items_by_id = {item.item_id: item}

            window.start_processing([item])

            self.assertIsNone(window._output_directory)
            self.assertEqual(item.state, ItemState.QUEUED)
            self.assertFalse(window._processing)
            self.assertIn("저장 위치를 지정", window.status_label.text())
            window.close()

    def test_items_added_during_processing_continue_as_next_batch(self):
        class FakeWorker:
            job_count = 0

            def set_performance_policy(self, policy):
                self.policy = policy

            def set_jobs(self, jobs):
                self.jobs = jobs
                self.job_count = len(jobs)

            def start(self):
                self.started = True

            @staticmethod
            def isRunning():
                return False

        with (
            tempfile.TemporaryDirectory() as directory,
            patch("main.model_is_installed", return_value=True),
            patch.object(MainWindow, "_refresh_acceleration_status"),
        ):
            window = MainWindow()
            worker = FakeWorker()
            window._worker = worker
            window._output_directory = Path(directory)
            completed = BatchItem(
                "completed",
                "C:/images/first.png",
                state=ItemState.COMPLETED,
            )
            pending = BatchItem("pending", "C:/images/next.png")
            window._items = [completed, pending]
            window._items_by_id = {
                completed.item_id: completed,
                pending.item_id: pending,
            }
            pending_list_item = QListWidgetItem(pending.filename)
            pending_list_item.setData(
                Qt.ItemDataRole.UserRole,
                pending.item_id,
            )
            window.file_list.addItem(pending_list_item)
            pending_list_item.setSelected(True)
            window._active_job_ids = {completed.item_id}
            window._processing = True

            window._on_batch_finished(False)
            self.app.processEvents()

            self.assertTrue(worker.started)
            self.assertEqual(worker.jobs[0][0], pending.item_id)
            window._processing = False
            window.close()

    def test_unselected_queued_item_does_not_auto_continue(self):
        class FakeWorker:
            job_count = 0
            started = False

            def set_performance_policy(self, policy):
                self.policy = policy

            def set_jobs(self, jobs):
                self.jobs = jobs
                self.job_count = len(jobs)

            def start(self):
                self.started = True

            @staticmethod
            def isRunning():
                return False

        with (
            tempfile.TemporaryDirectory() as directory,
            patch("main.model_is_installed", return_value=True),
            patch.object(MainWindow, "_refresh_acceleration_status"),
        ):
            window = MainWindow()
            worker = FakeWorker()
            window._worker = worker
            window._output_directory = Path(directory)
            completed = BatchItem(
                "completed",
                "C:/images/first.png",
                state=ItemState.COMPLETED,
            )
            pending = BatchItem("pending", "C:/images/next.png")
            window._items = [completed, pending]
            window._items_by_id = {
                completed.item_id: completed,
                pending.item_id: pending,
            }
            pending_list_item = QListWidgetItem(pending.filename)
            pending_list_item.setData(
                Qt.ItemDataRole.UserRole,
                pending.item_id,
            )
            window.file_list.addItem(pending_list_item)
            window.file_list.clearSelection()
            window._active_job_ids = {completed.item_id}
            window._processing = True

            window._on_batch_finished(False)
            self.app.processEvents()

            self.assertFalse(worker.started)
            self.assertEqual(pending.state, ItemState.QUEUED)
            self.assertIn("선택하지 않은 1개", window.status_label.text())
            window.close()


if __name__ == "__main__":
    unittest.main()
