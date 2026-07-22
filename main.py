import argparse
import sys
import uuid
from pathlib import Path

from PySide6.QtCore import (
    QEvent,
    QItemSelection,
    QItemSelectionModel,
    QPointF,
    QProcess,
    QSettings,
    QSize,
    Qt,
    QTimer,
    QUrl,
)
from PySide6.QtGui import (
    QDesktopServices,
    QFont,
    QFontDatabase,
    QGuiApplication,
    QIcon,
    QImageReader,
    QMouseEvent,
    QPixmap,
)
from PySide6.QtWidgets import (
    QApplication,
    QAbstractItemView,
    QButtonGroup,
    QCheckBox,
    QDialog,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QListView,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSlider,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from app_core import (
    MAX_IMAGE_PIXELS,
    STATE_PRESENTATION,
    SUPPORTED_SUFFIXES,
    BatchItem,
    ItemState,
    OutputNamingPolicy,
    available_output_path,
    validate_output_affix,
)
from app_settings import (
    LEGACY_OUTPUT_DIRECTORY_SETTING,
    MODEL_DIRECTORY_SETTING,
    OUTPUT_NAMING_PREFIX_SETTING,
    OUTPUT_NAMING_SUFFIX_SETTING,
    default_model_directory,
    default_update_directory,
    ensure_writable_directory,
    ensure_writable_model_directory,
    format_storage_size,
)
from app_version import APP_VERSION, UPDATE_MANIFEST_URL
from acceleration import AccelerationInfo, AccelerationMode
from gpu_download import GpuPackRelease, current_gpu_pack_release
from gpu_runtime import (
    GPU_RUNTIME_SETTING,
    default_gpu_runtime_directory,
    gpu_runtime_is_compatible,
    gpu_runtime_is_installed,
    load_gpu_runtime_manifest,
)
from model_installation import (
    model_is_installed,
    model_storage_size,
)
from inference import cleanup_abandoned_output_files
from mascot import TsunaoWidget
from preview import ImagePreview
from performance import (
    MODE_LABELS,
    PERFORMANCE_COOLDOWN_SETTING,
    PERFORMANCE_CPU_THREADS_SETTING,
    PERFORMANCE_GPU_MEMORY_SETTING,
    PERFORMANCE_MODE_SETTING,
    PerformanceMode,
    PerformancePolicy,
    preset_policy,
)
from privileges import is_running_as_admin, request_elevated_restart
from processing import (
    AccelerationDetectionThread,
    ExternalInferenceThread,
    GpuRuntimeFileThread,
    InferenceThread,
    ModelFileThread,
    ModelInstallProcess,
    run_model_cleanup_worker,
    run_model_install_worker,
)
from styles import APP_STYLESHEET
from update_config import UPDATE_PUBLIC_KEY_B64
from update_service import (
    UpdateError,
    UpdateRelease,
    prune_update_cache,
    verify_installer_file,
)
from update_workers import UpdateCheckThread, UpdateDownloadThread
from process_safety import start_parent_exit_watchdog
from widgets import (
    AccelerationDialog,
    ImageListWidget,
    ModelInstallDialog,
    ModelManagementDialog,
    ModelOperationDialog,
    OutputNamingDialog,
    PerformanceDialog,
    ToonOutComboBox,
    paths_from_drop_event,
)


SMALL_THUMBNAIL_MIN_WIDTH = 440
LARGE_THUMBNAIL_MIN_WIDTH = 700
BRAND_MASCOT_SIZE = 68
BRAND_MASCOT_PIXEL_RATIO = 4.0


def bundled_resource_path(*parts: str) -> Path:
    bundle_root = Path(
        getattr(sys, "_MEIPASS", Path(__file__).resolve().parent)
    )
    return bundle_root.joinpath(*parts)


def application_icon_path() -> Path:
    return bundled_resource_path("assets", "toonout.ico")


def brand_mascot_pixmap() -> QPixmap:
    """Return a dense pixmap that stays sharp on high-DPI displays."""

    source = QPixmap(str(bundled_resource_path("assets", "toonout-icon.png")))
    if source.isNull():
        return source
    pixel_size = round(BRAND_MASCOT_SIZE * BRAND_MASCOT_PIXEL_RATIO)
    pixmap = source.scaled(
        pixel_size,
        pixel_size,
        Qt.AspectRatioMode.KeepAspectRatio,
        Qt.TransformationMode.SmoothTransformation,
    )
    pixmap.setDevicePixelRatio(BRAND_MASCOT_PIXEL_RATIO)
    return pixmap


def configure_application_font(application: QApplication) -> None:
    """운영체제 UI 글꼴에 색 번짐 없는 안티앨리어싱을 적용한다."""

    system_font = QFontDatabase.systemFont(
        QFontDatabase.SystemFont.GeneralFont
    )
    font = QFont(system_font)
    # Segoe UI로 ASCII 경로의 역슬래시를 보존하고, 맑은 고딕으로 한글
    # 글리프를 보완한다. QFont의 families API를 써야 실제 fallback 목록이 된다.
    families = ["Segoe UI", system_font.family(), "Malgun Gothic"]
    font.setFamilies(list(dict.fromkeys(families)))
    font.setStyleStrategy(
        QFont.StyleStrategy.PreferAntialias
        | QFont.StyleStrategy.NoSubpixelAntialias
    )
    application.setFont(font)


class MainWindow(QMainWindow):
    def __init__(
        self,
        startup_model_directory: str | None = None,
        *,
        cleanup_update_cache: bool = False,
    ):
        super().__init__()

        self.setWindowTitle(f"ToonOut {APP_VERSION} · 배경 제거")
        self.resize(1440, 900)
        self.setMinimumSize(900, 620)

        self._items: list[BatchItem] = []
        self._items_by_id: dict[str, BatchItem] = {}
        self._processing = False
        self._pause_requested = False
        self._paused = False
        self._close_after_worker = False
        self._last_output_folder: str | None = None
        self._active_auto_save_ids: set[str] = set()
        self._active_job_ids: set[str] = set()
        self._auto_continue_allowed = False

        self._settings = QSettings("ToonOut", "ToonOut")
        # 출력 위치는 세션 전용이다. 이전 버전이 남긴 값도 재사용하지 않는다.
        self._settings.remove(LEGACY_OUTPUT_DIRECTORY_SETTING)
        self._settings.sync()
        self._output_directory: Path | None = None
        self._output_naming_policy = OutputNamingPolicy(
            prefix=str(
                self._settings.value(OUTPUT_NAMING_PREFIX_SETTING, "") or ""
            ),
            suffix=str(
                self._settings.value(OUTPUT_NAMING_SUFFIX_SETTING, "") or ""
            ),
        )
        if (
            validate_output_affix(self._output_naming_policy.prefix)
            or validate_output_affix(self._output_naming_policy.suffix)
        ):
            self._output_naming_policy = OutputNamingPolicy()
        self._performance_policy = self._load_performance_policy()
        saved_model_directory = self._settings.value(MODEL_DIRECTORY_SETTING)
        self._model_directory = Path(
            startup_model_directory
            or saved_model_directory
            or default_model_directory()
        )
        self._model_install_dialog: ModelInstallDialog | None = None
        self._model_install_process: ModelInstallProcess | None = None
        self._model_operation = "idle"
        self._model_file_thread: ModelFileThread | None = None
        self._gpu_runtime_directory = default_gpu_runtime_directory()
        self._gpu_enabled = bool(
            self._settings.value(GPU_RUNTIME_SETTING, False, type=bool)
        ) and gpu_runtime_is_installed(self._gpu_runtime_directory)
        self._acceleration_info: AccelerationInfo | None = None
        self._acceleration_error: str | None = None
        self._acceleration_thread: AccelerationDetectionThread | None = None
        self._gpu_runtime_thread: GpuRuntimeFileThread | None = None
        self._gpu_runtime_dialog: ModelOperationDialog | None = None
        self._gpu_runtime_action: str | None = None
        self._update_check_thread: UpdateCheckThread | None = None
        self._update_download_thread: UpdateDownloadThread | None = None
        self._update_release: UpdateRelease | None = None
        self._update_installer_path: Path | None = None
        self._update_error: str | None = None
        self._update_cleanup_attempts = 0

        if startup_model_directory:
            self._save_model_directory(self._model_directory)

        self._worker = self._new_inference_worker()
        self._connect_worker()
        self._build_interface()
        self._apply_style()
        self._update_interface_state()
        self._refresh_model_status()
        self._refresh_acceleration_status()
        if cleanup_update_cache:
            QTimer.singleShot(1_000, self._retry_update_cache_cleanup)
        QTimer.singleShot(1_500, self.check_for_updates)

    def _load_performance_policy(self) -> PerformancePolicy:
        saved_mode = str(
            self._settings.value(
                PERFORMANCE_MODE_SETTING,
                PerformanceMode.BALANCED.value,
            )
        )
        try:
            mode = PerformanceMode(saved_mode)
        except ValueError:
            mode = PerformanceMode.BALANCED

        if mode != PerformanceMode.CUSTOM:
            return preset_policy(mode)

        balanced = preset_policy(PerformanceMode.BALANCED)
        try:
            return PerformancePolicy(
                mode,
                int(
                    self._settings.value(
                        PERFORMANCE_CPU_THREADS_SETTING,
                        balanced.cpu_threads,
                    )
                ),
                float(
                    self._settings.value(
                        PERFORMANCE_GPU_MEMORY_SETTING,
                        balanced.gpu_memory_fraction,
                    )
                ),
                int(
                    self._settings.value(
                        PERFORMANCE_COOLDOWN_SETTING,
                        balanced.cooldown_ms,
                    )
                ),
            ).normalized()
        except (TypeError, ValueError):
            return balanced

    def _save_performance_policy(self) -> None:
        policy = self._performance_policy
        self._settings.setValue(PERFORMANCE_MODE_SETTING, policy.mode.value)
        self._settings.setValue(
            PERFORMANCE_CPU_THREADS_SETTING,
            policy.cpu_threads,
        )
        self._settings.setValue(
            PERFORMANCE_GPU_MEMORY_SETTING,
            policy.gpu_memory_fraction,
        )
        self._settings.setValue(
            PERFORMANCE_COOLDOWN_SETTING,
            policy.cooldown_ms,
        )
        self._settings.sync()

    def _new_inference_worker(self):
        if self._gpu_enabled:
            try:
                manifest = load_gpu_runtime_manifest(
                    self._gpu_runtime_directory,
                    verify_files=False,
                )
                if not gpu_runtime_is_compatible(manifest):
                    raise RuntimeError("GPU 가속 팩 업데이트가 필요합니다.")
                return ExternalInferenceThread(
                    [str(manifest.worker_path)],
                    str(self._model_directory),
                    self._performance_policy,
                )
            except Exception:
                self._gpu_enabled = False
                self._settings.setValue(GPU_RUNTIME_SETTING, False)
        return InferenceThread(
            str(self._model_directory),
            self._performance_policy,
        )

    def _replace_inference_worker(self, gpu_enabled: bool):
        if self._worker.isRunning():
            raise RuntimeError("처리 중에는 CPU/GPU를 바꿀 수 없습니다.")
        self._gpu_enabled = gpu_enabled
        self._settings.setValue(GPU_RUNTIME_SETTING, gpu_enabled)
        self._settings.sync()
        self._worker = self._new_inference_worker()
        self._connect_worker()

    def _connect_worker(self):
        self._worker.model_status.connect(self._on_model_status)
        self._worker.model_ready.connect(self._on_model_ready)
        self._worker.model_failed.connect(self._on_model_failed)
        self._worker.item_started.connect(self._on_item_started)
        self._worker.item_completed.connect(self._on_item_completed)
        self._worker.item_failed.connect(self._on_item_failed)
        self._worker.progress_changed.connect(self._on_progress_changed)
        self._worker.pause_reached.connect(self._on_pause_reached)
        self._worker.batch_finished.connect(self._on_batch_finished)

    def _build_interface(self):
        root = QWidget()
        root.setObjectName("root")
        root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(28, 24, 28, 24)
        root_layout.setSpacing(18)

        header = QHBoxLayout()
        header.setSpacing(10)
        brand_layout = QVBoxLayout()
        brand_layout.setSpacing(0)

        mascot_directory = bundled_resource_path("assets", "mascot")
        self.brand_mascot = QLabel()
        self.brand_mascot.setObjectName("brandMascot")
        self.brand_mascot.setFixedSize(
            BRAND_MASCOT_SIZE,
            BRAND_MASCOT_SIZE,
        )
        self.brand_mascot.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.brand_mascot.setAccessibleName("츠나오 아이콘")
        brand_pixmap = brand_mascot_pixmap()
        if not brand_pixmap.isNull():
            self.brand_mascot.setPixmap(brand_pixmap)

        title = QLabel("ToonOut")
        title.setObjectName("appTitle")
        title.setAlignment(Qt.AlignmentFlag.AlignVCenter)
        brand_layout.addStretch()
        brand_layout.addWidget(title)
        brand_layout.addStretch()

        self.header_add_button = QPushButton("이미지 추가")
        self.header_add_button.setObjectName("secondaryButton")
        self.header_folder_button = QPushButton("폴더 추가")
        self.header_folder_button.setObjectName("secondaryButton")
        self.tsunao_hide_checkbox = QCheckBox("츠나오 끄기")
        self.tsunao_hide_checkbox.setObjectName("mascotToggle")
        self.tsunao_hide_checkbox.setAccessibleName("츠나오 표시 끄기")
        self.tsunao_hide_checkbox.setToolTip(
            "체크하면 창 안에서 움직이는 츠나오를 숨깁니다"
        )
        self.tsunao_hide_checkbox.setCursor(
            Qt.CursorShape.PointingHandCursor
        )
        self.acceleration_button = QPushButton("장치 확인 중")
        self.acceleration_button.setObjectName("accelerationCheckingButton")
        self.acceleration_button.setToolTip(
            "CPU/GPU 처리 상태와 선택형 NVIDIA 가속 팩을 관리합니다"
        )
        self.acceleration_button.setEnabled(False)
        self.update_status_button = QPushButton("업데이트 확인")
        self.update_status_button.setObjectName("updateCheckingButton")
        self.update_status_button.setAccessibleName("ToonOut 업데이트 상태")
        self.update_status_button.setEnabled(bool(UPDATE_PUBLIC_KEY_B64))
        self.update_status_button.setToolTip(
            "GitHub Releases에서 최신 ToonOut 버전을 확인합니다"
            if UPDATE_PUBLIC_KEY_B64
            else "이 개발 빌드에는 업데이트 검증 키가 없습니다"
        )
        self.model_status_button = QPushButton()
        self.model_status_button.setToolTip(
            "모델 설치 상태와 저장 위치를 확인합니다"
        )

        header.addLayout(brand_layout)
        header.addWidget(
            self.brand_mascot,
            0,
            Qt.AlignmentFlag.AlignTop,
        )
        header.addStretch()
        header.addWidget(self.tsunao_hide_checkbox)
        header.addWidget(self.header_folder_button)
        header.addWidget(self.header_add_button)
        header.addWidget(self.update_status_button)
        header.addWidget(self.acceleration_button)
        header.addWidget(self.model_status_button)
        root_layout.addLayout(header)

        self.workspace_page = self._build_workspace_page()
        root_layout.addWidget(self.workspace_page, 1)

        self.action_bar = self._build_action_bar()
        root_layout.addWidget(self.action_bar)

        self.header_add_button.clicked.connect(self.choose_images)
        self.header_folder_button.clicked.connect(self.choose_folder)
        self.acceleration_button.clicked.connect(self.open_acceleration_status)
        self.update_status_button.clicked.connect(self.handle_update_action)
        self.model_status_button.clicked.connect(self.open_model_status)
        self.setCentralWidget(root)
        self.tsunao = TsunaoWidget(root)
        self.tsunao.set_assets(mascot_directory)
        self.tsunao_hide_checkbox.toggled.connect(
            self.tsunao.set_user_hidden
        )
        self.setAcceptDrops(True)
        root.setAcceptDrops(True)
        for child in root.findChildren(QWidget):
            child.setAcceptDrops(True)
        application = QApplication.instance()
        if application is not None:
            application.installEventFilter(self)
        QTimer.singleShot(0, self._place_tsunao_initial)

    def _place_tsunao_initial(self):
        self.tsunao.place_initial(
            QPointF(
                132,
                max(120, self.action_bar.geometry().top() - 250),
            )
        )

    @staticmethod
    def _repolish_control(control: QWidget):
        style = control.style()
        style.unpolish(control)
        style.polish(control)
        control.update()

    def _set_update_button(
        self,
        text: str,
        object_name: str,
        *,
        enabled: bool,
        tooltip: str,
    ):
        self.update_status_button.setText(text)
        self.update_status_button.setObjectName(object_name)
        self.update_status_button.setEnabled(enabled)
        self.update_status_button.setToolTip(tooltip)
        self.update_status_button.show()
        self._repolish_control(self.update_status_button)

    def _retry_update_cache_cleanup(self):
        """Retry while the just-finished installer still holds its own EXE."""

        self._update_cleanup_attempts += 1
        if prune_update_cache(default_update_directory(), None):
            return
        if self._update_cleanup_attempts < 30:
            QTimer.singleShot(2_000, self._retry_update_cache_cleanup)

    def check_for_updates(self):
        if not UPDATE_PUBLIC_KEY_B64:
            self._set_update_button(
                "업데이트 확인",
                "updateCheckingButton",
                enabled=False,
                tooltip="이 개발 빌드에는 업데이트 검증 키가 없습니다",
            )
            return
        if (
            self._update_check_thread is not None
            or self._update_download_thread is not None
        ):
            return

        self._update_error = None
        self._set_update_button(
            "업데이트 확인 중",
            "updateCheckingButton",
            enabled=False,
            tooltip="GitHub Releases에서 최신 버전을 확인하고 있습니다",
        )

        thread = UpdateCheckThread(
            UPDATE_MANIFEST_URL,
            UPDATE_PUBLIC_KEY_B64,
            APP_VERSION,
            default_update_directory(),
            self,
        )
        self._update_check_thread = thread
        thread.available.connect(self._on_update_available)
        thread.up_to_date.connect(self._on_update_up_to_date)
        thread.failed.connect(self._on_update_check_failed)
        thread.finished.connect(self._on_update_check_finished)
        thread.start()

    def _on_update_available(self, release: UpdateRelease):
        self._update_release = release
        self._update_installer_path = None
        self._start_update_download()

    def _on_update_up_to_date(self):
        self._update_release = None
        self._update_installer_path = None
        self._update_error = None
        self._set_update_button(
            "업데이트 확인",
            "updateCheckingButton",
            enabled=True,
            tooltip=f"현재 ToonOut {APP_VERSION}이 최신 버전입니다. 눌러서 다시 확인합니다",
        )

    def _on_update_check_failed(self, error: str):
        self._update_error = error
        self._set_update_button(
            "업데이트 확인",
            "updateWarningButton",
            enabled=True,
            tooltip=f"업데이트를 확인하지 못했습니다. 눌러서 다시 시도하세요.\n{error}",
        )

    def _on_update_check_finished(self):
        thread = self._update_check_thread
        self._update_check_thread = None
        if thread is not None:
            thread.deleteLater()

    def _start_update_download(self):
        release = self._update_release
        if release is None or self._update_download_thread is not None:
            return
        self._update_error = None
        self._set_update_button(
            f"업데이트 다운로드 · {release.version}",
            "updateDownloadingButton",
            enabled=False,
            tooltip="검증할 ToonOut 설치 파일을 자동으로 다운로드하고 있습니다",
        )
        thread = UpdateDownloadThread(
            release,
            default_update_directory() / release.version,
            self,
        )
        self._update_download_thread = thread
        thread.progress_changed.connect(self._on_update_download_progress)
        thread.ready.connect(self._on_update_download_ready)
        thread.failed.connect(self._on_update_download_failed)
        thread.finished.connect(self._on_update_download_finished)
        thread.start()

    def _on_update_download_progress(self, received: int, total: int):
        if total <= 0 or self._update_release is None:
            return
        percent = min(100, int(received * 100 / total))
        self._set_update_button(
            f"업데이트 다운로드 · {percent}%",
            "updateDownloadingButton",
            enabled=False,
            tooltip=f"ToonOut {self._update_release.version} 설치 파일을 다운로드하고 있습니다",
        )

    def _on_update_download_ready(self, installer_path: str):
        self._update_installer_path = Path(installer_path)
        self._update_error = None
        self._refresh_update_button()

    def _on_update_download_failed(self, error: str):
        self._update_installer_path = None
        self._update_error = error
        self._set_update_button(
            "업데이트 다시 받기",
            "updateWarningButton",
            enabled=True,
            tooltip=f"업데이트 다운로드를 완료하지 못했습니다.\n{error}",
        )

    def _on_update_download_finished(self):
        thread = self._update_download_thread
        self._update_download_thread = None
        if thread is not None:
            thread.deleteLater()

    def _refresh_update_button(self):
        release = self._update_release
        if release is None or self._update_installer_path is None:
            return
        if self._processing:
            self._set_update_button(
                "작업 후 업데이트",
                "updateReadyButton",
                enabled=False,
                tooltip="현재 이미지 처리가 끝나면 업데이트를 설치할 수 있습니다",
            )
            return
        self._set_update_button(
            f"업데이트 설치 · {release.version}",
            "updateReadyButton",
            enabled=True,
            tooltip="ToonOut을 종료하고 검증된 업데이트를 설치합니다",
        )

    def handle_update_action(self):
        if self._update_installer_path is not None:
            self.install_ready_update()
        elif self._update_release is not None:
            self._start_update_download()
        else:
            self.check_for_updates()

    def install_ready_update(self):
        release = self._update_release
        installer = self._update_installer_path
        if release is None or installer is None or self._processing:
            return
        try:
            verify_installer_file(release, installer)
        except UpdateError as error:
            self._on_update_download_failed(str(error))
            return

        arguments = [
            "/SILENT",
            "/SUPPRESSMSGBOXES",
            "/CLOSEAPPLICATIONS",
            "/NORESTART",
        ]
        result = QProcess.startDetached(
            str(installer),
            arguments,
            str(installer.parent),
        )
        started = result[0] if isinstance(result, tuple) else bool(result)
        if not started:
            QMessageBox.warning(
                self,
                "업데이트를 시작하지 못했습니다",
                "설치 프로그램을 실행할 수 없습니다. 업데이트를 다시 "
                "다운로드하거나 ToonOut을 다시 실행해 주세요.",
            )
            return

        self._set_update_button(
            "업데이트 시작 중",
            "updateReadyButton",
            enabled=False,
            tooltip="ToonOut을 종료하고 새 버전을 설치합니다",
        )
        application = QApplication.instance()
        if application is not None:
            QTimer.singleShot(0, application.quit)

    def eventFilter(self, watched, event):
        if (
            isinstance(event, QMouseEvent)
            and event.type()
            in {
                QEvent.Type.MouseButtonPress,
                QEvent.Type.MouseMove,
                QEvent.Type.MouseButtonRelease,
            }
            and isinstance(watched, QWidget)
            and watched.window() is self
            and QApplication.activeModalWidget() is None
            and hasattr(self, "tsunao")
        ):
            root_position = self.centralWidget().mapFromGlobal(
                event.globalPosition().toPoint()
            )
            if self.tsunao.handle_mouse_event(
                event,
                QPointF(root_position),
            ):
                event.accept()
                return True

        if (
            watched is getattr(self, "workspace_splitter_handle", None)
            and event.type() == QEvent.Type.KeyPress
            and event.key() in {Qt.Key.Key_Left, Qt.Key.Key_Right}
        ):
            direction = -1 if event.key() == Qt.Key.Key_Left else 1
            sizes = self.workspace_splitter.sizes()
            self.workspace_splitter.setSizes(
                [sizes[0] + direction * 40, sizes[1] - direction * 40]
            )
            self._sync_queue_layout()
            return True

        drop_event_types = {
            QEvent.Type.DragEnter,
            QEvent.Type.DragMove,
            QEvent.Type.Drop,
        }
        if (
            event.type() in drop_event_types
            and isinstance(watched, QWidget)
            and watched.window() is self
            and event.mimeData().hasUrls()
            and any(url.isLocalFile() for url in event.mimeData().urls())
        ):
            event.acceptProposedAction()
            if event.type() == QEvent.Type.Drop:
                self.add_paths(paths_from_drop_event(event))
            return True
        return super().eventFilter(watched, event)

    def _refresh_acceleration_status(self):
        if (
            self._acceleration_thread is not None
            and self._acceleration_thread.isRunning()
        ):
            return
        self.acceleration_button.setText("장치 확인 중")
        self.acceleration_button.setObjectName("accelerationCheckingButton")
        self.acceleration_button.setEnabled(False)
        self._repolish_control(self.acceleration_button)
        thread = AccelerationDetectionThread(
            str(self._gpu_runtime_directory),
            self._gpu_enabled,
            self,
        )
        self._acceleration_thread = thread
        thread.detected.connect(self._on_acceleration_detected)
        thread.failed.connect(self._on_acceleration_failed)
        thread.start()

    def _on_acceleration_detected(self, info: AccelerationInfo):
        self._acceleration_info = info
        self._acceleration_error = None
        self.acceleration_button.setText(info.status_text)
        self.acceleration_button.setObjectName(
            "accelerationWarningButton"
            if (
                info.runtime is not None
                and not gpu_runtime_is_compatible(info.runtime)
            )
            else {
                AccelerationMode.GPU_ACTIVE: "accelerationActiveButton",
                AccelerationMode.GPU_PACK_INSTALLED: (
                    "accelerationInstalledButton"
                ),
                AccelerationMode.GPU_PACK_AVAILABLE: (
                    "accelerationAvailableButton"
                ),
                AccelerationMode.GPU_UNAVAILABLE: (
                    "accelerationWarningButton"
                ),
                AccelerationMode.CPU_ONLY: "accelerationCpuButton",
            }[info.mode]
        )
        self.acceleration_button.setEnabled(not self._processing)
        self._repolish_control(self.acceleration_button)

    def _on_acceleration_failed(self, message: str):
        self._acceleration_error = message
        self.acceleration_button.setText("! 장치 확인 실패")
        self.acceleration_button.setObjectName("accelerationWarningButton")
        self.acceleration_button.setEnabled(not self._processing)
        self._repolish_control(self.acceleration_button)

    def open_acceleration_status(self):
        if self._processing:
            return
        if self._gpu_runtime_thread is not None:
            if self._gpu_runtime_dialog is not None:
                self._gpu_runtime_dialog.exec()
            return
        if self._acceleration_info is None:
            QMessageBox.warning(
                self,
                "처리 장치 확인 실패",
                "CPU/GPU 상태를 확인하지 못했습니다.\n"
                f"{self._acceleration_error or '잠시 후 다시 시도하세요.'}",
            )
            return

        dialog = AccelerationDialog(self._acceleration_info, self)
        def run_after_close(action):
            dialog.accept()
            QTimer.singleShot(0, action)

        dialog.install_requested.connect(
            lambda: run_after_close(self._install_gpu_runtime)
        )
        dialog.gpu_enabled_changed.connect(
            lambda enabled: run_after_close(
                lambda: self._set_gpu_runtime_enabled(enabled)
            )
        )
        dialog.remove_requested.connect(
            lambda: run_after_close(self._remove_gpu_runtime)
        )
        dialog.driver_page_requested.connect(
            lambda: QDesktopServices.openUrl(
                QUrl("https://www.nvidia.com/ko-kr/drivers/")
            )
        )
        dialog.exec()

    def _set_gpu_runtime_enabled(self, enabled: bool):
        if enabled and not gpu_runtime_is_installed(
            self._gpu_runtime_directory
        ):
            QMessageBox.warning(
                self,
                "GPU 가속 팩이 없습니다",
                "먼저 ToonOut NVIDIA GPU 가속 팩을 설치하세요.",
            )
            return
        if enabled:
            try:
                manifest = load_gpu_runtime_manifest(
                    self._gpu_runtime_directory,
                    verify_files=False,
                )
            except Exception as error:
                QMessageBox.warning(self, "GPU 가속 팩 확인 실패", str(error))
                return
            if not gpu_runtime_is_compatible(manifest):
                QMessageBox.warning(
                    self,
                    "GPU 가속 팩 업데이트 필요",
                    "설치된 GPU 팩은 성능 모드와 일시정지를 지원하지 "
                    "않습니다. 최신 ToonOut GPU 가속 팩을 설치하세요.",
                )
                return
        self._replace_inference_worker(enabled)
        self._acceleration_info = None
        self._refresh_acceleration_status()
        self.status_label.setText(
            "다음 작업부터 NVIDIA GPU를 사용합니다"
            if enabled
            else "다음 작업부터 CPU를 사용합니다"
        )

    def _install_gpu_runtime(self):
        try:
            release = current_gpu_pack_release()
        except Exception as error:
            QMessageBox.warning(
                self,
                "GPU 가속 팩 자동 설치 불가",
                str(error),
            )
            return
        self._run_gpu_runtime_operation("install", pack_release=release)

    def _remove_gpu_runtime(self):
        choice = QMessageBox.warning(
            self,
            "GPU 가속 팩 삭제",
            "ToonOut의 NVIDIA GPU 가속 파일만 삭제할까요?\n\n"
            f"위치: {self._gpu_runtime_directory}\n\n"
            "모델과 사용자 이미지는 삭제하지 않으며 이후 CPU로 처리합니다.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel,
        )
        if choice != QMessageBox.StandardButton.Yes:
            return
        self._replace_inference_worker(False)
        self._run_gpu_runtime_operation("delete")

    def _run_gpu_runtime_operation(
        self,
        action: str,
        pack_path: str | None = None,
        pack_release: GpuPackRelease | None = None,
    ):
        if self._gpu_runtime_thread is not None:
            return
        title = "GPU 가속 팩 설치" if action == "install" else "GPU 가속 팩 삭제"
        status = (
            "GPU 가속 팩 배포 정보 확인 중"
            if action == "install"
            else "GPU 가속 파일을 삭제하는 중"
        )
        dialog = ModelOperationDialog(
            title,
            status,
            parent=self,
            allow_background=action == "install",
            cancellable=action == "install",
            determinate=action == "install",
        )
        outcome: dict[str, object] = {"result": None}
        thread = GpuRuntimeFileThread(
            action,
            str(self._gpu_runtime_directory),
            pack_path,
            pack_release,
        )
        self._gpu_runtime_thread = thread
        self._gpu_runtime_dialog = dialog
        self._gpu_runtime_action = action
        self.acceleration_button.setText("● GPU 팩 설치 중")
        self.acceleration_button.setObjectName("accelerationCheckingButton")
        self.acceleration_button.setToolTip(
            "눌러서 백그라운드 GPU 가속 팩 설치 현황을 봅니다"
        )
        self.acceleration_button.setEnabled(True)
        self._repolish_control(self.acceleration_button)
        self._update_action_state()

        def handle_status(message: str):
            dialog.set_status(message)
            self.acceleration_button.setToolTip(
                f"{message}\n눌러서 설치 창을 다시 엽니다"
            )

        def handle_progress(status: str, percent: int):
            dialog.set_progress(status, percent)
            self.acceleration_button.setText(f"● GPU 팩 설치 · {percent}%")
            self.acceleration_button.setToolTip(
                f"[{status}] {percent}%\n눌러서 설치 창을 다시 엽니다"
            )

        def handle_installed(manifest):
            if not gpu_runtime_is_compatible(manifest):
                outcome["result"] = "failure"
                outcome["error"] = "최신 GPU 가속 팩이 필요합니다."
                dialog.fail()
                return
            outcome["result"] = "success"
            dialog.finish()

        def handle_deleted():
            outcome["result"] = "success"
            dialog.finish()

        def handle_failure(error: str):
            outcome["result"] = "failure"
            outcome["error"] = error
            dialog.fail()

        def handle_cancelled():
            outcome["result"] = "cancelled"
            dialog.show_cancelled()

        dialog.cancel_requested.connect(self._cancel_gpu_runtime_operation)
        thread.status_changed.connect(handle_status)
        thread.progress_changed.connect(handle_progress)
        thread.installed.connect(handle_installed)
        thread.deleted.connect(handle_deleted)
        thread.failed.connect(handle_failure)
        thread.cancelled.connect(handle_cancelled)
        thread.finished.connect(
            lambda: self._finish_gpu_runtime_operation(
                thread,
                dialog,
                action,
                title,
                outcome,
            )
        )
        thread.start()
        dialog.exec()

    def _cancel_gpu_runtime_operation(self):
        thread = self._gpu_runtime_thread
        if thread is None or self._gpu_runtime_action != "install":
            return
        if self._gpu_runtime_dialog is not None:
            self._gpu_runtime_dialog.set_status(
                "GPU 가속 팩 설치를 취소하고 임시 파일을 정리하는 중"
            )
        thread.request_cancel()

    def _finish_gpu_runtime_operation(
        self,
        thread: GpuRuntimeFileThread,
        dialog: ModelOperationDialog,
        action: str,
        title: str,
        outcome: dict[str, object],
    ):
        if self._gpu_runtime_thread is not thread:
            return
        self._gpu_runtime_thread = None
        self._gpu_runtime_dialog = None
        self._gpu_runtime_action = None
        result = outcome.get("result")
        self._update_action_state()

        if result == "success":
            if action == "install":
                self._replace_inference_worker(True)
                message = "GPU 가속 팩을 설치했습니다. 다음 작업부터 GPU를 사용합니다."
            else:
                message = "GPU 가속 팩을 삭제했습니다. 이후 CPU로 처리합니다."
            self._acceleration_info = None
            self._refresh_acceleration_status()
            QMessageBox.information(self, f"{title} 완료", message)
        elif result == "cancelled":
            self._acceleration_info = None
            self._refresh_acceleration_status()
            self.status_label.setText("GPU 가속 팩 설치를 취소했습니다")
        else:
            self._acceleration_info = None
            self._refresh_acceleration_status()
            QMessageBox.warning(
                self,
                f"{title} 실패",
                str(outcome.get("error", "GPU 가속 팩 작업을 완료하지 못했습니다.")),
            )

        if dialog.isVisible():
            dialog.finished.connect(dialog.deleteLater)
        else:
            dialog.deleteLater()
        thread.deleteLater()
        if self._close_after_worker:
            self.close()

    def _save_model_directory(self, directory: str | Path):
        self._model_directory = Path(directory)
        self._settings.setValue(
            MODEL_DIRECTORY_SETTING,
            str(self._model_directory),
        )
        self._settings.sync()

    def _refresh_model_status(self):
        if self._model_operation == "install":
            self.model_status_button.setText("● 모델 설치 중")
            self.model_status_button.setObjectName("modelBusyButton")
            self.model_status_button.setToolTip(
                "눌러서 백그라운드 모델 설치 현황을 봅니다"
            )
        elif self._model_operation != "idle":
            self.model_status_button.setText("● 모델 작업 중")
            self.model_status_button.setObjectName("modelBusyButton")
            self.model_status_button.setToolTip("모델 파일 작업이 진행 중입니다")
        elif model_is_installed(self._model_directory):
            self.model_status_button.setText("● 모델 설치됨")
            self.model_status_button.setObjectName("modelInstalledButton")
            self.model_status_button.setToolTip(
                "모델 설치 상태와 저장 위치를 확인합니다"
            )
        else:
            self.model_status_button.setText("● 모델 설치 필요")
            self.model_status_button.setObjectName("modelRequiredButton")
            self.model_status_button.setToolTip(
                "모델 설치 상태와 저장 위치를 확인합니다"
            )

        style = self.model_status_button.style()
        style.unpolish(self.model_status_button)
        style.polish(self.model_status_button)
        self.model_status_button.update()

    def open_model_status(self):
        if self._processing:
            return
        if self._model_operation == "install":
            self.show_model_installer()
            return
        if self._model_operation != "idle":
            return
        if model_is_installed(self._model_directory):
            self._show_model_management()
        else:
            self.show_model_installer()

    def show_model_installer(self) -> bool:
        if self._worker.isRunning():
            return False

        dialog = self._model_install_dialog
        if self._model_operation == "install":
            if dialog is None:
                return False
        elif self._model_operation != "idle":
            return False
        else:
            dialog = ModelInstallDialog(
                self._model_directory,
                allow_elevation=not is_running_as_admin(),
                parent=self,
            )
            dialog.install_requested.connect(self._begin_model_install)
            dialog.cancel_requested.connect(self._cancel_model_install)
            dialog.elevation_requested.connect(self._restart_as_admin)
            self._model_install_dialog = dialog

        result = dialog.exec()
        installed = model_is_installed(self._model_directory)
        if self._model_operation != "install":
            self._release_model_install_dialog(dialog)
        self._refresh_model_status()
        return result == ModelInstallDialog.DialogCode.Accepted and installed

    def _release_model_install_dialog(self, dialog: ModelInstallDialog):
        if self._model_install_dialog is dialog:
            self._model_install_dialog = None
            dialog.deleteLater()

    def _begin_model_install(self, directory: str):
        if self._model_install_process is not None:
            return
        self._save_model_directory(directory)
        self._worker.reset_engine(directory)
        self._worker.set_jobs([])
        self._model_operation = "install"
        self._refresh_model_status()
        self._update_action_state()

        process = ModelInstallProcess(directory, self)
        self._model_install_process = process
        process.status_changed.connect(self._on_model_install_status)
        process.progress_changed.connect(self._on_model_install_progress)
        process.installation_succeeded.connect(self._on_model_install_succeeded)
        process.installation_failed.connect(self._on_model_install_failed)
        process.installation_cancelled.connect(self._on_model_install_cancelled)
        process.start_installation()

    def _cancel_model_install(self):
        process = self._model_install_process
        if process is None:
            return
        if self._model_install_dialog is not None:
            self._model_install_dialog.show_cancelling()
        process.request_cancel()

    def _on_model_install_status(self, message: str):
        if self._model_install_dialog is not None:
            self._model_install_dialog.set_status(message)
        self.model_status_button.setToolTip(
            f"{message}\n눌러서 설치 창을 다시 엽니다"
        )

    def _on_model_install_progress(self, status: str, percent: int):
        if self._model_install_dialog is not None:
            self._model_install_dialog.set_progress(status, percent)
        self.model_status_button.setText(f"● 모델 설치 · {percent}%")
        self.model_status_button.setToolTip(
            f"[{status}] {percent}%\n눌러서 설치 창을 다시 엽니다"
        )

    def _finish_model_install_process(self):
        process = self._model_install_process
        self._model_install_process = None
        self._model_operation = "idle"
        self._worker.reset_engine(str(self._model_directory))
        self._refresh_model_status()
        self._update_action_state()
        if process is not None:
            process.deleteLater()

    def _on_model_install_succeeded(self):
        dialog = self._model_install_dialog
        self._finish_model_install_process()
        if dialog is not None:
            dialog.show_success()
            if not dialog.isVisible():
                self._release_model_install_dialog(dialog)
        if self._close_after_worker:
            self.close()

    def _on_model_install_failed(self, error: str):
        dialog = self._model_install_dialog
        self._finish_model_install_process()
        if dialog is not None:
            dialog.show_failure(error)
            if not dialog.isVisible():
                self._release_model_install_dialog(dialog)
        else:
            QMessageBox.warning(self, "모델 설치 실패", error)
        if self._close_after_worker:
            self.close()

    def _on_model_install_cancelled(self):
        dialog = self._model_install_dialog
        self._finish_model_install_process()
        if dialog is not None:
            dialog.show_cancelled()
            if not dialog.isVisible():
                self._release_model_install_dialog(dialog)
        self.status_label.setText("모델 설치를 취소했습니다")
        if self._close_after_worker:
            self.close()

    def _restart_as_admin(self, directory: str):
        self._save_model_directory(directory)
        image_paths = [item.source_path for item in self._items]
        if not request_elevated_restart(directory, image_paths):
            QMessageBox.warning(
                self,
                "관리자 권한을 얻지 못했습니다",
                "Windows 권한 요청이 취소되었거나 앱을 다시 실행하지 못했습니다. "
                "다른 저장 위치를 선택할 수 있습니다.",
            )
            return

        if self._model_install_dialog is not None:
            self._model_install_dialog.reject()
        QTimer.singleShot(0, self.close)

    def _show_model_management(self):
        if not model_is_installed(self._model_directory):
            return
        size_text = format_storage_size(
            model_storage_size(self._model_directory)
        )
        dialog = ModelManagementDialog(
            self._model_directory,
            size_text,
            parent=self,
        )
        dialog.open_requested.connect(self._open_model_folder)
        result = dialog.exec()

        if result == ModelManagementDialog.MOVE_RESULT:
            self._move_model_installation()
        elif result == ModelManagementDialog.DELETE_RESULT:
            self._delete_model_installation()

    def _open_model_folder(self):
        self._model_directory.mkdir(parents=True, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(self._model_directory)))

    def _move_model_installation(self):
        destination = QFileDialog.getExistingDirectory(
            self,
            "모델을 옮길 폴더 선택",
            str(self._model_directory.parent),
            QFileDialog.Option.ShowDirsOnly,
        )
        if not destination:
            return

        destination_path = Path(destination)
        if destination_path == self._model_directory:
            QMessageBox.information(
                self,
                "같은 위치입니다",
                "현재 모델 저장 위치와 다른 폴더를 선택하세요.",
            )
            return
        try:
            ensure_writable_model_directory(destination_path)
        except OSError as error:
            QMessageBox.warning(
                self,
                "새 위치를 사용할 수 없습니다",
                "선택한 폴더에 모델 파일을 쓸 수 없습니다. "
                f"다른 위치를 선택하세요.\n\n세부 정보: {error}",
            )
            return

        choice = QMessageBox.question(
            self,
            "모델 이동",
            f"모델을 다음 위치로 옮길까요?\n\n{destination_path}\n\n"
            "새 위치의 파일을 확인한 뒤 기존 파일을 삭제합니다.",
        )
        if choice != QMessageBox.StandardButton.Yes:
            return
        self._run_model_file_operation("move", destination_path)

    def _delete_model_installation(self):
        choice = QMessageBox.warning(
            self,
            "모델 삭제",
            "설치된 ToonOut 모델 파일을 삭제할까요?\n\n"
            f"위치: {self._model_directory}\n\n"
            "이미지와 다른 파일은 삭제하지 않습니다. 다시 배경을 제거하려면 "
            "모델을 다시 다운로드해야 합니다.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel,
        )
        if choice != QMessageBox.StandardButton.Yes:
            return
        self._run_model_file_operation("delete")

    def _run_model_file_operation(
        self,
        action: str,
        destination: Path | None = None,
    ):
        source = self._model_directory
        self._worker.reset_engine(str(source))
        self._model_operation = action
        self._refresh_model_status()

        title = "모델 이동" if action == "move" else "모델 삭제"
        initial_status = (
            "모델 파일을 새 위치로 복사하는 중"
            if action == "move"
            else "모델 파일을 삭제하는 중"
        )
        dialog = ModelOperationDialog(title, initial_status, parent=self)
        outcome: dict[str, object] = {"success": False, "source_removed": True}
        thread = ModelFileThread(
            action,
            str(source),
            str(destination) if destination is not None else None,
        )
        self._model_file_thread = thread

        def handle_success(source_removed: bool):
            outcome["success"] = True
            outcome["source_removed"] = source_removed
            dialog.finish()

        def handle_failure(error: str):
            outcome["error"] = error
            dialog.fail()

        thread.status_changed.connect(dialog.set_status)
        thread.succeeded.connect(handle_success)
        thread.failed.connect(handle_failure)
        thread.start()
        dialog.exec()
        thread.wait()
        self._model_file_thread = None
        self._model_operation = "idle"

        if outcome["success"]:
            if action == "move" and destination is not None:
                self._save_model_directory(destination)
            else:
                self._save_model_directory(source)
            self._worker.reset_engine(str(self._model_directory))
            self._refresh_model_status()

            if action == "move" and not outcome["source_removed"]:
                QMessageBox.warning(
                    self,
                    "모델은 이동했지만 기존 파일이 남았습니다",
                    "새 위치는 정상적으로 사용할 수 있습니다. 기존 위치의 모델 "
                    f"파일은 직접 확인하세요.\n\n{source}",
                )
            else:
                message = (
                    f"모델을 새 위치로 옮겼습니다.\n\n{self._model_directory}"
                    if action == "move"
                    else "모델 파일을 삭제했습니다."
                )
                QMessageBox.information(self, f"{title} 완료", message)
            return

        self._worker.reset_engine(str(source))
        self._refresh_model_status()
        QMessageBox.warning(
            self,
            f"{title} 실패",
            str(outcome.get("error", "모델 파일 작업을 완료하지 못했습니다.")),
        )

    def _build_workspace_page(self):
        page = QWidget()
        page_layout = QVBoxLayout(page)
        page_layout.setContentsMargins(0, 0, 0, 0)
        page_layout.setSpacing(10)

        queue_toolbar = QHBoxLayout()
        self.batch_count_label = QLabel("0개 이미지")
        self.batch_count_label.setObjectName("sectionTitle")
        self.retry_button = QPushButton("다시 시도")
        self.retry_button.setObjectName("secondaryButton")
        self.remove_button = QPushButton("목록에서 제거")
        self.remove_button.setObjectName("secondaryButton")
        self.clear_button = QPushButton("모두 비우기")
        self.clear_button.setObjectName("quietButton")

        queue_toolbar.addWidget(self.batch_count_label)
        queue_toolbar.addWidget(self.retry_button)
        queue_toolbar.addWidget(self.remove_button)
        queue_toolbar.addWidget(self.clear_button)
        queue_toolbar.addStretch()
        page_layout.addLayout(queue_toolbar)

        self.file_list = ImageListWidget()
        self.file_list.setObjectName("fileList")
        self.file_list.setAccessibleDescription(
            "이미지를 창 어디에나 끌어 놓으면 대기열에 추가됩니다. "
            "드래그로 여러 이미지를 선택하고 Ctrl을 누르면 선택을 "
            "추가하거나 해제할 수 있습니다. 오른쪽 경계를 넓히면 "
            "썸네일이 커집니다"
        )
        self.file_list.setMinimumWidth(260)
        self.file_list.setResizeMode(QListView.ResizeMode.Adjust)
        self.file_list.setMovement(QListView.Movement.Static)
        self.file_list.setWordWrap(True)
        self.file_list.setSelectionMode(
            QAbstractItemView.SelectionMode.ExtendedSelection
        )
        self.file_list.setSelectionRectVisible(True)

        self.preview_panel = self._build_preview_panel()
        self.preview.show_message(
            "이미지를 추가하면 이곳에서 결과를 확인할 수 있습니다"
        )

        self.workspace_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.workspace_splitter.addWidget(self.file_list)
        self.workspace_splitter.addWidget(self.preview_panel)
        self.workspace_splitter.setStretchFactor(0, 1)
        self.workspace_splitter.setStretchFactor(1, 3)
        self.workspace_splitter.setChildrenCollapsible(False)
        self.workspace_splitter.setHandleWidth(10)
        self.workspace_splitter.setSizes([340, 980])
        self.workspace_splitter_handle = self.workspace_splitter.handle(1)
        self.workspace_splitter_handle.setAccessibleName(
            "대기열과 미리보기 크기 조절"
        )
        self.workspace_splitter_handle.setToolTip(
            "좌우로 끌어 대기열 폭과 썸네일 크기를 조절하세요"
        )
        self.workspace_splitter_handle.setFocusPolicy(
            Qt.FocusPolicy.StrongFocus
        )
        self.workspace_splitter_handle.installEventFilter(self)
        page_layout.addWidget(self.workspace_splitter, 1)

        self.file_list.files_dropped.connect(self.add_paths)
        self.file_list.currentRowChanged.connect(self.show_selected_preview)
        self.file_list.itemSelectionChanged.connect(
            self._update_selection_actions
        )
        self.workspace_splitter.splitterMoved.connect(
            self._sync_queue_layout
        )
        self.retry_button.clicked.connect(self.retry_selected)
        self.remove_button.clicked.connect(self.remove_selected)
        self.clear_button.clicked.connect(self.clear_all)
        self._sync_queue_layout()
        QTimer.singleShot(0, self._sync_queue_layout)
        return page

    def _build_preview_panel(self):
        panel = QFrame()
        panel.setObjectName("previewPanel")
        panel.setMinimumWidth(440)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        preview_header = QHBoxLayout()
        self.preview_title = QLabel("미리보기")
        self.preview_title.setObjectName("previewTitle")

        self.preview_mode_group = QButtonGroup(self)
        self.preview_mode_group.setExclusive(True)
        self.original_mode_button = QPushButton("원본")
        self.result_mode_button = QPushButton("결과")
        self.compare_mode_button = QPushButton("비교")

        for button in (
            self.original_mode_button,
            self.result_mode_button,
            self.compare_mode_button,
        ):
            button.setCheckable(True)
            button.setObjectName("modeButton")
            self.preview_mode_group.addButton(button)

        self.original_mode_button.setChecked(True)

        preview_header.addWidget(self.preview_title)
        preview_header.addStretch()
        preview_header.addWidget(self.original_mode_button)
        preview_header.addWidget(self.result_mode_button)
        preview_header.addWidget(self.compare_mode_button)
        layout.addLayout(preview_header)

        self.preview = ImagePreview()
        layout.addWidget(self.preview, 1)

        self.compare_slider = QSlider(Qt.Orientation.Horizontal)
        self.compare_slider.setRange(0, 100)
        self.compare_slider.setValue(50)
        self.compare_slider.setToolTip("왼쪽은 결과, 오른쪽은 원본")
        self.compare_slider.hide()
        layout.addWidget(self.compare_slider)

        self.preview_error_label = QLabel("")
        self.preview_error_label.setObjectName("errorText")
        self.preview_error_label.setWordWrap(True)
        self.preview_error_label.hide()
        layout.addWidget(self.preview_error_label)

        controls = QHBoxLayout()
        hint = QLabel("휠 확대 · 좌클릭 드래그 · 더블클릭 화면 맞춤")
        hint.setObjectName("mutedText")
        self.zoom_out_button = QPushButton("축소")
        self.zoom_in_button = QPushButton("확대")
        self.actual_size_button = QPushButton("100%")
        self.reset_view_button = QPushButton("화면 맞춤")

        for button in (
            self.zoom_out_button,
            self.zoom_in_button,
            self.actual_size_button,
            self.reset_view_button,
        ):
            button.setObjectName("quietButton")

        controls.addWidget(hint)
        controls.addStretch()
        controls.addWidget(self.zoom_out_button)
        controls.addWidget(self.zoom_in_button)
        controls.addWidget(self.actual_size_button)
        controls.addWidget(self.reset_view_button)
        layout.addLayout(controls)

        self.original_mode_button.clicked.connect(
            lambda: self.set_preview_mode("original")
        )
        self.result_mode_button.clicked.connect(
            lambda: self.set_preview_mode("result")
        )
        self.compare_mode_button.clicked.connect(
            lambda: self.set_preview_mode("compare")
        )
        self.compare_slider.valueChanged.connect(
            lambda value: self.preview.set_split(value / 100)
        )
        self.zoom_out_button.clicked.connect(self.preview.zoom_out)
        self.zoom_in_button.clicked.connect(self.preview.zoom_in)
        self.actual_size_button.clicked.connect(self.preview.actual_size)
        self.reset_view_button.clicked.connect(self.preview.reset_view)
        return panel

    def _build_action_bar(self):
        action_bar = QFrame()
        action_bar.setObjectName("actionBar")
        layout = QVBoxLayout(action_bar)
        layout.setContentsMargins(18, 14, 18, 14)
        layout.setSpacing(10)

        status_row = QHBoxLayout()
        self.status_label = QLabel("이미지를 추가하세요")
        self.status_label.setObjectName("statusText")
        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setMinimumWidth(260)
        self.progress_bar.hide()
        status_row.addWidget(self.status_label)
        status_row.addStretch()
        status_row.addWidget(self.progress_bar)
        layout.addLayout(status_row)

        settings_row = QHBoxLayout()
        self.output_note = QLabel()
        self.output_note.setObjectName("mutedText")
        self.output_folder_button = QPushButton()
        self.output_folder_button.setObjectName("secondaryButton")
        self.output_naming_button = QPushButton()
        self.output_naming_button.setObjectName("secondaryButton")

        performance_label = QLabel("성능")
        performance_label.setObjectName("fieldLabel")
        self.performance_combo = ToonOutComboBox()
        self.performance_combo.setAccessibleName("성능 모드")
        self.performance_combo.setMinimumWidth(150)
        for mode in (
            PerformanceMode.ECO,
            PerformanceMode.BALANCED,
            PerformanceMode.MAXIMUM,
            PerformanceMode.CUSTOM,
        ):
            self.performance_combo.addItem(MODE_LABELS[mode], mode.value)

        settings_row.addWidget(self.output_note)
        settings_row.addWidget(self.output_folder_button)
        settings_row.addWidget(self.output_naming_button)
        settings_row.addStretch()
        settings_row.addWidget(performance_label)
        settings_row.addWidget(self.performance_combo)
        layout.addLayout(settings_row)

        action_row = QHBoxLayout()
        self.new_task_button = QPushButton("새 작업")
        self.new_task_button.setObjectName("quietButton")
        self.open_folder_button = QPushButton("저장 폴더 열기")
        self.open_folder_button.setObjectName("secondaryButton")
        self.pause_button = QPushButton("일시정지")
        self.pause_button.setObjectName("secondaryButton")
        self.cancel_button = QPushButton("중지")
        self.cancel_button.setObjectName("dangerButton")
        self.select_all_button = QPushButton("모두 선택하기")
        self.select_all_button.setObjectName("secondaryButton")
        self.primary_button = QPushButton("배경 제거 시작")
        self.primary_button.setObjectName("primaryButton")

        self.new_task_button.hide()
        self.open_folder_button.hide()
        self.pause_button.hide()
        self.cancel_button.hide()

        action_row.addStretch()
        action_row.addWidget(self.new_task_button)
        action_row.addWidget(self.open_folder_button)
        action_row.addWidget(self.pause_button)
        action_row.addWidget(self.cancel_button)
        action_row.addWidget(self.select_all_button)
        action_row.addWidget(self.primary_button)
        layout.addLayout(action_row)

        self._refresh_output_controls()
        self._select_performance_combo(self._performance_policy.mode)
        self._refresh_performance_tooltip()
        self.output_folder_button.clicked.connect(self.choose_output_folder)
        self.output_naming_button.clicked.connect(
            self.open_output_naming_dialog
        )
        self.performance_combo.currentIndexChanged.connect(
            self._on_performance_mode_selected
        )
        self.select_all_button.clicked.connect(self.select_all_items)
        self.primary_button.clicked.connect(self.handle_primary_action)
        self.pause_button.clicked.connect(self.toggle_pause)
        self.cancel_button.clicked.connect(self.cancel_processing)
        self.open_folder_button.clicked.connect(self.open_output_folder)
        self.new_task_button.clicked.connect(self.start_new_task)
        return action_bar

    def _apply_style(self):
        self.setStyleSheet(APP_STYLESHEET)

    def _refresh_output_controls(self):
        policy = self._output_naming_policy
        if policy.is_default:
            self.output_naming_button.setText("이름 형식 · 원본")
            self.output_naming_button.setToolTip(
                "기본: 원본 이름.png · 같은 이름이 있으면 _no_bg 추가"
            )
        else:
            self.output_naming_button.setText("이름 형식 · 사용자 지정")
            self.output_naming_button.setToolTip(
                f"{policy.prefix}[원본 이름]{policy.suffix}.png"
            )
        if self._output_directory is None:
            self.output_note.setText("저장 · 위치를 먼저 지정하세요")
            self.output_folder_button.setText("저장 위치 지정")
            self.output_folder_button.setToolTip(
                "각 이미지가 끝날 때마다 저장할 폴더를 지정합니다"
            )
            return

        directory = self._output_directory
        short_name = directory.name or str(directory)
        self.output_note.setText("자동 저장 · 이미지별 즉시 저장")
        self.output_folder_button.setText(f"저장 위치 · {short_name}")
        self.output_folder_button.setToolTip(str(directory))

    def open_output_naming_dialog(self):
        if self._processing:
            return
        dialog = OutputNamingDialog(self._output_naming_policy, self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        self._output_naming_policy = dialog.policy
        self._settings.setValue(
            OUTPUT_NAMING_PREFIX_SETTING,
            self._output_naming_policy.prefix,
        )
        self._settings.setValue(
            OUTPUT_NAMING_SUFFIX_SETTING,
            self._output_naming_policy.suffix,
        )
        self._settings.sync()
        self._refresh_output_controls()
        self.status_label.setText("출력 이미지 이름 형식을 변경했습니다")

    def choose_output_folder(self) -> bool:
        if self._processing:
            return False
        start_directory = str(self._output_directory or Path.home())
        selected = QFileDialog.getExistingDirectory(
            self,
            "결과 저장 위치 지정",
            start_directory,
        )
        if not selected:
            return False
        try:
            directory = ensure_writable_directory(selected)
        except OSError as error:
            QMessageBox.warning(
                self,
                "저장 폴더를 사용할 수 없습니다",
                "선택한 폴더에 파일을 쓸 수 없습니다. 다른 위치를 "
                f"선택하세요.\n\n세부 정보: {error}",
            )
            return False

        self._output_directory = directory
        self._refresh_output_controls()
        self.status_label.setText(
            "각 이미지가 끝날 때마다 지정한 위치에 저장합니다"
        )
        return True

    def _select_performance_combo(self, mode: PerformanceMode):
        index = self.performance_combo.findData(mode.value)
        self.performance_combo.blockSignals(True)
        self.performance_combo.setCurrentIndex(max(0, index))
        self.performance_combo.blockSignals(False)

    def _refresh_performance_tooltip(self):
        policy = self._performance_policy
        self.performance_combo.setToolTip(
            f"CPU {policy.cpu_threads}개 스레드 · GPU 메모리 상한 "
            f"{policy.gpu_memory_fraction:.0%} · 이미지 사이 "
            f"{policy.cooldown_ms}ms 휴식\n"
            "GPU 사용률을 정확한 퍼센트로 고정하는 기능은 아닙니다."
        )

    def _on_performance_mode_selected(self, index: int):
        if index < 0 or self._processing:
            return
        try:
            mode = PerformanceMode(self.performance_combo.itemData(index))
        except (TypeError, ValueError):
            return

        previous_mode = self._performance_policy.mode
        if mode == PerformanceMode.CUSTOM:
            dialog = PerformanceDialog(self._performance_policy, self)
            if dialog.exec() != QDialog.DialogCode.Accepted:
                self._select_performance_combo(previous_mode)
                return
            self._performance_policy = dialog.policy
        else:
            self._performance_policy = preset_policy(mode)

        self._worker.set_performance_policy(self._performance_policy)
        self._save_performance_policy()
        self._select_performance_combo(self._performance_policy.mode)
        self._refresh_performance_tooltip()
        self.status_label.setText(
            f"다음 작업에 {MODE_LABELS[self._performance_policy.mode]} "
            "성능 모드를 사용합니다"
        )

    def choose_images(self):
        was_processing = self._processing
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "배경을 제거할 이미지 선택",
            "",
            "이미지 파일 (*.png *.jpg *.jpeg *.webp)",
        )
        self.add_paths(paths)
        self._continue_after_modal_add(was_processing)

    def choose_folder(self):
        was_processing = self._processing
        folder = QFileDialog.getExistingDirectory(
            self,
            "이미지가 들어 있는 폴더 선택",
        )
        if folder:
            self.add_paths([folder])
        self._continue_after_modal_add(was_processing)

    def _continue_after_modal_add(self, was_processing: bool):
        """파일 대화상자가 열린 사이 배치가 끝난 경우에도 자동 재개한다."""
        if (
            not was_processing
            or self._processing
            or not self._auto_continue_allowed
        ):
            return
        pending = [
            item
            for item in self.selected_items()
            if item.state == ItemState.QUEUED
        ]
        if pending:
            self.start_processing(pending)

    def _expand_paths(self, raw_paths: list[str]) -> tuple[list[Path], int]:
        expanded: list[Path] = []
        rejected = 0
        for raw_path in raw_paths:
            path = Path(raw_path)
            if path.is_dir():
                try:
                    children = sorted(
                        path.iterdir(),
                        key=lambda entry: entry.name.casefold(),
                    )
                except OSError:
                    rejected += 1
                    continue
                for child in children:
                    try:
                        if child.is_file():
                            expanded.append(child)
                    except OSError:
                        rejected += 1
            elif path.is_file():
                expanded.append(path)
            else:
                rejected += 1
        return expanded, rejected

    def _validate_image(self, path: Path) -> str | None:
        if path.suffix.casefold() not in SUPPORTED_SUFFIXES:
            return "지원하지 않는 형식"

        reader = QImageReader(str(path))
        reader.setAutoTransform(True)
        if not reader.canRead():
            return "이미지 파일을 읽을 수 없음"

        size = reader.size()
        if size.isValid() and size.width() * size.height() > MAX_IMAGE_PIXELS:
            return "이미지 크기가 너무 큼"
        return None

    def add_paths(self, raw_paths: list[str]):
        if not raw_paths:
            return

        existing_paths = {
            str(Path(item.source_path).resolve()).casefold()
            for item in self._items
        }
        added_count = 0
        expanded_paths, rejected_count = self._expand_paths(raw_paths)
        added_list_items: list[QListWidgetItem] = []

        for path in expanded_paths:
            try:
                resolved = str(path.resolve(strict=True))
            except OSError:
                rejected_count += 1
                continue
            if resolved.casefold() in existing_paths:
                continue

            validation_error = self._validate_image(path)
            if validation_error is not None:
                rejected_count += 1
                continue

            batch_item = BatchItem(
                item_id=uuid.uuid4().hex,
                source_path=resolved,
            )
            self._items.append(batch_item)
            self._items_by_id[batch_item.item_id] = batch_item
            existing_paths.add(resolved.casefold())

            list_item = QListWidgetItem(QIcon(resolved), "")
            list_item.setData(Qt.ItemDataRole.UserRole, batch_item.item_id)
            list_item.setToolTip(resolved)
            self.file_list.addItem(list_item)
            added_list_items.append(list_item)
            self._refresh_list_item(batch_item)
            added_count += 1

        if added_list_items:
            first_index = self.file_list.indexFromItem(added_list_items[0])
            last_index = self.file_list.indexFromItem(added_list_items[-1])
            selection = QItemSelection(first_index, last_index)
            selection_model = self.file_list.selectionModel()
            selection_model.select(
                selection,
                QItemSelectionModel.SelectionFlag.Select,
            )
            selection_model.setCurrentIndex(
                first_index,
                QItemSelectionModel.SelectionFlag.NoUpdate,
            )

        if self._processing:
            if added_count and rejected_count:
                self.status_label.setText(
                    f"대기열에 {added_count}개 추가 · 읽을 수 없는 파일 "
                    f"{rejected_count}개 제외 · 현재 작업 후 이어서 처리합니다"
                )
            elif added_count:
                self.status_label.setText(
                    f"대기열에 {added_count}개 추가 · 현재 작업 후 이어서 처리합니다"
                )
            elif rejected_count:
                self.status_label.setText(
                    f"읽을 수 없는 파일 {rejected_count}개 제외 · 현재 작업은 계속합니다"
                )
        elif rejected_count:
            self.status_label.setText(
                f"{added_count}개 추가 · 읽을 수 없는 파일 {rejected_count}개 제외"
            )
        elif added_count:
            self.status_label.setText(f"이미지 {added_count}개를 추가했습니다")
        else:
            self.status_label.setText("새로 추가할 이미지가 없습니다")

        if added_count:
            self.tsunao.image_loaded()
        self._update_interface_state(keep_status=True)

    def _list_widget_item(self, item_id: str) -> QListWidgetItem | None:
        for row in range(self.file_list.count()):
            list_item = self.file_list.item(row)
            if list_item.data(Qt.ItemDataRole.UserRole) == item_id:
                return list_item
        return None

    def _refresh_list_item(self, item: BatchItem):
        list_item = self._list_widget_item(item.item_id)
        if list_item is None:
            return

        symbol, state_text = STATE_PRESENTATION[item.state]
        list_item.setText(f"{symbol} {item.filename}\n{state_text}")
        icon_path = item.result_path or item.source_path
        list_item.setIcon(QIcon(icon_path))

    def selected_item(self) -> BatchItem | None:
        list_item = self.file_list.currentItem()
        if list_item is None:
            return None
        item_id = list_item.data(Qt.ItemDataRole.UserRole)
        return self._items_by_id.get(item_id)

    def selected_items(self) -> list[BatchItem]:
        selected_ids = {
            list_item.data(Qt.ItemDataRole.UserRole)
            for list_item in self.file_list.selectedItems()
        }
        return [
            item for item in self._items if item.item_id in selected_ids
        ]

    def select_all_items(self):
        if not self._items:
            return
        self.file_list.selectAll()
        if self.file_list.currentRow() < 0:
            first_index = self.file_list.model().index(0, 0)
            self.file_list.selectionModel().setCurrentIndex(
                first_index,
                QItemSelectionModel.SelectionFlag.NoUpdate,
            )
        self.file_list.setFocus()

    def show_selected_preview(self, row: int):
        item = self.selected_item()
        if item is None:
            self.preview_title.setText("미리보기")
            self.preview.show_message(
                "목록에서 이미지를 선택하세요"
                if self._items
                else "이미지를 추가하면 이곳에서 결과를 확인할 수 있습니다"
            )
            self._update_selection_actions()
            return

        original = QPixmap(item.source_path)
        result = QPixmap(item.result_path) if item.result_path else QPixmap()
        if original.isNull():
            self.preview.show_message("원본 이미지를 불러올 수 없습니다")
        else:
            self.preview.set_images(original, result)

        self.preview_title.setText(item.filename)
        has_result = not result.isNull()
        self.result_mode_button.setEnabled(has_result)
        self.compare_mode_button.setEnabled(has_result)

        if not has_result:
            self.original_mode_button.setChecked(True)
            self.set_preview_mode("original")
        elif self.compare_mode_button.isChecked():
            self.set_preview_mode("compare")
        elif self.result_mode_button.isChecked():
            self.set_preview_mode("result")
        else:
            self.set_preview_mode("original")

        if item.error:
            self.preview_error_label.setText(item.error)
            self.preview_error_label.show()
        else:
            self.preview_error_label.hide()

        self._update_selection_actions()

    def set_preview_mode(self, mode: str):
        if mode in {"result", "compare"} and not self.preview.has_result():
            mode = "original"
            self.original_mode_button.setChecked(True)

        self.preview.set_mode(mode)
        self.compare_slider.setVisible(mode == "compare")

    def _sync_queue_layout(self, *_):
        list_width = self.file_list.width()
        if list_width >= LARGE_THUMBNAIL_MIN_WIDTH:
            layout_mode = "large"
        elif list_width >= SMALL_THUMBNAIL_MIN_WIDTH:
            layout_mode = "small"
        else:
            layout_mode = "list"

        if getattr(self, "_queue_layout_mode", None) == layout_mode:
            return
        self._queue_layout_mode = layout_mode

        if layout_mode == "large":
            self.file_list.setViewMode(QListView.ViewMode.IconMode)
            self.file_list.setIconSize(QSize(160, 160))
            self.file_list.setGridSize(QSize(195, 220))
            self.file_list.setSpacing(8)
            self.file_list.setWrapping(True)
        elif layout_mode == "small":
            self.file_list.setViewMode(QListView.ViewMode.IconMode)
            self.file_list.setIconSize(QSize(80, 80))
            self.file_list.setGridSize(QSize(115, 138))
            self.file_list.setSpacing(4)
            self.file_list.setWrapping(True)
        else:
            self.file_list.setViewMode(QListView.ViewMode.ListMode)
            self.file_list.setIconSize(QSize(42, 42))
            self.file_list.setGridSize(QSize())
            self.file_list.setSpacing(2)
            self.file_list.setWrapping(False)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, "file_list"):
            QTimer.singleShot(0, self._sync_queue_layout)
        if hasattr(self, "tsunao"):
            QTimer.singleShot(0, self.tsunao.parent_resized)

    def _update_selection_actions(self):
        selected = self.selected_items()
        selected_count = len(selected)
        retryable_count = sum(
            item.state in {ItemState.FAILED, ItemState.CANCELLED}
            for item in selected
        )
        can_edit = bool(selected) and not self._processing

        if self._items:
            self.batch_count_label.setText(
                f"{len(self._items)}개 이미지 · {selected_count}개 선택"
            )
        else:
            self.batch_count_label.setText("0개 이미지")

        self.remove_button.setText(
            f"목록에서 제거 ({selected_count})"
            if selected_count > 1
            else "목록에서 제거"
        )
        self.remove_button.setEnabled(can_edit)
        self.retry_button.setText(
            f"다시 시도 ({retryable_count})"
            if retryable_count > 1
            else "다시 시도"
        )
        self.retry_button.setEnabled(
            can_edit and retryable_count > 0
        )
        self.select_all_button.setEnabled(bool(self._items))
        self._update_action_state()

    def remove_selected(self):
        if self._processing:
            return

        selected = self.selected_items()
        if not selected:
            return

        selected_ids = {item.item_id for item in selected}
        for row in reversed(range(self.file_list.count())):
            list_item = self.file_list.item(row)
            if (
                list_item.data(Qt.ItemDataRole.UserRole)
                in selected_ids
            ):
                self.file_list.takeItem(row)

        self._items = [
            item for item in self._items if item.item_id not in selected_ids
        ]
        for item_id in selected_ids:
            self._items_by_id.pop(item_id, None)
        self._update_interface_state()

    def retry_selected(self):
        if self._processing:
            return
        retryable = [
            item
            for item in self.selected_items()
            if item.state in {ItemState.FAILED, ItemState.CANCELLED}
        ]
        if not retryable:
            return

        for item in retryable:
            item.state = ItemState.QUEUED
            item.error = None
            self._refresh_list_item(item)
        self.start_processing(retryable)

    def clear_all(self):
        if self._processing or not self._items:
            return

        if any(
            item.state == ItemState.COMPLETED and not item.result_saved
            for item in self._items
        ):
            choice = QMessageBox.question(
                self,
                "목록 비우기",
                "저장하지 않은 결과가 있을 수 있습니다. 목록을 비울까요?",
            )
            if choice != QMessageBox.StandardButton.Yes:
                return

        self._clear_items()

    def _clear_items(self):
        self.file_list.clear()
        self._items.clear()
        self._items_by_id.clear()
        self._last_output_folder = None
        self._active_job_ids.clear()
        self.preview.show_message(
            "이미지를 추가하면 이곳에서 결과를 확인할 수 있습니다"
        )
        self.preview_title.setText("미리보기")
        self.preview_error_label.hide()
        self._update_interface_state()

    def handle_primary_action(self):
        selected = self.selected_items()
        queued = [
            item
            for item in selected
            if item.state == ItemState.QUEUED
        ]
        retryable = [
            item
            for item in selected
            if item.state in {ItemState.FAILED, ItemState.CANCELLED}
        ]

        if queued or retryable:
            self.start_processing([*queued, *retryable])

    def start_processing(self, items: list[BatchItem]):
        if self._processing or not items:
            return

        if self._output_directory is None:
            if not self.choose_output_folder():
                self.status_label.setText(
                    "배경 제거를 시작하려면 저장 위치를 지정하세요"
                )
                return

        if not model_is_installed(self._model_directory):
            if not self.show_model_installer():
                self.status_label.setText(
                    "배경 제거를 시작하려면 모델 설치가 필요합니다"
                )
                return

        try:
            result_folder = ensure_writable_directory(
                self._output_directory
            )
        except OSError as error:
            self._output_directory = None
            self._refresh_output_controls()
            self.status_label.setText("저장 위치를 다시 지정해야 합니다")
            QMessageBox.warning(
                self,
                "저장 위치를 사용할 수 없습니다",
                "지정한 폴더에 파일을 쓸 수 없습니다. 저장 위치를 다시 "
                f"지정하세요.\n\n{error}",
            )
            return

        cleanup_abandoned_output_files(result_folder)

        jobs = []
        reserved_paths: set[Path] = set()
        self._active_auto_save_ids.clear()
        self._active_job_ids = {item.item_id for item in items}
        for item in items:
            item.state = ItemState.QUEUED
            item.error = None
            item.result_path = None
            item.result_saved = False
            output_path = available_output_path(
                result_folder,
                item.source_path,
                reserved_paths,
                naming_policy=self._output_naming_policy,
            )
            reserved_paths.add(output_path)
            self._active_auto_save_ids.add(item.item_id)
            jobs.append((item.item_id, item.source_path, str(output_path)))
            self._refresh_list_item(item)

        self._worker.set_performance_policy(self._performance_policy)
        self._worker.set_jobs(jobs)
        self._processing = True
        self.tsunao.processing_started()
        self._auto_continue_allowed = True
        self._pause_requested = False
        self._paused = False
        self._set_input_controls_enabled(False)
        self.progress_bar.setRange(0, 0)
        self.progress_bar.show()
        self.cancel_button.show()
        self.cancel_button.setEnabled(True)
        self.pause_button.setText("일시정지")
        self.pause_button.setEnabled(True)
        self.pause_button.show()
        self.status_label.setText("모델을 준비하고 있습니다")
        self._update_action_state()
        self._worker.start()

    def cancel_processing(self):
        if not self._processing:
            return
        self._auto_continue_allowed = False
        self._worker.request_cancel()
        self.pause_button.setEnabled(False)
        self.cancel_button.setEnabled(False)
        self.status_label.setText("현재 이미지를 마친 뒤 중지합니다")

    def toggle_pause(self):
        if not self._processing:
            return
        if self._pause_requested or self._paused:
            self._worker.request_resume()
            self._pause_requested = False
            self._paused = False
            self.pause_button.setText("일시정지")
            self.status_label.setText("작업을 재개합니다")
            return

        self._worker.request_pause()
        self._pause_requested = True
        self.pause_button.setText("재개")
        self.status_label.setText("현재 이미지를 마친 뒤 일시정지합니다")

    def _on_pause_reached(self):
        if not self._processing or not self._pause_requested:
            return
        self._paused = True
        self.status_label.setText("일시정지됨 · 재개를 눌러 계속하세요")

    def _on_model_status(self, message: str):
        if self._model_operation == "install":
            if self._model_install_dialog is not None:
                self._model_install_dialog.set_status(message)
            return
        self.progress_bar.setRange(0, 0)
        self.status_label.setText(message)

    def _on_model_ready(self, device_label: str):
        if self._model_operation == "install":
            if self._model_install_dialog is not None:
                self._model_install_dialog.set_status(
                    f"모델 파일 확인 완료 · {device_label} 준비 중"
                )
            return
        self.progress_bar.setRange(0, max(1, self._worker.job_count))
        self.progress_bar.setValue(0)
        self.status_label.setText(f"모델 준비 완료 · {device_label} 사용")

    def _on_model_failed(self, error: str):
        if self._gpu_enabled and any(
            word in error.lower()
            for word in ("cuda", "nvidia", "gpu worker")
        ):
            previous = self._acceleration_info
            self._on_acceleration_detected(
                AccelerationInfo(
                    mode=AccelerationMode.GPU_UNAVAILABLE,
                    device=previous.device if previous else None,
                    runtime=previous.runtime if previous else None,
                    runtime_size=previous.runtime_size if previous else 0,
                    runtime_directory=str(self._gpu_runtime_directory),
                )
            )
        if self._model_operation == "install":
            self._model_operation = "idle"
            self._refresh_model_status()
            if self._model_install_dialog is not None:
                self._model_install_dialog.show_failure(error)
            return

        for item in self._items:
            if (
                item.item_id in self._active_job_ids
                and item.state in {ItemState.QUEUED, ItemState.PROCESSING}
            ):
                item.state = ItemState.FAILED
                item.error = error
                self._refresh_list_item(item)
        self._processing = False
        self.tsunao.processing_finished(completed=False)
        self._auto_continue_allowed = False
        self._active_job_ids.clear()
        self._active_auto_save_ids.clear()
        self._set_input_controls_enabled(True)
        self.progress_bar.hide()
        self.pause_button.hide()
        self.cancel_button.hide()
        self.status_label.setText("모델을 준비하지 못했습니다")
        self._update_interface_state(keep_status=True)

        message = (
            "모델 파일을 준비하지 못했습니다.\n"
            "인터넷 연결과 저장 공간을 확인하거나, 오른쪽 위 모델 상태 "
            "버튼에서 다른 폴더를 선택한 뒤 다시 시도하세요.\n\n"
            f"현재 위치: {self._model_directory}\n"
            f"세부 정보: {error}"
        )
        QMessageBox.warning(self, "모델 준비 실패", message)
        if self._close_after_worker:
            self.close()

    def _on_item_started(self, item_id: str):
        item = self._items_by_id.get(item_id)
        if item is None:
            return
        item.state = ItemState.PROCESSING
        self.status_label.setText(f"처리 중 · {item.filename}")
        self._refresh_list_item(item)

    def _on_item_completed(self, item_id: str, result_path: str):
        item = self._items_by_id.get(item_id)
        if item is None:
            return

        item.state = ItemState.COMPLETED
        item.result_path = result_path
        item.result_saved = item_id in self._active_auto_save_ids
        item.error = None
        if item.result_saved:
            self._last_output_folder = str(Path(result_path).parent)
        self._refresh_list_item(item)

        if self.selected_item() is item:
            self.result_mode_button.setChecked(True)
            self.show_selected_preview(self.file_list.currentRow())
            self.set_preview_mode("result")

    def _on_item_failed(self, item_id: str, error: str):
        item = self._items_by_id.get(item_id)
        if item is None:
            return

        item.state = ItemState.FAILED
        item.error = error
        self._refresh_list_item(item)
        if self.selected_item() is item:
            self.show_selected_preview(self.file_list.currentRow())

    def _on_progress_changed(self, completed: int, total: int):
        self.progress_bar.setRange(0, total)
        self.progress_bar.setValue(completed)
        self.progress_bar.setFormat(f"{completed} / {total}")

    def _on_batch_finished(self, cancelled: bool):
        if self._model_operation == "install":
            self._model_operation = "idle"
            self._refresh_model_status()
            if self._model_install_dialog is not None:
                self._model_install_dialog.show_success()
            return

        batch_had_completion = not cancelled and any(
            item.item_id in self._active_job_ids
            and item.state == ItemState.COMPLETED
            for item in self._items
        )
        self._processing = False
        self.tsunao.processing_finished(completed=batch_had_completion)
        self._auto_continue_allowed = not cancelled
        self._pause_requested = False
        self._paused = False
        self._set_input_controls_enabled(True)
        self.pause_button.hide()
        self.cancel_button.hide()

        if cancelled:
            for item in self._items:
                if (
                    item.item_id in self._active_job_ids
                    and item.state == ItemState.QUEUED
                ):
                    item.state = ItemState.CANCELLED
                    self._refresh_list_item(item)
            pending_count = sum(
                item.state == ItemState.QUEUED for item in self._items
            )
            if pending_count:
                self.status_label.setText(
                    "작업을 중지했습니다 · 완료된 결과는 유지됩니다 · "
                    f"새로 추가한 {pending_count}개는 대기 중입니다"
                )
            else:
                self.status_label.setText(
                    "작업을 중지했습니다 · 완료된 결과는 유지됩니다"
                )
        else:
            completed = sum(
                item.state == ItemState.COMPLETED for item in self._items
            )
            failed = sum(item.state == ItemState.FAILED for item in self._items)
            queued_items = [
                item
                for item in self._items
                if item.state == ItemState.QUEUED
            ]
            selected_ids = {
                item.item_id for item in self.selected_items()
            }
            pending_items = [
                item
                for item in queued_items
                if item.item_id in selected_ids
            ]
            if pending_items and not self._close_after_worker:
                self.status_label.setText(
                    f"현재 작업 완료 · 선택된 {len(pending_items)}개를 "
                    "이어서 처리합니다"
                )
            elif queued_items and failed:
                self.status_label.setText(
                    f"{completed}개 완료 · {failed}개 실패 · "
                    f"선택하지 않은 {len(queued_items)}개는 대기 중입니다"
                )
            elif queued_items:
                self.status_label.setText(
                    f"{completed}개 완료 · 선택하지 않은 "
                    f"{len(queued_items)}개는 대기 중입니다"
                )
            elif failed:
                self.status_label.setText(
                    f"{completed}개 완료 · {failed}개 실패 · 실패 항목을 선택해 다시 시도하세요"
                )
            else:
                all_saved = all(
                    item.result_saved
                    for item in self._items
                    if item.state == ItemState.COMPLETED
                )
                if completed and all_saved:
                    self.status_label.setText(
                        f"{completed}개 이미지 처리 및 저장이 완료되었습니다"
                    )
                else:
                    self.status_label.setText(
                        f"{completed}개 이미지 처리가 완료되었습니다"
                    )

        pending_to_continue = (
            [
                item
                for item in self.selected_items()
                if item.state == ItemState.QUEUED
            ]
            if not cancelled and not self._close_after_worker
            else []
        )
        self._active_job_ids.clear()
        self._active_auto_save_ids.clear()
        self._update_interface_state(keep_status=True)
        if self._close_after_worker:
            self.close()
        elif pending_to_continue:
            QTimer.singleShot(
                0,
                lambda pending=pending_to_continue: self.start_processing(
                    [
                        item
                        for item in pending
                        if item.state == ItemState.QUEUED
                    ]
                ),
            )

    def open_output_folder(self):
        if self._last_output_folder:
            QDesktopServices.openUrl(
                QUrl.fromLocalFile(self._last_output_folder)
            )

    def start_new_task(self):
        if self._processing:
            return
        self._clear_items()

    def _set_input_controls_enabled(self, enabled: bool):
        # 파일과 폴더 추가는 worker가 실행 중이어도 계속 받을 수 있다.
        # 새 항목은 현재 worker의 고정 작업 목록과 분리된 QUEUED 상태가 되고,
        # batch_finished에서 다음 배치로 자동 이어진다.
        self.header_add_button.setEnabled(True)
        self.header_folder_button.setEnabled(True)
        for control in (
            self.model_status_button,
            self.clear_button,
            self.output_folder_button,
            self.output_naming_button,
            self.performance_combo,
        ):
            control.setEnabled(enabled)
        self.acceleration_button.setEnabled(
            enabled
            and (
                self._acceleration_info is not None
                or self._acceleration_error is not None
            )
        )
        self._refresh_update_button()
        self._update_selection_actions()

    def _update_action_state(self):
        selected = self.selected_items()
        queued_count = sum(
            item.state == ItemState.QUEUED for item in selected
        )
        retryable_count = sum(
            item.state in {ItemState.FAILED, ItemState.CANCELLED}
            for item in selected
        )
        saved_count = sum(
            item.state == ItemState.COMPLETED and item.result_saved
            for item in self._items
        )

        if self._gpu_runtime_thread is not None:
            self.primary_button.setText("GPU 팩 설치 중")
            self.primary_button.setEnabled(False)
        elif self._model_operation == "install":
            self.primary_button.setText("모델 설치 중")
            self.primary_button.setEnabled(False)
        elif self._processing:
            self.primary_button.setText("처리 중")
            self.primary_button.setEnabled(False)
        elif queued_count:
            process_count = queued_count + retryable_count
            self.primary_button.setText(
                f"배경 제거 시작 ({process_count})"
            )
            self.primary_button.setEnabled(True)
        elif retryable_count:
            self.primary_button.setText(
                f"선택 항목 다시 시도 ({retryable_count})"
            )
            self.primary_button.setEnabled(True)
        elif self._items and not selected:
            self.primary_button.setText("처리할 이미지 선택")
            self.primary_button.setEnabled(False)
        elif any(item.state == ItemState.QUEUED for item in self._items):
            self.primary_button.setText("처리할 이미지 선택")
            self.primary_button.setEnabled(False)
        elif saved_count:
            self.primary_button.setText("모두 저장됨")
            self.primary_button.setEnabled(False)
        else:
            self.primary_button.setText("배경 제거 시작")
            self.primary_button.setEnabled(False)

    def _update_interface_state(self, keep_status: bool = False):
        has_items = bool(self._items)
        self.action_bar.show()
        self.header_add_button.show()
        self.header_folder_button.show()
        if not keep_status:
            if has_items:
                self.status_label.setText(
                    "배경을 제거할 이미지를 선택하세요"
                )
            else:
                self.status_label.setText(
                    "창 어디에나 이미지를 놓거나 이미지 추가를 누르세요"
                )

        self.open_folder_button.setVisible(
            bool(self._last_output_folder) and has_items
        )
        self.new_task_button.setVisible(
            bool(self._last_output_folder) and has_items
        )
        self._update_action_state()
        self._update_selection_actions()

    def closeEvent(self, event):
        if self._gpu_runtime_thread is not None and self._gpu_runtime_thread.isRunning():
            if self._gpu_runtime_action == "install":
                choice = QMessageBox.question(
                    self,
                    "GPU 가속 팩 설치 중",
                    "GPU 가속 팩 설치를 취소하고 ToonOut을 종료할까요?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.No,
                )
                if choice == QMessageBox.StandardButton.Yes:
                    self._close_after_worker = True
                    self._cancel_gpu_runtime_operation()
            else:
                QMessageBox.information(
                    self,
                    "GPU 가속 팩 작업 중",
                    "GPU 가속 팩 삭제가 끝난 뒤 종료할 수 있습니다.",
                )
            event.ignore()
            return

        if self._model_file_thread is not None and self._model_file_thread.isRunning():
            QMessageBox.information(
                self,
                "모델 파일 작업 중",
                "모델 파일 이동 또는 삭제가 끝난 뒤 종료할 수 있습니다.",
            )
            event.ignore()
            return

        if (
            self._model_install_process is not None
            and self._model_install_process.state()
            != QProcess.ProcessState.NotRunning
        ):
            choice = QMessageBox.question(
                self,
                "모델 설치 중",
                "모델 설치를 취소하고 ToonOut을 종료할까요?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if choice == QMessageBox.StandardButton.Yes:
                self._close_after_worker = True
                self._cancel_model_install()
            event.ignore()
            return

        if self._worker.isRunning():
            if self._model_operation == "install":
                QMessageBox.information(
                    self,
                    "모델 설치 중",
                    "모델 설치가 끝난 뒤 종료할 수 있습니다.",
                )
                event.ignore()
                return
            choice = QMessageBox.question(
                self,
                "처리 중인 작업",
                "현재 이미지를 마친 뒤 ToonOut을 종료할까요?",
            )
            if choice == QMessageBox.StandardButton.Yes:
                self._close_after_worker = True
                self.cancel_processing()
            event.ignore()
            return

        if (
            self._acceleration_thread is not None
            and self._acceleration_thread.isRunning()
            and not self._acceleration_thread.wait(10_000)
        ):
            event.ignore()
            return

        for thread in (
            self._update_check_thread,
            self._update_download_thread,
        ):
            if thread is not None and thread.isRunning():
                thread.requestInterruption()
                if not thread.wait(15_000):
                    event.ignore()
                    return

        application = QApplication.instance()
        if application is not None:
            application.removeEventFilter(self)
        self.tsunao.shutdown()
        self._output_directory = None
        self._settings.remove(LEGACY_OUTPUT_DIRECTORY_SETTING)
        self._settings.sync()
        event.accept()


def parse_app_arguments(arguments: list[str]):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--model-directory")
    parser.add_argument("--model-install-worker")
    parser.add_argument("--model-cleanup-worker")
    parser.add_argument("--model-install-status")
    parser.add_argument("--open-model-installer", action="store_true")
    parser.add_argument("--restore-image", action="append", default=[])
    parser.add_argument("--cleanup-update-cache", action="store_true")
    parser.add_argument("--parent-pid", type=int)
    return parser.parse_known_args(arguments)


if __name__ == "__main__":
    app_arguments, qt_arguments = parse_app_arguments(sys.argv[1:])
    if app_arguments.parent_pid:
        start_parent_exit_watchdog(app_arguments.parent_pid)
    if app_arguments.model_install_worker:
        if not app_arguments.model_install_status:
            raise SystemExit(2)
        raise SystemExit(
            run_model_install_worker(
                app_arguments.model_install_worker,
                app_arguments.model_install_status,
            )
        )
    if app_arguments.model_cleanup_worker:
        raise SystemExit(
            run_model_cleanup_worker(app_arguments.model_cleanup_worker)
        )
    # Qt Widgets는 Windows의 125%·150%·175% 같은 분수 배율에서 글자와
    # 테두리 픽셀을 서로 다르게 반올림할 수 있다. 앱 생성 전에 정수 배율로
    # 맞춰 글꼴과 위젯 좌표가 같은 픽셀 격자를 사용하게 한다.
    QGuiApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.Round
    )
    app = QApplication([sys.argv[0], *qt_arguments])
    app.setStyle("Fusion")
    configure_application_font(app)
    app_icon = QIcon(str(application_icon_path()))
    if not app_icon.isNull():
        app.setWindowIcon(app_icon)
    window = MainWindow(
        app_arguments.model_directory,
        cleanup_update_cache=app_arguments.cleanup_update_cache,
    )
    window.show()
    if app_arguments.restore_image:
        QTimer.singleShot(
            0,
            lambda: window.add_paths(app_arguments.restore_image),
        )
    if app_arguments.open_model_installer:
        QTimer.singleShot(0, window.open_model_status)
    sys.exit(app.exec())
