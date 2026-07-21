"""ToonOut에서 재사용하는 작은 UI 위젯."""

import ctypes
import os
from pathlib import Path

from PySide6.QtCore import (
    QAbstractAnimation,
    QEasingCurve,
    QPointF,
    QPropertyAnimation,
    QSize,
    Qt,
    Signal,
)
from PySide6.QtGui import QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListView,
    QListWidget,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QStyle,
    QStyledItemDelegate,
    QVBoxLayout,
    QWidget,
)

from app_settings import (
    available_storage_bytes,
    ensure_writable_model_directory,
    format_storage_size,
    is_permission_error,
)
from app_core import (
    MAX_OUTPUT_AFFIX_LENGTH,
    OutputNamingPolicy,
    validate_output_affix,
)
from acceleration import AccelerationInfo, AccelerationMode
from gpu_runtime import gpu_runtime_is_compatible
from performance import PerformanceMode, PerformancePolicy


def paths_from_drop_event(event) -> list[str]:
    return [url.toLocalFile() for url in event.mimeData().urls()]


def _client_animations_enabled() -> bool:
    """Windows의 '애니메이션 효과' 접근성 설정을 따른다."""
    if os.name != "nt":
        return True
    try:
        enabled = ctypes.c_int()
        success = ctypes.windll.user32.SystemParametersInfoW(
            0x1042,  # SPI_GETCLIENTAREAANIMATION
            0,
            ctypes.byref(enabled),
            0,
        )
        return bool(enabled.value) if success else True
    except (AttributeError, OSError):
        return True


class ComboItemDelegate(QStyledItemDelegate):
    """선택 텍스트와 현재 적용 상태가 항상 보이는 메뉴 행."""

    def __init__(self, combo: QComboBox):
        super().__init__(combo)
        self._combo = combo

    def sizeHint(self, option, index) -> QSize:
        size = super().sizeHint(option, index)
        size.setHeight(38)
        return size

    def paint(self, painter: QPainter, option, index):
        painter.save()
        row_rect = option.rect.adjusted(5, 2, -5, -2)
        highlighted = bool(
            option.state & QStyle.StateFlag.State_Selected
        )
        hovered = bool(option.state & QStyle.StateFlag.State_MouseOver)
        current = index.row() == self._combo.currentIndex()

        if highlighted:
            painter.setBrush(QColor("#e9eaff"))
        elif hovered:
            painter.setBrush(QColor("#f7f7ff"))
        else:
            painter.setBrush(QColor("#ffffff"))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRoundedRect(row_rect, 7, 7)

        font = QFont(option.font)
        if current:
            font.setWeight(QFont.Weight.DemiBold)
        painter.setFont(font)
        painter.setPen(QColor("#252b3a"))
        text_rect = row_rect.adjusted(11, 0, -34, 0)
        painter.drawText(
            text_rect,
            Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
            str(index.data(Qt.ItemDataRole.DisplayRole)),
        )

        if current:
            check_pen = QPen(QColor("#5054dc"))
            check_pen.setWidthF(1.8)
            check_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            check_pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
            painter.setPen(check_pen)
            check_x = row_rect.right() - 19
            check_y = row_rect.center().y()
            painter.drawLine(
                QPointF(check_x - 3, check_y),
                QPointF(check_x - 1, check_y + 2),
            )
            painter.drawLine(
                QPointF(check_x - 1, check_y + 2),
                QPointF(check_x + 4, check_y - 3),
            )
        painter.restore()


class ToonOutComboBox(QComboBox):
    """ToonOut의 선택 상태와 팝업 폭을 일관되게 보여주는 콤보박스."""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        popup_view = QListView(self)
        popup_view.setObjectName("comboPopup")
        popup_view.setMouseTracking(True)
        popup_view.setUniformItemSizes(True)
        popup_view.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.setView(popup_view)
        self.setItemDelegate(ComboItemDelegate(self))

        popup = popup_view.window()
        popup.setObjectName("comboPopupContainer")
        popup.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)

    def showPopup(self):
        metrics = self.fontMetrics()
        content_width = max(
            (
                metrics.horizontalAdvance(self.itemText(index))
                for index in range(self.count())
            ),
            default=0,
        ) + 62
        popup_width = max(self.width(), content_width)
        self.view().setMinimumWidth(popup_width)
        self.view().window().setMinimumWidth(popup_width)
        super().showPopup()

    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        arrow_pen = QPen(
            QColor("#737b8c" if self.isEnabled() else "#adb2bf")
        )
        arrow_pen.setWidthF(1.6)
        arrow_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        arrow_pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        painter.setPen(arrow_pen)
        center_x = self.width() - 15
        center_y = self.height() / 2
        painter.drawLine(
            QPointF(center_x - 4, center_y - 2),
            QPointF(center_x, center_y + 2),
        )
        painter.drawLine(
            QPointF(center_x, center_y + 2),
            QPointF(center_x + 4, center_y - 2),
        )


class ImageListWidget(QListWidget):
    files_dropped = Signal(list)

    def __init__(self):
        super().__init__()
        self.setAcceptDrops(True)
        self.setDragDropMode(QAbstractItemView.DragDropMode.DropOnly)
        self.setVerticalScrollMode(
            QAbstractItemView.ScrollMode.ScrollPerPixel
        )
        self.verticalScrollBar().setSingleStep(48)
        self._scroll_target = 0
        self._smooth_scroll_enabled = _client_animations_enabled()
        self._scroll_animation = QPropertyAnimation(
            self.verticalScrollBar(),
            b"value",
            self,
        )
        self._scroll_animation.setDuration(135)
        self._scroll_animation.setEasingCurve(
            QEasingCurve.Type.OutCubic
        )

    def wheelEvent(self, event):
        pixel_delta = event.pixelDelta().y()
        angle_delta = event.angleDelta().y()
        if pixel_delta == 0 and angle_delta == 0:
            super().wheelEvent(event)
            return

        distance = int(
            pixel_delta * 1.8
            if pixel_delta
            else angle_delta * 1.4
        )
        if event.inverted():
            distance = -distance

        scroll_bar = self.verticalScrollBar()
        if (
            self._scroll_animation.state()
            == QAbstractAnimation.State.Running
        ):
            base_value = self._scroll_target
        else:
            base_value = scroll_bar.value()

        self._scroll_target = max(
            scroll_bar.minimum(),
            min(scroll_bar.maximum(), base_value - distance),
        )

        if self._smooth_scroll_enabled:
            self._scroll_animation.stop()
            self._scroll_animation.setStartValue(scroll_bar.value())
            self._scroll_animation.setEndValue(self._scroll_target)
            self._scroll_animation.start()
        else:
            scroll_bar.setValue(self._scroll_target)
        event.accept()

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            super().dragEnterEvent(event)

    def dragMoveEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            super().dragMoveEvent(event)

    def dropEvent(self, event):
        if event.mimeData().hasUrls():
            self.files_dropped.emit(paths_from_drop_event(event))
            event.acceptProposedAction()
        else:
            super().dropEvent(event)

    def paintEvent(self, event):
        super().paintEvent(event)
        if self.count():
            return

        painter = QPainter(self.viewport())
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(QColor("#6e7687"))
        painter.drawText(
            self.viewport().rect().adjusted(24, 24, -24, -24),
            (
                Qt.AlignmentFlag.AlignHCenter
                | Qt.AlignmentFlag.AlignBottom
                | Qt.TextFlag.TextWordWrap
            ),
            "대기열이 비어 있습니다\n이미지를 창 어디에나 놓으세요\nPNG · JPG · WEBP",
        )
        painter.end()


class OutputNamingDialog(QDialog):
    """출력 PNG의 원본 이름 앞뒤에 붙일 텍스트를 설정한다."""

    def __init__(
        self,
        policy: OutputNamingPolicy,
        parent: QWidget | None = None,
    ):
        super().__init__(parent)
        self.setWindowTitle("출력 이미지 이름")
        self.setModal(True)
        self.setMinimumWidth(560)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(26, 24, 26, 22)
        layout.setSpacing(14)

        title = QLabel("출력 이미지 이름")
        title.setObjectName("dialogTitle")
        description = QLabel(
            "원본 파일 이름 앞뒤에 원하는 문자를 붙일 수 있습니다. "
            "결과는 항상 PNG로 저장됩니다."
        )
        description.setObjectName("mutedText")
        description.setWordWrap(True)

        form_card = QFrame()
        form_card.setObjectName("storageCard")
        form = QFormLayout(form_card)
        form.setContentsMargins(16, 14, 16, 14)
        form.setSpacing(10)

        self.prefix_field = QLineEdit(policy.prefix)
        self.prefix_field.setObjectName("namingField")
        self.prefix_field.setAccessibleName("출력 파일 이름 접두사")
        self.prefix_field.setPlaceholderText("예: toonout_")
        self.prefix_field.setMaxLength(MAX_OUTPUT_AFFIX_LENGTH)
        self.suffix_field = QLineEdit(policy.suffix)
        self.suffix_field.setObjectName("namingField")
        self.suffix_field.setAccessibleName("출력 파일 이름 접미사")
        self.suffix_field.setPlaceholderText("예: _cutout")
        self.suffix_field.setMaxLength(MAX_OUTPUT_AFFIX_LENGTH)
        form.addRow("접두사", self.prefix_field)
        form.addRow("접미사", self.suffix_field)

        preview_card = QFrame()
        preview_card.setObjectName("guideCard")
        preview_layout = QVBoxLayout(preview_card)
        preview_layout.setContentsMargins(16, 13, 16, 13)
        preview_layout.setSpacing(5)
        preview_title = QLabel("파일 이름 미리보기")
        preview_title.setObjectName("guideTitle")
        self.preview_label = QLabel()
        self.preview_label.setObjectName("namingPreview")
        self.collision_label = QLabel()
        self.collision_label.setObjectName("mutedText")
        self.collision_label.setWordWrap(True)
        preview_layout.addWidget(preview_title)
        preview_layout.addWidget(self.preview_label)
        preview_layout.addWidget(self.collision_label)

        self.error_label = QLabel()
        self.error_label.setObjectName("errorText")
        self.error_label.setWordWrap(True)
        self.error_label.hide()

        button_row = QHBoxLayout()
        reset_button = QPushButton("기본값으로 되돌리기")
        reset_button.setObjectName("quietButton")
        cancel_button = QPushButton("취소")
        cancel_button.setObjectName("quietButton")
        self.save_button = QPushButton("적용")
        self.save_button.setObjectName("primaryButton")
        self.save_button.setDefault(True)
        button_row.addWidget(reset_button)
        button_row.addStretch()
        button_row.addWidget(cancel_button)
        button_row.addWidget(self.save_button)

        layout.addWidget(title)
        layout.addWidget(description)
        layout.addWidget(form_card)
        layout.addWidget(preview_card)
        layout.addWidget(self.error_label)
        layout.addLayout(button_row)

        self.prefix_field.textChanged.connect(self._refresh_preview)
        self.suffix_field.textChanged.connect(self._refresh_preview)
        reset_button.clicked.connect(self._reset_to_default)
        cancel_button.clicked.connect(self.reject)
        self.save_button.clicked.connect(self.accept)
        self._refresh_preview()

    @property
    def policy(self) -> OutputNamingPolicy:
        return OutputNamingPolicy(
            prefix=self.prefix_field.text(),
            suffix=self.suffix_field.text(),
        )

    def _reset_to_default(self):
        self.prefix_field.clear()
        self.suffix_field.clear()
        self.prefix_field.setFocus()

    def _refresh_preview(self):
        prefix_error = validate_output_affix(self.prefix_field.text())
        suffix_error = validate_output_affix(self.suffix_field.text())
        error = (
            f"접두사: {prefix_error}"
            if prefix_error
            else f"접미사: {suffix_error}"
            if suffix_error
            else None
        )
        self.error_label.setVisible(error is not None)
        self.error_label.setText(error or "")
        self.save_button.setEnabled(error is None)

        policy = self.policy
        preferred = policy.filename_for("character.jpg")
        self.preview_label.setText(f"character.jpg  →  {preferred}")
        preferred_stem = Path(preferred).stem
        if policy.is_default:
            collision_name = f"{preferred_stem}_no_bg.png"
        else:
            collision_name = f"{preferred_stem}_2.png"
        self.collision_label.setText(
            f"같은 이름이 이미 있으면 {collision_name}으로 저장합니다."
        )


class PerformanceDialog(QDialog):
    """CPU 스레드, GPU 메모리 상한, 이미지 사이 휴식 시간을 정한다."""

    def __init__(
        self,
        policy: PerformancePolicy,
        parent: QWidget | None = None,
    ):
        super().__init__(parent)
        self.setWindowTitle("사용자 지정 성능")
        self.setModal(True)
        self.setMinimumWidth(540)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(26, 24, 26, 22)
        layout.setSpacing(14)

        title = QLabel("사용자 지정 성능")
        title.setObjectName("dialogTitle")
        description = QLabel(
            "CPU는 동시에 쓰는 계산 스레드 수를 제한하고, GPU는 ToonOut이 "
            "예약할 수 있는 메모리 상한을 정합니다. GPU 사용률을 정확한 "
            "퍼센트로 고정하는 기능은 아닙니다."
        )
        description.setObjectName("mutedText")
        description.setWordWrap(True)

        form_card = QFrame()
        form_card.setObjectName("storageCard")
        form = QFormLayout(form_card)
        form.setContentsMargins(16, 14, 16, 14)
        form.setSpacing(10)

        self.cpu_threads = QSpinBox()
        self.cpu_threads.setRange(1, max(1, os.cpu_count() or 1))
        self.cpu_threads.setValue(policy.cpu_threads)
        self.cpu_threads.setSuffix(" 개")
        self.cpu_threads.setToolTip("PyTorch가 CPU 계산에 사용할 최대 스레드 수")

        self.gpu_memory = QSpinBox()
        self.gpu_memory.setRange(10, 100)
        self.gpu_memory.setSingleStep(5)
        self.gpu_memory.setValue(round(policy.gpu_memory_fraction * 100))
        self.gpu_memory.setSuffix(" %")
        self.gpu_memory.setToolTip("ToonOut 프로세스의 GPU 메모리 예약 상한")

        self.cooldown = QSpinBox()
        self.cooldown.setRange(0, 10_000)
        self.cooldown.setSingleStep(100)
        self.cooldown.setValue(policy.cooldown_ms)
        self.cooldown.setSuffix(" ms")
        self.cooldown.setToolTip("한 이미지가 끝난 뒤 다음 이미지를 시작하기 전 대기")

        form.addRow("CPU 계산 스레드", self.cpu_threads)
        form.addRow("GPU 메모리 상한", self.gpu_memory)
        form.addRow("이미지 사이 휴식", self.cooldown)

        caution = QLabel(
            "메모리 상한이 너무 낮으면 모델이 실행되지 않을 수 있습니다. "
            "그때는 상한을 높이거나 CPU 처리로 전환하세요."
        )
        caution.setObjectName("mutedText")
        caution.setWordWrap(True)

        button_row = QHBoxLayout()
        cancel_button = QPushButton("취소")
        cancel_button.setObjectName("quietButton")
        save_button = QPushButton("적용")
        save_button.setObjectName("primaryButton")
        save_button.setDefault(True)
        button_row.addStretch()
        button_row.addWidget(cancel_button)
        button_row.addWidget(save_button)

        layout.addWidget(title)
        layout.addWidget(description)
        layout.addWidget(form_card)
        layout.addWidget(caution)
        layout.addLayout(button_row)

        cancel_button.clicked.connect(self.reject)
        save_button.clicked.connect(self.accept)

    @property
    def policy(self) -> PerformancePolicy:
        return PerformancePolicy(
            PerformanceMode.CUSTOM,
            self.cpu_threads.value(),
            self.gpu_memory.value() / 100,
            self.cooldown.value(),
        ).normalized()


class ModelInstallDialog(QDialog):
    """저장 위치 선택부터 다운로드 결과까지 한곳에서 보여주는 설치 창."""

    install_requested = Signal(str)
    elevation_requested = Signal(str)
    cancel_requested = Signal()

    def __init__(
        self,
        current_directory: str | Path,
        allow_elevation: bool,
        parent: QWidget | None = None,
    ):
        super().__init__(parent)
        self._directory = Path(current_directory)
        self._allow_elevation = allow_elevation
        self._installing = False
        self._installed = False
        self.setWindowTitle("ToonOut 모델 설치")
        self.setModal(True)
        self.setMinimumWidth(640)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(26, 24, 26, 22)
        layout.setSpacing(15)

        title_row = QHBoxLayout()
        title = QLabel("배경 제거 모델 설치")
        title.setObjectName("dialogTitle")
        self.state_label = QLabel("● 설치 필요")
        self.state_label.setObjectName("modelRequiredLabel")
        title_row.addWidget(title)
        title_row.addStretch()
        title_row.addWidget(self.state_label)

        description = QLabel(
            "모델은 선택한 폴더에 저장되고 이 컴퓨터에서만 실행됩니다. "
            "다운로드와 캐시를 위해 여러 GB의 여유 공간을 권장합니다."
        )
        description.setObjectName("mutedText")
        description.setWordWrap(True)

        storage_card = QFrame()
        storage_card.setObjectName("storageCard")
        card_layout = QVBoxLayout(storage_card)
        card_layout.setContentsMargins(16, 14, 16, 14)
        card_layout.setSpacing(9)
        path_label = QLabel("모델 저장 폴더")
        path_label.setObjectName("fieldLabel")
        path_row = QHBoxLayout()
        self.path_field = QLineEdit(str(self._directory))
        self.path_field.setObjectName("pathField")
        self.path_field.setReadOnly(True)
        self.browse_button = QPushButton("위치 변경")
        self.browse_button.setObjectName("secondaryButton")
        path_row.addWidget(self.path_field, 1)
        path_row.addWidget(self.browse_button)
        self.space_label = QLabel()
        self.space_label.setObjectName("storageSpace")
        card_layout.addWidget(path_label)
        card_layout.addLayout(path_row)
        card_layout.addWidget(self.space_label)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.hide()
        self.status_label = QLabel("설치를 시작하면 필요한 파일을 내려받습니다")
        self.status_label.setObjectName("mutedText")
        self.status_label.setWordWrap(True)

        button_row = QHBoxLayout()
        self.cancel_button = QPushButton("취소")
        self.cancel_button.setObjectName("quietButton")
        self.install_button = QPushButton("모델 설치")
        self.install_button.setObjectName("primaryButton")
        self.install_button.setDefault(True)
        button_row.addStretch()
        button_row.addWidget(self.cancel_button)
        button_row.addWidget(self.install_button)

        layout.addLayout(title_row)
        layout.addWidget(description)
        layout.addWidget(storage_card)
        layout.addWidget(self.progress_bar)
        layout.addWidget(self.status_label)
        layout.addLayout(button_row)

        self.browse_button.clicked.connect(self._browse)
        self.cancel_button.clicked.connect(self._handle_cancel_action)
        self.install_button.clicked.connect(self._handle_primary_action)
        self._refresh_space_label()

    @property
    def selected_directory(self) -> Path:
        return self._directory

    def _browse(self):
        directory = QFileDialog.getExistingDirectory(
            self,
            "모델을 저장할 폴더 선택",
            str(self._directory),
            QFileDialog.Option.ShowDirsOnly,
        )
        if directory:
            self._directory = Path(directory)
            self.path_field.setText(str(self._directory))
            self._refresh_space_label()

    def _refresh_space_label(self):
        available_bytes = available_storage_bytes(self._directory)
        if available_bytes is None:
            self.space_label.setText("사용 가능한 공간을 확인할 수 없습니다")
        else:
            self.space_label.setText(
                f"사용 가능한 공간 · {format_storage_size(available_bytes)}"
            )

    def _handle_primary_action(self):
        if self._installed:
            self.accept()
            return
        if self._installing:
            return

        try:
            self._directory = ensure_writable_model_directory(self._directory)
        except OSError as error:
            if is_permission_error(error) and self._allow_elevation:
                self._offer_elevation(error)
            else:
                QMessageBox.warning(
                    self,
                    "이 폴더를 사용할 수 없습니다",
                    "선택한 폴더에 파일을 쓸 수 없습니다. "
                    f"다른 위치를 선택하세요.\n\n세부 정보: {error}",
                )
            return

        self._installing = True
        self.browse_button.setEnabled(False)
        self.cancel_button.setText("설치 취소")
        self.cancel_button.setEnabled(True)
        self.install_button.setEnabled(False)
        self.progress_bar.show()
        self.status_label.setObjectName("mutedText")
        self.status_label.setStyleSheet("")
        self.status_label.setText("모델 설치를 시작하는 중")
        self.install_requested.emit(str(self._directory))

    def _handle_cancel_action(self):
        if self._installing:
            self.cancel_requested.emit()
            return
        self.reject()

    def _offer_elevation(self, error: OSError):
        message = QMessageBox(self)
        message.setWindowTitle("관리자 권한 필요")
        message.setIcon(QMessageBox.Icon.Warning)
        message.setText("선택한 폴더에 모델을 설치하려면 관리자 권한이 필요합니다.")
        message.setInformativeText(
            "앱을 관리자 권한으로 다시 실행하거나 다른 위치를 선택할 수 있습니다. "
            f"현재 위치: {self._directory}\n\n세부 정보: {error}"
        )
        change_button = message.addButton(
            "다른 위치 선택",
            QMessageBox.ButtonRole.RejectRole,
        )
        elevate_button = message.addButton(
            "관리자 권한으로 다시 실행",
            QMessageBox.ButtonRole.AcceptRole,
        )
        message.setDefaultButton(change_button)
        message.exec()
        if message.clickedButton() is elevate_button:
            self.elevation_requested.emit(str(self._directory))

    def set_status(self, text: str):
        self.status_label.setText(text)

    def show_failure(self, error: str):
        self._installing = False
        self.browse_button.setEnabled(True)
        self.cancel_button.setText("취소")
        self.cancel_button.setEnabled(True)
        self.install_button.setEnabled(True)
        self.progress_bar.hide()
        self.status_label.setObjectName("errorText")
        self.status_label.setStyleSheet("")
        self.status_label.setText(error)

    def show_cancelling(self):
        self.cancel_button.setEnabled(False)
        self.install_button.setEnabled(False)
        self.status_label.setObjectName("mutedText")
        self.status_label.setStyleSheet("")
        self.status_label.setText("모델 설치를 취소하는 중")

    def show_cancelled(self):
        self._installing = False
        self.browse_button.setEnabled(True)
        self.cancel_button.setText("취소")
        self.cancel_button.setEnabled(True)
        self.install_button.setEnabled(True)
        self.progress_bar.hide()
        self.status_label.setObjectName("mutedText")
        self.status_label.setStyleSheet("")
        self.status_label.setText(
            "모델 설치를 취소했습니다 · 내려받은 일부 파일은 다음 설치 때 재사용됩니다"
        )

    def show_success(self):
        self._installing = False
        self._installed = True
        self.state_label.setText("● 설치됨")
        self.state_label.setObjectName("modelInstalledLabel")
        self.state_label.setStyleSheet("")
        self.progress_bar.setRange(0, 1)
        self.progress_bar.setValue(1)
        self.status_label.setObjectName("mutedText")
        self.status_label.setStyleSheet("")
        self.status_label.setText("모델 설치가 완료되었습니다")
        self.cancel_button.setText("닫기")
        self.install_button.setText("완료")
        self.install_button.setEnabled(True)

    def closeEvent(self, event):
        super().closeEvent(event)


class ModelManagementDialog(QDialog):
    """설치된 모델의 위치 확인, 이동, 삭제 진입점."""

    MOVE_RESULT = 2
    DELETE_RESULT = 3
    open_requested = Signal()

    def __init__(
        self,
        directory: str | Path,
        size_text: str,
        parent: QWidget | None = None,
    ):
        super().__init__(parent)
        self.setWindowTitle("설치된 모델 관리")
        self.setModal(True)
        self.setMinimumWidth(620)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(26, 24, 26, 22)
        layout.setSpacing(15)

        title_row = QHBoxLayout()
        title = QLabel("설치된 모델 관리")
        title.setObjectName("dialogTitle")
        state = QLabel("● 설치됨")
        state.setObjectName("modelInstalledLabel")
        title_row.addWidget(title)
        title_row.addStretch()
        title_row.addWidget(state)

        description = QLabel(
            "이 모델은 로컬에서 배경 제거에 사용됩니다. 위치를 열거나, "
            "다른 드라이브로 옮기거나, 다시 다운로드할 수 있도록 삭제할 수 있습니다."
        )
        description.setObjectName("mutedText")
        description.setWordWrap(True)

        card = QFrame()
        card.setObjectName("storageCard")
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(16, 14, 16, 14)
        card_layout.setSpacing(9)
        path_label = QLabel("현재 위치")
        path_label.setObjectName("fieldLabel")
        path_field = QLineEdit(str(directory))
        path_field.setObjectName("pathField")
        path_field.setReadOnly(True)
        size_label = QLabel(f"모델 파일 크기 · {size_text}")
        size_label.setObjectName("storageSpace")
        card_layout.addWidget(path_label)
        card_layout.addWidget(path_field)
        card_layout.addWidget(size_label)

        action_row = QHBoxLayout()
        open_button = QPushButton("폴더 열기")
        open_button.setObjectName("secondaryButton")
        move_button = QPushButton("다른 위치로 이동")
        move_button.setObjectName("secondaryButton")
        delete_button = QPushButton("모델 삭제")
        delete_button.setObjectName("dangerButton")
        action_row.addWidget(open_button)
        action_row.addWidget(move_button)
        action_row.addStretch()
        action_row.addWidget(delete_button)

        close_button = QPushButton("닫기")
        close_button.setObjectName("quietButton")
        close_row = QHBoxLayout()
        close_row.addStretch()
        close_row.addWidget(close_button)

        layout.addLayout(title_row)
        layout.addWidget(description)
        layout.addWidget(card)
        layout.addLayout(action_row)
        layout.addLayout(close_row)

        open_button.clicked.connect(self.open_requested.emit)
        move_button.clicked.connect(lambda: self.done(self.MOVE_RESULT))
        delete_button.clicked.connect(lambda: self.done(self.DELETE_RESULT))
        close_button.clicked.connect(self.reject)


class AccelerationDialog(QDialog):
    """한 앱 안에서 CPU/GPU 선택과 GPU 팩 관리를 안내한다."""

    install_requested = Signal()
    gpu_enabled_changed = Signal(bool)
    remove_requested = Signal()
    driver_page_requested = Signal()

    def __init__(
        self,
        info: AccelerationInfo,
        parent: QWidget | None = None,
    ):
        super().__init__(parent)
        self.setWindowTitle("처리 장치와 GPU 가속")
        self.setModal(True)
        self.setMinimumWidth(680)

        title_text, description, guidance = self._copy_for_mode(info)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(26, 24, 26, 22)
        layout.setSpacing(15)

        title_row = QHBoxLayout()
        title = QLabel(title_text)
        title.setObjectName("dialogTitle")
        state = QLabel(info.status_text)
        runtime_outdated = (
            info.runtime is not None
            and not gpu_runtime_is_compatible(info.runtime)
        )
        state.setObjectName(
            "accelerationWarningLabel"
            if runtime_outdated
            else self._state_label_name(info.mode)
        )
        title_row.addWidget(title)
        title_row.addStretch()
        title_row.addWidget(state)

        description_label = QLabel(description)
        description_label.setObjectName("mutedText")
        description_label.setWordWrap(True)

        details_card = QFrame()
        details_card.setObjectName("storageCard")
        details_layout = QVBoxLayout(details_card)
        details_layout.setContentsMargins(16, 14, 16, 14)
        details_layout.setSpacing(8)
        self._add_detail_row(
            details_layout,
            "현재 처리",
            "NVIDIA GPU" if info.mode == AccelerationMode.GPU_ACTIVE else "CPU",
        )
        if info.device is not None:
            self._add_detail_row(details_layout, "그래픽 카드", info.device.name)
            if info.device.driver_version:
                self._add_detail_row(
                    details_layout,
                    "NVIDIA 드라이버",
                    info.device.driver_version,
                )
            if info.device.memory_gib is not None:
                self._add_detail_row(
                    details_layout,
                    "GPU 메모리",
                    f"{info.device.memory_gib:.1f} GB",
                )
        if info.runtime is not None:
            self._add_detail_row(
                details_layout,
                "GPU 가속 팩",
                f"PyTorch {info.runtime.torch_version} · CUDA {info.runtime.cuda_runtime}",
            )
            self._add_detail_row(
                details_layout,
                "설치 용량",
                format_storage_size(info.runtime_size),
            )
        if info.runtime_directory:
            path_title = QLabel("GPU 팩 설치 위치")
            path_title.setObjectName("fieldLabel")
            path_label = QLineEdit(info.runtime_directory)
            path_label.setObjectName("pathField")
            path_label.setReadOnly(True)
            details_layout.addWidget(path_title)
            details_layout.addWidget(path_label)

        guide_card = QFrame()
        guide_card.setObjectName("guideCard")
        guide_layout = QVBoxLayout(guide_card)
        guide_layout.setContentsMargins(16, 14, 16, 14)
        guide_layout.setSpacing(5)
        guide_title = QLabel("설치 전 알아두기")
        guide_title.setObjectName("guideTitle")
        guide_text = QLabel(guidance)
        guide_text.setObjectName("mutedText")
        guide_text.setWordWrap(True)
        guide_layout.addWidget(guide_title)
        guide_layout.addWidget(guide_text)

        action_row = QHBoxLayout()
        if runtime_outdated:
            install_button = QPushButton("GPU 가속 팩 업데이트")
            install_button.setObjectName("primaryButton")
            install_button.clicked.connect(self.install_requested.emit)
            action_row.addWidget(install_button)
        elif info.mode == AccelerationMode.GPU_PACK_AVAILABLE:
            install_button = QPushButton("GPU 가속 팩 설치")
            install_button.setObjectName("primaryButton")
            install_button.clicked.connect(self.install_requested.emit)
            action_row.addWidget(install_button)
        elif info.mode == AccelerationMode.GPU_ACTIVE:
            cpu_button = QPushButton("CPU로 전환")
            cpu_button.setObjectName("secondaryButton")
            cpu_button.clicked.connect(
                lambda: self.gpu_enabled_changed.emit(False)
            )
            action_row.addWidget(cpu_button)
        elif info.mode == AccelerationMode.GPU_PACK_INSTALLED:
            gpu_button = QPushButton("GPU 가속 사용")
            gpu_button.setObjectName("primaryButton")
            gpu_button.setEnabled(info.device is not None)
            gpu_button.clicked.connect(
                lambda: self.gpu_enabled_changed.emit(True)
            )
            action_row.addWidget(gpu_button)
        elif info.mode == AccelerationMode.GPU_UNAVAILABLE:
            cpu_button = QPushButton("CPU로 전환")
            cpu_button.setObjectName("secondaryButton")
            cpu_button.clicked.connect(
                lambda: self.gpu_enabled_changed.emit(False)
            )
            action_row.addWidget(cpu_button)

        if info.runtime is not None:
            remove_button = QPushButton("가속 팩 삭제")
            remove_button.setObjectName("dangerButton")
            remove_button.clicked.connect(self.remove_requested.emit)
            action_row.addWidget(remove_button)
        if info.device is not None or info.runtime is not None:
            driver_button = QPushButton("NVIDIA 드라이버 확인")
            driver_button.setObjectName("quietButton")
            driver_button.clicked.connect(self.driver_page_requested.emit)
            action_row.addWidget(driver_button)

        action_row.addStretch()
        close_button = QPushButton("닫기")
        close_button.setObjectName("quietButton")
        close_button.clicked.connect(self.accept)
        action_row.addWidget(close_button)

        layout.addLayout(title_row)
        layout.addWidget(description_label)
        layout.addWidget(details_card)
        layout.addWidget(guide_card)
        layout.addLayout(action_row)

    @staticmethod
    def _state_label_name(mode: AccelerationMode) -> str:
        return {
            AccelerationMode.GPU_ACTIVE: "accelerationActiveLabel",
            AccelerationMode.GPU_PACK_INSTALLED: "accelerationInstalledLabel",
            AccelerationMode.GPU_PACK_AVAILABLE: "accelerationAvailableLabel",
            AccelerationMode.GPU_UNAVAILABLE: "accelerationWarningLabel",
            AccelerationMode.CPU_ONLY: "accelerationCpuLabel",
        }[mode]

    @staticmethod
    def _copy_for_mode(info: AccelerationInfo) -> tuple[str, str, str]:
        if info.runtime is not None and not gpu_runtime_is_compatible(
            info.runtime
        ):
            return (
                "GPU 가속 팩 업데이트가 필요합니다",
                "설치된 팩은 현재 ToonOut의 성능 모드와 일시정지를 지원하지 않습니다.",
                "최신 ToonOut 배포 파일에 포함된 GPU 가속 팩으로 업데이트하세요. "
                "업데이트 전에도 CPU 처리는 계속 사용할 수 있습니다.",
            )
        if info.mode == AccelerationMode.GPU_ACTIVE:
            return (
                "GPU 가속을 사용합니다",
                f"다음 배경 제거 작업은 {info.device.name}에서 실행됩니다.",
                "GPU 가속 팩은 ToonOut 사용자 폴더에만 설치됩니다. 시스템 CUDA "
                "Toolkit이나 Python을 설치하지 않으며 언제든 CPU로 전환하거나 "
                "가속 팩만 삭제할 수 있습니다.",
            )
        if info.mode == AccelerationMode.GPU_PACK_INSTALLED:
            return (
                "GPU 가속 팩이 설치되어 있습니다",
                "현재는 CPU로 처리하도록 선택되어 있습니다.",
                "호환되는 NVIDIA 그래픽 카드가 있으면 GPU 가속을 켤 수 있습니다. "
                "이미지와 모델은 GPU를 사용해도 컴퓨터 밖으로 전송되지 않습니다.",
            )
        if info.mode == AccelerationMode.GPU_PACK_AVAILABLE:
            return (
                "이 PC에서 GPU 가속을 추가할 수 있습니다",
                f"{info.device.name}을 찾았습니다. 같은 ToonOut 앱에 GPU 지원을 추가할 수 있습니다.",
                "GPU 가속 팩은 수 GB의 저장 공간과 다운로드가 필요합니다. 팩에는 "
                "PyTorch와 CUDA 실행 파일이 포함되므로 CUDA Toolkit을 따로 설치하지 "
                "마세요. 최신 NVIDIA 드라이버를 권장하며 ToonOut 공식 배포 팩만 "
                "설치하세요.",
            )
        if info.mode == AccelerationMode.GPU_UNAVAILABLE:
            return (
                "GPU 가속을 시작할 수 없습니다",
                "GPU 가속 팩은 설치되어 있지만 호환되는 NVIDIA 장치를 찾지 못했습니다.",
                "CPU로 전환한 뒤 NVIDIA 드라이버와 그래픽 카드 호환성을 확인하세요. "
                "문제 해결을 위해 시스템 CUDA Toolkit을 설치할 필요는 없습니다.",
            )
        return (
            "현재 CPU로 처리합니다",
            "호환되는 NVIDIA 그래픽 카드를 찾지 못했습니다.",
            "CPU 처리는 추가 설치 없이 사용할 수 있습니다. 현재 GPU 가속은 NVIDIA "
            "그래픽 카드만 지원하며 AMD·Intel GPU는 지원하지 않습니다.",
        )

    @staticmethod
    def _add_detail_row(layout: QVBoxLayout, label: str, value: str):
        row = QHBoxLayout()
        name_label = QLabel(label)
        name_label.setObjectName("fieldLabel")
        value_label = QLabel(value)
        value_label.setTextInteractionFlags(
            value_label.textInteractionFlags()
            | Qt.TextInteractionFlag.TextSelectableByMouse
        )
        row.addWidget(name_label)
        row.addStretch()
        row.addWidget(value_label)
        layout.addLayout(row)


class ModelOperationDialog(QDialog):
    """긴 파일 작업을 표시하고 선택적으로 백그라운드·취소를 지원한다."""

    cancel_requested = Signal()

    def __init__(
        self,
        title: str,
        status: str,
        parent: QWidget | None = None,
        *,
        allow_background: bool = False,
        cancellable: bool = False,
    ):
        super().__init__(parent)
        self._running = True
        self._allow_background = allow_background
        self.setWindowTitle(title)
        self.setModal(True)
        self.setMinimumWidth(520)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 22, 24, 22)
        layout.setSpacing(12)
        title_label = QLabel(title)
        title_label.setObjectName("dialogTitle")
        self.status_label = QLabel(status)
        self.status_label.setObjectName("mutedText")
        self.status_label.setWordWrap(True)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)
        self.progress_bar.setTextVisible(False)
        layout.addWidget(title_label)
        layout.addWidget(self.status_label)
        layout.addWidget(self.progress_bar)

        self.cancel_button: QPushButton | None = None
        self.background_button: QPushButton | None = None
        if cancellable or allow_background:
            button_row = QHBoxLayout()
            button_row.addStretch()
            if cancellable:
                self.cancel_button = QPushButton("설치 취소")
                self.cancel_button.setObjectName("dangerButton")
                self.cancel_button.clicked.connect(self._request_cancel)
                button_row.addWidget(self.cancel_button)
            if allow_background:
                self.background_button = QPushButton("백그라운드에서 계속")
                self.background_button.setObjectName("quietButton")
                self.background_button.clicked.connect(self.reject)
                button_row.addWidget(self.background_button)
            layout.addLayout(button_row)

    def _request_cancel(self):
        if not self._running:
            return
        if self.cancel_button is not None:
            self.cancel_button.setEnabled(False)
        self.set_status("GPU 가속 팩 설치를 취소하는 중")
        self.cancel_requested.emit()

    def set_status(self, status: str):
        self.status_label.setText(status)

    def finish(self):
        self._running = False
        self.accept()

    def fail(self):
        self._running = False
        self.reject()

    def show_cancelled(self):
        self._running = False
        self.progress_bar.hide()
        self.status_label.setText("GPU 가속 팩 설치를 취소했습니다")
        if self.cancel_button is not None:
            self.cancel_button.hide()
        if self.background_button is not None:
            self.background_button.setText("닫기")

    def closeEvent(self, event):
        if self._running and not self._allow_background:
            event.ignore()
            return
        super().closeEvent(event)
