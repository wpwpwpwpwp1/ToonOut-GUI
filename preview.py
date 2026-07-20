"""원본/결과 비교와 확대·이동을 제공하는 이미지 캔버스."""

from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QBrush, QColor, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QFrame,
    QGraphicsItem,
    QGraphicsScene,
    QGraphicsView,
)


class ComparePixmapItem(QGraphicsItem):
    def __init__(self):
        super().__init__()
        self._original = QPixmap()
        self._result = QPixmap()
        self._mode = "original"
        self._split = 0.5

    def boundingRect(self):
        if self._original.isNull():
            return QRectF()
        return QRectF(self._original.rect())

    def set_images(self, original: QPixmap, result: QPixmap | None = None):
        self.prepareGeometryChange()
        self._original = original
        self._result = result or QPixmap()
        self.update()

    def clear_images(self):
        self.prepareGeometryChange()
        self._original = QPixmap()
        self._result = QPixmap()
        self.update()

    def set_mode(self, mode: str):
        self._mode = mode
        self.update()

    def set_split(self, split: float):
        self._split = max(0.0, min(1.0, split))
        self.update()

    def has_image(self) -> bool:
        return not self._original.isNull()

    def has_result(self) -> bool:
        return not self._result.isNull()

    def _draw_pixmap(self, painter, pixmap, clip_rect=None):
        if pixmap.isNull():
            return

        painter.save()
        if clip_rect is not None:
            painter.setClipRect(clip_rect)
        painter.drawPixmap(
            self.boundingRect(),
            pixmap,
            QRectF(pixmap.rect()),
        )
        painter.restore()

    def paint(self, painter, option, widget=None):
        if not self.has_image():
            return

        if self._mode == "result" and self.has_result():
            self._draw_pixmap(painter, self._result)
            return

        if self._mode != "compare" or not self.has_result():
            self._draw_pixmap(painter, self._original)
            return

        bounds = self.boundingRect()
        split_x = bounds.width() * self._split
        result_rect = QRectF(
            bounds.left(),
            bounds.top(),
            split_x,
            bounds.height(),
        )
        original_rect = QRectF(
            split_x,
            bounds.top(),
            bounds.width() - split_x,
            bounds.height(),
        )

        self._draw_pixmap(painter, self._result, result_rect)
        self._draw_pixmap(painter, self._original, original_rect)

        shadow_pen = QPen(QColor(0, 0, 0, 90), 4)
        shadow_pen.setCosmetic(True)
        painter.setPen(shadow_pen)
        painter.drawLine(int(split_x), 0, int(split_x), int(bounds.height()))

        divider_pen = QPen(QColor("#ffffff"), 2)
        divider_pen.setCosmetic(True)
        painter.setPen(divider_pen)
        painter.drawLine(int(split_x), 0, int(split_x), int(bounds.height()))


class ImagePreview(QGraphicsView):
    def __init__(self):
        super().__init__()

        self._scene = QGraphicsScene(self)
        self._image_item = ComparePixmapItem()
        self._scene.addItem(self._image_item)
        self.setScene(self._scene)

        self._zoom_ratio = 1.0
        self._auto_fit = True
        self._message = "목록에서 이미지를 선택하세요"

        self.setMinimumSize(420, 320)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(
            QGraphicsView.ViewportAnchor.AnchorUnderMouse
        )
        self.setResizeAnchor(
            QGraphicsView.ViewportAnchor.AnchorViewCenter
        )
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        self.setFrameShape(QFrame.Shape.NoFrame)

        checkerboard = QPixmap(24, 24)
        checkerboard.fill(QColor("#f0f2f6"))
        painter = QPainter(checkerboard)
        painter.fillRect(0, 0, 12, 12, QColor("#dce0e7"))
        painter.fillRect(12, 12, 12, 12, QColor("#dce0e7"))
        painter.end()
        self.setBackgroundBrush(QBrush(checkerboard))

    def set_images(self, original: QPixmap, result: QPixmap | None = None):
        self._image_item.set_images(original, result)
        self._scene.setSceneRect(self._image_item.boundingRect())
        self._message = ""
        self.reset_view()
        self.viewport().update()

    def show_message(self, message: str):
        self._image_item.clear_images()
        self._scene.setSceneRect(QRectF())
        self._message = message
        self.resetTransform()
        self._zoom_ratio = 1.0
        self._auto_fit = True
        self.viewport().update()

    def set_mode(self, mode: str):
        self._image_item.set_mode(mode)

    def set_split(self, split: float):
        self._image_item.set_split(split)

    def has_result(self) -> bool:
        return self._image_item.has_result()

    def zoom_in(self):
        self._zoom(1.25)

    def zoom_out(self):
        self._zoom(0.8)

    def _zoom(self, factor: float):
        if not self._image_item.has_image():
            return

        next_ratio = self._zoom_ratio * factor
        if not 0.1 <= next_ratio <= 20:
            return

        self.scale(factor, factor)
        self._zoom_ratio = next_ratio
        self._auto_fit = False

    def reset_view(self):
        if not self._image_item.has_image():
            return

        self.resetTransform()
        self.fitInView(
            self._image_item,
            Qt.AspectRatioMode.KeepAspectRatio,
        )
        self._zoom_ratio = 1.0
        self._auto_fit = True

    def actual_size(self):
        if not self._image_item.has_image():
            return

        self.resetTransform()
        self.centerOn(self._image_item)
        self._zoom_ratio = 1.0
        self._auto_fit = False

    def wheelEvent(self, event):
        if event.angleDelta().y() > 0:
            self.zoom_in()
        else:
            self.zoom_out()
        event.accept()

    def mouseDoubleClickEvent(self, event):
        self.reset_view()
        event.accept()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._auto_fit:
            self.reset_view()

    def paintEvent(self, event):
        super().paintEvent(event)

        if not self._message:
            return

        painter = QPainter(self.viewport())
        painter.setPen(QColor("#6e7687"))
        painter.drawText(
            self.viewport().rect(),
            Qt.AlignmentFlag.AlignCenter,
            self._message,
        )
        painter.end()
