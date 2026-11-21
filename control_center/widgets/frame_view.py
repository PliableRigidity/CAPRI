"""Shows one BGR camera frame, scaled to fit. Reports clicks in image pixel coordinates."""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPen
from PySide6.QtWidgets import QSizePolicy, QWidget


DRAG_MIN_PX = 6.0                           # a press-and-release shorter than this (in widget pixels) is a click, not a drag


class FrameView(QWidget):
    clicked = Signal(float, float)          # image pixel coordinates of a left click
    region_selected = Signal(float, float, float, float)   # x, y, width, height in IMAGE pixels, after a drag (drag_select only)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.drag_select = False            # True: a left drag draws a rubber band and emits region_selected; a click then fires on release
        self._drag_start: QPointF | None = None
        self._drag_now: QPointF | None = None
        self._image: QImage | None = None
        self._size: tuple[int, int] | None = None
        self.placeholder = "No camera frame"
        self.badge = ""                     # short text in the top-left corner, e.g. "MOCK"
        self.setMinimumSize(320, 180)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def set_frame(self, frame: np.ndarray | None) -> None:
        """``frame`` is BGR uint8. It is copied, so the caller may reuse or modify it afterwards."""
        if frame is None:
            self._image, self._size = None, None
        else:
            height, width = frame.shape[:2]
            image = QImage(frame.data, width, height, frame.strides[0], QImage.Format.Format_BGR888)
            self._image, self._size = image.copy(), (width, height)
        self.update()

    def set_placeholder(self, text: str) -> None:
        self.placeholder = text
        self.update()

    def _target(self) -> QRectF | None:
        if self._size is None:
            return None
        width, height = self._size
        scale = min(self.width() / width, self.height() / height)
        w, h = width * scale, height * scale
        return QRectF((self.width() - w) / 2.0, (self.height() - h) / 2.0, w, h)

    def widget_to_image(self, point: QPointF) -> tuple[float, float] | None:
        target = self._target()
        if target is None or not target.contains(point):
            return None
        width, height = self._size
        return ((point.x() - target.left()) * width / target.width(), (point.y() - target.top()) * height / target.height())

    def mousePressEvent(self, event) -> None:  # noqa: N802 - Qt API
        if event.button() != Qt.MouseButton.LeftButton:
            return
        if self.drag_select:
            if self._target() is not None:
                self._drag_start = self._drag_now = event.position()
            return
        position = self.widget_to_image(event.position())
        if position is not None:
            self.clicked.emit(position[0], position[1])

    def mouseMoveEvent(self, event) -> None:  # noqa: N802 - Qt API
        if self._drag_start is not None:
            self._drag_now = event.position()
            self.update()

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802 - Qt API
        if self._drag_start is None or event.button() != Qt.MouseButton.LeftButton:
            return
        start, end = self._drag_start, event.position()
        self._drag_start = self._drag_now = None
        self.update()
        if abs(end.x() - start.x()) < DRAG_MIN_PX and abs(end.y() - start.y()) < DRAG_MIN_PX:
            position = self.widget_to_image(end)
            if position is not None:
                self.clicked.emit(position[0], position[1])
            return
        rect = self.widget_rect_to_image(start, end)
        if rect is not None:
            self.region_selected.emit(*rect)

    def widget_rect_to_image(self, a: QPointF, b: QPointF) -> tuple[float, float, float, float] | None:
        """A dragged rectangle in widget coordinates -> (x, y, w, h) in image pixels, clipped to the picture.
        None when the drag did not touch the picture at all."""
        target = self._target()
        if target is None or self._size is None:
            return None
        width, height = self._size
        scale_x, scale_y = width / target.width(), height / target.height()
        x0, x1 = sorted(((a.x() - target.left()) * scale_x, (b.x() - target.left()) * scale_x))
        y0, y1 = sorted(((a.y() - target.top()) * scale_y, (b.y() - target.top()) * scale_y))
        x0, x1, y0, y1 = max(0.0, x0), min(float(width), x1), max(0.0, y0), min(float(height), y1)
        if x1 <= x0 or y1 <= y0:
            return None
        return (x0, y0, x1 - x0, y1 - y0)

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt API
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#0b0f14"))
        target = self._target()
        if target is None or self._image is None:
            painter.setPen(QColor("#7f8996"))
            painter.setFont(QFont("Segoe UI", 14))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap, self.placeholder)
        else:
            painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
            painter.drawImage(target, self._image)
        if self._drag_start is not None and self._drag_now is not None:
            pen = QPen(QColor("#ffd84d"), 2, Qt.PenStyle.DashLine)
            painter.setPen(pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRect(QRectF(self._drag_start, self._drag_now).normalized())
        if self.badge:
            painter.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
            box = QRectF(8, 8, painter.fontMetrics().horizontalAdvance(self.badge) + 16, 24)
            painter.fillRect(box, QColor("#e8b85b"))
            painter.setPen(QColor("#11161d"))
            painter.drawText(box, Qt.AlignmentFlag.AlignCenter, self.badge)
        painter.end()
