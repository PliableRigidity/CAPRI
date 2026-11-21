"""Top-down arena map in the arena-fixed frame of config/arena_overlay.yaml (origin = plain corner, +y up).

Used live (Operate) and for recorded rounds (Review). It draws what it is given: trails are lists
of separate runs, so a gap in tracking stays a gap on screen.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QFont, QPainter, QPainterPath, QPen, QPolygonF
from PySide6.QtWidgets import QSizePolicy, QWidget

from control_center.tools.overlay_core import arena_outline, flipper_boundary, pit_arcs


@dataclass
class Trail:
    runs: list[np.ndarray]              # each (n, 2) in metres; drawn as separate polylines
    colour: str = "#55c889"
    dashed: bool = False                # dashed = reconstructed offline, solid = estimated during the round
    label: str = ""
    width: float = 2.0


@dataclass
class Marker:
    x: float
    y: float
    heading_deg: float | None = None
    colour: str = "#55c889"
    label: str = ""
    hollow: bool = False                # hollow = not a current measurement (e.g. last known position)


class ArenaMapWidget(QWidget):
    def __init__(self, arena: dict[str, Any], parent=None) -> None:
        super().__init__(parent)
        self.arena = arena
        self.trails: list[Trail] = []
        self.markers: list[Marker] = []
        self.caption = ""
        self.note = ""                  # large centred text, e.g. "Map not calibrated"
        self.setMinimumSize(220, 220)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._outline = arena_outline(arena)
        self._pits = pit_arcs(arena, samples=96)
        self._flipper = flipper_boundary(arena)

    def set_arena(self, arena: dict[str, Any]) -> None:
        """Show a different arena (another profile, or a newly saved set-up)."""
        self.arena = arena
        self._outline = arena_outline(arena)
        self._pits = pit_arcs(arena, samples=96)
        self._flipper = flipper_boundary(arena)
        self.update()

    def set_content(self, trails: list[Trail] | None = None, markers: list[Marker] | None = None,
                    caption: str | None = None, note: str | None = None) -> None:
        if trails is not None:
            self.trails = trails
        if markers is not None:
            self.markers = markers
        if caption is not None:
            self.caption = caption
        if note is not None:
            self.note = note
        self.update()

    # ---- geometry ----

    def _arena_rect(self) -> QRectF:
        margin = 18.0
        side = max(10.0, min(self.width(), self.height() - 22) - 2 * margin)
        return QRectF((self.width() - side) / 2.0, margin, side, side)

    def _to_widget(self, rect: QRectF, x: float, y: float) -> QPointF:
        w, h = float(self.arena["width_m"]), float(self.arena["height_m"])
        return QPointF(rect.left() + rect.width() * x / w, rect.bottom() - rect.height() * y / h)

    def _polygon(self, rect: QRectF, points: np.ndarray) -> QPolygonF:
        return QPolygonF([self._to_widget(rect, float(x), float(y)) for x, y in points])

    # ---- painting ----

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt API
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor("#0d1218"))
        rect = self._arena_rect()
        painter.setPen(QPen(QColor("#5c6b7d"), 2))
        painter.setBrush(QColor("#151c25"))
        painter.drawPolygon(self._polygon(rect, self._outline))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(QPen(QColor("#d9a441"), 2))
        for arc in self._pits:
            painter.drawPolyline(self._polygon(rect, arc))
        painter.setPen(QPen(QColor("#c05a5a"), 2))
        painter.drawPolyline(self._polygon(rect, self._flipper))
        painter.setPen(QColor("#5c6b7d"))
        painter.setFont(QFont("Segoe UI", 8))
        origin = self._to_widget(rect, 0.0, 0.0)
        painter.drawText(QPointF(origin.x() + 4, origin.y() - 4), "(0,0) plain corner")

        for trail in self.trails:
            pen = QPen(QColor(trail.colour), trail.width)
            if trail.dashed:
                pen.setStyle(Qt.PenStyle.DashLine)
            painter.setPen(pen)
            for run in trail.runs:
                if len(run) == 1:
                    painter.drawPoint(self._to_widget(rect, float(run[0][0]), float(run[0][1])))
                elif len(run) > 1:
                    painter.drawPolyline(self._polygon(rect, run))

        for marker in self.markers:
            centre = self._to_widget(rect, marker.x, marker.y)
            colour = QColor(marker.colour)
            painter.setPen(QPen(colour, 2))
            painter.setBrush(Qt.BrushStyle.NoBrush if marker.hollow else colour)
            painter.drawEllipse(centre, 7.0, 7.0)
            if marker.heading_deg is not None and math.isfinite(marker.heading_deg):
                angle = math.radians(marker.heading_deg)
                tip = QPointF(centre.x() + 20.0 * math.cos(angle), centre.y() - 20.0 * math.sin(angle))
                painter.drawLine(centre, tip)
            if marker.label:
                painter.setPen(colour)
                painter.drawText(QPointF(centre.x() + 10, centre.y() - 10), marker.label)

        if self.note:
            painter.setPen(QColor("#9eabb8"))
            painter.setFont(QFont("Segoe UI", 13, QFont.Weight.Bold))
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap, self.note)
        if self.caption:
            painter.setPen(QColor("#9eabb8"))
            painter.setFont(QFont("Segoe UI", 9))
            painter.drawText(QRectF(4, self.height() - 20, self.width() - 8, 18), Qt.AlignmentFlag.AlignCenter, self.caption)
        painter.end()
