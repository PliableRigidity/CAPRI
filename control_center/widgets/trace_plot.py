"""A small line plot of one to three named series against time, with a cursor at the selected time.

It draws what it is given. A NaN sample is a gap: the line is never joined across it, and (if a
series sets ``max_gap_s``) it is not joined across a pause in the samples either. Nothing is
resampled, smoothed or integrated here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QPen, QPixmap, QPolygonF
from PySide6.QtWidgets import QSizePolicy, QWidget

SERIES_COLOURS = ("#6fa8dc", "#e8a05b", "#b79be8")
MAX_SERIES = 3

BACKGROUND = "#0d1218"
PLOT_BACKGROUND = "#101720"
GRID = "#212b37"
AXIS_TEXT = "#9eabb8"
TITLE_TEXT = "#c7cdd4"
CURSOR = "#f2f5f8"


@dataclass
class Series:
    name: str
    t: np.ndarray
    y: np.ndarray
    colour: str = ""                    # empty = the next colour of SERIES_COLOURS
    dashed: bool = False                # dashed = not measured during the round (e.g. reconstructed offline)
    max_gap_s: float | None = None      # also break the line where consecutive samples are further apart


def finite_runs(t: np.ndarray, y: np.ndarray, max_gap_s: float | None = None) -> list[np.ndarray]:
    """Consecutive finite samples as (n, 2) arrays of (t, y). A NaN (or a long pause) ends a run."""
    t = np.asarray(t, float)
    y = np.asarray(y, float)
    n = min(len(t), len(y))
    if n == 0:
        return []
    t, y = t[:n], y[:n]
    keep = np.flatnonzero(np.isfinite(t) & np.isfinite(y))
    if len(keep) == 0:
        return []
    split = np.diff(keep) > 1
    if max_gap_s is not None:
        split |= np.diff(t[keep]) > max_gap_s
    return [np.column_stack([t[part], y[part]]) for part in np.split(keep, np.flatnonzero(split) + 1)]


def nice_ticks(low: float, high: float, target: int = 4) -> list[float]:
    span = high - low
    if not math.isfinite(span) or span <= 0:
        return [low]
    magnitude = 10.0 ** math.floor(math.log10(span / target))
    step = magnitude
    for multiple in (1.0, 2.0, 5.0, 10.0):
        step = multiple * magnitude
        if span / step <= target + 1:
            break
    first = math.ceil(low / step - 1e-9) * step
    count = int(math.floor((high - first) / step + 1e-9)) + 1
    return [round(first + i * step, 10) for i in range(max(0, count))]


def _tick_text(value: float, step: float) -> str:
    if step >= 1.0:
        return f"{value:.0f}"
    return f"{value:.{min(4, max(1, int(math.ceil(-math.log10(step)))))}f}"


def _time_text(seconds: float, step: float) -> str:
    if step < 1.0:
        return f"{seconds:.1f}s"
    whole = int(round(seconds))
    return f"{whole // 60}:{whole % 60:02d}"


class TracePlot(QWidget):
    """``set_series`` once per round, ``set_cursor`` as the scrubber moves. A click reports its time."""

    time_clicked = Signal(float)

    def __init__(self, title: str = "", unit: str = "", empty_message: str = "No data", parent=None) -> None:
        super().__init__(parent)
        self.title = title
        self.unit = unit
        self.empty_message = empty_message
        self.series: list[Series] = []
        self.cursor: float | None = None
        self._runs: list[list[np.ndarray]] = []
        self._t_range: tuple[float, float] | None = None
        self._y_range: tuple[float, float] | None = None
        self._cache: QPixmap | None = None
        self.setMinimumSize(240, 84)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    # ---- content ----

    def set_series(self, series: Sequence[Series | tuple], t_range: tuple[float, float] | None = None) -> None:
        """``series`` is up to three ``Series`` (or ``(name, t, y)`` tuples). ``t_range`` fixes the time axis."""
        if len(series) > MAX_SERIES:
            raise ValueError(f"a TracePlot shows at most {MAX_SERIES} series")
        self.series = []
        for position, item in enumerate(series):
            entry = item if isinstance(item, Series) else Series(*item)
            entry = Series(entry.name, np.asarray(entry.t, float), np.asarray(entry.y, float),
                           entry.colour or SERIES_COLOURS[position], entry.dashed, entry.max_gap_s)
            self.series.append(entry)
        self._runs = [finite_runs(entry.t, entry.y, entry.max_gap_s) for entry in self.series]
        points = [run for runs in self._runs for run in runs]
        if points:
            values = np.concatenate([run[:, 1] for run in points])
            low, high = float(values.min()), float(values.max())
            pad = 0.08 * (high - low) if high > low else max(0.5, 0.05 * abs(high))
            self._y_range = (low - pad, high + pad)
        else:
            self._y_range = None
        if t_range is not None:
            self.set_time_range(*t_range)
        elif points:
            times = np.concatenate([run[:, 0] for run in points])
            self.set_time_range(float(times.min()), float(times.max()))
        else:
            self._t_range = None
        self._invalidate()

    def set_time_range(self, start: float, end: float) -> None:
        if not (math.isfinite(start) and math.isfinite(end)):
            self._t_range = None
        else:
            self._t_range = (start, end if end > start else start + 1.0)
        self._invalidate()

    def set_cursor(self, t: float | None) -> None:
        if t != self.cursor:
            self.cursor = t
            self.update()

    def set_empty_message(self, text: str) -> None:
        self.empty_message = text
        self._invalidate()

    # ---- what is shown (also used by tests) ----

    def has_data(self) -> bool:
        return any(len(runs) > 0 for runs in self._runs)

    def runs(self, index: int = 0) -> list[np.ndarray]:
        """The separately drawn pieces of series ``index``."""
        return self._runs[index]

    def message(self) -> str:
        """The centred text shown instead of a plot ('' when there is data)."""
        return "" if self.has_data() else self.empty_message

    def y_range(self) -> tuple[float, float] | None:
        return self._y_range

    def time_range(self) -> tuple[float, float] | None:
        return self._t_range

    def value_at_cursor(self, index: int) -> float | None:
        """The newest sample of a series at or before the cursor, or None if there is none (a gap)."""
        if self.cursor is None or not 0 <= index < len(self.series):
            return None
        entry = self.series[index]
        if len(entry.t) == 0:
            return None
        i = int(np.searchsorted(entry.t, self.cursor, side="right")) - 1
        if i < 0 or not math.isfinite(entry.y[i]):
            return None
        limit = entry.max_gap_s if entry.max_gap_s is not None else 0.5
        if self.cursor - entry.t[i] > limit:
            return None
        return float(entry.y[i])

    # ---- geometry ----

    def plot_rect(self) -> QRectF:
        return QRectF(56.0, 22.0, max(10.0, self.width() - 56.0 - 12.0), max(10.0, self.height() - 22.0 - 20.0))

    def time_to_x(self, t: float) -> float:
        rect = self.plot_rect()
        start, end = self._t_range or (0.0, 1.0)
        return rect.left() + rect.width() * (t - start) / (end - start)

    def x_to_time(self, x: float) -> float:
        rect = self.plot_rect()
        start, end = self._t_range or (0.0, 1.0)
        fraction = min(1.0, max(0.0, (x - rect.left()) / rect.width()))
        return start + fraction * (end - start)

    def value_to_y(self, value: float) -> float:
        rect = self.plot_rect()
        low, high = self._y_range or (0.0, 1.0)
        return rect.bottom() - rect.height() * (value - low) / (high - low)

    # ---- input ----

    def _report(self, event) -> None:
        if self._t_range is not None and self.has_data():
            self.time_clicked.emit(self.x_to_time(event.position().x()))

    def mousePressEvent(self, event) -> None:  # noqa: N802 - Qt API
        if event.button() == Qt.MouseButton.LeftButton:
            self._report(event)

    def mouseMoveEvent(self, event) -> None:  # noqa: N802 - Qt API
        if event.buttons() & Qt.MouseButton.LeftButton:
            self._report(event)

    # ---- painting ----

    def _invalidate(self) -> None:
        self._cache = None
        self.update()

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt API
        self._cache = None
        super().resizeEvent(event)

    def _thin(self, run: np.ndarray, columns: int) -> np.ndarray:
        """More samples than pixels: keep each pixel column's minimum and maximum so spikes stay visible."""
        if len(run) <= 4 * columns:
            return run
        start, end = self._t_range
        column = np.clip(((run[:, 0] - start) / (end - start) * columns).astype(int), 0, columns - 1)
        first = np.flatnonzero(np.diff(column, prepend=column[0] - 1))
        out = np.empty((2 * len(first), 2))
        out[0::2, 0] = out[1::2, 0] = run[first, 0]
        out[0::2, 1] = np.minimum.reduceat(run[:, 1], first)
        out[1::2, 1] = np.maximum.reduceat(run[:, 1], first)
        return out

    def _render(self) -> QPixmap:
        ratio = self.devicePixelRatioF()
        pixmap = QPixmap(max(1, int(self.width() * ratio)), max(1, int(self.height() * ratio)))
        pixmap.setDevicePixelRatio(ratio)
        pixmap.fill(QColor(BACKGROUND))
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = self.plot_rect()
        painter.fillRect(rect, QColor(PLOT_BACKGROUND))
        painter.setFont(QFont("Segoe UI", 9, QFont.Weight.DemiBold))
        painter.setPen(QColor(TITLE_TEXT))
        heading = f"{self.title} ({self.unit})" if self.unit else self.title
        painter.drawText(QRectF(6.0, 2.0, self.width() - 12.0, 18.0), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, heading)

        if not self.has_data() or self._t_range is None or self._y_range is None:
            painter.setPen(QColor(AXIS_TEXT))
            painter.setFont(QFont("Segoe UI", 11))
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap, self.empty_message)
            painter.end()
            return pixmap

        painter.setFont(QFont("Segoe UI", 8))
        low, high = self._y_range
        y_ticks = nice_ticks(low, high, 3 if rect.height() < 110 else 4)
        y_step = y_ticks[1] - y_ticks[0] if len(y_ticks) > 1 else 1.0
        for value in y_ticks:
            y = self.value_to_y(value)
            painter.setPen(QPen(QColor(GRID), 1))
            painter.drawLine(QPointF(rect.left(), y), QPointF(rect.right(), y))
            painter.setPen(QColor(AXIS_TEXT))
            painter.drawText(QRectF(0.0, y - 8.0, rect.left() - 5.0, 16.0),
                             Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter, _tick_text(value, y_step))
        start, end = self._t_range
        t_ticks = nice_ticks(start, end, max(2, int(rect.width() // 110)))
        t_step = t_ticks[1] - t_ticks[0] if len(t_ticks) > 1 else 1.0
        for value in t_ticks:
            x = self.time_to_x(value)
            painter.setPen(QPen(QColor(GRID), 1))
            painter.drawLine(QPointF(x, rect.top()), QPointF(x, rect.bottom()))
            painter.setPen(QColor(AXIS_TEXT))
            painter.drawText(QRectF(x - 30.0, rect.bottom() + 2.0, 60.0, 16.0), Qt.AlignmentFlag.AlignCenter, _time_text(value, t_step))

        painter.setClipRect(rect.adjusted(-1.0, -1.0, 1.0, 1.0))
        columns = max(1, int(rect.width()))
        for entry, runs in zip(self.series, self._runs):
            pen = QPen(QColor(entry.colour), 1.6)
            if entry.dashed:
                pen.setStyle(Qt.PenStyle.DashLine)
            for run in runs:
                run = self._thin(run, columns)
                points = [QPointF(self.time_to_x(float(t)), self.value_to_y(float(v))) for t, v in run]
                if len(points) == 1:
                    dot = QPen(QColor(entry.colour), 4.0)
                    dot.setCapStyle(Qt.PenCapStyle.RoundCap)
                    painter.setPen(dot)
                    painter.drawPoint(points[0])
                else:
                    painter.setPen(pen)
                    painter.drawPolyline(QPolygonF(points))
        painter.end()
        return pixmap

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt API
        if self._cache is None:
            self._cache = self._render()
        painter = QPainter(self)
        painter.drawPixmap(0, 0, self._cache)
        if self.has_data() and self._t_range is not None:
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            rect = self.plot_rect()
            # Legend, right-aligned in the top margin: line sample, name, value at the cursor.
            painter.setFont(QFont("Segoe UI", 8, QFont.Weight.DemiBold))
            metrics = painter.fontMetrics()
            x = rect.right()
            for index in range(len(self.series) - 1, -1, -1):
                entry = self.series[index]
                value = self.value_at_cursor(index)
                text = entry.name if self.cursor is None else f"{entry.name} {'–' if value is None else format(value, '.3f')}"
                width = metrics.horizontalAdvance(text)
                x -= width
                painter.setPen(QColor(entry.colour))
                painter.drawText(QRectF(x, 2.0, width + 2.0, 18.0), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, text)
                pen = QPen(QColor(entry.colour), 2.0)
                if entry.dashed:
                    pen.setStyle(Qt.PenStyle.DashLine)
                painter.setPen(pen)
                painter.drawLine(QPointF(x - 22.0, 11.0), QPointF(x - 5.0, 11.0))
                x -= 34.0
            if self.cursor is not None:
                start, end = self._t_range
                if start <= self.cursor <= end:
                    cx = self.time_to_x(self.cursor)
                    painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)
                    painter.setPen(QPen(QColor(CURSOR), 1))
                    painter.drawLine(QPointF(cx, rect.top()), QPointF(cx, rect.bottom()))
        painter.end()
