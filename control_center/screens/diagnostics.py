"""Diagnostics: sim-to-real comparison and policy shadow. Kept apart from recording and normal operation.

Nothing on this screen arms, hands over or drives the robot, and nothing here can: it reads files,
reads the hub, and runs a checkpoint against the SIMULATOR. A policy's proposed action is shown and
never sent anywhere.
"""

from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from PySide6.QtCore import QCoreApplication, QPointF, QRectF, Qt, QTimer
from PySide6.QtGui import QBrush, QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import (QAbstractItemView, QComboBox, QDoubleSpinBox, QFileDialog, QFrame, QGridLayout, QHBoxLayout,
                               QHeaderView, QLabel, QLineEdit, QPushButton, QScrollArea, QSizePolicy, QStackedWidget,
                               QTableWidget, QTableWidgetItem, QTabWidget, QToolTip, QVBoxLayout, QWidget)

from control_center.backend import shadow, sim_compare
from control_center.backend.round_reader import RoundData
from control_center.widgets.arena_map import ArenaMapWidget, Marker, Trail

ROOT = Path(__file__).resolve().parents[2]

GREEN, AMBER, RED, GREY, BLUE, TEXT_DIM = "#55c889", "#e8b85b", "#ef6b73", "#7f8996", "#6fa8dc", "#9eabb8"
CLASS_COLOURS = {
    "stop": "#4a5563", "forward": "#55c889", "reverse": "#e8b85b", "turn_left": "#6fa8dc", "turn_right": "#b48ead",
    "forward_left": "#5fb7b0", "forward_right": "#8fbf6a", "reverse_left": "#c9a26b", "reverse_right": "#d08770",
}
TABLE_STYLE = ("QTableWidget { background: #0d1218; gridline-color: #2b3542; border: 1px solid #2b3542; }"
               "QHeaderView::section { background: #1b232d; color: #9eabb8; border: 0; padding: 4px; font-weight: 700; }"
               "QTableCornerButton::section { background: #1b232d; border: 0; }")
TAB_STYLE = ("QTabWidget::pane { border: 1px solid #2b3542; }"
             "QTabBar::tab { background: #171e27; color: #9eabb8; padding: 7px 16px; border: 1px solid #2b3542; }"
             "QTabBar::tab:selected { background: #263241; color: #d9dee5; font-weight: 700; }")
SIM_ONLY_STATEMENT = ("A model running in the simulator is not evidence that it can run from live observations. "
                      "Proposed actions are display only and are never sent: this screen has no way to send them.")
COMMAND_IS_NOT_MOTION = ("Command agreement and motion agreement are separate checks. Matching commands say nothing about "
                         "whether the robot moved the same way.")


def _card(title: str) -> tuple[QFrame, QVBoxLayout]:
    frame = QFrame()
    frame.setObjectName("panel")
    layout = QVBoxLayout(frame)
    layout.setContentsMargins(12, 10, 12, 12)
    layout.setSpacing(6)
    heading = QLabel(title)
    heading.setObjectName("panelTitle")
    layout.addWidget(heading)
    return frame, layout


def _scrolling(parent: QWidget) -> QVBoxLayout:
    """Put a vertical scroll area in `parent` and return the layout of its content."""
    scroll = QScrollArea()
    scroll.setWidgetResizable(True)
    scroll.setFrameShape(QFrame.Shape.NoFrame)
    body = QWidget()
    scroll.setWidget(body)
    wrapper = QVBoxLayout(parent)
    wrapper.setContentsMargins(0, 0, 0, 0)
    wrapper.addWidget(scroll)
    layout = QVBoxLayout(body)
    layout.setContentsMargins(10, 10, 10, 10)
    layout.setSpacing(8)
    return layout


def _caption(text: str) -> QLabel:
    label = QLabel(text)
    label.setStyleSheet("background: transparent;")
    return label


def _text(text: str = "", colour: str | None = None, bold: bool = False) -> QLabel:
    label = QLabel(text)
    label.setWordWrap(True)
    label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
    style = "background: transparent;"
    if colour:
        style += f" color: {colour};"
    if bold:
        style += " font-weight: 700;"
    label.setStyleSheet(style)
    return label


def _spin(minimum: float, maximum: float, step: float = 0.05, suffix: str = " s") -> QDoubleSpinBox:
    box = QDoubleSpinBox()
    box.setRange(minimum, maximum)
    box.setDecimals(2)
    box.setSingleStep(step)
    box.setSuffix(suffix)
    box.setKeyboardTracking(False)
    box.setMinimumWidth(96)
    return box


# ----------------------------------------------------------------------------------------------
# Timeline


@dataclass
class Bar:
    start: float
    end: float
    text: str
    colour: str
    tip: str = ""


@dataclass
class TimelineRow:
    title: str
    bars: list[Bar] = field(default_factory=list)
    empty_note: str = ""


class CommandTimelineWidget(QWidget):
    """Rows of coloured bars on one shared time axis (round time). Gaps are left empty, never filled."""

    GUTTER, ROW_H, ROW_GAP, TOP, AXIS_H = 176, 22, 6, 20, 34

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.rows: list[TimelineRow] = []
        self.window: tuple[float, float] | None = None
        self.markers: list[tuple[float, str, str]] = []
        self.blocked_after: float | None = None
        self.axis_caption = "round time (s)"
        self._range = (0.0, 10.0)
        self.setMouseTracking(True)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.set_content([])

    def set_content(self, rows: list[TimelineRow], window: tuple[float, float] | None = None,
                    markers: list[tuple[float, str, str]] | None = None, blocked_after: float | None = None,
                    axis_caption: str | None = None) -> None:
        self.rows, self.window, self.markers, self.blocked_after = rows, window, markers or [], blocked_after
        if axis_caption is not None:
            self.axis_caption = axis_caption
        points = [value for row in rows for bar in row.bars for value in (bar.start, bar.end)]
        points += [t for t, _, _ in self.markers] + (list(window) if window else [])
        low, high = (min(points), max(points)) if points else (0.0, 10.0)
        if high - low < 0.5:
            high = low + 0.5
        pad = 0.03 * (high - low)
        self._range = (low - pad, high + pad)
        self.setFixedHeight(self.TOP + max(1, len(rows)) * (self.ROW_H + self.ROW_GAP) + self.AXIS_H)
        self.update()

    def _x(self, t: float) -> float:
        low, high = self._range
        return self.GUTTER + (t - low) / (high - low) * max(10.0, self.width() - self.GUTTER - 12)

    def _row_top(self, index: int) -> float:
        return self.TOP + index * (self.ROW_H + self.ROW_GAP)

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt API
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        painter.fillRect(self.rect(), QColor("#11161d"))
        right = self.width() - 12.0
        bottom = self._row_top(max(1, len(self.rows))) - self.ROW_GAP
        small = QFont("Segoe UI", 8)
        painter.setFont(QFont("Segoe UI", 9))
        for index, row in enumerate(self.rows):
            top = self._row_top(index)
            lane = QRectF(self.GUTTER, top, right - self.GUTTER, self.ROW_H)
            painter.setPen(QColor(TEXT_DIM))
            painter.drawText(QRectF(0, top, self.GUTTER - 8, self.ROW_H),
                             Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter, row.title)
            painter.fillRect(lane, QColor("#0d1218"))
            if not row.bars:
                painter.setPen(QColor(GREY))
                painter.drawText(lane.adjusted(6, 0, 0, 0), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, row.empty_note)
            for bar in row.bars:
                x0, x1 = max(self.GUTTER, self._x(bar.start)), min(right, self._x(bar.end))
                if x1 <= x0:
                    continue
                rect = QRectF(x0, top, max(1.0, x1 - x0), self.ROW_H)
                painter.fillRect(rect, QColor(bar.colour))
                if rect.width() > 46:
                    painter.setPen(QColor("#0b0f14"))
                    painter.setFont(small)
                    painter.drawText(rect.adjusted(3, 0, -2, 0), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, bar.text)
                    painter.setFont(QFont("Segoe UI", 9))
        if self.window is not None:
            x0, x1 = self._x(self.window[0]), self._x(self.window[1])
            painter.fillRect(QRectF(x0, self.TOP - 4, x1 - x0, bottom - self.TOP + 8), QColor(111, 168, 220, 38))
            painter.setPen(QPen(QColor(BLUE), 1))
            painter.drawLine(QPointF(x0, self.TOP - 4), QPointF(x0, bottom + 4))
            painter.drawLine(QPointF(x1, self.TOP - 4), QPointF(x1, bottom + 4))
        if self.blocked_after is not None and self._x(self.blocked_after) < right:
            x0 = max(self.GUTTER, self._x(self.blocked_after))
            area = QRectF(x0, self.TOP - 4, right - x0, bottom - self.TOP + 8)
            painter.fillRect(area, QBrush(QColor(239, 107, 115, 120), Qt.BrushStyle.BDiagPattern))
        painter.setFont(small)
        merged: list[list] = []                     # markers at the same place share one label
        for t, text, colour in sorted(self.markers):
            x = self._x(t)
            if not self.GUTTER <= x <= right:
                continue
            if merged and abs(merged[-1][0] - x) < 3.0:
                merged[-1][1] += " · " + text
            else:
                merged.append([x, text, colour])
        for x, text, colour in merged:
            pen = QPen(QColor(colour), 1)
            pen.setStyle(Qt.PenStyle.DashLine)
            painter.setPen(pen)
            painter.drawLine(QPointF(x, self.TOP - 6), QPointF(x, bottom + 4))
            painter.drawText(QPointF(x + 3, 11), text)
        low, high = self._range
        step = _nice_step((high - low) / max(2.0, (right - self.GUTTER) / 90.0))
        painter.setPen(QColor("#5c6b7d"))
        painter.drawLine(QPointF(self.GUTTER, bottom + 6), QPointF(right, bottom + 6))
        tick = math.ceil(low / step) * step
        while tick <= high:
            x = self._x(tick)
            painter.drawLine(QPointF(x, bottom + 6), QPointF(x, bottom + 10))
            painter.drawText(QRectF(x - 40, bottom + 10, 80, 14), Qt.AlignmentFlag.AlignCenter, f"{tick:g}")
            tick += step
        painter.setPen(QColor(TEXT_DIM))
        painter.drawText(QRectF(0, bottom + 8, self.GUTTER - 8, 16), Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter, self.axis_caption)
        painter.end()

    def bar_at(self, x: float, y: float) -> Bar | None:
        for index, row in enumerate(self.rows):
            top = self._row_top(index)
            if top <= y <= top + self.ROW_H:
                for bar in row.bars:
                    if self._x(bar.start) <= x <= self._x(bar.end):
                        return bar
        return None

    def mouseMoveEvent(self, event) -> None:  # noqa: N802 - Qt API
        position = event.position()
        bar = self.bar_at(position.x(), position.y())
        if bar is not None and bar.tip:
            QToolTip.showText(event.globalPosition().toPoint(), bar.tip, self)
        else:
            QToolTip.hideText()


def _nice_step(raw: float) -> float:
    if raw <= 0:
        return 1.0
    power = 10.0 ** math.floor(math.log10(raw))
    for factor in (1.0, 2.0, 5.0, 10.0):
        if raw <= factor * power:
            return factor * power
    return 10.0 * power


def _command_bars(segments: list[sim_compare.CommandSegment], shift: float, time_name: str) -> list[Bar]:
    bars = []
    for segment in segments:
        wheels = "" if segment.left is None else f"  L={segment.left:g} R={segment.right:g}"
        tip = (f"{segment.label} ({segment.detail}){wheels}\n{time_name} {segment.start_s:.2f}–{segment.end_s:.2f} s"
               + (f"\nround time {segment.start_s + shift:.2f}–{segment.end_s + shift:.2f} s" if shift else ""))
        bars.append(Bar(segment.start_s + shift, segment.end_s + shift, segment.label, CLASS_COLOURS.get(segment.label, GREY), tip))
    return bars


def _pose_bars(track) -> list[Bar]:
    """Stretches of consecutive valid poses, in time. Anything else stays empty."""
    bars: list[Bar] = []
    start = previous = None
    colour = GREEN if track.source == "live" else "#3f8f6a"
    for i in range(len(track)):
        t = float(track.t[i])
        ok = bool(track.valid[i]) and (previous is None or t - previous <= sim_compare.POSE_MAX_STEP_S)
        if start is not None and not ok:
            bars.append(Bar(start, previous, "valid", colour, f"valid {track.source} pose {start:.2f}–{previous:.2f} s"))
            start = None
        if bool(track.valid[i]):
            if start is None:
                start = t
            previous = t
        else:
            previous = None
    if start is not None and previous is not None:
        bars.append(Bar(start, previous, "valid", colour, f"valid {track.source} pose {start:.2f}–{previous:.2f} s"))
    return bars


# ----------------------------------------------------------------------------------------------
# Sim-to-real


class SimToRealTab(QWidget):
    def __init__(self, hub, parent=None) -> None:
        super().__init__(parent)
        self.hub = hub
        self.sim_log: sim_compare.SimLog | None = None
        self.round: RoundData | None = None
        self.comparison: sim_compare.Comparison | None = None
        self.measured: sim_compare.MeasuredMotion | None = None
        self.motion: sim_compare.MotionAgreement | None = None
        self.arena_map: ArenaMapWidget | None = None
        self._loading = False

        outer = _scrolling(self)

        sources, sources_layout = _card("WHAT IS BEING COMPARED")
        grid = QGridLayout()
        grid.setHorizontalSpacing(10)
        self.sim_button = QPushButton("Choose simulator log…")
        self.sim_button.clicked.connect(self.choose_sim_log)
        self.sim_path = _text("No simulator log chosen", TEXT_DIM)
        self.sim_round_box = QComboBox()
        self.sim_round_box.setMinimumWidth(330)
        self.sim_round_box.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
        self.sim_round_box.currentIndexChanged.connect(self._on_inputs_changed)
        self.round_button = QPushButton("Choose recorded round…")
        self.round_button.clicked.connect(self.choose_round)
        self.round_path = _text("No recorded round chosen", TEXT_DIM)
        self.sim_note = _text("", TEXT_DIM)
        self.round_note = _text("", TEXT_DIM)
        grid.addWidget(self.sim_button, 0, 0)
        grid.addWidget(self.sim_path, 0, 1)
        grid.addWidget(self.sim_round_box, 0, 2)
        grid.addWidget(self.sim_note, 1, 1, 1, 2)
        grid.addWidget(self.round_button, 2, 0)
        grid.addWidget(self.round_path, 2, 1, 1, 2)
        grid.addWidget(self.round_note, 3, 1, 1, 2)
        grid.setColumnStretch(1, 1)
        sources_layout.addLayout(grid)
        outer.addWidget(sources)

        timeline_card, timeline_layout = _card("COMMANDS OVER TIME")
        offset_row = QHBoxLayout()
        offset_row.addWidget(_caption("Simulated t = 0 is at round time"))
        self.offset_box = _spin(-3600.0, 3600.0)
        self.offset_box.valueChanged.connect(self._on_inputs_changed)
        offset_row.addWidget(self.offset_box)
        self.align_button = QPushButton("Align to ROUND START marker")
        self.align_button.setEnabled(False)
        self.align_button.clicked.connect(self.align_to_round_start)
        offset_row.addWidget(self.align_button)
        self.align_note = _text("", TEXT_DIM)
        offset_row.addWidget(self.align_note, 1)
        timeline_layout.addLayout(offset_row)
        self.timeline = CommandTimelineWidget()
        timeline_layout.addWidget(self.timeline)
        legend = "   ".join(f'<span style="color:{colour}">■</span> {name}' for name, colour in CLASS_COLOURS.items())
        legend_label = QLabel(legend + '   <span style="color:#7f8996">empty = unknown / not recorded</span>')
        legend_label.setTextFormat(Qt.TextFormat.RichText)
        legend_label.setStyleSheet("background: transparent; font-size: 8pt;")
        timeline_layout.addWidget(legend_label)
        outer.addWidget(timeline_card)

        lower = QHBoxLayout()
        lower.setSpacing(8)
        bench, bench_layout = _card("SHORT STRAIGHT / TURN BENCH TEST")
        window_row = QHBoxLayout()
        window_row.addWidget(_caption("Window (round time): start"))
        self.start_box = _spin(-3600.0, 36000.0)
        self.end_box = _spin(-3600.0, 36000.0)
        self.end_box.setValue(2.0)
        self.start_box.valueChanged.connect(self._on_inputs_changed)
        self.end_box.valueChanged.connect(self._on_inputs_changed)
        window_row.addWidget(self.start_box)
        window_row.addWidget(_caption("end"))
        window_row.addWidget(self.end_box)
        self.whole_button = QPushButton("Use the whole simulated round")
        self.whole_button.clicked.connect(self.use_whole_sim_round)
        window_row.addWidget(self.whole_button)
        window_row.addStretch(1)
        bench_layout.addLayout(window_row)

        self.table = QTableWidget(2, 3)
        self.table.setStyleSheet(TABLE_STYLE)
        self.table.setHorizontalHeaderLabels(["Measured (camera)", "Simulated (log or typed)", "Measured − simulated"])
        self.table.setVerticalHeaderLabels(["Distance travelled (m)", "Heading change (deg, + = counter-clockwise)"])
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.verticalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Fixed)
        self.table.verticalHeader().setDefaultSectionSize(32)
        self.table.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.table.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.table.setFixedHeight(32 * 2 + self.table.horizontalHeader().sizeHint().height() + 6)
        self.sim_distance_edit = QLineEdit()
        self.sim_heading_edit = QLineEdit()
        for row, edit in enumerate((self.sim_distance_edit, self.sim_heading_edit)):
            edit.setPlaceholderText("type the simulated value")
            edit.setStyleSheet("background: #171e27; border: 1px solid #3b4a5d; padding: 3px;")
            edit.editingFinished.connect(self._on_inputs_changed)
            self.table.setCellWidget(row, 1, edit)
            for column in (0, 2):
                self.table.setItem(row, column, QTableWidgetItem("—"))
        bench_layout.addWidget(self.table)
        self.measured_note = _text("", TEXT_DIM)
        self.simulated_note = _text("", TEXT_DIM)
        bench_layout.addWidget(self.measured_note)
        bench_layout.addWidget(self.simulated_note)

        self.command_line = _text("COMMAND agreement: choose a simulator log and a recorded round")
        self.motion_line = _text("MOTION agreement: choose a recorded round")
        self.separate_note = _text(COMMAND_IS_NOT_MOTION, TEXT_DIM)
        self.warning_line = _text("", AMBER, bold=True)
        self.notes_line = _text("", TEXT_DIM)
        for label in (self.command_line, self.motion_line, self.separate_note, self.warning_line, self.notes_line):
            bench_layout.addWidget(label)
        bench_layout.addStretch(1)
        lower.addWidget(bench, 3)

        path_card, path_layout = _card("MEASURED PATH (CAMERA)")
        self.map_stack = QStackedWidget()
        self.no_pose_note = _text("Choose a recorded round.", TEXT_DIM, bold=True)
        self.no_pose_note.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.no_pose_note.setMinimumSize(240, 240)
        self.map_holder = QWidget()
        self.map_holder_layout = QVBoxLayout(self.map_holder)
        self.map_holder_layout.setContentsMargins(0, 0, 0, 0)
        self.map_stack.addWidget(self.no_pose_note)
        self.map_stack.addWidget(self.map_holder)
        self.map_stack.setMinimumHeight(340)
        path_layout.addWidget(self.map_stack, 1)
        lower.addWidget(path_card, 2)
        outer.addLayout(lower, 1)
        self._refresh()

    # ---- choosing inputs ----

    def choose_sim_log(self) -> None:
        start = ROOT / "artifacts"
        name, _ = QFileDialog.getOpenFileName(self, "Choose a simulator play log", str(start if start.is_dir() else ROOT),
                                              "Simulator play logs (*.jsonl);;All files (*)")
        if name:
            self.load_sim_log(name)

    def choose_round(self) -> None:
        root = Path(self.hub.options.rounds_root)
        name = QFileDialog.getExistingDirectory(self, "Choose a recorded round folder", str(root if root.is_dir() else ROOT))
        if name:
            self.load_round(name)

    def load_sim_log(self, path: Path | str) -> bool:
        try:
            log = sim_compare.load_sim_log(path)
        except (OSError, UnicodeDecodeError) as error:
            self.sim_log = None
            self.sim_path.setText(f"Could not read {path}: {error}")
            self._refresh()
            return False
        self.sim_log = log
        self.sim_path.setText(str(log.path))
        self._loading = True
        self.sim_round_box.clear()
        for sim_round in log.rounds:
            self.sim_round_box.addItem(sim_round.describe())
        self._loading = False
        notes = [f"Step fields in this log: {', '.join(log.step_fields) or 'none'}."] + [note[:1].upper() + note[1:] + "." for note in log.notes]
        self.sim_note.setText(" ".join(notes))
        self._default_window()
        self._refresh()
        return bool(log.rounds)

    def load_round(self, folder: Path | str) -> bool:
        try:
            data = RoundData(folder)
        except Exception as error:      # not a round folder, or an unreadable one: say so, keep the screen alive
            self.round = None
            self.round_path.setText(f"Could not open {folder}: {error}")
            self.round_note.setText("")
            self._refresh()
            return False
        self.round = data
        self.round_path.setText(str(data.folder))
        starts = sim_compare.round_start_times(data)
        parts = [f"{data.duration_s:.1f} s", f"{len(data.controller_status)} controller status records",
                 f"{len(data.pc_sent)} PC-sent commands",
                 f"ROUND START marked at {', '.join(f'{t:.2f} s' for t in starts)}" if starts else "no ROUND START marker"]
        if data.manifest.get("mock"):
            parts.insert(0, "MOCK round: synthetic camera and replayed controller lines, not hardware")
        self.round_note.setText(" · ".join(parts) + ("" if not data.warnings else " · " + "; ".join(data.warnings)))
        self.align_button.setEnabled(bool(starts))
        self.align_note.setText("" if starts else "This round has no ROUND START marker: set the offset by hand.")
        if starts:
            self._set_quietly(self.offset_box, starts[0])
            self.align_note.setText(f"Offset set to the ROUND START marker at {starts[0]:.2f} s.")
        self._rebuild_map()
        self._default_window()
        self._refresh()
        return True

    def align_to_round_start(self) -> None:
        if self.round is None:
            return
        starts = sim_compare.round_start_times(self.round)
        if starts:
            self.offset_box.setValue(starts[0])
            self.align_note.setText(f"Offset set to the ROUND START marker at {starts[0]:.2f} s.")

    def use_whole_sim_round(self) -> None:
        sim_round = self.current_sim_round()
        if sim_round is None:
            return
        offset = self.offset_box.value()
        self._set_quietly(self.start_box, offset + sim_round.start_s)
        self._set_quietly(self.end_box, offset + sim_round.end_s)
        self._refresh()

    def current_sim_round(self) -> sim_compare.SimRound | None:
        if self.sim_log is None or not self.sim_log.rounds:
            return None
        index = self.sim_round_box.currentIndex()
        return self.sim_log.rounds[index] if 0 <= index < len(self.sim_log.rounds) else self.sim_log.rounds[0]

    def set_window(self, t_start: float, t_end: float) -> None:
        self._set_quietly(self.start_box, t_start)
        self._set_quietly(self.end_box, t_end)
        self._refresh()

    def _set_quietly(self, box: QDoubleSpinBox, value: float) -> None:
        box.blockSignals(True)
        box.setValue(value)
        box.blockSignals(False)

    def _default_window(self) -> None:
        offset = self.offset_box.value()
        sim_round = self.current_sim_round()
        end = offset + (sim_round.end_s if sim_round is not None else 2.0)
        if self.round is not None and self.round.duration_s > offset:
            end = min(end, self.round.duration_s)
        self._set_quietly(self.start_box, offset)
        self._set_quietly(self.end_box, max(end, offset + 0.1))

    def _on_inputs_changed(self, *args) -> None:
        if not self._loading:
            self._refresh()

    # ---- map ----

    def _rebuild_map(self) -> None:
        if self.arena_map is not None:
            self.map_holder_layout.removeWidget(self.arena_map)
            self.arena_map.deleteLater()
            self.arena_map = None
        if self.round is None or sim_compare.pick_pose_track(self.round) is None:
            return
        arena = self.round.manifest.get("arena") or self.hub.overlay_config.arena
        try:
            self.arena_map = ArenaMapWidget(dict(arena))
        except Exception:               # a manifest from an older build without the arena outline
            self.arena_map = ArenaMapWidget(dict(self.hub.overlay_config.arena))
        self.map_holder_layout.addWidget(self.arena_map)

    def _refresh_map(self) -> None:
        if self.round is None:
            self.no_pose_note.setText("Choose a recorded round.")
            self.map_stack.setCurrentWidget(self.no_pose_note)
            return
        track = sim_compare.pick_pose_track(self.round)
        if track is None or self.arena_map is None:
            self.no_pose_note.setText("No calibrated pose in this round.\n\nNo measured path can be drawn and no distance or "
                                      "heading can be measured from it.")
            self.map_stack.setCurrentWidget(self.no_pose_note)
            return
        offline = track.source != "live"
        trails = [Trail(track.segments(), "#4a5563", offline, "whole round", 1.5)]
        markers: list[Marker] = []
        if self.measured is not None and self.measured.available and self.measured.path:
            trails.append(Trail(self.measured.path, GREEN, offline, "window", 2.5))
            first, last = self.measured.path[0][0], self.measured.path[-1][-1]
            markers.append(Marker(float(first[0]), float(first[1]), None, GREEN, "window start", hollow=True))
            end_pose = track.at(self.measured.t_end, max_age_s=1e9)
            heading = end_pose["heading_deg"] if end_pose and end_pose["valid"] else None
            markers.append(Marker(float(last[0]), float(last[1]), heading, GREEN, "window end"))
        kind = "reconstructed OFFLINE after the round (dashed)" if offline else "estimated LIVE during the round"
        self.arena_map.set_content(trails, markers, f"Camera pose track {kind}. Gaps are not joined.", "")
        self.map_stack.setCurrentWidget(self.map_holder)

    # ---- the comparison ----

    def _typed(self, edit: QLineEdit) -> float | None:
        text = edit.text().strip().replace(",", ".")
        if not text:
            edit.setToolTip("")
            return None
        try:
            value = float(text)
        except ValueError:
            value = float("nan")
        if not math.isfinite(value):
            edit.setToolTip("Not a number: ignored")
            return None
        edit.setToolTip("")
        return value

    def _refresh(self) -> None:
        sim_round = self.current_sim_round()
        offset, t_start, t_end = self.offset_box.value(), self.start_box.value(), self.end_box.value()
        typed_distance, typed_heading = self._typed(self.sim_distance_edit), self._typed(self.sim_heading_edit)
        typed = None
        if typed_distance is not None or typed_heading is not None:
            typed = sim_compare.typed_simulated_motion(typed_distance, typed_heading)
        self.comparison = None
        self.measured = None
        self.motion = None
        simulated = typed

        rows: list[TimelineRow] = []
        markers: list[tuple[float, str, str]] = []
        blocked_after = None
        if sim_round is not None:
            rows.append(TimelineRow("SIM blue command", _command_bars(sim_compare.command_timeline(sim_round), offset, "simulated time"),
                                    "this log has no blue command"))
            markers.append((offset, "sim t=0", BLUE))
        else:
            rows.append(TimelineRow("SIM blue command", [], "no simulator log chosen"))
        if self.round is not None:
            rows.append(TimelineRow("Controller L/R (reported)", _command_bars(sim_compare.controller_timeline(self.round), 0.0, "round time"),
                                    "no controller status in this round"))
            rows.append(TimelineRow("PC sent (not confirmed)", _command_bars(sim_compare.pc_sent_timeline(self.round), 0.0, "round time"),
                                    "no PC-sent commands in this round"))
            track = sim_compare.pick_pose_track(self.round)
            rows.append(TimelineRow("Camera pose valid", [] if track is None else _pose_bars(track), "no calibrated pose in this round"))
            markers += [(t, "ROUND START", AMBER) for t in sim_compare.round_start_times(self.round)]
        else:
            rows.append(TimelineRow("Recorded round", [], "no recorded round chosen"))

        if self.round is not None and sim_round is not None:
            self.comparison = sim_compare.compare(self.sim_log, sim_round, self.round, offset, t_start, t_end, simulated=typed)
            self.measured, simulated = self.comparison.motion.measured, self.comparison.motion.simulated
            blocked_after = self.comparison.not_comparable_after_s
            self.command_line.setText("COMMAND agreement — " + self.comparison.command_text())
            self.motion_line.setText("MOTION agreement — " + self.comparison.motion_text())
            self.warning_line.setText(self.comparison.warning_text())
            self.warning_line.setStyleSheet(f"background: transparent; font-weight: 700; color: {RED if blocked_after is not None else AMBER};")
            self.notes_line.setText(" ".join(self.comparison.notes))
            if blocked_after is not None:
                markers.append((blocked_after, "simulated contact: not comparable after this", RED))
        else:
            missing = "a simulator log" if sim_round is None else "a recorded round"
            self.command_line.setText(f"COMMAND agreement — not available: choose {missing}")
            if self.round is None:
                self.motion_line.setText("MOTION agreement — not available: choose a recorded round")
            else:
                # Motion needs no simulator log when the simulated values are typed.
                self.measured = sim_compare.measured_motion(self.round, t_start, t_end)
                if simulated is None:
                    simulated = sim_compare.typed_simulated_motion(None, None)
                self.motion = sim_compare.motion_agreement(self.measured, simulated)
                self.motion_line.setText("MOTION agreement — " + self.motion.text())
            self.warning_line.setText("" if self.sim_log is None else self.sim_log.contact_note[:1].upper() + self.sim_log.contact_note[1:] + ".")
            self.notes_line.setText("")
        if self.comparison is not None:
            self.motion = self.comparison.motion
        self.timeline.set_content(rows, (t_start, t_end) if t_end > t_start else None, markers, blocked_after)
        self._fill_table(self.measured, simulated)
        self._refresh_map()

    def _fill_table(self, measured: sim_compare.MeasuredMotion | None, simulated: sim_compare.SimulatedMotion | None) -> None:
        def cell(row: int, column: int, text: str) -> None:
            self.table.item(row, column).setText(text)

        if measured is None:
            cell(0, 0, "—")
            cell(1, 0, "—")
            self.measured_note.setText("Measured: choose a recorded round.")
        elif not measured.available:
            cell(0, 0, "not available")
            cell(1, 0, "not available")
            self.measured_note.setText("Measured: " + measured.reason + ".")
        else:
            cell(0, 0, f"{measured.distance_m:.3f}")
            cell(1, 0, f"{measured.heading_change_deg:+.1f}")
            self.measured_note.setText(
                f"Measured from the {measured.track_note}: {measured.known_s:.2f} s of valid pose in {measured.runs} run(s); "
                f"position unknown for {measured.unknown_s:.2f} s of the {measured.t_end - measured.t_start:.2f} s window "
                f"(not bridged). Straight-line displacement inside the runs: {measured.straight_line_m:.3f} m."
                + (f" {measured.tag_switches} marker-switch step(s) excluded." if measured.tag_switches else ""))
        from_log = simulated is not None and simulated.available and simulated.source != "typed by the user"
        for row, edit, value in ((0, self.sim_distance_edit, None if not from_log else simulated.distance_m),
                                 (1, self.sim_heading_edit, None if not from_log else simulated.heading_change_deg)):
            edit.setPlaceholderText("type the simulated value" if value is None else f"{value:.3f} (from the log; type to override)")
        if simulated is None:
            self.simulated_note.setText("Simulated: " + (self.sim_log.pose_note[:1].upper() + self.sim_log.pose_note[1:] + "."
                                                          if self.sim_log is not None else "choose a simulator log, or type the values."))
        elif simulated.available:
            self.simulated_note.setText(f"Simulated values: {simulated.source}.")
        else:
            self.simulated_note.setText("Simulated: " + simulated.reason + ".")
        motion = self.motion
        differences = (None, None)
        if motion is not None and motion.comparable:
            differences = (motion.distance_difference_m, motion.heading_difference_deg)
        blocked = motion is not None and not motion.comparable
        cell(0, 2, "not comparable" if blocked else "—" if differences[0] is None else f"{differences[0]:+.3f}")
        cell(1, 2, "not comparable" if blocked else "—" if differences[1] is None else f"{differences[1]:+.1f}")


# ----------------------------------------------------------------------------------------------
# Shadow


class StateChip(QLabel):
    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumWidth(150)
        self.show_state(False, GREY)

    def show_state(self, active: bool, colour: str) -> None:
        if active:
            self.setStyleSheet(f"background: {colour}; color: #0b0f14; font-weight: 800; padding: 8px 14px; border-radius: 4px;")
        else:
            self.setStyleSheet(f"background: #1b232d; color: {colour}; font-weight: 700; padding: 8px 14px; "
                               f"border: 1px solid #2b3542; border-radius: 4px;")


class ShadowTab(QWidget):
    def __init__(self, hub, parent=None) -> None:
        super().__init__(parent)
        self.hub = hub
        self.info: shadow.CheckpointInfo | None = None
        self.runner: shadow.SimShadowRunner | None = None
        self._last_shown = 0.0

        outer = _scrolling(self)

        outer.addWidget(_text(SIM_ONLY_STATEMENT, AMBER, bold=True))

        checkpoint, layout = _card("CHECKPOINT")
        row = QHBoxLayout()
        self.choose_button = QPushButton("Choose checkpoint…")
        self.choose_button.clicked.connect(self.choose_checkpoint)
        self.path_label = _text("No checkpoint chosen", TEXT_DIM)
        row.addWidget(self.choose_button)
        row.addWidget(self.path_label, 1)
        layout.addLayout(row)
        self.kind_label = _text("—", bold=True)
        self.kind_label.setStyleSheet("background: transparent; font-size: 15pt; font-weight: 800;")
        self.evidence_label = _text("The label comes from the checkpoint's metadata (its run record and the timestep count "
                                    "stored in the zip), never from the file name.", TEXT_DIM)
        self.facts_label = _text("", None)
        self.warnings_label = _text("", AMBER)
        for label in (self.kind_label, self.evidence_label, self.facts_label, self.warnings_label):
            layout.addWidget(label)
        self.requirements = QTableWidget(0, 4)
        self.requirements.setStyleSheet(TABLE_STYLE)
        self.requirements.setHorizontalHeaderLabels(["Observation values", "What it is", "In the simulator it comes from",
                                                     "On the real robot it would have to come from"])
        self.requirements.verticalHeader().setVisible(False)
        self.requirements.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.requirements.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.requirements.setWordWrap(True)
        header = self.requirements.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        self.requirements.verticalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.requirements.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.requirements.setFixedHeight(60)
        layout.addWidget(self.requirements)
        outer.addWidget(checkpoint)

        states, layout = _card("POLICY STATE")
        chips = QHBoxLayout()
        self.chips = {"OFF": StateChip("OFF"), "SIM": StateChip("SIM"), "LIVE SHADOW": StateChip("LIVE SHADOW")}
        for chip in self.chips.values():
            chips.addWidget(chip)
        chips.addStretch(1)
        layout.addLayout(chips)
        self.state_label = _text("", TEXT_DIM)
        layout.addWidget(self.state_label)
        outer.addWidget(states)

        sim, layout = _card("SIM — THE CHECKPOINT ON SIMULATOR OBSERVATIONS")
        self.sim_button = QPushButton("Run on simulator observations (SIM)")
        self.sim_button.setEnabled(False)
        self.sim_button.clicked.connect(self.toggle_sim)
        row = QHBoxLayout()
        row.addWidget(self.sim_button)
        self.sim_status = _text("Choose a checkpoint first.", TEXT_DIM)
        row.addWidget(self.sim_status, 1)
        layout.addLayout(row)
        grid = QGridLayout()
        grid.setHorizontalSpacing(14)
        self.sim_values: dict[str, QLabel] = {}
        for index, name in enumerate(("Checkpoint", "Input source", "Simulated time", "Observation", "Proposed action",
                                      "Decodes to", "Action probabilities")):
            title = _text(name, TEXT_DIM)
            value = _text("—")
            grid.addWidget(title, index, 0, Qt.AlignmentFlag.AlignTop)
            grid.addWidget(value, index, 1)
            self.sim_values[name] = value
        grid.setColumnStretch(1, 1)
        layout.addLayout(grid)
        layout.addWidget(_text("The simulated robot follows these actions inside the simulator (frozen configuration "
                               f"{shadow.FROZEN_CONFIG.relative_to(ROOT).as_posix()}, default spawn, opponent standing still). "
                               "Nothing leaves this computer.", TEXT_DIM))
        outer.addWidget(sim)

        live, layout = _card("LIVE SHADOW — THE CHECKPOINT ON LIVE OBSERVATIONS")
        self.live_state = _text("UNAVAILABLE", RED, bold=True)
        self.live_state.setStyleSheet(f"background: transparent; color: {RED}; font-size: 13pt; font-weight: 800;")
        layout.addWidget(self.live_state)
        self.live_missing = _text("")
        layout.addWidget(self.live_missing)
        self.live_button = QPushButton("Run on live observations (LIVE SHADOW)")
        self.live_button.setEnabled(False)
        self.live_button.setToolTip("Unavailable: the inputs listed above are missing")
        row = QHBoxLayout()
        row.addWidget(self.live_button)
        row.addStretch(1)
        layout.addLayout(row)
        outer.addWidget(live)
        outer.addStretch(1)

        self._timer = QTimer(self)
        self._timer.timeout.connect(self.refresh_status)
        self._timer.start(1000)
        self.refresh_status()

    # ---- checkpoint ----

    def choose_checkpoint(self) -> None:
        start = shadow.RUNS_DIR if shadow.RUNS_DIR.is_dir() else ROOT
        name, _ = QFileDialog.getOpenFileName(self, "Choose a checkpoint", str(start), "Checkpoints (*.zip);;All files (*)")
        if name:
            self.load_checkpoint(name)

    def load_checkpoint(self, path: Path | str) -> bool:
        if self.runner is not None:
            self.sim_status.setText("Stop SIM before choosing another checkpoint.")
            return False
        try:
            info = shadow.inspect_checkpoint(path)
        except Exception as error:      # unreadable file: report it, keep the screen alive
            self.info = None
            self.path_label.setText(str(path))
            self.kind_label.setText("not a readable checkpoint")
            self.evidence_label.setText(str(error))
            self.facts_label.setText("")
            self.warnings_label.setText("")
            self.requirements.setRowCount(0)
            self.sim_button.setEnabled(False)
            self.refresh_status()
            return False
        self.info = info
        self.path_label.setText(str(info.path))
        self.kind_label.setText(info.kind)
        self.kind_label.setStyleSheet(f"background: transparent; font-size: 15pt; font-weight: 800; "
                                      f"color: {GREY if info.kind == shadow.KIND_UNKNOWN else BLUE};")
        self.evidence_label.setText("Evidence (metadata, not the file name): " + info.evidence + ".")
        steps = "not recorded" if info.num_timesteps is None else f"{info.num_timesteps:,}"
        self.facts_label.setText(
            f"PPO timesteps in the zip: {steps}\n"
            f"Observation: {info.observation_size} values · profile {info.observation_profile or 'unknown'} ({info.observation_profile_source})\n"
            f"Actions: {info.action_count} · profile {info.action_profile or 'unknown'} ({info.action_profile_source})\n"
            f"Tactics: {', '.join(info.tactics) or 'unknown'}\n"
            f"Previous-action encoding: {info.previous_action_encoding or 'unknown'} ({info.previous_action_encoding_source})")
        self.warnings_label.setText("\n".join("Warning: " + warning for warning in info.warnings))
        self.requirements.setRowCount(len(info.requirements))
        for row, item in enumerate(info.requirements):
            for column, text in enumerate((item.index, item.name, item.in_simulator, item.live_source)):
                self.requirements.setItem(row, column, QTableWidgetItem(text))
        self._fit_requirements()
        QTimer.singleShot(0, self._fit_requirements)
        self.sim_button.setEnabled(info.layout_known)
        self.sim_status.setText("Ready. Loading takes about 15 s, and up to two minutes the first time after a restart." if info.layout_known else
                                "This checkpoint's observation layout is not recorded, so the simulator cannot be set up for it.")
        self.sim_values["Checkpoint"].setText(info.label)
        for name in ("Input source", "Simulated time", "Observation", "Proposed action", "Decodes to", "Action probabilities"):
            self.sim_values[name].setText("—")
        self.refresh_status()
        return True

    def _fit_requirements(self) -> None:
        """Make the table exactly as tall as its rows, so it never needs its own scroll bar."""
        table = self.requirements
        table.resizeRowsToContents()
        rows = sum(table.rowHeight(row) for row in range(table.rowCount()))
        table.setFixedHeight(table.horizontalHeader().height() + rows + 6)

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt API
        super().resizeEvent(event)
        QTimer.singleShot(0, self._fit_requirements)

    # ---- status ----

    def refresh_status(self) -> None:
        mode = self.hub.policy.get("mode", "OFF")
        available, missing = shadow.live_shadow_status(self.info, self.hub)
        self.chips["OFF"].show_state(mode == "OFF", GREY)
        self.chips["SIM"].show_state(mode == "SIM", AMBER)
        self.chips["LIVE SHADOW"].show_state(mode == "LIVE SHADOW", RED if not available else BLUE)
        described = {"OFF": "OFF: no policy is running.",
                     "SIM": "SIM: a checkpoint is reading SIMULATOR observations and driving the simulated robot only.",
                     "LIVE SHADOW": "LIVE SHADOW: a checkpoint is reading live observations; its proposals are display only."}
        self.state_label.setText(described.get(mode, mode) + "  LIVE SHADOW is "
                                 + ("available." if available else "UNAVAILABLE (see below)."))
        if available:
            # Reaching this needs a live observation adapter, which does not exist; the button stays disabled
            # until someone builds one and wires it here.
            self.live_state.setText("Inputs present, but no LIVE SHADOW runner exists in this build")
            self.live_missing.setText("")
        else:
            self.live_state.setText("UNAVAILABLE")
            self.live_missing.setText("Missing inputs:\n" + "\n".join("  •  " + item for item in missing))
        self.live_button.setEnabled(False)

    # ---- SIM ----

    def toggle_sim(self) -> None:
        if self.runner is None:
            self.start_sim()
        else:
            self.stop_sim()

    def start_sim(self) -> bool:
        if self.runner is not None or self.info is None or not self.info.layout_known:
            return False
        runner = shadow.SimShadowRunner(self.hub, self.info)
        runner.progress.connect(self.sim_status.setText)
        runner.decision.connect(self._on_decision)
        runner.failed.connect(self._on_failed)
        runner.finished.connect(self._on_finished)
        self.runner = runner
        self.sim_button.setText("Stop SIM")
        self.choose_button.setEnabled(False)
        self.sim_values["Checkpoint"].setText(self.info.label)
        self.sim_values["Input source"].setText("SIMULATOR (simulated observations; simulated robot)")
        self.sim_status.setText("Starting…")
        runner.start()
        return True

    def stop_sim(self) -> None:
        if self.runner is None:
            return
        self.runner.stop()
        self.sim_button.setEnabled(False)
        self.sim_status.setText("Stopping… (if the checkpoint is still loading, it stops when the load finishes)")
        self.refresh_status()

    def _on_decision(self, decision: shadow.ShadowDecision) -> None:
        now = time.monotonic()
        if now - self._last_shown < 0.1 and decision.observation_valid and decision.round_end is None:
            return
        self._last_shown = now
        values = self.sim_values
        values["Simulated time"].setText(f"t = {decision.sim_time_s:.2f} s · simulated round {decision.round_index} · decision {decision.index}"
                                         + (f" · round ended: {decision.round_end}" if decision.round_end else ""))
        if decision.observation_valid:
            size = "?" if self.info is None else self.info.observation_size
            values["Observation"].setText(f"VALID ({size} finite values, simulator)")
            values["Observation"].setStyleSheet(f"background: transparent; color: {GREEN}; font-weight: 700;")
            values["Proposed action"].setText(f"{decision.tactic}   (display only, never sent)")
            left, right = decision.wheel_command
            values["Decodes to"].setText(f"L {left:+.2f}  R {right:+.2f}  simulator wheel units (negative = forward), simulated robot only")
        else:
            values["Observation"].setText(f"INVALID: {decision.observation_problem}")
            values["Observation"].setStyleSheet(f"background: transparent; color: {RED}; font-weight: 700;")
            values["Proposed action"].setText("none: no action is proposed from an invalid observation")
            values["Decodes to"].setText("—")
        if decision.probabilities:
            values["Action probabilities"].setText("   ".join(f"{name} {value:.2f}" for name, value in decision.probabilities.items()))
        else:
            values["Action probabilities"].setText("not available")
        self.refresh_status()

    def _on_failed(self, message: str) -> None:
        self.sim_status.setText("SIM failed: " + message)

    def _on_finished(self) -> None:
        runner, self.runner = self.runner, None
        if runner is not None:
            runner.wait(2000)
            if runner.error is None:
                self.sim_status.setText("SIM stopped.")
        self.sim_button.setText("Run on simulator observations (SIM)")
        self.sim_button.setEnabled(self.info is not None and self.info.layout_known)
        self.choose_button.setEnabled(True)
        self.refresh_status()

    def shutdown(self) -> None:
        self._timer.stop()
        runner = self.runner
        if runner is not None:
            runner.stop()
            if not runner.wait(5000):
                _LINGERING.append(runner)       # still importing torch: keep it alive until it returns


_LINGERING: list = []


# ----------------------------------------------------------------------------------------------


class DiagnosticsScreen(QWidget):
    """Two diagnostic workflows: sim-to-real comparison, and policy shadow."""

    def __init__(self, hub, parent=None) -> None:
        super().__init__(parent)
        self.hub = hub
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        self.tabs = QTabWidget()
        self.tabs.setStyleSheet(TAB_STYLE)
        self.sim_to_real = SimToRealTab(hub)
        self.shadow = ShadowTab(hub)
        self.tabs.addTab(self.sim_to_real, "Sim-to-real")
        self.tabs.addTab(self.shadow, "Shadow")
        layout.addWidget(self.tabs)
        app = QCoreApplication.instance()
        if app is not None:
            app.aboutToQuit.connect(self.shutdown)
        # Importing the simulator package costs about a second; do it off the UI thread before it is needed.
        threading.Thread(target=_prewarm_simulator_import, name="diagnostics-prewarm").start()

    def load_sim_log(self, path: Path | str) -> bool:
        return self.sim_to_real.load_sim_log(path)

    def load_round(self, folder: Path | str) -> bool:
        return self.sim_to_real.load_round(folder)

    def load_checkpoint(self, path: Path | str) -> bool:
        return self.shadow.load_checkpoint(path)

    def shutdown(self) -> None:
        """Stop the SIM runner (if any) and report the policy OFF. Safe to call more than once."""
        self.shadow.shutdown()

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt API
        self.shutdown()
        super().closeEvent(event)


def _prewarm_simulator_import() -> None:
    try:
        import autobot.sim.cli  # noqa: F401
    except Exception:           # the Shadow tab reports the real error when a checkpoint is chosen
        pass
