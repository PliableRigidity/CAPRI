"""Setup screen: camera, floor calibration, marker checks and read-only controller / IMU diagnostics.

Everything here is kept off the Operate screen on purpose. The screen reads from the hub and asks
the hub to do things; it never opens a camera or a port itself, and it has no control that sends
anything to the robot (no arm, no mode, no drive).
"""

from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (QAbstractItemView, QFileDialog, QFrame, QGridLayout, QHBoxLayout, QHeaderView, QLabel,
                               QLineEdit, QPlainTextEdit, QPushButton, QScrollArea, QSpinBox, QSplitter, QTableWidget,
                               QTableWidgetItem, QTabWidget, QVBoxLayout, QWidget)

from control_center.backend.setup_tools import (FLOOR_LEVEL_NOTE, LIVE_PROFILE_LABEL, MARKER_ROW_LABELS, ONE_PORT_NOTE,
                                                READ_ONLY_NOTE, SINGLE_TAG_HOWTO, SPREAD_STATEMENT, WHEN_TO_RECALIBRATE,
                                                CalibrationModel, RawLineLog, SingleTagCheck, camera_rows,
                                                controller_summary, firmware_text, imu_report, list_profiles,
                                                marker_layout_rows, marker_placeholder_note, marker_report)
from control_center.tools.arena_overlay import draw_arena, draw_reference_points
from control_center.tools.overlay_core import LIVE_PROFILE, REPO_ROOT
from control_center.widgets.frame_view import FrameView

REFRESH_MS = 150                      # about 7 Hz
SINGLE_TAG_SAMPLES = 20
LEVEL_COLOURS = {"ok": "#55c889", "warn": "#e8b85b", "bad": "#ef6b73", "off": "#9aa5b1", "plain": "#d9dee5"}
FIT_LEVELS = {"ok": "ok", "weak": "warn", "insufficient": "bad"}

SETUP_STYLE = """
QLabel { background: transparent; }
QFrame#panel { background: #171e27; border: 1px solid #2b3542; border-radius: 5px; }
QLabel#panelTitle { color: #9eabb8; font-size: 10pt; font-weight: 700; letter-spacing: 1px; }
QLabel#rowName { color: #9eabb8; }
QLabel#hint { color: #9eabb8; }
QLabel#profileLabel { color: #6fa8dc; font-size: 12pt; font-weight: 800; }
QLabel#fitLine { font-size: 13pt; font-weight: 800; }
QLabel#mono { font-family: Consolas; }
QTabWidget::pane { border: 1px solid #2b3542; border-radius: 5px; top: -1px; }
QTabBar::tab { background: #171e27; color: #9eabb8; padding: 8px 18px; border: 1px solid #2b3542; border-bottom: none;
               margin-right: 2px; font-weight: 650; }
QTabBar::tab:selected { background: #263241; color: #d9dee5; }
QTableWidget { background: #0d1218; alternate-background-color: #121922; gridline-color: #2b3542;
               border: 1px solid #2b3542; selection-background-color: #2f4a66; selection-color: #ffffff; }
QHeaderView::section { background: #171e27; color: #9eabb8; border: none; border-bottom: 1px solid #2b3542;
                       padding: 4px 6px; font-weight: 700; }
QLineEdit, QSpinBox { background: #0d1218; border: 1px solid #3b4a5d; border-radius: 4px; padding: 5px 6px; }
QScrollArea { border: none; background: transparent; }
QWidget#scrollBody { background: transparent; }
"""


def _label(text: str = "", name: str | None = None, selectable: bool = False) -> QLabel:
    label = QLabel(text)
    label.setTextFormat(Qt.TextFormat.PlainText)
    label.setWordWrap(True)
    if name:
        label.setObjectName(name)
    if selectable:
        label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
    return label


def _set(label: QLabel, text: str, level: str | None = None) -> None:
    if label.text() != text:
        label.setText(text)
    if level is not None and label.property("level") != level:
        label.setProperty("level", level)
        label.setStyleSheet(f"color: {LEVEL_COLOURS.get(level, LEVEL_COLOURS['plain'])};")


def _panel(title: str) -> tuple[QFrame, QVBoxLayout]:
    frame = QFrame()
    frame.setObjectName("panel")
    layout = QVBoxLayout(frame)
    layout.setContentsMargins(12, 10, 12, 12)
    layout.setSpacing(8)
    heading = QLabel(title)
    heading.setObjectName("panelTitle")
    layout.addWidget(heading)
    return frame, layout


def _table(headers: list[str], rows: int = 0) -> QTableWidget:
    table = QTableWidget(rows, len(headers))
    table.setHorizontalHeaderLabels(headers)
    table.verticalHeader().setVisible(False)
    table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
    table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
    table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
    table.setAlternatingRowColors(True)
    table.verticalHeader().setDefaultSectionSize(25)
    table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
    table.horizontalHeader().setStretchLastSection(True)
    return table


def _fill(table: QTableWidget, rows: list[tuple[str, ...]]) -> None:
    table.setRowCount(len(rows))
    for row, values in enumerate(rows):
        for column, value in enumerate(values):
            item = table.item(row, column)
            if item is None:
                item = QTableWidgetItem()
                table.setItem(row, column, item)
            if item.text() != value:
                item.setText(value)


def _scroll(body: QWidget) -> QScrollArea:
    body.setObjectName("scrollBody")
    area = QScrollArea()
    area.setWidgetResizable(True)
    area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    area.setWidget(body)
    return area


def render_calibration_view(frame: np.ndarray, model: CalibrationModel) -> np.ndarray:
    """A COPY of ``frame`` with the fitted arena outline and the placed points drawn on it."""
    out = frame.copy()
    size = (frame.shape[1], frame.shape[0])
    if model.clicks.image_size is not None and tuple(model.clicks.image_size) != size:
        return out                                  # the points belong to another resolution: draw nothing
    if model.mapping.valid:
        draw_arena(out, model.mapping, model.config.arena)
    draw_reference_points(out, model.mapping, model.session, model.config.warn_check_error_mm)
    current = model.current
    if current is not None and current.name in model.clicks.points:
        u, v = model.clicks.points[current.name]
        cv2.circle(out, (int(round(u)), int(round(v))), 11, (0, 255, 255), 1, cv2.LINE_AA)
    return out


class SetupScreen(QWidget):
    """Tabs: Camera, Floor calibration, Markers, Controller & IMU.

    ``live_points_path`` and ``profiles_dir`` exist so tests never touch the real calibration files;
    the application leaves them at their defaults.
    """

    def __init__(self, hub, parent=None, live_points_path: Path | str | None = None,
                 profiles_dir: Path | str | None = None) -> None:
        super().__init__(parent)
        self.hub = hub
        self.config = hub.vision.config
        self.calibration = CalibrationModel(self.config, live_points_path, mock=bool(hub.options.mock),
                                            profiles_dir=profiles_dir)
        start = hub.vision.clicks
        if start is not None and not start.historical and not start.synthetic:
            self.calibration.adopt(start)             # continue from the calibration in use; never a footage profile
            self.calibration.message = ""
        self.calibration.extra_block = self._profile_block
        self.single_tag = SingleTagCheck(self.config.markers, SINGLE_TAG_SAMPLES)
        self.raw_log = RawLineLog(keep=8)
        self._frozen: np.ndarray | None = None
        self._single_tag_version = -1
        self._last_frame_index = -2

        self.setStyleSheet(SETUP_STYLE)
        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        self.tabs = QTabWidget()
        self.camera_tab = self._build_camera_tab()
        self.calibration_tab = self._build_calibration_tab()
        self.markers_tab = self._build_markers_tab()
        self.controller_tab = self._build_controller_tab()
        self.tabs.addTab(self.camera_tab, "Camera")
        self.tabs.addTab(self.calibration_tab, "Floor calibration")
        self.tabs.addTab(self.markers_tab, "Markers")
        self.tabs.addTab(self.controller_tab, "Controller && IMU")
        self.tabs.currentChanged.connect(lambda _index: self.refresh())
        root.addWidget(self.tabs)

        hub.sample_ready.connect(self._on_sample)
        hub.controller_line.connect(self._on_controller_line)
        if hasattr(hub, "profile_changed"):
            hub.profile_changed.connect(lambda _name: self._calibration_changed())
        self._timer = QTimer(self)
        self._timer.setInterval(REFRESH_MS)
        self._timer.timeout.connect(self.refresh)
        self.refresh_profiles()
        self._calibration_changed()
        self.refresh()

    # ================================================================ A. camera

    def _build_camera_tab(self) -> QWidget:
        tab = QWidget()
        layout = QHBoxLayout(tab)
        side = QWidget()
        side.setMaximumWidth(520)
        column = QVBoxLayout(side)
        column.setContentsMargins(0, 0, 0, 0)

        panel, body = _panel("CAMERA SOURCE")
        grid = QGridLayout()
        grid.setHorizontalSpacing(14)
        self.camera_values: dict[str, QLabel] = {}
        for row, name in enumerate(("Source", "Description", "Index", "Resolution", "Measured fps", "State")):
            grid.addWidget(_label(name, "rowName"), row, 0, Qt.AlignmentFlag.AlignTop)
            self.camera_values[name] = _label("", selectable=True)
            grid.addWidget(self.camera_values[name], row, 1)
        grid.setColumnStretch(1, 1)
        body.addLayout(grid)
        column.addWidget(panel)

        panel, body = _panel("CAMERA INDEX")
        row = QHBoxLayout()
        self.camera_index = QSpinBox()
        self.camera_index.setRange(0, 15)
        self.camera_index.setValue(int(self.hub.options.camera_index))
        self.reopen_button = QPushButton("Reopen camera")
        self.reopen_button.clicked.connect(self.reopen_camera)
        row.addWidget(_label("Camera index", "rowName"))
        row.addWidget(self.camera_index)
        row.addWidget(self.reopen_button)
        row.addStretch(1)
        body.addLayout(row)
        self.reopen_message = _label("Not reopened in this session.", "hint")
        body.addWidget(self.reopen_message)
        body.addWidget(_label("The Logitech C525 has been index 1 on this PC. The index can change when cameras are "
                              "plugged in or removed; try another index if the wrong picture appears. A camera cannot "
                              "be changed while recording, or in a mock / replay session.", "hint"))
        column.addWidget(panel)
        column.addStretch(1)
        layout.addWidget(side)

        self.camera_preview = FrameView()
        self.camera_preview.set_placeholder("No camera frame")
        layout.addWidget(self.camera_preview, 1)
        return tab

    def reopen_camera(self) -> None:
        index = self.camera_index.value()
        ok, message = self.hub.reopen_camera(index)
        _set(self.reopen_message, f"Camera index {index}: {message}" if ok else f"Not reopened: {message}",
             "ok" if ok else "warn")

    def _refresh_camera(self) -> None:
        for name, text, level in camera_rows(self.hub.capture, self.hub.options):
            _set(self.camera_values[name], text, level)

    # ================================================================ B. floor calibration

    def _build_calibration_tab(self) -> QWidget:
        tab = QWidget()
        outer = QHBoxLayout(tab)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        outer.addWidget(splitter)

        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 6, 0)
        self.calibration_view = FrameView()
        self.calibration_view.set_placeholder("No camera frame")
        self.calibration_view.clicked.connect(self._on_frame_clicked)
        left_layout.addWidget(self.calibration_view, 1)
        self.calibration_prompt = _label("", selectable=True)
        left_layout.addWidget(self.calibration_prompt)
        row = QHBoxLayout()
        self.freeze_button = QPushButton("Freeze frame")
        self.freeze_button.clicked.connect(lambda: self.freeze())
        self.unfreeze_button = QPushButton("Unfreeze")
        self.unfreeze_button.clicked.connect(self.unfreeze)
        row.addWidget(self.freeze_button)
        row.addWidget(self.unfreeze_button)
        row.addStretch(1)
        left_layout.addLayout(row)
        self.calibration_message = _label("", "hint")
        left_layout.addWidget(self.calibration_message)
        splitter.addWidget(left)

        body = QWidget()
        column = QVBoxLayout(body)
        column.setContentsMargins(6, 0, 6, 0)

        panel, layout = _panel("PROFILE BEING EDITED")
        self.profile_label = _label(LIVE_PROFILE_LABEL, "profileLabel")
        layout.addWidget(self.profile_label)
        self.profile_status = _label("", "hint", selectable=True)
        layout.addWidget(self.profile_status)
        self.recalibrate_label = _label(WHEN_TO_RECALIBRATE, "hint")
        layout.addWidget(self.recalibrate_label)
        column.addWidget(panel)

        panel, layout = _panel("LANDMARKS")
        layout.addWidget(_label("Freeze a frame, select a landmark, then click it in the picture. " + FLOOR_LEVEL_NOTE, "hint"))
        self.landmark_table = _table(["Landmark", "Arena x, y (m)", "Role", "Placed", "Fit residual mm", "CHECK error mm"],
                                     len(self.config.references))
        self.landmark_table.setMinimumHeight(min(440, 50 + 25 * len(self.config.references)))
        self.landmark_table.itemSelectionChanged.connect(self._on_landmark_selected)
        self.landmark_table.cellDoubleClicked.connect(self._on_landmark_double_clicked)
        layout.addWidget(self.landmark_table)
        row = QHBoxLayout()
        self.role_button = QPushButton("Swap fit / check")
        self.role_button.setToolTip("Swap the selected landmark between FIT (used in the fit) and CHECK (held out)")
        self.role_button.clicked.connect(self.toggle_role)
        self.skip_button = QPushButton("Skip")
        self.skip_button.clicked.connect(self.skip_landmark)
        self.undo_button = QPushButton("Undo")
        self.undo_button.clicked.connect(self.undo_click)
        self.clear_button = QPushButton("Clear")
        self.clear_button.clicked.connect(self.clear_clicks)
        for button in (self.role_button, self.skip_button, self.undo_button, self.clear_button):
            row.addWidget(button)
        row.addStretch(1)
        layout.addLayout(row)
        column.addWidget(panel)

        panel, layout = _panel("FIT")
        self.fit_label = _label("", "fitLine", selectable=True)
        layout.addWidget(self.fit_label)
        self.fit_errors = _label("", selectable=True)
        layout.addWidget(self.fit_errors)
        row = QHBoxLayout()
        self.save_button = QPushButton("Save live calibration")
        self.save_button.clicked.connect(self.save_calibration)
        self.reload_button = QPushButton("Reload saved")
        self.reload_button.setToolTip("Discard unsaved clicks and reload the saved LIVE calibration")
        self.reload_button.clicked.connect(self.reload_saved)
        row.addWidget(self.save_button)
        row.addWidget(self.reload_button)
        row.addStretch(1)
        layout.addLayout(row)
        self.save_message = _label("", selectable=True)
        layout.addWidget(self.save_message)
        column.addWidget(panel)

        panel, layout = _panel("OTHER PROFILES (READ-ONLY, NEVER LOADED HERE)")
        self.profiles_label = _label("", selectable=True)
        layout.addWidget(self.profiles_label)
        column.addWidget(panel)
        column.addStretch(1)

        area = _scroll(body)
        area.setMinimumWidth(600)
        splitter.addWidget(area)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 0)
        splitter.setSizes([800, 680])
        return tab

    def _profile_block(self) -> str | None:
        """This tab edits the LIVE landmark file only, and only when no arena set-up owns the corner clicks."""
        name = getattr(self.hub, "profile_name", LIVE_PROFILE)
        if name != LIVE_PROFILE:
            return (f"NOT saved: the active profile is '{name}'. Floor-landmark calibration here edits only the LIVE profile. "
                    f"Use the ARENA tab for '{name}', or switch the profile at the top of the window.")
        if getattr(self.hub, "overlay_config", None) is not None and self.hub.overlay_config.arena.get("source") == "arena setup file":
            return ("NOT saved: an arena set-up is saved for LIVE, and its camera corners live in the ARENA tab. "
                    "Use ARENA > Recalibrate camera; saving landmarks here would replace those corner clicks.")
        return None

    @property
    def frozen(self) -> bool:
        return self._frozen is not None

    def freeze(self, frame: np.ndarray | None = None) -> bool:
        """Calibrate on one still frame: the newest camera frame, or ``frame`` when one is handed in."""
        if frame is None:
            _, _, frame = self.hub.capture.latest()
        if frame is None:
            _set(self.calibration_message, "No camera frame to freeze: the camera is not delivering frames.", "warn")
            return False
        self._frozen = frame.copy()
        self.calibration.set_frame_size((frame.shape[1], frame.shape[0]))
        _set(self.calibration_message, "Frame frozen. Select a landmark and click it at floor level.", "off")
        self._calibration_changed()
        return True

    def unfreeze(self) -> None:
        self._frozen = None
        self._last_frame_index = -2
        _set(self.calibration_message, "Live view. Freeze a frame to click landmarks.", "off")
        self._calibration_changed()

    def _on_frame_clicked(self, x: float, y: float) -> None:
        if not self.frozen:
            _set(self.calibration_message, "Freeze a frame before clicking landmarks (clicks on the live picture are ignored).", "warn")
            return
        self.place_click(x, y)

    def place_click(self, x: float, y: float) -> str | None:
        name = self.calibration.place(x, y)
        _set(self.calibration_message, self.calibration.message, "off" if name else "warn")
        self._calibration_changed()
        return name

    def _on_landmark_selected(self) -> None:
        row = self.landmark_table.currentRow()
        if row >= 0 and self.landmark_table.selectedItems():
            self.calibration.select(row)
            self._calibration_changed(sync_selection=False)

    def _on_landmark_double_clicked(self, row: int, column: int) -> None:
        if column == 2:
            self.calibration.toggle_role(row)
            _set(self.calibration_message, self.calibration.message, "off")
            self._calibration_changed()

    def toggle_role(self) -> None:
        role = self.calibration.toggle_role()
        _set(self.calibration_message, self.calibration.message if role else "Select a landmark first.", "off" if role else "warn")
        self._calibration_changed()

    def skip_landmark(self) -> None:
        self.calibration.skip()
        _set(self.calibration_message, self.calibration.message, "off")
        self._calibration_changed()

    def undo_click(self) -> None:
        name = self.calibration.undo()
        _set(self.calibration_message, self.calibration.message, "off" if name else "warn")
        self._calibration_changed()

    def clear_clicks(self) -> None:
        self.calibration.clear()
        _set(self.calibration_message, self.calibration.message, "off")
        self._calibration_changed()

    def reload_saved(self) -> None:
        ok, message = self.calibration.reload_saved()
        _set(self.save_message, message, "ok" if ok else "warn")
        self._calibration_changed()

    def save_calibration(self) -> bool:
        """Write the live profile, then hand a copy of the points to the hub."""
        info = self.hub.capture.source_info or {}
        result = self.calibration.save(source=f"Command Centre Setup: {info.get('description') or 'camera frame'}")
        if result.ok:
            self.hub.apply_clicks(self.calibration.clicks_copy())
        _set(self.save_message, result.message, ("ok" if self.calibration.quality.level == "ok" else "warn") if result.ok else "bad")
        self._calibration_changed(keep_save_message=True)
        return result.ok

    @staticmethod
    def _short_path(path: Path) -> str:
        """Repo-relative when possible: a long absolute path has no spaces and cannot wrap."""
        try:
            return path.resolve().relative_to(REPO_ROOT).as_posix()
        except ValueError:
            return f"{path.name} (in folder {path.parent.name}; hover for the full path)"

    def refresh_profiles(self) -> None:
        profiles = list_profiles(self.calibration.profiles_dir)
        text = "\n".join(profile.label() for profile in profiles) if profiles else "No other profiles found."
        _set(self.profiles_label, text, "off")

    def _calibration_changed(self, sync_selection: bool = True, keep_save_message: bool = False) -> None:
        model = self.calibration
        self.freeze_button.setEnabled(not self.frozen)
        self.unfreeze_button.setEnabled(self.frozen)
        self._show_calibration_frame()               # first: a live frame may change the frame size, and so the fit
        rows = []
        for row in model.rows():
            rows.append((row.name, f"{row.arena_xy[0]:.3f}, {row.arena_xy[1]:.3f}",
                         row.role.upper() + (" (changed)" if row.role_overridden else ""),
                         "placed" if row.placed else "not placed",
                         "-" if row.fit_residual_mm is None else f"{row.fit_residual_mm:.1f}",
                         "-" if row.check_error_mm is None else f"{row.check_error_mm:.1f}"))
        _fill(self.landmark_table, rows)
        for row in model.rows():
            self.landmark_table.item(row.index, 0).setToolTip(row.note)
        if sync_selection:
            self.landmark_table.blockSignals(True)
            if model.current is not None:
                self.landmark_table.selectRow(model.session.index)
            else:
                self.landmark_table.clearSelection()
            self.landmark_table.blockSignals(False)
        _set(self.fit_label, model.fit_line(), FIT_LEVELS[model.quality.level])
        _set(self.fit_errors, model.error_summary())
        _set(self.calibration_prompt, model.prompt(), "plain")
        _set(self.profile_status, f"{model.status_line()}\nFile: {self._short_path(model.target_path)}")
        self.profile_status.setToolTip(str(model.target_path))
        permanent = model.permanent_save_block()
        blocked = model.save_block_reason()
        self.save_button.setEnabled(blocked is None)
        self.save_button.setToolTip(blocked or "Write the LIVE profile (the previous file is kept as a .bak copy) and use it now")
        if permanent:
            _set(self.save_message, permanent, "warn")
        elif not keep_save_message:
            if blocked:
                _set(self.save_message, blocked, "bad")
            elif model.quality.level == "weak":
                _set(self.save_message, "Fit is WEAK: it can be saved, and it will be shown as weak everywhere.", "warn")
            else:
                _set(self.save_message, "Ready to save." if model.unsaved else "No unsaved changes.", "off")

    def _show_calibration_frame(self) -> None:
        mock = "MOCK · " if self.hub.options.mock else ""
        if self._frozen is not None:
            self.calibration_view.badge = f"{mock}FROZEN - click landmarks at floor level"
            self.calibration_view.set_frame(render_calibration_view(self._frozen, self.calibration))
            return
        index, _, frame = self.hub.capture.latest()
        self.calibration_view.badge = f"{mock}LIVE - freeze to click landmarks" if frame is not None else ""
        if frame is None:
            self.calibration_view.set_placeholder(f"No camera frame ({self.hub.capture.state})")
            self.calibration_view.set_frame(None)
            return
        self._last_frame_index = index
        self.calibration.set_frame_size((frame.shape[1], frame.shape[0]))
        self.calibration_view.set_frame(render_calibration_view(frame, self.calibration))

    def _refresh_calibration(self) -> None:
        if self._frozen is not None:
            return                                   # a frozen picture only changes when the clicks do
        index = self.hub.capture.latest()[0]
        if index == self._last_frame_index:
            return
        before = (self.calibration.frame_size, self.calibration.quality.level)
        self._show_calibration_frame()
        if before != (self.calibration.frame_size, self.calibration.quality.level):
            self._calibration_changed()              # the camera resolution changed under the saved points

    # ================================================================ C. markers

    def _build_markers_tab(self) -> QWidget:
        tab = QWidget()
        outer = QVBoxLayout(tab)
        outer.setContentsMargins(0, 0, 0, 0)
        body = QWidget()
        layout = QHBoxLayout(body)

        left = QVBoxLayout()
        panel, box = _panel("CONFIGURED MARKER LAYOUT")
        box.addWidget(_label(f"Dictionary {self.config.dictionary}. Offsets are measured from the bot centre: forward "
                             "toward the nose, left to the bot's left; yaw is the marker's top edge relative to the nose.", "hint"))
        layout_rows = marker_layout_rows(self.config)
        self.layout_table = _table(["id", "face", "forward_m", "left_m", "yaw_deg", "size_m", "height_m"], len(layout_rows))
        self.layout_table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        _fill(self.layout_table, layout_rows)
        self.layout_table.setMinimumHeight(min(330, 34 + 30 * max(1, len(layout_rows))))
        box.addWidget(self.layout_table)
        note, level = marker_placeholder_note(self.config)
        self.placeholder_label = _label()
        _set(self.placeholder_label, note, level)
        box.addWidget(self.placeholder_label)
        left.addWidget(panel)

        panel, box = _panel("LIVE DETECTION")
        grid = QGridLayout()
        grid.setHorizontalSpacing(14)
        self.marker_values: dict[str, QLabel] = {}
        for row, name in enumerate(MARKER_ROW_LABELS):
            grid.addWidget(_label(name, "rowName"), row, 0, Qt.AlignmentFlag.AlignTop)
            self.marker_values[name] = _label("", selectable=True)
            grid.addWidget(self.marker_values[name], row, 1)
        grid.setColumnStretch(1, 1)
        box.addLayout(grid)
        left.addWidget(panel)
        left.addStretch(1)
        layout.addLayout(left, 1)

        right = QVBoxLayout()
        panel, box = _panel("SINGLE-TAG CHECK (STATIONARY BOT)")
        box.addWidget(_label(SINGLE_TAG_HOWTO, "hint"))
        row = QHBoxLayout()
        self.capture_button = QPushButton(f"Capture ({SINGLE_TAG_SAMPLES} samples)")
        self.capture_button.clicked.connect(self.capture_single_tag)
        self.single_tag_reset = QPushButton("Reset")
        self.single_tag_reset.clicked.connect(self.reset_single_tag)
        row.addWidget(self.capture_button)
        row.addWidget(self.single_tag_reset)
        row.addStretch(1)
        box.addLayout(row)
        self.single_tag_status = _label("", selectable=True)
        box.addWidget(self.single_tag_status)
        box.addWidget(_label("Bot centre (x, y) and nose heading estimated from each tag alone. Scatter is the "
                             "frame-to-frame noise inside one capture.", "hint"))
        self.single_tag_table = _table(["tag", "x (m)", "y (m)", "heading (deg)", "samples", "scatter (mm)", "scatter (deg)"])
        self.single_tag_table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.single_tag_table.setMinimumHeight(190)
        box.addWidget(self.single_tag_table)
        self.single_tag_spread = _label("", "fitLine", selectable=True)
        box.addWidget(self.single_tag_spread)
        self.single_tag_statement = _label(SPREAD_STATEMENT)
        _set(self.single_tag_statement, SPREAD_STATEMENT, "warn")
        box.addWidget(self.single_tag_statement)
        right.addWidget(panel)
        right.addStretch(1)
        layout.addLayout(right, 1)

        outer.addWidget(_scroll(body))
        return tab

    def capture_single_tag(self) -> None:
        if self.single_tag.capturing:
            self.single_tag.cancel()
        else:
            self.single_tag.start()
        self._refresh_markers()

    def reset_single_tag(self) -> None:
        self.single_tag.reset()
        self._refresh_markers()

    def _on_sample(self, sample) -> None:
        self.single_tag.add(sample)

    def markers_text(self) -> str:
        return "\n".join(f"{name}: {label.text()}" for name, label in self.marker_values.items())

    def _refresh_markers(self, now: float | None = None) -> None:
        now = time.monotonic() if now is None else now
        report = marker_report(self.config, self.hub.latest_sample, now)
        for name, text, level in report.rows:
            _set(self.marker_values[name], text, level)
        check = self.single_tag
        self.capture_button.setText("Cancel capture" if check.capturing else f"Capture ({SINGLE_TAG_SAMPLES} samples)")
        _set(self.single_tag_status, check.status, "warn" if check.capturing else "off")
        if check.version != self._single_tag_version:
            self._single_tag_version = check.version
            _fill(self.single_tag_table, [
                (str(e.tag_id), f"{e.x_m:.4f}", f"{e.y_m:.4f}", f"{e.heading_deg:+.2f}", str(e.samples),
                 f"{e.position_scatter_mm:.2f}", f"{e.heading_scatter_deg:.2f}")
                for e in (check.estimates[tag] for tag in sorted(check.estimates))])
            _set(self.single_tag_spread, check.spread_text(), "warn" if check.spread() is not None else "off")

    # ================================================================ D. controller & IMU (read-only)

    def _build_controller_tab(self) -> QWidget:
        tab = QWidget()
        outer = QVBoxLayout(tab)
        outer.setContentsMargins(0, 0, 0, 0)
        body = QWidget()
        layout = QHBoxLayout(body)

        left = QVBoxLayout()
        panel, box = _panel("TELEMETRY SOURCE (LISTEN-ONLY)")
        grid = QGridLayout()
        grid.setHorizontalSpacing(14)
        self.link_source = _label("", selectable=True)
        self.link_state = _label("", selectable=True)
        self.firmware_label = _label("", selectable=True)
        for row, (name, widget) in enumerate((("Source", self.link_source), ("State", self.link_state),
                                              ("Firmware", self.firmware_label))):
            grid.addWidget(_label(name, "rowName"), row, 0, Qt.AlignmentFlag.AlignTop)
            grid.addWidget(widget, row, 1)
        grid.setColumnStretch(1, 1)
        box.addLayout(grid)
        row = QHBoxLayout()
        self.port_edit = QLineEdit()
        self.port_edit.setPlaceholderText("e.g. COM5")
        self.port_edit.setMaximumWidth(110)
        row.addWidget(_label("COM port", "rowName"))
        if self.hub.options.controller_port:
            self.port_edit.setText(str(self.hub.options.controller_port))
        self.listen_button = QPushButton("Listen (read-only)")
        self.listen_button.clicked.connect(self.listen_on_port)
        self.follow_button = QPushButton("Follow client log…")
        self.follow_button.clicked.connect(self.pick_client_log)
        self.disconnect_button = QPushButton("Disconnect")
        self.disconnect_button.clicked.connect(self.disconnect_controller)
        for widget in (self.port_edit, self.listen_button, self.follow_button, self.disconnect_button):
            row.addWidget(widget)
        row.addStretch(1)
        box.addLayout(row)
        self.link_message = _label("", "hint", selectable=True)
        box.addWidget(self.link_message)
        box.addWidget(_label(ONE_PORT_NOTE, "hint"))
        box.addWidget(_label(READ_ONLY_NOTE, "hint"))
        left.addWidget(panel)

        panel, box = _panel("IMU")
        self.imu_label = _label("", "mono", selectable=True)
        box.addWidget(self.imu_label)
        left.addWidget(panel)
        left.addStretch(1)
        layout.addLayout(left, 1)

        right = QVBoxLayout()
        panel, box = _panel("LATEST RAW LINES")
        grid = QGridLayout()
        grid.setHorizontalSpacing(14)
        self.raw_values: dict[str, QLabel] = {}
        for row, kind in enumerate(RawLineLog.KINDS):
            grid.addWidget(_label(kind, "rowName"), row, 0, Qt.AlignmentFlag.AlignTop)
            self.raw_values[kind] = _label("", "mono", selectable=True)
            grid.addWidget(self.raw_values[kind], row, 1)
        grid.setColumnStretch(1, 1)
        box.addLayout(grid)
        box.addWidget(_label("Last lines, newest at the bottom (IMU lines are left out; the newest one is above). "
                             "RX = from the controller. TX = a line the PC drive client's log says that client sent; "
                             "this dashboard sends nothing.", "hint"))
        self.raw_tail = QPlainTextEdit()
        self.raw_tail.setReadOnly(True)
        self.raw_tail.setMinimumHeight(330)             # lines wrap, so the end of a long ST line is never hidden
        box.addWidget(self.raw_tail)
        right.addWidget(panel)
        right.addStretch(1)
        layout.addLayout(right, 1)

        outer.addWidget(_scroll(body))
        return tab

    def listen_on_port(self) -> None:
        port = self.port_edit.text().strip()
        if not port:
            _set(self.link_message, "Type a COM port first, e.g. COM5.", "warn")
            return
        self.raw_log.clear()
        ok, message = self.hub.connect_controller_port(port)
        _set(self.link_message, message if ok else f"Could not listen on {port}: {message}", "ok" if ok else "bad")
        self._refresh_controller()

    def pick_client_log(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Follow the PC drive client's --log file", "", "Log files (*.log *.txt);;All files (*)")
        if path:
            self.follow_client_log(path)

    def follow_client_log(self, path: Path | str) -> bool:
        self.raw_log.clear()
        ok, message = self.hub.connect_controller_log(path)
        _set(self.link_message, message if ok else f"Could not follow {path}: {message}", "ok" if ok else "bad")
        self._refresh_controller()
        return ok

    def disconnect_controller(self) -> None:
        self.hub.disconnect_controller()
        _set(self.link_message, "Telemetry source closed.", "off")
        self._refresh_controller()

    def _on_controller_line(self, direction: str, text: str) -> None:
        self.raw_log.add(direction, text)

    def controller_text(self) -> str:
        return "\n".join([f"Source: {self.link_source.text()}", f"State: {self.link_state.text()}",
                          f"Firmware: {self.firmware_label.text()}", self.imu_label.text()]
                         + [f"{kind}: {label.text()}" for kind, label in self.raw_values.items()])

    def _refresh_controller(self, now: float | None = None) -> None:
        now = time.monotonic() if now is None else now
        monitor = self.hub.monitor                    # the hub makes a new monitor for every source
        source, state, level = controller_summary(monitor, self.hub.link_label, self.hub.link_error, now)
        _set(self.link_source, source, "plain" if self.hub.link_label else "off")
        _set(self.link_state, state, level)
        _set(self.firmware_label, firmware_text(monitor), "plain" if monitor.firmware() else "off")
        text, level = imu_report(monitor, now)
        _set(self.imu_label, text, level)
        for kind, label in self.raw_values.items():
            _set(label, self.raw_log.latest_text(kind), "plain" if kind in self.raw_log.latest else "off")
        tail = self.raw_log.tail_text()
        if self.raw_tail.toPlainText() != tail:
            self.raw_tail.setPlainText(tail)
            bar = self.raw_tail.verticalScrollBar()
            bar.setValue(bar.maximum())               # keep the newest line in view
        self.disconnect_button.setEnabled(self.hub.link_label is not None)

    # ================================================================ refresh

    def refresh(self, now: float | None = None) -> None:
        """Update every panel from the hub. Safe without a camera, a controller or a visible window."""
        self._refresh_camera()
        self._refresh_markers(now)
        self._refresh_controller(now)
        current = self.tabs.currentWidget()
        if current is self.calibration_tab:
            self._refresh_calibration()
        elif current is self.camera_tab:
            self.camera_preview.badge = "MOCK" if self.hub.options.mock else ""
            self.camera_preview.set_frame(self.hub.capture.latest()[2])

    def showEvent(self, event) -> None:  # noqa: N802 - Qt API
        super().showEvent(event)
        self.refresh_profiles()
        self.refresh()
        self._timer.start()

    def hideEvent(self, event) -> None:  # noqa: N802 - Qt API
        self._timer.stop()
        super().hideEvent(event)
