"""ROBOT TRACKING & LIVE MONITORING: what the camera makes of the robots, and whether the policy WOULD act. Read-only.

This screen shows the camera with the calibrated arena drawn on it, the own robot's tracker state and the markers behind it,
the click-selected opponent, warnings, timing, and an optional display-only SHADOW MODE.

  * It never opens a camera: frames come from ``hub.capture`` (the one owner of the camera).
  * It has no connection to the robot and no control that sends anything. It does not import a serial port or a socket and
    writes nothing to the controller link. A policy proposal is text on this screen; there is no path from it to a motor.
  * All vision work (marker detection, trackers, shadow inference) runs on ONE worker thread that takes the newest frame.
    The GUI thread only paints the newest finished result, about 16 times a second, and only while the tab is visible.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (QAbstractItemView, QCheckBox, QFrame, QGridLayout, QHBoxLayout, QHeaderView, QLabel, QPushButton,
                               QScrollArea, QSplitter, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget)

from control_center.backend import tracking_model as tm
from control_center.backend.readiness import (CHIP_COLOURS, FAILED, NOT_CONFIGURED, READY, WARNING, Readiness)
from control_center.backend.tracking_worker import (CheckpointLoader, HubTelemetry, ShadowRunner, TrackingEngine, TrackingResult,
                                                    TrackingWorker)
from control_center.backend.marker_settings import dictionary_status
from control_center.tools.chassis_tracker import TrackerSettings
from control_center.tools.live_telemetry import DEFAULT_MOUNTING, ImuMounting, load_mounting
from control_center.tools.observation_contract import RECOMMENDED_CHECKPOINT, load_contract
from control_center.tools.overlay_core import ClickSet
from control_center.widgets.arena_map import ArenaMapWidget, Marker, Trail
from control_center.widgets.frame_view import FrameView

REFRESH_MS = 60                  # about 16 Hz
SHADOW_EVERY_N_REFRESHES = 4     # the prerequisite table is rebuilt about 4 times a second
RESULT_STALE_S = 2.0
DRAG_HINT = "DRAG a rectangle around the opponent in the picture to select it (any object, any size)."
SHADOW_GRACE_S = 2.0             # how long a cheap prerequisite may be missing before a running shadow is switched off
LEVEL_COLOURS = {"ok": "#55c889", "warn": "#e8b85b", "bad": "#ef6b73", "off": "#9aa5b1", "plain": "#d9dee5"}
TITLE = "ROBOT TRACKING & LIVE MONITORING"
SAFE_ARENA = {"width_m": 1.0, "height_m": 1.0, "pit_centres_m": [], "pit_radius_m": 0.1, "flipper_vertices_m": [[0, 0], [0, 0], [0, 0]]}

STYLE = """
TrackingScreen { background: #11161d; }
TrackingScreen QLabel, TrackingScreen QCheckBox { color: #d9dee5; }
QLabel { background: transparent; }
QFrame#panel { background: #171e27; border: 1px solid #2b3542; border-radius: 5px; }
QLabel#panelTitle { color: #9eabb8; font-size: 10pt; font-weight: 700; letter-spacing: 1px; }
QLabel#screenTitle { color: #d9dee5; font-size: 15pt; font-weight: 800; letter-spacing: 1px; }
QLabel#rowName { color: #9eabb8; }
QLabel#hint { color: #9eabb8; }
QLabel#banner { border-radius: 5px; padding: 8px 12px; font-size: 11pt; font-weight: 700; }
QLabel#chip { border-radius: 10px; padding: 3px 12px; font-weight: 800; color: #0d1218; }
QLabel#headline { font-size: 13pt; font-weight: 800; }
QTableWidget { background: #0d1218; alternate-background-color: #121922; gridline-color: #2b3542;
               border: 1px solid #2b3542; selection-background-color: #2f4a66; selection-color: #ffffff; }
QHeaderView::section { background: #171e27; color: #9eabb8; border: none; border-bottom: 1px solid #2b3542;
                       padding: 4px 6px; font-weight: 700; }
QScrollArea { border: none; background: transparent; }
QWidget#scrollBody { background: transparent; }
QPushButton { padding: 6px 14px; }
"""


# ---------------------------------------------------------------- small widgets


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
    layout.setSpacing(6)
    heading = QLabel(title)
    heading.setObjectName("panelTitle")
    layout.addWidget(heading)
    return frame, layout


class KeyValueGrid(QWidget):
    """Rows of (name, value) whose labels are reused, so refreshing does not rebuild widgets."""

    def __init__(self) -> None:
        super().__init__()
        self._grid = QGridLayout(self)
        self._grid.setContentsMargins(0, 0, 0, 0)
        self._grid.setHorizontalSpacing(14)
        self._grid.setColumnStretch(1, 1)
        self._names: list[QLabel] = []
        self._values: list[QLabel] = []

    def set_rows(self, rows: list[tuple[str, str, str]]) -> None:
        while len(self._names) < len(rows):
            row = len(self._names)
            name, value = _label("", "rowName"), _label("", selectable=True)
            self._grid.addWidget(name, row, 0, Qt.AlignmentFlag.AlignTop)
            self._grid.addWidget(value, row, 1)
            self._names.append(name)
            self._values.append(value)
        for index, (name, value, level) in enumerate(rows):
            self._names[index].show()
            self._values[index].show()
            _set(self._names[index], name)
            _set(self._values[index], value, level)
        for index in range(len(rows), len(self._names)):
            self._names[index].hide()
            self._values[index].hide()

    def texts(self) -> list[tuple[str, str]]:
        return [(n.text(), v.text()) for n, v in zip(self._names, self._values) if not n.isHidden()]


def _table(headers: list[str]) -> QTableWidget:
    table = QTableWidget(0, len(headers))
    table.setHorizontalHeaderLabels(headers)
    table.verticalHeader().setVisible(False)
    table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
    table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
    table.setAlternatingRowColors(True)
    table.verticalHeader().setDefaultSectionSize(24)
    table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
    table.horizontalHeader().setStretchLastSection(True)
    return table


def _fill(table: QTableWidget, rows: list[tuple[list[str], str]]) -> None:
    """rows: ([cell texts], level). Colours the text by level."""
    from PySide6.QtGui import QColor

    table.setRowCount(len(rows))
    for r, (cells, level) in enumerate(rows):
        for c, text in enumerate(cells):
            item = table.item(r, c)
            if item is None:
                item = QTableWidgetItem()
                table.setItem(r, c, item)
            if item.text() != text:
                item.setText(text)
            item.setForeground(QColor(LEVEL_COLOURS.get(level, LEVEL_COLOURS["plain"])))


class TrackingMap(ArenaMapWidget):
    """The shared top-down arena map, with one radius per pit when the profile has ``pit_radii_m``."""

    def __init__(self, arena: dict[str, Any]) -> None:
        super().__init__(SAFE_ARENA)
        self.set_arena(arena)

    def set_arena(self, arena: dict[str, Any]) -> None:
        from control_center.tools.overlay_core import arena_outline, flipper_boundary, pit_arcs

        usable = arena if not tm.arena_problems(arena) else SAFE_ARENA
        self.arena = usable
        self._outline = arena_outline(usable)
        self._flipper = flipper_boundary(usable)
        self._pits = [arc for centre, radius in tm.pit_list(usable)
                      for arc in pit_arcs({**usable, "pit_centres_m": [list(centre)], "pit_radius_m": radius}, samples=96)]
        self.arena_ok = usable is arena
        self.update()


# ---------------------------------------------------------------- the screen


class TrackingScreen(QWidget):
    """Tab: ROBOT TRACKING & LIVE MONITORING.

    ``policy_factory`` / ``contract_loader`` / ``mounting_path`` exist so tests never load the real 30-60 s checkpoint; the
    application leaves them at their defaults.
    """

    def __init__(self, hub, parent=None, *, policy_factory: Callable[[Any], Any] | None = None,
                 contract_loader: Callable[[], Any] | None = None, mounting_path: Path | str | None = None) -> None:
        super().__init__(parent)
        self.hub = hub
        self.setWindowTitle(TITLE)
        self.setStyleSheet(STYLE)
        self._policy_factory = policy_factory or self._default_policy_factory
        self._contract_loader = contract_loader or (lambda: load_contract(RECOMMENDED_CHECKPOINT))
        self._contract: Any = None
        self._contract_error = ""
        self._policy: Any = None
        self._loader: CheckpointLoader | None = None
        self._load_state, self._load_detail = "not_loaded", "not loaded (shadow is off)"
        self._shadow_on = False
        self._shadow_message = ""
        self._blocked_since: float | None = None
        self._closed = False

        self.config = hub.vision.config
        self._signature = self._calibration_signature()
        self._epoch = 0
        self.telemetry = HubTelemetry(self._load_mounting(mounting_path))
        self.engine = TrackingEngine(self.config, hub.vision.clicks, self.profile_name())
        self.worker = TrackingWorker(hub.capture, self.engine, self)
        self.warnings = tm.WarningMonitor()
        self.own_trail, self.opponent_trail = tm.TrailBuffer(), tm.TrailBuffer()
        self._result: TrackingResult | None = None
        self._painted_seq = -1
        self._click_message = ""
        self._refreshes = 0
        self._diagnostics: dict[str, Any] = {}
        self._own_aged_view: tuple | None = None          # (state, age to 0.1 s) of the own-robot panel as last painted from the clock
        self._own_display: dict[str, Any] | None = None   # the own-robot report AS SHOWN (aged to the refresh clock); warnings and readiness read this

        self._build()
        hub.controller_line.connect(self._on_controller_line)
        if hasattr(hub, "profile_changed"):
            hub.profile_changed.connect(self._on_profile_changed)
        self._timer = QTimer(self)
        self._timer.setInterval(REFRESH_MS)
        self._timer.timeout.connect(self.refresh)
        self.refresh()

    # ================================================================ construction

    def _build(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        header = QHBoxLayout()
        title = _label(TITLE, "screenTitle")
        title.setWordWrap(False)
        header.addWidget(title)
        header.addStretch(1)
        self.profile_label = _label("", "hint")
        header.addWidget(self.profile_label)
        self.readiness_chip = QLabel(NOT_CONFIGURED)
        self.readiness_chip.setObjectName("chip")
        header.addWidget(self.readiness_chip)
        root.addLayout(header)
        self.camera_line = _label("", "hint")
        root.addWidget(self.camera_line)
        self.banner = _label("", "banner")
        self.banner.hide()
        root.addWidget(self.banner)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        root.addWidget(splitter, 1)

        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 6, 0)
        self.frame_view = FrameView()
        self.frame_view.set_placeholder("No camera frame yet")
        self.frame_view.drag_select = True                         # drag a box around any object; a bare click only explains
        self.frame_view.clicked.connect(self._on_frame_clicked)
        self.frame_view.region_selected.connect(self._on_region_selected)
        left_layout.addWidget(self.frame_view, 1)
        self.click_line = _label(DRAG_HINT, "hint")
        left_layout.addWidget(self.click_line)
        splitter.addWidget(left)

        body = QWidget()
        column = QVBoxLayout(body)
        column.setContentsMargins(0, 0, 6, 0)
        column.setSpacing(8)

        panel, box = _panel("AUTOBOT (OWN ROBOT)")
        self.own_grid = KeyValueGrid()
        box.addWidget(self.own_grid)
        self.confidence_label = _label("", "headline")
        box.addWidget(self.confidence_label)
        self.confidence_inputs = _label("", selectable=True)
        box.addWidget(self.confidence_inputs)
        box.addWidget(_label(tm.CONFIDENCE_NOTE, "hint"))
        column.addWidget(panel)

        panel, box = _panel("MARKERS")
        self.dictionary_label = _label("")
        box.addWidget(self.dictionary_label)
        self.marker_table = _table(["ID", "Face", "Seen now", "Rate (last 5 s)", "Status"])
        self.marker_table.setMinimumHeight(190)
        box.addWidget(self.marker_table)
        self.marker_summary = _label("")
        box.addWidget(self.marker_summary)
        column.addWidget(panel)

        panel, box = _panel("OPPONENT")
        self.status_label = _label("TRACKING STATUS: no opponent selected", "headline")
        box.addWidget(self.status_label)
        self.opponent_grid = KeyValueGrid()
        box.addWidget(self.opponent_grid)
        row = QHBoxLayout()
        self.select_button = QPushButton("Select Opponent / Draw Box")
        self.select_button.setToolTip("Then drag a rectangle around any object in the camera picture (dragging works at any time)")
        self.select_button.clicked.connect(self.arm_selection)
        self.reselect_button = QPushButton("Reselect Opponent")
        self.reselect_button.setToolTip("Drag a new box. The current track stays until the new box is accepted.")
        self.reselect_button.clicked.connect(self.arm_reselection)
        self.clear_button = QPushButton("Clear Opponent")
        self.clear_button.clicked.connect(self.clear_opponent)
        for button in (self.select_button, self.reselect_button, self.clear_button):
            row.addWidget(button)
        row.addStretch(1)
        box.addLayout(row)
        box.addWidget(_label(tm.OPPONENT_HINT, "hint"))
        column.addWidget(panel)

        panel, box = _panel("ARENA MAP (floor plane, metres)")
        self.map = TrackingMap(self.config.arena)
        self.map.setMinimumHeight(260)
        box.addWidget(self.map)
        column.addWidget(panel)

        panel, box = _panel("DIAGNOSTICS")
        self.diagnostics_grid = KeyValueGrid()
        box.addWidget(self.diagnostics_grid)
        column.addWidget(panel)

        panel, box = _panel("SHADOW MODE (display only: nothing is sent)")
        box.addWidget(_label(tm.SHADOW_ONLY_STATEMENT, "hint"))
        box.addWidget(_label(tm.NO_IMU_STATEMENT, "hint"))
        self.shadow_checkbox = QCheckBox("Run policy in shadow mode (display only, nothing is sent)")
        self.shadow_checkbox.setChecked(False)
        self.shadow_checkbox.setEnabled(False)                     # enabled only once the prerequisites that can be checked hold
        self.shadow_checkbox.toggled.connect(self._on_shadow_toggled)
        box.addWidget(self.shadow_checkbox)
        self.shadow_why = _label("", "hint")
        box.addWidget(self.shadow_why)
        self.prereq_table = _table(["Prerequisite", "Status", "Detail"])
        self.prereq_table.setMinimumHeight(260)
        box.addWidget(self.prereq_table)
        self.shadow_hub = _label("", "hint")
        box.addWidget(self.shadow_hub)
        self.load_label = _label("", "hint")
        box.addWidget(self.load_label)
        self.shadow_headline = _label("SHADOW OFF (default)", "headline")
        box.addWidget(self.shadow_headline)
        self.shadow_detail = _label("", selectable=True)
        box.addWidget(self.shadow_detail)
        column.addWidget(panel)
        column.addStretch(1)

        body.setObjectName("scrollBody")
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setWidget(body)
        scroll.setMinimumWidth(430)
        splitter.addWidget(scroll)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)

    # ================================================================ hub access (every attribute guarded)

    def profile_name(self) -> str:
        name = getattr(self.hub, "profile_name", None)
        return str(name) if name else str(getattr(self.hub.vision, "profile", "live"))

    def _calibration_signature(self) -> tuple:
        vision = self.hub.vision
        clicks = vision.clicks
        points = () if clicks is None else (tuple(sorted((k, tuple(v)) for k, v in clicks.points.items())), clicks.image_size,
                                            tuple(sorted(clicks.roles.items())))
        arena = repr(sorted(((k, repr(v)) for k, v in vision.config.arena.items())))
        return (self.profile_name(), id(vision.config), id(clicks), points, arena)

    def _add_event(self, kind: str, text: str) -> None:
        add = getattr(self.hub, "add_event", None)
        if add is not None:
            add(kind, text)

    def _camera(self) -> tuple[str, str, float, float | None]:
        capture = self.hub.capture
        return (capture.state, capture.detail, float(getattr(capture, "measured_fps", 0.0)), capture.last_frame_age())

    @staticmethod
    def _load_mounting(path: Path | str | None) -> ImuMounting:
        try:
            return load_mounting(path or DEFAULT_MOUNTING)
        except Exception as error:                               # an unreadable mounting file can never read as "verified"
            return ImuMounting(verified=False, note=f"mounting file unusable: {error}")

    @staticmethod
    def _default_policy_factory(contract):
        from control_center.tools.stream_adapter import CheckpointPolicy

        return CheckpointPolicy(contract)

    # ================================================================ lifecycle

    def showEvent(self, event) -> None:  # noqa: N802 - Qt API
        super().showEvent(event)
        if self._closed:
            return
        self.worker.resume()
        if not self.worker.isRunning():
            self.worker.start()
        self._timer.start()
        self.refresh()

    def hideEvent(self, event) -> None:  # noqa: N802 - Qt API
        self.worker.pause()                                       # the hub's own vision keeps running; this one does not compete
        self._timer.stop()
        super().hideEvent(event)

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt API
        self.shutdown()
        super().closeEvent(event)

    def shutdown(self) -> None:
        """Stop and join the vision thread (and the checkpoint loader). Safe to call more than once."""
        if self._closed:
            return
        self._closed = True
        self._timer.stop()
        for signal, slot in ((self.hub.controller_line, self._on_controller_line),):
            try:
                signal.disconnect(slot)
            except (TypeError, RuntimeError):
                pass
        if hasattr(self.hub, "profile_changed"):
            try:
                self.hub.profile_changed.disconnect(self._on_profile_changed)
            except (TypeError, RuntimeError):
                pass
        self.worker.stop()
        if self.worker.isRunning() and not self.worker.wait(3000):
            self._add_event("tracking", "tracking vision thread did not stop within 3 s")
        if self._loader is not None:
            self._loader.abandon()            # a checkpoint load cannot be interrupted: it is a daemon thread and ends with the process

    # ================================================================ profile / calibration changes

    def _on_profile_changed(self, _name: str = "") -> None:
        self._check_profile()

    def _check_profile(self) -> None:
        signature = self._calibration_signature()
        if signature == self._signature:
            return
        self._signature = signature
        self._epoch += 1
        self.config = self.hub.vision.config
        clicks = self.hub.vision.clicks
        profile = self.profile_name()
        self.worker.post("rebuild", self.config, clicks, profile,
                         f"profile or calibration changed (now '{profile}'): every track and the opponent selection were cleared",
                         self._epoch)
        self.own_trail.clear(), self.opponent_trail.clear()
        self.warnings.reset()
        self._result, self._painted_seq = None, -1
        self._own_aged_view, self._own_display = None, None
        self._click_message = "profile or calibration changed: select the opponent again"
        self.click_line.setText(self._click_message)
        self.map.set_arena(self.config.arena)
        self.map.set_content(trails=[], markers=[])
        self.frame_view.set_frame(None)
        self.frame_view.set_placeholder("Waiting for a camera frame with the new calibration")
        if self._shadow_on:                                       # the runner holds the old config: restart it on the new one
            self._post_shadow(True)
        self._add_event("tracking", f"Tracking restarted for profile '{profile}': trackers and opponent selection cleared")

    # ================================================================ clicks

    def _on_frame_clicked(self, u: float, v: float) -> None:
        """A bare click does not say how big the object is; it only reminds the operator to drag a box."""
        self._click_message = "a click does not say how big the object is: DRAG a box around it"
        self.click_line.setText(self._click_message)

    def _on_region_selected(self, x: float, y: float, w: float, h: float) -> None:
        """A box dragged on the picture: it becomes the initial region of interest, on the frame the operator was looking at."""
        result = self._result
        if result is None:
            self._click_message = "no camera picture to select on yet"
        elif self.worker.paused or not self.worker.isRunning():
            self._click_message = "tracking is not running"
        else:
            self.worker.post("roi", (x, y, w, h), result.raw, result.t)
            self._click_message = f"box {w:.0f} x {h:.0f} px at ({x:.0f}, {y:.0f}) sent: waiting for the answer"
        self.frame_view.unsetCursor()
        self.click_line.setText(self._click_message)

    def arm_selection(self) -> None:
        self.frame_view.setCursor(Qt.CursorShape.CrossCursor)
        self._click_message = "DRAG a rectangle around the opponent (any object, any size)"
        self.click_line.setText(self._click_message)

    def arm_reselection(self) -> None:
        self.frame_view.setCursor(Qt.CursorShape.CrossCursor)
        self._click_message = "DRAG a new rectangle: the current track stays until the new box is accepted"
        self.click_line.setText(self._click_message)

    def clear_opponent(self) -> None:
        self.worker.post("clear")
        self.opponent_trail.clear()
        self.frame_view.unsetCursor()
        self._click_message = "opponent selection cleared: DRAG a box around the opponent to select it again"
        self.click_line.setText(self._click_message)

    # ================================================================ controller lines (listen-only) -> telemetry

    def _on_controller_line(self, direction: str, text: str) -> None:
        self.telemetry.feed(direction, text, time.monotonic())

    # ================================================================ refresh

    def refresh(self, now: float | None = None) -> None:
        """Paint the newest finished result. Cheap and safe without a camera, a result or a visible window."""
        if self._closed:
            return
        now = time.monotonic() if now is None else now
        self._refreshes += 1
        self._check_profile()
        result = self.worker.latest()
        if result is not None and result.epoch == self._epoch and result.seq != self._painted_seq:
            self._paint(result)
        self._paint_live(now)
        self.update_warnings(now)
        if self.isVisible() and (self._refreshes % SHADOW_EVERY_N_REFRESHES == 1 or self._shadow_on):
            self._refresh_shadow(now)                              # (not while hidden: reading the checkpoint record is not free)
        self._refresh_chip()

    def _paint(self, result: TrackingResult) -> None:
        self._result, self._painted_seq = result, result.seq
        self._own_aged_view, self._own_display = None, result.own
        mock = bool(self.hub.options.mock)
        self.frame_view.badge = ("MOCK  " if mock else "") + f"profile {self.profile_name()}"
        self.frame_view.set_frame(result.annotated)
        for note in result.notes:
            self._add_event("tracking", note)
        for event in result.click_events:
            if event["event"] == "roi":
                self._click_message = (("opponent selected: " if event["accepted"] else "box REFUSED: ") + event["reason"])
                self._add_event("tracking", self._click_message)
            elif event["event"] == "click":
                self._click_message = (("opponent selected: " if event["accepted"] else "click REFUSED: ") + event["reason"])
                self._add_event("tracking", self._click_message)
        self.click_line.setText(self._click_message or DRAG_HINT)

        own, opponent = result.own, result.opponent
        self._paint_own(result, own)
        self.opponent_grid.set_rows(tm.opponent_rows(opponent))
        selected = opponent is not None and opponent["state"] != "not_selected"
        self.clear_button.setEnabled(selected)
        self.reselect_button.setEnabled(selected)
        _set(self.status_label, tm.opponent_status_line(opponent), tm.OPPONENT_STATE_LEVEL.get(opponent["state"], "plain") if opponent else "off")

        dictionary_text, dictionary_level = dictionary_status(result.dictionary)
        _set(self.dictionary_label, f"Dictionary in use: {dictionary_text}.  Frames in the rate window: {result.marker_frames}.  "
                                    f"Decoded this frame: {result.marker_ids or 'none'}", dictionary_level)
        _fill(self.marker_table, [([str(row.marker_id), row.face, "yes" if row.seen_now else "no", row.rate_text, row.status], row.level)
                                  for row in result.marker_rows])
        unknown = [row.marker_id for row in result.marker_rows if not row.configured]
        never = [row.marker_id for row in result.marker_rows if row.configured and not row.ever_seen]
        _set(self.marker_summary, ("UNKNOWN ids seen (not in the marker layout): " + str(unknown) if unknown else
                                   "No unknown ids seen.") + "   " +
             ("Configured but NEVER seen since tracking started: " + str(never) if never else "Every configured id has been seen."),
             "bad" if unknown else ("warn" if never else "ok"))

        # top-down map
        own_xy = None if own is None or own.get("x_m") is None else (own["x_m"], own["y_m"])
        self.own_trail.add(result.t, own_xy if own is not None and own["state"] in ("measured", "predicted") else None)
        opp_xy = None if opponent is None or opponent.get("x_m") is None else (opponent["x_m"], opponent["y_m"])
        self.opponent_trail.add(result.t, opp_xy)
        self._paint_map(result, own)
        self._diagnostics = result.diagnostics

    # ---- the own robot: shown at the time of PAINTING, not only at the time of the frame

    def _lost_after_s(self) -> float:
        track = getattr(self.engine, "track", None)
        return float(getattr(getattr(track, "tracker_settings", None), "lost_after_s", TrackerSettings().lost_after_s))

    def _own_now(self, result: TrackingResult, now: float) -> dict[str, Any] | None:
        """The own-robot report aged to ``now``: frames that stop arriving turn a pose STALE and then LOST on screen."""
        return tm.age_own_report(result.own, max(0.0, now - result.done_at), result.limits["predict_hold_s"], self._lost_after_s())

    def _paint_own(self, result: TrackingResult, own: dict[str, Any] | None) -> None:
        self.own_grid.set_rows(tm.own_rows(own))
        confidence = tm.tracking_confidence(own, result.evidence, result.limits["position_m"], result.limits["heading_deg"],
                                            result.limits["predict_hold_s"])
        _set(self.confidence_label, "Tracking confidence: " + confidence.text(), confidence.ui_level)
        _set(self.confidence_inputs, "from: " + "; ".join(f"{name} {value}" for name, value in confidence.inputs))

    def _paint_map(self, result: TrackingResult, own: dict[str, Any] | None) -> None:
        usable = result.map_level in ("ok", "weak") and not result.arena_problems
        own_xy = None if own is None or own.get("x_m") is None else (own["x_m"], own["y_m"])
        opponent = result.opponent
        opp_xy = None if opponent is None or opponent.get("x_m") is None else (opponent["x_m"], opponent["y_m"])
        markers: list[Marker] = []
        if usable and own_xy is not None:
            colour = {"measured": "#55c889", "predicted": "#e8b85b"}.get(own["state"], "#ef6b73")
            age = own.get("age_s")
            markers.append(Marker(own_xy[0], own_xy[1], own.get("heading_deg"), colour,
                                  "AutoBot " + own["state"].upper() + ("" if age is None or own["state"] == "measured" else f" {1000 * age:.0f} ms"),
                                  hollow=own["state"] != "measured"))
        if usable and opp_xy is not None:
            markers.append(Marker(opp_xy[0], opp_xy[1], None, "#ef6b73", "opponent"))
        trails = [Trail([self._np(run) for run in self.own_trail.runs()], "#55c889", label="own"),
                  Trail([self._np(run) for run in self.opponent_trail.runs()], "#ef6b73", label="opponent")]
        note = ""
        if result.arena_problems:
            note = "Arena geometry unusable: " + "; ".join(result.arena_problems)
        elif result.map_level == "none":
            note = tm.NOT_CALIBRATED_TEXT
        elif result.map_level == "invalid":
            note = "Arena mapping INVALID: " + result.map_reason
        self.map.set_content(trails=trails if usable else [], markers=markers if usable else [], note=note,
                             caption="approximate positions (not height-corrected) for the opponent")

    @staticmethod
    def _np(run):
        import numpy as np

        return np.asarray(run, float)

    def _paint_live(self, now: float) -> None:
        """Things that change without a new result: camera state, ages, the profile, diagnostics."""
        state, detail, fps, age = self._camera()
        text, level = tm.camera_summary(state, detail, fps, age)
        _set(self.camera_line, text, level)
        _set(self.profile_label, f"profile: {self.profile_name()}")
        result = self._result
        diagnostics = self._diagnostics
        if result is not None and result.own is not None:
            aged = self._own_now(result, now)
            if aged is not None and self._own_aged_view != (aged["state"], round(aged.get("age_s") or 0.0, 1)):
                self._own_aged_view = (aged["state"], round(aged.get("age_s") or 0.0, 1))        # repaint when the state or the age (to 0.1 s) changes
                self._own_display = aged
                self._paint_own(result, aged)
                self._paint_map(result, aged)

        def ms(value):
            return "-" if value is None else f"{value:.1f} ms"

        rows = [("Software time per frame", f"mean {ms(diagnostics.get('ms_mean'))}, p95 {ms(diagnostics.get('ms_p95'))}, "
                                            f"last {ms(diagnostics.get('ms_last'))}  (markers + trackers; camera latency not included)", "plain"),
                ("Processed rate", "-" if diagnostics.get("fps") is None else f"{diagnostics['fps']:.1f} frames/s", "plain"),
                ("Skipped frames", f"{diagnostics.get('skipped_total', 0)} since tracking started "
                                   "(the vision thread always takes the newest frame, so a slow frame skips the ones behind it)",
                 "warn" if diagnostics.get("skipped_total", 0) else "plain"),
                ("Queue age", "-" if diagnostics.get("queue_age_ms") is None else
                 f"{diagnostics['queue_age_ms']:.0f} ms from capture until the vision thread took the frame", "plain"),
                ("Camera frame age", "no frame yet" if age is None else f"{1000 * age:.0f} ms", "plain" if age is not None and age < 0.5 else "warn"),
                ("Result age", "-" if result is None else f"{1000 * max(0.0, now - result.t):.0f} ms since the shown frame was captured", "plain")]
        self.diagnostics_grid.set_rows(rows)

    # ================================================================ warnings

    def conditions(self, now: float) -> dict[str, tm.Condition]:
        out: dict[str, tm.Condition] = {}
        state, detail, _, age = self._camera()
        if state in ("LOST", "UNAVAILABLE"):
            out["camera"] = tm.Condition("camera", f"CAMERA {state}: {detail or 'no frames'}", "bad")
        elif state in ("LIVE", "MOCK", "REPLAY") and age is not None and age > 2.0:
            out["camera"] = tm.Condition("camera", f"CAMERA frames have stopped ({age:.1f} s since the last one)", "bad")
        result = self._result
        if result is None:
            return out
        if result.arena_problems:
            out["mapping_invalid"] = tm.Condition("mapping_invalid", "ARENA GEOMETRY UNUSABLE: " + "; ".join(result.arena_problems), "bad")
        elif result.map_level == "none":
            out["mapping_missing"] = tm.Condition("mapping_missing", tm.NOT_CALIBRATED_TEXT, "bad")
        elif result.map_level == "invalid":
            out["mapping_invalid"] = tm.Condition("mapping_invalid", "ARENA MAPPING INVALID: " + result.map_reason, "bad")
        elif result.map_level == "weak":
            out["mapping_weak"] = tm.Condition("mapping_weak", "ARENA FIT IS WEAK: " + (result.map_reason or "see the ARENA tab"), "warn")
        if result.placeholders:
            out["placeholders"] = tm.Condition("placeholders", "PLACEHOLDER config values still in use: " + "; ".join(result.placeholders), "warn")
        if result.map_level in ("ok", "weak"):
            own = self._own_display          # the report as painted: aged to the refresh clock, see _paint_live
            if own is not None and own["state"] == "lost":
                out["own_lost"] = tm.Condition("own_lost", "OWN ROBOT LOST: " + own["reason"], "bad")
            elif own is not None and own["state"] == "stale":
                out["own_stale"] = tm.Condition("own_stale", "OWN ROBOT STALE (last pose is display only): " + own["reason"], "bad")
            sized = [] if own is None else [f"{k}: {v}" for k, v in (own.get("rejected") or {}).items() if "but is configured as" in v]
            if sized and own["state"] != "measured":
                out["marker_size"] = tm.Condition("marker_size", "MARKERS ARE DECODED BUT REJECTED AS THE WRONG SIZE (nothing can be tracked): "
                                                  + sized[0] + ("" if len(sized) == 1 else f"  (+{len(sized) - 1} more)"), "bad")
            opponent = result.opponent
            if opponent is not None and opponent["state"] == "lost":
                out["opponent_lost"] = tm.Condition("opponent_lost", "OPPONENT LOST: drag a new box around it. " + opponent["reason"], "bad")
            elif opponent is not None and opponent["state"] == "uncertain":
                out["opponent_uncertain"] = tm.Condition("opponent_uncertain", "OPPONENT TRACKING UNCERTAIN (no position): " + opponent["reason"], "warn")
            elif opponent is not None and opponent["state"] == "ambiguous":
                out["opponent_ambiguous"] = tm.Condition("opponent_ambiguous", "OPPONENT AMBIGUOUS (no position): " + opponent["reason"], "warn")
        if now - result.done_at > RESULT_STALE_S and not self.worker.paused and "camera" not in out:
            out["vision_stalled"] = tm.Condition("vision_stalled", "the vision thread has produced no new result for "
                                                                   f"{now - result.done_at:.1f} s", "bad")
        return out

    def update_warnings(self, now: float | None = None) -> list[tm.WarningEvent]:
        now = time.monotonic() if now is None else now
        events = self.warnings.update(now, self.conditions(now))
        for event in events:
            self._add_event("tracking", event.text)
        active = self.warnings.active
        if active:
            self.banner.setText("\n".join(c.text for c in active[:5]) + ("" if len(active) <= 5 else f"\n... and {len(active) - 5} more"))
            worst = "bad" if any(c.severity == "bad" for c in active) else "warn"
            colours = {"bad": ("#5b1f25", "#ffd9dc"), "warn": ("#4d3a12", "#ffeab5")}[worst]
            self.banner.setStyleSheet(f"background: {colours[0]}; color: {colours[1]};")
            self.banner.show()
        else:
            self.banner.hide()
            self.banner.setText("")
        return events

    # ================================================================ readiness

    def readiness(self) -> Readiness:
        result = self._result
        fresh = (result is not None and result.epoch == self._epoch and not self.worker.paused and self.worker.isRunning()
                 and time.monotonic() - result.done_at <= RESULT_STALE_S)
        return self.readiness_for(result if fresh else None, own=self._own_display if fresh else None)

    def readiness_for(self, result: TrackingResult | None, own: dict[str, Any] | None = None) -> Readiness:
        """The readiness a given result implies. ``None``: no fresh result (the tab is hidden, or nothing has arrived).

        ``own``: the own-robot report aged to the time of the question (``readiness()`` passes it); without it the result's own report is used."""
        state, detail, _, _ = self._camera()
        if state in ("LOST", "UNAVAILABLE"):
            return Readiness(FAILED, f"Camera {state.lower()}", detail or "the camera delivers no frames",
                             "Check the USB cable, then reopen the camera in SETUP")
        clicks = self.hub.vision.clicks
        if result is None:
            if clicks is None or clicks.historical or not clicks.points:
                return Readiness(NOT_CONFIGURED, "Arena not calibrated", tm.NOT_CALIBRATED_TEXT, "Calibrate the arena in the ARENA tab")
            return Readiness(WARNING, "Tracking not verified", "this tab has no fresh result yet",
                             "Open ROBOT TRACKING with the robot in view of the camera")
        if result.arena_problems:
            return Readiness(FAILED, "Arena geometry unusable", "; ".join(result.arena_problems), "Fix the arena geometry in the ARENA tab")
        if result.map_level == "none":
            return Readiness(NOT_CONFIGURED, "Arena not calibrated", tm.NOT_CALIBRATED_TEXT, "Calibrate the arena in the ARENA tab")
        if result.map_level == "invalid":
            return Readiness(FAILED, "Arena mapping invalid", result.map_reason, "Re-click the reference points in the ARENA tab")
        problems: list[str] = []
        own = own if own is not None else result.own
        if own is None or own["state"] in ("stale", "lost"):
            problems.append("own robot " + ("not tracked" if own is None else own["state"].upper()))
        elif own.get("rejected"):
            problems.append("visible markers disagree: " + "; ".join(f"{k}: {v}" for k, v in own["rejected"].items()))
        if result.map_level == "weak":
            problems.append("arena fit is WEAK")
        if result.placeholders:
            problems.append("placeholder config values still in use")
        if problems:
            return Readiness(WARNING, problems[0][0].upper() + problems[0][1:], "; ".join(problems),
                             "Put the robot in view, then check the markers and the arena fit")
        return Readiness(READY, "Tracking OK", f"own robot {own['state'].upper()}, markers agreeing {own['used_ids']}")

    def _refresh_chip(self) -> None:
        readiness = self.readiness()
        text = f"{readiness.level.upper()}: {readiness.headline}"
        if self.readiness_chip.text() != text:
            self.readiness_chip.setText(text)
            self.readiness_chip.setStyleSheet(f"background: {CHIP_COLOURS[readiness.level]};")
            self.readiness_chip.setToolTip(readiness.detail + (("\n" + readiness.fix) if readiness.fix else ""))

    # ================================================================ shadow mode (display only)

    def _ensure_contract(self) -> None:
        if self._contract is not None or self._contract_error:
            return
        try:
            self._contract = self._contract_loader()
        except Exception as error:
            self._contract_error = f"{type(error).__name__}: {error}"

    def prerequisites(self, now: float) -> list[tm.Prerequisite]:
        self._ensure_contract()
        result = self._result
        report = self.telemetry.report(now)
        size = None if self._contract is None else (self._contract.arena_width_m, self._contract.arena_height_m)
        frame_ok, frame_detail = tm.frame_compatibility(self.config.arena, self.profile_name(), size)
        if result is None:
            mapping_ok, mapping_detail = False, "no camera result yet"
        elif result.arena_problems:
            mapping_ok, mapping_detail = False, "arena geometry unusable: " + "; ".join(result.arena_problems)
        else:
            mapping_ok = result.map_level in ("ok", "weak")
            mapping_detail = ({"ok": "mapping valid", "weak": "mapping valid, fit WEAK",
                               "none": tm.NOT_CALIBRATED_TEXT, "invalid": "mapping INVALID: " + result.map_reason}[result.map_level])
        own = None if result is None else self._own_display
        opponent = None if result is None else result.opponent
        own_ok = own is not None and own["state"] in ("measured", "predicted")
        opponent_ok = opponent is not None and opponent["state"] == "tracked"
        return tm.shadow_prerequisites(
            frame_ok=frame_ok, frame_detail=frame_detail, mapping_ok=mapping_ok, mapping_detail=mapping_detail,
            contract_ok=self._contract is not None,
            contract_detail=(f"checkpoint {self._contract.identifier}" if self._contract is not None
                             else f"cannot read the checkpoint record: {self._contract_error}"),
            imu_present=report["present"], imu_present_detail=report["present_detail"], imu_fresh=report["fresh"],
            imu_fresh_detail=report["fresh_detail"], mounting_verified=report["mounting_verified"], mounting_detail=report["mounting_detail"],
            own_ok=own_ok, own_detail=("no result yet" if own is None else f"own robot {own['state'].upper()}"),
            opponent_ok=opponent_ok, opponent_detail=("no result yet" if opponent is None else f"opponent {opponent['state'].upper().replace('_', ' ')}"),
            checkpoint_state=self._load_state, checkpoint_detail=self._load_detail)

    def _refresh_shadow(self, now: float) -> None:
        prereqs = self.prerequisites(now)
        _fill(self.prereq_table, [([p.label + ("" if p.cheap else "  (live input)"), p.word, p.detail], "ok" if p.ok else "bad") for p in prereqs])
        enabled, why = tm.shadow_checkbox_state(prereqs)
        if enabled or not self._shadow_on:
            self._blocked_since = None
        else:                                                      # a brief IMU hiccup only blocks proposals (the assembler says so);
            if self._blocked_since is None:                        # shadow is switched off only when the prerequisite stays missing
                self._blocked_since = now
            if now - self._blocked_since >= SHADOW_GRACE_S:
                self._set_shadow_checked(False)
                self._shadow_message = "shadow stopped: " + why
        self.shadow_checkbox.setEnabled(enabled or self._shadow_on)
        _set(self.shadow_why, (self._shadow_message + "  |  " if self._shadow_message else "") + why, "ok" if enabled else "bad")
        try:
            missing = list(self.hub.live_shadow_missing()) if hasattr(self.hub, "live_shadow_missing") else []
        except Exception as error:
            missing = [f"hub could not report: {error}"]
        _set(self.shadow_hub, "Hub says it still lacks: " + ("; ".join(missing) if missing else "nothing"))
        if self._load_state == "loading" and self._loader is not None:
            _set(self.load_label, f"Loading the checkpoint on a background thread: {time.monotonic() - self._loader.started_at:.0f} s "
                                  "elapsed (usually 30-60 s). The rest of the screen keeps working.", "warn")
        else:
            _set(self.load_label, "Checkpoint: " + self._load_detail, "ok" if self._load_state == "loaded" else "off")
        summary = None if self._result is None else self._result.shadow
        text, level, detail = tm.shadow_headline(self._shadow_on, self._load_state, summary)
        _set(self.shadow_headline, text, level)
        _set(self.shadow_detail, "\n".join(detail))

    def _set_shadow_checked(self, checked: bool) -> None:
        blocked = self.shadow_checkbox.blockSignals(True)
        self.shadow_checkbox.setChecked(checked)
        self.shadow_checkbox.blockSignals(blocked)
        if not checked and self._shadow_on:
            self._shadow_on = False
            self.worker.post("shadow", None)

    def _on_shadow_toggled(self, checked: bool) -> None:
        if not checked:
            self._shadow_on = False
            self._shadow_message = "shadow off"
            self.worker.post("shadow", None)
            self._refresh_shadow(time.monotonic())
            return
        enabled, why = tm.shadow_checkbox_state(self.prerequisites(time.monotonic()))
        if not enabled:                                            # the checkbox is disabled then, but never trust the widget alone
            self._set_shadow_checked(False)
            self._shadow_message = "shadow refused: " + why
            return
        self._shadow_on = True
        self._shadow_message = "shadow on: display only"
        self._post_shadow(True)
        if self._policy is None and self._load_state != "loading":
            self._start_loading()

    def _post_shadow(self, on: bool) -> None:
        if not on:
            self.worker.post("shadow", None)
            return
        capture = self.hub.capture
        source = (f"{capture.source_info.get('description') or capture.state} profile '{self.profile_name()}' "
                  "+ controller telemetry from the Command Centre monitor (listen-only)")
        runner = ShadowRunner(self._contract, self.config, self.hub.vision.clicks or ClickSet(), self.telemetry, source, self._policy)
        self.worker.post("shadow", runner)

    def _start_loading(self) -> None:
        self._load_state, self._load_detail = "loading", "loading"
        self._loader = CheckpointLoader(self._policy_factory, self._contract, self)
        self._loader.loaded.connect(self._on_policy_loaded)
        self._loader.failed.connect(self._on_policy_failed)
        self._loader.start()

    def _on_policy_loaded(self, policy) -> None:
        self._policy = policy
        self._load_state, self._load_detail = "loaded", f"checkpoint loaded ({self._contract.identifier})"
        self._add_event("tracking", "Shadow checkpoint loaded (display only)")
        if self._shadow_on:
            self.worker.post("policy", policy)

    def _on_policy_failed(self, message: str) -> None:
        self._load_state, self._load_detail = "failed", f"checkpoint FAILED to load: {message}"
        self._add_event("tracking", self._load_detail)
