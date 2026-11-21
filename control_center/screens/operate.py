"""Operate: the match screen. One large view, who has control, what is working, and recording.

Nothing on this screen arms, hands over or drives the robot. The control source shown here is what
the controller last reported; when that report is missing or old the screen says UNKNOWN or STALE.
"""

from __future__ import annotations

import time
from collections import deque

import numpy as np
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (QButtonGroup, QCheckBox, QFrame, QGridLayout, QHBoxLayout, QLabel, QLineEdit,
                               QPlainTextEdit, QPushButton, QSizePolicy, QStackedWidget, QVBoxLayout, QWidget)

from control_center.backend import status as st
from control_center.backend.hub import OperationsHub
from control_center.backend.live_vision import VisionSample
from control_center.tools.overlay_core import PoseStatus
from control_center.widgets.arena_map import ArenaMapWidget, Marker, Trail
from control_center.widgets.frame_view import FrameView

LEVEL_COLOURS = {"ok": "#55c889", "warn": "#e8b85b", "bad": "#ef6b73", "off": "#7f8996"}
TRAIL_SECONDS = 20.0
QUIET_EVENT_KINDS = {"app"}


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


class StatusRow(QWidget):
    """A dot, a name and a value. The detail goes in the tooltip so the row stays one line."""

    def __init__(self, name: str) -> None:
        super().__init__()
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 2, 0, 2)
        layout.setSpacing(8)
        self.dot = QLabel("●")
        self.name = QLabel(name)
        self.name.setStyleSheet("color: #9eabb8;")
        self.name.setFixedWidth(92)
        self.value = QLabel("—")
        self.value.setWordWrap(True)
        self.value.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        layout.addWidget(self.dot)
        layout.addWidget(self.name)
        layout.addWidget(self.value, 1)

    def show_item(self, item: st.Item) -> None:
        colour = LEVEL_COLOURS[item.level]
        self.dot.setStyleSheet(f"color: {colour};")
        self.value.setText(item.text)
        self.value.setStyleSheet(f"color: {'#d9dee5' if item.level != 'off' else '#9aa4b0'}; font-weight: 600;")
        self.setToolTip(item.detail)


class OperateScreen(QWidget):
    review_requested = Signal(str)          # folder of the round that was just saved

    def __init__(self, hub: OperationsHub, parent=None) -> None:
        super().__init__(parent)
        self.hub = hub
        self._trail: deque[tuple[float, float, float, bool]] = deque()      # t, x, y, starts_new_run
        self._last_sample: VisionSample | None = None
        self._last_valid: VisionSample | None = None
        self._shown_frame = -1
        self._last_folder: str | None = None
        self._build()
        hub.snapshot_changed.connect(self.show_snapshot)
        hub.event_added.connect(self._on_event)
        hub.sample_ready.connect(self._on_sample)
        hub.recording_finished.connect(self._on_recording_finished)
        for stamp, kind, text in hub.events:
            self._on_event(stamp, kind, text)
        self._display_timer = QTimer(self)
        self._display_timer.timeout.connect(self._refresh_view)
        self._display_timer.start(33)

    # ---- layout ----

    def _build(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(10, 8, 10, 10)
        root.setSpacing(8)
        body = QHBoxLayout()
        body.setSpacing(10)
        root.addLayout(body, 1)

        # Left: the view.
        view_column = QVBoxLayout()
        view_column.setSpacing(6)
        bar = QHBoxLayout()
        self.camera_button = QPushButton("Camera")
        self.map_button = QPushButton("Map")
        group = QButtonGroup(self)
        for index, button in enumerate((self.camera_button, self.map_button)):
            button.setCheckable(True)
            button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            button.setMinimumWidth(96)
            group.addButton(button, index)
            bar.addWidget(button)
        self.camera_button.setChecked(True)
        self.overlay_box = QCheckBox("Draw arena and bot on the preview (display only; the recording stays raw)")
        self.overlay_box.setChecked(True)
        self.overlay_box.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        bar.addSpacing(12)
        bar.addWidget(self.overlay_box)
        bar.addStretch(1)
        self.caption = QLabel("")
        self.caption.setStyleSheet("font-size: 12pt; font-weight: 700;")
        bar.addWidget(self.caption)
        view_column.addLayout(bar)
        self.stack = QStackedWidget()
        self.frame_view = FrameView()
        self.map_view = ArenaMapWidget(self.hub.overlay_config.arena)
        profile_changed = getattr(self.hub, "profile_changed", None)
        if profile_changed is not None:                      # another profile or a newly saved arena: redraw the map
            profile_changed.connect(lambda _name: self.map_view.set_arena(self.hub.overlay_config.arena))
        self.stack.addWidget(self.frame_view)
        self.stack.addWidget(self.map_view)
        group.idClicked.connect(self.stack.setCurrentIndex)
        view_column.addWidget(self.stack, 1)
        body.addLayout(view_column, 1)

        # Right: control, status, recording.
        side = QVBoxLayout()
        side.setSpacing(8)
        side_widget = QWidget()
        side_widget.setLayout(side)
        side_widget.setFixedWidth(400)
        body.addWidget(side_widget)

        control, layout = _card("CONTROL SOURCE  ·  reported by the controller")
        self.source_label = QLabel("UNKNOWN")
        self.source_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.source_label)
        grid = QGridLayout()
        grid.setHorizontalSpacing(10)
        self.armed_label = QLabel("UNKNOWN")
        self.requested_label = QLabel("UNKNOWN")
        for column, (title, label) in enumerate((("ARM STATE", self.armed_label), ("RADIOMASTER REQUESTS", self.requested_label))):
            heading = QLabel(title)
            heading.setStyleSheet("color: #9eabb8; font-size: 8pt; font-weight: 700;")
            heading.setAlignment(Qt.AlignmentFlag.AlignCenter)
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            grid.addWidget(heading, 0, column)
            grid.addWidget(label, 1, column)
        layout.addLayout(grid)
        self.control_detail = QLabel("")
        self.control_detail.setWordWrap(True)
        self.control_detail.setStyleSheet("color: #9eabb8;")
        layout.addWidget(self.control_detail)
        control.setToolTip("This screen cannot arm, hand over or drive. The RadioMaster and the controller decide.")
        side.addWidget(control)

        status_card, layout = _card("STATUS")
        self.rows = {key: StatusRow(name) for key, name in (
            ("robot_link", "Robot link"), ("camera", "Camera"), ("map", "Map"), ("own_bot", "Own bot"),
            ("opponent", "Opponent"), ("policy", "Policy"), ("imu", "IMU"))}
        for row in self.rows.values():
            layout.addWidget(row)
        side.addWidget(status_card)

        record_card, layout = _card("ROUND RECORDING")
        self.record_state = QLabel("Not recording")
        self.record_state.setStyleSheet("font-size: 15pt; font-weight: 800;")
        layout.addWidget(self.record_state)
        self.label_edit = QLineEdit()
        self.label_edit.setPlaceholderText("Round label (optional), e.g. heat1")
        layout.addWidget(self.label_edit)
        self.record_button = QPushButton("●  RECORD ROUND   (F8)")
        self.record_button.setObjectName("recordButton")
        self.record_button.setMinimumHeight(50)
        self.mark_button = QPushButton("MARK ROUND START   (F9 or Space)")
        self.mark_button.setMinimumHeight(40)
        self.stop_button = QPushButton("■  STOP RECORDING")
        self.stop_button.setObjectName("warningButton")
        self.stop_button.setMinimumHeight(40)
        for button in (self.record_button, self.mark_button, self.stop_button):
            button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            layout.addWidget(button)
        self.record_note = QLabel("Records the raw camera video and whatever telemetry is present. "
                                  "Mark Round Start only adds a timestamp.")
        self.record_note.setWordWrap(True)
        self.record_note.setStyleSheet("color: #9eabb8;")
        layout.addWidget(self.record_note)
        self.review_button = QPushButton("Review the round just saved")
        self.review_button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.review_button.setVisible(False)
        layout.addWidget(self.review_button)
        side.addWidget(record_card)
        side.addStretch(1)

        self.record_button.clicked.connect(self.start_recording)
        self.stop_button.clicked.connect(self.stop_recording)
        self.mark_button.clicked.connect(self.mark_round_start)
        self.review_button.clicked.connect(lambda: self._last_folder and self.review_requested.emit(self._last_folder))
        for keys, slot in (("F8", self.start_recording), ("F9", self.mark_round_start), ("Space", self.mark_round_start)):
            shortcut = QShortcut(QKeySequence(keys), self)
            shortcut.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
            shortcut.activated.connect(slot)

        # Bottom: a small event log.
        self.event_log = QPlainTextEdit()
        self.event_log.setReadOnly(True)
        self.event_log.setMaximumBlockCount(300)
        self.event_log.setFixedHeight(120)
        self.event_log.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        root.addWidget(self.event_log)
        self._show_recording(False, 0.0)

    # ---- recording ----

    def start_recording(self) -> None:
        if self.hub.recorder is not None:
            return
        ok, message = self.hub.start_recording(self.label_edit.text())
        if not ok:
            self.record_note.setText(f"Could not start: {message}")
            self.record_note.setStyleSheet("color: #ef6b73;")
        else:
            self.label_edit.clearFocus()
            self.review_button.setVisible(False)
            self.record_note.setText(f"Recording to {message}")
            self.record_note.setStyleSheet("color: #9eabb8;")

    def stop_recording(self) -> None:
        self.hub.stop_recording()

    def mark_round_start(self) -> None:
        ok, message = self.hub.mark_round_start()
        self.record_note.setText(message if ok else f"Round start NOT marked: {message}")
        self.record_note.setStyleSheet(f"color: {'#55c889' if ok else '#ef6b73'};")

    def _on_recording_finished(self, summary) -> None:
        self._last_folder = str(summary.folder)
        clean = summary.video_verified and not summary.write_errors and not summary.frames_dropped
        self.record_note.setText(summary.headline())
        self.record_note.setStyleSheet(f"color: {'#55c889' if clean else '#e8b85b'};")
        self.review_button.setVisible(True)

    def _show_recording(self, active: bool, elapsed: float, item: st.Item | None = None) -> None:
        # Idle: label + Record. Recording: Mark Round Start + Stop. Never both sets at once.
        self.record_button.setVisible(not active)
        self.label_edit.setVisible(not active)
        self.stop_button.setVisible(active)
        self.mark_button.setVisible(active)
        if item is not None:
            self.record_state.setText(("● " if active else "") + item.text)
            colour = "#ef6b73" if active else LEVEL_COLOURS[item.level] if item.level != "off" else "#9aa4b0"
            self.record_state.setStyleSheet(f"font-size: 15pt; font-weight: 800; color: {colour};")
            self.record_state.setToolTip(item.detail)

    # ---- status ----

    def show_snapshot(self, snapshot: st.Snapshot) -> None:
        control = snapshot.control
        colour = LEVEL_COLOURS[control.source.level]
        self.source_label.setText(control.source.text)
        self.source_label.setStyleSheet(f"font-size: 36pt; font-weight: 800; color: {colour}; padding: 2px;")
        for label, item in ((self.armed_label, control.armed), (self.requested_label, control.requested)):
            label.setText(item.text)
            label.setStyleSheet(f"font-size: 17pt; font-weight: 800; color: {LEVEL_COLOURS[item.level]};")
        self.control_detail.setText(control.detail)
        for key, row in self.rows.items():
            row.show_item(getattr(snapshot, key))
        self.caption.setText(snapshot.camera_map_caption)
        self.frame_view.badge = {"MOCK": "MOCK · SYNTHETIC, NOT A CAMERA", "REPLAY": "REPLAY · NOT LIVE"}.get(self.hub.capture.state, "")
        if self.hub.capture.state not in ("LIVE", "MOCK", "REPLAY"):
            self.frame_view.set_frame(None)
            self.frame_view.set_placeholder(f"{snapshot.camera.text}\n{snapshot.camera.detail}")
        self._show_recording(snapshot.recording_active, snapshot.recording_elapsed_s, snapshot.recording)
        self._refresh_map(snapshot)

    def _on_event(self, stamp: str, kind: str, text: str) -> None:
        if kind not in QUIET_EVENT_KINDS:
            self.event_log.appendPlainText(f"[{stamp}] {text}")

    # ---- the view ----

    def _on_sample(self, sample: VisionSample) -> None:
        previous = self._last_sample
        self._last_sample = sample
        if sample.pose.ok:
            new_run = previous is None or not previous.pose.ok or sample.tag_switch
            self._trail.append((sample.t_mono, sample.pose.x_m, sample.pose.y_m, new_run))
            self._last_valid = sample
        while self._trail and sample.t_mono - self._trail[0][0] > TRAIL_SECONDS:
            self._trail.popleft()

    def _refresh_view(self) -> None:
        if self.stack.currentIndex() != 0 or not self.isVisible():
            return
        index, _, frame = self.hub.capture.latest()
        if frame is None or index == self._shown_frame:
            return
        self._shown_frame = index
        if self.overlay_box.isChecked():
            frame = self._annotated(frame)
        self.frame_view.set_frame(frame)

    def _annotated(self, frame: np.ndarray) -> np.ndarray:
        """Draw on a COPY. The frame object itself is the one being recorded."""
        from control_center.tools.arena_overlay import draw_arena, draw_bot

        vision = self.hub.vision
        mapping, sample = vision.mapping, self._last_sample
        if not mapping.valid and (sample is None or not sample.detections):
            return frame
        canvas = frame.copy()
        try:
            if mapping.valid:
                draw_arena(canvas, mapping, vision.config.arena)
            if sample is not None:
                draw_bot(canvas, mapping, vision.camera, PoseStatus(sample.pose, 0.0, None), sample.detections, vision.config)
        except Exception:          # a drawing problem must never stop the preview
            return frame
        return canvas

    def _trail_runs(self) -> list[np.ndarray]:
        runs: list[list[tuple[float, float]]] = []
        for _, x, y, new_run in self._trail:
            if new_run or not runs:
                runs.append([])
            runs[-1].append((x, y))
        return [np.array(run) for run in runs if run]

    def _refresh_map(self, snapshot: st.Snapshot) -> None:
        if self.stack.currentIndex() != 1:
            return
        markers: list[Marker] = []
        note = ""
        now = time.monotonic()
        sample, last = self._last_sample, self._last_valid
        if snapshot.map.level in ("off", "bad"):
            note = f"{snapshot.map.text}\nCalibrate in Setup to see positions here"
        elif sample is not None and sample.pose.ok and now - sample.t_mono < 0.5:
            markers.append(Marker(sample.pose.x_m, sample.pose.y_m, sample.pose.heading_deg, "#55c889", "own bot"))
        elif last is not None:
            markers.append(Marker(last.pose.x_m, last.pose.y_m, None, "#e8b85b",
                                  f"last seen {now - last.t_mono:.1f} s ago", hollow=True))
        self.map_view.set_content(
            trails=[Trail(self._trail_runs(), "#55c889", label="own bot (camera)")], markers=markers,
            caption=f"{snapshot.own_bot.text}   ·   opponent: {snapshot.opponent.text.lower()}", note=note)
