"""ARENA tab: calibrate the arena and its hazards by clicking on the camera picture.

LIVE (the competition arena, whose size comes from the design drawing; nothing is typed):

    1 freeze a frame   2 click the four corners   3 click the centre of pit 1   4 click the centre of pit 2
    5 click the flipper   6 check the overlay   7 Save   8 reload and confirm
    (optional: Advanced overrides for a measured difference from the design)

HOME TEST (a patch of any size, so its dimensions are typed): corners, size, pits with radius, flipper vertices.

The logic is ``control_center.backend.arena_workflow`` (tested without Qt) on top of
``control_center.tools.arena_setup``. This screen only shows it and passes clicks and typing through.
It reads the camera through the hub, writes the active profile's own two files when you press Save, and has
no connection to the robot.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (QAbstractItemView, QCheckBox, QFormLayout, QFrame, QGroupBox, QHBoxLayout, QHeaderView, QLabel,
                               QLineEdit, QPushButton, QScrollArea, QSplitter, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget)

from control_center.backend.arena_workflow import MODE_TITLES, ArenaWorkflow, render_saved_view, render_setup_view
from control_center.backend.profiles import profile_label
from control_center.backend.readiness import CHIP_COLOURS, FAILED, NOT_CONFIGURED, READY, WARNING, Readiness
from control_center.tools import arena_setup as A
from control_center.tools.arena_design import DesignError, Overrides
from control_center.widgets.frame_view import FrameView

log = logging.getLogger(__name__)
REFRESH_MS = 120
STATE_COLOURS = {"empty": "#7b8794", "placed": "#55c889", "saved": "#b28cf0", "current": "#ffd84d"}
STYLE = """
QLabel { background: transparent; }
QLabel#prompt { font-size: 13pt; font-weight: 800; color: #ffd84d; }
QLabel#chip { font-weight: 800; padding: 3px 10px; border-radius: 4px; color: #11161d; }
QLabel#hint { color: #9eabb8; }
QLabel#mono { font-family: Consolas; }
QTableWidget { background: #0d1218; alternate-background-color: #121922; gridline-color: #2b3542; border: 1px solid #2b3542; }
QLineEdit { background: #0d1218; border: 1px solid #3b4a5d; border-radius: 4px; padding: 6px; font-size: 12pt; }
"""


def _text(text: str = "", name: str | None = None) -> QLabel:
    label = QLabel(text)
    label.setTextFormat(Qt.TextFormat.PlainText)
    label.setWordWrap(True)
    if name:
        label.setObjectName(name)
    return label


def _color_text(label: QLabel, text: str, colour: str) -> None:
    if label.text() != text:
        label.setText(text)
    label.setStyleSheet(f"color: {colour};")


class ArenaScreen(QWidget):
    """Guided arena and hazard calibration for the active profile (HOME TEST or LIVE)."""

    def __init__(self, hub, parent=None, workflow: ArenaWorkflow | None = None) -> None:
        super().__init__(parent)
        self.hub = hub
        self._injected = workflow is not None
        self.workflow = workflow or ArenaWorkflow(hub.profile_name if hasattr(hub, "profile_name") else "live", getattr(hub, "app_state", None))
        self._frozen: np.ndarray | None = None
        self._checking_saved = False
        self._last_index = -2
        self._dirty = True
        self._shown_key: Any = None
        self.setStyleSheet(STYLE)

        root = QVBoxLayout(self)
        root.setContentsMargins(10, 8, 10, 8)
        header = QHBoxLayout()
        self.profile_label = QLabel()
        self.profile_label.setStyleSheet("font-size: 12pt; font-weight: 800; color: #6fa8dc;")
        header.addWidget(self.profile_label)
        header.addStretch(1)
        self.chip = QLabel(NOT_CONFIGURED)
        self.chip.setObjectName("chip")
        header.addWidget(self.chip)
        root.addLayout(header)
        self.status_line = _text("", "hint")
        root.addWidget(self.status_line)

        split = QSplitter(Qt.Orientation.Horizontal)
        self.view = FrameView()
        self.view.setMinimumSize(560, 340)
        self.view.clicked.connect(self._on_frame_clicked)
        split.addWidget(self.view)
        split.addWidget(self._build_side())
        split.setStretchFactor(0, 3)
        split.setStretchFactor(1, 2)
        split.setSizes([900, 640])
        root.addWidget(split, 1)

        self._timer = QTimer(self)
        self._timer.setInterval(REFRESH_MS)
        self._timer.timeout.connect(self.refresh)
        if hasattr(hub, "profile_changed"):
            hub.profile_changed.connect(self._on_profile_changed)
        if hasattr(hub, "restore_changed"):
            hub.restore_changed.connect(lambda _report: self.refresh())
        self.refresh()

    # ================================================================ layout
    def _build_side(self) -> QWidget:
        body = QWidget()
        column = QVBoxLayout(body)
        column.setContentsMargins(4, 0, 4, 0)
        column.setSpacing(6)

        row = QHBoxLayout()
        self.new_button = QPushButton("New set-up")
        self.new_button.setToolTip(MODE_TITLES["new"] + ": clicks the four corners, size, both pits and the flipper")
        self.recal_button = QPushButton("Recalibrate camera")
        self.recal_button.setToolTip(MODE_TITLES["recalibrate"] + ". Use after the camera or stand moved; the arena is unchanged.")
        self.edit_button = QPushButton("Edit arena")
        self.edit_button.setToolTip(MODE_TITLES["edit"] + ". Use when the arena itself changed; the camera clicks stay.")
        self.resume_button = QPushButton("Continue unfinished")
        for button, mode in ((self.new_button, "new"), (self.recal_button, "recalibrate"), (self.edit_button, "edit")):
            button.clicked.connect(lambda _checked=False, m=mode: self.start(m))
            row.addWidget(button)
        self.resume_button.clicked.connect(self.resume)
        row.addWidget(self.resume_button)
        column.addLayout(row)

        self.prompt_label = _text("Press New set-up to start.", "prompt")
        column.addWidget(self.prompt_label)

        entry = QHBoxLayout()
        self.entry = QLineEdit()
        self.entry.setPlaceholderText("type a size here (metres or millimetres), then Enter")
        self.entry.returnPressed.connect(self.submit_entry)
        self.entry_button = QPushButton("OK")
        self.entry_button.clicked.connect(self.submit_entry)
        entry.addWidget(self.entry, 1)
        entry.addWidget(self.entry_button)
        column.addLayout(entry)
        self.entry_hint = _text("", "hint")
        column.addWidget(self.entry_hint)
        self.entry.textChanged.connect(self._update_entry_hint)

        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["Step", "State", "Value"])
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setAlternatingRowColors(True)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self.table.setMinimumHeight(230)
        self.table.cellClicked.connect(self._on_row_clicked)
        column.addWidget(self.table, 1)
        self.table_hint = _text("Click a row to place that item again (the correction of one point). Violet rows are saved values that have not been re-clicked.", "hint")
        column.addWidget(self.table_hint)

        nav = QHBoxLayout()
        self.back_button = QPushButton("◀ Back")
        self.next_button = QPushButton("Next ▶")
        self.undo_button = QPushButton("Undo")
        self.reset_button = QPushButton("Reset")
        self.retry_button = QPushButton("Retry (new frame)")
        for button, slot in ((self.back_button, self.back), (self.next_button, self.next), (self.undo_button, self.undo),
                             (self.reset_button, self.reset), (self.retry_button, self.retry)):
            button.clicked.connect(lambda _checked=False, f=slot: f())
            nav.addWidget(button)
        column.addLayout(nav)

        flipper = QHBoxLayout()
        self.finish_button = QPushButton("Finish flipper")
        self.add_vertex_button = QPushButton("Add flipper vertex")
        self.remove_vertex_button = QPushButton("Remove selected vertex")
        for button, slot in ((self.finish_button, self.finish_flipper), (self.add_vertex_button, self.add_vertex),
                             (self.remove_vertex_button, self.remove_vertex)):
            button.clicked.connect(lambda _checked=False, f=slot: f())
            flipper.addWidget(button)
        column.addLayout(flipper)

        self.freeze_button = QPushButton("Freeze frame")
        self.freeze_button.setCheckable(True)
        self.freeze_button.clicked.connect(self.toggle_freeze)
        self.save_button = QPushButton("Save")
        self.save_button.setStyleSheet("font-weight: 800;")
        self.save_button.clicked.connect(lambda: self.save())
        self.reload_button = QPushButton("Reload saved")
        self.reload_button.clicked.connect(self.reload_saved)
        self.check_button = QPushButton("Check saved overlay")
        self.check_button.setCheckable(True)
        self.check_button.clicked.connect(self.toggle_check)
        actions = QHBoxLayout()
        for button in (self.freeze_button, self.save_button, self.reload_button, self.check_button):
            actions.addWidget(button)
        column.addLayout(actions)
        self.verify_box = QCheckBox("The saved overlay matches the floor: outline on the arena edges, circles on the pits, shape on the flipper")
        self.verify_box.setStyleSheet("font-weight: 700;")
        self.verify_box.clicked.connect(self.confirm_overlay)
        column.addWidget(self.verify_box)

        column.addWidget(self._build_restore_panel())
        self.design_label = _text("", "hint")
        column.addWidget(self.design_label)
        column.addWidget(self._build_overrides())
        self.accuracy_label = _text("", "mono")
        column.addWidget(self.accuracy_label)

        self.problem_label = _text()
        self.warning_label = _text()
        self.result_label = _text("", "mono")
        self.files_label = _text("", "hint")
        for label in (self.problem_label, self.warning_label, self.result_label, self.files_label):
            column.addWidget(label)
        column.addStretch(1)
        area = QScrollArea()
        area.setWidgetResizable(True)
        area.setWidget(body)
        area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        return area

    def _build_restore_panel(self) -> QWidget:
        """The remembered calibration between rounds: loaded? same camera settings? still aligned? -> continue or recalibrate."""
        box = QGroupBox("Saved calibration (restored automatically)")
        layout = QVBoxLayout(box)
        self.restore_label = _text("", "mono")
        layout.addWidget(self.restore_label)
        self.camera_label = _text("", "hint")
        layout.addWidget(self.camera_label)
        row = QHBoxLayout()
        self.quick_button = QPushButton("Quick alignment check")
        self.quick_button.setToolTip("Draw the saved arena over the live picture and compare it with the floor")
        self.align_ok_button = QPushButton("Alignment OK, continue")
        self.align_ok_button.setStyleSheet("font-weight: 800;")
        self.recheck_button = QPushButton("Re-check")
        self.reconnect_button = QPushButton("Reconnect camera")
        self.quick_button.clicked.connect(lambda _c=False: self.quick_check())
        self.align_ok_button.clicked.connect(lambda _c=False: self.confirm_alignment())
        self.recheck_button.clicked.connect(lambda _c=False: self.recheck())
        self.reconnect_button.clicked.connect(lambda _c=False: self.reconnect_camera())
        for button in (self.quick_button, self.align_ok_button, self.recheck_button, self.reconnect_button):
            row.addWidget(button)
        layout.addLayout(row)
        self.restore_box = box
        return box

    def _restore_report(self):
        return getattr(self.hub, "restore_report", None)

    def quick_check(self) -> None:
        """Show the SAVED arena over the live picture (the visual alignment check)."""
        self.workflow.reload()
        if self.workflow.loaded.arena is None:
            self._say("Nothing is saved to check yet. Run a New set-up.", "warn")
            return
        self.unfreeze()
        self._checking_saved = True
        self.check_button.setChecked(True)
        self._say("Showing the saved arena over the live picture. If the outline sits on the floor edges, the circles on the pits "
                  "and the shape on the flipper, press Alignment OK. If not, Recalibrate camera.", "ok")
        self._dirty = True
        self.refresh()

    def confirm_alignment(self) -> bool:
        """The operator says the overlay matches the real arena (this run only; asked again at the next start)."""
        if hasattr(self.hub, "confirm_alignment") and self._restore_report() is not None:
            ok, message = self.hub.confirm_alignment()
        else:
            ok = self.workflow.mark_verified()
            message = "Confirmed by eye." if ok else "Nothing valid is saved to confirm."
        if ok:
            self.workflow.mark_verified()
        self.verify_box.setChecked(ok and self.workflow.is_verified())
        self._say(message, "ok" if ok else "bad")
        self.refresh()
        return ok

    def recheck(self) -> None:
        if hasattr(self.hub, "evaluate_restore"):
            self.hub.evaluate_restore()
        self.refresh()

    def reconnect_camera(self) -> None:
        if hasattr(self.hub, "reconnect_camera"):
            ok, message = self.hub.reconnect_camera()
            self._say(message, "ok" if ok else "warn")
        self.refresh()

    def _build_overrides(self) -> QWidget:
        """Advanced: a MEASURED difference from the design (e.g. 1130 mm because walls stand 10 mm in). Off by default."""
        box = QGroupBox("Advanced: measured differences from the design (leave empty unless you measured)")
        box.setCheckable(True)
        box.setChecked(False)
        form = QFormLayout(box)
        self.override_fields: dict[str, QLineEdit] = {}
        for key, label in (("width_m", "Arena width"), ("height_m", "Arena height"), ("pit_radius_m", "Pit radius"),
                           ("flipper_leg_m", "Flipper leg length"), ("inset_per_side_m", "Inset per side (wall/floor edge)")):
            edit = QLineEdit()
            edit.setPlaceholderText("design value (metres or millimetres)")
            self.override_fields[key] = edit
            form.addRow(label, edit)
        self.override_reason = QLineEdit()
        self.override_reason.setPlaceholderText("how it was measured (kept in the saved file)")
        form.addRow("Reason", self.override_reason)
        row = QHBoxLayout()
        self.override_apply = QPushButton("Apply override")
        self.override_clear = QPushButton("Use the design")
        self.override_apply.clicked.connect(lambda _checked=False: self.apply_overrides())
        self.override_clear.clicked.connect(lambda _checked=False: self.clear_overrides())
        row.addWidget(self.override_apply)
        row.addWidget(self.override_clear)
        form.addRow(row)
        self.override_status = _text("", "hint")
        form.addRow(self.override_status)
        self.override_box = box
        return box

    def apply_overrides(self) -> bool:
        """Read the advanced fields. Empty fields stay at the design. Nothing is applied unless every entry is valid."""
        if self.workflow.design is None:
            self.override_status.setText("Overrides apply to the competition arena's design-based set-up. Start a New set-up first.")
            return False
        values: dict[str, float] = {}
        try:
            for key, edit in self.override_fields.items():
                if edit.text().strip():
                    values[key] = A.parse_length(edit.text())
        except ValueError as error:
            self.override_status.setText(str(error))
            return False
        ok, message = self.workflow.set_overrides(Overrides(reason=self.override_reason.text().strip(), **values))
        self.override_status.setText(message)
        self._say(message, "ok" if ok else "bad")
        self._dirty = True
        self.refresh()
        return ok

    def clear_overrides(self) -> bool:
        for edit in self.override_fields.values():
            edit.clear()
        self.override_reason.clear()
        if self.workflow.design is None:
            return False
        return self.apply_overrides()

    # ================================================================ actions
    @property
    def frozen(self) -> bool:
        return self._frozen is not None

    def _latest_frame(self) -> np.ndarray | None:
        try:
            _index, _t, frame = self.hub.capture.latest()
        except Exception:                                    # no capture session in some test hubs
            return None
        return frame

    def freeze(self, frame: np.ndarray | None = None) -> bool:
        frame = self._latest_frame() if frame is None else frame
        if frame is None:
            self._say("No camera frame to freeze. Check the camera in the GUIDED SETUP tab (camera step) and try Retry.", "bad")
            return False
        self._frozen = frame.copy()
        self.workflow.set_frame_size((frame.shape[1], frame.shape[0]))
        self.freeze_button.setChecked(True)
        self.freeze_button.setText("Live view")
        self._dirty = True
        return True

    def unfreeze(self) -> None:
        self._frozen = None
        self.freeze_button.setChecked(False)
        self.freeze_button.setText("Freeze frame")
        self._dirty = True

    def toggle_freeze(self) -> None:
        if self.frozen:
            self.unfreeze()
        else:
            self.freeze()
        self.refresh()

    def start(self, mode: str) -> bool:
        if not self.frozen and not self.freeze():
            return False
        ok, message = self.workflow.begin(mode, (self._frozen.shape[1], self._frozen.shape[0]))
        self._checking_saved = False
        self.check_button.setChecked(False)
        self._say(message, "ok" if ok else "bad")
        self._dirty = True
        self.refresh()
        return ok

    def resume(self) -> bool:
        if not self.frozen and not self.freeze():
            return False
        ok, message = self.workflow.resume_draft((self._frozen.shape[1], self._frozen.shape[0]))
        self._say(message, "ok" if ok else "warn")
        self._dirty = True
        self.refresh()
        return ok

    def _on_frame_clicked(self, x: float, y: float) -> None:
        if self.workflow.session is None:
            self._say("Press New set-up (or Recalibrate camera / Edit arena) first.", "warn")
            return
        if not self.frozen:
            self._say("Freeze the frame first: clicks on the live picture are ignored.", "warn")
            return
        self.place_click(x, y)

    def place_click(self, x: float, y: float) -> str:
        key = self.workflow.click(x, y)
        self._say(self.workflow.message, "ok" if key else "warn")
        self._dirty = True
        self.refresh()
        return key

    def submit_entry(self) -> bool:
        if self.workflow.session is None:
            self._say("Start a set-up first.", "warn")
            return False
        ok, message = self.workflow.type_value(self.entry.text())
        self._say(message, "ok" if ok else "bad")
        if ok:
            self.entry.clear()
        self._dirty = True
        self.refresh()
        return ok

    def _update_entry_hint(self, text: str) -> None:
        if not text.strip():
            self.entry_hint.setText("")
            return
        try:
            value = A.parse_length(text)
            self.entry_hint.setText(f"= {value:.3f} m ({value * 1000:.0f} mm)")
        except ValueError as error:
            self.entry_hint.setText(str(error))

    def _on_row_clicked(self, row: int, _column: int) -> None:
        rows = self.workflow.rows()
        if 0 <= row < len(rows) and self.workflow.goto(rows[row].key):
            self._say(self.workflow.message, "ok")
            self._dirty = True
            self.refresh()

    def _step(self, direction: int) -> None:
        rows = self.workflow.rows()
        if not rows:
            return
        index = next((i for i, row in enumerate(rows) if row.key == self.workflow.session.current), 0 if direction > 0 else len(rows))
        target = max(0, min(len(rows) - 1, index + direction))
        self.workflow.goto(rows[target].key)
        self._say(self.workflow.message, "ok")
        self._dirty = True
        self.refresh()

    def back(self) -> None:
        self._step(-1)

    def next(self) -> None:
        self._step(1)

    def undo(self) -> None:
        if self.workflow.session is not None:
            self.workflow.undo()
            self._say(self.workflow.message, "ok")
            self._dirty = True
            self.refresh()

    def reset(self) -> None:
        if self.workflow.session is not None:
            self.workflow.reset()
            self._say("Reset: every click and typed value of this set-up was cleared (saved files are untouched).", "warn")
            self._dirty = True
            self.refresh()

    def retry(self) -> None:
        """A fresh frame, keeping every click: use when the picture was blurred or someone stood in the way."""
        fresh = self._latest_frame()
        if fresh is None:
            self._say("The camera is not delivering frames, so the frozen picture was kept. Fix the camera (GUIDED SETUP > camera step) and Retry.", "bad")
            self.refresh()
            return
        self.freeze(fresh)
        self._say("New frame taken. Your clicks were kept: check that they still sit on the corners; if the camera moved, Reset.", "warn")
        self.refresh()

    def finish_flipper(self) -> None:
        if self.workflow.session is not None:
            self.workflow.finish_flipper()
            self._say(self.workflow.message, "ok")
            self._dirty = True
            self.refresh()

    def add_vertex(self) -> None:
        if self.workflow.session is not None:
            self.workflow.reopen_flipper()
            self._say(self.workflow.message, "ok")
            self._dirty = True
            self.refresh()

    def remove_vertex(self) -> None:
        row = self.table.currentRow()
        rows = self.workflow.rows()
        if self.workflow.session is None or not 0 <= row < len(rows) or not rows[row].key.startswith("flipper"):
            self._say("Select a flipper vertex row first.", "warn")
            return
        self.workflow.remove_flipper_vertex(int(rows[row].key[7:]))
        self._say(self.workflow.message, "ok")
        self._dirty = True
        self.refresh()

    def save(self, apply_to_hub: bool = True) -> bool:
        corners = self.workflow.session.corner_pixels() if self.workflow.session is not None else None
        reference = self._frozen.copy() if self._frozen is not None else None
        ok, lines = self.workflow.save()
        if ok:
            if apply_to_hub and corners and reference is not None and hasattr(self.hub, "calibration_saved"):
                self.hub.calibration_saved(reference, list(corners.values()))
            self._say("SAVED. " + " ".join(lines), "ok")
            if apply_to_hub and hasattr(self.hub, "reload_calibration"):
                self.hub.reload_calibration()                # the rest of the Command Centre now uses the saved files
            self.workflow.session = None
            self.workflow.solution = A.Solution()
            self.verify_box.setChecked(False)
        else:
            self._say(" ".join(lines), "bad")
        self._dirty = True
        self.refresh()
        return ok

    def reload_saved(self) -> None:
        report = self.workflow.verify_report()
        if hasattr(self.hub, "reload_calibration"):
            self.hub.reload_calibration()
        if report["problem"]:
            self._say("Reloaded, but: " + report["problem"], "bad")
        elif not report["arena_saved"]:
            self._say("Nothing is saved for this profile yet.", "warn")
        else:
            self._say("Reloaded from " + report["arena_file"] + (" and the camera corners." if report["corners_saved"] else "; the camera corners are not saved yet."), "ok")
        self._dirty = True
        self.refresh()

    def toggle_check(self) -> None:
        self._checking_saved = self.check_button.isChecked()
        if self._checking_saved:
            self.workflow.reload()
            if self.workflow.loaded.arena is None:
                self._checking_saved = False
                self.check_button.setChecked(False)
                self._say("Nothing is saved to check yet.", "warn")
        if self._checking_saved:
            self.unfreeze()
            self._say("Showing the SAVED arena over the live picture. Compare it with the floor, then tick the box below.", "ok")
        self._dirty = True
        self.refresh()

    def confirm_overlay(self) -> None:
        if self.verify_box.isChecked():
            if self.workflow.mark_verified():
                self._say("Confirmed by eye. This is remembered until the arena or camera corners are saved again.", "ok")
            else:
                self.verify_box.setChecked(False)
                self._say("Nothing valid is saved to confirm.", "warn")
        else:
            self.workflow.state.clear_arena_verified(self.workflow.profile)
        self.refresh()

    def _say(self, text: str, level: str = "ok") -> None:
        colours = {"ok": "#d9dee5", "warn": "#e8b85b", "bad": "#ef6b73"}
        _color_text(self.status_line, text, colours.get(level, "#d9dee5"))

    # ================================================================ profile changes
    def _on_profile_changed(self, name: str) -> None:
        if self._injected:
            self.workflow.reload()
        elif name != self.workflow.profile:
            self.workflow = ArenaWorkflow(name, getattr(self.hub, "app_state", None))
            self._frozen = None
            self._checking_saved = False
            self.unfreeze()
        else:
            self.workflow.reload()
        self.verify_box.setChecked(self.workflow.is_verified())
        self._dirty = True
        self.refresh()

    # ================================================================ readiness and drawing
    def readiness(self) -> Readiness:
        level, reason = None, ""
        try:
            state = self.hub.vision.map_state()
            level, reason = state.level, state.detail
        except Exception:                                    # a hub without live vision (some tests)
            pass
        base = self.workflow.readiness(level, reason, self._restore_report())
        session = self.workflow.session
        if base.level == READY and session is not None:
            return Readiness(WARNING, "A set-up is in progress", "The saved arena is fine, but this tab is editing it.", "Finish and Save it, or Reset.")
        return base

    def refresh(self) -> None:
        workflow = self.workflow
        self.profile_label.setText(f"ARENA CALIBRATION · {profile_label(workflow.profile)}")
        readiness = self.readiness()
        self.chip.setText(readiness.level.upper())
        self.chip.setStyleSheet(f"background: {CHIP_COLOURS[readiness.level]};")
        self.chip.setToolTip(f"{readiness.headline}\n{readiness.detail}\n{readiness.fix}".strip())
        session, solution = workflow.session, workflow.solution

        for button in (self.recal_button, self.edit_button):
            button.setEnabled(workflow.loaded.arena is not None and not workflow.loaded.problem)
        self.resume_button.setEnabled(workflow.has_draft)
        in_session = session is not None
        for button in (self.back_button, self.next_button, self.undo_button, self.reset_button, self.finish_button,
                       self.add_vertex_button, self.remove_vertex_button):
            button.setEnabled(in_session)
        typed_flow = in_session and session.mode != "recalibrate" and not session.design_flow
        self.finish_button.setEnabled(typed_flow and not session.flipper_closed)
        self.add_vertex_button.setEnabled(typed_flow and session.flipper_closed)
        self.remove_vertex_button.setEnabled(typed_flow)
        for button in (self.finish_button, self.add_vertex_button, self.remove_vertex_button):
            button.setVisible(not (in_session and session.design_flow))
        designed = workflow.design
        self.override_box.setVisible(designed is not None or not in_session)
        self.override_box.setEnabled(designed is not None)
        self.design_label.setText("\n".join(designed.lines()) if designed is not None else
                                  ("Dimensions are typed for this profile (a test patch has no design drawing)." if in_session else ""))
        can_save, why = workflow.can_save()
        self.save_button.setEnabled(can_save)
        self.save_button.setToolTip("" if can_save else why)
        slot = session.current_slot() if session else None
        number_step = slot is not None and slot.kind == A.NUMBER
        self.entry.setEnabled(number_step)
        self.entry_button.setEnabled(number_step)
        if number_step:
            self.entry.setPlaceholderText(f"{slot.label}: type metres or millimetres, Enter accepts the offered default")
        elif in_session and session.design_flow:
            self.entry.setPlaceholderText("nothing to type: the size comes from the design drawing")
        self.entry.setVisible(not (in_session and session.design_flow))
        self.entry_button.setVisible(not (in_session and session.design_flow))
        self.entry_hint.setVisible(not (in_session and session.design_flow))
        report = self._restore_report()
        self.restore_box.setVisible(report is not None or getattr(self.hub, "settings_store", None) is not None)
        if report is not None:
            self.restore_label.setText("\n".join(report.lines))
            self.align_ok_button.setEnabled(report.can_confirm)
            self.align_ok_button.setToolTip(report.confirm_refusal or "")
            self.quick_button.setEnabled(workflow.loaded.arena is not None)
        else:
            self.restore_label.setText("Waiting for a camera picture to compare the saved calibration with.")
            self.align_ok_button.setEnabled(False)
        status = getattr(self.hub, "camera_status_text", None)
        self.camera_label.setText(status() if status is not None else "")
        self.verify_box.setEnabled(workflow.loaded.arena is not None and workflow.loaded.corners_saved and not workflow.loaded.problem)
        if not self.verify_box.isChecked() and workflow.is_verified():
            self.verify_box.setChecked(True)
        self.prompt_label.setText(session.prompt() if session is not None else
                                  ("Saved arena loaded. Use Check saved overlay to compare it with the floor, or start a set-up." if workflow.loaded.arena
                                   else "No arena saved for this profile. Press New set-up."))
        self._fill_table()
        self._fill_results(readiness)
        self._paint()

    def _fill_table(self) -> None:
        rows = self.workflow.rows()
        key = tuple((r.key, r.state, r.value) for r in rows)
        if key == self._shown_key:
            return
        self._shown_key = key
        self.table.setRowCount(len(rows))
        for index, row in enumerate(rows):
            for column, text in enumerate((row.label + ("" if row.required else "  (optional)"), row.state.upper(), row.value)):
                item = self.table.item(index, column)
                if item is None:
                    item = QTableWidgetItem()
                    self.table.setItem(index, column, item)
                item.setText(text)
                if column == 1:
                    from PySide6.QtGui import QColor
                    item.setForeground(QColor(STATE_COLOURS.get(row.state, "#d9dee5")))

    def _fill_results(self, readiness: Readiness) -> None:
        solution, workflow = self.workflow.solution, self.workflow
        _color_text(self.problem_label, ("Still needed: " + "; ".join(solution.problems)) if workflow.session is not None and solution.problems else "", "#ef6b73")
        _color_text(self.warning_label, ("Warnings: " + "; ".join(solution.warnings)) if solution.warnings else "", "#e8b85b")
        lines = list(workflow.hazard_lines()) if solution.setup is not None else (workflow.loaded.arena.summary_lines() if workflow.loaded.arena else [])
        if solution.complete:
            lines.append(f"fit: {solution.level.upper()}" + (f" - {solution.quality.reasons[0]}" if solution.quality.reasons else ""))
            if solution.rim_check_mm:
                lines.append("pit-rim checks: " + ", ".join(f"{v:.0f} mm" for v in solution.rim_check_mm))
        lines.append(f"{readiness.level}: {readiness.headline}" + (f" - {readiness.detail}" if readiness.detail else ""))
        if readiness.fix:
            lines.append("Next: " + readiness.fix)
        self.result_label.setText("\n".join(lines))
        accuracy = workflow.accuracy_lines() if solution.mapping.valid else []
        self.accuracy_label.setVisible(bool(accuracy))
        _color_text(self.accuracy_label, "Calibration accuracy:\n" + "\n".join(accuracy) if accuracy else "",
                    "#ef6b73" if solution.unreliable else "#9eabb8")
        loaded = workflow.loaded
        placeholders = loaded.config.placeholders
        files = [f"arena file: {loaded.arena_path}", f"camera corners file: {loaded.clicks_path}",
                 "placeholders still in use: " + ("; ".join(placeholders) if placeholders else "none")]
        if workflow.last_save is not None:
            files = workflow.last_save.lines() + files
        self.files_label.setText("\n".join(files))

    def _paint(self) -> None:
        frame = self._frozen if self.frozen else self._latest_frame()
        if frame is None:
            self.view.set_frame(None)
            self.view.set_placeholder("No camera frame. Open the camera step in GUIDED SETUP.")
            return
        key = (id(frame), self.workflow.message, str(self.workflow.solution.level), self._checking_saved, self._shown_key)
        if self.frozen and key == getattr(self, "_painted_key", None) and not self._dirty:
            return
        self._painted_key, self._dirty = key, False
        self.workflow.set_frame_size((frame.shape[1], frame.shape[0])) if self.workflow.session is not None else None
        if self._checking_saved:
            mapping = self.hub.vision.mapping
            self.view.set_frame(render_saved_view(frame, self.workflow.loaded, mapping))
        elif self.workflow.session is not None:
            self.view.set_frame(render_setup_view(frame, self.workflow))
        else:
            self.view.set_frame(frame)
        self.view.badge = "FROZEN" if self.frozen else ("MOCK" if getattr(self.hub.options, "mock", False) else "")

    # ================================================================ lifecycle
    def showEvent(self, event) -> None:  # noqa: N802 - Qt API
        self._timer.start()
        self.workflow.reload()
        super().showEvent(event)

    def hideEvent(self, event) -> None:  # noqa: N802 - Qt API
        self._timer.stop()
        super().hideEvent(event)

    def shutdown(self) -> None:
        self._timer.stop()
