"""ELRS & CONTROL: who is driving, how the ELRS link and the controller are doing, and the safety checks.

Everything on this screen is a display, except one thing: the operator can open a supervised console
running ``scripts/motion_check.py`` (the wheels-raised safety checks, or a floor run that motion_check
itself refuses without a passing safety record). That is only ever started by a button press, after the
operator has selected the port and the power mode and ticked every confirmation, and never as a
consequence of anything else finishing.

This screen is listen-only: it has no call that writes to the controller, never opens a port or a camera,
and does not import ``serial``. The RadioMaster works without this computer.
"""

from __future__ import annotations

import subprocess
import time
from pathlib import Path
from typing import Any, Callable

from PySide6.QtCore import QObject, QRunnable, Qt, QThreadPool, QTimer, Signal
from PySide6.QtGui import QBrush, QColor
from PySide6.QtWidgets import (QAbstractItemView, QCheckBox, QComboBox, QFrame, QHBoxLayout, QHeaderView, QLabel,
                               QLineEdit, QPushButton, QScrollArea, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget)

from control_center.backend import control_status as cs
from control_center.backend.readiness import CHIP_COLOURS, Readiness

REFRESH_MS = 250                      # 4 Hz
LEVEL_COLOURS = {"ok": "#55c889", "warn": "#e8b85b", "bad": "#ef6b73", "off": "#9aa5b1", "plain": "#d9dee5"}
TITLE = "ELRS & CONTROL"

STYLE = """
QLabel { background: transparent; }
QFrame#panel { background: #171e27; border: 1px solid #2b3542; border-radius: 5px; }
QFrame#panelDisabled { background: #141a22; border: 1px dashed #3b4a5d; border-radius: 5px; }
QLabel#panelTitle { color: #9eabb8; font-size: 10pt; font-weight: 700; letter-spacing: 1px; }
QLabel#hint { color: #9eabb8; }
QLabel#mono { font-family: Consolas; }
QTableWidget { background: #0d1218; alternate-background-color: #121922; gridline-color: #2b3542; color: #d9dee5;
               border: 1px solid #2b3542; }
QHeaderView::section { background: #171e27; color: #9eabb8; border: none; border-bottom: 1px solid #2b3542;
                       padding: 4px 6px; font-weight: 700; }
QLineEdit, QComboBox { background: #0d1218; border: 1px solid #3b4a5d; border-radius: 4px; padding: 5px 6px; }
QScrollArea { border: none; background: transparent; }
QWidget#scrollBody { background: transparent; }
QPushButton:disabled { color: #6b7684; }
"""


def _label(text: str = "", name: str | None = None, wrap: bool = True) -> QLabel:
    label = QLabel(text)
    label.setTextFormat(Qt.TextFormat.PlainText)
    label.setWordWrap(wrap)
    label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
    if name:
        label.setObjectName(name)
    return label


def _colour(label: QLabel, level: str, extra: str = "") -> None:
    label.setStyleSheet(f"color: {LEVEL_COLOURS.get(level, LEVEL_COLOURS['plain'])}; {extra}")


def _panel(title: str, disabled: bool = False) -> tuple[QFrame, QVBoxLayout]:
    frame = QFrame()
    frame.setObjectName("panelDisabled" if disabled else "panel")
    layout = QVBoxLayout(frame)
    layout.setContentsMargins(12, 10, 12, 12)
    layout.setSpacing(8)
    heading = QLabel(title)
    heading.setObjectName("panelTitle")
    layout.addWidget(heading)
    return frame, layout


class _Table(QTableWidget):
    """A read-only table that sizes itself to its rows (the page scrolls, the tables do not)."""

    def __init__(self, headers: list[str], stretch_column: int) -> None:
        super().__init__(0, len(headers))
        self.setHorizontalHeaderLabels(headers)
        self.verticalHeader().setVisible(False)
        self.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.setAlternatingRowColors(True)
        self.setWordWrap(True)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        header = self.horizontalHeader()
        for column in range(len(headers)):
            header.setSectionResizeMode(column, QHeaderView.ResizeMode.Stretch if column == stretch_column
                                        else QHeaderView.ResizeMode.ResizeToContents)
        self._signature: tuple = ()

    def fill(self, rows: list[tuple[str, ...]], colours: list[str | None] | None = None, column: int = 1,
             tooltips: list[str] | None = None) -> None:
        resize = len(rows) != self.rowCount()
        self.setRowCount(len(rows))
        for r, values in enumerate(rows):
            for c, text in enumerate(values):
                cell = self.item(r, c)
                if cell is None:
                    cell = QTableWidgetItem()
                    self.setItem(r, c, cell)
                if cell.text() != text:
                    cell.setText(text)
                    resize = resize or c == column
                if tooltips is not None and cell.toolTip() != tooltips[r]:
                    cell.setToolTip(tooltips[r])
            if colours is not None and colours[r]:
                self.item(r, column).setForeground(QBrush(QColor(LEVEL_COLOURS[colours[r]])))
        if resize:
            self.resizeRowsToContents()
            self.setFixedHeight(self.horizontalHeader().height() + sum(self.rowHeight(i) for i in range(self.rowCount())) + 6)

    def show_items(self, items: list[cs.Item]) -> None:
        self.fill([(i.label, i.text, i.source_line()) for i in items], [i.level for i in items], 1, [i.detail for i in items])


class _ReferenceSignals(QObject):
    finished = Signal(object, object, str)       # plan, reference, error text ("" when it worked)


class _ReferenceJob(QRunnable):
    """build_plan + sim_reference on the simulator: pure computation, no hardware."""

    def __init__(self, runner: Callable[[], tuple[Any, dict[str, Any]]], signals: _ReferenceSignals) -> None:
        super().__init__()
        self._runner, self._signals = runner, signals

    def run(self) -> None:
        try:
            plan, reference = self._runner()
            result = (plan, reference, "")
        except BaseException as error:           # the worker must report, never die silently
            result = (None, None, f"{type(error).__name__}: {error}")
        try:
            self._signals.finished.emit(*result)
        except RuntimeError:                     # the screen was closed while this ran
            pass


class ControlScreen(QWidget):
    """``record_path`` (the safety record), ``reference_runner`` and the two clocks are injectable for tests."""

    def __init__(self, hub, parent=None, *, record_path: Path | str | None = None,
                 reference_runner: Callable[[], tuple[Any, dict[str, Any]]] | None = None,
                 clock: Callable[[], float] = time.monotonic, unix_clock: Callable[[], float] = time.time) -> None:
        super().__init__(parent)
        self.hub = hub
        self.record_path = Path(record_path) if record_path is not None else None
        self._reference_runner = reference_runner or cs.compute_plan_and_reference
        self._clock, self._unix = clock, unix_clock
        self.tracker = cs.ControlTracker()
        self.status: cs.ControlStatus | None = None
        self._closed = False
        self._proc: Any = None
        self._proc_kind = ""
        self._proc_command: list[str] = []
        self._exit_note = ""
        self._reference_started = False
        self._reference_running = False
        self._pool = QThreadPool(self)
        self._pool.setMaxThreadCount(1)
        self._signals = _ReferenceSignals(self)
        self._signals.finished.connect(self._on_reference)
        self.setStyleSheet(STYLE)
        self._build()
        self.refresh()
        self._timer = QTimer(self)
        self._timer.timeout.connect(self.refresh)
        self._timer.start(REFRESH_MS)

    # ================================================================ layout

    def _build(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        body = QWidget()
        body.setObjectName("scrollBody")
        root = QVBoxLayout(body)
        root.setContentsMargins(10, 8, 10, 10)
        root.setSpacing(10)
        area = QScrollArea()
        area.setWidgetResizable(True)
        area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        area.setWidget(body)
        outer.addWidget(area)

        head = QHBoxLayout()
        title = QLabel(TITLE)
        title.setStyleSheet("font-size: 15pt; font-weight: 800;")
        head.addWidget(title)
        head.addStretch(1)
        self.readiness_chip = QLabel("")
        head.addWidget(self.readiness_chip)
        root.addLayout(head)
        self.mock_badge = _label("MOCK SESSION: the controller lines are a replay and the camera is synthetic. Nothing on this "
                                 "screen is from the robot.", "hint")
        self.mock_badge.setStyleSheet("color: #e8b85b; font-weight: 700;")
        self.mock_badge.setVisible(bool(getattr(getattr(self.hub, "options", None), "mock", False)))
        root.addWidget(self.mock_badge)
        self.readiness_text = _label("", "hint")
        root.addWidget(self.readiness_text)

        # ---- operating mode banner + handoff
        panel, box = _panel("OPERATING MODE  ·  what the controller itself reports")
        self.banner = QLabel("UNKNOWN")
        self.banner.setAlignment(Qt.AlignmentFlag.AlignCenter)
        box.addWidget(self.banner)
        self.banner_detail = _label("", "hint")
        box.addWidget(self.banner_detail)
        chips = QHBoxLayout()
        self.mode_chips: list[QLabel] = []
        for _ in range(3):
            chip = _label("")
            chip.setMargin(4)
            chips.addWidget(chip, 1)
            self.mode_chips.append(chip)
        box.addLayout(chips)
        self.controller_line = _label("", "mono")
        box.addWidget(self.controller_line)
        self.handoff_headline = _label("")
        self.handoff_headline.setStyleSheet("font-weight: 700;")
        box.addWidget(self.handoff_headline)
        self.handoff_transition = _label("", "hint")
        box.addWidget(self.handoff_transition)
        for rule in cs.HANDOFF_RULES:
            box.addWidget(_label("•  " + rule, "hint"))
        root.addWidget(panel)

        # ---- diagnostics
        panel, box = _panel("ELRS LINK AND RECEIVER  ·  only what the firmware reports")
        self.elrs_table = _Table(["Item", "Value", "Source · age"], 1)
        box.addWidget(self.elrs_table)
        root.addWidget(panel)
        panel, box = _panel("FIRMWARE AND RESTARTS")
        self.firmware_table = _Table(["Item", "Value", "Source · age"], 1)
        box.addWidget(self.firmware_table)
        root.addWidget(panel)
        panel, box = _panel("POWER")
        self.power_table = _Table(["Item", "Value", "Source · age"], 1)
        box.addWidget(self.power_table)
        root.addWidget(panel)
        panel, box = _panel("COMMUNICATION")
        self.comms_table = _Table(["Item", "Value", "Source · age"], 1)
        box.addWidget(self.comms_table)
        root.addWidget(panel)

        # ---- transports
        panel, box = _panel("TRANSPORT INVENTORY  ·  every route, with its evidence (no state is upgraded here)")
        self.transport_table = _Table(["Route", "Path", "State", "Evidence", "Detail", "Seen now"], 4)
        box.addWidget(self.transport_table)
        root.addWidget(panel)

        # ---- manual
        panel, box = _panel("MANUAL  ·  RADIOMASTER OVER ELRS")
        self.manual_text = _label(cs.MANUAL_TEXT)
        self.manual_text.setStyleSheet("font-weight: 700;")
        box.addWidget(self.manual_text)
        self.manual_return = _label(cs.RETURN_TO_MANUAL, "hint")
        box.addWidget(self.manual_return)
        root.addWidget(panel)

        # ---- autonomous
        panel, box = _panel("AUTONOMOUS MODE  ·  DISABLED", disabled=True)
        self.autonomous_state = QLabel("DISABLED: not enableable in this build")
        self.autonomous_state.setStyleSheet("font-size: 13pt; font-weight: 800; color: #9aa5b1;")
        box.addWidget(self.autonomous_state)
        box.addWidget(_label("The trained policy runs on the laptop and produces the robot's commands. It stays DISABLED until each "
                             "prerequisite below has been verified on hardware. This build has no control that enables it, even if "
                             "every prerequisite were met.", "hint"))
        self.autonomous_button = QPushButton("Enable AUTONOMOUS (not available in this build)")
        self.autonomous_button.setEnabled(False)
        self.autonomous_button.setToolTip("Disabled: " + "; ".join(["no verified command path", "observation pipeline not verified"]))
        box.addWidget(self.autonomous_button)
        self.prereq_table = _Table(["Prerequisite", "State", "Reason"], 2)
        box.addWidget(self.prereq_table)
        box.addWidget(_label("AI SHADOW is a separate, non-actuating display: the policy's proposals are shown and never sent to "
                             "the robot. It does not need the gate above and does not depend on the motors.", "hint"))
        self.shadow_state = _label("", "hint")
        box.addWidget(self.shadow_state)
        root.addWidget(panel)

        # ---- motion check
        panel, box = _panel("MOTION CHECK  ·  simulator reference, safety record, supervised console")
        box.addWidget(_label("The simulator reference below is SIM: a prediction in the simulator, not a hardware result. It is "
                             "computed in this program in the background; nothing is sent anywhere.", "hint"))
        row = QHBoxLayout()
        self.reference_button = QPushButton("Compute plan and simulator reference")
        self.reference_button.clicked.connect(self.compute_reference)
        row.addWidget(self.reference_button)
        self.reference_status = _label("Not computed yet.", "hint")
        row.addWidget(self.reference_status, 1)
        box.addLayout(row)
        self.plan_table = _Table(["#", "Tactic", "Seconds", "Throttle", "Steer", "Firmware L / R", "Sim forward m",
                                  "Sim heading deg", "Sim reads as"], 1)
        box.addWidget(self.plan_table)
        box.addWidget(_label("Wheels-raised safety checks, from artifacts/motion-check/safety-checks.json for the connected firmware:", "hint"))
        self.safety_table = _Table(["Check", "State", "Evidence", "Age"], 2)
        box.addWidget(self.safety_table)
        self.safety_note = _label("", "hint")
        box.addWidget(self.safety_note)
        box.addWidget(_label("Floor-run rule: `motion_check.py run --surface floor` refuses to start unless all five checks passed on "
                             "this firmware within the last 12 hours. There is no override. This window applies the same rule "
                             "first and never launches a floor run that would be refused.", "hint"))

        form = QHBoxLayout()
        form.addWidget(QLabel("COM port"))
        self.port_edit = QLineEdit()
        self.port_edit.setPlaceholderText("e.g. COM5")
        self.port_edit.setMaximumWidth(120)
        port = getattr(getattr(self.hub, "options", None), "controller_port", None)
        if port:
            self.port_edit.setText(str(port))
        form.addWidget(self.port_edit)
        form.addWidget(QLabel("Power"))
        self.power_combo = QComboBox()
        self.power_combo.addItem("select how the robot is powered", None)
        for choice in cs._power_choices():
            self.power_combo.addItem(choice, choice)
        form.addWidget(self.power_combo)
        form.addStretch(1)
        box.addLayout(form)
        self.check_boxes = {key: QCheckBox(text) for key, text in cs.CHECKS_CONFIRMATIONS.items()}
        self.floor_boxes = {key: QCheckBox(text) for key, text in cs.FLOOR_CONFIRMATIONS.items()}
        box.addWidget(_label("Confirmations for the safety checks:", "hint"))
        for widget in self.check_boxes.values():
            box.addWidget(widget)
        self.checks_button = QPushButton("Open supervised safety checks in a console")
        self.checks_button.clicked.connect(lambda: self.launch("checks"))
        box.addWidget(self.checks_button)
        self.checks_reasons = _label("", "hint")
        box.addWidget(self.checks_reasons)
        box.addWidget(_label("Confirmations for a floor run:", "hint"))
        for widget in self.floor_boxes.values():
            box.addWidget(widget)
        self.floor_button = QPushButton("Open supervised FLOOR run in a console")
        self.floor_button.clicked.connect(lambda: self.launch("floor"))
        box.addWidget(self.floor_button)
        self.floor_reasons = _label("", "hint")
        box.addWidget(self.floor_reasons)
        for widget in (*self.check_boxes.values(), *self.floor_boxes.values()):
            widget.toggled.connect(self._update_launch_controls)
        self.port_edit.textChanged.connect(self._update_launch_controls)
        self.power_combo.currentIndexChanged.connect(self._update_launch_controls)

        stop_row = QHBoxLayout()
        self.stop_button = QPushButton("Stop the console")
        self.stop_button.clicked.connect(self.stop_process)
        stop_row.addWidget(self.stop_button)
        self.process_label = _label("No console started from this window.", "hint")
        stop_row.addWidget(self.process_label, 1)
        box.addLayout(stop_row)
        box.addWidget(_label("What the console does: the console asks its own questions and waits for you to type GO; it cannot arm "
                             "the robot or select a mode (those are RadioMaster switches). Its limits, the deadman behaviour and the "
                             "neutral-on-stop rules are those of scripts/motion_check.py and the firmware. Stop closes the console "
                             "abruptly: the controller then stops by itself about 250 ms after the last command. Physical emergency "
                             "stop: ARM switch off on the RadioMaster, or the power cut-off.", "hint"))
        root.addWidget(panel)
        root.addStretch(1)

    # ================================================================ status

    def _inputs(self) -> cs.ControlInputs:
        return cs.inputs_from_hub(self.hub)

    def _compute(self) -> cs.ControlStatus:
        return cs.build_control_status(self._inputs(), self.tracker, self._clock(), self._unix(), self.record_path)

    def readiness(self) -> Readiness:
        self.status = self._compute()
        return self.status.readiness

    def refresh(self) -> None:
        if self._closed:
            return
        self._poll_process()
        status = self.status = self._compute()
        view = status.authority
        self.banner.setText(view.banner.text)
        self.banner.setStyleSheet(f"font-size: 30pt; font-weight: 800; color: {LEVEL_COLOURS[view.banner.level]};")
        self.banner_detail.setText(view.banner.detail + (f"  ({view.banner.source}, {cs.age_text(view.banner.age_s)})"
                                                         if view.banner.age_s is not None else ""))
        for chip, item in zip(self.mode_chips, view.modes):
            chip.setText(f"{item.label}\n{item.text}")
            _colour(chip, item.level, "font-weight: 700;")
            chip.setToolTip(item.detail)
        self.controller_line.setText(f"{view.controller_line.label}: {view.controller_line.text}")
        _colour(self.controller_line, view.controller_line.level)
        self.handoff_headline.setText(status.handoff.headline)
        _colour(self.handoff_headline, status.handoff.level, "font-weight: 700;")
        self.handoff_transition.setText(status.handoff.transition)
        self.handoff_transition.setVisible(bool(status.handoff.transition))

        chip = status.readiness
        self.readiness_chip.setText(chip.level.upper())
        self.readiness_chip.setStyleSheet(f"color: white; background: {CHIP_COLOURS[chip.level]}; border-radius: 4px; "
                                          "padding: 3px 10px; font-weight: 800;")
        self.readiness_text.setText(chip.headline + (f" · {chip.detail}" if chip.detail else "") + (f" · Next: {chip.fix}" if chip.fix else ""))

        self.elrs_table.show_items(status.elrs)
        self.firmware_table.show_items(status.firmware)
        self.power_table.show_items([status.power])
        self.comms_table.show_items(status.communication)
        self.transport_table.fill([(t.route, t.path, t.state, t.evidence, t.detail, t.seen_now) for t in status.transports], None, 4)

        auto = status.autonomous
        self.autonomous_state.setText("DISABLED: not enableable in this build" + ("" if not auto.allowed else
                                                                                   " (prerequisites met, but this build offers no way to enable it)"))
        self.autonomous_button.setEnabled(auto.enable_offered)       # always False in this build
        self.prereq_table.fill([(p.label, "MET" if p.met else "UNMET", p.reason) for p in auto.prerequisites],
                               ["ok" if p.met else "bad" for p in auto.prerequisites], 1)
        shadow = view.modes[2]
        self.shadow_state.setText(f"{shadow.label}: {shadow.text}")

        rows, note = cs.safety_rows(self.hub.monitor.firmware() if hasattr(self.hub, "monitor") else None, self._unix(), self.record_path)
        self.safety_table.fill([(r.name, r.state, r.evidence, r.age_text) for r in rows],
                               ["ok" if r.state == "PASS" else "bad" if r.state == "FAIL" else "off" for r in rows], 1)
        self.safety_note.setText(note)
        self._update_launch_controls()

    # ================================================================ simulator reference (worker thread)

    def showEvent(self, event) -> None:                 # computed once, when the screen is first seen
        super().showEvent(event)
        if not self._reference_started:
            self.compute_reference()

    def compute_reference(self) -> None:
        if self._closed or self._reference_running:
            return
        self._reference_started = self._reference_running = True
        self.reference_button.setEnabled(False)
        self.reference_status.setText("Computing the plan and the simulator reference in the background (SIM, no hardware)...")
        self._pool.start(_ReferenceJob(self._reference_runner, self._signals))

    def _on_reference(self, plan: Any, reference: Any, error: str) -> None:
        self._reference_running = False
        if self._closed:
            return
        self.reference_button.setEnabled(True)
        if error or plan is None:
            self.reference_status.setText(f"Could not compute the plan or the simulator reference: {error}")
            return
        self.plan_table.fill(cs.motion_plan_rows(plan, reference))
        self.reference_status.setText("SIM, not a hardware result. " + str(reference.get("note", "")))

    # ================================================================ supervised console

    def confirmations(self, kind: str) -> dict[str, bool]:
        boxes = self.floor_boxes if kind == "floor" else self.check_boxes
        return {key: box.isChecked() for key, box in boxes.items()}

    def launch_blockers(self, kind: str) -> list[str]:
        inputs = self._inputs()
        return cs.launch_blockers(kind, port=self.port_edit.text(), power=self.power_combo.currentData(),
                                  confirmations=self.confirmations(kind), link_label=inputs.link_label, link_error=inputs.link_error,
                                  mock=inputs.mock, process_running=self._proc is not None,
                                  firmware=inputs.monitor.firmware(), now_unix=self._unix(), record_path=self.record_path)

    def _update_launch_controls(self) -> None:
        for kind, button, reasons in (("checks", self.checks_button, self.checks_reasons), ("floor", self.floor_button, self.floor_reasons)):
            blockers = self.launch_blockers(kind)
            button.setEnabled(not blockers)
            reasons.setText("Disabled because: " + "; ".join(blockers) if blockers else
                            "Ready to open a console. Nothing starts until you press the button.")
            button.setToolTip("; ".join(blockers))
        self.stop_button.setEnabled(self._proc is not None)

    def launch(self, kind: str) -> bool:
        """Open the console. Only ever called from a button; refuses again here if anything is missing."""
        if kind not in ("checks", "floor"):
            return False
        blockers = self.launch_blockers(kind)
        if blockers:
            self.process_label.setText("Not started: " + blockers[0])
            return False
        port, power = self.port_edit.text().strip(), self.power_combo.currentData()
        command = cs.checks_command(port, power) if kind == "checks" else cs.floor_command(port, power)
        flags = getattr(subprocess, "CREATE_NEW_CONSOLE", 0)
        try:
            process = subprocess.Popen(command, cwd=str(cs.ROOT), creationflags=flags)
        except (OSError, ValueError) as error:
            self.process_label.setText(f"Could not start the console: {error}")
            return False
        self._proc, self._proc_kind, self._proc_command, self._exit_note = process, kind, command, ""
        for box in (*self.check_boxes.values(), *self.floor_boxes.values()):
            box.setChecked(False)                      # each launch needs its own confirmations
        self._poll_process()
        self._update_launch_controls()
        return True

    def process_state(self) -> str:
        if self._proc is not None:
            return f"RUNNING ({self._proc_kind}) pid {getattr(self._proc, 'pid', '?')}: " + " ".join(self._proc_command[1:])
        return self._exit_note or "No console started from this window."

    def _poll_process(self) -> None:
        if self._proc is not None:
            code = self._proc.poll()
            if code is not None:
                self._exit_note = f"EXITED ({self._proc_kind}) with code {code}. Read the console's own result; nothing here interprets it."
                self._proc = None
        self.process_label.setText(self.process_state())

    def stop_process(self) -> None:
        process = self._proc
        if process is None:
            return
        self._terminate(process)
        self._exit_note = f"STOPPED ({self._proc_kind}) from this window. The controller stops by itself about 250 ms after the last command."
        self._proc = None
        self.process_label.setText(self._exit_note)
        self._update_launch_controls()

    @staticmethod
    def _terminate(process: Any) -> None:
        try:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=1.0)
                except Exception:
                    kill = getattr(process, "kill", None)
                    if kill is not None and process.poll() is None:
                        kill()
        except Exception:
            pass

    # ================================================================ lifecycle

    def shutdown(self) -> None:
        """Stop the timer, finish the worker and terminate any console started here. Safe to call twice."""
        if self._closed:
            return
        self._closed = True
        self._timer.stop()
        process, self._proc = self._proc, None
        if process is not None:
            self._terminate(process)
        self._pool.waitForDone(3000)

    def closeEvent(self, event) -> None:
        self.shutdown()
        super().closeEvent(event)
