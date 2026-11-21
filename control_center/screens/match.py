"""MATCH: the competition-day round screen. Mode, Start / Stop Round, what must be true first, what the AI and the robot are doing.

This screen is a thin shell over three finished backends and decides nothing about safety itself:

* ``match_controller.MatchController``   the single round state machine (prerequisites, downgrade to AI SHADOW, faults, timer).
* ``link_service.LinkService``            the ONE Wi-Fi link / ModeManager worker. Requests go through ``match_support.ServiceModes``;
                                          the ModeManager is never touched from this thread.
* ``ai_model.AiModelScreen``              the loaded model, the observation, the single inference loop, the mapping self-check.

Rules kept here, all of them tested:
  * The application is in MANUAL and non-transmitting. Selecting a mode (MANUAL / AI SHADOW / AUTONOMOUS) only chooses what Start
    Round will ask for. A round starts ONLY from the Start Round button (``operator_action=True`` is passed from that handler and
    nowhere else): not when the camera connects, not when an opponent is selected, not after a restart.
  * AUTONOMOUS is available only when the ModeManager's gate is open (it needs hardware acceptance that does not exist yet): the
    closed gate items are shown with their reasons. With AUTONOMOUS requested and anything missing, Start Round starts AI SHADOW
    instead and says so. Opening the gate also needs the typed phrase and three ticks.
  * A fault during AUTONOMOUS drops to AI SHADOW at once and needs a new deliberate confirmation.
  * The opponent selection is cleared at the end of every round. The inference loop runs only during a round and is released at its end.
  * A software stop cannot work when Wi-Fi is down; the screen says so next to the stop buttons.
  * Everything logged says what was planned and whether a datagram was sent; nothing says a command was physically carried out.

Evidence: results from a mock, replay or fake link are labelled SIMULATED. Nothing on this screen is hardware verification.
"""

from __future__ import annotations

import dataclasses
import logging
import time
from pathlib import Path
from typing import Any, Callable

from PySide6.QtCore import Qt, QTimer, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (QButtonGroup, QCheckBox, QDialog, QDialogButtonBox, QHBoxLayout, QLabel, QLineEdit, QListWidget, QPlainTextEdit,
                               QPushButton, QScrollArea, QVBoxLayout, QWidget)

from control_center.backend import match_support as ms
from control_center.backend import tracking_model as tm
from control_center.backend.control_modes import CONFIRM_PHRASE, CONFIRM_TTL_S, Confirmation, Mode
from control_center.backend.match_controller import MatchController, MatchInputs, StartResult, missing, prerequisites
from control_center.backend.profiles import profile_label
from control_center.backend.readiness import CHIP_COLOURS, NOT_CONFIGURED, Readiness
from control_center.backend.session_log import describe
from control_center.widgets.authority_strip import AuthorityStrip
from control_center.widgets.panel_kit import STYLE, KeyValueGrid, fill_table, label, make_table, panel, set_text, table_texts

log = logging.getLogger(__name__)
TITLE = "MATCH"
TICK_MS = 50                          # 20 Hz: the controller tick, the inference loop and the logging
SLOW_EVERY = 6                        # heavy widgets (tables, board) repaint every 6th tick = about 3 Hz
LOG_EVERY = 20                        # the log panel re-reads the session file about once a second
DECISION_LOG_PERIOD_S = 0.5
INFERENCE_STALL_S = 1.0
ERROR_REPEAT_S = 2.0
TRANSMIT_WINDOW_S = 1.5
CONFIRMATION_REUSE_S = CONFIRM_TTL_S - 5.0
MODES = (Mode.MANUAL, Mode.AI_SHADOW, Mode.AUTONOMOUS)
MODE_TEXT = {Mode.MANUAL: "MANUAL", Mode.AI_SHADOW: "AI SHADOW", Mode.AUTONOMOUS: "AUTONOMOUS"}
MODE_MEANING = {Mode.MANUAL: "the RadioMaster drives; this program transmits nothing",
                Mode.AI_SHADOW: "the AI proposes, it is shown and logged; every transmitted value is 0 / 0",
                Mode.AUTONOMOUS: "the AI drives; only with the gate open and a deliberate confirmation"}
START_NOTE = ("Start Round is a deliberate button press. Selecting a mode, connecting the camera or selecting an opponent never starts "
              "a round. The opponent selection is cleared at the end of every round.")


class AutonomousConfirmationDialog(QDialog):
    """The typed phrase and three ticks (control_modes.Confirmation). Shown only when the ModeManager's gate is open."""

    def __init__(self, parent: QWidget | None, now: Callable[[], float]) -> None:
        super().__init__(parent)
        self._now = now
        self.setWindowTitle("Confirm AUTONOMOUS")
        layout = QVBoxLayout(self)
        layout.addWidget(label("AUTONOMOUS lets the AI drive the robot. The confirmation is valid for 30 s and works once. Type the phrase "
                               f"exactly and tick all three boxes: {CONFIRM_PHRASE}", selectable=False))
        self.phrase = QLineEdit()
        self.phrase.setPlaceholderText(CONFIRM_PHRASE)
        layout.addWidget(self.phrase)
        self.ticks = [QCheckBox(text) for text in ("Wheels raised, or the floor is clear all round",
                                                   "The RadioMaster is in my hand with the ARM switch reachable",
                                                   "I understand the AI stops when disarmed, and the robot stops by itself if Wi-Fi fails")]
        for box in self.ticks:
            layout.addWidget(box)
            box.toggled.connect(self._check)
        self.buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)
        self.phrase.textChanged.connect(self._check)
        self._check()

    def _check(self) -> None:
        ok = self.phrase.text().strip() == CONFIRM_PHRASE and all(box.isChecked() for box in self.ticks)
        self.buttons.button(QDialogButtonBox.StandardButton.Ok).setEnabled(ok)

    def confirmation(self) -> Confirmation:
        return Confirmation(self.phrase.text(), *(box.isChecked() for box in self.ticks), created_at=self._now())


class MatchScreen(QWidget):
    def __init__(self, hub: Any, parent: QWidget | None = None, *, tracking: Any = None, ai: Any = None, service: Any = None,
                 clock: Callable[[], float] = time.monotonic, drive: Callable[[], None] | None = None,
                 confirm: Callable[[], Confirmation | None] | None = None, wait_s: float = 1.0, start_timer: bool = True) -> None:
        super().__init__(parent)
        self.hub = hub
        self.tracking = tracking
        self.ai = ai if (ai is not None and hasattr(ai, "model_loaded")) else None
        self.service = service if service is not None else getattr(hub, "link_service", None)
        self._clock = clock
        self._confirm_hook = confirm
        self.setWindowTitle(TITLE)
        self.setStyleSheet(STYLE)
        self._closed = False
        self.modes = ms.ServiceModes(self.service, wait_s=wait_s, drive=drive)
        self.controller = MatchController(self.modes, clock=clock, session_log=getattr(hub, "session_log", None),
                                          on_round_end=self._round_ended, on_round_start=self._round_started)
        self._selected = Mode.MANUAL
        self._confirmation: Confirmation | None = None
        self._action_text = ""
        self._ticks = 0
        self._inputs = MatchInputs()
        self._last_error_at = float("-inf")
        self._last_error = ""
        self._last_plan_t = float("-inf")
        self._last_mode_seen: Mode | None = None
        self._last_decision_key: tuple | None = None
        self._last_decision_logged = float("-inf")
        self._last_worker_error = ""
        self._open_folder_calls: list[Path] = []
        self.start_attempts = 0

        self._build()
        self._sync_mode_buttons()
        self._timer = QTimer(self)
        self._timer.setInterval(TICK_MS)
        self._timer.timeout.connect(self._on_timer)
        if start_timer:
            self._timer.start()
        self.paint_all()

    # ================================================================ construction

    def _build(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(10, 10, 10, 10)
        header = QHBoxLayout()
        title = label(TITLE, "screenTitle", selectable=False)
        title.setWordWrap(False)
        header.addWidget(title)
        header.addStretch(1)
        self.profile_label = label("", "hint")
        header.addWidget(self.profile_label)
        self.readiness_chip = QLabel(NOT_CONFIGURED)
        self.readiness_chip.setObjectName("chip")
        header.addWidget(self.readiness_chip)
        outer.addLayout(header)
        self.evidence_label = label("", "banner")
        outer.addWidget(self.evidence_label)

        body = QWidget()
        body.setObjectName("scrollBody")
        columns = QHBoxLayout(body)
        columns.setContentsMargins(0, 0, 6, 0)
        left, right = QVBoxLayout(), QVBoxLayout()
        left.setSpacing(8)
        right.setSpacing(8)
        columns.addLayout(left, 3)
        columns.addLayout(right, 2)

        # ---- mode and round
        frame, box = panel("MODE (selecting a mode does not start a round)")
        row = QHBoxLayout()
        self.mode_buttons: dict[Mode, QPushButton] = {}
        self.mode_group = QButtonGroup(self)
        self.mode_group.setExclusive(True)
        for mode in MODES:
            button = QPushButton(MODE_TEXT[mode])
            button.setObjectName("modeButton")
            button.setCheckable(True)
            button.clicked.connect(lambda _checked=False, m=mode: self.select_mode(m))
            self.mode_group.addButton(button)
            self.mode_buttons[mode] = button
            row.addWidget(button, 1)
        box.addLayout(row)
        self.mode_note = label("", "hint")
        box.addWidget(self.mode_note)
        self.gate_label = label("", "headline")
        box.addWidget(self.gate_label)
        self.gate_reasons = label("", "hint")
        box.addWidget(self.gate_reasons)
        left.addWidget(frame)

        frame, box = panel("ROUND")
        row = QHBoxLayout()
        self.start_button = QPushButton("Start Round")
        self.start_button.setObjectName("startButton")
        self.start_button.clicked.connect(self._on_start_clicked)
        self.stop_button = QPushButton("Stop Round")
        self.stop_button.setObjectName("stopButton")
        self.stop_button.clicked.connect(self._on_stop_clicked)
        self.emergency_button = QPushButton("EMERGENCY STOP")
        self.emergency_button.setObjectName("emergencyButton")
        self.emergency_button.clicked.connect(self._on_emergency_clicked)
        for button in (self.start_button, self.stop_button, self.emergency_button):
            row.addWidget(button, 1)
        box.addLayout(row)
        self.stop_note = label(ms.SOFTWARE_STOP_WIFI_TEXT, "hint")
        box.addWidget(self.stop_note)
        self.timer_label = label("00:00.0", "timer")
        self.timer_label.setWordWrap(False)
        box.addWidget(self.timer_label)
        self.round_label = label("", "headline")
        box.addWidget(self.round_label)
        self.message_label = label("", "hint")
        box.addWidget(self.message_label)
        self.action_label = label("", "hint")
        box.addWidget(self.action_label)
        box.addWidget(label(START_NOTE, "hint"))
        self.banner = label("", "banner")
        self.banner.setStyleSheet("background: #5b1f25; color: #ffd9dc;")
        self.banner.hide()
        box.addWidget(self.banner)
        left.addWidget(frame)

        frame, box = panel("PREREQUISITES (all must PASS for AUTONOMOUS; AI SHADOW needs the first two)")
        self.prereq_table = make_table(["Prerequisite", "Status", "Needed for", "Detail"])
        self.prereq_table.setMinimumHeight(330)
        box.addWidget(self.prereq_table)
        left.addWidget(frame)

        frame, box = panel("RECENT EVENTS (session log)")
        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMinimumHeight(200)
        box.addWidget(self.log_view)
        self.log_where = label("", "hint")
        box.addWidget(self.log_where)
        left.addWidget(frame)
        left.addStretch(1)

        # ---- status board
        frame, box = panel("STATUS BOARD")
        self.board = KeyValueGrid()
        box.addWidget(self.board)
        right.addWidget(frame)

        frame, box = panel("AUTHORITY")
        self.authority_strip = AuthorityStrip()
        box.addWidget(self.authority_strip)
        right.addWidget(frame)

        frame, box = panel("FAULTS AND SAFETY")
        self.safety_label = label("", "headline")
        box.addWidget(self.safety_label)
        self.faults_label = label("No faults.", "hint")
        box.addWidget(self.faults_label)
        right.addWidget(frame)

        frame, box = panel("PREVIOUS SESSIONS AND ROUNDS (logs/sessions)")
        self.sessions_list = QListWidget()
        self.sessions_list.setMinimumHeight(130)
        box.addWidget(self.sessions_list)
        row = QHBoxLayout()
        self.open_folder_button = QPushButton("Open folder")
        self.open_folder_button.clicked.connect(self._on_open_folder)
        row.addWidget(self.open_folder_button)
        row.addStretch(1)
        box.addLayout(row)
        right.addWidget(frame)
        right.addStretch(1)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setWidget(body)
        outer.addWidget(scroll, 1)

    # ================================================================ inputs for the controller

    def snapshot(self) -> Any:
        try:
            return None if self.service is None else self.service.latest()
        except Exception:
            return None

    def _service_now(self) -> float:
        try:
            return float(self.service.now())
        except Exception:
            return self._clock()

    def gate_state(self) -> tuple[bool, list[str]]:
        """(gate open, the closed items with their reasons). Exactly what the ModeManager says."""
        snap = self.snapshot()
        if snap is None:
            return False, ["link service: there is no link service in this session, so AUTONOMOUS cannot be requested"]
        reasons = [f"{item.name}: {item.reason} [{item.kind}]" for item in snap.gate if not item.ok]
        return bool(snap.mode.gate_open), reasons

    def _arena(self) -> tuple[bool | None, str]:
        trusted = getattr(self.hub, "world_coordinates_trusted", None)
        report = getattr(self.hub, "restore_report", None)
        if trusted is True:
            return True, "arena calibration loaded, aligned and confirmed this run"
        if trusted is None:
            return None, "no remembered calibration to judge in this session (MOCK, REPLAY or no settings folder): not trusted"
        return False, "calibration not confirmed this run" + (f": {report.headline}" if report is not None else "")

    def _controller_report(self) -> tuple[bool, str]:
        monitor = getattr(self.hub, "monitor", None)
        if monitor is None:
            return False, "no controller monitor in this session"
        try:
            authority = monitor.authority(time.monotonic())
        except Exception as error:
            return False, f"controller monitor failed: {error}"
        if not authority.fresh or authority.status is None:
            return False, f"the controller reports nothing fresh ({authority.freshness}): the radio link cannot be confirmed"
        if not authority.status.link:
            return False, "the controller reports the RadioMaster link DOWN"
        return True, "the controller reports the RadioMaster link UP (this is the controller's own report)"

    def assemble_inputs(self, now: float | None = None) -> MatchInputs:
        now = self._clock() if now is None else now
        hub = self.hub
        capture = getattr(hub, "capture", None)
        view = live = None
        if self.ai is not None:
            view = self.ai.observation_status(now)
            live = self.ai.live_inputs()
        else:
            try:
                live = ms.collect_inputs(hub, self.tracking, now)
            except Exception:
                live = None
        up, up_detail = self._controller_report()
        trusted, arena_detail = self._arena()
        mapping_ok = bool(self.ai is not None and self.ai.mapping_validated())
        check = self.ai.mapping_check() if self.ai is not None and hasattr(self.ai, "mapping_check") else None
        error = self.ai.last_inference_error() if self.ai is not None else ""
        try:
            age = capture.last_frame_age()
        except Exception:
            age = None
        return ms.build_match_inputs(
            camera_state=str(getattr(capture, "state", "")), camera_detail=str(getattr(capture, "detail", "") or ""), camera_age_s=age,
            arena_trusted=trusted, arena_detail=arena_detail, own=None if live is None else live.own,
            opponent=None if live is None else live.opponent, observation=view,
            model_loaded=bool(self.ai is not None and self.ai.model_loaded()),
            model_detail=self.ai.model_detail() if self.ai is not None else "there is no AI / MODEL screen in this session",
            mapping_validated=mapping_ok,
            mapping_detail=("software self-check PASS" if mapping_ok else
                            "software self-check FAILED: " + "; ".join(l.name for l in check.failures) if check is not None and not check.passed else
                            "the mapping self-check has not been run (AI / MODEL tab, Check mapping)"),
            snapshot=self.snapshot(), controller_up=up, controller_detail=up_detail, inference_error=error)

    # ================================================================ mode selection (never starts a round)

    def select_mode(self, mode: Mode) -> tuple[bool, str]:
        """Choose the mode Start Round will ask for; during a round, change the running mode through the controller."""
        running = self.controller.running
        if mode is Mode.AUTONOMOUS:
            open_, reasons = self.gate_state()
            if not open_:
                self._selected = Mode.AUTONOMOUS
                self._action_text = ("AUTONOMOUS is NOT available: the ModeManager's gate is closed. " + "; ".join(reasons)
                                     + ". Start Round will start AI SHADOW instead.")
                if running:
                    self._selected = self.controller.mode
                    self._action_text = "AUTONOMOUS refused while running: " + "; ".join(reasons)
                self._sync_mode_buttons()
                self.paint_all()
                return False, self._action_text
            confirmation = self._ask_confirmation()
            if confirmation is None:
                self._action_text = "AUTONOMOUS was not confirmed: the selection is unchanged"
                self._sync_mode_buttons()
                self.paint_all()
                return False, self._action_text
            self._confirmation = confirmation
            self._selected = Mode.AUTONOMOUS
            if running:
                ok, why = self.controller.set_mode(Mode.AUTONOMOUS, self.assemble_inputs(), None, confirmation)
                self._confirmation = None
                self._action_text = "AUTONOMOUS requested" if ok else "AUTONOMOUS refused: " + "; ".join(why)
                if not ok:
                    self._selected = self.controller.mode
                self._sync_mode_buttons()
                self.paint_all()
                return ok, self._action_text
            self._action_text = "AUTONOMOUS confirmed for Start Round (valid for 30 s). Nothing has started."
            self._sync_mode_buttons()
            self.paint_all()
            return True, self._action_text
        self._selected = mode
        self._confirmation = None
        if running:
            ok, why = self.controller.set_mode(mode, self.assemble_inputs())
            self._action_text = f"{MODE_TEXT[mode]} requested" if ok else f"{MODE_TEXT[mode]} refused: " + "; ".join(why)
        else:
            self._action_text = f"{MODE_TEXT[mode]} selected for the next round. Nothing has started; nothing is transmitted."
        self._sync_mode_buttons()
        self.paint_all()
        return True, self._action_text

    def _ask_confirmation(self) -> Confirmation | None:
        if self._confirm_hook is not None:
            return self._confirm_hook()
        dialog = AutonomousConfirmationDialog(self, self._service_now)
        return dialog.confirmation() if dialog.exec() == QDialog.DialogCode.Accepted else None

    def _sync_mode_buttons(self) -> None:
        shown = self.controller.mode if self.controller.running else self._selected
        for mode, button in self.mode_buttons.items():
            blocked = button.blockSignals(True)
            button.setChecked(mode is shown)
            button.blockSignals(blocked)

    # ================================================================ start / stop (the only way a round begins)

    def _on_start_clicked(self) -> None:
        self.start_round(operator_action=True)

    def _on_stop_clicked(self) -> None:
        self.stop_round()

    def _on_emergency_clicked(self) -> None:
        self.emergency_stop()

    def start_round(self, *, operator_action: bool = False) -> StartResult:
        """Begin a round. ``operator_action=True`` is passed ONLY by the Start Round button's handler; any other caller is refused."""
        self.start_attempts += 1
        now = self._clock()
        inputs = self.assemble_inputs(now)
        requested = self._selected
        confirmation = None
        if operator_action and requested is Mode.AUTONOMOUS and self.gate_state()[0]:
            confirmation = self._usable_confirmation()
            if confirmation is None:
                self._action_text = "Start cancelled: AUTONOMOUS was not confirmed"
                self.paint_all()
                return StartResult(False, None, requested, False, ("AUTONOMOUS was not confirmed",))
        result = self.controller.start_round(requested, inputs, now, confirmation, operator_action=operator_action)
        self._confirmation = None
        self._action_text = result.message
        if result.started and result.mode in (Mode.AI_SHADOW, Mode.AUTONOMOUS):
            ok, why = self.ai.start_loop("MATCH round") if self.ai is not None else (False, "there is no AI / MODEL screen")
            if not ok:
                self.controller.stop_round(f"the inference loop could not start: {why}")
                self._action_text = f"Round stopped at once: the inference loop could not start: {why}"
                result = StartResult(False, None, requested, False, (why,))
        self._sync_mode_buttons()
        self.paint_all()
        return result

    def _usable_confirmation(self) -> Confirmation | None:
        held = self._confirmation
        if held is not None and self._service_now() - held.created_at <= CONFIRMATION_REUSE_S:
            return held
        return self._ask_confirmation()

    def stop_round(self, reason: str = "operator pressed Stop Round") -> None:
        """Always works: neutral, MANUAL, the opponent selection cleared, the round log closed."""
        self.controller.stop_round(reason, self._clock())
        self._selected = Mode.MANUAL
        self._confirmation = None
        self._action_text = self.controller.message
        self._sync_mode_buttons()
        self.paint_all()

    def emergency_stop(self) -> None:
        """ModeManager.emergency_stop through the link service (neutral where the link allows it, MANUAL), then the round ends.
        It travels over Wi-Fi like any command: when Wi-Fi is down it cannot reach the robot."""
        service = self.service
        if service is not None:
            try:
                service.return_to_manual()
            except Exception:
                log.exception("return_to_manual failed")
        self.stop_round("EMERGENCY STOP pressed")
        self._action_text = ("EMERGENCY STOP sent through the link service. " + self.controller.message +
                             " If Wi-Fi is down the stop could not reach the robot: use the RadioMaster ARM switch.")
        self.paint_all()

    # ---- the controller's callbacks

    def _round_started(self, _number: int) -> None:
        snap = self.snapshot()
        plans = getattr(snap, "plans", ()) or ()
        self._last_plan_t = max((p.t for p in plans), default=float("-inf"))      # earlier plans belong to no round
        self._last_decision_key, self._last_decision_logged = None, float("-inf")
        self._last_mode_seen = getattr(getattr(snap, "mode", None), "mode", None)

    def _round_ended(self) -> None:
        """Called by the controller on EVERY way a round ends (button, emergency, fault, time, application exit)."""
        if self.ai is not None:
            self.ai.stop_loop()
        clear = getattr(self.tracking, "clear_opponent", None)
        if clear is not None:
            try:
                clear()                                        # a new opponent every round
            except Exception:
                log.exception("clearing the opponent selection failed")

    # ================================================================ the 20 Hz tick

    def _on_timer(self) -> None:
        if self._closed:
            return
        if self.controller.running or self.isVisible():
            self.tick()

    def tick(self, now: float | None = None) -> list[str]:
        """One step: keep tracking resumed, run the inference loop (only during a round), feed the decision, check the faults, log, repaint.
        Returns the faults found in this step. Never starts a round."""
        if self._closed:
            return []
        now = self._clock() if now is None else now
        self._ticks += 1
        running = self.controller.running
        if running or self.isVisible():
            ms.ensure_tracking_active(self.tracking)
        decision = None
        if running and self.ai is not None and self.ai.loop_active:
            decision = self.ai.infer_once(now)
        inputs = self.assemble_inputs(now)
        stalled = ""
        if running and self.ai is not None and self.ai.loop_active:
            at = self.ai.last_decision_at()
            if at is not None and now - at > INFERENCE_STALL_S:
                stalled = f"the inference loop has produced no decision for {now - at:.1f} s"
        error = inputs.inference_error or stalled
        controller_inputs = dataclasses.replace(inputs, inference_error=self._throttled_error(error, now))
        if decision is not None:
            view = self.ai.observation_status(now)
            allow = decision.valid_for_control or self.controller.mode is not Mode.AUTONOMOUS
            feed = ms.decision_feed(decision, view.result, allow_tactic=allow)
            snap = self.snapshot()
            mock = decision.source != "LIVE" or bool(getattr(snap, "simulated", False))
            self.controller.feed_decision(feed, controller_inputs, now, observation_at=decision.input_time, mock=mock)
            self._log_decision(decision, now)
        faults = self.controller.tick(controller_inputs, now) if running else []
        self._inputs = inputs
        if running:
            self._log_round_state(inputs, now)
        if running and not self.controller.running:
            self._selected = Mode.MANUAL                          # the controller ended the round itself (time limit, fault): back to MANUAL
        self.paint(now, heavy=(self._ticks % SLOW_EVERY == 1))
        return faults

    def _throttled_error(self, error: str, now: float) -> str:
        """The controller records an inference error on every tick it sees one; hand it the same error at most every 2 s."""
        if not error:
            self._last_error = ""
            return ""
        if error != self._last_error or now - self._last_error_at >= ERROR_REPEAT_S:
            self._last_error, self._last_error_at = error, now
            return error
        return ""

    # ================================================================ session log (what was seen, decided and attempted)

    def _log(self, kind: str, event: str, **data: Any) -> None:
        session = getattr(self.hub, "session_log", None)
        if session is not None:
            session.log(kind, event, **data)

    def _log_decision(self, decision: Any, now: float) -> None:
        key = (decision.tactic, decision.valid_for_control, tuple(decision.blocked_by), decision.source)
        if key == self._last_decision_key and now - self._last_decision_logged < DECISION_LOG_PERIOD_S:
            return
        self._last_decision_key, self._last_decision_logged = key, now
        self._log("decision", decision.tactic or "no tactic", tactic=decision.tactic, source=decision.source,
                  valid_for_control=decision.valid_for_control, blocked_by=list(decision.blocked_by),
                  label="LIVE" if decision.valid_for_control else "DIAGNOSTIC", probability=decision.probability,
                  note="a proposal only: nothing is sent unless the mode manager's gated sender accepts it")

    def _log_round_state(self, inputs: MatchInputs, now: float) -> None:
        session = getattr(self.hub, "session_log", None)
        view = self.ai.observation_status(now) if self.ai is not None else None
        if session is not None and view is not None:
            session.log("observation", view.label, complete=view.complete, source=view.source,
                        groups={row.label: row.word for row in view.groups})
        snap = self.snapshot()
        if snap is None or session is None:
            return
        for plan in snap.plans:                                  # every NEW command plan: what was planned and whether a datagram went out
            if plan.t > self._last_plan_t:
                self._last_plan_t = plan.t
                session.command(plan, mode=plan.mode, authority=plan.authority)
        mode = snap.mode.mode
        if self._last_mode_seen is not None and mode is not self._last_mode_seen:
            session.mode_change(self._last_mode_seen.value, mode.value, True, [])
        self._last_mode_seen = mode
        worker_error = getattr(snap, "worker_error", None) or ""
        if worker_error and worker_error != self._last_worker_error:
            session.fault("link service reported an internal error", detail=worker_error)
        self._last_worker_error = worker_error

    # ================================================================ readiness for the guided page and SYSTEM STATUS

    def readiness(self) -> Readiness:
        return ms.match_readiness(self.assemble_inputs())

    def autonomous_readiness(self) -> Readiness:
        """BLOCKED with the gate's exact reasons while it is closed (expected); READY only if it is open AND every prerequisite holds."""
        inputs = self.assemble_inputs()
        open_, reasons = self.gate_state()
        return ms.autonomous_readiness_for(open_, reasons, missing(inputs, Mode.AUTONOMOUS))

    # ================================================================ painting

    def showEvent(self, event) -> None:  # noqa: N802 - Qt API
        super().showEvent(event)
        self.paint_all()

    def paint_all(self) -> None:
        self._inputs = self.assemble_inputs(self._clock())
        self.paint(self._clock(), heavy=True, inputs=self._inputs, force=True)

    def paint(self, now: float, heavy: bool = False, inputs: MatchInputs | None = None, force: bool = False) -> None:
        if self._closed:
            return
        inputs = inputs or self._inputs
        controller = self.controller
        running = controller.running
        set_text(self.timer_label, ms.format_elapsed(controller.elapsed(now)), "ok" if running else "off")
        self.start_button.setEnabled(not running)
        self.stop_button.setEnabled(True)                       # always enabled
        self.emergency_button.setEnabled(True)
        set_text(self.round_label, (f"ROUND {controller.round_number} RUNNING in {MODE_TEXT[controller.mode]}" if running else
                                    "NO ROUND RUNNING: nothing is transmitted"), "ok" if running else "off")
        set_text(self.message_label, controller.message)
        set_text(self.action_label, self._action_text)
        stopped = controller.needs_reconfirm
        if stopped:
            self.banner.setText("AUTONOMOUS STOPPED: " + (controller.faults[-1] if controller.faults else controller.message)
                                + "  Now AI SHADOW. Re-enter AUTONOMOUS deliberately: select AUTONOMOUS and confirm again.")
            self.banner.show()
        else:
            self.banner.hide()
        open_, reasons = self.gate_state()
        effective = "AI SHADOW instead (AUTONOMOUS is unavailable)" if self._selected is Mode.AUTONOMOUS and not open_ else MODE_TEXT[self._selected]
        self.start_button.setText("Start Round" if not running else "Round running")
        self.start_button.setToolTip(f"Starts a round in {effective}.")
        shown = controller.mode if running else self._selected
        set_text(self.mode_note, f"{MODE_TEXT[shown]}: {MODE_MEANING[shown]}.  Next Start Round: {effective}.")
        set_text(self.gate_label, ("AUTONOMOUS: AVAILABLE (gate open; needs the typed confirmation)" if open_ else "AUTONOMOUS: UNAVAILABLE (gate closed)"),
                 "warn" if open_ else "blocked")
        set_text(self.gate_reasons, "" if open_ else "Why: " + "; ".join(reasons))
        set_text(self.profile_label, f"profile: {profile_label(str(getattr(self.hub, 'profile_name', 'live')))}")
        snap = self.snapshot()
        simulated = bool(getattr(snap, "simulated", True)) if snap is not None else True
        camera_state = str(getattr(getattr(self.hub, "capture", None), "state", ""))
        evidence = ("SIMULATED: " + ("a mock or fake link; " if simulated else "") +
                    ("MOCK camera; " if camera_state == "MOCK" else "REPLAY, not a live camera; " if camera_state == "REPLAY" else "") +
                    "nothing shown here comes from the robot." if simulated or camera_state in ("MOCK", "REPLAY") else
                    "LIVE link and camera. Software results are SOFTWARE evidence; only the hardware acceptance record is HARDWARE evidence.")
        set_text(self.evidence_label, evidence)
        self.evidence_label.setStyleSheet("background: " + ("#4d3a12; color: #ffeab5;" if simulated or camera_state in ("MOCK", "REPLAY")
                                                            else "#1f3a2b; color: #cdeedb;"))
        readiness = ms.match_readiness(inputs)
        text = f"{readiness.level.upper()}: {readiness.headline}"
        if self.readiness_chip.text() != text:
            self.readiness_chip.setText(text)
            self.readiness_chip.setStyleSheet(f"background: {CHIP_COLOURS[readiness.level]};")
            self.readiness_chip.setToolTip(readiness.detail)
        self.authority_strip.update_from(snap)
        self._sync_mode_buttons()
        if not (heavy or force):
            return
        if not (self.isVisible() or force):
            return
        self._paint_prerequisites(inputs)
        self._paint_board(inputs, snap, now)
        self._paint_faults(inputs, snap)
        if force or self._ticks % LOG_EVERY == 1:
            self._paint_log()

    def _paint_prerequisites(self, inputs: MatchInputs) -> None:
        rows = []
        for p in prerequisites(inputs):
            needed = "AI SHADOW and AUTONOMOUS" if p.shadow else "AUTONOMOUS"
            rows.append(([p.name, "PASS" if p.ok else "FAIL", needed, p.detail], "ok" if p.ok else ("bad" if p.shadow else "warn")))
        fill_table(self.prereq_table, rows)

    def _latest_plan(self, snap: Any) -> Any:
        plans = getattr(snap, "plans", ()) or ()
        return plans[-1] if plans else None

    def _paint_board(self, inputs: MatchInputs, snap: Any, now: float) -> None:
        ai = self.ai
        rows: list[tuple[str, str, str]] = []
        loaded = bool(ai is not None and ai.model_loaded())
        rows.append(("Model loaded", "YES" if loaded else "NO: " + (ai.model_detail() if ai is not None else "no AI / MODEL screen"), "ok" if loaded else "bad"))
        decision = ai.last_decision() if ai is not None else None
        if decision is None:
            rows.append(("Current AI tactic", "none: no decision (the loop runs only during an AI SHADOW or AUTONOMOUS round)", "off"))
        else:
            at = ai.last_decision_at()
            age = "" if at is None else f", decided {max(0.0, now - at):.1f} s ago"
            live = decision.valid_for_control
            rows.append(("Current AI tactic", (decision.tactic.upper() if decision.tactic else "NONE (blocked)") + f"  [{ms.LIVE_LABEL if live else ms.DIAGNOSTIC_LABEL}, "
                         f"source {decision.source}{age}]" + ("" if decision.tactic else ": " + "; ".join(decision.blocked_by[:2])), "ok" if live else "warn"))
        plan = self._latest_plan(snap)
        if plan is None:
            rows.append(("INTENDED throttle / steer", "- (no command plan yet)", "off"))
        else:
            rows.append(("INTENDED throttle / steer", f"{plan.intended_throttle} / {plan.intended_steer}  (tactic {plan.tactic}; what the table says, before any cap)", "plain"))
        recent_sent = [p for p in (getattr(snap, "plans", ()) or ()) if p.sent and (p.transmitted_throttle or p.transmitted_steer)]
        transmitted = recent_sent[-1] if recent_sent else None
        mode_now = self.controller.mode if self.controller.running else (snap.mode.mode if snap is not None else Mode.MANUAL)
        if getattr(snap, "transmitting", False) and transmitted is not None and mode_now is Mode.AUTONOMOUS:
            rows.append(("ACTUAL TRANSMITTED throttle / steer", f"{transmitted.transmitted_throttle} / {transmitted.transmitted_steer}  (a datagram was written; "
                                                                 "this is NOT evidence the robot moved)", "bad"))
        else:
            why = "MANUAL: the RadioMaster drives" if mode_now is Mode.MANUAL else "AI SHADOW: every transmitted value is 0" if mode_now is Mode.AI_SHADOW else (plan.why_not if plan else "")
            rows.append(("ACTUAL TRANSMITTED throttle / steer", f"0 / 0  -  {ms.NOT_TRANSMITTED_TEXT} ({why})", "ok"))
        rows += self._own_rows(inputs)
        rows += self._opponent_rows()
        capture = getattr(self.hub, "capture", None)
        try:
            text, level = tm.camera_summary(capture.state, capture.detail, float(getattr(capture, "measured_fps", 0.0)), capture.last_frame_age())
        except Exception:
            text, level = "camera status unavailable", "bad"
        rows.append(("Camera", text, level))
        health = getattr(snap, "health", None)
        if health is None:
            rows.append(("Wi-Fi", "no link service", "bad"))
        else:
            state = getattr(health.state, "value", str(health.state))
            beat = "" if health.heartbeat_age_s is None else f", heartbeat {1000 * health.heartbeat_age_s:.0f} ms old"
            rows.append(("Wi-Fi", f"{state}{beat}" + ("  SIMULATED link" if snap.simulated else "") + (f"  ({health.error})" if health.error else ""),
                         "ok" if state == "CONNECTED" and not snap.simulated else "warn"))
        self.board.set_rows(rows)

    def _own_rows(self, inputs: MatchInputs) -> list[tuple[str, str, str]]:
        live = self.ai.live_inputs() if self.ai is not None else None
        own = None if live is None else live.own
        if own is None:
            return [("Own robot", "no track: " + inputs.own_detail, "bad")]
        position = "no position" if own.get("x_m") is None else f"x {own['x_m']:.3f}, y {own['y_m']:.3f} m, heading {own['heading_deg']:.1f} deg"
        markers = f"markers used {own.get('used_ids') or 'none'}, visible {own.get('visible_ids') or 'none'}"
        return [("Own robot", f"{str(own['state']).upper()}  {position}  ({markers})", "ok" if own["state"] == "measured" else "warn" if own["state"] == "predicted" else "bad")]

    def _opponent_rows(self) -> list[tuple[str, str, str]]:
        live = self.ai.live_inputs() if self.ai is not None else None
        opponent = None if live is None else live.opponent
        if opponent is None:
            return [("Opponent", "not selected (select it on the TRACKING tab; it is cleared at the end of every round)", "off")]
        confidence = opponent.get("confidence")
        position = "no position" if opponent.get("x_m") is None else f"x {opponent['x_m']:.3f}, y {opponent['y_m']:.3f} m (IMAGE-BASED, approximate)"
        return [("Opponent", f"{str(opponent['state']).upper().replace('_', ' ')}  confidence "
                             f"{'n/a' if confidence is None else f'{confidence:.2f}'}  {position}", "ok" if opponent["state"] == "tracked" else "warn")]

    def _paint_faults(self, inputs: MatchInputs, snap: Any) -> None:
        controller = self.controller
        mode_now = controller.mode if controller.running else (snap.mode.mode if snap is not None else Mode.MANUAL)
        physical = "Physical safety checks: " + ("recorded as HARDWARE evidence" if inputs.safety_done else "NOT completed (none recorded as hardware evidence)")
        set_text(self.safety_label, f"Mode {MODE_TEXT[mode_now]}.  {physical}.", "ok" if inputs.safety_done else "warn")
        items = list(controller.faults[-5:])
        if inputs.inference_error:
            items.append("inference: " + inputs.inference_error)
        worker = getattr(snap, "worker_error", None)
        if worker:
            items.append("link service: " + str(worker))
        set_text(self.faults_label, "FAULTS:\n" + "\n".join(items) if items else "No faults.", "bad" if items else "off")

    def _log_root(self) -> Path | None:
        session = getattr(self.hub, "session_log", None)
        return None if session is None else session.root

    def _paint_log(self) -> None:
        session = getattr(self.hub, "session_log", None)
        if session is None:
            self.log_view.setPlainText("No session log in this session (it is written to logs/sessions when the application runs normally).")
            self.log_where.setText("")
        else:
            events = ms.tail_events(session.folder / "session.jsonl", 60)
            self.log_view.setPlainText("\n".join(describe(e) for e in events))
            self.log_view.verticalScrollBar().setValue(self.log_view.verticalScrollBar().maximum())
            self.log_where.setText(f"{session.folder}  (rounds: round-NNN.jsonl in the same folder)")
        selected = self.sessions_list.currentRow()
        self.sessions_list.clear()
        for folder, rounds in ms.sessions_with_rounds(self._log_root()):
            self.sessions_list.addItem(f"{folder.name}   rounds: {', '.join(r.replace('.jsonl', '') for r in rounds) or 'none'}")
            self.sessions_list.item(self.sessions_list.count() - 1).setData(Qt.ItemDataRole.UserRole, str(folder))
        if 0 <= selected < self.sessions_list.count():
            self.sessions_list.setCurrentRow(selected)

    def _on_open_folder(self) -> None:
        item = self.sessions_list.currentItem()
        session = getattr(self.hub, "session_log", None)
        folder = Path(item.data(Qt.ItemDataRole.UserRole)) if item is not None else (session.folder if session is not None else None)
        if folder is not None:
            self.open_folder(folder)

    def open_folder(self, folder: Path) -> None:
        self._open_folder_calls.append(folder)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(folder)))

    # ================================================================ test helpers

    def prerequisite_texts(self) -> list[list[str]]:
        return table_texts(self.prereq_table)

    def board_texts(self) -> list[tuple[str, str]]:
        return self.board.texts()

    # ================================================================ shutdown

    def shutdown(self) -> None:
        """Stop a running round (neutral, MANUAL), the timer and the inference loop. Closes nothing it does not own: the link service,
        the camera and the session log belong to the hub, the model to the AI / MODEL screen."""
        if self._closed:
            return
        self._timer.stop()
        try:
            self.controller.shutdown()
        except Exception:
            log.exception("stopping the round at shutdown failed")
        if self.ai is not None:
            self.ai.stop_loop()
        self._closed = True

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt API
        self.shutdown()
        super().closeEvent(event)
