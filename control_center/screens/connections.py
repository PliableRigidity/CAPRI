"""CONNECTIONS: the Wi-Fi link to the robot, who has authority, the autonomous gate and the acceptance workflow.

This screen displays a ``ServiceSnapshot`` from ``control_center/backend/link_service.py`` and sends the operator's requests
to that service. It never touches the link, the mode manager or the acceptance record itself, never opens a socket, serial
port or camera, and creates no transport: only the service does, and only when the operator presses Connect (or Discover).

What it will not do, whatever is pressed:
  * It has no button that arms the robot, selects a mode other than MANUAL, or sends a drive command. AI SHADOW and
    AUTONOMOUS are chosen on the Match tab; this screen shows the current mode and the reason for every closed gate.
  * It cannot mark a physical step PASS. It records what the operator REPORTS (kept as NOT TESTED) and the two things the
    app measures itself (step 2 and the step 3 stability window); the backend decides, and its refusal text is shown.
  * Anything that comes from a mock, a replay or a fake link is labelled SIMULATED and can never make the chip READY.

``service`` may be None (the screen then shows NOT CONFIGURED and every control is disabled). Ownership: the dashboard owns
the service and must call ``service.shutdown()``; ``ConnectionsScreen.shutdown()`` only detaches this screen.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QBrush, QColor
from PySide6.QtWidgets import (QAbstractItemView, QApplication, QComboBox, QFrame, QGridLayout, QHBoxLayout, QHeaderView, QLabel,
                               QLineEdit, QPlainTextEdit, QProgressBar, QPushButton, QScrollArea, QTableWidget, QTableWidgetItem,
                               QVBoxLayout, QWidget)

from control_center.backend.acceptance import (BLOCKED as STEP_BLOCKED, FAIL, NOT_TESTED, PASS, SUBCHECKS, Evidence)
from control_center.backend.readiness import (BLOCKED, CHIP_COLOURS, FAILED, NOT_CONFIGURED, READY, WARNING, Readiness)
from control_center.backend.wifi_link import LinkState, validate_endpoint
from control_center.widgets.authority_strip import AuthorityStrip

TITLE = "CONNECTIONS"
STALE_AFTER_S = 1.5                   # a running service that has not published for this long is not trusted
REFRESH_MS = 1000                     # re-evaluates staleness when no snapshot arrives
LOG_LINES = 200
ACCEPTANCE_PHRASE = "I AM AT THE REAL ROBOT"
BLOCKED_HEADLINE = "Wi-Fi connected, but motor command transmission has not been physically verified"
NOT_SUPPORTED_ACK = "not supported by this firmware (it sends no acknowledgement of a command)"
OPEN_STATES = (LinkState.CONNECTING, LinkState.CONNECTED, LinkState.DEGRADED)

LEVEL_COLOURS = {"ok": "#55c889", "warn": "#e8b85b", "bad": "#ef6b73", "off": "#9aa5b1", "plain": "#d9dee5",
                 "blocked": "#b48be8"}
STATUS_LEVEL = {PASS: "ok", FAIL: "bad", STEP_BLOCKED: "blocked", NOT_TESTED: "off"}
EVIDENCE_WORDS = {Evidence.HARDWARE: "HARDWARE", Evidence.MOCK: "SIMULATED (mock)", Evidence.SIM: "SIMULATED (simulator)",
                  Evidence.OPERATOR_REPORTED: "OPERATOR-REPORTED"}
GATE_KIND_WORDS = {"HARDWARE": "HARDWARE record", "LIVE": "LIVE (measured now)", "SOFTWARE": "SOFTWARE", "SESSION": "SESSION"}

STYLE = """
QLabel { background: transparent; }
QFrame#panel { background: #171e27; border: 1px solid #2b3542; border-radius: 5px; }
QFrame#panel QLabel { color: #d9dee5; }
QLabel#panelTitle { color: #9eabb8; font-size: 10pt; font-weight: 700; letter-spacing: 1px; }
QLabel#hint { color: #9eabb8; }
QLabel#mono { font-family: Consolas; }
QTableWidget { background: #0d1218; alternate-background-color: #121922; gridline-color: #2b3542; color: #d9dee5;
               border: 1px solid #2b3542; }
QHeaderView::section { background: #171e27; color: #9eabb8; border: none; border-bottom: 1px solid #2b3542;
                       padding: 4px 6px; font-weight: 700; }
QLineEdit, QComboBox, QPlainTextEdit { background: #0d1218; border: 1px solid #3b4a5d; border-radius: 4px; padding: 5px 6px; }
QPlainTextEdit { font-family: Consolas; }
QScrollArea { border: none; background: transparent; }
QWidget#scrollBody { background: transparent; }
QPushButton:disabled { color: #6b7684; }
QProgressBar { border: 1px solid #3b4a5d; border-radius: 4px; text-align: center; }
"""


# ================================================================ pure helpers (tested directly, no Qt widgets)


def _age_ms(seconds: float | None) -> str:
    return "none yet" if seconds is None else f"{seconds * 1000:.0f} ms"


def wifi_readiness(snapshot: Any, now: float | None = None) -> Readiness:
    """The Wi-Fi link's readiness from a ``ServiceSnapshot`` alone (so a screen cannot say Ready without the backend saying so).

    NOT CONFIGURED  never connected / disconnected, or no service at all
    FAILED          the connection failed: the backend's plain-language error and troubleshooting hint
    WARNING         connected but degraded, stale, protocol unknown, no HELLO; or connecting; or a SIMULATED link
    BLOCKED         link fine, but motor transmission has not been physically verified (the expected state)
    READY           connected AND the acceptance record allows autonomous AND this build enables it, never when simulated
    """
    if snapshot is None:
        return Readiness(NOT_CONFIGURED, "No Wi-Fi link service in this session",
                         "This window has no link service, so it cannot connect to the robot. Nothing is sent.",
                         "Restart the Command Centre; see logs/ if it repeats. The RadioMaster drives either way.")
    health = snapshot.health
    state = health.state
    endpoint = health.endpoint.label
    if now is not None and snapshot.running and now - snapshot.t > STALE_AFTER_S:
        return Readiness(WARNING, "The link status is not updating",
                         f"The newest update is {now - snapshot.t:.1f} s old, so what is shown may be out of date.",
                         "Press Return to MANUAL. If this does not clear, restart the Command Centre.")
    if snapshot.worker_error:
        return Readiness(WARNING, "The link service reported an internal error",
                         f"{snapshot.worker_error}. Neutral and MANUAL were forced.", "Check the diagnostics log below.")
    if snapshot.simulated:
        if state is LinkState.FAILED:
            return Readiness(FAILED, "SIMULATED link failed: " + (health.error or "unknown"), "Nothing here is from the robot.",
                             health.hint or "")
        if state in (LinkState.CONNECTED, LinkState.DEGRADED, LinkState.CONNECTING):
            return Readiness(WARNING, "SIMULATED Wi-Fi link: nothing here is from the robot",
                             "A mock, replay or fake link is never physical verification and can never be Ready.",
                             "Connect to the real robot to get a real status.")
        return Readiness(NOT_CONFIGURED, "SIMULATED session: not connected", "Nothing here is from the robot.",
                         "Press Connect to start the simulated link, or restart without --mock for the real robot.")
    if state is LinkState.FAILED:
        return Readiness(FAILED, health.error or "The Wi-Fi link failed",
                         "The mode is MANUAL. If the robot was being driven by the AI it stops by itself 250 ms after its "
                         "last command.", health.hint or "Press Reconnect, then check the robot's power and Wi-Fi.")
    if state is LinkState.DISCONNECTED:
        ssid = snapshot.preferred_ssid or "AutoBot-Handoff"
        return Readiness(NOT_CONFIGURED, "Not connected to the robot's Wi-Fi link",
                         f"Endpoint {endpoint}. While disconnected this program sends nothing and the RadioMaster drives.",
                         f"Join the robot's own network ({ssid}) in the Windows Wi-Fi list, then press Connect.")
    if state is LinkState.CONNECTING:
        return Readiness(WARNING, f"Connecting to {endpoint}",
                         "Waiting for the robot's first status line and its HELLO line (status requests only).",
                         "Wait up to 4 seconds. If it fails, the reason and what to try appear here.")
    problems: list[str] = []
    if state is LinkState.DEGRADED:
        problems.append("the link is DEGRADED (status late, or packet loss high)")
    if health.heartbeat_age_s is None or health.heartbeat_age_s >= 0.5:
        problems.append("no fresh status heartbeat (older than 0.5 s)")
    if health.protocol_ok is None:
        problems.append("no HELLO line yet: the firmware is not identified")
    elif health.protocol_ok is False:
        problems.append("protocol not confirmed: " + health.protocol_note)
    if health.loss is not None and health.loss.high:
        problems.append("packet loss or status gaps are high")
    if problems:
        return Readiness(WARNING, "Wi-Fi connected, but the link is not healthy", "; ".join(problems) + ".",
                         "Move the laptop closer, or press Reconnect. Do not rely on Wi-Fi until this clears.")
    acceptance = snapshot.acceptance
    build_open = all(item.ok for item in snapshot.gate if item.name == "build")
    if acceptance.complete and build_open:
        return Readiness(READY, "Wi-Fi connected and the acceptance record allows autonomous",
                         "Every acceptance step is PASS from HARDWARE evidence for this firmware.")
    if acceptance.complete:
        return Readiness(BLOCKED, "Wi-Fi connected; the acceptance record is complete, but this build does not enable autonomous mode",
                         next((item.reason for item in snapshot.gate if item.name == "build" and not item.ok), ""),
                         "Autonomous mode changes only together with a hardware result, not by a code edit alone.")
    missing = [line.strip() for line in acceptance.missing if not line.startswith("   ")]
    shown = "; ".join(missing[:3]) + (f" (and {len(missing) - 3} more)" if len(missing) > 3 else "")
    return Readiness(BLOCKED, BLOCKED_HEADLINE, "Missing: " + shown if shown else "",
                     "Keep driving with the RadioMaster. Work through the ACCEPTANCE WORKFLOW below on the real robot.")


def gate_evidence_text(kind: str, simulated: bool) -> str:
    if kind == "LIVE" and simulated:
        return "SIMULATED (from a fake link)"
    return GATE_KIND_WORDS.get(kind, kind)


def health_rows(snapshot: Any) -> list[tuple[str, str, str, str]]:
    """(key, label, value, level) for the live health grid. Anything not measured says so and why: nothing is estimated."""
    health = snapshot.health
    missing = health.unavailable
    sim = "  [SIMULATED]" if snapshot.simulated else ""
    rows: list[tuple[str, str, str, str]] = []

    def add(key: str, label: str, value: str, level: str = "plain") -> None:
        rows.append((key, label, value, level))

    state_level = {LinkState.CONNECTED: "ok", LinkState.DEGRADED: "warn", LinkState.CONNECTING: "warn", LinkState.FAILED: "bad",
                   LinkState.DISCONNECTED: "off"}[health.state]
    add("state", "Link state", health.state.value + sim, state_level)
    add("endpoint", "Robot address", health.endpoint.label)
    add("firmware", "Firmware id", health.firmware_id or "not identified (no HELLO line yet)",
        "plain" if health.firmware_id else "off")
    protocol = ("OK: " if health.protocol_ok else "NOT OK: " if health.protocol_ok is False else "unknown: ") + health.protocol_note
    add("protocol", "Protocol", protocol, "ok" if health.protocol_ok else "bad" if health.protocol_ok is False else "off")
    add("heartbeat", "Heartbeat age (newest ST line)", _age_ms(health.heartbeat_age_s) if health.heartbeat_age_s is not None
        else "none: " + missing.get("heartbeat_age_s", "no ST line"),
        "ok" if health.heartbeat_age_s is not None and health.heartbeat_age_s < 0.5 else "warn")
    add("last_rx", "Last line of any kind", _age_ms(health.last_rx_age_s) if health.last_rx_age_s is not None
        else "none: " + missing.get("last_rx_age_s", "nothing received"))
    if snapshot.last_command_age_s is not None:
        command = f"{snapshot.last_command_age_s * 1000:.0f} ms ago (newest planned command that was really sent)"
    elif snapshot.ever_sent:
        command = "a command datagram was sent this session; its age is not recorded"
    else:
        command = "none sent this session"
    add("last_command", "Last command sent", command)
    add("latency", "Round-trip latency", f"{health.latency_ms:.0f} ms (median of {health.latency_samples} status requests)"
        if health.latency_ms is not None else "not measurable: " + missing.get("latency_ms", "no sample yet"))
    loss = health.loss
    if loss is None:
        add("loss", "Loss indicators", "not measurable: " + missing.get("loss", "not connected"))
    else:
        parts = []
        if loss.firmware_rx is not None:
            parts.append(f"robot counted rx {loss.firmware_rx}, bad {loss.firmware_bad}, old {loss.firmware_old}")
        else:
            parts.append("robot counters: not measurable (" + missing.get("loss.firmware", "no NET line") + ")")
        parts.append(f"loss ratio {loss.firmware_loss_ratio * 100:.1f} %" if loss.firmware_loss_ratio is not None
                     else "loss ratio: not measurable (" + missing.get("loss.firmware_loss_ratio", "not enough data") + ")")
        parts.append(f"late-heartbeat gaps {loss.pc_status_gaps} (recent {loss.pc_status_gaps_recent}), send errors "
                     f"{loss.pc_send_errors}, unanswered probes {loss.pc_probe_timeouts}")
        add("loss", "Loss indicators", "; ".join(parts), "warn" if loss.high else "plain")
    add("local_ip", "This laptop, as the robot sees it", health.local_ip or "not measurable: " + missing.get("local_ip", "unknown"))
    add("reconnects", "Reconnects / controller restarts", f"{health.reconnects} / {health.restarts}",
        "warn" if health.restarts else "plain")
    add("rearm", "Re-arm", f"REQUIRED: {health.latched_reason}" if health.needs_rearm else "the operator re-armed on this connection",
        "off" if health.needs_rearm else "ok")
    controller = snapshot.controller
    if controller is None:
        add("controller", "Controller report", "none: no status line from the controller (it reports nothing while there is no link)", "off")
    else:
        add("controller", "Controller report (as the firmware says)",
            f"armed {int(controller.armed)} · MODE switch asks {controller.requested} · active source {controller.active} · "
            f"radio link {'UP' if controller.link_up else 'DOWN'} · accepts PC commands {int(controller.pc_ok)} · block "
            f"{controller.block} · fault {controller.fault} · report age {controller.age_s * 1000:.0f} ms",
            "ok" if controller.fresh and controller.fault == "NONE" else "warn")
    add("ack", "Last command acknowledged", NOT_SUPPORTED_ACK, "off")
    if health.error:
        add("error", "Last error", health.error, "bad")
    return rows


def step_rows(snapshot: Any) -> tuple[list[tuple[str, ...]], list[str]]:
    """Rows and level keys for the acceptance table: the 7 steps and the five sub-checks of step 5."""
    rows: list[tuple[str, ...]] = []
    levels: list[str] = []

    def when(unix: float | None) -> str:
        return "" if unix is None else datetime.fromtimestamp(unix).strftime("%Y-%m-%d %H:%M")

    def note(view: Any) -> str:
        parts = [view.reason] if view.reason else []
        if view.notes and view.notes != view.reason:
            parts.append(view.notes)
        return " | ".join(parts)

    for view in snapshot.acceptance.steps:
        rows.append((f"{view.number}. {view.title}", view.status, EVIDENCE_WORDS.get(view.evidence or "", view.evidence or "-"),
                     when(view.t_unix), note(view)))
        levels.append(STATUS_LEVEL.get(view.status, "plain"))
        for child in view.subchecks:
            rows.append(("      " + child.title, child.status, EVIDENCE_WORDS.get(child.evidence or "", child.evidence or "-"),
                         when(child.t_unix), note(child)))
            levels.append(STATUS_LEVEL.get(child.status, "plain"))
    return rows, levels


def diagnostics_text(snapshot: Any, readiness: Readiness, events: tuple[str, ...]) -> str:
    """Everything a person needs to read over the phone: copied by the Copy diagnostics button."""
    lines = [f"Capri Command Centre - CONNECTIONS diagnostics ({datetime.now().isoformat(timespec='seconds')})"]
    if snapshot is None:
        lines += ["No link service is attached to this window.", "", f"Readiness: {readiness.level} - {readiness.headline}"]
        return "\n".join(lines)
    lines += [f"{'SIMULATED SESSION: nothing below is from the robot' if snapshot.simulated else 'Real link session'}",
              f"Readiness: {readiness.level} - {readiness.headline}", f"  {readiness.detail}", f"  Next: {readiness.fix}", "", "LINK"]
    lines += [f"  {label}: {value}" for _key, label, value, _level in health_rows(snapshot)]
    lines += ["", f"MODE {snapshot.mode.mode.value}; authority {snapshot.authority.authority.value}: {snapshot.authority.reason}",
              f"Last mode event: {snapshot.mode.last_event}", "", "AUTONOMOUS GATE (" + ("OPEN" if snapshot.mode.gate_open else "CLOSED") + ")"]
    lines += [f"  [{'PASS' if item.ok else 'FAIL'}] {item.name} ({gate_evidence_text(item.kind, snapshot.simulated)}): {item.reason}"
              for item in snapshot.gate]
    lines += ["", "ACCEPTANCE (only HARDWARE evidence counts)"]
    rows, _levels = step_rows(snapshot)
    lines += ["  " + " | ".join(part for part in row if part) for row in rows]
    lines += ["", f"EVENTS (newest last, up to {LOG_LINES})"] + [f"  {line}" for line in events[-LOG_LINES:]]
    return "\n".join(lines)


# ================================================================ small widget helpers


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

    def fill(self, rows: list[tuple[str, ...]], levels: list[str | None], colour_column: int = 1) -> None:
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
                    resize = True
            if levels[r]:
                self.item(r, colour_column).setForeground(QBrush(QColor(LEVEL_COLOURS[levels[r]])))
        if resize:
            self.resizeRowsToContents()
            self.setFixedHeight(self.horizontalHeader().height() + sum(self.rowHeight(i) for i in range(self.rowCount())) + 6)

    def cell_texts(self) -> list[list[str]]:
        return [[self.item(r, c).text() if self.item(r, c) else "" for c in range(self.columnCount())] for r in range(self.rowCount())]


# ================================================================ the screen


class ConnectionsScreen(QWidget):
    def __init__(self, hub, service=None, parent=None) -> None:
        super().__init__(parent)
        self.hub = hub
        self.service = service
        self._closed = False
        self._snap: Any = None
        self._dirty = False                            # the operator has typed in the endpoint boxes
        self._seen_endpoint: Any = None
        self._endpoint_valid = False
        self._last_readiness: Readiness | None = None
        self.setStyleSheet(STYLE)
        self._build()
        self._validate()
        if service is not None:
            self._snap = service.latest()
            for line in service.event_log()[-LOG_LINES:]:
                self.log_view.appendPlainText(line)
            service.snapshot_changed.connect(self._on_snapshot)
            service.event_added.connect(self._on_event)
        self._render()
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._render)
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

        # ---- (a) the status chip, the headline and the next action
        head = QHBoxLayout()
        title = QLabel(TITLE)
        title.setStyleSheet("font-size: 15pt; font-weight: 800;")
        head.addWidget(title)
        head.addStretch(1)
        self.status_chip = QLabel("")
        head.addWidget(self.status_chip)
        root.addLayout(head)
        self.sim_badge = _label("SIMULATED: this is a mock, replay or fake link. Nothing on this screen is from the robot, and "
                                "none of it can count as hardware evidence.", "hint")
        self.sim_badge.setStyleSheet("color: #e8b85b; font-weight: 700;")
        root.addWidget(self.sim_badge)
        self.headline_label = _label("")
        self.headline_label.setStyleSheet("font-size: 12pt; font-weight: 700;")
        root.addWidget(self.headline_label)
        self.detail_label = _label("", "hint")
        root.addWidget(self.detail_label)
        self.next_label = _label("")
        root.addWidget(self.next_label)

        # ---- (b) the endpoint row
        panel, box = _panel("ROBOT ADDRESS  ·  the robot's own Wi-Fi network (default 192.168.4.1, port 4210)")
        row = QHBoxLayout()
        row.addWidget(QLabel("Host"))
        self.host_edit = QLineEdit()
        self.host_edit.setPlaceholderText("192.168.4.1")
        self.host_edit.setMaximumWidth(220)
        row.addWidget(self.host_edit)
        row.addWidget(QLabel("Port"))
        self.port_edit = QLineEdit()
        self.port_edit.setPlaceholderText("4210")
        self.port_edit.setMaximumWidth(80)
        row.addWidget(self.port_edit)
        self.apply_button = QPushButton("Apply")
        self.connect_button = QPushButton("Connect")
        self.disconnect_button = QPushButton("Disconnect")
        self.reconnect_button = QPushButton("Reconnect")
        self.discover_button = QPushButton("Discover")
        for button in (self.apply_button, self.connect_button, self.disconnect_button, self.reconnect_button, self.discover_button):
            row.addWidget(button)
        row.addStretch(1)
        box.addLayout(row)
        self.endpoint_message = _label("", "hint")
        box.addWidget(self.endpoint_message)
        self.action_label = _label("", "hint")
        box.addWidget(self.action_label)
        self.discover_label = _label("Discover asks only the address above whether a robot answers (status requests, no scanning).", "hint")
        box.addWidget(self.discover_label)
        box.addWidget(_label("Connect sends status requests only: it can never move the robot. Reconnect goes back to MANUAL and "
                             "the link then needs a deliberate re-arm; nothing resumes by itself.", "hint"))
        root.addWidget(panel)
        self.host_edit.textEdited.connect(self._on_edited)
        self.port_edit.textEdited.connect(self._on_edited)
        self.apply_button.clicked.connect(self.apply_endpoint)
        self.connect_button.clicked.connect(self.press_connect)
        self.disconnect_button.clicked.connect(self.press_disconnect)
        self.reconnect_button.clicked.connect(self.press_reconnect)
        self.discover_button.clicked.connect(self.press_discover)

        # ---- (c) the live health grid
        panel, box = _panel("LIVE LINK HEALTH  ·  only what the firmware and this link really report")
        self.health_grid = QGridLayout()
        self.health_grid.setColumnStretch(1, 1)
        box.addLayout(self.health_grid)
        self.health_labels: dict[str, tuple[QLabel, QLabel]] = {}
        root.addWidget(panel)
        self._health_box = box

        # ---- (d) the authority strip and Return to MANUAL
        panel, box = _panel("MODE AND AUTHORITY")
        self.strip = AuthorityStrip()
        box.addWidget(self.strip)
        row = QHBoxLayout()
        self.manual_button = QPushButton("Return to MANUAL")
        self.manual_button.clicked.connect(self.press_return_to_manual)
        row.addWidget(self.manual_button)
        row.addWidget(_label("Always available. Sends a neutral command if the link is connected. AI SHADOW and AUTONOMOUS are "
                             "chosen on the Match tab.", "hint"), 1)
        box.addLayout(row)
        root.addWidget(panel)

        # ---- (e) the gate checklist
        panel, box = _panel("AUTONOMOUS GATE  ·  every item must pass; each closed item says why")
        self.gate_summary = _label("")
        box.addWidget(self.gate_summary)
        self.gate_table = _Table(["Item", "State", "Reason", "Evidence"], 2)
        box.addWidget(self.gate_table)
        root.addWidget(panel)

        # ---- (f) the acceptance workflow
        panel, box = _panel("ACCEPTANCE WORKFLOW  ·  Wi-Fi motor transmission stays BLOCKED until every step is PASS from HARDWARE")
        box.addWidget(_label("Only HARDWARE evidence counts. SIMULATED results are stored as NOT TESTED. OPERATOR-REPORTED means you "
                             "told this program: it did not measure it, it is kept as NOT TESTED and opens nothing. This screen "
                             "cannot mark a physical step PASS; a supervised hardware tool records those.", "hint"))
        self.acceptance_table = _Table(["Step", "Status", "Evidence", "When", "Notes / why"], 4)
        box.addWidget(self.acceptance_table)
        self.motion_check_label = _label("", "hint")
        box.addWidget(self.motion_check_label)

        box.addWidget(_label("Typed confirmation. Type the sentence below to enable the buttons that record anything. It says you are "
                             "physically with the real robot; it does not make a result true.", "hint"))
        row = QHBoxLayout()
        self.phrase_hint = _label(ACCEPTANCE_PHRASE, "mono", wrap=False)
        row.addWidget(self.phrase_hint)
        self.confirm_edit = QLineEdit()
        self.confirm_edit.setPlaceholderText("type the sentence")
        row.addWidget(self.confirm_edit, 1)
        self.confirm_state = _label("", "hint", wrap=False)
        row.addWidget(self.confirm_state)
        box.addLayout(row)
        self.confirm_edit.textChanged.connect(lambda _text: self._update_controls())

        row = QHBoxLayout()
        self.wifi_record_button = QPushButton("Record step 2: Wi-Fi established without motor commands")
        self.wifi_record_button.clicked.connect(self.press_record_wifi)
        row.addWidget(self.wifi_record_button)
        row.addStretch(1)
        box.addLayout(row)

        row = QHBoxLayout()
        self.stability_button = QPushButton("Start stability window (60 s)")
        self.stability_button.clicked.connect(self.press_stability)
        row.addWidget(self.stability_button)
        self.stability_bar = QProgressBar()
        self.stability_bar.setRange(0, 60)
        row.addWidget(self.stability_bar, 1)
        box.addLayout(row)
        self.stability_label = _label("", "hint")
        box.addWidget(self.stability_label)

        box.addWidget(_label("OPERATOR-REPORTED inputs for the physical steps (1 baseline, 5 transitions and failsafes). These are what "
                             "you say you saw; this program did not measure them. They are kept in the record as NOT TESTED.", "hint"))
        row = QHBoxLayout()
        self.report_step = QComboBox()
        self.report_step.addItem("Step 1: RadioMaster-only baseline on battery", ("baseline", None))
        for check, text in SUBCHECKS.items():
            self.report_step.addItem(f"Step 5: {text}", ("transitions", check))
        row.addWidget(self.report_step, 1)
        self.report_status = QComboBox()
        self.report_status.addItems([PASS, FAIL])
        row.addWidget(self.report_status)
        box.addLayout(row)
        row = QHBoxLayout()
        self.reading_edits: dict[str, QLineEdit] = {}
        for key, text in (("pack_rest_v", "pack at rest (V)"), ("converter_out_idle_v", "converter idle (V)"),
                          ("converter_out_wheels_v", "converter, wheels running (V)"), ("minutes_without_restart", "minutes without restart")):
            edit = QLineEdit()
            edit.setPlaceholderText(text)
            edit.setToolTip(text)
            row.addWidget(edit)
            self.reading_edits[key] = edit
        box.addLayout(row)
        self.report_notes = QLineEdit()
        self.report_notes.setPlaceholderText("notes (what you saw)")
        box.addWidget(self.report_notes)
        self.report_button = QPushButton("Record OPERATOR-REPORTED result")
        self.report_button.clicked.connect(self.press_report)
        box.addWidget(self.report_button)
        self.acceptance_message = _label("", "hint")
        box.addWidget(self.acceptance_message)
        root.addWidget(panel)

        # ---- (g) diagnostics
        panel, box = _panel("DIAGNOSTICS LOG  ·  connection events, hints, newest last (last 200 lines)")
        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(LOG_LINES)
        self.log_view.setMinimumHeight(180)
        box.addWidget(self.log_view)
        row = QHBoxLayout()
        self.copy_button = QPushButton("Copy diagnostics")
        self.copy_button.clicked.connect(self.copy_diagnostics)
        row.addWidget(self.copy_button)
        self.copy_label = _label("", "hint")
        row.addWidget(self.copy_label, 1)
        box.addLayout(row)
        root.addWidget(panel)
        root.addStretch(1)

    # ================================================================ readiness and rendering

    def readiness(self) -> Readiness:
        snapshot = self._snap if self._snap is not None else (self.service.latest() if self.service is not None else None)
        now = self.service.now() if self.service is not None else None
        return wifi_readiness(snapshot, now)

    def show_snapshot(self, snapshot: Any) -> None:
        """Render a given snapshot (what the signal does; also used by tests with scripted snapshots)."""
        self._snap = snapshot
        self._render()

    def _on_snapshot(self, snapshot: Any) -> None:
        if not self._closed:
            self.show_snapshot(snapshot)

    def _on_event(self, line: str) -> None:
        if not self._closed:
            self.log_view.appendPlainText(line)

    def _render(self) -> None:
        if self._closed:
            return
        snap = self._snap
        now = self.service.now() if self.service is not None else None
        chip = self._last_readiness = wifi_readiness(snap, now)
        self.status_chip.setText(chip.level.upper())
        self.status_chip.setStyleSheet(f"color: white; background: {CHIP_COLOURS[chip.level]}; border-radius: 4px; "
                                       "padding: 4px 14px; font-size: 13pt; font-weight: 800;")
        simulated = bool(snap is not None and snap.simulated) or (snap is None and bool(getattr(getattr(self.hub, "options", None), "mock", False)))
        self.sim_badge.setVisible(simulated)
        self.headline_label.setText(chip.headline)
        self.detail_label.setText(chip.detail)
        self.detail_label.setVisible(bool(chip.detail))
        self.next_label.setText(f"Next: {chip.fix}" if chip.fix else "")
        self.next_label.setVisible(bool(chip.fix))
        if snap is not None:
            self._render_endpoint(snap)
            self._render_health(snap)
            self.strip.update_from(snap)
            self._render_gate(snap)
            self._render_acceptance(snap)
            self._render_messages(snap)
        else:
            self.strip.update_from(None)
        self._update_controls()

    def _render_endpoint(self, snap: Any) -> None:
        if snap.endpoint != self._seen_endpoint:
            self._seen_endpoint = snap.endpoint
            shown = (snap.endpoint.host, str(snap.endpoint.port))
            if not self._dirty and (self.host_edit.text(), self.port_edit.text()) != shown:
                for edit, text in ((self.host_edit, shown[0]), (self.port_edit, shown[1])):
                    edit.blockSignals(True)
                    edit.setText(text)
                    edit.blockSignals(False)
                self._validate()

    def _render_health(self, snap: Any) -> None:
        for index, (key, label, value, level) in enumerate(health_rows(snap)):
            pair = self.health_labels.get(key)
            if pair is None:
                name, text = _label(label, "hint"), _label("")
                self.health_grid.addWidget(name, index, 0, Qt.AlignmentFlag.AlignTop)
                self.health_grid.addWidget(text, index, 1)
                pair = self.health_labels[key] = (name, text)
            if pair[1].text() != value:
                pair[1].setText(value)
            _colour(pair[1], level)
        for key in [k for k in self.health_labels if k not in {row[0] for row in health_rows(snap)}]:
            for widget in self.health_labels.pop(key):
                widget.setParent(None)

    def _render_gate(self, snap: Any) -> None:
        open_ = snap.mode.gate_open
        self.gate_summary.setText(("GATE OPEN: every item passes." if open_ else
                                   f"GATE CLOSED: {sum(1 for item in snap.gate if not item.ok)} of {len(snap.gate)} items fail. "
                                   "AUTONOMOUS cannot be entered, and nothing but neutral can be sent over Wi-Fi."))
        _colour(self.gate_summary, "ok" if open_ else "blocked", "font-weight: 700;")
        self.gate_table.fill([(item.name, "PASS" if item.ok else "FAIL", item.reason, gate_evidence_text(item.kind, snap.simulated))
                              for item in snap.gate], ["ok" if item.ok else "bad" for item in snap.gate], 1)

    def _render_acceptance(self, snap: Any) -> None:
        rows, levels = step_rows(snap)
        self.acceptance_table.fill(rows, levels, 1)
        info = snap.acceptance.motion_check
        self.motion_check_label.setText(("Motion-check record: " + info.text) if info is not None else "")
        self.motion_check_label.setVisible(info is not None)
        stability = snap.stability
        self.stability_bar.setValue(int(min(stability.progress_s, 60)))
        kind = "SIMULATED (mock): it will count for nothing" if snap.simulated else "real link: counts only with a typed confirmation"
        tail = f" {stability.recorded}" if stability.recorded else ""
        self.stability_label.setText(f"Stability window: {stability.state} ({stability.progress_s:.0f} of {stability.window_s:.0f} s) · "
                                     f"evidence {EVIDENCE_WORDS.get(stability.evidence, stability.evidence)} · {kind}." + tail
                                     + (f" {stability.reason}." if stability.state in ("FAILED", "RUNNING") else ""))

    def _render_messages(self, snap: Any) -> None:
        request = snap.last_request
        if request is None:
            self.action_label.setText("")
        else:
            self.action_label.setText(f"{request.what}: {'ok' if request.ok else 'REFUSED'}" + (f". {request.text}" if request.text else ""))
            _colour(self.action_label, "plain" if request.ok else "bad")
            if request.what.startswith("acceptance") or request.what.startswith("stability"):
                self.acceptance_message.setText(self.action_label.text())
                _colour(self.acceptance_message, "plain" if request.ok else "bad")
        if snap.discovering:
            self.discover_label.setText("Probing the address above with status requests only...")
        elif snap.discovery is not None:
            found = snap.discovery
            tag = "  [SIMULATED]" if snap.simulated else ""
            if found.found:
                known = "known to this app" if found.known_firmware else "NOT known to this app"
                self.discover_label.setText(f"Probe: {found.endpoint.label} answered. Firmware {found.firmware_id} ({known}); "
                                            f"protocol {'OK' if found.protocol_ok else 'not confirmed'}: {found.protocol_note}.{tag}")
            else:
                self.discover_label.setText(f"Probe: no usable answer from {found.endpoint.label}. {found.error or ''} "
                                            f"What to try: {found.hint or '-'}{tag}")

    # ================================================================ controls

    def _typed_ok(self) -> bool:
        return self.confirm_edit.text().strip() == ACCEPTANCE_PHRASE

    def _validate(self) -> None:
        endpoint, why = validate_endpoint(self.host_edit.text(), self.port_edit.text())
        self._endpoint_valid = endpoint is not None
        if endpoint is None:
            self.endpoint_message.setText(why)
            _colour(self.endpoint_message, "bad")
        else:
            self.endpoint_message.setText(f"{endpoint.label} is a usable address.")
            _colour(self.endpoint_message, "off")

    def _on_edited(self, _text: str = "") -> None:
        self._dirty = True
        self._validate()
        self._update_controls()

    def _update_controls(self) -> None:
        service, snap = self.service, self._snap
        has = service is not None and snap is not None
        state = snap.health.state if snap is not None else LinkState.DISCONNECTED
        open_ = state in OPEN_STATES
        discovering = bool(snap is not None and snap.discovering)
        valid = self._endpoint_valid

        def setup(button: QPushButton, enabled: bool, why: str) -> None:
            button.setEnabled(enabled)
            button.setToolTip("" if enabled else "Disabled: " + why)

        none_why = "there is no link service in this window"
        setup(self.apply_button, has and not open_ and valid, none_why if not has else "disconnect first" if open_ else "the address is not valid")
        setup(self.connect_button, has and not open_ and valid, none_why if not has else "already connected or connecting" if open_
              else "the address is not valid")
        setup(self.disconnect_button, has and open_, none_why if not has else "the link is not open")
        setup(self.reconnect_button, has and state in (LinkState.CONNECTED, LinkState.DEGRADED, LinkState.FAILED),
              none_why if not has else "there is nothing to reconnect: use Connect")
        setup(self.discover_button, has and not open_ and valid and not discovering,
              none_why if not has else "disconnect first (a second socket would take the status stream)" if open_ else
              "a probe is running" if discovering else "the address is not valid")
        self.host_edit.setEnabled(has and not open_)
        self.port_edit.setEnabled(has and not open_)
        setup(self.manual_button, service is not None, none_why)

        typed = self._typed_ok()
        self.confirm_state.setText("confirmed" if typed else "not typed")
        _colour(self.confirm_state, "ok" if typed else "off")
        why_typed = "type the confirmation sentence"
        setup(self.report_button, has and typed, none_why if not has else why_typed)
        connected = state is LinkState.CONNECTED
        setup(self.wifi_record_button, has and typed and connected, none_why if not has else why_typed if not typed else
              f"the link is {state.value}, not CONNECTED")
        running = bool(snap is not None and snap.stability.state == "RUNNING")
        setup(self.stability_button, has and typed and connected and not running,
              none_why if not has else why_typed if not typed else f"the link is {state.value}, not CONNECTED" if not connected
              else "a window is already running")
        self.report_status.setEnabled(has)
        selected = self.report_step.currentData()
        for edit in self.reading_edits.values():
            edit.setVisible(bool(selected and selected[0] == "baseline"))

    # ================================================================ actions (every one only sends a request)

    def _fields(self) -> tuple[str, str]:
        return self.host_edit.text(), self.port_edit.text()

    def apply_endpoint(self) -> None:
        if self.service is None:
            return
        ok, why = self.service.configure_endpoint(*self._fields())
        if ok:
            self._dirty = False
            self.endpoint_message.setText("Address applied" + (f" ({why})" if why else "") + ". It is remembered for next time.")
            _colour(self.endpoint_message, "ok")
        else:
            self.endpoint_message.setText(why)
            _colour(self.endpoint_message, "bad")

    def press_connect(self) -> None:
        if self.service is None or not self._endpoint_valid:
            return
        self._dirty = False
        self.service.connect(*self._fields())
        self.endpoint_message.setText("Connecting: status requests only.")
        _colour(self.endpoint_message, "off")

    def press_disconnect(self) -> None:
        if self.service is not None:
            self.service.disconnect()

    def press_reconnect(self) -> None:
        if self.service is not None:
            self.service.reconnect()

    def press_discover(self) -> None:
        if self.service is not None and self._endpoint_valid:
            self.service.discover()
            self.discover_label.setText("Probing the address above with status requests only...")

    def press_return_to_manual(self) -> None:
        if self.service is not None:
            self.service.return_to_manual()
            self.action_label.setText("Return to MANUAL requested: neutral is sent if the link is connected.")
            _colour(self.action_label, "plain")

    def press_record_wifi(self) -> None:
        if self.service is not None and self._typed_ok():
            self.service.record_wifi_established(operator_confirmed=True)

    def press_stability(self) -> None:
        if self.service is not None and self._typed_ok():
            self.service.start_stability_window(operator_confirmed=True)

    def press_report(self) -> None:
        if self.service is None or not self._typed_ok():
            return
        step, check = self.report_step.currentData()
        data: dict[str, float] = {}
        if step == "baseline":
            for key, edit in self.reading_edits.items():
                text = edit.text().strip()
                if not text:
                    continue
                try:
                    data[key] = float(text)
                except ValueError:
                    self.acceptance_message.setText(f"{key}: {text!r} is not a number. Nothing was recorded.")
                    _colour(self.acceptance_message, "bad")
                    return
        self.service.record_acceptance(step, self.report_status.currentText(), Evidence.OPERATOR_REPORTED, check=check,
                                       notes=self.report_notes.text().strip(), data=data,
                                       produced_by="operator (typed in the Command Centre)", operator_confirmed=True)

    def diagnostics_text(self) -> str:
        events = self.service.event_log() if self.service is not None else ()
        return diagnostics_text(self._snap, self.readiness(), events)

    def copy_diagnostics(self) -> str:
        text = self.diagnostics_text()
        QApplication.clipboard().setText(text)
        self.copy_label.setText(f"Copied {len(text.splitlines())} lines to the clipboard.")
        return text

    # ================================================================ lifecycle

    def shutdown(self) -> None:
        """Detach this screen: stop its timer and disconnect from the service. Safe to call twice. The service itself is
        owned by the dashboard, which calls ``service.shutdown()``."""
        if self._closed:
            return
        self._closed = True
        self._timer.stop()
        service = self.service
        if service is not None:
            for signal, slot in ((service.snapshot_changed, self._on_snapshot), (service.event_added, self._on_event)):
                try:
                    signal.disconnect(slot)
                except (RuntimeError, TypeError):
                    pass

    def closeEvent(self, event) -> None:
        self.shutdown()
        super().closeEvent(event)
