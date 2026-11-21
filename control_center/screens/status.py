"""SYSTEM STATUS: the whole system on one page, and the one thing to do next.

A table of the critical components (camera, arena, own-robot tracking, opponent, Wi-Fi, manual ELRS / safety, AI,
autonomous drive), each PASS / WARNING / FAIL / NOT TESTED / BLOCKED with why and what to do, a "next action"
banner, a one-click "Run software checks" and "Copy report". Software checks only: nothing here touches the
robot, and a PASS is a software result unless the row says HARDWARE.

The rows are answered by the screens themselves (their ``readiness()``); this page never decides one.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import (QFrame, QGridLayout, QHBoxLayout, QLabel, QPushButton, QScrollArea, QVBoxLayout, QWidget)

from control_center.backend import system_status as ss
from control_center.backend.profiles import profile_label
from control_center.backend.readiness import CHIP_COLOURS, Readiness

log = logging.getLogger(__name__)
REFRESH_MS = 1000
STYLE = """
QLabel { background: transparent; }
QFrame#row { background: #171e27; border: 1px solid #2b3542; border-radius: 6px; }
QLabel#chip { font-weight: 800; padding: 3px 10px; border-radius: 4px; color: #11161d; }
QLabel#next { font-size: 13pt; font-weight: 800; }
QLabel#safe { font-weight: 800; padding: 6px 10px; border-radius: 4px; background: #1f3b2b; color: #9be3b4; }
QLabel#detail { color: #b9c3ce; }
QLabel#fix { color: #e8b85b; }
"""


def _label(text: str = "", name: str | None = None) -> QLabel:
    label = QLabel(text)
    label.setTextFormat(Qt.TextFormat.PlainText)
    label.setWordWrap(True)
    if name:
        label.setObjectName(name)
    return label


class StatusRow(QFrame):
    def __init__(self, component_key: str, title: str, on_open: Callable[[str], None]) -> None:
        super().__init__()
        self.key = component_key
        self.setObjectName("row")
        layout = QGridLayout(self)
        layout.setContentsMargins(12, 8, 12, 8)
        self.title = _label(title)
        self.title.setStyleSheet("font-weight: 800;")
        self.chip = QLabel("NOT TESTED")
        self.chip.setObjectName("chip")
        self.evidence = _label("")
        self.evidence.setStyleSheet("color: #7b8794;")
        self.headline = _label("")
        self.headline.setStyleSheet("font-weight: 700;")
        self.detail = _label("", "detail")
        self.fix = _label("", "fix")
        self.open_button = QPushButton("Open")
        self.open_button.clicked.connect(lambda: on_open(self.key))
        layout.addWidget(self.title, 0, 0)
        layout.addWidget(self.evidence, 0, 1, Qt.AlignmentFlag.AlignRight)
        layout.addWidget(self.chip, 0, 2, Qt.AlignmentFlag.AlignRight)
        layout.addWidget(self.headline, 1, 0, 1, 3)
        layout.addWidget(self.detail, 2, 0, 1, 3)
        layout.addWidget(self.fix, 3, 0, 1, 2)
        layout.addWidget(self.open_button, 3, 2, Qt.AlignmentFlag.AlignRight)
        layout.setColumnStretch(0, 1)

    def show_component(self, component: ss.Component) -> None:
        readiness = component.readiness
        self.chip.setText(component.word)
        self.chip.setStyleSheet(f"background: {CHIP_COLOURS[readiness.level]};")
        self.evidence.setText(f"a PASS here = {component.evidence} check")
        self.headline.setText(readiness.headline)
        self.detail.setText(readiness.detail)
        self.detail.setVisible(bool(readiness.detail))
        self.fix.setText(("Next: " + readiness.fix) if readiness.fix else "")
        self.fix.setVisible(bool(readiness.fix))
        self.open_button.setVisible(component.tab is not None)


class SystemStatusScreen(QWidget):
    def __init__(self, hub, navigate: Callable[[str], None] | None = None, providers: dict[str, Callable[[], Readiness | None]] | None = None,
                 safe_state: Callable[[], str] | None = None, parent=None) -> None:
        super().__init__(parent)
        self.hub = hub
        self.navigate = navigate or (lambda _tab: None)
        self.providers = providers if providers is not None else {}
        self.safe_state = safe_state
        self.components: list[ss.Component] = []
        self.setStyleSheet(STYLE)
        root = QVBoxLayout(self)
        root.setContentsMargins(12, 8, 12, 8)
        self.profile_label = _label("")
        self.profile_label.setStyleSheet("font-size: 12pt; font-weight: 800; color: #6fa8dc;")
        root.addWidget(self.profile_label)
        self.safe_label = _label("", "safe")
        root.addWidget(self.safe_label)
        self.summary = _label("", "next")
        root.addWidget(self.summary)
        self.next_label = _label("", "fix")
        root.addWidget(self.next_label)
        buttons = QHBoxLayout()
        self.check_button = QPushButton("Run software checks")
        self.check_button.setToolTip("Re-check the camera picture against the saved calibration and refresh every row. Software only.")
        self.copy_button = QPushButton("Copy report")
        self.check_button.clicked.connect(lambda _c=False: self.run_checks())
        self.copy_button.clicked.connect(lambda _c=False: self.copy_report())
        buttons.addWidget(self.check_button)
        buttons.addWidget(self.copy_button)
        buttons.addStretch(1)
        self.note = _label("A PASS is a SOFTWARE check unless the row says HARDWARE. Software tests are never hardware verification.", "detail")
        buttons.addWidget(self.note, 2)
        root.addLayout(buttons)
        body = QWidget()
        column = QVBoxLayout(body)
        column.setSpacing(6)
        self.rows: dict[str, StatusRow] = {}
        for key, title, _tab, _needed, _evidence in ss.ROWS:
            row = StatusRow(key, title, self.open_tab)
            self.rows[key] = row
            column.addWidget(row)
        column.addStretch(1)
        area = QScrollArea()
        area.setWidgetResizable(True)
        area.setWidget(body)
        area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        root.addWidget(area, 1)
        self._timer = QTimer(self)
        self._timer.setInterval(REFRESH_MS)
        self._timer.timeout.connect(self.refresh)
        self.refresh()

    # ---------------------------------------------------------------- actions
    def open_tab(self, key: str) -> None:
        tab = next((tab for k, _title, tab, _needed, _evidence in ss.ROWS if k == key), None)
        if tab:
            self.navigate(tab)

    def run_checks(self) -> None:
        """Re-check the calibration against the picture, then refresh every row. Reads only."""
        evaluate = getattr(self.hub, "evaluate_restore", None)
        if evaluate is not None:
            try:
                evaluate()
            except Exception:
                log.exception("restore re-check failed")
        self.refresh()

    def report(self) -> str:
        profile = getattr(self.hub, "profile_name", "live")
        return ss.report_text(self.components, profile_label(profile), self._safe_text())

    def copy_report(self) -> str:
        text = self.report()
        clipboard = QGuiApplication.clipboard()
        if clipboard is not None:
            clipboard.setText(text)
        return text

    def _safe_text(self) -> str:
        if self.safe_state is not None:
            try:
                return self.safe_state()
            except Exception:
                log.exception("safe-state text failed")
        return "SAFE STATE: MANUAL control. This application is not transmitting motor commands."

    # ---------------------------------------------------------------- refresh
    def refresh(self) -> None:
        profile = getattr(self.hub, "profile_name", "live")
        self.profile_label.setText(f"SYSTEM STATUS · {profile_label(profile)}")
        self.components = ss.build_components(self.providers)
        for component in self.components:
            self.rows[component.key].show_component(component)
        self.safe_label.setText(self._safe_text())
        overall = ss.round_summary(self.components)
        self.summary.setText(overall.headline)
        self.summary.setStyleSheet(f"font-size: 13pt; font-weight: 800; color: {CHIP_COLOURS[overall.level]};")
        self.next_label.setText("NEXT: " + overall.fix)

    def readiness(self) -> Readiness:
        return ss.round_summary(self.components or ss.build_components(self.providers))

    def showEvent(self, event) -> None:  # noqa: N802 - Qt API
        self._timer.start()
        self.refresh()
        super().showEvent(event)

    def hideEvent(self, event) -> None:  # noqa: N802 - Qt API
        self._timer.stop()
        super().hideEvent(event)

    def shutdown(self) -> None:
        self._timer.stop()
