"""The authority strip: which mode is selected, who has authority, and whether anything non-zero is being sent.

One small reusable widget, shown on the CONNECTIONS tab and on the Match tab. It only displays a ``ServiceSnapshot`` (see
``control_center/backend/link_service.py``); it has no buttons, opens nothing and sends nothing. Call ``update_from(snapshot)``
(10 Hz is plenty). ``snapshot`` may be None: the strip then says it has no data.

The words match the ELRS & CONTROL tab: MANUAL is the RadioMaster, AI SHADOW is the non-actuating display, AUTONOMOUS is
blocked until the hardware acceptance record exists. Physical disarm always wins.
"""

from __future__ import annotations

from typing import Any

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QVBoxLayout, QWidget

COLOURS = {"ok": "#55c889", "warn": "#e8b85b", "bad": "#ef6b73", "off": "#9aa5b1", "plain": "#d9dee5"}

MODE_LABELS = {"MANUAL": "MANUAL", "AI_SHADOW": "AI SHADOW", "AUTONOMOUS": "AUTONOMOUS"}
MODE_MEANING = {
    "MANUAL": "the RadioMaster drives; this program sends nothing",
    "AI_SHADOW": "the AI's proposals are shown and logged; only zero commands can be sent",
    "AUTONOMOUS": "the AI drives; every gate must be open and the operator must confirm",
}
DISARM_NOTE = ("Physical disarm always wins: ARM switch off, MODE to RADIO or radio off on the RadioMaster takes authority "
               "away from the AI at once.")
SOFTWARE_STOP_NOTE = ("A software stop travels over this Wi-Fi link, so it cannot work while Wi-Fi is down. The robot's own "
                      "protection is the controller firmware, which stops the motors 250 ms after the last command, and the "
                      "RadioMaster (ARM switch off, or MODE to RADIO).")
AUTHORITY_LEVEL = {"RADIO": "ok", "AI": "warn", "NONE": "off"}


def _label(text: str = "", wrap: bool = True) -> QLabel:
    label = QLabel(text)
    label.setTextFormat(Qt.TextFormat.PlainText)
    label.setWordWrap(wrap)
    label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
    return label


class AuthorityStrip(QFrame):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("authorityStrip")
        self.setStyleSheet("QFrame#authorityStrip { background: #171e27; border: 1px solid #2b3542; border-radius: 5px; } "
                           "QLabel { background: transparent; color: #d9dee5; }")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(6)
        chips = QHBoxLayout()
        self.mode_chips: dict[str, QLabel] = {}
        for key in MODE_LABELS:
            chip = _label("", wrap=False)
            chip.setAlignment(Qt.AlignmentFlag.AlignCenter)
            chip.setMargin(5)
            chips.addWidget(chip, 1)
            self.mode_chips[key] = chip
        layout.addLayout(chips)
        self.mode_label = _label("")
        layout.addWidget(self.mode_label)
        self.authority_label = _label("")
        self.authority_label.setStyleSheet("font-weight: 700;")
        layout.addWidget(self.authority_label)
        self.neutral_label = _label("")
        layout.addWidget(self.neutral_label)
        self.sim_label = _label("SIMULATED: this is a mock or fake link. Nothing shown here comes from the robot.")
        self.sim_label.setStyleSheet(f"color: {COLOURS['warn']}; font-weight: 700;")
        layout.addWidget(self.sim_label)
        self.disarm_label = _label(DISARM_NOTE)
        self.disarm_label.setStyleSheet(f"color: {COLOURS['off']};")
        layout.addWidget(self.disarm_label)
        self.stop_label = _label(SOFTWARE_STOP_NOTE)
        self.stop_label.setStyleSheet(f"color: {COLOURS['off']};")
        layout.addWidget(self.stop_label)
        self.update_from(None)

    def update_from(self, snapshot: Any) -> None:
        if snapshot is None:
            for key, chip in self.mode_chips.items():
                chip.setText(MODE_LABELS[key])
                chip.setStyleSheet(f"color: {COLOURS['off']}; border: 1px solid #3b4a5d; border-radius: 4px;")
            self.mode_label.setText("No mode information yet: the link service has not reported.")
            self.authority_label.setText("Authority: unknown")
            self.authority_label.setStyleSheet(f"font-weight: 700; color: {COLOURS['off']};")
            self.neutral_label.setText("Neutral status: unknown")
            self.sim_label.setVisible(False)
            return
        mode = snapshot.mode
        current = mode.mode.value
        for key, chip in self.mode_chips.items():
            text = MODE_LABELS[key]
            if key == current:
                colour = {"MANUAL": COLOURS["ok"], "AI_SHADOW": COLOURS["warn"], "AUTONOMOUS": COLOURS["bad"]}[key]
                chip.setText(f"{text}  (selected)")
                chip.setStyleSheet(f"color: white; background: {colour}; border-radius: 4px; font-weight: 800;")
            else:
                blocked = key == "AUTONOMOUS" and not mode.gate_open
                chip.setText(f"{text}" + ("  (blocked)" if blocked else ""))
                chip.setStyleSheet(f"color: {COLOURS['off']}; border: 1px solid #3b4a5d; border-radius: 4px;")
        self.mode_label.setText(f"{MODE_LABELS.get(current, current)}: {MODE_MEANING.get(current, '')}")
        authority = mode.authority
        self.authority_label.setText(f"Authority: {authority.authority.value}. {authority.reason}")
        self.authority_label.setStyleSheet(f"font-weight: 700; color: {COLOURS[AUTHORITY_LEVEL.get(authority.authority.value, 'plain')]};")
        if snapshot.transmitting:
            self.neutral_label.setText("NOT neutral: a non-zero drive command was sent in the last 1.5 s.")
            self.neutral_label.setStyleSheet(f"font-weight: 700; color: {COLOURS['bad']};")
        else:
            hold = " Neutral hold is ON: zero commands are streamed so the controller can accept PC commands." if mode.neutral_hold else ""
            self.neutral_label.setText("Neutral: nothing non-zero is being sent." + hold)
            self.neutral_label.setStyleSheet(f"font-weight: 700; color: {COLOURS['ok']};")
        self.sim_label.setVisible(bool(snapshot.simulated))
