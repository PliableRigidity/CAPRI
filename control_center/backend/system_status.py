"""The whole system on one page: what is ready, what is not, and what to do next. Pure logic, no Qt.

Every row is answered by the code that checked it (a screen's ``readiness()``); this module only orders the rows,
maps the words and picks the ONE next thing to fix. A row can never be PASS because a guess says so.

Words: the chips elsewhere in the Command Centre say READY / NOT CONFIGURED / WARNING / FAILED / BLOCKED; the
readiness table says PASS / NOT TESTED / WARNING / FAIL / BLOCKED. They are the same five states.

Evidence classes (``evidence``): a PASS here is a SOFTWARE check (the camera delivers frames, the file loads, the
tracker sees markers in the picture) unless the row says HARDWARE (a physical test the operator ran and recorded).
Nothing in this module can turn a software result into hardware readiness.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Mapping

from control_center.backend.readiness import BLOCKED, FAILED, NOT_CONFIGURED, READY, WARNING, Readiness, worst

log = logging.getLogger(__name__)

DIAG_WORDS = {READY: "PASS", WARNING: "WARNING", FAILED: "FAIL", NOT_CONFIGURED: "NOT TESTED", BLOCKED: "BLOCKED"}

# key, title, tab that fixes it, whether it must be fine before a round (False = informational), evidence class of a PASS
ROWS: tuple[tuple[str, str, str | None, bool, str], ...] = (
    ("camera", "Camera", "guided", True, "SOFTWARE"),
    ("arena", "Arena calibration and alignment", "arena", True, "SOFTWARE"),
    ("tracking", "Own robot tracking (ArUco markers)", "tracking", True, "SOFTWARE"),
    ("opponent", "Opponent tracking (drag-box ROI)", "tracking", True, "SOFTWARE"),
    ("connections", "Wi-Fi connection to the robot", "connections", False, "SOFTWARE"),
    ("control", "Manual ELRS control and safety checks", "control", True, "HARDWARE"),
    ("ai", "AI model and observation", "ai", False, "SOFTWARE"),
    ("autonomous", "Autonomous drive (Wi-Fi motor commands)", "connections", False, "HARDWARE"),
)
TITLES = {key: title for key, title, _tab, _needed, _evidence in ROWS}


def diag_word(level: str) -> str:
    return DIAG_WORDS.get(level, level.upper())


@dataclass(frozen=True)
class Component:
    key: str
    title: str
    readiness: Readiness
    tab: str | None
    needed_for_round: bool
    evidence: str                     # what a PASS on this row means: SOFTWARE or HARDWARE

    @property
    def word(self) -> str:
        return diag_word(self.readiness.level)

    def line(self) -> str:
        parts = [f"{self.word:<10} {self.title}: {self.readiness.headline}"]
        if self.readiness.detail:
            parts.append(f"    {self.readiness.detail}")
        if self.readiness.fix:
            parts.append(f"    Next: {self.readiness.fix}")
        return "\n".join(parts)


def opponent_readiness(opponent: Mapping[str, Any] | None, calibrated: bool = True) -> Readiness:
    """From the opponent track dict of a ``TrackingResult`` (None = no fresh result)."""
    if not calibrated:
        return Readiness(NOT_CONFIGURED, "Arena not calibrated", "An opponent position needs the calibrated arena.", "Calibrate the arena first.")
    if opponent is None:
        return Readiness(NOT_CONFIGURED, "Opponent not selected", "No tracking result yet. The opponent is chosen again every round.",
                         "Open TRACKING and drag a box around the opponent.")
    state = opponent.get("state")
    if state == "tracked":
        confidence = opponent.get("confidence")
        extra = "" if confidence is None else f", confidence {confidence:.2f}"
        return Readiness(READY, "Opponent tracked", f"position is IMAGE-BASED and approximate{extra}")
    if state == "not_selected":
        return Readiness(NOT_CONFIGURED, "Opponent not selected", "Selection is never remembered between rounds.",
                         "Open TRACKING and drag a box around the opponent.")
    if state == "uncertain":
        return Readiness(WARNING, "Opponent tracking uncertain", f"no position is shown: {opponent.get('reason', '')}",
                         "Wait for it to recover or drag a new box.")
    if state == "lost":
        return Readiness(FAILED, "Opponent lost", str(opponent.get("reason", "")), "Drag a new box around the opponent.")
    return Readiness(WARNING, f"Opponent {str(state).replace('_', ' ')}", str(opponent.get("reason", "")), "Check the TRACKING tab.")


def build_components(providers: Mapping[str, Callable[[], Readiness | None]]) -> list[Component]:
    """Ask every provider; a missing or broken one is a visible NOT TESTED / FAIL row, never a silent PASS."""
    out: list[Component] = []
    for key, title, tab, needed, evidence in ROWS:
        provider = providers.get(key)
        if provider is None:
            result = Readiness(NOT_CONFIGURED, "Not available in this build", "No screen answers this check.", "See logs/ for the reason.")
        else:
            try:
                result = provider() or Readiness(NOT_CONFIGURED, "No status yet")
            except Exception as error:
                log.exception("status of %s failed", key)
                result = Readiness(FAILED, "The status check itself failed", f"{type(error).__name__}: {error}", "Open the tab; see logs/.")
        out.append(Component(key, title, result, tab, needed, evidence))
    return out


def next_action(components: list[Component]) -> tuple[str, Component | None]:
    """The ONE next thing: the first row, in workflow order, that must be fixed before a round (FAIL first, then
    NOT TESTED, then WARNING). Rows that are only informational never block this."""
    for wanted in (FAILED, NOT_CONFIGURED, WARNING):
        for component in components:
            if component.needed_for_round and component.readiness.level == wanted:
                fix = component.readiness.fix or component.readiness.headline
                return f"{component.title}: {fix}", component
    blocked = [c for c in components if c.readiness.level == BLOCKED]
    if blocked:
        return (f"Ready for MANUAL driving and AI SHADOW. {blocked[0].title} stays BLOCKED: {blocked[0].readiness.detail or blocked[0].readiness.headline}",
                blocked[0])
    return "Everything that can be checked in software is ready. Hardware checks are done by you, on the robot, step by step (see the playbook).", None


def round_summary(components: list[Component]) -> Readiness:
    needed = [c.readiness for c in components if c.needed_for_round]
    level = worst(needed) if needed else NOT_CONFIGURED
    text, _ = next_action(components)
    headline = {READY: "Ready for a round (MANUAL / AI SHADOW)", WARNING: "Usable, with warnings", FAILED: "Not ready: something failed",
                NOT_CONFIGURED: "Not ready: setup incomplete", BLOCKED: "Blocked"}[level]
    return Readiness(level, headline, "", text)


def report_text(components: list[Component], profile: str, safe_state: str = "") -> str:
    """Plain text of the whole table, for 'Copy report' and the session log."""
    lines = [f"Capri system readiness (profile {profile})"]
    if safe_state:
        lines.append(safe_state)
    lines.append("PASS here means a SOFTWARE check passed unless the row says HARDWARE; software is never hardware verification.")
    lines += [component.line() for component in components]
    lines.append("NEXT: " + next_action(components)[0])
    return "\n".join(lines)
