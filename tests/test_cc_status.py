"""SYSTEM STATUS: the words, the ordering of what to fix next, and the page. Pure logic plus offscreen Qt; software only."""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("QT_QPA_FONTDIR", r"C:\Windows\Fonts")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pytest

from control_center.backend import system_status as ss
from control_center.backend.app_state import LEGACY_STEPS, STEP_IDS, AppState
from control_center.backend.readiness import BLOCKED, FAILED, LEVELS, NOT_CONFIGURED, READY, WARNING, Readiness, worst
from test_cc_guided import hub, qapp, sandbox  # noqa: E402,F401


def providers(**levels):
    base = {key: Readiness(READY, f"{key} fine") for key, *_ in ss.ROWS}
    base.update(levels)
    return {key: (lambda value=value: value) for key, value in base.items()}


def test_every_level_has_a_diagnostic_word_and_blocked_is_a_level():
    assert {ss.diag_word(level) for level in LEVELS} == {"PASS", "WARNING", "FAIL", "NOT TESTED", "BLOCKED"}
    assert worst([Readiness(READY, "a"), Readiness(BLOCKED, "b"), Readiness(WARNING, "c")]) == BLOCKED
    assert worst([Readiness(BLOCKED, "b"), Readiness(FAILED, "c")]) == FAILED


def test_a_missing_or_broken_provider_is_never_a_pass():
    rows = ss.build_components({"camera": lambda: (_ for _ in ()).throw(RuntimeError("boom"))})
    by_key = {c.key: c for c in rows}
    assert by_key["camera"].word == "FAIL" and "boom" in by_key["camera"].readiness.detail
    assert all(c.word == "NOT TESTED" for c in rows if c.key != "camera")


def test_the_next_action_is_the_first_unfinished_step_in_workflow_order():
    rows = ss.build_components(providers(arena=Readiness(NOT_CONFIGURED, "no arena", fix="Calibrate"),
                                         opponent=Readiness(NOT_CONFIGURED, "none", fix="Drag a box"),
                                         tracking=Readiness(FAILED, "lost", fix="Put the robot in view")))
    text, component = ss.next_action(rows)
    assert component.key == "tracking" and "Put the robot in view" in text            # a failure comes before a missing step
    rows = ss.build_components(providers(arena=Readiness(NOT_CONFIGURED, "no arena", fix="Calibrate"), opponent=Readiness(NOT_CONFIGURED, "none", fix="Drag a box")))
    assert ss.next_action(rows)[1].key == "arena"


def test_blocked_autonomy_does_not_stop_manual_or_shadow_and_is_not_called_ready():
    rows = ss.build_components(providers(autonomous=Readiness(BLOCKED, "blocked", "no hardware acceptance record"),
                                         connections=Readiness(BLOCKED, "Wi-Fi connected, motor transmission not verified")))
    text, component = ss.next_action(rows)
    assert "MANUAL" in text and "BLOCKED" in text and component.readiness.level == BLOCKED
    assert ss.round_summary(rows).level == READY and "MANUAL / AI SHADOW" in ss.round_summary(rows).headline
    report = ss.report_text(rows, "LIVE", "SAFE STATE: MANUAL")
    assert "never hardware verification" in report and "BLOCKED" in report and "SAFE STATE" in report


def test_opponent_readiness_for_each_state_and_nothing_is_kept_from_before():
    assert ss.opponent_readiness(None).level == NOT_CONFIGURED
    assert ss.opponent_readiness({"state": "not_selected"}).level == NOT_CONFIGURED
    assert ss.opponent_readiness({"state": "tracked", "confidence": 0.9}).level == READY
    assert "IMAGE-BASED" in ss.opponent_readiness({"state": "tracked", "confidence": 0.9}).detail
    assert ss.opponent_readiness({"state": "uncertain", "reason": "hidden"}).level == WARNING
    assert ss.opponent_readiness({"state": "lost", "reason": "gone"}).level == FAILED
    assert ss.opponent_readiness({"state": "tracked"}, calibrated=False).level == NOT_CONFIGURED


def test_older_progress_files_use_the_new_step_names(tmp_path):
    state = AppState(tmp_path / "s.json")
    state.data["profiles"]["live"] = {"wizard_step": "monitoring", "steps": {}, "arena_verified": None}
    assert state.wizard_step("live") == "operate"
    state.data["profiles"]["live"]["wizard_step"] = "control"
    assert state.wizard_step("live") == "safety" and set(LEGACY_STEPS.values()) <= set(STEP_IDS)


def test_the_page_shows_every_row_the_safe_state_and_the_next_action(hub):
    from control_center.screens.status import SystemStatusScreen
    visited = []
    screen = SystemStatusScreen(hub, navigate=visited.append, providers=providers(autonomous=Readiness(BLOCKED, "blocked", "no acceptance"),
                                                                                 ai=Readiness(WARNING, "model not loaded")))
    screen.show()
    screen.refresh()
    assert list(screen.rows) == [row[0] for row in ss.ROWS]
    assert screen.rows["autonomous"].chip.text() == "BLOCKED" and screen.rows["ai"].chip.text() == "WARNING"
    assert "SAFE STATE" in screen.safe_label.text() and screen.next_label.text().startswith("NEXT:")
    screen.rows["connections"].open_button.click()
    assert visited == ["connections"]
    text = screen.copy_report()
    assert "AI model" in text and "BLOCKED" in text
    assert "SOFTWARE" in screen.rows["camera"].evidence.text() and "HARDWARE" in screen.rows["autonomous"].evidence.text()


def test_the_page_survives_having_no_providers_and_never_says_ready(hub):
    from control_center.screens.status import SystemStatusScreen
    screen = SystemStatusScreen(hub)
    screen.show()
    screen.refresh()
    assert screen.readiness().level != READY and all(row.chip.text() == "NOT TESTED" for row in screen.rows.values())
