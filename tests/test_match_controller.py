"""Start / Stop Round logic, prerequisites, faults and consecutive rounds. Fake mode manager and clock; software only.

Nothing here is hardware verification: the 'all prerequisites met' inputs are scripted booleans, not measurements.
"""
from __future__ import annotations

import dataclasses

import pytest

from control_center.backend.control_modes import Confirmation, Mode, ModeManager
from control_center.backend.match_controller import MatchController, MatchInputs, missing, prerequisites
from control_center.backend.session_log import SessionLog, read_events


class FakeModes:
    """The part of ModeManager the controller uses. ``autonomous_ok`` scripts the gate and confirmation outcome."""

    def __init__(self, autonomous_ok=True):
        self.mode = Mode.MANUAL
        self.autonomous_ok = autonomous_ok
        self.calls: list[tuple] = []
        self.submitted: list = []

    def request(self, mode, now=None, confirmation=None):
        self.calls.append(("request", mode))
        if mode is Mode.AUTONOMOUS:
            if self.mode is not Mode.AI_SHADOW:
                return False, ["AUTONOMOUS cannot be entered directly from MANUAL"]
            if not self.autonomous_ok or confirmation is None:
                return False, ["confirmation missing" if confirmation is None else "gate closed"]
        self.mode = mode
        return True, []

    def submit(self, decision, now=None, **kw):
        self.submitted.append((decision, kw))


ALL = MatchInputs(camera_ok=True, arena_trusted=True, own_valid=True, opponent_valid=True, observation_complete=True, model_loaded=True,
                  mapping_validated=True, wifi_ready=True, authority_ok=True, safety_done=True, manual_available=True)
CONFIRM = Confirmation("ENABLE AUTONOMOUS", True, True, True, 0.0)


class Clock:
    t = 1000.0

    def __call__(self):
        return self.t


def controller(modes=None, **kw):
    clock = Clock()
    return MatchController(modes or FakeModes(), clock=clock, **kw), clock


def test_a_round_never_starts_unless_the_operator_presses_start():
    mc, _ = controller()
    result = mc.start_round(Mode.MANUAL, ALL)                    # camera connected, opponent selected, everything ready: still nothing
    assert not result.started and "presses Start Round" in result.message and not mc.running
    mc.tick(ALL)
    assert not mc.running and mc.round_number == 0


def test_manual_is_always_allowed_and_transmits_nothing():
    mc, _ = controller()
    assert mc.start_round(Mode.MANUAL, MatchInputs(), operator_action=True).started and mc.mode is Mode.MANUAL
    assert mc.modes.submitted == []
    assert not mc.feed_decision({"tactic": "advance"}, ALL) or mc.modes.submitted            # in MANUAL nothing reaches a sender


def test_ai_shadow_needs_camera_and_model_and_nothing_else():
    mc, _ = controller()
    refused = mc.start_round(Mode.AI_SHADOW, MatchInputs(camera_ok=True), operator_action=True)
    assert not refused.started and any("Policy loaded" in r for r in refused.reasons)
    result = mc.start_round(Mode.AI_SHADOW, MatchInputs(camera_ok=True, model_loaded=True), operator_action=True)
    assert result.started and result.mode is Mode.AI_SHADOW and not result.downgraded


def test_autonomous_with_a_missing_prerequisite_runs_in_shadow_instead_with_the_reasons():
    mc, _ = controller()
    inputs = dataclasses.replace(ALL, safety_done=False, safety_detail="physical safety checks not completed", opponent_valid=False)
    result = mc.start_round(Mode.AUTONOMOUS, inputs, confirmation=CONFIRM, operator_action=True)
    assert result.started and result.downgraded and result.mode is Mode.AI_SHADOW and mc.mode is Mode.AI_SHADOW
    assert any("safety checks" in r for r in result.reasons) and any("Opponent" in r for r in result.reasons)
    assert "AUTONOMOUS was refused" in result.message
    assert (Mode.AUTONOMOUS) not in [c[1] for c in mc.modes.calls]       # never even asked the manager


def test_autonomous_needs_the_managers_gate_and_the_confirmation_too():
    mc, _ = controller(FakeModes(autonomous_ok=False))
    result = mc.start_round(Mode.AUTONOMOUS, ALL, confirmation=CONFIRM, operator_action=True)
    assert result.downgraded and mc.mode is Mode.AI_SHADOW and "gate closed" in " ".join(result.reasons)
    mc.stop_round()
    mc2, _ = controller()
    result = mc2.start_round(Mode.AUTONOMOUS, ALL, confirmation=None, operator_action=True)
    assert result.downgraded and "confirmation missing" in " ".join(result.reasons)
    mc3, _ = controller()
    result = mc3.start_round(Mode.AUTONOMOUS, ALL, confirmation=CONFIRM, operator_action=True)
    assert result.started and not result.downgraded and mc3.mode is Mode.AUTONOMOUS            # only with every input scripted true


def test_a_failure_during_autonomous_stops_it_at_once_and_needs_a_new_confirmation():
    modes = FakeModes()
    mc, clock = controller(modes)
    mc.start_round(Mode.AUTONOMOUS, ALL, confirmation=CONFIRM, operator_action=True)
    assert mc.mode is Mode.AUTONOMOUS
    for field_name, text in (("own_valid", "own robot"), ("opponent_valid", "Opponent"), ("observation_complete", "Observation"),
                             ("camera_ok", "Camera"), ("wifi_ready", "Wi-Fi"), ("model_loaded", "Policy")):
        modes.mode = Mode.AUTONOMOUS
        mc.needs_reconfirm = False
        found = mc.tick(dataclasses.replace(ALL, **{field_name: False}), clock.t)
        assert found and text in found[0] and mc.mode is Mode.AI_SHADOW and mc.needs_reconfirm, field_name
    modes.mode = Mode.AUTONOMOUS
    mc.tick(dataclasses.replace(ALL, inference_error="shape mismatch"), clock.t)
    assert mc.mode is Mode.AI_SHADOW and "Policy" in mc.faults[-1]
    # recovery is deliberate: the inputs being fine again does not put it back
    mc.tick(ALL, clock.t)
    assert mc.mode is Mode.AI_SHADOW and mc.needs_reconfirm
    ok, _ = mc.set_mode(Mode.AUTONOMOUS, ALL, clock.t, CONFIRM)
    assert ok and mc.mode is Mode.AUTONOMOUS and not mc.needs_reconfirm


def test_no_decision_is_passed_on_after_an_inference_error_or_outside_a_round():
    modes = FakeModes()
    mc, clock = controller(modes)
    assert not mc.feed_decision({"tactic": "advance"}, ALL)                       # no round
    mc.start_round(Mode.AI_SHADOW, ALL, operator_action=True)
    assert mc.feed_decision({"tactic": "advance"}, ALL, observation_at=clock.t) and len(modes.submitted) == 1
    assert not mc.feed_decision({"tactic": "advance"}, dataclasses.replace(ALL, inference_error="boom"))
    assert not mc.feed_decision(None, ALL) and len(modes.submitted) == 1


def test_stop_round_is_neutral_manual_and_clears_the_opponent_for_the_next_round(tmp_path):
    ended, started = [], []
    modes = FakeModes()
    log = SessionLog(tmp_path)
    mc, clock = controller(modes, session_log=log, on_round_end=lambda: ended.append(1), on_round_start=started.append)
    mc.start_round(Mode.AI_SHADOW, ALL, operator_action=True)
    clock.t += 12.5
    assert mc.elapsed(clock.t) == pytest.approx(12.5)
    mc.stop_round("operator pressed Stop Round", clock.t)
    assert modes.mode is Mode.MANUAL and not mc.running and ended == [1] and mc.elapsed(clock.t + 30) == pytest.approx(12.5)
    assert ("request", Mode.MANUAL) in modes.calls
    mc.start_round(Mode.AI_SHADOW, ALL, operator_action=True)
    mc.stop_round("again", clock.t)
    assert started == [1, 2] and ended == [1, 1] and mc.round_number == 2
    log.close()
    assert [e["event"] for e in read_events(log.folder / "round-001.jsonl")] == ["start", "stop"]
    assert [e["event"] for e in read_events(log.folder / "round-002.jsonl")] == ["start", "stop"]


def test_a_second_start_while_running_is_refused_and_a_stop_while_idle_is_safe():
    mc, _ = controller()
    mc.start_round(Mode.AI_SHADOW, ALL, operator_action=True)
    assert not mc.start_round(Mode.AI_SHADOW, ALL, operator_action=True).started
    mc.stop_round()
    mc.stop_round()                                           # idempotent
    assert mc.modes.mode is Mode.MANUAL


def test_application_exit_during_a_round_stops_it(tmp_path):
    log = SessionLog(tmp_path)
    mc, _ = controller(session_log=log)
    mc.start_round(Mode.AI_SHADOW, ALL, operator_action=True)
    mc.shutdown()
    assert not mc.running and mc.modes.mode is Mode.MANUAL and "application closed" in mc.last_stop_reason


def test_round_time_limit_is_optional_and_never_assumed():
    mc, clock = controller()
    mc.start_round(Mode.AI_SHADOW, ALL, operator_action=True)
    clock.t += 10_000
    mc.tick(ALL, clock.t)
    assert mc.running                                          # no limit configured: it only counts up
    limited, clock2 = controller(duration_s=120)
    limited.start_round(Mode.AI_SHADOW, ALL, operator_action=True)
    clock2.t += 121
    limited.tick(ALL, clock2.t)
    assert not limited.running and "round time" in limited.last_stop_reason


def test_prerequisite_table_lists_every_item_the_operator_named():
    names = [p.name for p in prerequisites(MatchInputs())]
    for expected in ("Camera operational", "Policy loaded", "LIVE arena calibration", "Own robot tracking", "Opponent selected",
                     "Observation complete", "Command mapping", "Wi-Fi transport", "Control authority", "Physical safety checks", "Manual recovery"):
        assert any(expected in n for n in names), expected
    assert len(missing(MatchInputs(), Mode.AUTONOMOUS)) == len(names) and len(missing(MatchInputs(), Mode.AI_SHADOW)) == 2


def test_with_the_real_mode_manager_autonomous_is_refused_in_the_current_repository_state():
    """No Wi-Fi link, no acceptance record: the real gate is closed whatever the scripted inputs say."""
    manager = ModeManager(None, None)
    mc = MatchController(manager)
    result = mc.start_round(Mode.AUTONOMOUS, ALL, confirmation=CONFIRM, operator_action=True)
    assert result.started and result.downgraded and manager.mode is Mode.AI_SHADOW
    assert any("gate closed" in r for r in result.reasons)
    mc.stop_round()
    assert manager.mode is Mode.MANUAL
