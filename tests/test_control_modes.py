"""Modes, the autonomous gate and authority (control_center/backend/control_modes.py).

SOFTWARE TESTS ONLY. The link is a fake transport on a fake clock (``link.real`` is set True only to exercise the gates),
and the acceptance records are SYNTHETIC entries in a temporary file. None of this is physical verification: the gate is
closed in the repository's real state (no hardware record, no IMU, no real link), and the first test says so.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tests"))

import test_acceptance as ta  # noqa: E402  (synthetic records helper)
import test_wifi_link as tw  # noqa: E402  (fake port and clock)

from control_center.backend import control_modes as cm  # noqa: E402
from control_center.backend.acceptance import Acceptance  # noqa: E402
from control_center.backend.ai_observation import IMU_ABSENT_REASON  # noqa: E402
from control_center.backend.control_modes import (CONFIRM_PHRASE, Authority, Confirmation, Mode, ModeManager)  # noqa: E402
from control_center.backend.wifi_link import LinkState, WifiLink  # noqa: E402


def entry(valid=True, age_ms=20.0, reason="ok", source="stream"):
    return SimpleNamespace(valid=valid, age_ms=age_ms, reason=reason, source=source)


def make_decision(tactic="advance", imu=None, own=None, opponent=None):
    inputs = {"imu": imu or entry(source="controller IMU"),
              "previous_action": entry(age_ms=None, source="masked"),
              "own_pose": own or entry(age_ms=30.0, source="ArUco"),
              "opponent_position": opponent or entry(age_ms=30.0, source="blob")}
    complete = all(e.valid and (e.age_ms is None or e.age_ms <= 250.0) for e in inputs.values())
    return SimpleNamespace(inputs=inputs, observation=[0.0] * 19 if complete else None, blocked_by=[],
                           proposed={"tactic": tactic, "probability": 0.9} if tactic else None)


class ModeRig:
    """A fake link, a temporary acceptance record and a manager, all on one fake clock."""

    def __init__(self, tmp_path, *, records=True, real=True, mock_feed=False, ai=True) -> None:
        self.r = tw.Rig()
        self.clock = self.r.clock
        self.link = self.r.link
        self.link.real = real
        self.acc = Acceptance(tmp_path / "wifi_acceptance.json", unix_clock=lambda: ta.NOW,
                              motion_check_path=tmp_path / "none.json")
        if records:
            ta.pass_everything(self.acc)
        self.mgr = ModeManager(self.link, self.acc, clock=self.clock, unix_clock=lambda: ta.NOW, sketch_hash=ta.SKETCH,
                               capabilities=ta.CAPS, build_allows_autonomous=True)    # the build lock is tested on its own
        self.decision = make_decision()
        self.ai = ai
        self.mock_feed = mock_feed
        self.uptime = 5000

    # -- time

    def step(self, seconds: float = 0.02, status: bool = True, **fields) -> None:
        self.uptime += int(seconds * 1000)
        now = self.clock.advance(seconds)
        if status:
            values = dict(t=self.uptime)
            values.update(fields)
            self.r.port.feed(tw.st(**values))
        if self.ai is True:
            self.mgr.submit(self.decision, now, policy_loaded=True, mock=self.mock_feed)
        elif self.ai == "decision_only":                 # the decision stays fresh but the AI loop's heartbeat stops
            self.mgr.feed.decision, self.mgr.feed.decided_at = self.decision, now
        self.mgr.tick(now)

    def run(self, seconds: float, **fields) -> None:
        for _ in range(int(round(seconds / 0.02))):
            self.step(0.02, **fields)

    # -- scenarios

    def connect(self, **fields) -> None:
        assert self.link.connect(self.clock())
        values = dict(armed=1, req="PC", act="NONE", pc_ok=0, block="WAIT_PC_NEUTRAL")
        values.update(fields)
        self.r.port.feed(tw.HELLO, tw.st(t=self.uptime, **values))
        self.mgr.tick(self.clock.advance(0.05))
        assert self.link.state is LinkState.CONNECTED

    def shadow_ready(self) -> "ModeRig":
        """Armed, MODE=PC, link up; the operator holds neutral; the controller accepts PC commands; gate open."""
        self.connect()
        assert self.mgr.request(Mode.AI_SHADOW) == (True, [])
        self.run(1.2, armed=1, req="PC", act="NONE", pc_ok=0, block="WAIT_PC_NEUTRAL")
        ok, why = self.mgr.set_neutral_hold(True)
        assert ok, why
        self.run(0.4, act="PC", pc_ok=1, block="NONE")
        open_, items = self.mgr.gate()
        assert open_, [(i.name, i.reason) for i in items if not i.ok]
        return self

    def confirmation(self, **overrides) -> Confirmation:
        values = dict(phrase=CONFIRM_PHRASE, wheels_clear=True, radiomaster_in_hand=True, understands_stops=True,
                      created_at=self.clock())
        values.update(overrides)
        return Confirmation(**values)

    def autonomous(self) -> "ModeRig":
        self.shadow_ready()
        self.last_confirmation = self.confirmation()
        ok, why = self.mgr.request(Mode.AUTONOMOUS, confirmation=self.last_confirmation)
        assert ok, why
        self.run(0.2, act="PC", pc_ok=1, block="NONE")
        return self

    def drives(self) -> list[bytes]:
        return self.r.port.drives()

    def last_drive_fields(self) -> tuple[int, int]:
        parts = self.drives()[-1].decode().strip().split(",")
        return int(parts[2]), int(parts[3])


@pytest.fixture
def rig(tmp_path):
    return ModeRig(tmp_path)


# ---------------------------------------------------------------- the repository's real state

def test_the_gate_is_closed_in_the_repositorys_current_state(tmp_path):
    link = WifiLink(clock=tw.FakeClock())                   # never connected: nothing is opened until connect()
    mgr = ModeManager(link, Acceptance(tmp_path / "wifi_acceptance.json", motion_check_path=tmp_path / "none.json"),
                      clock=link._clock)
    open_, items = mgr.gate(1000.0)
    assert open_ is False
    failing = {item.name: item.reason for item in items if not item.ok}
    assert {"acceptance", "link", "controller", "radio_override", "physical_disarm", "ai_process", "tracking", "imu",
            "observation", "policy", "build"} <= set(failing)
    assert "AUTONOMOUS_ENABLEABLE_IN_THIS_BUILD" in failing["build"]
    assert failing["imu"] == IMU_ABSENT_REASON == "the baseline firmware sends no IMU data: nothing is invented"
    assert all(item.reason for item in items) and {i.kind for i in items} <= {"HARDWARE", "LIVE", "SOFTWARE", "SESSION"}
    assert mgr.mode is Mode.MANUAL and mgr.authority(1000.0).authority is Authority.NONE
    assert mgr.request(Mode.AUTONOMOUS, 1000.0)[0] is False
    mgr.request(Mode.AI_SHADOW, 1000.0)
    ok, reasons = mgr.request(Mode.AUTONOMOUS, 1000.0, Confirmation(CONFIRM_PHRASE, True, True, True, 1000.0))
    assert not ok and reasons[0] == "gate closed" and any("acceptance" in r for r in reasons)
    assert link.drive_command_ever_sent is False


def test_the_default_acceptance_record_in_the_repository_cannot_open_the_gate():
    link = WifiLink(clock=tw.FakeClock())
    mgr = ModeManager(link, Acceptance(), clock=link._clock)          # the real config/wifi_acceptance.json, read only
    assert mgr.gate(1000.0)[0] is False


# ---------------------------------------------------------------- the gate, item by item

def test_with_everything_in_order_the_gate_opens(rig):
    rig.shadow_ready()
    open_, items = rig.mgr.gate()
    assert open_ and all(item.ok for item in items)
    assert {item.name for item in items} == {"real_session", "acceptance", "link", "controller", "radio_override",
                                              "physical_disarm", "ai_process", "tracking", "imu", "observation", "policy",
                                              "build"}


def _close_real(rig):
    rig.link.real = False


def _close_acceptance(rig):
    rig.acc._results.pop("supervised")


def _close_link(rig):
    rig.link.state = LinkState.DEGRADED


def _close_controller(rig):
    rig.step(armed=0, act="NONE", pc_ok=0, block="DISARMED")


def _close_radio_override(rig):
    rig.acc._results.pop("transitions.takeover")


def _close_disarm(rig):
    rig.acc._results.pop("transitions.arm_off")


def _close_ai(rig):
    rig.ai = "decision_only"                             # fresh data, but the AI loop itself has stopped reporting
    rig.run(0.6, act="PC", pc_ok=1, block="NONE")


def _close_tracking(rig):
    rig.decision = make_decision(opponent=entry(valid=False, reason="no unmarked blob selected"))
    rig.step(act="PC", pc_ok=1, block="NONE")


def _close_imu(rig):
    rig.decision = make_decision(imu=entry(valid=False, age_ms=None, reason="no IMU telemetry (NoTelemetry)", source="none"))
    rig.step(act="PC", pc_ok=1, block="NONE")


def _close_observation(rig):
    rig.decision = make_decision(own=entry(age_ms=300.0))
    rig.step(act="PC", pc_ok=1, block="NONE")


def _close_policy(rig):
    rig.mgr.feed.policy_loaded = False
    rig.ai = False
    rig.mgr.ai_heartbeat(rig.clock())


def _close_build(rig):
    rig.mgr._build_override = None                          # follow control_status again: False in this repository's state


@pytest.mark.parametrize("name, close", [("real_session", _close_real), ("acceptance", _close_acceptance),
                                         ("link", _close_link), ("controller", _close_controller),
                                         ("radio_override", _close_radio_override), ("physical_disarm", _close_disarm),
                                         ("ai_process", _close_ai), ("tracking", _close_tracking), ("imu", _close_imu),
                                         ("observation", _close_observation), ("policy", _close_policy),
                                         ("build", _close_build)])
def test_closing_any_single_gate_closes_the_whole_gate_with_a_reason(tmp_path, name, close):
    rig = ModeRig(tmp_path).shadow_ready()
    close(rig)
    open_, items = rig.mgr.gate(rig.clock())
    failing = {item.name: item.reason for item in items if not item.ok}
    assert open_ is False and name in failing and failing[name], (name, failing)
    ok, reasons = rig.mgr.request(Mode.AUTONOMOUS, confirmation=rig.confirmation())
    assert not ok and any(name in reason for reason in reasons)
    assert rig.mgr.mode is Mode.AI_SHADOW
    assert rig.drives() and all(d.split(b",")[2:4] == [b"0", b"0"] for d in rig.drives())      # only zeros, ever


def test_mock_sessions_never_open_the_gate(tmp_path):
    for kwargs in (dict(real=False), dict(mock_feed=True)):
        rig = ModeRig(tmp_path / ("a" if "real" in kwargs else "b"), **kwargs)
        rig.connect()
        rig.mgr.request(Mode.AI_SHADOW)
        rig.run(1.5, act="PC", pc_ok=1, block="NONE")
        open_, items = rig.mgr.gate()
        failing = {item.name for item in items if not item.ok}
        assert not open_ and "real_session" in failing, kwargs
        assert rig.mgr.request(Mode.AUTONOMOUS, confirmation=rig.confirmation())[0] is False
    rig = ModeRig(tmp_path / "c")
    rig.mgr.mock_session = True                                       # the flag alone must close it
    rig.connect()
    rig.mgr.request(Mode.AI_SHADOW)
    rig.run(1.5, act="PC", pc_ok=1, block="NONE")
    assert "real_session" in {i.name for i in rig.mgr.gate()[1] if not i.ok}


# ---------------------------------------------------------------- mode transitions and the confirmation

def test_default_is_manual_and_manual_cannot_jump_to_autonomous(rig):
    assert rig.mgr.mode is Mode.MANUAL
    assert rig.mgr.authority(rig.clock()).authority is Authority.NONE
    ok, reasons = rig.mgr.request(Mode.AUTONOMOUS, confirmation=rig.confirmation())
    assert not ok and "AI_SHADOW first" in reasons[0] and rig.mgr.mode is Mode.MANUAL
    assert rig.mgr.request(Mode.MANUAL) == (True, [])
    assert rig.mgr.request("AUTONOMOUS")[0] is False


def test_autonomous_needs_the_typed_phrase_three_ticks_and_a_fresh_single_use_confirmation(rig):
    rig.shadow_ready()
    now = rig.clock()
    cases = [(None, "confirmation is required"), (rig.confirmation(phrase="enable autonomous"), "typed phrase"),
             (rig.confirmation(phrase=""), "typed phrase"), (rig.confirmation(wheels_clear=False), "wheels"),
             (rig.confirmation(radiomaster_in_hand=False), "RadioMaster"), (rig.confirmation(understands_stops=False), "stops"),
             (rig.confirmation(created_at=now - 31.0), "30 s"), (rig.confirmation(created_at=now + 60.0), "future")]
    for confirmation, expected in cases:
        ok, reasons = rig.mgr.request(Mode.AUTONOMOUS, confirmation=confirmation)
        assert not ok and any(expected in r for r in reasons), (expected, reasons)
        assert rig.mgr.mode is Mode.AI_SHADOW
    assert rig.mgr.request(Mode.AUTONOMOUS, confirmation=rig.confirmation(created_at=now - 29.0)) == (True, [])
    assert rig.mgr.mode is Mode.AUTONOMOUS


def test_a_confirmation_is_single_use(rig):
    rig.autonomous()
    rig.mgr.request(Mode.AI_SHADOW)
    rig.run(0.3, act="PC", pc_ok=1, block="NONE")
    ok, reasons = rig.mgr.request(Mode.AUTONOMOUS, confirmation=rig.last_confirmation)     # the one that was just used
    assert not ok and any("already used" in r for r in reasons) and rig.mgr.mode is Mode.AI_SHADOW


def test_leaving_autonomous_sends_neutral_first_and_reentry_needs_a_new_confirmation(tmp_path):
    rig = ModeRig(tmp_path).autonomous()
    assert rig.mgr.mode is Mode.AUTONOMOUS and rig.last_drive_fields() != (0, 0)
    rig.mgr.request(Mode.AI_SHADOW)
    assert rig.last_drive_fields() == (0, 0) and rig.mgr.mode is Mode.AI_SHADOW
    rig.run(0.3, act="PC", pc_ok=1, block="NONE")
    ok, reasons = rig.mgr.request(Mode.AUTONOMOUS, confirmation=rig.last_confirmation)
    assert not ok and any("already used" in r for r in reasons)
    ok, why = rig.mgr.request(Mode.AUTONOMOUS, confirmation=rig.confirmation())
    assert ok, why
    rig.mgr.request(Mode.MANUAL)
    assert rig.last_drive_fields() == (0, 0) and rig.mgr.mode is Mode.MANUAL


def test_autonomous_sends_capped_commands_with_rising_sequence_numbers(tmp_path):
    rig = ModeRig(tmp_path).autonomous()
    rig.run(0.2, act="PC", pc_ok=1, block="NONE")
    assert rig.last_drive_fields() == (60, 0)                        # the runtime mapping (command_mapping.py): advance peaks at max_command 60
    plan = rig.mgr.plans[-1]
    assert plan.sent and (plan.intended_throttle, plan.intended_steer) == (60, 0)
    assert (plan.transmitted_throttle, plan.transmitted_steer) == (60, 0)
    numbers = [int(d.split(b",")[4]) for d in rig.drives()]
    assert numbers == sorted(set(numbers))
    assert rig.mgr.authority(rig.clock()).authority is Authority.AI
    for tactic, expected in (("turn_left", (0, -42)), ("reverse", (-39, 0)), ("stop", (0, 0))):
        rig.decision = make_decision(tactic)
        rig.run(0.5, act="PC", pc_ok=1, block="NONE")                 # slew limiting and neutral-first on a reversal take a few ticks
        assert rig.last_drive_fields() == expected, tactic
    rig.decision = make_decision("forward_left")                      # not one of the policy's five tactics: nothing is sent for it
    rig.run(0.1, act="PC", pc_ok=1, block="NONE")
    assert rig.mgr.plans[-1].sent is False and "could not be mapped" in rig.mgr.plans[-1].why_not


def test_a_decision_that_is_no_longer_refreshed_ages_out_and_is_never_replayed(tmp_path):
    rig = ModeRig(tmp_path).autonomous()
    rig.ai = False                                                    # the AI loop stops: the last decision just sits there
    before = len(rig.drives())
    rig.run(0.6, act="PC", pc_ok=1, block="NONE")
    commands = rig.drives()[before:]
    moving = [c for c in commands if c.split(b",")[2:4] != [b"0", b"0"]]
    assert len(moving) <= 5                                           # the IMU (100 ms) goes stale first: at most ~5 ticks
    assert rig.mgr.mode is Mode.AI_SHADOW and "stale" in rig.mgr.last_event
    assert rig.last_drive_fields() == (0, 0)
    assert rig.mgr.authority(rig.clock()).authority is Authority.NONE


# ---------------------------------------------------------------- shadow and manual never transmit

def test_manual_and_shadow_plans_are_zero_and_not_sent(rig):
    rig.connect()
    assert rig.mgr.mode is Mode.MANUAL
    rig.run(0.5, act="PC", pc_ok=1, block="NONE")
    assert rig.drives() == []                                         # MANUAL sends nothing at all
    plan = rig.mgr.compute_command(make_decision("advance"), rig.clock())
    assert (plan.intended_throttle, plan.intended_steer) == (60, 0)
    assert (plan.transmitted_throttle, plan.transmitted_steer, plan.sent) == (0, 0, False) and "MANUAL" in plan.why_not
    rig.mgr.request(Mode.AI_SHADOW)
    rig.run(0.5, act="PC", pc_ok=1, block="NONE")
    assert rig.drives() == []                                         # SHADOW without the operator's neutral hold: nothing
    plan = rig.mgr.compute_command(make_decision("turn_right"), rig.clock())
    assert (plan.transmitted_throttle, plan.transmitted_steer, plan.sent) == (0, 0, False) and "AI_SHADOW" in plan.why_not
    assert plan.intended_steer != 0
    assert all(p.transmitted_throttle == 0 and p.transmitted_steer == 0 and not p.sent for p in rig.mgr.plans)
    assert rig.mgr.plans                                              # displayed and logged


def test_shadow_with_the_neutral_hold_sends_only_zero_commands(tmp_path):
    rig = ModeRig(tmp_path)
    rig.connect()
    assert rig.mgr.set_neutral_hold(True)[0] is False                 # only in AI_SHADOW
    rig.mgr.request(Mode.AI_SHADOW)
    rig.run(1.2, armed=1, req="PC", act="NONE", pc_ok=0, block="WAIT_PC_NEUTRAL")
    assert rig.mgr.set_neutral_hold(True) == (True, [])
    rig.decision = make_decision("advance")
    rig.run(1.0, act="PC", pc_ok=1, block="NONE")
    assert rig.drives() and all(d.split(b",")[2:4] == [b"0", b"0"] for d in rig.drives())
    assert all(not p.sent for p in rig.mgr.plans)
    rig.step(armed=0, act="NONE", pc_ok=0, block="DISARMED")           # disarm: the hold switches itself off
    assert rig.mgr.neutral_hold is False
    count = len(rig.drives())
    rig.run(0.3, armed=0, act="NONE", pc_ok=0, block="DISARMED")
    assert len(rig.drives()) == count


def test_the_link_refuses_drive_commands_from_outside_the_manager(rig):
    rig.shadow_ready()
    now = rig.clock()
    result = rig.link.send_drive(30, 0, now, now)
    assert result.sent is False and result.code == "NOT_ALLOWED"


def test_the_plan_log_is_bounded_and_deduplicated(tmp_path):
    rig = ModeRig(tmp_path)
    rig.mgr._plans = type(rig.mgr._plans)(maxlen=5)
    for _ in range(300):
        rig.mgr.compute_command(make_decision("advance"), rig.clock.advance(0.02))
    assert len(rig.mgr.plans) <= 5
    log = rig.mgr.plans
    assert len({(p.tactic, p.mode, p.why_not) for p in log}) == 1 and log[-1].t > log[0].t


# ---------------------------------------------------------------- physical authority wins

def test_authority_values_follow_the_controller(rig):
    rig.autonomous()
    assert rig.mgr.authority(rig.clock()).authority is Authority.AI
    rig.step(req="RADIO", act="RADIO", pc_ok=0, block="NONE")
    state = rig.mgr.authority(rig.clock())
    assert state.authority is Authority.RADIO and "RadioMaster" in state.reason


@pytest.mark.parametrize("fields, word", [
    (dict(armed=0, act="NONE", pc_ok=0, block="DISARMED"), "disarm"),
    (dict(req="RADIO", act="RADIO", pc_ok=0, block="NONE"), "took over"),
    (dict(req="RADIO", act="NONE", pc_ok=0, block="WAIT_RADIO_NEUTRAL"), "took over"),
    (dict(link=0, act="NONE", pc_ok=0, block="RADIO_LOST", fault="RADIO_LOST", stop="RADIO_LOST"), "radio"),
    (dict(fault="PC_STALE", act="NONE", pc_ok=0, block="REARM_REQUIRED", stop="PC_STALE"), "fault"),
    (dict(tok=99), "token"),
])
def test_physical_disarm_mode_change_radio_loss_and_faults_override_the_ai_at_once(tmp_path, fields, word):
    rig = ModeRig(tmp_path).autonomous()
    assert rig.mgr.mode is Mode.AUTONOMOUS
    drives_before = len(rig.drives())
    rig.step(**fields)
    assert rig.mgr.mode is Mode.MANUAL, rig.mgr.last_event
    assert word.lower() in rig.mgr.last_event.lower(), rig.mgr.last_event
    assert rig.mgr.authority(rig.clock()).authority is not Authority.AI
    new = rig.drives()[drives_before:]
    assert new and new[-1].split(b",")[2:4] == [b"0", b"0"]           # the last thing sent is neutral
    assert not any(d.split(b",")[2:4] != [b"0", b"0"] for d in new[1:]), "nothing non-zero after the change"
    # it stays dropped: the controller coming back to a perfect state does not resume anything
    rig.run(2.0, armed=1, req="PC", act="PC", pc_ok=1, block="NONE", fault="NONE", tok=fields.get("tok", 7))
    assert rig.mgr.mode is Mode.MANUAL and rig.last_drive_fields() == (0, 0)
    assert rig.link.needs_rearm
    ok, reasons = rig.mgr.request(Mode.AUTONOMOUS, confirmation=rig.confirmation())
    assert not ok                                                       # not from MANUAL, and never without a new confirmation


def test_after_a_drop_reentry_is_shadow_then_a_fresh_confirmation(tmp_path):
    rig = ModeRig(tmp_path).autonomous()
    old = rig.last_confirmation
    rig.step(armed=0, act="NONE", pc_ok=0, block="DISARMED")
    assert rig.mgr.mode is Mode.MANUAL
    # the operator cycles ARM on the RadioMaster: a new token, the controller waits for a neutral
    rig.run(0.5, armed=1, req="PC", act="NONE", pc_ok=0, block="WAIT_PC_NEUTRAL", tok=9)
    assert rig.mgr.mode is Mode.MANUAL
    assert rig.mgr.request(Mode.AI_SHADOW) == (True, [])
    rig.run(1.2, armed=1, req="PC", act="NONE", pc_ok=0, block="WAIT_PC_NEUTRAL", tok=9)
    assert rig.mgr.set_neutral_hold(True)[0]
    rig.run(1.2, act="PC", pc_ok=1, block="NONE", tok=9)
    assert rig.mgr.gate()[0]
    assert rig.mgr.request(Mode.AUTONOMOUS, confirmation=old)[0] is False          # the earlier confirmation is spent
    ok, why = rig.mgr.request(Mode.AUTONOMOUS, confirmation=rig.confirmation())
    assert ok, why
    rig.run(0.2, tok=9, act="PC", pc_ok=1, block="NONE")
    assert rig.mgr.authority(rig.clock()).authority is Authority.AI


def test_losing_the_link_invalidates_authority_locally_with_no_help_from_wifi(tmp_path):
    rig = ModeRig(tmp_path).autonomous()
    rig.link.disconnect(rig.clock())                                    # no tick needed: it is invalid at once
    state = rig.mgr.authority(rig.clock())
    assert state.authority is Authority.NONE and "no controller report" in state.reason
    writes = len(rig.r.port.written)
    rig.mgr.tick(rig.clock.advance(0.02))
    assert rig.mgr.mode is Mode.MANUAL and "controller report is gone" in rig.mgr.last_event
    assert len(rig.r.port.written) == writes                            # there was no link to send a stop on
    assert rig.mgr.emergency_stop(rig.clock()) is None                  # a software stop cannot reach a closed link; it must not raise


def test_a_silent_link_drops_authority_then_fails_and_a_stop_over_a_dead_link_is_harmless(tmp_path):
    rig = ModeRig(tmp_path).autonomous()
    for _ in range(35):                                                 # 0.7 s with no ST line
        rig.step(0.02, status=False)
    assert rig.mgr.mode is Mode.MANUAL and "report is" in rig.mgr.last_event
    assert rig.mgr.authority(rig.clock()).authority is Authority.NONE
    rig.r.port.fail_writes = True                                       # Wi-Fi is gone: the stop cannot be delivered
    rig.mgr.emergency_stop(rig.clock.advance(0.1))
    for _ in range(10):
        rig.step(0.5, status=False)
    assert rig.link.state is LinkState.FAILED
    assert rig.mgr.authority(rig.clock()).authority is Authority.NONE and rig.mgr.mode is Mode.MANUAL


# ---------------------------------------------------------------- failures on the AI side

@pytest.mark.parametrize("name, close, most_moving", [("ai_process", _close_ai, 26), ("tracking", _close_tracking, 1),
                                                      ("imu", _close_imu, 1), ("observation", _close_observation, 1),
                                                      ("policy", _close_policy, 1)])
def test_ai_loop_tracking_observation_and_policy_failures_end_autonomous_with_neutral(tmp_path, name, close, most_moving):
    rig = ModeRig(tmp_path).autonomous()
    assert rig.mgr.mode is Mode.AUTONOMOUS
    before = len(rig.drives())
    close(rig)
    rig.run(0.1, act="PC", pc_ok=1, block="NONE")
    assert rig.mgr.mode is Mode.AI_SHADOW and name in rig.mgr.last_event, (rig.mgr.mode, rig.mgr.last_event)
    assert rig.mgr.authority(rig.clock()).authority is Authority.NONE
    after = rig.drives()[before:]
    assert after and after[-1].split(b",")[2:4] == [b"0", b"0"]               # neutral, immediately
    moving = [d for d in after if d.split(b",")[2:4] != [b"0", b"0"]]
    assert len(moving) <= most_moving                                         # nothing is held (the AI heartbeat limit is 0.5 s = 25 ticks)
    count = len(rig.drives())
    rig.run(0.3, act="PC", pc_ok=1, block="NONE")
    assert all(d.split(b",")[2:4] == [b"0", b"0"] for d in rig.drives()[count:])
    ok, reasons = rig.mgr.request(Mode.AUTONOMOUS, confirmation=rig.confirmation())
    assert not ok and reasons[0] == "gate closed"                              # and it needs the gate AND a new confirmation


def test_acceptance_going_stale_mid_run_ends_autonomous(tmp_path):
    clock_unix = {"t": ta.NOW}
    rig = ModeRig(tmp_path)
    rig.mgr._unix = lambda: clock_unix["t"]
    rig.autonomous()
    clock_unix["t"] = ta.NOW + 13 * 3600
    rig.run(0.1, act="PC", pc_ok=1, block="NONE")
    assert rig.mgr.mode is not Mode.AUTONOMOUS and rig.last_drive_fields() == (0, 0)


# ---------------------------------------------------------------- shutdown

def test_shutdown_sends_neutral_sets_manual_and_is_idempotent(tmp_path):
    rig = ModeRig(tmp_path).autonomous()
    count = len(rig.drives())
    rig.mgr.shutdown()
    assert rig.mgr.mode is Mode.MANUAL and len(rig.drives()) == count + 1 and rig.last_drive_fields() == (0, 0)
    rig.mgr.shutdown()
    rig.mgr.shutdown()
    assert len(rig.drives()) == count + 1
    assert rig.mgr.request(Mode.AI_SHADOW)[0] is False
    rig.mgr.tick(rig.clock.advance(0.02))
    assert len(rig.drives()) == count + 1
    assert rig.link.send_drive(30, 0, rig.clock(), rig.clock()).sent is False


def test_emergency_stop_returns_to_manual_with_a_neutral(tmp_path):
    rig = ModeRig(tmp_path).autonomous()
    rig.mgr.emergency_stop(rig.clock.advance(0.02), "operator pressed STOP")
    assert rig.mgr.mode is Mode.MANUAL and rig.last_drive_fields() == (0, 0)
    assert "STOP" in rig.mgr.last_event


def test_snapshot_is_complete_for_the_ui(tmp_path):
    rig = ModeRig(tmp_path).shadow_ready()
    snap = rig.mgr.snapshot(rig.clock())
    assert snap.mode is Mode.AI_SHADOW and snap.gate_open and snap.needs_confirmation and snap.neutral_hold
    assert "NOT physical verification" in snap.evidence_note and len(snap.gate) == 12
    assert snap.authority.authority is Authority.NONE
