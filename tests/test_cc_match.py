"""MATCH screen (control_center/screens/match.py): mode, Start / Stop Round, prerequisites, status board, faults, session log.

Offscreen Qt, a MOCK hub that is never started, a fake policy engine, a scripted tracking screen, a fake link service with a SCRIPTED
gate (and, for the real mode manager, the fake-transport rig of tests/test_link_service.py). No camera, socket or serial port is
opened, nothing is flashed or sent, and every folder is inside tmp_path (nothing is written under config/ or logs/).
EVERYTHING below is software, simulated or scripted: it says nothing about the real robot, and the 'all prerequisites met' states are
scripted booleans, not measurements.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("QT_QPA_FONTDIR", r"C:\Windows\Fonts")

import pytest
from PySide6.QtWidgets import QApplication, QLabel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tests"))

import match_rig as rig  # noqa: E402
import test_link_service as tls  # noqa: E402  (the fake-transport rig for the REAL mode manager)

from control_center.backend import match_support as ms  # noqa: E402
from control_center.backend import policy_engine as pe  # noqa: E402
from control_center.backend import session_log  # noqa: E402
from control_center.backend import wifi_link as wl  # noqa: E402
from control_center.backend.control_modes import CONFIRM_PHRASE, Confirmation, Mode  # noqa: E402
from control_center.backend.match_controller import prerequisites  # noqa: E402
from control_center.backend.readiness import BLOCKED, NOT_CONFIGURED, READY, WARNING  # noqa: E402
from control_center.backend.session_log import read_events  # noqa: E402
from control_center.screens.ai_model import AiModelScreen  # noqa: E402
from control_center.screens.match import AutonomousConfirmationDialog, MatchScreen  # noqa: E402

ADVANCE_60 = "60 / 0"


@pytest.fixture(autouse=True)
def safe_environment(monkeypatch, tmp_path):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("a real transport was requested in a test")
    monkeypatch.setattr(wl.client, "UdpPort", forbidden)
    monkeypatch.setattr(wl, "default_transport_factory", forbidden)
    default_logs = tmp_path / "must-stay-unused-default-logs"
    monkeypatch.setattr(session_log, "DEFAULT_ROOT", default_logs)
    rig.forget_loops()
    yield
    rig.forget_loops()
    assert not default_logs.exists()


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


class World:
    """A hub, a manual clock, a scripted tracking screen, a fake engine and service, and the two screens under test."""

    def __init__(self, tmp_path: Path) -> None:
        self.tmp_path = tmp_path
        self.hub = rig.make_hub(tmp_path, session_log=True)
        self.clock = rig.Clock()
        self.tracking = rig.FakeTracking(self.clock)
        self.engine = rig.FakeEngine()
        self.service = rig.FakeService(self.clock)
        self.confirmations = 0
        self.screens: list = []
        self.ai = self.make_ai()
        self.match = self.make_match(self.ai, self.service)

    def confirm(self) -> Confirmation:
        self.confirmations += 1
        return Confirmation(CONFIRM_PHRASE, True, True, True, created_at=self.clock())

    def make_ai(self, engine=None) -> AiModelScreen:
        chosen = engine or self.engine
        screen = AiModelScreen(self.hub, tracking=self.tracking, engine_factory=lambda _p: chosen, clock=self.clock, start_timer=False,
                               autoload=False)
        self.screens.append(screen)
        return screen

    def make_match(self, ai, service, **kwargs) -> MatchScreen:
        kwargs.setdefault("confirm", self.confirm)
        screen = MatchScreen(self.hub, tracking=self.tracking, ai=ai, service=service, clock=self.clock, start_timer=False, **kwargs)
        self.screens.append(screen)
        return screen

    # ---- scripted states
    def frame(self, state: str | None = None) -> None:
        rig.publish_frame(self.hub, state)

    def model(self) -> None:
        assert self.ai.load_model() or self.ai.model_loaded()
        assert rig.wait_for(self.ai.model_loaded)

    def shadow_ready(self, imu: bool = False) -> None:
        """A loaded model and a camera that delivers frames: what AI SHADOW needs. ``imu`` scripts a SIMULATED IMU (the MOCK hub's replay)."""
        self.model()
        self.frame("MOCK")
        self.tracking.telemetry.flow = imu

    def full_live(self) -> None:
        """SCRIPT every prerequisite and the gate true, as if a live session had all of them. A scripted state, not a measurement."""
        self.model()
        self.hub.options.mock = False
        self.frame("LIVE")
        self.hub.restore_report = SimpleNamespace(world_coordinates_trusted=True, headline="scripted")
        self.hub.monitor = rig.FakeMonitor(up=True)
        self.tracking.telemetry.flow = True
        self.tracking.set(own="measured", opponent="tracked")
        self.service.gate_open = True
        self.service.simulated = False
        assert self.ai.check_mapping().passed

    def press(self, mode: Mode) -> None:
        self.match.mode_buttons[mode].click()

    def start(self, mode: Mode = Mode.AI_SHADOW):
        self.frame()
        self.press(mode)
        before = self.match.controller.round_number
        self.match.start_button.click()
        return self.match.controller.round_number > before

    def tick(self, seconds: float = 0.05, count: int = 1) -> list[str]:
        found: list[str] = []
        for _ in range(count):
            self.frame()
            self.clock.advance(seconds)
            found += self.match.tick()
        return found

    def texts(self) -> str:
        return "\n".join(item.text() for item in self.match.findChildren(QLabel))

    def board(self) -> dict[str, str]:
        self.match.paint_all()
        return dict(self.match.board_texts())

    def log_events(self, name: str = "session.jsonl") -> list[dict]:
        return list(read_events(self.hub.session_log.folder / name))

    def close(self) -> None:
        for screen in reversed(self.screens):
            screen.shutdown()
        self.hub.close()


@pytest.fixture
def world(qapp, tmp_path):
    made = World(tmp_path)
    yield made
    made.close()


# ---------------------------------------------------------------- the safe starting state

def test_it_starts_in_manual_transmitting_nothing_with_stop_always_available(world):
    match = world.match
    assert not match.controller.running and match.controller.round_number == 0
    assert world.service.calls == [] and world.service.mode is Mode.MANUAL and world.service.decisions == []
    assert match.mode_buttons[Mode.MANUAL].isChecked() and not match.mode_buttons[Mode.AI_SHADOW].isChecked()
    assert match.start_button.isEnabled() and match.stop_button.isEnabled() and match.emergency_button.isEnabled()
    assert world.engine.active_loop is None
    assert "NO ROUND RUNNING" in match.round_label.text() and "nothing is transmitted" in match.round_label.text()


def test_the_software_stop_cannot_work_without_wifi_statement_is_on_the_tab(world):
    text = world.texts()
    assert "software stop cannot work when Wi-Fi is down" in text
    assert "250 ms" in text and "ARM switch" in text and "RadioMaster" in text


def test_the_round_timer_counts_up_and_keeps_its_value_after_stop(world):
    world.shadow_ready()
    assert world.start()
    world.tick(12.5)
    assert world.match.timer_label.text() == "00:12.5"
    world.match.stop_button.click()
    world.clock.advance(30)
    world.match.tick()
    assert world.match.timer_label.text() == "00:12.5"


# ---------------------------------------------------------------- a round starts only from the button

def test_a_round_never_starts_by_itself_selecting_a_mode_connecting_the_camera_or_selecting_an_opponent(world):
    match = world.match
    world.shadow_ready()
    refused = match.start_round()                                       # no operator action: the controller refuses
    assert not refused.started and "presses Start Round" in refused.message and match.controller.round_number == 0
    world.press(Mode.AI_SHADOW)                                          # selecting a mode starts nothing
    assert not match.controller.running and world.service.calls == []
    world.tracking.set(opponent="not_selected")
    world.frame("LIVE")                                                  # the camera connects
    world.tick(count=5)
    world.tracking.set(opponent="tracked")                               # an opponent is selected
    world.tick(count=5)
    rig.process(0.1)
    assert not match.controller.running and match.controller.round_number == 0 and world.service.calls == []
    assert world.engine.active_loop is None and world.service.mode is Mode.MANUAL and world.service.decisions == []
    match.start_button.click()                                           # only the button starts one
    assert match.controller.running and match.controller.round_number == 1 and match.controller.mode is Mode.AI_SHADOW
    assert "Round started in AI_SHADOW" in match.message_label.text()


def test_a_restarted_screen_does_not_resume_or_start_a_round(world):
    world.shadow_ready()
    assert world.start()
    world.match.shutdown()
    again = world.make_match(world.ai, rig.FakeService(world.clock))
    world.tick()
    again.tick()
    assert not again.controller.running and again.controller.round_number == 0 and again.service.calls == []
    assert world.engine.active_loop is None


# ---------------------------------------------------------------- readiness

def test_match_readiness_levels_and_autonomous_stays_blocked_with_the_exact_gate_reasons(world):
    match = world.match
    assert match.readiness().level == NOT_CONFIGURED                    # no model
    world.model()
    assert match.readiness().level == WARNING and "Camera" in match.readiness().headline    # no camera frames yet
    world.frame("MOCK")
    ready = match.readiness()
    assert ready.level == READY and "MANUAL and AI SHADOW" in ready.headline
    autonomous = match.autonomous_readiness()
    assert autonomous.level == BLOCKED and not autonomous.ok
    for item in world.service.gate_items():
        assert f"{item.name}: {item.reason}" in autonomous.detail                # the gate's exact reasons, every one
    assert match.readiness().level == READY                              # unaffected: MANUAL / AI SHADOW rounds can start


def test_autonomous_readiness_is_ready_only_with_the_gate_open_and_every_prerequisite_scripted_true(world):
    match = world.match
    world.full_live()
    assert match.autonomous_readiness().level == READY and match.autonomous_readiness().ok
    assert not [p for p in prerequisites(match.assemble_inputs()) if not p.ok]
    for change, word in ((lambda: world.tracking.set(opponent="lost"), "Opponent"), (lambda: world.tracking.set(own="lost"), "Own robot"),
                         (lambda: setattr(world.tracking.telemetry, "flow", False), "Observation")):
        world.full_live()
        change()
        result = match.autonomous_readiness()
        assert result.level == WARNING and word in result.detail, word
    world.full_live()
    world.service.gate_open = False
    assert match.autonomous_readiness().level == BLOCKED
    world.full_live()
    world.hub.options.mock = True                                          # a MOCK / SIMULATED camera can never complete the LIVE observation
    world.frame("MOCK")
    assert match.autonomous_readiness().level == WARNING


def test_the_prerequisite_table_lists_every_item_with_pass_or_fail_and_details(world):
    match = world.match
    match.paint_all()
    rows = match.prerequisite_texts()
    assert [r[0] for r in rows] == [p.name for p in prerequisites(match.assemble_inputs())] and len(rows) == 11
    by_name = {r[0]: r for r in rows}
    assert by_name["Policy loaded"][1] == "FAIL" and "not loaded" in by_name["Policy loaded"][3]
    assert by_name["Command mapping validated"][1] == "FAIL" and "Check mapping" in by_name["Command mapping validated"][3]
    assert by_name["Control authority established"][3].startswith("gate closed:") and "acceptance" in by_name["Control authority established"][3]
    assert by_name["Opponent selected and tracked"][1] == "PASS"
    world.shadow_ready()
    world.ai.check_mapping()
    match.paint_all()
    by_name = {r[0]: r for r in match.prerequisite_texts()}
    assert by_name["Policy loaded"][1] == "PASS" and by_name["Command mapping validated"][1] == "PASS" and by_name["Camera operational"][1] == "PASS"
    assert by_name["Observation complete and fresh"][1] == "FAIL" and "no IMU" in by_name["Observation complete and fresh"][3]
    assert by_name["Physical safety checks completed"][1] == "FAIL"


# ---------------------------------------------------------------- AUTONOMOUS: refused, with reasons, AI SHADOW instead

def test_autonomous_is_refused_with_the_gate_reasons_and_start_round_falls_back_to_ai_shadow(world):
    match = world.match
    world.shadow_ready()
    world.confirm = None                                                   # (the dialog must never be opened for a closed gate)

    def must_not_open():
        raise AssertionError("the confirmation dialog opened although the gate is closed")

    match._confirm_hook = must_not_open
    ok, why = match.select_mode(Mode.AUTONOMOUS)
    assert not ok and "NOT available" in why and "Start Round will start AI SHADOW instead" in why
    assert "AUTONOMOUS: UNAVAILABLE" in match.gate_label.text()
    for item in world.service.gate_items():
        assert f"{item.name}: {item.reason}" in match.gate_reasons.text()
    assert not match.controller.running and world.service.calls == []
    assert "AI SHADOW instead" in match.start_button.toolTip()
    world.frame()
    match.start_button.click()
    assert match.controller.running and match.controller.mode is Mode.AI_SHADOW and match.controller.downgraded
    assert "AUTONOMOUS was refused" in match.message_label.text() and "AI_SHADOW" in match.message_label.text()
    assert ("mode", Mode.AUTONOMOUS, None) not in [c[:3] for c in world.service.calls]
    assert all(c[1] is not Mode.AUTONOMOUS for c in world.service.calls if c[0] == "mode")
    start = [e for e in world.log_events("round-001.jsonl") if e["event"] == "start"][0]
    assert start["mode"] == "AI_SHADOW" and start["requested"] == "AUTONOMOUS" and start["downgraded"] is True and start["reasons"]


def test_with_the_gate_open_the_typed_confirmation_is_asked_and_a_missing_prerequisite_still_means_ai_shadow(world):
    match = world.match
    world.shadow_ready()
    world.service.gate_open = True                                         # scripted: the gate opens, but wifi / safety / IMU still fail
    ok, why = match.select_mode(Mode.AUTONOMOUS)
    assert ok and world.confirmations == 1 and "Nothing has started" in why and not match.controller.running
    world.frame()
    match.start_button.click()
    assert match.controller.running and match.controller.downgraded and match.controller.mode is Mode.AI_SHADOW
    assert "AUTONOMOUS was refused" in match.message_label.text()
    assert world.confirmations == 1                                         # the held, still-valid confirmation was reused


def test_the_confirmation_dialog_needs_the_exact_phrase_and_all_three_ticks(qapp, world):
    dialog = AutonomousConfirmationDialog(None, world.clock)
    ok = dialog.buttons.button(dialog.buttons.StandardButton.Ok)
    assert not ok.isEnabled()
    dialog.phrase.setText("enable autonomous")
    for box in dialog.ticks:
        box.setChecked(True)
    assert not ok.isEnabled()
    dialog.phrase.setText(CONFIRM_PHRASE)
    assert ok.isEnabled()
    dialog.ticks[1].setChecked(False)
    assert not ok.isEnabled()
    dialog.ticks[1].setChecked(True)
    confirmation = dialog.confirmation()
    assert confirmation.phrase == CONFIRM_PHRASE and confirmation.wheels_clear and confirmation.radiomaster_in_hand
    assert confirmation.understands_stops and confirmation.created_at == world.clock()


# ---------------------------------------------------------------- Stop, Emergency STOP, consecutive rounds

def test_stop_always_works_ends_the_round_in_manual_and_clears_the_opponent(world):
    match = world.match
    world.shadow_ready()
    match.stop_button.click()                                               # idle: safe, and it still asks for MANUAL
    assert ("manual",) in world.service.calls and world.tracking.cleared == 0
    assert world.start()
    assert match.controller.running and world.engine.active_loop is not None and match.stop_button.isEnabled()
    world.tick(count=3)
    assert world.tracking.opponent_state == "tracked"
    match.stop_button.click()
    assert not match.controller.running and world.service.mode is Mode.MANUAL
    assert world.tracking.cleared == 1 and world.tracking.opponent_state == "not_selected"      # a new opponent every round
    assert world.engine.active_loop is None and not world.ai.loop_active
    assert "stopped" in match.message_label.text() and "Nothing is transmitted" in match.message_label.text()
    assert match.mode_buttons[Mode.MANUAL].isChecked()
    before = world.engine.decide_calls
    world.tick(count=3)
    assert world.engine.decide_calls == before and world.service.mode is Mode.MANUAL         # no inference outside a round


def test_emergency_stop_goes_through_the_service_ends_the_round_and_says_it_needs_wifi(world):
    world.shadow_ready()
    assert world.start()
    world.service.calls.clear()
    world.match.emergency_button.click()
    assert ("manual",) in world.service.calls and not world.match.controller.running and world.tracking.cleared == 1
    assert world.service.mode is Mode.MANUAL and world.engine.active_loop is None
    text = world.match.action_label.text()
    assert "EMERGENCY STOP" in text and "Wi-Fi is down" in text and "ARM switch" in text


def test_two_consecutive_rounds_make_two_round_logs_and_clear_the_opponent_twice(world):
    world.shadow_ready()
    for number in (1, 2):
        world.tracking.set(opponent="tracked")
        assert world.start()
        world.tick(count=4)
        world.match.stop_button.click()
        assert world.tracking.cleared == number
    folder = world.hub.session_log.folder
    assert (folder / "round-001.jsonl").is_file() and (folder / "round-002.jsonl").is_file()
    for name in ("round-001.jsonl", "round-002.jsonl"):
        events = world.log_events(name)
        assert [e["event"] for e in events if e["kind"] == "round"] == ["start", "stop"]
    world.match.paint_all()
    listing = [world.match.sessions_list.item(i).text() for i in range(world.match.sessions_list.count())]
    assert listing and "round-001" in listing[0] and "round-002" in listing[0]
    opened = []
    world.match.open_folder = opened.append
    world.match.sessions_list.setCurrentRow(0)
    world.match.open_folder_button.click()
    assert opened == [folder]


# ---------------------------------------------------------------- decisions, intended and transmitted values

def test_decisions_reach_the_mode_manager_only_during_a_round(world):
    world.shadow_ready(imu=True)                                            # a SIMULATED IMU: a complete DIAGNOSTIC observation exists
    world.tick(count=3)
    assert world.service.decisions == []                                     # no round, no loop, nothing fed
    assert world.start()
    world.tick(count=3)
    assert len(world.service.decisions) >= 3
    feed, kwargs = world.service.decisions[-1]
    assert feed["proposed"]["tactic"] == "advance" and feed["valid_for_control"] is False and feed["source"] == "SIMULATED"
    assert kwargs["policy_loaded"] is True and kwargs["mock"] is True and kwargs["observation_at"] is not None
    assert set(feed["inputs"]) == {"imu", "previous_action", "own_pose", "opponent_position"}
    world.match.stop_button.click()
    count = len(world.service.decisions)
    world.tick(count=3)
    assert len(world.service.decisions) == count


def test_in_ai_shadow_the_transmitted_values_are_zero_and_the_board_says_not_transmitted(world):
    world.shadow_ready(imu=True)
    assert world.start()
    world.tick(count=3)
    board = world.board()
    assert board["Model loaded"] == "YES"
    assert board["Current AI tactic"].startswith("ADVANCE") and "DIAGNOSTIC (not valid for control)" in board["Current AI tactic"]
    assert board["INTENDED throttle / steer"].startswith(ADVANCE_60)
    transmitted = board["ACTUAL TRANSMITTED throttle / steer"]
    assert transmitted.startswith("0 / 0") and "not transmitted" in transmitted and "AI SHADOW" in transmitted
    assert all(not plan.sent and plan.transmitted_throttle == 0 and plan.transmitted_steer == 0 for plan in world.service.plans)
    assert board["Own robot"].startswith("MEASURED") and "x 0.500, y 0.500" in board["Own robot"]
    assert board["Opponent"].startswith("TRACKED") and "IMAGE-BASED" in board["Opponent"] and "0.80" in board["Opponent"]
    assert "SIMULATED" in world.match.evidence_label.text()


def test_the_board_without_any_round_shows_no_decision_and_zero_transmitted(world):
    board = world.board()
    assert board["Model loaded"].startswith("NO") and board["Current AI tactic"].startswith("none")
    assert board["ACTUAL TRANSMITTED throttle / steer"].startswith("0 / 0") and "MANUAL" in board["ACTUAL TRANSMITTED throttle / steer"]
    assert board["INTENDED throttle / steer"].startswith("-")
    world.tracking.flow = False
    assert world.board()["Own robot"].startswith("no track")


def test_a_blocked_decision_without_an_imu_is_fed_as_no_tactic(world):
    world.shadow_ready(imu=False)
    assert world.start()
    world.tick(count=3)
    feed, _ = world.service.decisions[-1]
    assert feed["proposed"] is None and any("IMU" in item for item in feed["blocked_by"])
    assert world.service.plans == []                                         # nothing was planned from a decision with no tactic
    assert "NONE (blocked)" in world.board()["Current AI tactic"]


# ---------------------------------------------------------------- a fault during AUTONOMOUS (scripted gate)

def run_autonomous_round(world) -> None:
    world.full_live()
    assert world.match.select_mode(Mode.AUTONOMOUS)[0] and world.confirmations == 1
    world.frame("LIVE")
    world.match.start_button.click()
    assert world.match.controller.running and world.match.controller.mode is Mode.AUTONOMOUS and not world.match.controller.downgraded


def test_a_fault_during_autonomous_drops_to_ai_shadow_and_needs_a_new_deliberate_confirmation(world):
    match, service = world.match, world.service
    run_autonomous_round(world)
    assert service.mode is Mode.AUTONOMOUS and [c[1] for c in service.calls if c[0] == "mode"][-2:] == [Mode.AI_SHADOW, Mode.AUTONOMOUS]
    world.tick(count=3)
    sent = [p for p in service.plans if p.sent]
    assert sent and sent[-1].transmitted_throttle == 60 and sent[-1].transmitted_steer == 0
    board = world.board()
    assert board["ACTUAL TRANSMITTED throttle / steer"].startswith("60 / 0") and "NOT evidence the robot moved" in board["ACTUAL TRANSMITTED throttle / steer"]
    assert board["Current AI tactic"].startswith("ADVANCE") and "LIVE (valid for control)" in board["Current AI tactic"]
    # the fault: the opponent is lost
    world.tracking.set(opponent="lost")
    found = world.tick()
    assert found and "Opponent" in found[0]
    assert service.mode is Mode.AI_SHADOW and match.controller.mode is Mode.AI_SHADOW and match.controller.needs_reconfirm
    assert match.controller.running                                          # the round continues, as AI SHADOW
    assert not match.banner.isHidden() and "AUTONOMOUS STOPPED" in match.banner.text() and "Re-enter AUTONOMOUS deliberately" in match.banner.text()
    plans_after = len(service.plans)
    world.tracking.set(opponent="tracked")                                   # everything is fine again: it must NOT come back by itself
    world.tick(count=5)
    assert service.mode is Mode.AI_SHADOW and match.controller.needs_reconfirm and world.confirmations == 1
    assert all(not p.sent for p in service.plans[plans_after:]) and service.plans[plans_after:]
    # a new deliberate confirmation brings it back
    ok, why = match.select_mode(Mode.AUTONOMOUS)
    assert ok and world.confirmations == 2 and service.mode is Mode.AUTONOMOUS and not match.controller.needs_reconfirm
    faults = [e for e in world.log_events() if e["kind"] == "safety"]
    assert any(e["event"] == "autonomous stopped" for e in faults)
    match.stop_button.click()
    assert service.mode is Mode.MANUAL and not match.controller.running


def test_losing_other_prerequisites_during_autonomous_also_drops_it(world):
    for change in (lambda: world.tracking.set(own="lost"), lambda: setattr(world.tracking.telemetry, "flow", False),
                   lambda: setattr(world.service, "heartbeat", 0.9), lambda: setattr(world.hub, "monitor", rig.FakeMonitor(up=False))):
        run_autonomous_round(world) if not world.match.controller.running else None
        change()
        found = world.tick()
        assert found and world.service.mode is Mode.AI_SHADOW and world.match.controller.needs_reconfirm
        world.match.stop_button.click()
        world.service.heartbeat = 0.1
        world.hub.monitor = rig.FakeMonitor(up=True)
        world.confirmations = 0
        world.tracking.set(own="measured")
        world.tracking.telemetry.flow = True


def test_a_failing_inference_is_a_fault_and_is_not_repeated_every_tick(world):
    world.shadow_ready(imu=True)
    assert world.start()
    world.tick()
    real = world.engine.decide

    def broken(result, now=None):
        raise RuntimeError("scripted inference failure")

    world.engine.decide = broken
    found = []
    for _ in range(20):
        found += world.tick(0.05)
    assert 1 <= len(found) <= 2                                             # reported once, then again only every 2 s (not at 20 Hz)
    assert all("inference error" in item and "scripted inference failure" in item for item in found)
    world.match.paint_all()
    assert "scripted inference failure" in world.match.faults_label.text()
    count = len(world.service.decisions)
    world.tick(count=3)
    assert len(world.service.decisions) == count                            # nothing is fed while inference fails: no old action is reused
    world.engine.decide = real


# ---------------------------------------------------------------- the one inference loop, shutdown

def test_shutdown_mid_round_stops_it_neutral_manual_and_releases_everything(world):
    match = world.match
    world.shadow_ready(imu=True)
    assert world.start()
    world.tick(count=2)
    world.service.calls.clear()
    match.shutdown()
    assert not match.controller.running and "application closed during the round" in match.controller.last_stop_reason
    assert ("manual",) in world.service.calls and world.service.mode is Mode.MANUAL
    assert world.engine.active_loop is None and not world.ai.loop_active and world.tracking.cleared == 1
    assert not match._timer.isActive() and match.tick() == []
    match.shutdown()                                                         # idempotent
    events = world.log_events("round-001.jsonl")
    assert [e["event"] for e in events if e["kind"] == "round"] == ["start", "stop"]


def test_a_rebuilt_screen_cannot_start_a_second_inference_loop(world):
    world.shadow_ready()
    assert world.start()
    assert world.engine.active_loop is not None
    token = world.engine.active_loop
    ai2 = world.make_ai()                                                    # a restarted AI screen / match screen over the same engine
    service2 = rig.FakeService(world.clock)
    match2 = world.make_match(ai2, service2)
    assert ai2.model_loaded()
    match2.mode_buttons[Mode.AI_SHADOW].click()
    world.frame()
    match2.start_button.click()
    assert not match2.controller.running and "inference loop could not start" in match2.action_label.text()
    assert "already running" in match2.action_label.text()
    assert service2.mode is Mode.MANUAL and not ai2.loop_active and world.engine.active_loop == token
    world.match.stop_button.click()
    assert world.engine.active_loop is None
    match2.start_button.click()
    assert match2.controller.running and world.engine.active_loop is not None and world.engine.active_loop != token


def test_the_loop_runs_only_while_a_round_runs(world):
    world.shadow_ready(imu=True)
    assert world.engine.active_loop is None
    assert world.start()
    assert world.engine.active_loop is not None
    world.match.stop_button.click()
    assert world.engine.active_loop is None
    world.press(Mode.MANUAL)
    assert world.start(Mode.MANUAL) and world.engine.active_loop is None     # a MANUAL round needs no inference


# ---------------------------------------------------------------- the session log never claims execution

def test_the_session_log_records_plans_and_transmissions_but_never_claims_the_robot_acted(world):
    run_autonomous_round(world)
    world.tick(count=6)
    world.tracking.set(opponent="lost")
    world.tick(count=2)
    world.match.stop_button.click()
    world.shadow_ready(imu=True)
    world.tracking.set(opponent="tracked")                                  # a new opponent is selected for the second round
    assert world.start()
    world.tick(count=3)
    world.match.stop_button.click()
    folder = world.hub.session_log.folder
    files = sorted(folder.glob("*.jsonl"))
    assert {f.name for f in files} >= {"session.jsonl", "round-001.jsonl", "round-002.jsonl"}
    text = "\n".join(f.read_text(encoding="utf-8") for f in files)
    assert not re.search(r"execut(ed|es)\b|\bmoved\b|\bdrove\b|was driven|physically (moved|executed|performed|carried)|motors? (ran|turned|spun)", text, re.I)
    events = [e for f in files for e in read_events(f)]
    kinds = {e["kind"] for e in events}
    assert kinds >= {"round", "decision", "command_plan", "transmit", "mode", "safety", "observation"}
    transmits = [e for e in events if e["kind"] == "transmit"]
    assert transmits and all(e["physical_execution"] == "unknown: no evidence" for e in transmits)
    assert any(e["sent"] is True and e["throttle"] == 60 for e in transmits)                 # the scripted AUTONOMOUS datagrams: attempts, not outcomes
    assert any(e["sent"] is False and e["throttle"] == 0 for e in transmits)
    assert any(e["kind"] == "mode" and "AUTONOMOUS" in e["event"] for e in events)
    assert all(e["kind"] != "decision" or e.get("label") in ("LIVE", "DIAGNOSTIC") for e in events)


# ---------------------------------------------------------------- the REAL mode manager (fake transport rig)

def test_with_the_real_link_service_ai_shadow_transmits_zero_and_autonomous_is_refused_by_the_real_gate(qapp, tmp_path):
    world = World(tmp_path)
    service_rig = tls.Rig(tmp_path / "rig")
    try:
        match = world.make_match(world.ai, service_rig.service)
        world.shadow_ready(imu=True)
        assert service_rig.service.latest().mode.mode is Mode.MANUAL and not service_rig.service.latest().transmitting
        match.mode_buttons[Mode.AUTONOMOUS].click()                        # the real gate is closed: reasons, no dialog
        reasons = match.gate_reasons.text()
        for name in ("real_session", "acceptance", "build"):
            assert f"{name}:" in reasons
        assert "AUTONOMOUS: UNAVAILABLE" in match.gate_label.text() and world.confirmations == 0
        world.frame()
        match.start_button.click()
        service_rig.step(0.02, status=False)                              # the worker thread's job, by hand
        assert match.controller.running and match.controller.downgraded and "AUTONOMOUS was refused" in match.message_label.text()
        assert service_rig.service.latest().mode.mode is Mode.AI_SHADOW
        for _ in range(5):
            world.frame()
            world.clock.advance(0.05)
            match.tick()
            service_rig.step(0.05, status=False)
        snap = service_rig.service.latest()
        assert snap.plans and all(not p.sent and p.transmitted_throttle == 0 and p.transmitted_steer == 0 for p in snap.plans)
        assert not snap.transmitting and not snap.ever_sent and service_rig.port.written == []
        match.paint_all()
        board = dict(match.board_texts())
        assert board["ACTUAL TRANSMITTED throttle / steer"].startswith("0 / 0") and "not transmitted" in board["ACTUAL TRANSMITTED throttle / steer"]
        assert board["INTENDED throttle / steer"].startswith(ADVANCE_60)
        match.stop_button.click()
        service_rig.step(0.02, status=False)
        assert service_rig.service.latest().mode.mode is Mode.MANUAL and not match.controller.running
    finally:
        world.close()
        service_rig.service.shutdown()


def test_service_modes_waits_for_the_real_managers_verdict_on_autonomous_and_treats_silence_as_refusal(qapp, tmp_path):
    service_rig = tls.Rig(tmp_path)
    try:
        modes = ms.ServiceModes(service_rig.service, wait_s=0.5, drive=lambda: service_rig.step(0.02, status=False))
        assert modes.mode is Mode.MANUAL
        assert modes.request(Mode.AI_SHADOW) == (True, [])
        ok, why = modes.request(Mode.AUTONOMOUS, None, Confirmation(CONFIRM_PHRASE, True, True, True, service_rig.service.now()))
        assert not ok and why and any("gate closed" in item or "real_session" in item or "acceptance" in item for item in why)
        assert modes.mode is Mode.AI_SHADOW
        silent = ms.ServiceModes(service_rig.service, wait_s=0.05, drive=lambda: None, sleep=lambda s: None)
        ok, why = silent.request(Mode.AUTONOMOUS, None, None)
        assert not ok and "did not answer" in why[0]
        modes.request(Mode.MANUAL)
        service_rig.step(0.02, status=False)
        assert modes.mode is Mode.MANUAL
        local = ms.ServiceModes(None)
        assert local.request(Mode.AUTONOMOUS) == (False, ["there is no link service in this session: AUTONOMOUS cannot be entered"])
        assert local.request(Mode.AI_SHADOW) == (True, []) and local.mode is Mode.AI_SHADOW
    finally:
        service_rig.service.shutdown()


def test_without_a_link_service_the_screen_still_stops_and_never_enters_autonomous(qapp, tmp_path):
    world = World(tmp_path)
    try:
        match = world.make_match(world.ai, None)
        match.service = None
        match.modes.service = None
        world.shadow_ready()
        ok, why = match.select_mode(Mode.AUTONOMOUS)
        assert not ok and "no link service" in why
        world.frame()
        match.start_button.click()
        assert match.controller.running and match.controller.mode is Mode.AI_SHADOW
        match.stop_button.click()
        assert not match.controller.running and match.autonomous_readiness().level == BLOCKED
    finally:
        world.close()


# ---------------------------------------------------------------- the dashboard wires both screens

def test_the_dashboard_builds_both_screens_with_the_shared_services(qapp, tmp_path):
    from control_center.app import build_parser, hub_options, load_config
    from control_center.dashboard import DashboardWindow

    config = load_config(ROOT / "config" / "control_center.yaml")
    args = build_parser().parse_args(["--mock", "--state-file", str(tmp_path / "state.json"), "--rounds-dir", str(tmp_path / "rounds")])
    options = hub_options(args, dict(config))
    options.start_camera = False
    options.session_log_dir = tmp_path / "sessions"
    window = DashboardWindow(dict(config), options, start_tab=args.tab)
    try:
        assert isinstance(window.ai, AiModelScreen) and isinstance(window.match, MatchScreen)           # neither fell back to a placeholder
        assert window.match.ai is window.ai and window.match.tracking is window.tracking and window.ai.tracking is window.tracking
        assert window.match.service is window.hub.link_service and window.hub.link_service is not None
        assert [window.tabs.tabText(i) for i in range(window.tabs.count())].count("MATCH") == 1
        providers = window._status_providers()
        # (the engine's model cache is per process: an earlier test may already have loaded the real model, which is then reused)
        loaded = window.ai.model_loaded()
        assert providers["ai"]().level == (WARNING if loaded else NOT_CONFIGURED) and providers["autonomous"]().level == BLOCKED
        assert window._providers()["operate"]().level in ((WARNING, READY) if loaded else (NOT_CONFIGURED,))
        assert not window.match.controller.running and window.hub.link_service.latest().mode.mode is Mode.MANUAL
        assert pe.LOOPS.active is None
    finally:
        window.close()
    assert window.match._closed and window.ai._closed and pe.LOOPS.active is None
