"""AI / MODEL screen (control_center/screens/ai_model.py) and the pure helpers behind it (control_center/backend/match_support.py).

Offscreen Qt, a MOCK hub that is never started, a fake policy engine (no checkpoint, no torch), a scripted tracking screen and
scripted telemetry. No socket, serial port or camera is opened, nothing is flashed or sent, and nothing is written under config/ or
logs/ (every folder is inside tmp_path). Every result is SOFTWARE / SIMULATED: none of this is hardware verification.
One test loads the real checkpoint when torch and stable-baselines3 are importable and skips (printing why) otherwise.
"""

from __future__ import annotations

import os
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtWidgets import QApplication, QLabel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))

import match_rig as rig  # noqa: E402

from control_center.backend import match_support as ms  # noqa: E402
from control_center.backend import policy_engine as pe  # noqa: E402
from control_center.backend import session_log  # noqa: E402
from control_center.backend import wifi_link as wl  # noqa: E402
from control_center.backend.command_mapping import LIMITS, POLICY_TACTICS, describe_mapping  # noqa: E402
from control_center.backend.readiness import FAILED, NOT_CONFIGURED, READY, WARNING  # noqa: E402
from control_center.screens.ai_model import AiModelScreen  # noqa: E402
from control_center.tools.observation_contract import RECOMMENDED_CHECKPOINT  # noqa: E402


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


@pytest.fixture
def hub(qapp, tmp_path):
    made = rig.make_hub(tmp_path)
    yield made
    made.close()


@pytest.fixture
def world(hub):
    clock = rig.Clock()
    tracking = rig.FakeTracking(clock)
    engine = rig.FakeEngine()
    made: list[AiModelScreen] = []

    def build(**kwargs) -> AiModelScreen:
        kwargs.setdefault("tracking", tracking)
        chosen = kwargs.pop("engine", engine)
        screen = AiModelScreen(hub, engine_factory=lambda _path: chosen, clock=clock, start_timer=False, **kwargs)
        made.append(screen)
        return screen

    yield SimpleNamespace(hub=hub, clock=clock, tracking=tracking, engine=engine, build=build)
    for screen in made:
        screen.shutdown()


def load(screen: AiModelScreen) -> None:
    assert screen.load_model()
    assert rig.wait_for(screen.model_loaded), "the fake model did not finish loading"


def label_texts(screen) -> str:
    return "\n".join(item.text() for item in screen.findChildren(QLabel))


# ---------------------------------------------------------------- model card

def test_model_card_shows_identity_versions_spaces_and_the_plain_finding_about_what_the_checkpoint_is(world):
    screen = world.build()
    assert rig.wait_for(lambda: screen.model_info() is not None)
    screen.paint_all()
    card = dict(screen.card.texts())
    assert card["Path"].endswith("final_model.zip")
    assert card["SHA-256"] == "ab" * 32 and "64,426 bytes" in card["Size"]
    assert "stable-baselines3 2.9.2" in card["Trained with"] and "torch 2.15.0+cpu" in card["Trained with"]
    assert "stable-baselines3 2.9.2" in card["Installed here"]
    assert "Box(-1, 1, (19,), float32)" in card["Observation space"]
    assert "Discrete(5)" in card["Action space"] and "advance" in card["Action space"]
    assert card["Compatibility"].startswith("COMPATIBLE")
    assert card["Model state"] == "NOT LOADED"
    text = screen.origin_label.text()
    assert "behaviour cloning" in text and "DAgger" in text and "NOT trained by PPO" in text and "saved as a Stable-Baselines3 PPO object" in text
    assert "SIMULATOR evidence" in text and "nothing is a hardware result" in text


def test_an_unaudited_checkpoint_gets_no_claims_about_its_origin(world, tmp_path):
    other = tmp_path / "other.zip"
    other.write_bytes(b"not a real checkpoint")
    screen = world.build()
    assert screen.set_model_path(str(other), remember=False)[0]
    assert "UNKNOWN" in screen.origin_label.text() and "behaviour cloning" not in screen.origin_label.text()


def test_only_zip_files_can_be_chosen_and_an_incompatible_checkpoint_is_failed_before_any_load(world, tmp_path):
    screen = world.build()
    ok, why = screen.set_model_path(str(tmp_path / "model.pt"), remember=False)
    assert not ok and "*.zip" in why
    incompatible = rig.FakeEngine(compatible=False)
    other = world.build(engine=incompatible)
    assert rig.wait_for(lambda: other.model_info() is not None)
    readiness = other.readiness()
    assert readiness.level == FAILED and "scripted incompatibility" in readiness.detail
    paint = other.paint_all
    paint()
    assert "INCOMPATIBLE" in dict(other.card.texts())["Compatibility"]


# ---------------------------------------------------------------- loading

def test_nothing_loads_at_startup_without_a_remembered_preference(world):
    screen = world.build()
    rig.process(0.1)
    assert world.engine.load_calls == 0 and screen.model_state() == "not_loaded" and not screen.model_loaded()
    assert screen.readiness().level == NOT_CONFIGURED and world.engine.active_loop is None and not screen.loop_active


def test_a_remembered_preference_only_loads_it_never_starts_an_inference_loop(qapp, tmp_path):
    hub = rig.make_hub(tmp_path, settings=True)
    try:
        checkpoint = tmp_path / "remembered.zip"
        checkpoint.write_bytes(b"zip")
        assert hub.settings_store.update_section(hub.profile_name, "preferences", {"model_path": str(checkpoint)}).ok
        engine = rig.FakeEngine(checkpoint)
        screen = AiModelScreen(hub, tracking=rig.FakeTracking(rig.Clock()), engine_factory=lambda _p: engine, start_timer=False)
        try:
            assert screen.model_path() == str(checkpoint)
            assert rig.wait_for(screen.model_loaded)
            assert engine.load_calls == 1 and engine.active_loop is None and not screen.loop_active and engine.decide_calls == 0
        finally:
            screen.shutdown()
    finally:
        hub.close()


def test_the_chosen_path_is_remembered_per_profile_in_the_preferences_section(qapp, tmp_path):
    hub = rig.make_hub(tmp_path, settings=True)
    try:
        screen = AiModelScreen(hub, tracking=None, engine_factory=lambda _p: rig.FakeEngine(), start_timer=False, autoload=False)
        chosen = tmp_path / "chosen.zip"
        chosen.write_bytes(b"zip")
        ok, why = screen.set_model_path(str(chosen))
        assert ok and "remembered for profile" in why
        assert hub.settings_store.get_section(hub.profile_name, "preferences")["model_path"] == str(chosen)
        screen.shutdown()
    finally:
        hub.close()


def test_loading_runs_off_the_gui_thread_refuses_a_second_load_and_reuses_the_loaded_model(world):
    gate = threading.Event()
    engine = rig.FakeEngine(gate=gate)
    screen = world.build(engine=engine)
    assert screen.load_model() and screen.model_state() == "loading"
    assert not screen.load_model() and "already running" in screen.message_label.text()     # a second concurrent load is refused
    rig.process(0.1)                                                                         # the GUI thread is not blocked by the load
    assert screen.model_state() == "loading" and screen.readiness().level == NOT_CONFIGURED
    assert not screen.set_model_path("other.zip", remember=False)[0]                         # and the file cannot change meanwhile
    gate.set()
    assert rig.wait_for(screen.model_loaded)
    assert engine.load_calls == 1 and engine.load_threads[0] is not threading.main_thread()
    assert screen.load_model() and "reused" in screen.message_label.text() and engine.load_calls == 1


def test_a_failed_load_is_failed_with_the_reason(world):
    screen = world.build(engine=rig.FakeEngine(fail="boom: shape mismatch"))
    screen.load_model()
    assert rig.wait_for(lambda: screen.model_state() == "failed")
    readiness = screen.readiness()
    assert readiness.level == FAILED and "boom: shape mismatch" in readiness.detail and not screen.model_loaded()
    screen.paint_all()
    assert "LOAD FAILED" in dict(screen.card.texts())["Model state"]


# ---------------------------------------------------------------- the observation and readiness

def test_the_imu_row_says_the_baseline_firmware_sends_none_and_nothing_is_invented(world):
    screen = world.build()
    view = screen.observation_status(world.clock())
    screen.paint_all()
    imu = view.group("imu")
    assert imu.word == "MISSING" and imu.age_ms is None
    assert "baseline firmware sends no IMU data" in imu.reason and "BLOCKED" in imu.reason and "Nothing is invented" in imu.reason
    assert not view.complete and not view.valid_for_control and view.result.values is None and view.label == "NO OBSERVATION"
    rows = {row[0]: row for row in screen.group_texts()}
    assert rows["IMU"][2] == "MISSING" and "no IMU data" in rows["IMU"][5]
    assert rows["Previous action"][2:5] == ["VALID", "-", "CONSTANT"] and "constant 0.0" in rows["Previous action"][5]
    assert rows["Own pose"][2] == "VALID" and rows["Own pose"][4] == "SIMULATED"            # a MOCK camera is never LIVE
    assert rows["Opponent"][2] == "VALID" and rows["Own pose"][3].endswith("ms")
    assert "no IMU data" in label_texts(screen) and "constants the training environment defined" in label_texts(screen)


def test_the_19_channels_are_listed_with_constants_labelled(world):
    screen = world.build()
    screen.paint_all()
    rows = screen.channel_texts()
    assert len(rows) == 19 and [r[0] for r in rows] == [str(i) for i in range(19)]
    assert [r[2] for r in rows[:6]] == ["imu"] * 6 and all(r[4] == "MISSING" for r in rows[:6])
    assert all(r[3] == "+0.0000 (constant)" and r[6] == "CONSTANT" for r in rows[6:13])
    assert [r[2] for r in rows[13:]] == ["own_pose"] * 4 + ["opponent_position"] * 2 and all(r[3] == "-" for r in rows[13:])


def test_stale_inputs_and_an_untracked_opponent_are_flagged_never_substituted(world):
    screen = world.build()
    world.tracking.freeze_at(world.clock() - 0.3)                    # a result 300 ms old: older than the camera limit of 250 ms
    view = screen.observation_status(world.clock())
    assert view.group("own_pose").word == "STALE" and view.group("opponent_position").word == "STALE"
    assert "300 ms old" in view.group("own_pose").reason or "0.3" in view.group("own_pose").reason or "ms old" in view.group("own_pose").reason
    world.tracking._frozen = None
    world.tracking.set(opponent="uncertain")
    view = screen.observation_status(world.clock())
    opponent = view.group("opponent_position")
    assert opponent.word == "MISSING" and "scripted uncertain" in opponent.reason and view.result.values is None
    world.tracking.set(own="lost")
    assert screen.observation_status(world.clock()).group("own_pose").word == "MISSING"
    old = rig.FakeTracking(world.clock)
    old.freeze_at(world.clock() - 10.0)                              # a result from 10 s ago is no result at all
    stale_screen = world.build(tracking=old)
    view = stale_screen.observation_status(world.clock())
    assert view.group("own_pose").word == "MISSING" and any("ignored" in note for note in view.notes + (stale_screen.live_inputs().notes))


def test_readiness_levels_and_ready_is_unreachable_without_a_live_imu(world):
    screen = world.build()
    assert screen.readiness().level == NOT_CONFIGURED                                       # not loaded
    load(screen)
    readiness = screen.readiness()
    assert readiness.level == WARNING and "IMU unavailable" in readiness.detail and "baseline firmware sends no IMU data" in readiness.detail
    # Everything else perfect (own pose, opponent tracked, fresh): still never READY without a real IMU.
    world.tracking.set(own="measured", opponent="tracked")
    assert screen.observation_status(world.clock()).group("imu").word == "MISSING"
    assert screen.readiness().level == WARNING
    # A simulated / mock IMU (the MOCK hub replays controller lines): a complete observation exists but it is DIAGNOSTIC, not READY.
    world.tracking.telemetry.flow = True
    view = screen.observation_status(world.clock())
    assert view.complete and view.label == "DIAGNOSTIC" and not view.valid_for_control and view.source == "SIMULATED"
    assert screen.readiness().level == WARNING and not screen.readiness().ok


def test_ready_needs_every_group_live_and_is_scripted_here_only_to_show_the_rule_is_not_hard_coded(world, hub):
    screen = world.build()
    load(screen)
    world.tracking.telemetry.flow = True
    hub.options.mock = False                                                   # SCRIPTED: pretend a live session
    hub.capture.state = "LIVE"
    hub.restore_report = SimpleNamespace(world_coordinates_trusted=True, headline="scripted")
    view = screen.observation_status(world.clock())
    assert view.valid_for_control and view.label == "LIVE" and screen.readiness().level == READY
    world.tracking.telemetry.flow = False                                      # the baseline firmware: no IMU -> never READY
    assert screen.readiness().level == WARNING
    world.tracking.telemetry.flow = True
    hub.restore_report = None                                                  # the arena calibration is not confirmed this run
    view = screen.observation_status(world.clock())
    assert view.complete and not view.valid_for_control and any("not confirmed trusted" in b for b in view.blocked_by)
    assert screen.readiness().level == WARNING


# ---------------------------------------------------------------- decisions and the single inference loop

def test_a_decision_shows_tactic_source_validity_blocked_by_and_the_diagnostic_label(world):
    screen = world.build()
    load(screen)
    world.tracking.telemetry.flow = True
    assert screen.infer_once(world.clock()) is None and screen.last_decision() is None            # no loop registered: nothing is asked
    ok, why = screen.start_loop()
    assert ok and screen.loop_active and world.engine.active_loop is not None
    decision = screen.infer_once(world.clock())
    assert decision.tactic == "advance" and decision.source == "SIMULATED" and not decision.valid_for_control
    assert screen.last_decision() is decision and screen.last_inference_error() == ""
    screen.paint_all()
    grid = dict(screen.decision_grid.texts())
    assert grid["Tactic"] == "advance" and grid["valid_for_control"] == "False" and grid["Source"].startswith("SIMULATED")
    assert "not LIVE" in grid["blocked_by"] or "SIMULATED" in grid["blocked_by"]
    assert "DIAGNOSTIC (not valid for control)" in screen.decision_headline.text() and "ADVANCE" in screen.decision_headline.text()


def test_a_blocked_decision_has_no_tactic_and_says_why(world):
    screen = world.build()
    load(screen)
    screen.start_loop()
    decision = screen.infer_once(world.clock())                    # the baseline firmware: no IMU -> no vector -> no tactic
    assert decision.tactic is None and any("IMU" in item for item in decision.blocked_by)
    screen.paint_all()
    assert "NO TACTIC" in screen.decision_headline.text() and dict(screen.decision_grid.texts())["Tactic"].startswith("none")


def test_the_loop_cannot_start_without_a_loaded_model_and_never_twice_even_after_the_screen_is_rebuilt(world):
    screen = world.build()
    assert screen.start_loop() == (False, "the model is not loaded")
    load(screen)
    assert screen.start_loop()[0] and screen.start_loop() == (True, "the loop is already running")
    assert pe.LOOPS.active is not None
    restarted = world.build()                                                  # the screen is rebuilt without the old one being shut down
    assert restarted.model_loaded()                                            # the engine already holds the model: reused, not reloaded
    ok, why = restarted.start_loop()
    assert not ok and "already running" in why and not restarted.loop_active
    token = pe.LOOPS.active
    screen.shutdown()                                                          # releasing the first loop
    assert pe.LOOPS.active is None
    assert restarted.start_loop()[0] and pe.LOOPS.active is not None and pe.LOOPS.active != token
    assert world.engine.load_calls == 1


def test_shutdown_releases_the_loop_is_idempotent_and_ignores_late_results(world):
    gate = threading.Event()
    engine = rig.FakeEngine(gate=gate)
    screen = world.build(engine=engine)
    screen.load_model()
    screen.shutdown()
    screen.shutdown()
    gate.set()
    rig.process(0.2)                                                           # the late load result is dropped, no crash
    assert screen.model_state() == "loading" and engine.active_loop is None
    other = world.build()
    load(other)
    other.start_loop()
    other.shutdown()
    assert pe.LOOPS.active is None


# ---------------------------------------------------------------- diagnostic inference

def test_diagnostic_inference_is_simulated_never_valid_for_control_and_starts_no_loop(world):
    screen = world.build()
    assert screen.diagnostic_inference(world.clock()) is None and "not loaded" in screen.diagnostic_label.text()
    load(screen)
    decision = screen.diagnostic_inference(world.clock())
    assert decision.tactic == "advance" and decision.source == "SIMULATED" and decision.valid_for_control is False
    assert "DIAGNOSTIC" in screen.diagnostic_label.text() and "never valid for control" in screen.diagnostic_label.text()
    assert screen.last_decision() is None and screen.last_diagnostic() is decision        # it never becomes the loop's decision
    assert world.engine.active_loop is None and not screen.loop_active


# ---------------------------------------------------------------- the command mapping

def test_the_mapping_table_equals_describe_mapping_with_the_limits_and_the_plain_notes(world):
    screen = world.build()
    screen.paint_all()
    expected = describe_mapping(LIMITS)
    assert screen.mapping_texts() == [[r["tactic"], str(r["throttle"]), str(r["steer"]), str(r["left"]), str(r["right"]), r["meaning"]] for r in expected]
    assert [r["tactic"] for r in expected] == list(POLICY_TACTICS)
    limits = dict(screen.limits_grid.texts())
    assert "60 of 255" in limits["max_command"] and "250 ms" in limits["max decision age"] and "0.15 s" in limits["slew limit"]
    assert "0, 0 for 100 ms" in limits["neutral-first"]
    text = label_texts(screen)
    assert "+102 / -71 / +71 / -66" in text and "not the operating limit" in text and "verified physically" in text and "wheels raised" in text


def test_check_mapping_runs_the_software_self_check_and_feeds_mapping_validated(world, monkeypatch):
    screen = world.build()
    assert not screen.mapping_validated() and "Not run" in screen.check_result.text()
    screen.check_button.click()
    assert screen.mapping_validated() and screen.mapping_check().passed
    assert screen.check_result.text().startswith("PASS") and "SOFTWARE" in screen.check_result.text() and "hardware" in screen.check_result.text()
    # a broken mapping must FAIL: a mapper that lets an invalid tactic through as a drive command
    real = ms.map_tactic

    def leaky(tactic, now, decision_time, limits=LIMITS, state=None):
        return real("advance" if tactic is None else tactic, now, decision_time, limits, state)

    monkeypatch.setattr(ms, "map_tactic", leaky)
    screen.check_button.click()
    assert not screen.mapping_validated() and screen.check_result.text().startswith("FAIL")
    assert "invalid input" in screen.check_details.text()


def test_the_self_check_covers_every_action_the_caps_invalid_input_stale_and_neutral_first():
    check = ms.mapping_self_check()
    assert check.passed, check.failures
    names = " | ".join(line.name for line in check.lines)
    for tactic in POLICY_TACTICS:
        assert f"{tactic}:" in names
    for expected in ("invalid input", "stale decision", "future", "cap holds", "neutral-first", "turn_left has negative steer"):
        assert expected in names
    assert "SOFTWARE" in check.evidence and "Not a hardware result" in check.summary()


# ---------------------------------------------------------------- pure helpers

def test_decision_feed_carries_the_proposal_and_per_group_inputs_the_mode_manager_reads(world):
    screen = world.build()
    load(screen)
    world.tracking.telemetry.flow = True
    screen.start_loop()
    decision = screen.infer_once(world.clock())
    view = screen.observation_status(world.clock())
    feed = ms.decision_feed(decision, view.result)
    assert feed["proposed"]["tactic"] == "advance" and set(feed["inputs"]) == {"imu", "previous_action", "own_pose", "opponent_position"}
    assert feed["valid_for_control"] is False and feed["mode"] == "DIAGNOSTIC"
    from control_center.backend.control_modes import tactic_of
    from control_center.backend.ai_observation import observation_status

    assert tactic_of(feed) == "advance" and observation_status(feed).groups["own_pose"].valid
    assert ms.decision_feed(decision, view.result, allow_tactic=False)["proposed"] is None


def test_the_real_checkpoint_loads_once_in_a_worker_and_is_reused(qapp, hub):
    try:
        import stable_baselines3  # noqa: F401
        import torch  # noqa: F401
    except ImportError as error:
        print(f"SKIPPED real-model test: {error}")
        pytest.skip(f"torch / stable-baselines3 not importable: {error}")
    if not RECOMMENDED_CHECKPOINT.is_file():
        print("SKIPPED real-model test: the recommended checkpoint is not in runs/")
        pytest.skip("the recommended checkpoint is not in runs/")
    screen = AiModelScreen(hub, tracking=rig.FakeTracking(rig.Clock()), start_timer=False, autoload=False)
    second = None
    try:
        assert rig.wait_for(lambda: screen.model_info() is not None, 30)
        info = screen.model_info()
        assert info.compatible and info.obs_dim == 19 and len(info.sha256) == 64 and info.action_space == "Discrete(5)"
        screen.load_model()
        assert rig.wait_for(screen.model_loaded, 240), screen._load_error
        count = pe.LOAD_COUNT
        decision = screen.diagnostic_inference()
        assert decision is not None and decision.tactic in POLICY_TACTICS and decision.source == "SIMULATED" and not decision.valid_for_control
        second = AiModelScreen(hub, tracking=None, start_timer=False, autoload=False)       # the engine and the model are shared in the process
        assert second.model_loaded() and pe.LOAD_COUNT == count
        assert screen.start_loop()[0] and not second.start_loop()[0]
        assert screen.observation_status().group("imu").word == "MISSING"
    finally:
        screen.shutdown()
        if second is not None:
            second.shutdown()
