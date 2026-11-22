"""ROBOT TRACKING & LIVE MONITORING screen and its Qt-free model.

Offscreen Qt, a MOCK hub (synthetic camera, never started) and SYNTHETIC frames from overlay_fixture / track_fixture only:
no camera, serial port or socket is opened and nothing is sent to anything. Frames reach the screen by being published
into the hub's capture session exactly as its camera thread would publish them. Evidence class: FIXTURE; none of the
timings below is a real-camera figure.
"""

from __future__ import annotations

import ast
import dataclasses
import os
import threading
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import cv2
import numpy as np
import pytest
from PySide6.QtCore import QCoreApplication, QPoint, QPointF, Qt, Signal
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from control_center.app import load_config
from control_center.backend import tracking_model as tm
from control_center.backend.hub import HubOptions, OperationsHub
from control_center.backend.readiness import FAILED, NOT_CONFIGURED, READY, WARNING
from control_center.backend.tracking_worker import HubTelemetry, ShadowRunner, TrackingEngine, arena_shapes
from control_center.screens.tracking import TrackingScreen
from control_center.tools import overlay_fixture as fx
from control_center.tools import track_fixture as tf
from control_center.tools.live_telemetry import ImuMounting, load_mounting
from control_center.tools.observation_contract import RECOMMENDED_CHECKPOINT, load_contract
from control_center.tools.overlay_core import ClickSet

ROOT = Path(__file__).resolve().parents[1]
DT = 1.0 / 15.0
OWN = (0.575, 0.575, 0.3)
RED = (40, 40, 220)
needs_checkpoint = pytest.mark.skipif(not RECOMMENDED_CHECKPOINT.is_file(), reason="the recommended checkpoint is not in runs/")


# ---------------------------------------------------------------- fixtures and helpers


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture()
def hub(qapp, tmp_path):
    hub = OperationsHub(load_config(ROOT / "config" / "control_center.yaml"),
                        HubOptions(mock=True, start_camera=False, rounds_root=tmp_path / "rounds"))
    yield hub
    hub.close()


@pytest.fixture()
def make_screen(hub, tmp_path):
    made: list[TrackingScreen] = []

    def build(**kwargs) -> TrackingScreen:
        kwargs.setdefault("mounting_path", tmp_path / "no_such_mounting.yaml")      # never the real (unverified) file by accident
        screen = TrackingScreen(hub, **kwargs)
        made.append(screen)
        return screen

    yield build
    for screen in made:
        screen.shutdown()
        screen.close()


@pytest.fixture()
def screen(make_screen):
    return make_screen()


def pump(seconds: float = 0.02) -> None:
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        QCoreApplication.processEvents()
        time.sleep(0.003)


def wait_until(condition, timeout: float = 15.0, tick=None) -> bool:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if tick is not None:
            tick()
        QCoreApplication.processEvents()
        if condition():
            return True
        time.sleep(0.01)
    return condition()


def publish(hub, index: int, frame: np.ndarray) -> None:
    """What the hub's capture thread does for a new camera frame (the camera itself is never opened)."""
    capture = hub.capture
    with capture._new_frame:
        capture._latest = (index, time.monotonic(), frame)
        capture._new_frame.notify_all()
    capture.state = "MOCK"


def render(hub, bots=(), strays=()):
    return fx.render_frame(hub.vision.config, hub.mock_source.camera, fx.Scene(list(bots), list(strays)))


def bot_frame(hub, ids=None, objects=(), strays=()):
    frame = tf.render_bot(hub.vision.config, hub.mock_source.camera, *OWN, ids=ids)
    for centre, colour in objects:
        tf.paint_box(frame, hub.mock_source.camera, centre, 0.2, (0.11, 0.10), 0.034, colour)
    return frame


def pixel(hub, x, y, height=0.034):
    u, v = hub.mock_source.camera.project([[x, y, height]])[0]
    return float(u), float(v)


def slab_box(hub, x, y, half=(34.0, 30.0)):
    """The rectangle an operator would drag around the flat slab standing at floor point (x, y)."""
    u, v = pixel(hub, x, y)
    return (u - half[0], v - half[1], 2 * half[0], 2 * half[1])


def events(hub, needle: str) -> list[str]:
    return [text for _, kind, text in hub.events if kind == "tracking" and needle in text]


class Clock:
    def __init__(self, t: float = 0.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t


class FakePolicy:
    """Stands in for the 30-60 s checkpoint load. ``propose`` returns a dictionary; nothing is ever sent."""

    def __init__(self, contract, delay: float = 0.0) -> None:
        self.contract, self.delay = contract, delay
        self.loaded, self.proposals = False, 0
        self.load_thread: int | None = None

    def load(self) -> None:
        self.load_thread = threading.get_ident()
        time.sleep(self.delay)
        self.loaded = True

    def propose(self, observation) -> dict:
        self.proposals += 1
        return {"action": 0, "tactic": self.contract.tactics[0], "probability": 0.5}


@pytest.fixture()
def verified_mounting(tmp_path) -> Path:
    path = tmp_path / "imu_mounting.yaml"
    path.write_text("imu:\n  verified: true\n  sensor_to_body: [[1, 0, 0], [0, 1, 0], [0, 0, 1]]\n  note: test fixture\n", encoding="utf-8")
    return path


# ---------------------------------------------------------------- pure model


class Spec:
    def __init__(self, face: str) -> None:
        self.face = face


LAYOUT = {0: Spec("top"), 1: Spec("top"), 4: Spec("bottom")}


def test_marker_table_separates_seen_unseen_never_seen_and_unknown_ids():
    rates = tm.MarkerRateTracker(LAYOUT, window_s=5.0)
    rates.update(0.0, [0, 1, 30])
    rates.update(0.1, [0, 30])
    rows = {row.marker_id: row for row in rates.rows(expected_faces=("top", "rear"))}
    assert rows[0].seen_now and rows[0].rate == 1.0 and rows[0].status == "seen" and rows[0].level == "ok"
    assert not rows[1].seen_now and rows[1].rate == 0.5 and rows[1].status == "not seen this frame" and rows[1].level == "warn"
    assert rows[4].status.startswith("never seen (its face is not expected") and rows[4].level == "off"
    assert not rows[30].configured and rows[30].status.startswith("UNKNOWN ID") and rows[30].level == "bad"
    assert rates.unknown_ids() == [30] and rates.never_seen_ids() == [4]
    assert [row.marker_id for row in rates.rows()] == [0, 1, 4, 30]                      # configured first, unknown last
    never = {row.marker_id: row for row in rates.rows()}[4]
    assert never.status == "never seen since tracking started" and never.level == "warn"  # no orientation known: it IS a fault


def test_detection_rate_is_counted_per_frame_over_the_last_five_seconds_and_flags_intermittent_ids():
    rates = tm.MarkerRateTracker(LAYOUT, window_s=5.0)
    for step in range(60):
        rates.update(step * 0.1, [0] if step % 2 == 0 else [0, 1])                        # id 1 only in every second frame
    one = {row.marker_id: row for row in rates.rows()}[1]
    assert one.frames_total == rates.frames_in_window and 0.45 <= one.rate <= 0.55
    assert "INTERMITTENT" in one.status and one.level == "warn" and "%" in one.rate_text
    assert rates.frames_in_window <= 52                                                 # frames older than 5 s fell out of the window
    rates.update(100.0, [0])                                                              # a long gap: old frames are not "recent"
    assert rates.frames_in_window == 1 and {r.marker_id: r for r in rates.rows()}[1].rate == 0.0
    rates.reset()
    assert rates.rows(expected_faces=None)[0].rate is None and rates.rows()[0].rate_text == "-"


def test_marker_spread_is_only_defined_for_two_or_more_used_markers():
    details = [{"id": 0, "bot_x_m": 0.500, "bot_y_m": 0.500, "heading_deg": 10.0},
               {"id": 1, "bot_x_m": 0.503, "bot_y_m": 0.504, "heading_deg": 12.0},
               {"id": 2, "bot_x_m": 0.900, "bot_y_m": 0.900, "heading_deg": 90.0}]
    spread, turn = tm.marker_spread(details, [0, 1])
    assert spread == pytest.approx(5.0, abs=0.01) and turn == pytest.approx(2.0)
    assert tm.marker_spread(details, [0]) == (None, None) and tm.marker_spread(details, []) == (None, None)
    wrap = tm.marker_spread([{"id": 0, "bot_x_m": 0, "bot_y_m": 0, "heading_deg": 179.0},
                             {"id": 1, "bot_x_m": 0, "bot_y_m": 0, "heading_deg": -179.0}], [0, 1])
    assert wrap[1] == pytest.approx(2.0)


def own_report(state="measured", used=(0, 1), age=0.0, **extra):
    return {"state": state, "used_ids": list(used), "age_s": age, "position_quality": "height-corrected", "source": "top markers",
            "x_m": 0.5, "y_m": 0.5, "heading_deg": 10.0, "orientation": "upright", "rejected": {}, "reason": "ok", **extra}


def confidence(own, evidence=None):
    return tm.tracking_confidence(own, evidence, position_limit_m=0.02, heading_limit_deg=10.0, predict_hold_s=0.2)


def test_confidence_is_derived_only_from_measured_inputs_and_shows_them():
    good = confidence(own_report(), {"fit_rms_mm": 2.0, "spread_mm": 3.0, "heading_spread_deg": 1.0})
    assert good.level == "HIGH" and good.reasons == []
    names = [name for name, _ in good.inputs]
    assert {"tracker state", "age of last measurement", "markers agreeing", "marker fit rms", "marker position spread",
            "marker heading spread", "position quality"} <= set(names)
    assert dict(good.inputs)["marker fit rms"] == "2.0 mm"
    assert confidence(own_report(used=(0,)), {"fit_rms_mm": 2.0, "spread_mm": None, "heading_spread_deg": None}).level == "MEDIUM"
    assert "only one marker" in confidence(own_report(used=(0,))).text()
    bad_fit = confidence(own_report(), {"fit_rms_mm": 12.0, "spread_mm": 3.0, "heading_spread_deg": 1.0})
    assert bad_fit.level == "MEDIUM" and "fit rms" in bad_fit.text()                       # 12 mm > half of the 20 mm agreement limit
    both = confidence(own_report(used=(0,)), {"fit_rms_mm": 12.0, "spread_mm": None, "heading_spread_deg": None})
    assert both.level == "LOW"
    predicted = confidence(own_report("predicted", age=0.05))
    assert predicted.level == "MEDIUM" and dict(predicted.inputs)["marker fit rms"].startswith("n/a")      # nothing invented
    assert confidence(own_report("predicted", age=0.15)).level == "LOW"
    assert confidence(own_report("stale", age=0.5)).level == "NONE" and confidence(own_report("lost", used=(), age=2.0)).level == "NONE"
    assert confidence(None).level == "NONE"
    rear = confidence(own_report(source="rear marker only", used=(8,)))
    assert rear.level == "LOW" and "rear marker only" in rear.text() and "only one marker" in rear.text()
    assert "not a probability" in tm.CONFIDENCE_NOTE


def test_warning_monitor_debounces_and_logs_once_per_episode():
    monitor = tm.WarningMonitor(on_s=0.4, off_s=1.0, cooldown_s=15.0)
    lost = {"own_lost": tm.Condition("own_lost", "OWN LOST", "bad")}
    assert monitor.update(0.0, lost) == [] and monitor.active == []                   # not yet: a blip is not a warning
    assert monitor.update(0.2, {}) == [] and monitor.active == []                     # it went away before it was raised
    assert monitor.update(1.0, lost) == [] and monitor.update(1.2, lost) == []
    raised = monitor.update(1.5, lost)
    assert [e.key for e in raised] == ["own_lost"] and [c.key for c in monitor.active] == ["own_lost"]
    for step in range(30):                                                             # flicker for 3 s: no spam, still active
        events_now = monitor.update(2.0 + 0.1 * step, lost if step % 2 else {})
        assert events_now == []
    assert [c.key for c in monitor.active] == ["own_lost"]
    assert monitor.update(6.0, {}) == [] and monitor.update(7.1, {}) == [] and monitor.active == []       # cleared after off_s
    assert monitor.update(8.0, lost) == [] and monitor.update(8.5, lost) == []         # re-raised inside the cooldown: banner only
    assert [c.key for c in monitor.active] == ["own_lost"] and len(monitor.log) == 1
    assert monitor.update(20.0, {}) == [] and monitor.update(21.5, {}) == []
    assert monitor.update(30.0, lost) == [] and [e.key for e in monitor.update(30.5, lost)] == ["own_lost"]    # a new episode, after the cooldown
    assert [c.severity for c in monitor.active] == ["bad"]


def test_warning_banner_orders_bad_before_warn():
    monitor = tm.WarningMonitor(on_s=0.0)
    monitor.update(0.0, {"a": tm.Condition("a", "weak", "warn"), "b": tm.Condition("b", "lost", "bad")})
    assert [c.key for c in monitor.active] == ["b", "a"]


def test_trails_keep_gaps_as_gaps_and_forget_old_points():
    trail = tm.TrailBuffer(keep_s=2.0)
    for step in range(5):
        trail.add(step * 0.1, (step, 0))
    trail.add(0.5, None)
    trail.add(0.6, None)                                                               # repeated gap markers collapse
    trail.add(0.7, (9, 9))
    runs = trail.runs()
    assert [len(run) for run in runs] == [5, 1] and runs[1] == [(9.0, 9.0)]
    trail.add(5.0, (10, 10))
    assert trail.runs() == [[(10.0, 10.0)]]
    trail.clear()
    assert trail.runs() == []


def test_diagnostics_model_reports_software_time_rate_and_skips():
    diagnostics = tm.DiagnosticsModel(window_s=3.0)
    for step in range(11):
        diagnostics.record(done_at=step * 0.1, frame_index=step * 2, software_ms=10.0 + step, queue_age_s=0.01, skipped=1 if step else 0)
    summary = diagnostics.summary(now=1.0)
    assert summary["ms_mean"] == pytest.approx(15.0) and summary["ms_p95"] == 20.0 and summary["ms_last"] == 20.0
    assert summary["fps"] == pytest.approx(10.0) and summary["skipped_total"] == 10 and summary["queue_age_ms"] == pytest.approx(10.0)
    assert tm.DiagnosticsModel().summary(0.0)["fps"] is None


def test_pit_list_uses_per_pit_radii_when_present_and_arena_problems_are_named():
    arena = {"width_m": 1.15, "height_m": 1.15, "pit_centres_m": [[0, 1.15], [1.15, 0]], "pit_radius_m": 0.25,
             "flipper_vertices_m": [[0.8, 1.15], [1.15, 1.15], [1.15, 0.8]]}
    assert [r for _, r in tm.pit_list(arena)] == [0.25, 0.25] and tm.arena_problems(arena) == []
    assert [r for _, r in tm.pit_list({**arena, "pit_radii_m": [0.20, 0.30]})] == [0.20, 0.30]
    assert tm.arena_problems({**arena, "pit_centres_m": []}) and tm.arena_problems({}) and tm.arena_problems(None)
    assert any("flipper" in p for p in tm.arena_problems({**arena, "flipper_vertices_m": [[0, 0]]}))
    shapes = arena_shapes({**arena, "pit_radii_m": [0.20, 0.30]})
    assert [label for label, _, _ in shapes] == ["arena boundary", "pit boundary", "pit boundary", "fixed flipper boundary"]
    radii = [float(np.max(np.hypot(*(poly - np.array(centre)).T))) for (label, poly, _), (centre, _) in
             zip(shapes[1:3], tm.pit_list({**arena, "pit_radii_m": [0.20, 0.30]}))]
    assert radii == pytest.approx([0.20, 0.30], abs=1e-6)


def test_frame_compatibility_refuses_a_flagged_or_mismatched_profile():
    arena = {"width_m": 1.15, "height_m": 1.15}
    ok, why = tm.frame_compatibility(arena, "live", (1.15, 1.15))
    assert ok and "matches the checkpoint" in why
    ok, why = tm.frame_compatibility({**arena, "policy_frame_ok": False}, "home_test", (1.15, 1.15))
    assert not ok and "policy_frame_ok" in why and "home_test" in why
    ok, why = tm.frame_compatibility({**arena, "frame_ok": False, "policy_frame_ok": True}, "x", (1.15, 1.15))
    assert not ok and "frame_ok" in why                                                   # either flag being False refuses
    ok, why = tm.frame_compatibility({"width_m": 0.9, "height_m": 0.9}, "home_test", (1.15, 1.15))
    assert not ok and "0.900 x 0.900" in why
    assert not tm.frame_compatibility(arena, "live", None)[0]
    assert tm.frame_compatibility({**arena, "frame_ok": True}, "live", None)[0]


def prereqs(**overrides):
    facts = dict(frame_ok=True, frame_detail="f", mapping_ok=True, mapping_detail="m", contract_ok=True, contract_detail="c",
                 imu_present=True, imu_present_detail="i", imu_fresh=True, imu_fresh_detail="i", mounting_verified=True,
                 mounting_detail="v", own_ok=True, own_detail="o", opponent_ok=True, opponent_detail="p",
                 checkpoint_state="loaded", checkpoint_detail="k")
    facts.update(overrides)
    return tm.shadow_prerequisites(**facts)


def test_shadow_checkbox_depends_only_on_the_cheap_prerequisites():
    assert tm.shadow_checkbox_state(prereqs())[0]
    assert tm.shadow_checkbox_state(prereqs(own_ok=False, opponent_ok=False, checkpoint_state="not_loaded"))[0]   # live inputs gate proposals, not the box
    for override in ("imu_present", "imu_fresh", "mounting_verified", "mapping_ok", "frame_ok", "contract_ok"):
        enabled, why = tm.shadow_checkbox_state(prereqs(**{override: False}))
        assert not enabled and why.startswith("blocked:")
    assert all(p.word in ("Ready", "MISSING") for p in prereqs())


def test_shadow_headline_shows_proposed_only_when_the_assembler_produced_one():
    assert tm.shadow_headline(False, "not_loaded", None)[:2] == ("SHADOW OFF (default)", "off")
    proposed = {"proposed": {"tactic": "ram", "probability": 0.5}, "blocked_by": [], "policy_loaded": True, "error": ""}
    assert tm.shadow_headline(True, "loaded", proposed)[0] == "PROPOSED ram (p=0.50)"
    blocked = {"proposed": None, "blocked_by": ["[0-5] imu: no IMU telemetry"], "policy_loaded": True, "error": ""}
    headline, level, detail = tm.shadow_headline(True, "loaded", blocked)
    assert headline.startswith("BLOCKED") and "PROPOSED" not in headline and level == "bad" and detail == blocked["blocked_by"]
    assert "still loading" in tm.shadow_headline(True, "loading", {"proposed": None, "blocked_by": [], "policy_loaded": False, "error": ""})[0]
    assert "PROPOSED" not in tm.shadow_headline(True, "loaded", {"proposed": None, "blocked_by": [], "policy_loaded": False, "error": ""})[0]
    assert tm.shadow_headline(True, "loaded", {"error": "boom"})[0].startswith("SHADOW ERROR")


# ---------------------------------------------------------------- telemetry from the hub's lines


def test_hub_telemetry_copies_numbers_and_never_invents_a_sample(verified_mounting):
    telemetry = HubTelemetry(load_mounting(verified_mounting))
    report = telemetry.report(0.0)
    assert not report["present"] and "no controller lines" in report["present_detail"] and telemetry.latest(0.0) is None
    telemetry.feed("RX", "ST t=1000 tok=1 link=1 armed=0 req=RADIO act=RADIO pc=0 block=NONE fault=NONE stop=NONE L=0 R=0", 1.0)
    assert not telemetry.report(1.0)["present"] and "does not send IMU data" in telemetry.report(1.0)["present_detail"]
    assert telemetry.latest(1.0) is None
    telemetry.feed("RX", "IMU,2000,0.1,-0.2,9.8,0.0,0.0,18.0", 2.0)
    assert telemetry.report(2.01)["present"] and telemetry.report(2.01)["fresh"] and telemetry.report(2.01)["mounting_verified"]
    sample = telemetry.latest(2.01)
    assert sample is not None and sample.accel_m_s2 == pytest.approx((0.1, -0.2, 9.8)) and sample.gyro_rad_s[2] == pytest.approx(np.radians(18.0))
    assert not telemetry.report(2.5)["fresh"] and "ms old" in telemetry.report(2.5)["fresh_detail"]
    telemetry.feed("TX", "IMU,3000,1,1,1,1,1,1", 3.0)                              # a TX line says nothing about the robot
    assert telemetry.reading_at == 2.0
    telemetry.feed("RX", "IMU,bad,1,1", 4.0)
    assert telemetry.malformed >= 1 and telemetry.reading_at == 2.0


def test_unverified_mounting_blocks_even_with_a_stream(tmp_path):
    unverified = ImuMounting(verified=False, path=tmp_path / "imu_mounting.yaml")
    telemetry = HubTelemetry(unverified)
    telemetry.feed("RX", "IMU,2000,0.1,-0.2,9.8,0.0,0.0,18.0", 2.0)
    report = telemetry.report(2.0)
    assert report["present"] and not report["mounting_verified"] and "NOT verified" in report["mounting_detail"]
    assert telemetry.latest(2.0) is None and "not verified" in telemetry.description


# ---------------------------------------------------------------- the engine, driven with synthetic frames


def make_engine(hub, clock=None, clicks="mock") -> TrackingEngine:
    clicks = hub.mock_source.clicks() if clicks == "mock" else clicks
    return TrackingEngine(hub.vision.config, clicks, "synthetic fixture", clock=clock or time.monotonic)


def test_engine_tracks_the_bot_draws_the_overlay_on_a_copy_and_reports_inputs(hub):
    engine = make_engine(hub)
    frame = bot_frame(hub)
    before = frame.copy()
    result = None
    for step in range(4):
        result = engine.process(step, step * DT, frame, skipped=0, queue_age_s=0.01)
    assert np.array_equal(frame, before) and result.raw is frame                      # the raw frame is never drawn on
    assert not np.array_equal(result.annotated, frame)                                  # boundary, pits, flipper and the own ring are on the copy
    assert result.map_level == "ok" and result.own["state"] == "measured" and result.own["used_ids"] == [0, 1, 2, 3]
    assert result.own["x_m"] == pytest.approx(OWN[0], abs=0.02) and result.own["y_m"] == pytest.approx(OWN[1], abs=0.02)
    assert result.evidence["fit_rms_mm"] is not None and result.evidence["spread_mm"] is not None
    assert result.software_ms > 0 and result.engine_ms >= result.software_ms and result.diagnostics["processed"] == 4
    assert result.marker_ids == [0, 1, 2, 3] and result.dictionary == "DICT_4X4_50" and result.placeholders
    assert {row.marker_id: row.status for row in result.marker_rows}[0] == "seen"
    assert {row.marker_id: row.status for row in result.marker_rows}[4].startswith("never seen (its face is not expected")


def test_engine_without_a_calibration_still_reports_marker_ids_and_says_not_configured(hub):
    engine = make_engine(hub, clicks=None)
    result = engine.process(0, 0.0, bot_frame(hub))
    assert result.map_level == "none" and "no calibration" in result.map_reason
    assert result.marker_ids == [0, 1, 2, 3] and np.array_equal(result.annotated, result.raw)    # ids still decoded, nothing drawn
    assert result.own["state"] == "lost"
    historical = ClickSet(points={"corner_origin": (1.0, 1.0)}, meta={"profile_kind": "historical"})
    assert make_engine(hub, clicks=historical).process(0, 0.0, bot_frame(hub)).map_level == "none"


def test_engine_reports_an_invalid_mapping(hub):
    clicks = hub.mock_source.clicks()
    three = ClickSet(points=dict(list(clicks.points.items())[:3]), image_size=clicks.image_size, meta=dict(clicks.meta))
    result = make_engine(hub, clicks=three).process(0, 0.0, bot_frame(hub))
    assert result.map_level == "invalid" and result.map_reason
    assert result.marker_ids == [0, 1, 2, 3]


def test_engine_rebuilds_on_a_frame_size_change_and_after_a_profile_switch(hub):
    engine = make_engine(hub)
    frame = bot_frame(hub)
    engine.process(0, 0.0, frame)
    small = cv2.resize(frame, (640, 360))
    result = engine.process(1, DT, small)
    assert result.size == (640, 360) and any("changed size" in note for note in result.notes)
    assert result.map_level == "invalid" and "click them again at this resolution" in result.map_reason     # points belong to 960x540
    assert result.marker_ids                                                           # ... but markers are still decoded and shown
    engine.rebuild(note="profile switch", epoch=3)
    result = engine.process(2, 2 * DT, frame)
    assert result.epoch == 3 and result.notes == ["profile switch"] and result.opponent["state"] == "not_selected"
    older = engine.process(3, 3 * DT, frame)
    assert older.notes == []
    again = engine.process(4, 2 * DT, frame)                                          # out of order: restarted, not crashed
    assert any("restarted" in note for note in again.notes)


def test_engine_selects_the_opponent_by_dragged_box_refuses_the_own_robot_and_clears(hub):
    engine = make_engine(hub)
    frame = bot_frame(hub, objects=[((0.20, 0.90), RED)])
    for step in range(3):
        engine.process(step, step * DT, frame)
    engine.select_roi(slab_box(hub, *OWN[:2], half=(30.0, 26.0)), frame, 3 * DT)         # a box dragged on the own robot
    result = engine.process(3, 3 * DT, frame)
    refusal = result.click_events[0]
    assert refusal["event"] == "roi" and not refusal["accepted"] and "own tagged robot" in refusal["reason"]
    assert result.opponent["state"] == "not_selected"
    engine.select_opponent(pixel(hub, 0.20, 0.90), frame, 3.5 * DT)                      # a bare click selects nothing
    refused_click = engine.process(3, 3.5 * DT, frame).click_events[0]
    assert not refused_click["accepted"] and "drag a box" in refused_click["reason"]
    engine.select_roi(slab_box(hub, 0.20, 0.90), frame, 4 * DT)
    result = engine.process(4, 4 * DT, frame)
    assert result.click_events[0]["accepted"] and result.opponent["state"] == "tracked" and result.opponent["image_based"]
    assert result.opponent["x_m"] == pytest.approx(0.20, abs=0.05) and result.opponent["y_m"] == pytest.approx(0.90, abs=0.05)
    engine.clear_opponent()
    result = engine.process(5, 5 * DT, frame)
    assert result.opponent["state"] == "not_selected" and result.click_events == [{"event": "clear", "accepted": True, "reason": "selection cleared"}]


# ---------------------------------------------------------------- the screen


def test_screen_builds_in_mock_mode_without_a_camera_and_is_titled(hub, screen):
    assert screen.windowTitle() == "ROBOT TRACKING & LIVE MONITORING"
    texts = [label.text() for label in screen.findChildren(type(screen.camera_line))]
    assert "ROBOT TRACKING & LIVE MONITORING" in texts
    assert screen.shadow_checkbox.text() == "Run policy in shadow mode (display only, nothing is sent)"
    assert not screen.shadow_checkbox.isChecked() and not screen.shadow_checkbox.isEnabled()
    assert screen.shadow_headline.text() == "SHADOW OFF (default)"
    assert not screen.worker.isRunning()                                                  # nothing runs until the tab is shown
    screen.refresh()
    assert screen.readiness().level == WARNING                                            # calibrated, but nothing verified yet
    assert "baseline firmware sends no IMU data" in " ".join(label.text() for label in screen.findChildren(type(screen.camera_line)))


def test_marker_table_in_the_screen_shows_seen_unseen_and_unknown_ids(hub, screen):
    engine = screen.engine
    stray = fx.MarkerPlacement(30, (0.20, 0.90), 0.0, 0.04, 0.0)
    all_top = render(hub, [fx.BotInstance(*OWN)], [stray])
    one_top = render(hub, [fx.BotInstance(*OWN, ids=[0])], [stray])
    screen._paint(engine.process(0, 0.0, all_top))
    result = engine.process(1, DT, one_top)
    screen._paint(result)
    table = screen.marker_table
    cells = {int(table.item(r, 0).text()): [table.item(r, c).text() for c in range(5)] for r in range(table.rowCount())}
    assert cells[0][2] == "yes" and cells[0][4] == "seen" and cells[0][3].startswith("100%")
    assert cells[1][2] == "no" and cells[1][4] == "not seen this frame" and cells[1][3].startswith("50%")
    assert cells[4][4].startswith("never seen (its face is not expected")
    assert 30 in cells and cells[30][4].startswith("UNKNOWN ID")
    assert "[30]" in screen.marker_summary.text() and "UNKNOWN ids seen" in screen.marker_summary.text()
    assert "DICT_4X4_50" in screen.dictionary_label.text()
    assert ("State", "MEASURED") in screen.own_grid.texts()
    quality = dict(screen.own_grid.texts())["Position quality"]
    assert quality in ("height-corrected", "approximate (not height-corrected)")           # the tracker's own label, one of two
    assert screen.confidence_label.text().startswith("Tracking confidence: ") and "markers agreeing" in screen.confidence_inputs.text()


def test_a_dragged_box_selects_the_opponent_through_the_widget_to_frame_mapping(hub, screen):
    frame = bot_frame(hub, objects=[((0.20, 0.90), RED)])
    screen.resize(1400, 900)
    screen.show()
    counter = iter(range(1, 100000))

    def feed():
        publish(hub, next(counter), frame)
        time.sleep(0.02)

    assert wait_until(lambda: screen._result is not None and screen._result.own["state"] == "measured", tick=lambda: (feed(), screen.refresh()))
    view = screen.frame_view
    assert view.drag_select is True
    target = view._target()
    width, height = screen._result.size

    def widget_point(u, v):
        return QPointF(target.left() + u * target.width() / width, target.top() + v * target.height() / height)

    def drag(box):
        x, y, w, h = box
        start, end = widget_point(x, y), widget_point(x + w, y + h)
        QTest.mousePress(view, Qt.MouseButton.LeftButton, pos=QPoint(round(start.x()), round(start.y())))
        QTest.mouseRelease(view, Qt.MouseButton.LeftButton, pos=QPoint(round(end.x()), round(end.y())))

    u, v = pixel(hub, 0.20, 0.90)
    back = view.widget_to_image(widget_point(u, v))
    assert back == pytest.approx((u, v), abs=1e-6)                                      # the mapping is its own inverse
    rect = view.widget_rect_to_image(widget_point(u - 30, v - 20), widget_point(u + 30, v + 20))
    assert rect == pytest.approx((u - 30, v - 20, 60, 40), abs=1e-6)                       # a dragged rectangle in image pixels
    assert view.widget_rect_to_image(QPointF(-50, -50), QPointF(-10, -10)) is None         # entirely off the picture
    # a bare click only explains: it does not say how big the object is
    QTest.mouseClick(view, Qt.MouseButton.LeftButton, pos=QPoint(round(widget_point(u, v).x()), round(widget_point(u, v).y())))
    assert "DRAG a box" in screen.click_line.text() and screen._result.opponent["state"] == "not_selected"
    screen.select_button.click()
    assert "DRAG a rectangle" in screen.click_line.text()
    drag(slab_box(hub, 0.20, 0.90))
    assert "waiting for the answer" in screen.click_line.text()
    assert wait_until(lambda: screen._result.opponent is not None and screen._result.opponent["state"] == "tracked",
                      tick=lambda: (feed(), screen.refresh()))
    assert screen.click_line.text().startswith("opponent selected")
    assert ("State", "TRACKED") in screen.opponent_grid.texts() and screen.clear_button.isEnabled() and screen.reselect_button.isEnabled()
    assert screen.status_label.text().startswith("TRACKING STATUS: TRACKED   confidence")
    rows = dict(screen.opponent_grid.texts())
    assert "image-based" in rows["Position x, y"].lower() and rows["Tracking confidence"].startswith("1.00") or "confidence" in screen.status_label.text()
    assert "never used to accept or refuse" in rows["Apparent size (diagnostic only)"]
    assert events(hub, "opponent selected")
    # the own robot cannot be selected, and a refused box leaves the track alone
    screen.reselect_button.click()
    assert "current track stays" in screen.click_line.text()
    drag(slab_box(hub, *OWN[:2], half=(30.0, 26.0)))
    assert wait_until(lambda: "box REFUSED" in screen.click_line.text(), tick=lambda: (feed(), screen.refresh())), screen.click_line.text()
    assert "own tagged robot" in screen.click_line.text() and screen._result.opponent["state"] == "tracked"
    screen.clear_button.click()
    assert wait_until(lambda: screen._result.opponent["state"] == "not_selected", tick=lambda: (feed(), screen.refresh()))
    assert not screen.clear_button.isEnabled() and not screen.reselect_button.isEnabled()
    assert screen.status_label.text().startswith("TRACKING STATUS: no opponent selected")
    drag((40.0, 40.0, 0.0, 0.0))                                                         # an empty box is refused, not tracked
    QTest.mouseClick(view, Qt.MouseButton.LeftButton, pos=QPoint(1, 1))                  # outside the picture: nothing is attempted


def test_the_screen_can_reselect_a_lost_opponent_without_restarting_anything(hub, screen):
    frame = bot_frame(hub, objects=[((0.20, 0.90), RED)])
    bare = bot_frame(hub)
    t = 0.0
    for index in range(3):
        screen._paint(screen.engine.process(index, t, frame))
        t += DT
    screen.engine.select_roi(slab_box(hub, 0.20, 0.90), frame, t)
    screen._paint(screen.engine.process(3, t, frame))
    assert screen._result.opponent["state"] == "tracked"
    for index in range(4, 80):
        t += DT
        screen._paint(screen.engine.process(index, t, bare))
    assert screen._result.opponent["state"] == "lost" and screen._result.own["state"] == "measured"
    assert screen.status_label.text().startswith("TRACKING STATUS: LOST: drag a new box") and screen.reselect_button.isEnabled()
    t += DT
    screen.engine.select_roi(slab_box(hub, 0.20, 0.90), frame, t)                        # the camera, calibration and trackers are untouched
    screen._paint(screen.engine.process(80, t, frame))
    assert screen._result.opponent["state"] == "tracked" and screen._result.opponent["selection"] == 2


def lost_sequence(hub, screen, start_t=0.0, bot_frames=4, empty_frames=40, clock_offset=1000.0):
    """Own bot visible, then gone: feed engine results to the screen and run the warning logic on the same clock."""
    bot, empty = bot_frame(hub), render(hub)
    t = start_t
    index = int(start_t / DT)
    for step in range(bot_frames + empty_frames):
        result = screen.engine.process(index, t, bot if step < bot_frames else empty)
        screen._paint(result)
        screen.update_warnings(now=t + clock_offset)
        t, index = t + DT, index + 1
    return t


def test_own_lost_warning_appears_in_banner_and_log_and_is_debounced(hub, screen):
    end = lost_sequence(hub, screen, empty_frames=45)                                     # 3 s without the robot
    assert screen._result.own["state"] == "lost"
    assert not screen.banner.isHidden() and "OWN ROBOT LOST" in screen.banner.text()
    assert len(events(hub, "OWN ROBOT STALE")) == 1 and len(events(hub, "OWN ROBOT LOST")) == 1
    lost_lines = len(events(hub, "OWN ROBOT"))
    # the robot flickers back for a few frames and disappears again: still one episode, no new log lines
    bot, empty = bot_frame(hub), render(hub)
    t, index = end, 1000
    for cycle in range(4):
        for frame in [bot] * 3 + [empty] * 20:
            screen._paint(screen.engine.process(index, t, frame))
            screen.update_warnings(now=t + 1000.0)
            t, index = t + DT, index + 1
    assert len(events(hub, "OWN ROBOT")) == lost_lines
    for _ in range(30):                                                                   # steady and measured for 2 s: the banner clears
        screen._paint(screen.engine.process(index, t, bot))
        screen.update_warnings(now=t + 1000.0)
        t, index = t + DT, index + 1
    assert "OWN ROBOT" not in screen.banner.text()
    assert events(hub, "PLACEHOLDER") and len(events(hub, "PLACEHOLDER")) == 1             # persistent conditions are logged once too


def test_opponent_lost_and_ambiguous_warn_only_after_a_selection(hub, screen):
    frame = bot_frame(hub, objects=[((0.20, 0.90), RED)])
    bare = bot_frame(hub)
    t = 0.0
    for index in range(3):
        screen._paint(screen.engine.process(index, t, frame))
        screen.update_warnings(now=t + 1000)
        t += DT
    assert not events(hub, "OPPONENT")                                                    # nothing selected: not a warning
    screen.engine.select_roi(slab_box(hub, 0.20, 0.90), frame, t)
    for index in range(3, 70):                                                            # the object vanishes
        screen._paint(screen.engine.process(index, t, bare))
        screen.update_warnings(now=t + 1000)
        t += DT
    assert screen._result.opponent["state"] == "lost"
    assert len(events(hub, "OPPONENT LOST")) == 1 and "drag a new box" in events(hub, "OPPONENT LOST")[0]
    assert events(hub, "OPPONENT TRACKING UNCERTAIN")                                     # it was uncertain first, with no position
    assert "OPPONENT LOST" in screen.banner.text()
    assert len(events(hub, "OPPONENT AMBIGUOUS")) <= 1


def test_camera_loss_mapping_and_placeholder_warnings(hub, screen):
    screen._paint(screen.engine.process(0, 0.0, bot_frame(hub)))
    hub.capture.state, hub.capture.detail = "LOST", "camera stopped delivering frames"
    screen.update_warnings(now=100.0)
    screen.update_warnings(now=100.6)
    assert "CAMERA LOST" in screen.banner.text() and events(hub, "CAMERA LOST")
    assert screen.readiness().level == FAILED and "camera" in screen.readiness().headline.lower()
    hub.capture.state, hub.capture.detail = "MOCK", ""                                    # reconnected
    screen.update_warnings(now=101.0)
    screen.update_warnings(now=102.5)
    assert "CAMERA" not in screen.banner.text()
    assert len(events(hub, "CAMERA LOST")) == 1
    assert "PLACEHOLDER config values still in use" in screen.banner.text()

    weak = dataclasses.replace(screen._result, map_level="weak", map_reason="no check point clicked")
    screen._result = weak
    screen.update_warnings(now=200.0)
    screen.update_warnings(now=200.6)
    assert "ARENA FIT IS WEAK" in screen.banner.text()
    for level, text in (("none", "NOT CONFIGURED: calibrate the arena first (ARENA tab)"), ("invalid", "ARENA MAPPING INVALID")):
        screen._result = dataclasses.replace(screen._result, map_level=level, map_reason="three points only")
        screen.update_warnings(now=300.0)
        screen.update_warnings(now=300.6)
        assert text in screen.banner.text()


def test_readiness_for_every_level(hub, screen):
    engine = make_engine(hub)
    result = None
    for step in range(4):
        result = engine.process(step, step * DT, bot_frame(hub))
    clean = dataclasses.replace(result, placeholders=[])
    ready = screen.readiness_for(clean)
    assert ready.level == READY and ready.ok and "[0, 1, 2, 3]" in ready.detail
    predicted = dataclasses.replace(clean, own={**clean.own, "state": "predicted"})
    assert screen.readiness_for(predicted).level == READY
    for state in ("stale", "lost"):
        warning = screen.readiness_for(dataclasses.replace(clean, own={**clean.own, "state": state}))
        assert warning.level == WARNING and state.upper() in warning.detail and warning.fix
    assert screen.readiness_for(dataclasses.replace(clean, own=None)).level == WARNING
    assert screen.readiness_for(dataclasses.replace(clean, map_level="weak")).level == WARNING
    assert "WEAK" in screen.readiness_for(dataclasses.replace(clean, map_level="weak")).detail
    assert screen.readiness_for(result).level == WARNING and "placeholder" in screen.readiness_for(result).detail      # the default config's placeholders
    rejected = dataclasses.replace(clean, own={**clean.own, "rejected": {"3": "disagrees with the consensus"}})
    assert screen.readiness_for(rejected).level == WARNING
    assert screen.readiness_for(dataclasses.replace(clean, map_level="none")).level == NOT_CONFIGURED
    assert screen.readiness_for(dataclasses.replace(clean, map_level="invalid", map_reason="3 points")).level == FAILED
    assert screen.readiness_for(dataclasses.replace(clean, arena_problems=["no pits in the arena geometry"])).level == FAILED
    hub.capture.state = "LOST"
    assert screen.readiness_for(clean).level == FAILED
    hub.capture.state = "UNAVAILABLE"
    assert screen.readiness_for(None).level == FAILED
    hub.capture.state = "MOCK"
    assert screen.readiness_for(None).level == WARNING                                    # calibrated but not verified (hidden tab)
    hub.vision.set_clicks(None, "live", "no live calibration saved yet")
    assert screen.readiness_for(None).level == NOT_CONFIGURED and "ARENA" in screen.readiness_for(None).fix


def test_worker_starts_on_show_runs_off_the_ui_thread_pauses_when_hidden_and_stops_on_shutdown(hub, screen, monkeypatch):
    main = threading.get_ident()
    where = {"process": set(), "paint": []}
    original = screen.engine.process

    def spy(*args, **kwargs):
        where["process"].add(threading.get_ident())
        return original(*args, **kwargs)

    painter = screen.frame_view.set_frame

    def paint(frame):
        where["paint"].append(threading.get_ident())
        painter(frame)

    monkeypatch.setattr(screen.engine, "process", spy)
    monkeypatch.setattr(screen.frame_view, "set_frame", paint)
    frame = bot_frame(hub)
    counter = iter(range(1, 100000))

    def feed():
        publish(hub, next(counter), frame)
        time.sleep(0.02)

    assert not screen.worker.isRunning()
    screen.show()
    assert screen.worker.isRunning() and not screen.worker.paused
    assert wait_until(lambda: screen._result is not None, tick=lambda: (feed(), screen.refresh()))
    assert where["process"] == {screen.worker.thread_ident} and main not in where["process"]       # vision work is not on the GUI thread
    assert where["paint"] and set(where["paint"]) == {main}                                          # painting is
    latest = screen.worker.latest()
    screen.refresh()
    assert screen._painted_seq == latest.seq or screen._painted_seq == screen.worker.latest().seq

    screen.hide()                                                                         # hidden: the vision thread pauses
    assert screen.worker.paused
    pump(0.15)
    processed = screen.worker.processed
    for _ in range(10):
        feed()
    pump(0.3)
    assert screen.worker.processed == processed
    screen.show()
    assert not screen.worker.paused
    assert wait_until(lambda: screen.worker.processed > processed, tick=lambda: (feed(), screen.refresh()))

    screen.shutdown()
    assert not screen.worker.isRunning()
    screen.shutdown()                                                                     # idempotent
    screen.refresh()                                                                      # and harmless afterwards


def test_gui_paints_only_the_latest_finished_result_and_the_worker_skips_frames_it_cannot_keep_up_with(hub, screen, monkeypatch):
    engine = screen.engine
    first = engine.process(0, 0.0, bot_frame(hub))
    second = engine.process(1, DT, bot_frame(hub))
    painted = []
    original = screen.frame_view.set_frame
    monkeypatch.setattr(screen.frame_view, "set_frame", lambda image: (painted.append(image), original(image)))
    screen.worker._latest = second                                                        # two results existed; only the newest is kept
    screen.refresh()
    screen.refresh()
    screen.refresh()
    assert painted == [second.annotated] and painted[0] is second.annotated and screen._painted_seq == second.seq
    assert first.seq != second.seq

    seen_skips = []
    real = engine.process

    def slow(index, t, frame, skipped=0, queue_age_s=0.0):
        time.sleep(0.05)
        result = real(index, t, frame, skipped, queue_age_s)
        seen_skips.append(skipped)
        return result

    monkeypatch.setattr(engine, "process", slow)
    frame = bot_frame(hub)
    screen.show()
    for index in range(100, 130):
        publish(hub, index, frame)
        time.sleep(0.01)
    assert wait_until(lambda: sum(seen_skips) > 0 and len(seen_skips) >= 2, timeout=10)
    assert len(seen_skips) < 30 and sum(seen_skips) > 0                                   # old frames were dropped, never queued


def test_profile_switch_clears_selection_trails_and_rebuilds_from_the_new_config(hub, screen, monkeypatch):
    frame = bot_frame(hub, objects=[((0.20, 0.90), RED)])
    for index in range(3):
        screen._paint(screen.engine.process(index, index * DT, frame))
    screen.engine.select_roi(slab_box(hub, 0.20, 0.90), frame, 3 * DT)
    screen._paint(screen.engine.process(3, 3 * DT, frame))
    assert screen._result.opponent["state"] == "tracked" and screen.own_trail.runs()
    monkeypatch.setattr(hub, "profile_name", "home_test", raising=False)
    screen.refresh()
    assert screen._result is None and screen.own_trail.runs() == [] and screen._epoch == 1
    assert events(hub, "home_test") and "select the opponent again" in screen.click_line.text()
    assert screen.profile_name() == "home_test" and "home_test" in screen.profile_label.text()
    screen.worker._apply()                                                                # what the vision thread does with the command
    assert screen.engine.epoch == 1 and screen.engine.track is None
    result = screen.engine.process(4, 4 * DT, frame)
    assert result.epoch == 1 and result.opponent["state"] == "not_selected" and result.own["state"] == "measured"
    screen.worker._latest = result
    screen.refresh()
    assert screen._result is result and screen._epoch == 1
    stale = dataclasses.replace(result, epoch=0, seq=result.seq + 1)                     # a result of the old profile still in flight
    screen.worker._latest = stale
    screen.refresh()
    assert screen._result is result


def test_a_profile_changed_signal_from_the_hub_is_honoured(qapp, tmp_path):
    class SignalHub(OperationsHub):
        profile_changed = Signal(str)

    hub = SignalHub(load_config(ROOT / "config" / "control_center.yaml"),
                    HubOptions(mock=True, start_camera=False, rounds_root=tmp_path / "rounds"))
    screen = TrackingScreen(hub, mounting_path=tmp_path / "none.yaml")
    try:
        hub.profile_name = "home_test"
        hub.profile_changed.emit("home_test")
        assert screen._epoch == 1 and screen.profile_name() == "home_test"
        screen.shutdown()
        hub.profile_name = "live"
        hub.profile_changed.emit("live")                                                  # disconnected after shutdown
        assert screen._epoch == 1
    finally:
        screen.shutdown()
        hub.close()


def test_the_arena_calibration_changing_in_place_is_noticed(hub, screen, monkeypatch):
    screen.refresh()
    assert screen._epoch == 0
    monkeypatch.setitem(hub.vision.config.arena, "pit_radii_m", [0.2, 0.3])
    screen.refresh()
    assert screen._epoch == 1 and [r for _, r in tm.pit_list(screen.map.arena)] == [0.2, 0.3]
    monkeypatch.delitem(hub.vision.config.arena, "pit_radii_m")


# ---------------------------------------------------------------- shadow mode


def test_shadow_lists_every_prerequisite_and_stays_disabled_without_imu(hub, screen):
    screen._paint(screen.engine.process(0, 0.0, bot_frame(hub)))
    screen._refresh_shadow(time.monotonic())
    table = screen.prereq_table
    rows = {table.item(r, 0).text().replace("  (live input)", ""): (table.item(r, 1).text(), table.item(r, 2).text()) for r in range(table.rowCount())}
    assert set(rows) == {"IMU telemetry present", "IMU telemetry fresh", "IMU mounting verified", "Arena mapping valid",
                         "Profile layout matches the simulator frame", "Checkpoint record readable", "Own pose valid",
                         "Opponent position valid", "Checkpoint loaded"}
    assert rows["IMU telemetry present"][0] == "MISSING" and "no controller lines" in rows["IMU telemetry present"][1]
    assert rows["Arena mapping valid"][0] == "Ready" and rows["Own pose valid"][0] == "Ready"
    assert rows["Opponent position valid"][0] == "MISSING" and rows["Checkpoint loaded"][0] == "MISSING"
    assert not screen.shadow_checkbox.isEnabled() and not screen.shadow_checkbox.isChecked()
    assert screen.shadow_why.text().startswith("blocked:") and "IMU telemetry present" in screen.shadow_why.text()
    assert screen.shadow_headline.text() == "SHADOW OFF (default)"
    assert "Hub says it still lacks" in screen.shadow_hub.text()
    assert screen.telemetry.latest(time.monotonic()) is None                              # no IMU number exists anywhere
    screen.shadow_checkbox.setChecked(True)                                               # even a forced tick is refused
    assert not screen.shadow_checkbox.isChecked() and "refused" in screen.shadow_why.text() + screen._shadow_message
    assert screen.worker._commands is not None and not any(c[0] == "shadow" and c[1] is not None for c in screen.worker._commands)


def test_an_unverified_imu_mounting_keeps_the_box_disabled_even_with_a_stream(hub, make_screen, tmp_path):
    screen = make_screen(mounting_path=tmp_path / "missing.yaml")
    screen._paint(screen.engine.process(0, 0.0, bot_frame(hub)))
    hub.controller_line.emit("RX", "IMU,1000,0.1,0.0,9.8,0.0,0.0,5.0")
    screen._refresh_shadow(time.monotonic())
    rows = {screen.prereq_table.item(r, 0).text(): screen.prereq_table.item(r, 2).text() for r in range(screen.prereq_table.rowCount())}
    assert "IMU telemetry present" in rows and "IMU lines arriving" in rows["IMU telemetry present"]
    assert "NOT verified" in rows["IMU mounting verified"]
    assert not screen.shadow_checkbox.isEnabled() and "IMU mounting verified" in screen.shadow_why.text()


def test_a_profile_not_in_the_simulator_frame_refuses_shadow_with_the_reason(hub, make_screen, verified_mounting, monkeypatch):
    screen = make_screen(mounting_path=verified_mounting, contract_loader=lambda: type("C", (), {"identifier": "stub", "arena_width_m": 1.15, "arena_height_m": 1.15})())
    screen._paint(screen.engine.process(0, 0.0, bot_frame(hub)))
    hub.controller_line.emit("RX", "IMU,1000,0.1,0.0,9.8,0.0,0.0,5.0")
    screen._refresh_shadow(time.monotonic())
    assert screen.shadow_checkbox.isEnabled()                                              # everything cheap holds
    monkeypatch.setitem(hub.vision.config.arena, "policy_frame_ok", False)
    screen._signature = screen._calibration_signature()                                   # (the in-place change is not what is tested here)
    hub.controller_line.emit("RX", "IMU,1001,0.1,0.0,9.8,0.0,0.0,5.0")
    screen._refresh_shadow(time.monotonic())
    assert not screen.shadow_checkbox.isEnabled() and "policy_frame_ok is false" in screen.shadow_why.text()
    assert "shadow is refused" in screen.shadow_why.text()


@needs_checkpoint
def test_shadow_loads_the_checkpoint_off_the_ui_thread_with_progress_and_shows_why_there_is_no_proposal(hub, make_screen, verified_mounting):
    policies: list[FakePolicy] = []

    def factory(contract):
        policies.append(FakePolicy(contract, delay=0.6))
        return policies[-1]

    screen = make_screen(mounting_path=verified_mounting, policy_factory=factory)
    frame = bot_frame(hub)
    counter = iter(range(1, 100000))

    def feed():
        publish(hub, next(counter), frame)
        hub.controller_line.emit("RX", "IMU,1000,0.1,0.0,9.8,0.0,0.0,5.0")
        time.sleep(0.02)

    screen.show()
    assert wait_until(lambda: screen._result is not None and screen.shadow_checkbox.isEnabled(),
                      tick=lambda: (feed(), screen.refresh(), screen._refresh_shadow(time.monotonic())))
    feed()
    screen.shadow_checkbox.setChecked(True)
    assert screen._shadow_on and screen._load_state == "loading", (screen._shadow_message, screen.shadow_why.text())
    started = time.monotonic()
    for _ in range(6):                                                                     # the GUI keeps running while the load takes 0.6 s
        feed()
        screen.refresh()
    assert time.monotonic() - started < 0.5 and screen._load_state == "loading"            # the GUI thread was not blocked by the load
    screen._refresh_shadow(time.monotonic())
    assert "Loading the checkpoint on a background thread" in screen.load_label.text()
    assert wait_until(lambda: screen._load_state == "loaded", tick=lambda: (feed(), screen.refresh()))
    assert policies[0].load_thread not in (None, threading.get_ident()) and policies[0].loaded
    assert wait_until(lambda: screen._result.shadow is not None and screen._result.shadow["policy_loaded"],
                      tick=lambda: (feed(), screen.refresh(), screen._refresh_shadow(time.monotonic())))
    summary = screen._result.shadow
    assert summary["proposed"] is None and summary["blocked_by"]                           # no opponent selected: blocked, and says so
    assert any("opponent" in entry for entry in summary["blocked_by"])
    assert screen.shadow_headline.text().startswith("BLOCKED") and "PROPOSED" not in screen.shadow_headline.text()
    assert policies[0].proposals == 0                                                      # the policy was never asked
    screen.shadow_checkbox.setChecked(False)
    assert not screen._shadow_on
    assert wait_until(lambda: screen._result.shadow is None, tick=lambda: (feed(), screen.refresh()))
    assert screen.shadow_headline.text() == "SHADOW OFF (default)"


def test_a_brief_imu_gap_only_blocks_proposals_but_a_lasting_one_switches_shadow_off(hub, make_screen, verified_mounting):
    stub = type("C", (), {"identifier": "stub", "arena_width_m": 1.15, "arena_height_m": 1.15})()
    screen = make_screen(mounting_path=verified_mounting, contract_loader=lambda: stub, policy_factory=lambda c: FakePolicy(c, 5.0))
    screen._paint(screen.engine.process(0, 0.0, bot_frame(hub)))
    hub.controller_line.emit("RX", "IMU,1000,0.1,0.0,9.8,0.0,0.0,5.0")
    now = time.monotonic()
    screen._refresh_shadow(now)
    assert screen.shadow_checkbox.isEnabled()
    screen.shadow_checkbox.setChecked(True)
    assert screen._shadow_on
    screen._refresh_shadow(now + 0.5)                                                      # IMU 0.5 s old: stale, but only just
    assert screen._shadow_on and screen.shadow_checkbox.isChecked() and screen.shadow_checkbox.isEnabled()
    screen._refresh_shadow(now + 1.0)
    assert screen._shadow_on
    screen._refresh_shadow(now + 3.0)                                                      # missing for 2.5 s: switched off, with the reason
    assert not screen._shadow_on and not screen.shadow_checkbox.isChecked() and not screen.shadow_checkbox.isEnabled()
    assert "shadow stopped" in screen.shadow_why.text() and "IMU telemetry fresh" in screen.shadow_why.text()
    assert any(c[0] == "shadow" and c[1] is None for c in screen.worker._commands)


@needs_checkpoint
def test_shadow_proposal_appears_only_when_every_input_is_real_and_fresh(hub, verified_mounting):
    contract = load_contract(RECOMMENDED_CHECKPOINT)
    clock = Clock(0.0)
    engine = make_engine(hub, clock=clock)
    telemetry = HubTelemetry(load_mounting(verified_mounting))
    policy = FakePolicy(contract)
    runner = ShadowRunner(contract, hub.vision.config, hub.mock_source.clicks(), telemetry, "synthetic frames", policy)
    engine.set_shadow(runner)
    frame = bot_frame(hub, objects=[((0.20, 0.90), RED)])

    def step(index, imu=True, select=False):
        clock.t = index * DT
        if imu:
            telemetry.feed("RX", f"IMU,{index * 66},0.0,0.0,9.81,0.0,0.0,{10.0 + index}", clock.t)
        if select:
            engine.select_roi(slab_box(hub, 0.20, 0.90), frame, clock.t)
        return engine.process(index, clock.t, frame)

    no_imu = step(0, imu=False)
    assert no_imu.shadow["proposed"] is None and any("imu" in entry for entry in no_imu.shadow["blocked_by"])
    assert runner.last_decision.observation is None and policy.proposals == 0              # no value was filled in
    assert no_imu.shadow["output"].startswith("none: display only")
    no_opponent = step(1)
    assert no_opponent.shadow["proposed"] is None and any("opponent" in entry for entry in no_opponent.shadow["blocked_by"])
    assert not any("imu" in entry for entry in no_opponent.shadow["blocked_by"]) and policy.proposals == 0
    complete = step(2, select=True)
    complete = step(3)
    assert complete.opponent["state"] == "tracked" and complete.own["state"] == "measured"
    assert complete.shadow["blocked_by"] == [] and complete.shadow["proposed"] == {"action": 0, "tactic": contract.tactics[0], "probability": 0.5}
    assert policy.proposals >= 1
    imu_values = runner.last_decision.inputs["imu"].values
    assert imu_values["gyro_z_rad_s"] == pytest.approx(np.radians(10.0 + 3), abs=1e-6)   # exactly the received number, rotated by the verified mounting
    assert tm.shadow_headline(True, "loaded", complete.shadow)[0] == f"PROPOSED {contract.tactics[0]} (p=0.50)"
    stale_imu = step(4, imu=False)                                                         # the IMU stream stops: fresh within 0.10 s only
    clock.t = 4 * DT + 0.5
    telemetry_stale = engine.process(5, clock.t, frame)
    assert telemetry_stale.shadow["proposed"] is None and any("imu" in entry for entry in telemetry_stale.shadow["blocked_by"])
    assert stale_imu.shadow is not None


# ---------------------------------------------------------------- read-only by construction


def _imports_and_calls(path: Path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imported, hub_attributes, called = set(), set(), set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add(node.module or "")
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.Attribute):
            value = node.value
            if (isinstance(value, ast.Attribute) and value.attr == "hub") or (isinstance(value, ast.Name) and value.id == "hub"):
                hub_attributes.add(node.attr)
            if isinstance(node.ctx, ast.Load):
                called.add(node.attr)
    return imported, hub_attributes, called


@pytest.mark.parametrize("name", ["screens/tracking.py", "backend/tracking_worker.py", "backend/tracking_model.py"])
def test_the_tracking_modules_import_no_serial_or_socket_and_never_write_to_a_controller_link(name):
    imported, hub_attributes, used = _imports_and_calls(ROOT / "control_center" / name)
    forbidden_modules = {"serial", "socket", "bluetooth", "usb", "pyserial"}
    assert not {i.split(".")[0] for i in imported} & forbidden_modules
    forbidden_names = {"SerialLineSource", "TapSubscriber", "TelemetryRelay", "RelaySource", "ClientLogSource", "ScriptedSource",
                       "command_manager", "robot_link", "sendto", "sendall", "send", "write", "writelines", "connect_controller",
                       "connect_controller_port", "connect_controller_log", "start_recording", "reopen_camera", "VideoCapture"}
    assert not (imported | used) & forbidden_names, sorted((imported | used) & forbidden_names)
    # every hub attribute the screen touches is a read of state, a log line or a listen-only signal
    allowed = {"vision", "capture", "options", "events", "add_event", "live_shadow_missing", "profile_name", "profile_changed",
               "controller_line", "mock_source"}
    assert hub_attributes <= allowed, sorted(hub_attributes - allowed)


def test_the_screen_has_no_button_that_could_command_anything(screen):
    from PySide6.QtWidgets import QAbstractButton

    labels = sorted(button.text() for button in screen.findChildren(QAbstractButton) if button.text())     # (scroll-bar and corner buttons have no text)
    assert labels == ["Clear Opponent", "Reselect Opponent", "Run policy in shadow mode (display only, nothing is sent)",
                      "Select Opponent / Draw Box"]


# ---------------------------------------------------------------- an own-robot report that stops being refreshed must not read as current


def test_the_own_robot_panel_ages_into_stale_and_lost_when_no_new_result_arrives(hub, screen):
    """SYNTHETIC frames, mock hub. The newest result says MEASURED; if the vision thread or the camera then goes quiet, the SCREEN
    (painting on its own clock) must show the pose as old, then STALE (display only), then LOST, with the age growing."""
    result = screen.engine.process(0, 0.0, bot_frame(hub))
    screen._paint(result)
    assert result.own["state"] == "measured" and ("State", "MEASURED") in screen.own_grid.texts()
    rows = lambda: dict(screen.own_grid.texts())
    ids = rows()["Marker IDs the pose rests on"]
    assert ids == "[0, 1, 2, 3]" and rows()["Marker IDs in the newest frame"] == "[0, 1, 2, 3]" and rows()["Orientation"].startswith("UPRIGHT")
    assert "Heading" in rows() and "Position x, y" in rows() and screen.confidence_label.text().startswith("Tracking confidence: ")

    screen._paint_live(result.done_at + 0.05)
    assert ("State", "MEASURED") in screen.own_grid.texts() and rows()["Age"].startswith("50 ms")        # still current, age shown
    screen._paint_live(result.done_at + 0.5)
    assert rows()["State"].startswith("STALE") and "not valid" in rows()["State"] and rows()["Age"].startswith("500 ms")
    assert "display only" in rows()["Position x, y"] and "NONE" in screen.confidence_label.text()
    assert screen.map.markers and "STALE" in screen.map.markers[0].label and "500 ms" in screen.map.markers[0].label and screen.map.markers[0].hollow
    screen.update_warnings(now=result.done_at + 0.5)
    screen.update_warnings(now=result.done_at + 1.0)
    assert "OWN ROBOT STALE" in screen.banner.text()
    screen._paint_live(result.done_at + 2.5)
    assert rows()["State"] == "LOST" and rows()["Position x, y"] == "-" and not any(m.label.startswith("Capri") for m in screen.map.markers)
    assert screen.readiness_for(result, own=screen._own_display).level == WARNING and "LOST" in screen.readiness_for(result, own=screen._own_display).headline.upper()

    # a new result brings it straight back
    screen._paint(screen.engine.process(1, DT, bot_frame(hub)))
    screen._paint_live(time.monotonic())
    assert rows()["State"] == "MEASURED"


def test_a_flipped_robot_is_labelled_on_the_tracking_screen(hub, screen):
    frame = tf.render_bot(hub.vision.config, hub.mock_source.camera, *OWN, orientation="inverted")
    result = screen.engine.process(0, 0.0, frame)
    screen._paint(result)
    rows = dict(screen.own_grid.texts())
    assert result.own["flipped"] is True and rows["Orientation"].startswith("UPSIDE DOWN / FLIPPED") and rows["Marker IDs the pose rests on"] == "[4, 5, 6, 7]"
    assert "project dictionary" in screen.dictionary_label.text()


def test_markers_decoded_but_all_rejected_as_the_wrong_size_raise_a_banner(hub, screen):
    result = screen.engine.process(0, 0.0, bot_frame(hub))
    own = dict(result.own, state="lost", valid_for_observation=False, x_m=None, y_m=None, heading_deg=None, used_ids=[],
               rejected={"1": "measures 14 mm but is configured as 40 mm (size, height or dictionary wrong); the configured size is a PLACEHOLDER",
                         "2": "measures 13 mm but is configured as 40 mm (size, height or dictionary wrong)"})
    screen._paint(dataclasses.replace(result, own=own))
    conditions = screen.conditions(time.monotonic())
    assert "marker_size" in conditions and conditions["marker_size"].severity == "bad"
    assert "WRONG SIZE" in conditions["marker_size"].text and "14 mm" in conditions["marker_size"].text and "+1 more" in conditions["marker_size"].text
    screen._paint(result)                                                                  # measured again: the condition is gone
    assert "marker_size" not in screen.conditions(time.monotonic())
