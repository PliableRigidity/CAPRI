"""Click-selected opponent tracker, the live-track viewer loop, the opt-in flags and the marker survey.

Evidence class: FIXTURE only. Every picture is rendered by overlay_fixture / track_fixture (SYNTHETIC,
ideal pinhole camera, flat coloured slabs for the unmarked objects). Camera mode is driven through a
stand-in capture device and window: no real camera, serial port or robot link is opened. None of the
tolerances below is a real-camera accuracy figure.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import cv2
import numpy as np
import pytest

from control_center.tools import live_shadow, live_track, marker_survey
from control_center.tools import shadow_stream_demo as demo
from control_center.tools import track_fixture as tf
from control_center.tools.observation_contract import RECOMMENDED_CHECKPOINT, load_contract
from control_center.tools.opponent_select import (AMBIGUOUS, LOST, NOT_SELECTED, TRACKED, ClickedOpponentTracker, OpponentSettings,
                                                  OpponentState)
from control_center.tools.overlay_core import BotPose, load_config, save_clicks
from control_center.tools.overlay_fixture import MarkerPlacement, default_camera, fixture_clicks
from control_center.tools.stream_adapter import CameraPipeline, fit_profile

REPO = Path(__file__).resolve().parents[1]
DT = 1.0 / 30.0
RED, BLUE = (40, 40, 220), (200, 90, 30)
SIZE = (0.11, 0.10)
OWN = (0.30, 0.30, 0.3)
needs_checkpoint = pytest.mark.skipif(not RECOMMENDED_CHECKPOINT.is_file(), reason="the recommended checkpoint is not in runs/")


@pytest.fixture(scope="module")
def config():
    return load_config()


@pytest.fixture(scope="module")
def camera():
    return default_camera()


@pytest.fixture(scope="module")
def calibration(config, camera):
    mapping, estimate, _ = fit_profile(config, fixture_clicks(config, camera), camera.image_size)
    assert mapping.valid
    return mapping, estimate


def scene(config, camera, objects=(), own=OWN, ids=None, seed=3):
    """SYNTHETIC frame: the tagged own bot and flat coloured slabs ((x, y), colour[, size])."""
    frame = tf.render_bot(config, camera, *own, ids=ids, seed=seed)
    for item in objects:
        tf.paint_box(frame, camera, item[0], 0.2, item[2] if len(item) > 2 else SIZE, 0.034, item[1])
    return frame


def pixel(camera, x, y, height=0.034):
    u, v = camera.project([[x, y, height]])[0]
    return float(u), float(v)


def own_pose(x=OWN[0], y=OWN[1]):
    return BotPose(state="ok", reason="ok", x_m=x, y_m=y, heading_rad=OWN[2], floor_projection_xy=(x, y), used_ids=[0])


def error_mm(estimate, x, y):
    return 1000.0 * math.hypot(estimate.xy_m[0] - x, estimate.xy_m[1] - y)


# ---- selecting ------------------------------------------------------------------------------------------

def test_a_click_selects_the_object_and_bad_clicks_are_refused_with_the_reason(config, camera, calibration):
    tracker = ClickedOpponentTracker(calibration[0], config.arena)
    frame = scene(config, camera, [((0.70, 0.70), RED)])
    start = tracker.update(frame, 0.0, own_pose())
    assert start.state == NOT_SELECTED and start.xy_m is None and not tracker.to_opponent_estimate().ok
    refusals = {"own (tagged) robot": pixel(camera, *OWN[:2]), "stands out from the floor": pixel(camera, 0.50, 0.20, 0.0),
                "off the arena floor": (5.0, 5.0), "outside the camera picture": (-3.0, 10.0)}
    for expected, click in refusals.items():
        result = tracker.select(frame, click, 0.0, own_pose())
        assert not result and expected in result.reason and tracker.state == NOT_SELECTED, (expected, result.reason)
        assert "REFUSED" in tracker.to_report()["selection_note"]
    assert "implausibly large" in tracker.select(scene(config, camera, [((0.70, 0.70), RED, (0.40, 0.40))]), pixel(camera, 0.70, 0.70), 0.0, own_pose()).reason
    assert "implausibly small" in tracker.select(scene(config, camera, [((0.70, 0.70), RED, (0.02, 0.02))]), pixel(camera, 0.70, 0.70), 0.0, own_pose()).reason
    touching = scene(config, camera, [((OWN[0] + 0.125, OWN[1] + 0.02), RED)])
    assert not tracker.select(touching, pixel(camera, OWN[0] + 0.14, OWN[1] + 0.02), 0.0, own_pose())            # merged with / next to the own bot
    assert tracker.state == NOT_SELECTED and tracker.selection_id == 0

    result = tracker.select(frame, pixel(camera, 0.70, 0.70), 0.0, own_pose())
    assert result.ok and tracker.state == TRACKED and tracker.selection_id == 1 and "mm on the floor plane" in result.reason
    estimate = tracker.update(frame, 0.0, own_pose())
    assert estimate.state == TRACKED and error_mm(estimate, 0.70, 0.70) < 40.0 and "approximate" in estimate.position_quality
    opponent = tracker.to_opponent_estimate()
    assert opponent.ok and opponent.xy_m == estimate.xy_m and "clicked object #1" in opponent.source and "not height-corrected" in opponent.source
    report = tracker.to_report()
    assert report["state"] == "tracked" and report["box"] and report["x_m"] == pytest.approx(estimate.xy_m[0], abs=1e-4)
    refused = tracker.select(frame, pixel(camera, 0.50, 0.20, 0.0), DT, own_pose())                                # a stray click on the floor
    assert not refused and tracker.state == TRACKED and tracker.selection_id == 1                                  # ... leaves the track alone
    tracker.clear()
    assert tracker.update(frame, 2 * DT, own_pose()).state == NOT_SELECTED and isinstance(tracker.state, OpponentState)
    without_map = ClickedOpponentTracker(type(calibration[0])(), config.arena)
    assert "no valid floor map" in without_map.select(frame, pixel(camera, 0.70, 0.70), 0.0).reason
    with pytest.raises(ValueError):
        OpponentSettings.from_dict({"hold": 1})


def test_selection_without_an_own_pose_says_it_could_not_check_and_never_reports_the_own_bot(config, camera, calibration):
    tracker = ClickedOpponentTracker(calibration[0], config.arena)
    frame = scene(config, camera, [], ids=[])                          # every tag covered: the own bot is just a dark plate
    result = tracker.select(frame, pixel(camera, *OWN[:2]), 0.0)
    assert result.ok and "not checked against the own robot" in result.reason                                      # nothing said it was ours
    assert tracker.update(frame, DT, None).state == TRACKED
    for index in range(2, 6):                                          # the tag is read again: the track is sitting on the own bot
        estimate = tracker.update(scene(config, camera, [], seed=index), index * DT, own_pose())
        assert estimate.state != TRACKED and estimate.xy_m is None and not tracker.to_opponent_estimate().ok


# ---- following ------------------------------------------------------------------------------------------

def follow(tracker, config, camera, frames, start_t=0.0, own=True):
    """frames: a list of object lists. Returns (estimate, objects) per frame."""
    out = []
    for index, objects in enumerate(frames):
        t = start_t + (index + 1) * DT
        estimate = tracker.update(scene(config, camera, objects, seed=40 + index), t, own_pose() if own else None)
        out.append((estimate, objects))
    return out


def test_identity_is_kept_past_a_different_object_and_is_ambiguous_beside_a_similar_one(config, camera, calibration):
    def crossing(colour, gap):
        tracker = ClickedOpponentTracker(calibration[0], config.arena)
        first = scene(config, camera, [((0.70, 0.75), RED), ((0.95, 0.75 - gap), colour)])
        assert tracker.select(first, pixel(camera, 0.70, 0.75), 0.0, own_pose())
        frames = []
        for index in range(30):                                        # the opponent drifts; the other object drives past at 0.6 m/s
            frames.append([((0.70 + 0.002 * index, 0.75), RED), ((0.95 - 0.020 * index, 0.75 - gap), colour)])
        return tracker, follow(tracker, config, camera, frames)

    tracker, results = crossing(BLUE, 0.17)                             # a different-looking object passes 0.17 m away
    assert all(estimate.state == TRACKED for estimate, _ in results)
    assert max(error_mm(estimate, *objects[0][0]) for estimate, objects in results) < 40.0                  # never jumped to the blue one
    assert max(estimate.candidates for estimate, _ in results) == 2                                         # it did see the other object

    tracker, results = crossing(RED, 0.13)                              # a same-looking object passes within reach
    states = [estimate.state for estimate, _ in results]
    assert AMBIGUOUS in states and all(estimate.xy_m is None for estimate, _ in results if estimate.state != TRACKED)
    first = states.index(AMBIGUOUS)
    assert "similar objects" in results[first][0].reason and all(state == TRACKED for state in states[:first])
    assert all(error_mm(estimate, *objects[0][0]) < 40.0 for estimate, objects in results if estimate.state == TRACKED)   # never the wrong one
    assert not tracker.to_opponent_estimate().ok or tracker.state == TRACKED

    merged = ClickedOpponentTracker(calibration[0], config.arena)       # a DIFFERENT-looking object drives into the opponent: one region
    assert merged.select(scene(config, camera, [((0.70, 0.75), RED)]), pixel(camera, 0.70, 0.75), 0.0, own_pose())
    results = follow(merged, config, camera, [[((0.70, 0.75), RED), ((0.70, 0.75 - 0.105), BLUE)]] * 6)
    assert all(estimate.state == AMBIGUOUS and estimate.xy_m is None and "does not match" in estimate.reason for estimate, _ in results)


def test_contact_with_the_own_bot_is_ambiguous_with_no_position_and_recovers_when_they_part(config, camera, calibration):
    tracker = ClickedOpponentTracker(calibration[0], config.arena)
    start = np.array([0.80, 0.80])
    assert tracker.select(scene(config, camera, [(tuple(start), RED)]), pixel(camera, *start), 0.0, own_pose())
    direction = (np.array(OWN[:2]) - start) / np.linalg.norm(np.array(OWN[:2]) - start)
    path = [start + direction * min(0.6 * (index + 1) * DT, 0.582) for index in range(45)]                   # drives in, stops 0.125 m away
    path += [path[-1] - direction * 0.6 * (index + 1) * DT for index in range(25)]                           # and backs off again
    results = follow(tracker, config, camera, [[(tuple(point), RED)] for point in path])
    distances = [float(np.linalg.norm(point - np.array(OWN[:2]))) for point in path]
    for (estimate, _), distance in zip(results, distances):
        if distance < 0.15:                                            # outlines touching or nearly so
            assert estimate.state == AMBIGUOUS and estimate.xy_m is None and "own robot" in estimate.reason, (distance, estimate.reason)
            assert not estimate.to_opponent_estimate().ok and "AMBIGUOUS" in estimate.to_opponent_estimate().reason
            assert estimate.to_report()["x_m"] is None
        if distance > 0.24:
            assert estimate.state == TRACKED and error_mm(estimate, *path[results.index((estimate, _))]) < 40.0
    states = [estimate.state for estimate, _ in results]
    assert states[0] == TRACKED and states[-1] == TRACKED and states.count(AMBIGUOUS) >= 20 and LOST not in states
    assert sum(1 for estimate, _ in results if estimate.state != TRACKED and estimate.xy_m is not None) == 0


def test_a_vanished_object_is_lost_and_nothing_else_is_picked_up_until_reselected(config, camera, calibration):
    tracker = ClickedOpponentTracker(calibration[0], config.arena)                                           # hold_s 0.5
    where = (0.75, 0.70)
    assert tracker.select(scene(config, camera, [(where, RED)]), pixel(camera, *where), 0.0, own_pose())
    seen = follow(tracker, config, camera, [[(where, RED)]] * 3)
    assert all(estimate.state == TRACKED for estimate, _ in seen)
    gone = follow(tracker, config, camera, [[]] * 20, start_t=3 * DT)                                        # it disappears
    for index, (estimate, _) in enumerate(gone):
        missing = index * DT                                           # time since the first frame without it
        assert estimate.xy_m is None
        assert estimate.state == (AMBIGUOUS if missing <= 0.5 else LOST), (index, estimate.state, estimate.reason)
    assert "click the opponent again" in tracker.reason and "no acceptable match" in tracker.reason
    # The same-looking object back at the same place, and another one elsewhere: still LOST, nothing is re-acquired.
    back = follow(tracker, config, camera, [[(where, RED), ((0.5, 0.8), RED)]] * 10, start_t=23 * DT)
    assert all(estimate.state == LOST and estimate.xy_m is None for estimate, _ in back) and not tracker.to_opponent_estimate().ok
    again = scene(config, camera, [(where, RED), ((0.5, 0.8), RED)])
    assert tracker.select(again, pixel(camera, 0.5, 0.8), 34 * DT, own_pose()) and tracker.selection_id == 2
    estimate = tracker.update(again, 35 * DT, own_pose())
    assert estimate.state == TRACKED and error_mm(estimate, 0.5, 0.8) < 40.0 and estimate.selection_id == 2

    brief = ClickedOpponentTracker(calibration[0], config.arena)       # a short gap recovers on its own: same look, same place
    assert brief.select(scene(config, camera, [(where, RED)]), pixel(camera, *where), 0.0, own_pose())
    states = [estimate.state for estimate, _ in follow(brief, config, camera, [[(where, RED)]] * 2 + [[]] * 6 + [[(where, RED)]] * 3)]
    assert states == [TRACKED] * 2 + [AMBIGUOUS] * 6 + [TRACKED] * 3
    changed = ClickedOpponentTracker(calibration[0], config.arena)     # ... but not when what comes back looks different
    assert changed.select(scene(config, camera, [(where, RED)]), pixel(camera, *where), 0.0, own_pose())
    states = [estimate.state for estimate, _ in follow(changed, config, camera, [[]] * 3 + [[(where, BLUE)]] * 20)]
    assert TRACKED not in states and states[-1] == LOST

    leaving = ClickedOpponentTracker(calibration[0], config.arena)
    assert leaving.select(scene(config, camera, [((0.20, 0.80), RED)]), pixel(camera, 0.20, 0.80), 0.0, own_pose())
    states = [estimate.state for estimate, _ in follow(leaving, config, camera, [[((0.20 - 0.02 * (i + 1), 0.80), RED)] for i in range(40)])]
    assert states[0] == TRACKED and states[-1] == LOST and TRACKED not in states[states.index(LOST):]          # out through the wall: lost for good


# ---- the viewer loop and the opt-in flags -----------------------------------------------------------------

def test_live_track_loop_with_scripted_clicks_writes_a_trace(tmp_path, config, camera):
    track = live_track.LiveTrack(config, fixture_clicks(config, camera))
    opponent = (0.75, 0.70)
    frames = [(index * DT, scene(config, camera, [(opponent, RED)], ids=None if index % 4 else [], seed=60 + index)) for index in range(16)]
    events = {2: [("click", pixel(camera, *OWN[:2]))], 3: [("click", pixel(camera, *opponent))], 12: [("clear",)]}
    with (tmp_path / "trace.jsonl").open("w", encoding="utf-8") as trace:
        records = live_track.run_frames(track, frames, events, trace)
    written = [json.loads(line) for line in (tmp_path / "trace.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(records) == 16 and written == json.loads(json.dumps(records))
    assert [record["own_track"]["state"] for record in records[:5]] == ["lost", "measured", "measured", "measured", "predicted"]
    assert records[2]["events"][0]["accepted"] is False and "own (tagged) robot" in records[2]["events"][0]["reason"]
    assert records[3]["events"][0]["accepted"] is True and [record["opponent_track"]["state"] for record in records] == (
        ["not_selected"] * 3 + ["tracked"] * 9 + ["not_selected"] * 4)
    tracked = records[5]["opponent_track"]
    assert math.hypot(tracked["x_m"] - opponent[0], tracked["y_m"] - opponent[1]) < 0.04
    assert all(set(record["processing_ms"]) == {"markers", "pose", "blobs", "opponent", "total"} and "nothing is sent" in record["output"] for record in records)
    summary = live_track.summarise(records)
    assert summary["own_track_states"] == {"lost": 1, "measured": 12, "predicted": 3} and [event["event"] for event in summary["events"]] == ["click", "click", "clear"]
    print(f"\nFIXTURE live_track software time per frame: {summary['software_ms_per_frame']} (this computer; not camera latency)")
    sample = track.pipeline.process(frames[-1][1], 16 * DT)
    lines = [text for text, _ in track.lines(sample)]
    assert any(text.startswith("OWN TRACK   MEASURED") for text in lines) and any(text.startswith("OPPONENT    NOT SELECTED") for text in lines)
    assert any("PLACEHOLDER" in text for text in lines) and any("no connection to the robot" in text for text in lines)
    view = demo.annotate(frames[5][1], track.pipeline, sample)
    assert view.shape == frames[5][1].shape and np.abs(view.astype(int) - frames[5][1].astype(int)).max() > 50       # the footprint is drawn


def stand_in_window(monkeypatch, frames, script):
    """A fake capture device and window. ``script``: read count -> ("click", (x, y)) or a key code."""
    state = {"reads": 0, "opened": 0, "shown": [], "mouse": None}

    class FakeCapture:
        def __init__(self, *args): state["opened"] += 1
        def isOpened(self): return True
        def set(self, prop, value): return True
        def read(self):
            index = min(state["reads"], len(frames) - 1); state["reads"] += 1
            return True, frames[index].copy()
        def release(self): pass

    def wait_key(delay):
        action = script.get(state["reads"], 255)
        if isinstance(action, tuple):
            state["mouse"](cv2.EVENT_LBUTTONDOWN, int(round(action[1][0])), int(round(action[1][1])), 0, None)
            return 255
        return action

    monkeypatch.setattr(cv2, "VideoCapture", FakeCapture)
    monkeypatch.setattr(cv2, "namedWindow", lambda *a, **k: None)
    monkeypatch.setattr(cv2, "setMouseCallback", lambda name, callback: state.__setitem__("mouse", callback))
    monkeypatch.setattr(cv2, "imshow", lambda name, image: state["shown"].append(image.shape))
    monkeypatch.setattr(cv2, "waitKey", wait_key)
    monkeypatch.setattr(cv2, "getWindowProperty", lambda *a: 1.0)
    monkeypatch.setattr(cv2, "destroyAllWindows", lambda: None)
    return state


def test_live_track_camera_mode_through_a_stand_in_camera(tmp_path, monkeypatch, config, camera):
    points = tmp_path / "clicks.json"
    save_clicks(points, fixture_clicks(config, camera), source="synthetic")
    opponent = (0.75, 0.70)
    frames = [scene(config, camera, [(opponent, RED)], seed=80 + index) for index in range(4)]
    script = {3: ("click", pixel(camera, *opponent)), 6: ("click", (1280 + 50, 100)), 12: ord("c"), 16: ord("q")}
    state = stand_in_window(monkeypatch, frames, script)
    args = live_track.build_parser().parse_args(["--camera", "0", "--config", str(REPO / "config" / "arena_overlay.yaml"), "--points", str(points),
                                                 "--out", str(tmp_path / "out")])
    summary = live_track.run_camera(args)
    trace = [json.loads(line) for line in Path(summary["trace"]).read_text(encoding="utf-8").splitlines()]
    assert state["opened"] == 1 and summary["frames"] == len(trace) == 16 and state["shown"][0] == (720, 1280 + live_track.PANEL_WIDTH, 3)
    assert [event.get("accepted") for event in summary["events"]] == [True, None]                           # one click, one clear; the click on the text panel is not a selection
    states = [record["opponent_track"]["state"] for record in trace]
    assert states == ["not_selected"] * 3 + ["tracked"] * 9 + ["not_selected"] * 4                          # c cleared it
    assert all(record["own_track"]["state"] == "measured" for record in trace) and "no connection to the robot" in summary["output"]
    assert live_track.main(["--camera", "0", "--profile", demo.HISTORICAL_PROFILE, "--out", str(tmp_path / "refused")]) == 2 and state["opened"] == 1


@needs_checkpoint
def test_shadow_camera_mode_flags_are_opt_in(tmp_path, monkeypatch, config, camera):
    contract = load_contract()
    points = tmp_path / "clicks.json"
    save_clicks(points, fixture_clicks(config, camera), source="synthetic")
    opponent = (0.75, 0.70)
    frames = [scene(config, camera, [(opponent, RED)], ids=None if index < 6 else [], seed=90 + index) for index in range(12)]
    # The stand-in camera runs on the wall clock, so the covered-tag frames must not turn STALE just because a loaded or cold
    # machine took longer than 0.2 s to get through six frames: give the tracker a long hold in a copy of the config.
    slow_config = tmp_path / "arena_overlay_slow.yaml"
    slow_config.write_text((REPO / "config" / "arena_overlay.yaml").read_text(encoding="utf-8")
                           + "\ntracker: {predict_hold_s: 5.0, lost_after_s: 10.0}\n", encoding="utf-8")
    common = ["--camera", "0", "--config", str(slow_config), "--points", str(points)]
    assert demo.build_parser().parse_args(common).multi_marker is False and demo.build_parser().parse_args(common).click_opponent is False
    assert live_shadow.build_parser().parse_args(common).multi_marker is False and live_shadow.build_parser().parse_args(common).click_opponent is False

    state = stand_in_window(monkeypatch, frames, {2: ("click", pixel(camera, *opponent)), 12: ord("q")})
    summary = demo.run_camera(demo.build_parser().parse_args(common + ["--multi-marker", "--click-opponent", "--out", str(tmp_path / "new")]), contract)
    trace = [json.loads(line) for line in Path(summary["trace"]).read_text(encoding="utf-8").splitlines()]
    assert state["mouse"] is not None and len(trace) == 12 and all(d["observation"] is None and d["proposed"] is None for d in trace)   # no IMU: never complete
    assert all(d["inputs"]["opponent_position"]["valid"] and "clicked object" in d["inputs"]["opponent_position"]["source"] for d in trace[2:])
    assert "no opponent selected" in trace[0]["inputs"]["opponent_position"]["reason"]                       # no empty-floor reference was asked for
    own = [d["inputs"]["own_pose"] for d in trace]
    assert all(status["valid"] for status in own)
    # While the tags are visible a decision may still read "PREDICTED nn ms" if a frame took over 50 ms of wall-clock time to
    # process (a slow first frame, a loaded machine); that is timing, not behaviour, so only a young prediction is accepted.
    assert all("PREDICTED" not in status["source"] or int(status["source"].split("PREDICTED ")[1].split(" ms")[0]) < 150 for status in own[:6])
    assert all("PREDICTED" in status["source"] for status in own[6:])                                         # tags covered: predicted, and labelled
    assert [d["own_track"]["state"] for d in trace] == ["measured"] * 6 + ["predicted"] * 6 and trace[5]["opponent_track"]["state"] == "tracked"

    state = stand_in_window(monkeypatch, frames, {12: ord("q")})
    summary = demo.run_camera(demo.build_parser().parse_args(common + ["--out", str(tmp_path / "old")]), contract)
    trace = [json.loads(line) for line in Path(summary["trace"]).read_text(encoding="utf-8").splitlines()]
    assert state["mouse"] is None                                      # without the flags no mouse callback is even installed
    assert all("own_track" not in d and "opponent_track" not in d for d in trace)
    assert all("no startup reference" in d["inputs"]["opponent_position"]["reason"] for d in trace)
    assert [d["inputs"]["own_pose"]["valid"] for d in trace] == [True] * 6 + [False] * 6                      # no hold, no prediction: as before
    assert all("no configured bot markers" in d["inputs"]["own_pose"]["reason"] for d in trace[6:])


@needs_checkpoint
def test_live_shadow_with_both_flags_uses_the_trackers_and_blocks_without_them_valid(config, camera):
    from control_center.tools.live_telemetry import ImuMounting, LiveTelemetry

    def line(sequence, t):
        return f"IMU seq={sequence} t_us={int(t * 1e6)} ok=1 ax=0.0000 ay=0.0000 az=9.8067 gx=0.000 gy=0.000 gz=0.000 unit=m/s2,deg/s"

    telemetry = LiveTelemetry(ImuMounting(np.eye(3), True, Path("fixture-mounting.yaml")))
    shadow = live_shadow.LiveShadow(load_contract(), config, fixture_clicks(config, camera), telemetry, "SYNTHETIC fixture frames + scripted telemetry lines",
                                    shadow_label="FIXTURE SHADOW", multi_marker=True, click_opponent=True)
    shadow.start(camera.image_size)
    assert shadow.pipeline.chassis is not None and shadow.pipeline.clicked is not None and shadow.pipeline.tracker is None
    opponent, decisions, sequence = (0.75, 0.70), [], 0
    for index in range(20):
        t = index * DT
        for step in range(4):
            sequence += 1
            telemetry.feed("RX", line(sequence, t - 0.03 + 0.01 * step), t - 0.03 + 0.01 * step)
        objects = [(opponent, RED)] if index < 14 else []              # the opponent vanishes at frame 14
        frame = scene(config, camera, objects, ids=None if index not in (8, 9) else [], seed=120 + index)      # tags covered on frames 8 and 9
        if index == 2:
            assert shadow.pipeline.select_opponent(frame, pixel(camera, *opponent), t).ok
        decisions.append(shadow.step(frame, t)[1])
    assert all("no opponent selected" in d.inputs["opponent_position"].reason for d in decisions[:2])
    complete = [d.observation is not None for d in decisions]
    assert complete == [False] * 2 + [True] * 12 + [False] * 6                                               # policy not loaded: complete, no proposal
    assert all(d.proposed is None for d in decisions)
    assert all("PREDICTED" in decisions[i].inputs["own_pose"].source for i in (8, 9)) and "PREDICTED" not in decisions[10].inputs["own_pose"].source
    assert all("AMBIGUOUS" in d.inputs["opponent_position"].reason or "LOST" in d.inputs["opponent_position"].reason for d in decisions[14:])
    lines = [text for text, _ in shadow.lines(decisions[-1], decisions[-1].t)]
    assert any(text.startswith("OWN TRACK   MEASURED") for text in lines) and any(text.startswith("OPPONENT    AMBIGUOUS") for text in lines)
    plain = live_shadow.LiveShadow(load_contract(), config, fixture_clicks(config, camera), telemetry, "x")
    plain.start(camera.image_size)
    assert plain.pipeline.chassis is None and plain.pipeline.clicked is None and (plain.multi_marker, plain.click_opponent) == (False, False)


# ---- the marker survey ------------------------------------------------------------------------------------

def config_hashes():
    return {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted((REPO / "config").rglob("*")) if path.is_file()}


def test_survey_finds_the_dictionary_and_ids_and_never_writes_a_config(tmp_path, capsys, camera):
    config = tf.nine_marker_config()
    before = config_hashes()
    frame = tf.render_bot(config, camera, 0.55, 0.45, math.radians(30), strays=[MarkerPlacement(20, (0.30, 0.85), 0.4, 0.040, 0.0)])
    image = tmp_path / "frame.png"
    assert cv2.imwrite(str(image), frame)
    points = tmp_path / "clicks.json"
    save_clicks(points, fixture_clicks(config, camera), source="synthetic")
    args = marker_survey.build_parser().parse_args(["--image", str(image), "--config", str(REPO / "config" / "arena_overlay.yaml"),
                                                    "--points", str(points), "--out", str(tmp_path / "survey")])
    report = marker_survey.run(args)
    assert report["best_dictionary"] == "DICT_4X4_50" and "DICT_4X4_100" in report["best_dictionary_note"]       # the same ids exist in both
    mine = {row["id"]: row for row in report["markers"] if row["dictionary"] == "DICT_4X4_50"}
    assert sorted(mine) == [0, 1, 2, 3, 20] and all(row["frames_seen"] == row["frames_total"] == 1 and row["detection_rate"] == 1.0 for row in mine.values())
    assert set(report["dictionaries_tried"]) >= {"DICT_4X4_50", "DICT_4X4_100", "DICT_5X5_50", "DICT_6X6_50", "DICT_ARUCO_ORIGINAL",
                                                 "DICT_APRILTAG_36h11", "DICT_APRILTAG_25h9", "DICT_APRILTAG_16h5"}
    assert not [row for row in report["markers"] if row["dictionary"].startswith("DICT_APRILTAG") or row["dictionary"] in ("DICT_5X5_50", "DICT_6X6_50")]
    assert report["config"]["seen_but_not_configured"] == [20] and report["config"]["configured_but_never_seen"] == [4, 5, 6, 7]
    assert report["config"]["dictionary_matches_best"] and report["floor_calibration_used"] and report["writes_config"] is False
    for marker_id in (0, 1, 2, 3):                                     # 40 mm tags raised 34 mm: larger on the floor plane, right at their height
        assert 15.0 < mine[marker_id]["mean_side_px"] < 60.0
        assert mine[marker_id]["mean_side_at_configured_height_mm"] == pytest.approx(40.0, abs=1.0)
        assert mine[marker_id]["mean_side_on_floor_plane_mm"] > mine[marker_id]["mean_side_at_configured_height_mm"] + 0.5
    assert mine[20]["mean_side_on_floor_plane_mm"] == pytest.approx(40.0, abs=1.0) and mine[20]["mean_side_at_configured_height_mm"] is None
    saved = json.loads(Path(report["survey_file"]).read_text(encoding="utf-8"))
    assert saved["markers"] == report["markers"] and Path(report["survey_file"]).name == "survey.json" and (Path(report["survey_file"]).parent / "last_frame_annotated.png").is_file()
    assert Path(report["survey_file"]).parent.parent == tmp_path / "survey"
    assert marker_survey.main(["--image", str(image), "--out", str(tmp_path / "plain")]) == 0                      # no config: pixels only
    printed = capsys.readouterr().out
    assert "BEST DICTIONARY: DICT_4X4_50" in printed and "Nothing was written to any config file" in printed and "pixels only" in printed
    assert config_hashes() == before                                    # not one byte of config/ changed


def test_survey_measures_how_intermittent_each_marker_is(camera):
    config = tf.nine_marker_config()
    survey = marker_survey.MarkerSurvey(config=config)
    visible = [[0, 1, 2, 3], [0, 2], [0], [0, 1, 2], [], [0, 3], [0, 2], [0, 1]]
    marker_survey.run_frames(survey, (tf.render_bot(config, camera, 0.5, 0.5, 0.2, ids=ids, seed=140 + index) for index, ids in enumerate(visible)))
    report = survey.report()
    rates = {row["id"]: row["detection_rate"] for row in report["markers"] if row["dictionary"] == "DICT_4X4_50"}
    assert rates == {0: 0.875, 1: 0.375, 2: 0.5, 3: 0.25} and report["frames"] == 8
    assert report["config"]["configured_but_never_seen"] == [4, 5, 6, 7, 8] and report["config"]["seen_but_not_configured"] == []
    assert not report["floor_calibration_used"] and all(row["mean_side_on_floor_plane_mm"] is None for row in report["markers"])
    text = marker_survey.format_report(report)
    assert "37.5%" in text and "configured but NEVER seen: [4, 5, 6, 7, 8]" in text and "PLACEHOLDER" in text
    other = tf.nine_marker_config()
    other.dictionary = "DICT_APRILTAG_36h11"                            # a config that names the wrong dictionary is called out
    wrong = marker_survey.MarkerSurvey(config=other)
    wrong.add(tf.render_bot(config, camera, 0.5, 0.5, 0.2))
    assert wrong.report()["config"]["dictionary_matches_best"] is False and "MISMATCH" in marker_survey.format_report(wrong.report())
    empty = marker_survey.MarkerSurvey(["DICT_4X4_50"])
    empty.add(np.full((120, 160, 3), 128, np.uint8))
    assert empty.report()["best_dictionary"] is None and "no marker decoded" in marker_survey.format_report(empty.report())
