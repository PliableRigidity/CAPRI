"""Multi-marker chassis tracker, the rear marker, and its opt-in path into the observation. Read-only vision.

Evidence class: FIXTURE only. Marker corners are either projected exactly through the ideal pinhole
camera of overlay_fixture, or decoded from frames that camera rendered. No real camera, serial port or
robot link is opened. None of the tolerances below is a real-camera accuracy figure.
"""

from __future__ import annotations

import ast
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from control_center.tools import track_fixture as tf
from control_center.tools.chassis_tracker import (LOST, MEASURED, PREDICTED, REAR_ONLY, STALE, ChassisTracker, TrackerSettings,
                                                  TrackState, load_tracker_settings)
from control_center.tools.observation_contract import RECOMMENDED_CHECKPOINT, load_contract
from control_center.tools.overlay_core import (DEFAULT_CONFIG_PATH, CameraEstimate, MarkerDetector, estimate_bot_pose, load_config,
                                               parse_config)
from control_center.tools.overlay_fixture import MarkerPlacement, default_camera, fixture_clicks
from control_center.tools.stream_adapter import (MODE_COMPLETE, CameraPipeline, ImuSample, RecordedTelemetry, ShadowAssembler,
                                                 fit_profile)

REPO = Path(__file__).resolve().parents[1]
FPS = 30.0
DT = 1.0 / FPS
needs_checkpoint = pytest.mark.skipif(not RECOMMENDED_CHECKPOINT.is_file(), reason="the recommended checkpoint is not in runs/")


@pytest.fixture(scope="module")
def config():
    return tf.nine_marker_config()                       # top ids 0-3, bottom ids 4-7, rear id 8 (all placeholder offsets)


@pytest.fixture(scope="module")
def camera():
    return default_camera()


@pytest.fixture(scope="module")
def calibration(config, camera):
    mapping, estimate, _ = fit_profile(config, fixture_clicks(config, camera), camera.image_size)
    assert mapping.valid and estimate.source == "derived"
    return mapping, estimate


def seen(config, camera, x, y, heading, ids, orientation="upright", scale=None):
    """Exact pixel corners of the named markers for a chassis pose (what a perfect detector would return)."""
    out = []
    c, s = math.cos(heading), math.sin(heading)
    for marker_id in ids:
        spec = config.markers[marker_id]
        if spec.face == "rear":
            world = tf.rear_corners_world(spec, x, y, heading, orientation)
        else:
            model = spec.model_corners()
            if scale and marker_id in scale:                 # a marker printed at the wrong size
                centre = model.mean(axis=0)
                model = centre + (model - centre) * scale[marker_id]
            xy = np.array([x, y]) + model @ np.array([[c, s], [-s, c]])
            world = np.column_stack([xy, np.full(4, spec.height_m)])
        out.append(SimpleNamespace(tag_id=marker_id, corners=camera.project(world)))
    return out


def error_mm(estimate, x, y):
    return 1000.0 * math.hypot(estimate.x_m - x, estimate.y_m - y)


def heading_error_deg(estimate, heading):
    return abs(math.degrees((estimate.heading_rad - heading + math.pi) % (2 * math.pi) - math.pi))


def tracker_for(config, calibration, **settings):
    return ChassisTracker(config, calibration[0], calibration[1], TrackerSettings(**settings))


# ---- consensus --------------------------------------------------------------------------------------

def test_dropouts_on_rendered_frames_still_give_a_continuous_pose(config, camera, calibration):
    """FIXTURE: 45 rendered frames at 30 fps; each top marker visible 40% of the time, and 8 frames with none at all."""
    rng = np.random.default_rng(11)
    tracker = tracker_for(config, calibration)
    detector = MarkerDetector(config.dictionary)
    states, measured_errors, predicted_errors, heading_errors, single_frame_ok, empty = [], [], [], [], 0, 0
    nothing_visible = {7, 8, 20, 21, 22, 23, 24, 33}                  # up to five frames (167 ms) in a row
    for index in range(45):
        t = index * DT
        angle = math.radians(20 + 1.6 * index)
        x, y, heading = 0.575 + 0.28 * math.cos(angle), 0.50 + 0.28 * math.sin(angle), angle + math.pi / 2     # about 0.23 m/s
        ids = [marker for marker in (0, 1, 2, 3) if rng.random() < 0.40] or [int(rng.integers(4))]
        ids = [] if index in nothing_visible else ids
        empty += not ids
        detections = detector.detect_markers(tf.render_bot(config, camera, x, y, heading, ids=ids, seed=200 + index))
        single_frame_ok += estimate_bot_pose(detections, calibration[0], calibration[1], config).ok
        estimate = tracker.update(detections, t)
        states.append(estimate.state)
        assert estimate.valid_for_observation, (index, ids, estimate.reason)
        assert estimate.orientation == "upright" and estimate.to_bot_pose().ok
        (measured_errors if estimate.state == MEASURED else predicted_errors).append(error_mm(estimate, x, y))
        heading_errors.append(heading_error_deg(estimate, heading))
        if estimate.state == PREDICTED:
            assert "predicted" in estimate.to_bot_pose().reason and estimate.age_s <= 0.20 + 1e-9
    assert empty == 8 and states.count(PREDICTED) == empty and single_frame_ok == 45 - empty      # the old estimator has nothing on those frames
    assert set(states) == {MEASURED, PREDICTED}
    print(f"\nFIXTURE dropouts: measured max {max(measured_errors):.1f} mm, predicted max {max(predicted_errors):.1f} mm, "
          f"heading max {max(heading_errors):.1f} deg, {empty} frames with no marker")
    assert max(measured_errors) < 6.0 and max(predicted_errors) < 20.0 and max(heading_errors) < 4.0


def test_a_marker_with_a_wrong_offset_is_rejected_and_the_pose_stays_right(camera, calibration):
    truth = tf.nine_marker_config()
    wrong = tf.nine_marker_config(marker_overrides={2: {"forward_m": 0.025}})         # marker 2 really sits at -0.035: 60 mm off
    tracker = ChassisTracker(wrong, calibration[0], calibration[1], TrackerSettings())
    x, y, heading = 0.60, 0.50, math.radians(40)
    everything = seen(truth, camera, x, y, heading, [0, 1, 2, 3])
    assert not estimate_bot_pose(everything, calibration[0], calibration[1], wrong).ok           # the single-frame estimator gives up
    estimate = tracker.update(everything, 0.0)
    assert estimate.state == MEASURED and estimate.used_ids == [0, 1, 3] and set(estimate.rejected) == {2}
    assert "disagrees with the consensus of [0, 1, 3]" in estimate.rejected[2] and error_mm(estimate, x, y) < 1.0
    # Marker 2 on its own would now put the chassis 60 mm away. It was just outvoted, so it is not trusted alone.
    alone = tracker.update(seen(truth, camera, x, y, heading, [2]), DT)
    assert alone.state == PREDICTED and error_mm(alone, x, y) < 1.0 and "not trusted alone" in alone.frame_reason
    # With one good marker beside it the two disagree: the track's prediction picks the good one.
    pair = tracker.update(seen(truth, camera, x, y, heading, [1, 2]), 2 * DT)
    assert pair.state == MEASURED and pair.used_ids == [1] and "tie" in pair.rejected[2] and error_mm(pair, x, y) < 1.0
    report = pair.to_report()
    assert report["state"] == "measured" and report["rejected"] == {"2": pair.rejected[2]} and report["valid_for_observation"] is True


def test_two_disagreeing_markers_without_a_track_give_no_pose(camera, calibration):
    truth = tf.nine_marker_config()
    wrong = tf.nine_marker_config(marker_overrides={2: {"forward_m": 0.025}})
    tracker = ChassisTracker(wrong, calibration[0], calibration[1], TrackerSettings())
    estimate = tracker.update(seen(truth, camera, 0.6, 0.5, 0.3, [0, 2]), 0.0)
    assert estimate.state == LOST and estimate.x_m is None and not estimate.valid_for_observation
    assert "inconsistent" in estimate.reason and "no usable track" in estimate.reason and set(estimate.rejected) == {0, 2}
    assert not estimate.to_bot_pose().ok and estimate.to_bot_pose().x_m is None


def test_a_wrong_size_or_a_repeated_id_rejects_that_marker_not_the_frame(config, camera, calibration):
    x, y, heading = 0.45, 0.62, math.radians(-70)
    tracker = tracker_for(config, calibration)
    estimate = tracker.update(seen(config, camera, x, y, heading, [0, 1, 2, 3], scale={1: 1.5}), 0.0)
    assert estimate.state == MEASURED and estimate.used_ids == [0, 2, 3] and "configured as 40 mm" in estimate.rejected[1]
    assert error_mm(estimate, x, y) < 1.0
    twice = seen(config, camera, x, y, heading, [0, 2, 3]) + seen(config, camera, 0.9, 0.3, 1.0, [3]) + [SimpleNamespace(tag_id=30, corners=np.zeros((4, 2)))]
    estimate = tracker_for(config, calibration).update(twice, 0.0)
    assert estimate.used_ids == [0, 2] and "more than once" in estimate.rejected[3] and estimate.ignored_ids == [30]
    assert error_mm(estimate, x, y) < 1.0
    assert not estimate_bot_pose(twice, calibration[0], calibration[1], config).ok               # unchanged: the old estimator refuses the frame


# ---- states over time ---------------------------------------------------------------------------------

def test_a_total_dropout_goes_predicted_then_stale_then_lost_at_the_documented_times(config, camera, calibration):
    tracker = tracker_for(config, calibration)                       # predict_hold_s 0.20, lost_after_s 1.0
    speed, heading = 0.30, 0.0
    for index in range(6):                                           # driving along +x, all four tags visible
        estimate = tracker.update(seen(config, camera, 0.30 + speed * index * DT, 0.50, heading, [0, 1, 2, 3]), index * DT)
        assert estimate.state == MEASURED and estimate.age_s == 0.0
    last_t = 5 * DT
    expected = {}
    for index in range(6, 45):                                       # every tag covered from here on; frames keep arriving
        t = index * DT
        estimate = tracker.update([], t)
        age = t - last_t
        expected[index] = PREDICTED if age <= 0.20 else STALE if age <= 1.0 else LOST
        assert estimate.state == expected[index], (index, age, estimate.state)
        truth_x = 0.30 + speed * t
        pose = estimate.to_bot_pose()
        if estimate.state == PREDICTED:
            assert abs(estimate.x_m - truth_x) < 0.004 and abs(estimate.y_m - 0.50) < 0.002        # constant velocity carried forward
            assert pose.ok and f"predicted {1000 * age:.0f} ms" in pose.reason and estimate.used_ids == [0, 1, 2, 3]
        elif estimate.state == STALE:
            assert estimate.x_m == pytest.approx(0.30 + speed * last_t, abs=1e-3)                  # the LAST MEASURED pose, not moved on
            assert not estimate.valid_for_observation and not pose.ok and pose.x_m is None and "stale" in pose.reason
            assert "DISPLAY ONLY" in estimate.to_report()["pose_note"]
        else:
            assert estimate.x_m is None and estimate.orientation == "unknown" and not pose.ok and "lost" in pose.reason
    assert list(expected.values()).count(PREDICTED) == 6 and LOST in expected.values()
    back = tracker.update(seen(config, camera, 0.70, 0.50, heading, [0]), 45 * DT)       # after LOST any consistent detection starts a new track
    assert back.state == MEASURED and back.velocity_m_s == (0.0, 0.0)


def test_validity_runs_out_even_when_no_more_frames_arrive(config, camera, calibration):
    tracker = tracker_for(config, calibration)
    for index in range(4):
        tracker.update(seen(config, camera, 0.40 + 0.2 * index * DT, 0.40, 0.5, [1, 2]), index * DT)
    t0 = 3 * DT                                                      # the camera stops here: status() is only ever asked the time
    wanted = [(0.0, MEASURED), (0.04, MEASURED), (0.06, PREDICTED), (0.20, PREDICTED), (0.21, STALE), (1.0, STALE), (1.01, LOST), (60.0, LOST)]
    for age, state in wanted:
        status = tracker.status(t0 + age)
        assert status.state == state and status.valid_for_observation == (state in (MEASURED, PREDICTED)), (age, status.state)
        assert status.to_bot_pose().ok == status.valid_for_observation
    assert tracker.status(t0 + 0.10).x_m == pytest.approx(0.40 + 0.2 * (t0 + 0.10), abs=2e-3)       # predicted to the query time
    assert tracker.status(t0 + 0.02).state == MEASURED                # asking again later does not change the answer for an earlier time
    assert isinstance(tracker.status(t0).state, TrackState) and tracker.status(t0 + 2).to_report()["x_m"] is None


def test_prediction_speed_is_clamped(config, camera, calibration):
    tracker = tracker_for(config, calibration, max_speed_m_s=0.5, position_noise_m=0.10)       # the generous noise floor lets the steps in
    for index in range(4):
        tracker.update(seen(config, camera, 0.30 + 1.5 * index * DT, 0.50, 0.0, [0, 1, 2, 3]), index * DT)    # really moving at 1.5 m/s
    status = tracker.status(3 * DT + 0.10)
    assert status.state == PREDICTED and math.hypot(*status.velocity_m_s) == pytest.approx(0.5, abs=1e-6)
    assert status.x_m == pytest.approx(0.30 + 1.5 * 3 * DT + 0.5 * 0.10, abs=2e-3)


def test_a_teleport_is_rejected_but_a_persistent_new_place_restarts_the_track(config, camera, calibration):
    tracker = tracker_for(config, calibration)                       # 2.5 m/s, 30 mm noise floor, 3 frames to restart
    here, there = (0.30, 0.30, 0.2), (0.85, 0.80, 0.2)                # 0.74 m apart
    t = 0.0
    for _ in range(3):
        assert tracker.update(seen(config, camera, *here, [0, 1, 2, 3]), t).state == MEASURED
        t += DT
    jump = tracker.update(seen(config, camera, *there, [0, 1, 2, 3]), t)
    assert jump.state == PREDICTED and error_mm(jump, *here[:2]) < 1.0 and "inconsistent with the track" in jump.frame_reason
    assert "m/s" in jump.frame_reason and set(jump.rejected) == {0, 1, 2, 3} and tracker.gate_rejections == 1
    t += DT
    assert tracker.update(seen(config, camera, *here, [0, 1, 2, 3]), t).state == MEASURED       # back where it was: the count starts over
    for count in (1, 2):
        t += DT
        waiting = tracker.update(seen(config, camera, *there, [0, 1]), t)
        assert waiting.state == PREDICTED and error_mm(waiting, *here[:2]) < 1.0 and f"seen {count} of 3" in waiting.frame_reason
    t += DT
    moved = tracker.update(seen(config, camera, *there, [0, 1]), t)
    assert moved.state == MEASURED and error_mm(moved, *there[:2]) < 1.0 and tracker.reinitialisations == 1
    assert moved.velocity_m_s == (0.0, 0.0)                           # nothing is extrapolated across the jump
    spin = tracker.update(seen(config, camera, there[0], there[1], there[2] + math.pi, [0, 1]), t + DT)     # a half turn in one frame
    assert spin.state == PREDICTED and "rad/s" in spin.frame_reason


# ---- which face is up ---------------------------------------------------------------------------------

def test_bottom_markers_mean_inverted_and_mixed_faces_follow_the_majority_rule(config, camera, calibration):
    x, y, heading = 0.55, 0.55, math.radians(120)
    inverted = tracker_for(config, calibration).update(seen(config, camera, x, y, heading, [4, 5, 6, 7]), 0.0)
    assert inverted.state == MEASURED and inverted.orientation == "inverted" and inverted.source == "bottom markers"
    assert error_mm(inverted, x, y) < 1.0 and heading_error_deg(inverted, heading) < 0.5
    held = tracker_for(config, calibration)
    held.update(seen(config, camera, x, y, heading, [4, 5]), 0.0)
    assert held.update([], DT).orientation == "inverted"              # held from the last measurement while predicting
    stray = seen(config, camera, 0.25, 0.85, 1.0, [5])                 # one bottom marker somewhere else in the arena
    estimate = tracker_for(config, calibration).update(seen(config, camera, x, y, heading, [0, 1, 3]) + stray, 0.0)
    assert estimate.state == MEASURED and estimate.orientation == "upright" and estimate.used_ids == [0, 1, 3] and set(estimate.rejected) == {5}
    assert not estimate_bot_pose(seen(config, camera, x, y, heading, [0, 1, 3]) + stray, calibration[0], calibration[1], config).ok
    cases = {
        "one each": seen(config, camera, x, y, heading, [0]) + stray,                                             # a tie, and not a majority
        "two each": seen(config, camera, x, y, heading, [0, 1]) + seen(config, camera, 0.25, 0.85, 1.0, [5, 6]),   # two consensuses
        "no majority": seen(config, camera, x, y, heading, [0, 1]) + stray + seen(config, camera, 0.9, 0.2, 2.0, [6]),
        "bottom agrees": seen(config, camera, x, y, heading, [0, 1, 3]) + seen(config, camera, x, y, heading, [6])}   # both faces at one pose
    for name, detections in cases.items():
        estimate = tracker_for(config, calibration).update(detections, 0.0)
        assert estimate.state == LOST and estimate.x_m is None, name
        assert "conflict" in estimate.reason or "inconsistent" in estimate.reason, (name, estimate.reason)


# ---- the rear marker ----------------------------------------------------------------------------------

def test_rear_marker_config_is_additive_and_the_single_frame_estimator_ignores_it(config, camera, calibration):
    spec = config.markers[8]
    assert (spec.face, spec.height_m, spec.height_inverted_m) == ("rear", 0.026, None) and len(config.markers) == 9
    assert load_config().markers.keys() == {0, 1, 2, 3, 4, 5, 6, 7}                        # the shipped config is untouched
    data = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    data["bot"]["markers"].append({"id": 8, "face": "rear", "forward_m": -0.075})          # no height given anywhere
    with pytest.raises(ValueError, match="no height"):
        parse_config(data)
    data["bot"]["markers"][-1].update(height_m=0.02, height_inverted_m=0.03)
    assert parse_config(data).markers[8].height_inverted_m == 0.03
    data["bot"]["markers"][-1]["face"] = "side"
    with pytest.raises(ValueError, match="face must be"):
        parse_config(data)
    x, y, heading = 0.5, 0.35, math.radians(90)
    with_rear = estimate_bot_pose(seen(config, camera, x, y, heading, [0, 1, 2, 3, 8]), calibration[0], calibration[1], config)
    without = estimate_bot_pose(seen(config, camera, x, y, heading, [0, 1, 2, 3]), calibration[0], calibration[1], config)
    assert with_rear.ok and with_rear.rear_ids == [8] and with_rear.used_ids == [0, 1, 2, 3] and without.rear_ids == []
    assert (with_rear.x_m, with_rear.y_m, with_rear.heading_rad) == (without.x_m, without.y_m, without.heading_rad)
    only = estimate_bot_pose(seen(config, camera, x, y, heading, [8]), calibration[0], calibration[1], config)
    assert not only.ok and only.rear_ids == [8] and "rear" in only.reason
    assert "rear_ids" not in without.to_report() or without.to_report().get("rear_ids") == []


def test_rear_marker_alone_gives_a_labelled_pose_only_when_the_camera_position_is_known(config, camera, calibration):
    """FIXTURE: the rear marker is rendered and decoded; the camera position is the one derived from the floor map."""
    detector = MarkerDetector(config.dictionary)
    worst = [0.0, 0.0]
    for orientation in ("upright", "inverted"):
        for x, y, degrees in ((0.50, 0.35, 90), (0.60, 0.50, 80), (0.45, 0.30, 110), (0.70, 0.60, 60)):
            frame = tf.render_bot(config, camera, x, y, math.radians(degrees), orientation=orientation, ids=[], rear=True)
            detections = detector.detect_markers(frame)
            assert [d.tag_id for d in detections] == [8]
            estimate = tracker_for(config, calibration).update(detections, 0.0)
            assert estimate.state == MEASURED and estimate.orientation == orientation and estimate.used_ids == [8]
            assert estimate.source == REAR_ONLY and "LOWER CONFIDENCE" in estimate.position_label and estimate.to_report()["source"] == "rear marker only"
            worst = [max(worst[0], error_mm(estimate, x, y)), max(worst[1], heading_error_deg(estimate, math.radians(degrees)))]
            unknown_camera = ChassisTracker(config, calibration[0], CameraEstimate(), TrackerSettings()).update(detections, 0.0)
            assert unknown_camera.state == LOST and unknown_camera.x_m is None and "camera position" in unknown_camera.rejected[8]
    print(f"\nFIXTURE rear marker only: position max {worst[0]:.1f} mm, heading max {worst[1]:.1f} deg")
    assert worst[0] < 6.0 and worst[1] < 5.0


def test_rear_marker_with_other_markers_and_when_it_cannot_be_told_which_way_up(config, camera, calibration):
    x, y, heading = 0.50, 0.35, math.radians(95)
    estimate = tracker_for(config, calibration).update(seen(config, camera, x, y, heading, [0, 3, 8]), 0.0)
    assert estimate.used_ids == [0, 3, 8] and estimate.source == "top markers + rear" and estimate.orientation == "upright"
    assert error_mm(estimate, x, y) < 1.0
    # Bottom markers say inverted, so the rear marker is read as that of an inverted chassis; drawn upright it cannot fit.
    clash = tracker_for(config, calibration).update(seen(config, camera, x, y, heading, [4, 5]) + seen(config, camera, x, y, heading, [8], "upright"), 0.0)
    assert clash.used_ids == [4, 5] and clash.orientation == "inverted" and "does not fit as a rear marker" in clash.rejected[8]
    moved = tracker_for(config, calibration).update(seen(config, camera, x, y, heading, [0, 3]) + seen(config, camera, x + 0.1, y, heading, [8]), 0.0)
    assert moved.used_ids == [0, 3] and "disagrees with the top markers" in moved.rejected[8]
    # A camera almost straight above sees no difference between the two ways up: rejected, never guessed.
    above = CameraEstimate((x, y + 0.01, 1.3), "measured", None, "fixture: camera placed straight above")
    overhead = ChassisTracker(config, calibration[0], above, TrackerSettings()).update(seen(config, camera, x, y, heading, [8]), 0.0)
    assert overhead.state == LOST and ("ambiguous" in overhead.rejected[8] or "does not fit" in overhead.rejected[8])
    inconsistent = tracker_for(config, calibration).update(seen(config, camera, x, y, heading, [0]) + seen(config, camera, 0.8, 0.8, 0.0, [1])
                                                           + seen(config, camera, x, y, heading, [8]), 0.0)
    assert inconsistent.state == LOST and "no consensus" in inconsistent.rejected[8]            # the rear marker does not break a tie


# ---- settings -----------------------------------------------------------------------------------------

def test_tracker_section_is_optional_and_checked(tmp_path, config):
    assert load_tracker_settings(load_config()) == TrackerSettings() == load_tracker_settings(config)       # no section / no file: defaults
    defaults = TrackerSettings()
    assert (defaults.predict_hold_s, defaults.lost_after_s, defaults.max_speed_m_s, defaults.max_turn_rate_rad_s, defaults.reacquire_frames) == (0.20, 1.0, 2.5, 25.0, 3)
    data = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    data["tracker"] = {"predict_hold_s": 0.1, "lost_after_s": 0.5, "max_speed_m_s": 1.5, "max_turn_rate_rad_s": 12, "reacquire_frames": 5}
    path = tmp_path / "overlay.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    loaded = load_tracker_settings(load_config(path))
    assert (loaded.predict_hold_s, loaded.lost_after_s, loaded.max_speed_m_s, loaded.max_turn_rate_rad_s, loaded.reacquire_frames) == (0.1, 0.5, 1.5, 12.0, 5)
    for bad in ({"predict_hold_s": 2.0}, {"max_speed_m_s": 0}, {"reacquire_frames": 0}, {"hold": 1}):
        with pytest.raises(ValueError):
            TrackerSettings.from_dict(bad)
    with pytest.raises(ValueError, match="timestamp order"):
        tracker = ChassisTracker(config, *fit_profile(config, fixture_clicks(config, default_camera()), default_camera().image_size)[:2])
        tracker.update([], 1.0)
        tracker.update([], 0.5)


def test_new_modules_have_no_path_to_the_robot():
    forbidden = ("robot_link", "command_manager", "controller_link", "serial", "socket", "telemetry", "hub", "round_recorder")
    for name in ("chassis_tracker.py", "opponent_select.py", "marker_survey.py", "live_track.py", "track_fixture.py"):
        source = (REPO / "control_center" / "tools" / name).read_text(encoding="utf-8")
        imported: list[str] = []
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.Import):
                imported += [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                imported += [node.module or ""] + [alias.name for alias in node.names]
        assert not [item for item in imported if any(word in item.split(".")[-1] for word in forbidden)], name
        assert "192.168" not in source and "Serial(" not in source and "sendto(" not in source, name
        if name != "live_track.py" and name != "marker_survey.py":
            assert "VideoCapture" not in source and "CameraSource" not in source, name       # only the two tools YOU start can open a camera


# ---- into the observation: predicted is valid and says so, stale never arrives --------------------------

def box_frame(config, camera, own, ids, seed, opponent=(0.80, 0.75)):
    frame = tf.render_bot(config, camera, *own, ids=ids, seed=seed)
    tf.paint_box(frame, camera, opponent, 0.2, (0.11, 0.10), 0.034, (40, 40, 220))
    return frame


@needs_checkpoint
def test_predicted_pose_is_labelled_and_a_stale_one_never_reaches_the_observation(config, camera, calibration):
    contract = load_contract()                                        # the checkpoint's recorded contract only; the policy is not loaded
    imu = RecordedTelemetry([ImuSample(i * 0.005, (0.0, 0.0, 0.0), (0.0, 0.0, 9.80665)) for i in range(600)], "SYNTHETIC stand-in IMU")
    own = (0.35, 0.35, 0.4)

    def run(frames_with_tags, total_frames, decide_after_last=()):
        pipeline = CameraPipeline(config, *calibration, multi_marker=True, click_opponent=True)
        assembler = ShadowAssembler(contract, "SYNTHETIC fixture frames", imu, policy=None, max_camera_age_s=config.max_detection_age_s)
        decisions = []
        for index in range(total_frames):
            t = index * DT
            frame = box_frame(config, camera, own, None if index < frames_with_tags else [], seed=300 + index)
            if index == 0:
                pipeline.process(frame, t)                             # the own pose first, so the click can be checked against it
                assert pipeline.select_opponent(frame, camera.project([[0.80, 0.75, 0.034]])[0], t).ok
                t += 1e-3
            sample = pipeline.process(frame, t)
            assembler.offer(sample)
            decisions.append((t, sample, assembler.decide(t)))
        last_t = decisions[-1][0]
        return decisions, [assembler.decide(last_t + age) for age in decide_after_last]

    decisions, _ = run(frames_with_tags=4, total_frames=14)            # tags covered from frame 4; frames keep arriving
    last_measured_t = decisions[3][0]
    kinds = []
    for t, sample, decision in decisions:
        status = decision.inputs["own_pose"]
        age = t - last_measured_t
        assert decision.inputs["opponent_position"].valid and "clicked object" in decision.inputs["opponent_position"].source
        if age <= 0:
            kinds.append("measured")
            assert status.valid and "PREDICTED" not in status.source and status.source.startswith("ArUco ids [0, 1, 2, 3], height-corrected")
            assert decision.mode == MODE_COMPLETE and decision.observation is not None
        elif age <= 0.20:
            kinds.append("predicted")
            assert status.valid and f"ArUco ids [0, 1, 2, 3] PREDICTED {1000 * age:.0f} ms" in status.source
            assert decision.observation is not None and sample.own_track["state"] == "predicted" and sample.to_report()["own_track"]["age_s"] == pytest.approx(age, abs=1e-4)
            assert math.hypot(status.values["x_m"] - own[0], status.values["y_m"] - own[1]) < 0.003
        else:
            kinds.append("stale")
            assert not status.valid and "stale" in status.reason and status.values == {}
            assert decision.observation is None and decision.proposed is None and any("own_pose" in entry for entry in decision.blocked_by)
            assert sample.own_track["state"] == "stale" and sample.own.x_m is None and not sample.own.ok
    assert kinds.count("measured") == 4 and kinds.count("predicted") == 6 and kinds.count("stale") == 4

    # The camera stops right after a measured frame: the same decisions, asked later and later with no new frame.
    _, later = run(frames_with_tags=3, total_frames=3, decide_after_last=(0.03, 0.10, 0.19, 0.22, 0.26, 2.0))
    assert [d.inputs["own_pose"].valid for d in later] == [True, True, True, False, False, False]
    assert "PREDICTED" not in later[0].inputs["own_pose"].source and "PREDICTED 100 ms" in later[1].inputs["own_pose"].source
    assert "stale" in later[3].inputs["own_pose"].reason                         # the tracker's hold ran out before the camera limit
    assert "newest camera frame" in later[4].inputs["own_pose"].reason and all(d.observation is None for d in later[3:])
    assert all(d.observation is not None for d in later[:3])


def test_without_the_new_arguments_the_pipeline_is_what_it_was(config, camera, calibration):
    pipeline = CameraPipeline(config, *calibration)
    assert pipeline.chassis is None and pipeline.clicked is None
    frame = tf.render_bot(config, camera, 0.4, 0.4, 0.3, ids=[0, 1])
    sample = pipeline.process(frame, 0.0)
    reference = estimate_bot_pose(MarkerDetector(config.dictionary).detect_markers(frame), calibration[0], calibration[1], config)
    assert sample.own.to_report() == reference.to_report() and sample.own_at is None
    assert set(sample.to_report()) == {"frame", "t_s", "own", "opponent", "tracks", "blobs", "processing_ms"}
    assert "no startup reference" in sample.opponent.reason and sample.opponent.source == ""
    with pytest.raises(ValueError, match="click_opponent"):
        pipeline.select_opponent(frame, (10, 10), 0.0)
    covered = pipeline.process(tf.render_bot(config, camera, 0.4, 0.4, 0.3, ids=[]), DT)
    assert not covered.own.ok and "no configured bot markers" in covered.own.reason               # no hold, no prediction: as before
    opted = CameraPipeline(config, *calibration, multi_marker=True)
    assert set(opted.process(frame, 0.0).to_report()) == {"frame", "t_s", "own", "opponent", "tracks", "blobs", "processing_ms", "own_track"}
    stray = tf.render_bot(config, camera, 0.4, 0.4, 0.3, ids=[0, 1, 3], seed=5, strays=[MarkerPlacement(5, (0.9, 0.8), 0.3, 0.040, 0.034)])
    assert not pipeline.process(stray, 2 * DT).own.ok and opted.process(stray, DT).own.ok          # consensus instead of all-or-nothing
