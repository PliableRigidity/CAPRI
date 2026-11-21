"""Live ArUco tracking under controlled, SYNTHETIC conditions: blur, partial occlusion, one marker only, flicker, glare, wrong
dictionary, ids outside the layout, a repeated id, a flipped robot, and the age/state ladder MEASURED -> PREDICTED -> STALE -> LOST.

Evidence class: SYNTHETIC ONLY. Frames are rendered with ``cv2.aruco.generateImageMarker`` (DICT_4X4_50) and perspective-warped
through a known pinhole camera onto a textured floor (control_center/tools/aruco_synth.py, track_fixture.py); the true pose of every
marker is known exactly. No camera, serial port or socket is opened. Nothing here says anything about the real camera, its lens,
its lighting, rolling shutter, the real print quality or the real marker size and mounting: every size, height and offset in the
layout is a PLACEHOLDER (config/arena_overlay.yaml). The thresholds below check that the logic behaves, not how accurate the real
system is.
"""

from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from control_center.backend import tracking_model as tm
from control_center.backend.marker_settings import MARKER_DICTIONARY, OPENCV_DEFAULT_SET, build_detector
from control_center.tools import aruco_synth as S
from control_center.tools import track_fixture as tf
from control_center.tools.chassis_tracker import (LIVE_START_CONFIRM_FRAMES, LOST, MEASURED, PREDICTED, STALE, ChassisTracker,
                                                  TrackerSettings, live_tracker_settings)
from control_center.tools.overlay_core import MarkerDetector
from control_center.tools.overlay_fixture import make_camera, fixture_clicks
from control_center.tools.stream_adapter import fit_profile

REPO = Path(__file__).resolve().parents[1]
FPS = 30.0
DT = 1.0 / FPS
POSE = (0.575, 0.575, math.radians(20.0))          # x, y, heading of the chassis used unless a test says otherwise


# ---- a bench: the layout, a camera that shows the markers about as large as the real camera did, a calibration -----------------

@pytest.fixture(scope="module")
def config():
    return tf.nine_marker_config()                 # top 0-3, bottom 4-7, rear 8: all PLACEHOLDER offsets


@pytest.fixture(scope="module")
def camera():
    # About 34 px per marker side at 1280x720: the real camera's survey saw 36-41 px (artifacts/marker-survey), the placeholder size is 40 mm.
    return make_camera((0.575, 0.10, 0.95), (0.575, 0.65, 0.0), focal_px=930.0)


@pytest.fixture(scope="module")
def calibration(config, camera):
    mapping, estimate, _ = fit_profile(config, fixture_clicks(config, camera), camera.image_size)
    assert mapping.valid and estimate.source == "derived"
    return mapping, estimate


@pytest.fixture(scope="module")
def detector(config):
    return MarkerDetector(config.dictionary)


def live_tracker(config, calibration, **overrides):
    """The tracker as the live viewers build it (start_confirm_frames from ``live_tracker_settings``) unless overridden."""
    settings = TrackerSettings.from_dict({"start_confirm_frames": LIVE_START_CONFIRM_FRAMES, **overrides})
    return ChassisTracker(config, calibration[0], calibration[1], settings)


def corners_of(config, camera, marker_id, x, y, heading, orientation="upright"):
    """Exact pixel corners of one configured marker for a chassis pose (what a perfect detector would return)."""
    spec = config.markers[marker_id]
    c, s = math.cos(heading), math.sin(heading)
    model = spec.model_corners()
    xy = np.array([x, y]) + model @ np.array([[c, s], [-s, c]])
    return camera.project(np.column_stack([xy, np.full(4, spec.height_m)]))


def frame_of(config, camera, x, y, heading, ids=None, orientation="upright", seed=1, blur=0.0, blur_angle=0.0, noise=2.0):
    image = tf.render_bot(config, camera, x, y, heading, orientation=orientation, ids=ids, seed=seed)
    return S.camera_like(image, blur, blur_angle, noise, 80, seed)


def err_mm(estimate, x, y):
    return 1000.0 * math.hypot(estimate.x_m - x, estimate.y_m - y)


def err_deg(estimate, heading):
    return abs(math.degrees((estimate.heading_rad - heading + math.pi) % (2 * math.pi) - math.pi))


def detections_of(detector, image):
    return detector.detect_markers(image)


# =====================================================================================================================
# 1. The detector: blur, the parameter set, wrong dictionaries, ids outside the layout, partial occlusion, glare
# =====================================================================================================================

def _rate(detector_obj, side, blur, trials, seed):
    rng = np.random.default_rng(seed)
    found = total = 0
    wrong = 0                                   # a decoded id that is not in the picture
    far = 0                                     # the right id, but a corner more than 3 px from the truth
    for _ in range(trials):
        markers = [S.PlacedMarker(i, S.square_corners((250 + 300 * k, 300 + rng.uniform(-60, 60)), side, rng.uniform(0, 360),
                                                       (rng.uniform(-.15, .15), rng.uniform(-.15, .15)))) for k, i in enumerate((1, 2, 3))]
        image = S.camera_like(S.render(markers, seed=int(rng.integers(1, 9999))), blur, rng.uniform(0, 180), 2.0, 75, int(rng.integers(1, 9999)))
        corners, ids, _ = detector_obj.detectMarkers(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
        total += len(markers)
        truth = {m.marker_id: m for m in markers}
        for c, i in zip(corners, [] if ids is None else ids.flatten().tolist()):
            if i not in truth:
                wrong += 1
            else:
                found += 1
                far += S.corner_error_px(c, truth[i].corners_px) >= 3.0
    return found / total, wrong, far


def test_detection_rate_against_blur_and_the_effect_of_the_parameter_set():
    """SYNTHETIC: a 36 px marker (the size the real camera showed) under a straight motion streak. The named set is never worse than
    OpenCV's defaults and clearly better where blur starts to eat the markers; neither misreads anything. Not a real-camera rate."""
    live, default = build_detector(), build_detector(MARKER_DICTIONARY, OPENCV_DEFAULT_SET)
    table = {}
    for blur in (0.0, 4.0, 8.0, 10.0):
        table[blur] = (_rate(live, 36.0, blur, 14, 31), _rate(default, 36.0, blur, 14, 31))
    print("\nSYNTHETIC detection rate, 36 px marker, 42 markers per cell: blur px -> live_v1 / opencv_default")
    for blur, (a, b) in table.items():
        print(f"   blur {blur:4.1f} px   {a[0]:.2f} / {b[0]:.2f}   wrong ids {a[1]} / {b[1]}   corners >3 px off {a[2]} / {b[2]}")
    assert table[0.0][0][0] == 1.0 and table[4.0][0][0] >= 0.95                    # sharp and lightly blurred: every marker found
    for blur, (a, b) in table.items():
        assert a[1] == 0 and b[1] == 0, f"a decoded id that is not in the picture at blur {blur}"
        assert a[2] <= max(1, 0.1 * 42) or blur >= 8.0, f"corners more than 3 px off at blur {blur}"
        assert a[0] >= b[0] - 0.05, f"the named set is worse than the OpenCV default at blur {blur}"
    assert table[10.0][0][0] > table[10.0][1][0] + 0.05                             # where blur matters, the set helps (documented effect)
    assert table[10.0][0][0] < 0.95                                                 # ... and blur at 28% of the marker side still costs detections


def test_small_markers_are_found_down_to_the_expected_pixel_size_and_a_decode_never_lies():
    live = build_detector()
    sharp = {side: _rate(live, side, 0.0, 8, 5) for side in (14.0, 20.0, 26.0, 36.0)}
    print("\nSYNTHETIC sharp detection rate by marker side px:", {k: round(v[0], 2) for k, v in sharp.items()})
    assert sharp[36.0][0] == 1.0 and sharp[26.0][0] >= 0.9
    assert all(wrong == 0 and far == 0 for _, wrong, far in sharp.values())


def test_markers_of_other_dictionaries_are_not_read_as_ids_and_never_become_a_pose(config, camera, calibration, detector):
    """5x5, 6x6, ARUCO_ORIGINAL, AprilTag 36h11 and 16h5 markers, ids 0-8, at several sizes, angles and blurs."""
    rng = np.random.default_rng(4)
    decoded = rendered = 0
    tracker = live_tracker(config, calibration)
    states = set()
    for dictionary in ("DICT_5X5_50", "DICT_6X6_50", "DICT_ARUCO_ORIGINAL", "DICT_APRILTAG_36h11", "DICT_APRILTAG_16h5"):
        for trial in range(14):
            markers = [S.PlacedMarker(int(rng.integers(0, 9)), S.square_corners((300 + 250 * k, 300), float(rng.choice([26, 36, 50])),
                                                                                  rng.uniform(0, 360), (rng.uniform(-.15, .15), rng.uniform(-.15, .15))),
                                      dictionary=dictionary) for k in range(3)]
            image = S.camera_like(S.render(markers, seed=trial), float(rng.choice([0, 3, 6])), rng.uniform(0, 180), 2.0, 75, trial)
            found = detections_of(detector, image)
            rendered += len(markers)
            decoded += len(found)
            states.add(tracker.update(found, len(states) * 0.0 + (rendered / 3) * DT).state)
    print(f"\nSYNTHETIC wrong-dictionary markers decoded as a 4X4_50 id: {decoded} of {rendered}")
    assert decoded <= 0.01 * rendered                      # essentially never; an occasional false decode is possible, so not asserted as exactly 0
    assert states == {LOST}                                # and not once did that become a tracked pose


def test_ids_outside_the_layout_are_listed_as_ignored_and_never_used(config, camera, calibration, detector):
    """DICT_4X4_50 ids 9-49 are valid markers of the dictionary but not of the robot: seen, reported, never used."""
    tracker = live_tracker(config, calibration)
    stray = [S.PlacedMarker(i, S.square_corners((220 + 180 * k, 90), 34.0, 10.0 * k)) for k, i in enumerate((17, 33, 49))]
    x, y, heading = POSE
    for index in range(3):
        image = S.camera_like(S.render(stray, size=camera.image_size, floor=frame_of(config, camera, x, y, heading, ids=[], seed=index)), 0.0, 0.0, 2.0, 80, index)
        found = detections_of(detector, image)
        assert {17, 33, 49} <= {d.tag_id for d in found}
        estimate = tracker.update(found, index * DT)
    assert estimate.ignored_ids == [17, 33, 49] and estimate.used_ids == [] and estimate.visible_ids == []
    assert estimate.state == LOST and "no configured bot markers visible" in estimate.reason
    # with the robot's own markers in view as well: the same pose as without the strays
    clean, mixed = live_tracker(config, calibration), live_tracker(config, calibration)
    base = frame_of(config, camera, x, y, heading, seed=9)
    a = clean.update(detections_of(detector, base), 0.0)
    b = mixed.update(detections_of(detector, S.render(stray, size=camera.image_size, floor=base)), 0.0)
    assert b.state == MEASURED and b.ignored_ids == [17, 33, 49] and b.used_ids == a.used_ids
    assert (b.x_m, b.y_m, b.heading_rad) == pytest.approx((a.x_m, a.y_m, a.heading_rad), abs=2e-3)


def test_a_repeated_id_is_not_trusted_but_the_other_markers_still_give_the_pose(config, camera, calibration, detector):
    x, y, heading = POSE
    base = frame_of(config, camera, x, y, heading, ids=[0, 1, 3], seed=3)                     # marker 2 not drawn at its place ...
    twice = S.render([S.PlacedMarker(2, S.square_corners((150, 120), 34.0, 25.0)), S.PlacedMarker(2, S.square_corners((1100, 600), 34.0, 70.0))],
                     size=camera.image_size, floor=base)                                      # ... but two copies of id 2 elsewhere
    found = detections_of(detector, twice)
    assert sorted(d.tag_id for d in found).count(2) == 2
    estimate = live_tracker(config, calibration).update(found, 0.0)
    assert estimate.state == MEASURED and estimate.used_ids == [0, 1, 3] and "more than once" in estimate.rejected[2]
    assert err_mm(estimate, x, y) < 6.0
    # a repeated id that is the ONLY marker in view gives nothing at all
    lone = S.render([S.PlacedMarker(1, S.square_corners((150, 120), 34.0, 25.0)), S.PlacedMarker(1, S.square_corners((1100, 600), 34.0, 70.0))],
                    size=camera.image_size, floor=frame_of(config, camera, x, y, heading, ids=[], seed=4))
    alone = live_tracker(config, calibration, start_confirm_frames=1).update(detections_of(detector, lone), 0.0)
    assert alone.state == LOST and "more than once" in alone.rejected[1] and alone.x_m is None


def test_partial_occlusion_and_glare_never_produce_a_wrong_id_but_a_decode_can_have_shifted_corners(config, camera, detector):
    """Cover 15-60 % of a marker, or wash part of it out. FINDING (synthetic): the id is never wrong, but a marker that is still
    decoded with its edge covered can come back with its corners pulled 4-8 px (of a 34 px side) toward the cover."""
    x, y, heading = POSE
    truth = {i: corners_of(config, camera, i, x, y, heading) for i in (0, 1, 2, 3)}
    base = frame_of(config, camera, x, y, heading, seed=21, noise=0.0)
    wrong_id = missing = exact = shifted = 0
    worst = 0.0
    for fraction in (0.1, 0.15, 0.3, 0.45, 0.6):
        for side in ("left", "right", "top", "bottom"):
            image = base
            for i in (0, 2):
                image = S.occlude(image, truth[i], fraction, side)
            image = S.glare(image, truth[1].mean(axis=0), 26.0, 0.8 if fraction > 0.3 else 0.25)
            found = detections_of(detector, S.camera_like(image, 0.0, 0.0, 2.0, 80, 5))
            seen_ids = {d.tag_id: d for d in found}
            wrong_id += len(set(seen_ids) - {0, 1, 2, 3})
            for i in (0, 1, 2, 3):
                if i not in seen_ids:
                    missing += 1
                    continue
                error = S.corner_error_px(seen_ids[i].corners, truth[i])
                worst = max(worst, error)
                exact += error < 3.0
                shifted += error >= 3.0
    print(f"\nSYNTHETIC occlusion/glare battery: {exact} accurate, {shifted} decoded with corners >= 3 px off (worst {worst:.1f} px), "
          f"{missing} not decoded, {wrong_id} wrong ids")
    assert wrong_id == 0 and missing > 0 and exact > 0           # covering markers loses them, and no id is ever invented
    assert shifted > 0 and worst < 14.0                            # the documented weakness exists, and is bounded (< 40 % of the side)


def test_partly_covered_markers_do_not_drag_a_multi_marker_pose_far(config, camera, calibration, detector):
    """Markers 0 and 2 each lose 10-15 % of their width (still decoded, corners shifted); 1 and 3 are clean."""
    x, y, heading = POSE
    truth = {i: corners_of(config, camera, i, x, y, heading) for i in (0, 1, 2, 3)}
    base = frame_of(config, camera, x, y, heading, seed=21, noise=0.0)
    worst_position = worst_heading = 0.0
    for side in ("left", "right", "top", "bottom"):
        image = base
        for i in (0, 2):
            image = S.occlude(image, truth[i], 0.12, side)
        estimate = live_tracker(config, calibration).update(detections_of(detector, S.camera_like(image, 0.0, 0.0, 2.0, 80, 5)), 0.0)
        assert estimate.state == MEASURED and len(estimate.used_ids) >= 2, (side, estimate.reason)
        worst_position, worst_heading = max(worst_position, err_mm(estimate, x, y)), max(worst_heading, err_deg(estimate, heading))
    print(f"\nSYNTHETIC two markers 12% covered: worst position error {worst_position:.1f} mm, heading {worst_heading:.1f} deg")
    assert worst_position < 10.0 and worst_heading < 4.0


# =====================================================================================================================
# 2. The tracker on rendered frames: 1, 2, 3, 4 markers; one wrong marker; flicker
# =====================================================================================================================

def test_accuracy_with_one_two_three_and_four_visible_markers(config, camera, calibration, detector):
    rng = np.random.default_rng(8)
    errors = {n: [] for n in (1, 2, 3, 4)}
    heading_errors = {n: [] for n in (1, 2, 3, 4)}
    for trial in range(8):
        x, y, heading = rng.uniform(0.35, 0.8), rng.uniform(0.35, 0.8), rng.uniform(-math.pi, math.pi)
        for n in (1, 2, 3, 4):
            ids = sorted(rng.choice(4, n, replace=False).tolist())
            image = frame_of(config, camera, x, y, heading, ids=ids, seed=100 + trial)
            estimate = live_tracker(config, calibration, start_confirm_frames=1).update(detections_of(detector, image), 0.0)
            assert estimate.state == MEASURED and estimate.used_ids == ids, (trial, n, estimate.reason)
            errors[n].append(err_mm(estimate, x, y))
            heading_errors[n].append(err_deg(estimate, heading))
            assert estimate.orientation == "upright" and estimate.flipped is False
    mean = {n: round(float(np.mean(v)), 2) for n, v in errors.items()}
    worst = {n: round(float(np.max(v)), 2) for n, v in errors.items()}
    print(f"\nSYNTHETIC position error mm by markers visible: mean {mean} worst {worst}; heading deg worst "
          f"{ {n: round(float(np.max(v)), 2) for n, v in heading_errors.items()} }")
    assert all(max(v) < 12.0 for v in errors.values()) and all(max(v) < 6.0 for v in heading_errors.values())
    assert np.mean(errors[4]) <= np.mean(errors[1]) + 0.5                  # more markers do not make it worse


def test_one_marker_is_a_valid_but_low_confidence_pose_and_more_markers_raise_it(config, camera, calibration, detector):
    x, y, heading = POSE
    levels = {}
    for ids in ([2], [0, 3, 1, 2]):
        tracker = live_tracker(config, calibration, start_confirm_frames=1)
        estimate = tracker.update(detections_of(detector, frame_of(config, camera, x, y, heading, ids=ids, seed=7)), 0.0)
        evidence = {"fit_rms_mm": tracker._frame.fit_rms_mm, "spread_mm": None, "heading_spread_deg": None}
        levels[len(ids)] = tm.tracking_confidence(estimate.to_report(), evidence, 0.02, 10.0, 0.2).level
    assert levels[1] in ("LOW", "MEDIUM") and levels[1] != "HIGH" and levels[4] == "HIGH"


def test_a_marker_decoded_in_the_wrong_place_is_outvoted_by_the_others(config, camera, calibration, detector):
    """A copy of id 2 somewhere else (the real 2 is covered): markers 0, 1, 3 agree, the stray 2 is rejected, the pose is right."""
    x, y, heading = POSE
    base = frame_of(config, camera, x, y, heading, ids=[0, 1, 3], seed=12)
    stray = S.render([S.PlacedMarker(2, S.square_corners((900, 140), 34.0, 40.0))], size=camera.image_size, floor=base)
    estimate = live_tracker(config, calibration).update(detections_of(detector, stray), 0.0)
    assert estimate.state == MEASURED and estimate.used_ids == [0, 1, 3] and 2 in estimate.rejected
    assert err_mm(estimate, x, y) < 6.0 and err_deg(estimate, heading) < 3.0


def test_flicker_one_frame_in_three_never_reads_as_a_current_pose_older_than_two_frames(config, camera, calibration, detector):
    """30 fps, only marker 1 visible, and only every third frame; blur 3 px; the robot drives at 0.2 m/s."""
    tracker = live_tracker(config, calibration)
    x0, y0, heading = POSE
    rows, worst = [], 0.0
    for index in range(60):
        t = index * DT
        x = x0 + 0.2 * t
        visible = index % 3 == 0
        image = frame_of(config, camera, x, y0, heading, ids=[1] if visible else [], seed=index, blur=3.0)
        estimate = tracker.update(detections_of(detector, image), t)
        rows.append((estimate.state, estimate.age_s))
        if index >= 4:                                                   # after the start (two agreeing frames)
            assert estimate.valid_for_observation, (index, estimate.reason)
            assert estimate.age_s <= 2 * DT + 1e-6
            assert estimate.state == (MEASURED if visible else PREDICTED)
            worst = max(worst, err_mm(estimate, x, y0))
    print(f"\nSYNTHETIC flicker 1 in 3, one marker: worst position error {worst:.1f} mm")
    assert rows[0][0] == LOST and rows[3][0] == MEASURED                 # frame 0: one marker, one frame: not yet; frame 3: confirmed
    assert worst < 25.0


# =====================================================================================================================
# 3. Dropout: state ladder, age, never a stale pose as current; reacquisition
# =====================================================================================================================

def test_glare_dropout_walks_measured_predicted_stale_lost_with_the_age_always_reported(config, camera, calibration, detector):
    tracker = live_tracker(config, calibration)
    x, y, heading = POSE
    for index in range(4):                                               # four good frames with all markers
        last = tracker.update(detections_of(detector, frame_of(config, camera, x + 0.1 * index * DT, y, heading, seed=index)), index * DT)
    t_last = 3 * DT
    assert last.state == MEASURED and last.age_s == 0.0 and last.used_ids == [0, 1, 2, 3]
    # then glare wipes out every marker: frames keep arriving, and the markers are simply not there
    previous_age, ladder = 0.0, []
    for step in range(1, 45):
        now = t_last + step * DT
        washed = S.glare(frame_of(config, camera, x, y, heading, seed=50 + step), (640, 400), 900.0, 1.0)
        estimate = tracker.update(detections_of(detector, washed), now)
        age = now - t_last
        ladder.append(estimate.state)
        assert estimate.age_s == pytest.approx(age, abs=1e-9) and estimate.age_s >= previous_age
        previous_age = estimate.age_s
        expected = PREDICTED if age <= 0.20 + 1e-9 else STALE if age <= 1.0 + 1e-9 else LOST
        assert estimate.state == expected, (step, age, estimate.state)
        report = estimate.to_report()
        assert report["age_s"] == pytest.approx(round(age, 4), abs=1e-4) and report["state"] == expected.value
        if estimate.state == STALE:
            assert not estimate.valid_for_observation and not estimate.to_bot_pose().ok and "DISPLAY ONLY" in report["pose_note"]
            assert estimate.x_m == pytest.approx(tracker._last.x)                       # the last MEASURED pose, not moved on
        if estimate.state == LOST:
            assert estimate.x_m is None and estimate.heading_rad is None and estimate.orientation == "unknown" and estimate.flipped is None
            assert not estimate.to_bot_pose().ok
    assert ladder[0] == PREDICTED and STALE in ladder and ladder[-1] == LOST
    assert ladder == sorted(ladder, key=[PREDICTED, STALE, LOST].index)               # one way only: never back up without a detection


def test_the_state_follows_the_query_time_even_when_no_frame_arrives_at_all(config, camera, calibration, detector):
    tracker = live_tracker(config, calibration)
    x, y, heading = POSE
    for index in range(3):
        tracker.update(detections_of(detector, frame_of(config, camera, x, y, heading, seed=index)), index * DT)
    t_last = 2 * DT
    seen = [(dt, tracker.status(t_last + dt)) for dt in (0.0, 0.04, 0.1, 0.2, 0.21, 0.5, 1.0, 1.01, 5.0)]
    assert [e.state for _, e in seen] == [MEASURED, MEASURED, PREDICTED, PREDICTED, STALE, STALE, STALE, LOST, LOST]
    assert [round(e.age_s, 3) for _, e in seen] == [0.0, 0.04, 0.1, 0.2, 0.21, 0.5, 1.0, 1.01, 5.0]          # the age is reported in every state
    assert [e.valid_for_observation for _, e in seen] == [True, True, True, True, False, False, False, False, False]


def test_reacquisition_after_a_loss_is_immediate_with_two_markers_and_needs_a_second_frame_with_one(config, camera, calibration, detector):
    x, y, heading = POSE
    # a long loss, then the robot is somewhere else
    for ids, expect_first, expect_second in (([0, 3], MEASURED, MEASURED), ([2], LOST, MEASURED)):
        tracker = live_tracker(config, calibration)
        for index in range(3):
            tracker.update(detections_of(detector, frame_of(config, camera, x, y, heading, seed=index)), index * DT)
        assert tracker.status(5.0).state == LOST
        nx, ny, nh = 0.80, 0.40, math.radians(-100)
        first = tracker.update(detections_of(detector, frame_of(config, camera, nx, ny, nh, ids=ids, seed=70)), 5.0)
        second = tracker.update(detections_of(detector, frame_of(config, camera, nx, ny, nh, ids=ids, seed=71)), 5.0 + DT)
        assert (first.state, second.state) == (expect_first, expect_second), (ids, first.reason)
        assert second.used_ids == ids and err_mm(second, nx, ny) < 8.0
        if expect_first == LOST:
            assert first.x_m is None and "one marker" in first.frame_reason and 2 in first.rejected


def test_a_lone_marker_frame_cannot_start_a_track_but_two_markers_or_default_settings_can(config, camera, calibration, detector):
    x, y, heading = POSE
    one = detections_of(detector, frame_of(config, camera, x, y, heading, ids=[1], seed=3))
    assert live_tracker(config, calibration).update(one, 0.0).state == LOST                           # live default: needs confirming
    assert live_tracker(config, calibration, start_confirm_frames=1).update(one, 0.0).state == MEASURED   # explicit 1 = the old behaviour
    assert ChassisTracker(config, calibration[0], calibration[1]).settings.start_confirm_frames == 1      # bare TrackerSettings() unchanged
    assert live_tracker_settings(config).start_confirm_frames == LIVE_START_CONFIRM_FRAMES == 2
    both = detections_of(detector, frame_of(config, camera, x, y, heading, ids=[1, 2], seed=3))
    assert live_tracker(config, calibration).update(both, 0.0).state == MEASURED
    # two lone frames that do NOT agree (a false decode, then the real thing far away) do not confirm each other
    tracker = live_tracker(config, calibration)
    far = detections_of(detector, frame_of(config, camera, 0.95, 0.95, 1.0, ids=[1], seed=4))
    assert tracker.update(one, 0.0).state == LOST and tracker.update(far, DT).state == LOST
    assert tracker.update(far, 2 * DT).state == MEASURED                                              # the second agreeing frame does
    # frames further apart than the prediction hold do not confirm each other either
    sparse = live_tracker(config, calibration)
    assert sparse.update(one, 0.0).state == LOST and sparse.update(one, 0.5).state == LOST


# =====================================================================================================================
# 4. A flipped robot
# =====================================================================================================================

def test_a_flipped_robot_is_reported_as_inverted_from_the_bottom_markers(config, camera, calibration, detector):
    tracker = live_tracker(config, calibration)
    x, y, heading = 0.55, 0.60, math.radians(35)
    up = tracker.update(detections_of(detector, frame_of(config, camera, x, y, heading, seed=1)), 0.0)
    assert up.state == MEASURED and up.orientation == "upright" and up.flipped is False and up.used_ids == [0, 1, 2, 3]
    # the robot is flipped where it stands (rolled about its nose axis: same place, same nose direction); now only 4-7 can be seen
    flipped = tracker.update(detections_of(detector, frame_of(config, camera, x, y, heading, orientation="inverted", seed=2)), DT)
    assert flipped.state == MEASURED and flipped.orientation == "inverted" and flipped.flipped is True and flipped.used_ids == [4, 5, 6, 7]
    assert err_mm(flipped, x, y) < 6.0 and err_deg(flipped, heading) < 3.0
    assert flipped.to_report()["flipped"] is True and flipped.to_bot_pose().orientation == "inverted"
    # the orientation is held while predicting, and gone when LOST
    held = tracker.update(detections_of(detector, frame_of(config, camera, x, y, heading, ids=[], seed=3)), 2 * DT)
    assert held.state == PREDICTED and held.flipped is True
    assert tracker.status(10.0).state == LOST and tracker.status(10.0).flipped is None
    # one bottom marker alone is enough to say "flipped" (with a confirming second frame)
    solo = live_tracker(config, calibration)
    for index in range(2):
        solo_estimate = solo.update(detections_of(detector, frame_of(config, camera, x, y, heading, orientation="inverted", ids=[6], seed=9 + index)), index * DT)
    assert solo_estimate.state == MEASURED and solo_estimate.flipped is True and solo_estimate.used_ids == [6]


def test_top_and_bottom_markers_visible_together_give_no_pose_not_a_guess(config, camera, calibration, detector):
    x, y, heading = POSE
    image = S.render([S.PlacedMarker(5, S.square_corners((160, 520), 34.0, 20.0))], size=camera.image_size,
                     floor=frame_of(config, camera, x, y, heading, ids=[0, 1, 2], seed=14))           # three top markers and a stray bottom one
    estimate = live_tracker(config, calibration).update(detections_of(detector, image), 0.0)
    assert estimate.state != MEASURED or estimate.orientation == "upright" and estimate.used_ids == [0, 1, 2] and 5 in estimate.rejected
    assert 5 not in estimate.used_ids


# =====================================================================================================================
# 5. What the TRACKING screen is given: state, age, ids, position, heading, orientation, confidence (the Qt-free model)
# =====================================================================================================================

def _rows(own):
    return {name: (value, level) for name, value, level in tm.own_rows(own)}


def test_the_own_robot_panel_always_shows_state_age_ids_position_heading_orientation_and_confidence(config, camera, calibration, detector):
    tracker = live_tracker(config, calibration)
    x, y, heading = POSE
    measured = tracker.update(detections_of(detector, frame_of(config, camera, x, y, heading, ids=[0, 1, 2, 5 - 5 + 3], seed=1)), 0.0).to_report()
    rows = _rows(measured)
    assert rows["State"][0] == "MEASURED" and "ms since the last accepted detection" in rows["Age"][0]
    assert "Position x, y" in rows and "Heading" in rows and rows["Marker IDs in the newest frame"][0] == "[0, 1, 2, 3]"
    assert rows["Marker IDs the pose rests on"][0] == "[0, 1, 2, 3]" and rows["Orientation"][0].startswith("UPRIGHT")
    confidence = tm.tracking_confidence(measured, {"fit_rms_mm": 1.0, "spread_mm": 1.0, "heading_spread_deg": 0.5}, 0.02, 10.0, 0.2)
    assert confidence.level == "HIGH" and any(name == "age of last measurement" for name, _ in confidence.inputs)

    predicted = tracker.update(detections_of(detector, frame_of(config, camera, x, y, heading, ids=[], seed=2)), 0.1).to_report()
    rows = _rows(predicted)
    assert rows["State"] == ("PREDICTED", "warn") and "100 ms" in rows["Age"][0] and "[predicted]" in rows["Position x, y"][0]
    assert rows["Marker IDs in the newest frame"][0].startswith("none") and "held from the last detection" in rows["Orientation"][0]

    stale = tracker.status(0.6).to_report()
    rows = _rows(stale)
    assert rows["State"][0].startswith("STALE") and "not valid" in rows["State"][0] and rows["State"][1] == "bad"
    assert "display only" in rows["Position x, y"][0] and "600 ms" in rows["Age"][0]
    assert tm.tracking_confidence(stale, None, 0.02, 10.0, 0.2).level == "NONE"

    lost = tracker.status(3.0).to_report()
    rows = _rows(lost)
    assert rows["State"] == ("LOST", "bad") and rows["Position x, y"][0] == "-" and rows["Heading"][0] == "-"
    assert "3000 ms" in rows["Age"][0] and tm.tracking_confidence(lost, None, 0.02, 10.0, 0.2).level == "NONE"

    never = live_tracker(config, calibration).status(0.0).to_report()
    assert _rows(never)["Age"][0].startswith("no detection yet") and _rows(never)["State"][0] == "LOST"


def test_a_flipped_robot_and_an_ignored_id_are_spelled_out_on_the_panel(config, camera, calibration, detector):
    tracker = live_tracker(config, calibration)
    x, y, heading = POSE
    image = S.render([S.PlacedMarker(17, S.square_corners((200, 100), 34.0, 0.0))], size=camera.image_size,
                     floor=frame_of(config, camera, x, y, heading, orientation="inverted", seed=2))
    rows = _rows(tracker.update(detections_of(detector, image), 0.0).to_report())
    assert rows["Orientation"][0].startswith("UPSIDE DOWN / FLIPPED") and rows["Orientation"][1] == "warn"
    assert "[17]" in rows["Marker IDs ignored"][0] and rows["Marker IDs the pose rests on"][0] == "[4, 5, 6, 7]"


def test_a_report_that_stops_being_refreshed_ages_into_stale_and_lost_on_screen():
    """The screen paints the newest report; if the camera or the vision thread stops, that report must not read as current."""
    base = {"state": "measured", "valid_for_observation": True, "x_m": 0.5, "y_m": 0.6, "heading_deg": 20.0, "pose_note": None,
            "orientation": "upright", "flipped": False, "age_s": 0.0, "used_ids": [0, 1], "visible_ids": [0, 1], "rejected": {}, "ignored_ids": [],
            "position_quality": "height-corrected", "source": "top markers", "reason": "ok"}
    states = []
    for elapsed in (0.0, 0.1, 0.19, 0.21, 0.9, 1.01, 7.0):
        aged = tm.age_own_report(base, elapsed, 0.2, 1.0)
        states.append((aged["state"], round(aged["age_s"], 2)))
        assert aged["age_s"] == pytest.approx(elapsed) and base["state"] == "measured" and base["age_s"] == 0.0     # the input is not modified
        if aged["state"] == "stale":
            assert aged["x_m"] == 0.5 and aged["valid_for_observation"] is False and "DISPLAY ONLY" in aged["pose_note"]
        if aged["state"] == "lost":
            assert aged["x_m"] is None and aged["heading_deg"] is None and aged["valid_for_observation"] is False
    assert [s for s, _ in states] == ["measured", "measured", "measured", "stale", "stale", "lost", "lost"]
    already_lost = dict(base, state="lost", x_m=None, y_m=None, heading_deg=None, age_s=None)
    assert tm.age_own_report(already_lost, 5.0, 0.2, 1.0)["state"] == "lost" and tm.age_own_report(None, 1.0, 0.2, 1.0) is None


# =====================================================================================================================
# 6. OPERATE / recorder / reviewer share LiveVision: it now carries the tracker report beside its (unchanged) single-frame pose
# =====================================================================================================================

def test_live_vision_carries_the_tracker_state_and_age_beside_the_unchanged_single_frame_pose(config, camera):
    from control_center.backend.live_vision import LiveVision

    vision = LiveVision(config, fixture_clicks(config, camera), profile="synthetic fixture")
    x, y, heading = POSE
    full = frame_of(config, camera, x, y, heading, seed=1)
    empty = frame_of(config, camera, x, y, heading, ids=[], seed=2)
    first = vision.process(0, 0.0, full)
    assert first.pose.ok and first.own_track["state"] == "measured" and first.own_track["used_ids"] == first.pose.used_ids == [0, 1, 2, 3]
    assert first.own_track["flipped"] is False and first.own_track["age_s"] == 0.0 and first.own_track["visible_ids"] == [0, 1, 2, 3]
    gap = vision.process(1, DT, empty)                          # the single-frame pose is a GAP, as it always was ...
    assert not gap.pose.ok and gap.own_track["state"] == "predicted" and gap.own_track["age_s"] == pytest.approx(DT, abs=1e-4)      # ... the tracker says how old
    assert vision.process(2, 2.0, empty).own_track["state"] == "lost"
    again = vision.process(3, 0.0, full)                        # time went backwards (a new stream): no exception, a fresh track
    assert again.own_track is not None and again.pose.ok
    assert LiveVision(config, None).process(0, 0.0, full).own_track is None      # no calibration: no map, no pose, no track


# =====================================================================================================================
# 7. THE failure seen on the real camera: markers decoded, every one rejected because the configured size is a placeholder
# =====================================================================================================================

def test_a_configured_size_far_from_the_apparent_size_rejects_every_marker_and_says_what_to_measure(config, camera, calibration, detector):
    """REAL traces (artifacts/live-track, home_test): ids 0-8 decoded on hundreds of frames, 'measures 9-16 mm but is configured as 40 mm' on
    all of them, own track LOST on ~98 % of frames. Reproduced here with a SYNTHETIC mismatch: markers drawn at 40 mm, config says 14 mm."""
    small = tf.nine_marker_config(marker_overrides={i: {"size_m": 0.014} for i in range(8)})
    x, y, heading = POSE
    found = detections_of(detector, frame_of(config, camera, x, y, heading, seed=1))
    assert sorted(d.tag_id for d in found) == [0, 1, 2, 3]
    tracker = ChassisTracker(small, calibration[0], calibration[1], TrackerSettings())
    estimate = tracker.update(found, 0.0)
    assert estimate.state == LOST and estimate.x_m is None and estimate.visible_ids == [0, 1, 2, 3] and estimate.used_ids == []
    for marker_id in range(4):
        assert "but is configured as 14 mm" in estimate.rejected[marker_id] and "PLACEHOLDER" in estimate.rejected[marker_id]
        assert "bot.marker_size_m" in estimate.rejected[marker_id]
    assert "every visible marker was rejected" in estimate.frame_reason
    # once the size is right again the same frames give a pose
    assert ChassisTracker(config, calibration[0], calibration[1], TrackerSettings()).update(found, 0.0).state == MEASURED
