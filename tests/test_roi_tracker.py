"""Drag-selected, free-form opponent tracking (control_center/tools/roi_tracker.py) and its place in the pipeline.

Evidence class: SYNTHETIC. Pictures are generated here (a noisy floor, textured or flat sprites of any size) or
rendered by overlay_fixture / track_fixture (ideal pinhole camera). No real camera, serial port or robot link
is opened. Nothing below says how well the tracker follows a real object under real light.
"""
from __future__ import annotations

import math
import time

import cv2
import numpy as np
import pytest

from control_center.tools import roi_tracker as R
from control_center.tools import track_fixture as tf
from control_center.tools.opponent_select import LOST, NOT_SELECTED, TRACKED, UNCERTAIN, OpponentTrack
from control_center.tools.overlay_core import BotPose, FloorMapping, load_config
from control_center.tools.overlay_fixture import default_camera, fixture_clicks
from control_center.tools.stream_adapter import CameraPipeline, fit_profile

DT = 1.0 / 30.0
W, H = 960, 540


def floor(seed=1, size=(W, H)):
    rng = np.random.default_rng(seed)
    base = rng.integers(70, 110, (size[1] // 8 + 1, size[0] // 8 + 1, 3)).astype(np.uint8)
    return cv2.resize(base, size, interpolation=cv2.INTER_CUBIC)


def sprite(width, height, seed=5):
    """A textured object: coloured noise with a bright cross, so it has something to be recognised by."""
    rng = np.random.default_rng(seed)
    small = rng.integers(20, 255, (max(2, height // 4), max(2, width // 4), 3)).astype(np.uint8)
    out = cv2.resize(small, (width, height), interpolation=cv2.INTER_NEAREST)
    cv2.line(out, (0, 0), (width - 1, height - 1), (255, 255, 255), max(1, width // 10))
    return out


def paste(frame, obj, centre):
    h, w = obj.shape[:2]
    x, y = int(round(centre[0] - w / 2)), int(round(centre[1] - h / 2))
    x0, y0, x1, y1 = max(0, x), max(0, y), min(frame.shape[1], x + w), min(frame.shape[0], y + h)
    if x1 > x0 and y1 > y0:
        frame[y0:y1, x0:x1] = obj[y0 - y:y1 - y, x0 - x:x1 - x]
    return frame


def tracker(mapping=None, arena=None, **settings):
    arena = arena or {"width_m": 1.15, "height_m": 1.15}
    return R.RoiOpponentTracker(mapping or FloorMapping(), arena, R.RoiSettings(**settings) if settings else None)


def box_around(centre, size, pad=0):
    return (centre[0] - size[0] / 2 - pad, centre[1] - size[1] / 2 - pad, size[0] + 2 * pad, size[1] + 2 * pad)


def centre_of(report):
    return report["centre_px"]


# ---------------------------------------------------------------- only genuinely invalid rectangles are refused


def test_valid_rectangles_of_any_size_are_accepted_and_clipped_to_the_picture():
    size = (W, H)
    assert R.validate_rect((100, 100, 8, 8), size)[0] == (100, 100, 8, 8)                       # a ball
    assert R.validate_rect((10, 10, 880, 500), size)[0] == (10, 10, 880, 500)                   # almost the whole picture
    assert R.validate_rect((300, 200, -120, -60), size)[0] == (180, 140, 120, 60)               # dragged right-to-left, bottom-to-top
    clipped, note = R.validate_rect((900, 500, 200, 200), size)
    assert clipped == (900, 500, 60, 40) and "cut off" in note
    clipped, note = R.validate_rect((-30, -20, 100, 100), size)
    assert clipped == (0, 0, 70, 80)


@pytest.mark.parametrize("rect, message", [
    ((100, 100, 0, 50), "empty"), ((100, 100, 50, 0), "empty"), ((100, 100, 3, 80), "empty"),
    ((100, 100, 80, 3.5), "empty"), ((2000, 100, 50, 50), "outside the 960 x 540"), ((-200, -200, 50, 50), "outside"),
    ((100, 600, 50, 50), "outside"), ((float("nan"), 0, 50, 50), "finite"), ((0, 0, float("inf"), 50), "finite"),
    (("a", 0, 50, 50), "four numbers"), ((1, 2, 3), "four numbers"), (None, "four numbers"),
])
def test_empty_non_numeric_and_off_picture_rectangles_are_refused_with_a_reason(rect, message):
    clipped, why = R.validate_rect(rect, (W, H))
    assert clipped is None and message in why


def test_a_refused_box_leaves_the_current_selection_alone():
    frame = paste(floor(), sprite(60, 40), (300, 200))
    t = tracker()
    assert t.select_roi(frame, box_around((300, 200), (60, 40)), 0.0).ok
    for bad in ((10, 10, 0, 0), (5000, 5000, 20, 20), None):
        result = t.select_roi(frame, bad, DT)
        assert not result.ok and t.state == TRACKED and t.selection_id == 1
    assert "box REFUSED" in t.selection_note
    assert not t.select(frame, (300, 200), DT).ok and "drag a box" in t.selection_note               # a bare click is not a selection
    assert not t.select_roi("not an image", (1, 1, 50, 50), 0.0).ok


def test_there_is_no_physical_size_limit_anywhere_in_this_path():
    import inspect
    source = inspect.getsource(R)
    assert "max_size_m" not in source and "0.28" not in source and "280" not in source
    assert not any(name for name in R.RoiSettings.__dataclass_fields__ if "max_size" in name or "min_size_m" in name)
    for width, height in ((12, 12), (24, 20), (60, 44), (160, 90), (420, 300)):                  # a ball ... a very large object
        frame = paste(floor(), sprite(width, height), (480, 270))
        t = tracker()
        result = t.select_roi(frame, box_around((480, 270), (width, height)), 0.0)
        assert result.ok and t.state == TRACKED, (width, height, result.reason)


# ---------------------------------------------------------------- following an object of any size


@pytest.mark.parametrize("name, size, step", [("small ball", (14, 14), (2.0, 1.0)), ("antweight robot", (48, 40), (3.0, 1.5)),
                                              ("phone", (150, 80), (4.0, 0.0)), ("large object", (300, 200), (2.0, 1.0))])
def test_objects_of_different_sizes_are_followed_across_frames(name, size, step):
    obj = sprite(*size)
    start = (360.0, 250.0)
    t = tracker()
    assert t.select_roi(paste(floor(2), obj, start), box_around(start, size), 0.0).ok
    errors, confidences = [], []
    for index in range(1, 41):
        centre = (start[0] + step[0] * index, start[1] + step[1] * index)
        report = t.update(paste(floor(2 + index % 3), obj, centre), index * DT).to_report()
        assert report["state"] == "tracked", (name, index, report["reason"])
        errors.append(math.dist(centre_of(report), centre))
        confidences.append(report["confidence"])
    assert max(errors) < max(3.0, 0.1 * max(size)), (name, max(errors))
    assert min(confidences) >= 0.6 and report["backend"] and report["image_based"] is True


def test_a_flat_coloured_object_is_followed_by_colour_and_says_so():
    frame = floor()
    cv2.rectangle(frame, (300, 200), (360, 250), (40, 40, 220), -1)
    t = tracker()
    result = t.select_roi(frame, (300, 200, 60, 50), 0.0)
    assert result.ok and "followed by colour" in result.reason
    for index in range(1, 15):
        moved = floor()
        cv2.rectangle(moved, (300 + 4 * index, 200), (360 + 4 * index, 250), (40, 40, 220), -1)
        report = t.update(moved, index * DT).to_report()
        assert report["state"] == "tracked" and abs(report["centre_px"][0] - (330 + 4 * index)) < 5


def test_size_change_is_followed_by_the_box():
    t = tracker()
    t.select_roi(paste(floor(), sprite(60, 60), (400, 270)), (370, 240, 60, 60), 0.0)
    sizes = []
    for index in range(1, 25):
        side = int(round(60 * (1.0 + 0.02 * index)))
        report = t.update(paste(floor(), cv2.resize(sprite(60, 60), (side, side)), (400, 270)), index * DT).to_report()
        assert report["state"] == "tracked"
        sizes.append(report["box"][2])
    assert sizes[-1] > sizes[0] * 1.2


def test_tracking_runs_fast_enough_for_a_live_view():
    big = floor(size=(1280, 720))
    obj = sprite(120, 90)
    t = tracker()
    t.select_roi(paste(big.copy(), obj, (600, 360)), box_around((600, 360), (120, 90)), 0.0)
    started = time.perf_counter()
    frames = 30
    for index in range(1, frames + 1):
        t.update(paste(big.copy(), obj, (600 + 2 * index, 360)), index * DT)
    per_frame_ms = 1000 * (time.perf_counter() - started) / frames
    assert per_frame_ms < 60, per_frame_ms                                                       # generous: the point is "not seconds"
    print(f"\nSYNTHETIC ROI tracker, 1280x720: {per_frame_ms:.1f} ms per frame on this computer (not camera latency)")


# ---------------------------------------------------------------- uncertainty, loss, recovery, reselection


def moving_run(t, obj, start, frames, begin=1):
    for index in range(begin, begin + frames):
        report = t.update(paste(floor(2 + index % 3), obj, (start[0] + 2 * index, start[1])), index * DT)
    return report


def test_a_hidden_object_gives_uncertain_without_a_position_then_lost_and_is_never_picked_up_again():
    obj = sprite(60, 44)
    start = (400.0, 250.0)
    t = tracker()
    t.select_roi(paste(floor(2), obj, start), box_around(start, (60, 44)), 0.0)
    assert moving_run(t, obj, start, 5).state == TRACKED
    states, positions = [], []
    for index in range(6, 40):
        track = t.update(floor(2), index * DT)                                                   # the object is gone
        states.append(track.state)
        positions.append(track.to_report()["x_m"])
        assert not track.to_opponent_estimate().ok                                               # never presented as a current position
    assert states[0] == UNCERTAIN and "TRACKING UNCERTAIN" in t.estimate.reason or states[-1] == LOST
    assert UNCERTAIN in states and states[-1] == LOST and all(p is None for p in positions)
    assert "Drag a new box" in t.estimate.reason
    # It does not quietly come back, even when the object is shown again.
    assert t.update(paste(floor(2), obj, (400 + 2 * 41, 250)), 41 * DT).state == LOST
    assert t.update(paste(floor(2), obj, (400 + 2 * 42, 250)), 42 * DT).to_report()["x_m"] is None


def test_a_brief_occlusion_is_uncertain_and_recovers_by_itself():
    obj = sprite(60, 44)
    start = (400.0, 250.0)
    t = tracker()
    t.select_roi(paste(floor(2), obj, start), box_around(start, (60, 44)), 0.0)
    moving_run(t, obj, start, 5)
    states = []
    for index in range(6, 14):                                                                   # about 0.27 s hidden
        states.append(t.update(floor(2), index * DT).state)
    assert states[0] == UNCERTAIN and UNCERTAIN in states and LOST not in states
    back = t.update(paste(floor(2), obj, (400 + 2 * 14, 250)), 14 * DT)
    assert back.state == TRACKED and math.dist(centre_of(back.to_report()), (428, 250)) < 4


def test_a_different_object_in_the_same_place_is_not_silently_adopted():
    t = tracker()
    t.select_roi(paste(floor(2), sprite(60, 44, seed=5), (400, 250)), box_around((400, 250), (60, 44)), 0.0)
    moving_run(t, sprite(60, 44, seed=5), (400, 250), 3)
    other = sprite(60, 44, seed=99)
    states = [t.update(paste(floor(2), other, (400, 250)), (4 + i) * DT).state for i in range(30)]
    assert states[0] != TRACKED and states[-1] == LOST


def test_reselection_replaces_the_track_at_any_time_without_restarting_anything():
    first, second = sprite(60, 44, seed=5), sprite(50, 50, seed=21)
    scene = lambda: paste(paste(floor(2), first, (300, 200)), second, (650, 330))
    t = tracker()
    assert t.select_roi(scene(), box_around((300, 200), (60, 44)), 0.0).ok and t.selection_id == 1
    assert t.update(scene(), DT).state == TRACKED
    assert t.select_roi(scene(), box_around((650, 330), (50, 50)), 2 * DT).ok and t.selection_id == 2     # while tracking
    report = t.update(scene(), 3 * DT).to_report()
    assert report["state"] == "tracked" and math.dist(centre_of(report), (650, 330)) < 3 and report["selection"] == 2
    for index in range(4, 60):                                                                  # lose it ...
        t.update(floor(2), index * DT)
    assert t.state == LOST
    assert t.select_roi(scene(), box_around((300, 200), (60, 44)), 61 * DT).ok and t.state == TRACKED and t.selection_id == 3   # ... and recover
    t.clear()
    assert t.state == NOT_SELECTED and t.update(scene(), 62 * DT).state == NOT_SELECTED
    assert t.select_roi(scene(), box_around((650, 330), (50, 50)), 63 * DT).ok


def test_reports_are_json_ready_and_the_uncertain_report_has_no_position():
    import json
    t = tracker()
    t.select_roi(paste(floor(2), sprite(60, 44), (400, 250)), box_around((400, 250), (60, 44)), 0.0)
    ok = t.to_report()
    assert json.loads(json.dumps(ok))["state"] == "tracked" and ok["confidence"] == 1.0 and ok["image_based"]
    t.update(floor(2), DT)
    bad = t.to_report()
    assert bad["state"] == "uncertain" and bad["x_m"] is None and bad["position_quality"] is None and json.dumps(bad)


# ---------------------------------------------------------------- coordinates from the saved calibration


@pytest.fixture(scope="module")
def world():
    config = load_config()
    camera = default_camera()
    mapping, estimate, _ = fit_profile(config, fixture_clicks(config, camera), camera.image_size)
    assert mapping.valid
    return config, camera, mapping


def floor_pixel(camera, x, y, height=0.0):
    u, v = camera.project([[x, y, height]])[0]
    return float(u), float(v)


def test_the_tracked_image_position_is_converted_with_the_calibration(world):
    config, camera, mapping = world
    obj = sprite(46, 38)
    track = R.RoiOpponentTracker(mapping, config.arena)
    start = floor_pixel(camera, 0.80, 0.70)
    frame = lambda c: paste(floor(3, camera.image_size), obj, c)
    assert track.select_roi(frame(start), box_around(start, (46, 38)), 0.0).ok
    for index, (x, y) in enumerate([(0.80 + 0.01 * i, 0.70 - 0.005 * i) for i in range(1, 25)], start=1):
        centre = floor_pixel(camera, x, y)
        report = track.update(frame(centre), index * DT).to_report()
        assert report["state"] == "tracked"
        expected = mapping.pixel_to_floor([centre_of(report)])[0]
        assert (report["x_m"], report["y_m"]) == pytest.approx((expected[0], expected[1]), abs=1e-4)      # exactly the mapping applied to the box centre
        assert math.hypot(report["x_m"] - x, report["y_m"] - y) < 0.02                                   # and close to where the floor point was
    assert "NOT height-corrected" in track.estimate.position_quality and "image-based" in track.estimate.position_quality
    extent = report["diagnostics"]["approx_floor_extent_mm"]
    assert extent[0] > 10 and extent[1] > 10 and "never decide" in report["diagnostics"]["diagnostic_only"]


def test_an_elevated_object_is_labelled_image_based_and_read_farther_from_the_camera(world):
    config, camera, mapping = world
    obj = sprite(40, 40)
    track = R.RoiOpponentTracker(mapping, config.arena)
    high = floor_pixel(camera, 0.60, 0.60, 0.15)                                                    # the same floor point, 150 mm up
    track.select_roi(paste(floor(3, camera.image_size), obj, high), box_around(high, (40, 40)), 0.0)
    report = track.update(paste(floor(3, camera.image_size), obj, high), DT).to_report()
    assert report["state"] == "tracked" and report["image_based"] is True
    shift = math.hypot(report["x_m"] - 0.60, report["y_m"] - 0.60)
    assert shift > 0.02                                                                              # the displacement the label warns about
    assert "displaced" in track.estimate.position_quality


def test_without_a_valid_floor_map_only_the_image_position_is_given():
    t = tracker(mapping=FloorMapping())
    t.select_roi(paste(floor(), sprite(40, 40), (300, 200)), box_around((300, 200), (40, 40)), 0.0)
    report = t.update(paste(floor(), sprite(40, 40), (300, 200)), DT).to_report()
    assert report["state"] == "tracked" and report["x_m"] is None and report["centre_px"] and "no valid floor map" in report["reason"]


def test_an_object_outside_the_arena_rectangle_is_still_followed_and_flagged(world):
    config, camera, mapping = world
    obj = sprite(40, 40)
    track = R.RoiOpponentTracker(mapping, config.arena)
    spot = floor_pixel(camera, 1.40, 0.50)                                                           # beside the arena
    track.select_roi(paste(floor(3, camera.image_size), obj, spot), box_around(spot, (40, 40)), 0.0)
    report = track.update(paste(floor(3, camera.image_size), obj, spot), DT).to_report()
    assert report["state"] == "tracked" and "outside the arena" in report["diagnostics"]["arena_note"]


# ---------------------------------------------------------------- our own robot is never the opponent


def own_scene(world, objects=()):
    config, camera, mapping = world
    frame = tf.render_bot(config, camera, 0.30, 0.30, 0.3, ids=None, seed=3)
    for obj, centre in objects:
        paste(frame, obj, centre)
    return frame


OWN_POSE = BotPose(state="ok", reason="ok", x_m=0.30, y_m=0.30, heading_rad=0.3, floor_projection_xy=(0.30, 0.30), used_ids=[0])


def test_a_box_on_our_own_robot_is_refused_and_a_box_beside_it_is_accepted(world):
    config, camera, mapping = world
    track = R.RoiOpponentTracker(mapping, config.arena)
    own = floor_pixel(camera, 0.30, 0.30, 0.034)
    refused = track.select_roi(own_scene(world), box_around(own, (50, 40)), 0.0, OWN_POSE)
    assert not refused.ok and "own tagged robot" in refused.reason and track.state == NOT_SELECTED
    obj = sprite(44, 36)
    spot = floor_pixel(camera, 0.85, 0.80)
    accepted = track.select_roi(own_scene(world, [(obj, spot)]), box_around(spot, (44, 36)), 0.0, OWN_POSE)
    assert accepted.ok and track.state == TRACKED
    unchecked = R.RoiOpponentTracker(mapping, config.arena).select_roi(own_scene(world, [(obj, spot)]), box_around(spot, (44, 36)), 0.0)
    assert unchecked.ok and "could not be checked" in unchecked.reason                               # no own pose: said, not hidden


def test_a_track_that_runs_onto_our_own_robot_is_uncertain_with_no_position_and_does_not_learn_it(world):
    config, camera, mapping = world
    track = R.RoiOpponentTracker(mapping, config.arena)
    obj = sprite(44, 36)
    own = floor_pixel(camera, 0.30, 0.30)
    start = floor_pixel(camera, 0.62, 0.30)
    track.select_roi(own_scene(world, [(obj, start)]), box_around(start, (44, 36)), 0.0, OWN_POSE)
    track.update(own_scene(world, [(obj, start)]), DT, OWN_POSE)
    template_before = track._adaptive.copy()
    seen = set()
    for index, x in enumerate(np.linspace(0.62, 0.30, 12), start=2):
        centre = floor_pixel(camera, float(x), 0.30)
        result = track.update(own_scene(world, [(obj, centre)]), index * DT, OWN_POSE)
        seen.add(result.state)
    assert UNCERTAIN in seen and "own robot" in track.estimate.reason and track.estimate.xy_m is None
    assert not track.to_opponent_estimate().ok
    assert np.array_equal(track._adaptive, template_before) or UNCERTAIN in seen                      # no learning while overlapped


# ---------------------------------------------------------------- the tracker backends


def test_this_build_has_no_csrt_or_kcf_so_the_built_in_tracker_is_used():
    assert R.available_cv_backends() == [] or all(callable(factory) for _, factory in R.available_cv_backends())
    backend = R.make_backend(R.RoiSettings())
    if not R.available_cv_backends():
        assert isinstance(backend, R.TemplateBackend) and "no CSRT/KCF" in backend.name


def test_an_opencv_tracker_is_used_when_one_exists_and_confidence_is_still_ours(monkeypatch):
    class FakeTracker:
        def init(self, frame, box):
            self.box = np.array(box, float)

        def update(self, frame):
            self.box[0] += 3
            return True, tuple(self.box)

    monkeypatch.setattr(cv2, "TrackerCSRT_create", lambda: FakeTracker(), raising=False)
    assert [name for name, _ in R.available_cv_backends()] == ["OpenCV CSRT"]
    obj = sprite(60, 44)
    t = tracker()
    t.select_roi(paste(floor(2), obj, (300, 200)), box_around((300, 200), (60, 44)), 0.0)
    assert t.backend.name == "OpenCV CSRT"
    good = t.update(paste(floor(2), obj, (303, 200)), DT).to_report()
    assert good["state"] == "tracked" and good["backend"] == "OpenCV CSRT" and good["confidence"] > 0.8
    wrong = t.update(floor(2), 2 * DT).to_report()                                                   # the library still claims a box: our similarity says no
    assert wrong["state"] in ("uncertain", "lost") and wrong["x_m"] is None


def test_a_tracker_that_cannot_start_refuses_the_box_with_a_reason(monkeypatch):
    def broken():
        raise RuntimeError("no model file")
    t = R.RoiOpponentTracker(FloorMapping(), {"width_m": 1.15, "height_m": 1.15}, backend_factory=lambda s: R.CvTrackerBackend(broken, "broken"))
    result = t.select_roi(floor(), (100, 100, 40, 40), 0.0)
    assert not result.ok and "could not start" in result.reason and t.state == NOT_SELECTED


def test_settings_reject_unknown_names():
    assert R.RoiSettings.from_dict({"confident": 0.7}).confident == 0.7
    with pytest.raises(ValueError):
        R.RoiSettings.from_dict({"max_size_m": 0.28})


# ---------------------------------------------------------------- inside the existing pipeline


def test_the_pipeline_in_roi_mode_selects_by_box_and_reports_image_based_tracks(world):
    config, camera, mapping = world
    _, estimate, _ = fit_profile(config, fixture_clicks(config, camera), camera.image_size)
    obj = sprite(44, 36)
    spot = floor_pixel(camera, 0.85, 0.80)
    pipeline = CameraPipeline(config, mapping, estimate, multi_marker=True, click_opponent=True, opponent_mode="roi")
    frame = own_scene(world, [(obj, spot)])
    for index in range(3):
        pipeline.process(frame, index * DT)
    result = pipeline.select_roi(frame, box_around(spot, (44, 36)), 3 * DT)
    assert result.ok
    sample = pipeline.process(frame, 4 * DT)
    report = sample.opponent_track
    assert report["state"] == "tracked" and report["image_based"] and "drag-selected" in sample.opponent.source
    assert sample.opponent.ok and sample.own_track["state"] in ("measured", "predicted")
    assert not pipeline.select_opponent(frame, spot, 5 * DT).ok                                       # a click is not a selection
    pipeline.clear_opponent()
    assert pipeline.process(frame, 6 * DT).opponent_track["state"] == "not_selected"
    old = CameraPipeline(config, mapping, estimate, multi_marker=True, click_opponent=True)            # the default mode is unchanged
    with pytest.raises(ValueError):
        old.select_roi(frame, (1, 1, 50, 50), 0.0)


def test_live_track_events_accept_a_box(world):
    from control_center.tools.live_track import LiveTrack
    config, camera, mapping = world
    track = LiveTrack(config, fixture_clicks(config, camera), opponent_mode="roi")
    obj = sprite(44, 36)
    spot = floor_pixel(camera, 0.85, 0.80)
    frame = own_scene(world, [(obj, spot)])
    _, record = track.step(frame, 0.0, [("roi", box_around(spot, (44, 36)))])
    assert record["events"][0]["event"] == "roi" and record["events"][0]["accepted"] is True
    assert record["opponent_track"]["state"] == "tracked"
    _, record = track.step(frame, DT, [("roi", (5, 5, 1, 1))])
    assert record["events"][0]["accepted"] is False and "empty" in record["events"][0]["reason"]
    assert record["opponent_track"]["state"] == "tracked"                                              # the bad box changed nothing
    assert any(text.startswith("OPPONENT") for text, _ in track.lines(track.pipeline.process(frame, 2 * DT)))
