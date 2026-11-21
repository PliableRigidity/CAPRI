"""Profiles, fresh live calibration, fit grading, the fisheye lens model, footage calibration maths
and the unmarked-blob tracker.

Everything here runs on SYNTHETIC pictures or plain numbers, except the last test, which needs the
recorded footage and is skipped unless AUTOBOTS_FOOTAGE_MP4 points at it. None of the tolerances
are real-camera accuracy.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import cv2
import numpy as np
import pytest

from control_center.tools import arena_overlay, footage_reference
from control_center.tools.blob_tracker import LOST, OVERLAP, TRACKED, BlobTracker, arena_floor_polygon, floor_mask
from control_center.tools.overlay_core import (
    LIVE_PROFILE, PROFILES_DIR, ClickSet, PixelModel, apply_homography, assess_fit, fit_floor_mapping,
    load_clicks, load_config, profile_points_path, save_clicks, topdown_pixel, topdown_view)
from control_center.tools.overlay_fixture import default_camera, make_camera, reference_pixels, render_frame

REPO = Path(__file__).resolve().parents[1]
HISTORICAL = "historical_bristol_2026_03_22"


@pytest.fixture(scope="module")
def config():
    return load_config()


@pytest.fixture(scope="module")
def side_camera():
    """A low view from beyond the pit corner at (1.15, 0): nothing like the footage's viewpoint."""
    return make_camera((1.85, -0.55, 0.85), (0.50, 0.55, 0.0), focal_px=800.0)


def args_for(**overrides):
    base = dict(points=None, profile=LIVE_PROFILE, calibrate=False, camera=None, image=None, video=None)
    base.update(overrides)
    return argparse.Namespace(**base)


# ---- lens model -------------------------------------------------------------------------------

def test_fisheye_lens_round_trip_and_mapping():
    model = footage_reference.fisheye_model(350.0, (640, 360))
    pixels = np.array([[20.0, 15.0], [320.0, 180.0], [600.0, 340.0], [100.0, 300.0], [500.0, 40.0]])
    assert np.allclose(model.distort(model.undistort(pixels)), pixels, atol=1e-3)
    assert model.active and PixelModel.from_dict(model.to_dict()).model == "fisheye"
    assert not PixelModel().active and not PixelModel.from_dict(None).active
    # a known floor homography seen through that lens is recovered from distorted clicks
    H = np.array([[260.0, -40.0, 180.0], [30.0, -250.0, 330.0], [0.05, -0.2, 1.0]])
    refs = load_config().references
    clicks = {ref.name: tuple(model.distort(apply_homography(H, [ref.arena_xy]))[0]) for ref in refs}
    mapping = fit_floor_mapping(refs, clicks, (640, 360), model)
    assert mapping.valid and mapping.check_stats["max_mm"] < 0.5
    plain = fit_floor_mapping(refs, clicks, (640, 360))                      # ignoring the lens leaves a visible error
    assert not plain.valid or plain.check_stats["max_mm"] > 10.0


def test_focal_length_is_recovered_from_edge_straightness():
    truth = footage_reference.fisheye_model(350.0, (640, 360)); rng = np.random.default_rng(3)
    edges = {}
    for name, (a, b) in {"left": ((-260, -150), (-330, 250)), "top": ((-100, -230), (160, -240)), "diag": ((170, -240), (330, -130)),
                         "right": ((335, -120), (430, 190))}.items():
        ideal = np.linspace(a, b, 120) + np.array([320.0, 180.0])            # straight in an ideal image
        edges[name] = truth.distort(ideal) + rng.normal(0.0, 0.25, (120, 2))
    fit = footage_reference.estimate_fisheye_focal(edges, (640, 360))
    assert abs(fit["focal_px"] - 350.0) < 25.0 and not fit["at_search_limit"]
    assert fit["straightness_rms_px"] < 0.6 < 2.0 < fit["straightness_rms_px_uncorrected"]       # 0.25 px noise grows a little when undistorted
    low, high = fit["focal_range_within_0.1px_rms"]
    assert 290.0 < low <= fit["focal_px"] <= high < 420.0


def test_landmarks_are_where_fitted_edges_meet():
    H = np.array([[300.0, 20.0, 150.0], [10.0, -280.0, 340.0], [0.02, -0.15, 1.0]])
    angles = np.radians(np.linspace(-88, -2, 80))
    arc = apply_homography(H, np.column_stack([0.25 * np.cos(angles), 1.15 + 0.25 * np.sin(angles)]))    # pit rim at (0, 1.15)
    wall = apply_homography(H, np.column_stack([np.zeros(50), np.linspace(0.2, 0.88, 50)]))             # the x = 0 wall
    top = apply_homography(H, np.column_stack([np.linspace(0.27, 0.8, 50), np.full(50, 1.15)]))
    conic = footage_reference.Conic(arc + np.random.default_rng(0).normal(0, 0.1, arc.shape))
    truth_wall, truth_top = apply_homography(H, [[0.0, 0.90], [0.25, 1.15]])
    assert np.linalg.norm(conic.meet_line(footage_reference.fit_line(wall), truth_wall + 3.0) - truth_wall) < 0.6
    assert np.linalg.norm(conic.meet_line(footage_reference.fit_line(top), truth_top + 3.0) - truth_top) < 0.6
    corner = footage_reference.meet_lines(footage_reference.fit_line(wall), footage_reference.fit_line(top))
    assert np.linalg.norm(corner - apply_homography(H, [[0.0, 1.15]])[0]) < 0.3
    leaky = np.vstack([wall, wall[10:20] + [6.0, 0.0]])                      # a stretch where the colour mask leaked
    assert len(footage_reference.smooth_inliers(leaky, 1)) == 50


# ---- fit grading ------------------------------------------------------------------------------

def test_fit_is_graded_ok_weak_or_insufficient(config):
    camera = default_camera(); pixels = reference_pixels(config, camera); refs = config.references

    def grade(names, roles=None):
        clicks = {name: pixels[name] for name in names}
        mapping = fit_floor_mapping(refs, clicks, camera.image_size, roles=roles)
        return assess_fit(refs, clicks, roles, mapping, config.arena), mapping

    full, _ = grade([ref.name for ref in refs])
    assert full.level == "ok" and full.reasons == [] and full.hull_fraction > 0.5 and full.predicted_worst_mm < 25
    four = ["corner_origin", "pitA_rim_x0_wall", "pitB_rim_y0_wall", "flipper_edge_ymax_wall"]
    unchecked, mapping = grade(four)
    assert mapping.valid and unchecked.level == "weak" and "no check point" in unchecked.reasons[0]
    cluster = {"corner_origin": "fit", "tape_1": "fit", "tape_2": "fit", "pitB_rim_y0_wall": "fit", "pitA_rim_x0_wall": "check"}
    clustered, mapping = grade(list(cluster), cluster)                    # every fit point along the near side
    assert mapping.valid and clustered.level == "weak" and clustered.hull_fraction < 0.25
    assert any("span only" in reason or "click error moves the map" in reason for reason in clustered.reasons)
    too_few, mapping = grade(four[:3])
    assert not mapping.valid and too_few.level == "insufficient" and "at least 4" in too_few.reasons[0]
    line = {name: "fit" for name in ("corner_origin", "tape_1", "tape_centre", "tape_4")}                # all on the diagonal
    collinear, mapping = grade(list(line), line)
    assert not mapping.valid and collinear.level == "insufficient" and "collinear" in collinear.reasons[0]
    shifted = dict(pixels); shifted["tape_centre"] = (pixels["tape_centre"][0] + 25.0, pixels["tape_centre"][1])
    mapping = fit_floor_mapping(refs, shifted, camera.image_size)
    off = assess_fit(refs, shifted, None, mapping, config.arena)
    assert off.level == "weak" and any("tape_centre is off by" in reason for reason in off.reasons)


# ---- profiles: footage and live stay apart --------------------------------------------------------

def test_profile_paths_and_kinds(tmp_path):
    assert profile_points_path(LIVE_PROFILE).name == "arena_overlay_clicks.json"
    assert profile_points_path("abc") == PROFILES_DIR / "abc" / "clicks.json"
    for bad in ("", "../x", "a/b", ".hidden", "c:evil"):
        with pytest.raises(ValueError): profile_points_path(bad)
    live = ClickSet({"a": (1.0, 2.0)}, {}, (640, 360)); assert live.kind == "live" and not live.historical and not live.lens.active
    footage = ClickSet({"a": (1.0, 2.0)}, {}, (640, 360), {"profile_kind": "historical", "lens": footage_reference.fisheye_model(350, (640, 360)).to_dict()})
    save_clicks(tmp_path / "h.json", footage); again = load_clicks(tmp_path / "h.json")
    assert again.historical and again.lens.active and again.lens.model == "fisheye"


def test_a_footage_profile_is_never_loaded_for_a_live_camera(tmp_path):
    path = tmp_path / "footage.json"
    save_clicks(path, ClickSet({"corner_origin": (10.0, 10.0)}, {}, (640, 360), {"profile_kind": "historical", "source_video": "someone_elses.mp4"}))
    with pytest.raises(ValueError, match="HISTORICAL footage profile"):
        arena_overlay.load_profile_clicks(args_for(points=str(path), camera=0), "camera")
    clicks, _ = arena_overlay.load_profile_clicks(args_for(points=str(path), video="x.mp4"), "file")    # fine for replaying the footage
    assert clicks.historical
    with pytest.raises(ValueError, match="historical"):
        arena_overlay.load_profile_clicks(args_for(points=str(path), calibrate=True), "file")           # and never overwritten by --calibrate
    # the tool refuses before it would open a camera
    assert arena_overlay.main(["--camera", "0", "--points", str(path), "--no-window"]) == 2


def test_the_bristol_footage_profile_is_marked_historical_and_separate():
    path = profile_points_path(HISTORICAL)
    if not path.is_file(): pytest.skip("historical footage profile not generated")
    clicks = load_clicks(path)
    assert clicks.historical and clicks.meta["approximate"] and clicks.lens.model == "fisheye" and clicks.image_size == (640, 360)
    assert set(clicks.points) == {"pitA_rim_x0_wall", "pitA_rim_ymax_wall", "flipper_edge_ymax_wall", "flipper_edge_xmax_wall", "pitB_rim_xmax_wall"}
    assert sum(role == "fit" for role in clicks.roles.values()) == 4 and clicks.roles["flipper_edge_ymax_wall"] == "check"
    assert path != profile_points_path(LIVE_PROFILE) and path.parent.parent == PROFILES_DIR
    live = profile_points_path(LIVE_PROFILE)
    if live.is_file(): assert not load_clicks(live).historical
    assert load_config().camera_matrix is None                    # the footage lens lives in its profile, not in the shared config


# ---- fresh live calibration on a synthetic side view ---------------------------------------------

VISIBLE_FROM_THE_SIDE = {"corner_origin": "fit", "pitA_rim_x0_wall": "fit", "flipper_edge_ymax_wall": "fit", "flipper_edge_xmax_wall": "fit",
                         "pitA_rim_ymax_wall": "check", "tape_centre": "check"}


def calibrate_by_clicking(app, pixels, names, noise_px=0.0, seed=0):
    rng = np.random.default_rng(seed); order = [ref.name for ref in app.config.references]
    for name in names:
        chosen = app.session.select(order.index(name)); assert chosen.name == name       # pick the landmark by name, in any order
        placed = app.place_click(*(np.asarray(pixels[name]) + rng.normal(0.0, noise_px, 2)))
        assert placed == name


def test_fresh_live_calibration_from_a_side_view(config, side_camera, tmp_path):
    frame = render_frame(config, side_camera); pixels = reference_pixels(config, side_camera)
    w, h = side_camera.image_size
    assert all(0 <= pixels[name][0] < w and 0 <= pixels[name][1] < h for name in VISIBLE_FROM_THE_SIDE)
    source = arena_overlay.ImageSource.__new__(arena_overlay.ImageSource); source._frame = frame; source.description = "synthetic side view"
    points = tmp_path / "live_clicks.json"
    save_clicks(points, ClickSet({"corner_origin": (1.0, 1.0)}, {}, (w, h), {"profile_kind": "live"}))      # an older calibration exists
    clicks, path = arena_overlay.load_profile_clicks(args_for(points=str(points), calibrate=True, camera=0), "camera")
    assert clicks.points == {}                                             # fresh: nothing reused
    app = arena_overlay.OverlayApp(config, clicks, "camera: synthetic side view"); app.calibrating = True; app.click_mode = True
    app.process(frame, 0.0, source)
    assert not app.mapping.valid and app.quality.level == "insufficient" and "NOT saved" in arena_overlay.save_profile(app, path)
    assert "not calibrated yet" in app.profile_line()[0]
    names = list(VISIBLE_FROM_THE_SIDE)
    calibrate_by_clicking(app, pixels, names[:3]); app.process(frame, 0.0, source)
    assert app.quality.level == "insufficient"                             # three points: says so, draws no outline
    calibrate_by_clicking(app, pixels, names[3:4]); app.process(frame, 0.0, source)
    assert app.mapping.valid and app.quality.level == "weak" and "no check point" in " ".join(app.quality.reasons)
    for name in names[4:]:
        app.session.select([ref.name for ref in config.references].index(name))
        if app.session.role_of(app.session.current) != VISIBLE_FROM_THE_SIDE[name]: app.session.toggle_role()
        app.place_click(*pixels[name])
    out = app.process(frame, 0.0, source)
    assert app.mapping.valid and app.mapping.check_stats["count"] == 2 and app.mapping.check_stats["max_mm"] < 1.0
    assert app.quality.level in ("ok", "weak") and "FIT:" in app.fit_line()[0]
    outline = app.mapping.floor_to_pixel([[0.0, 0.0], [1.15, 1.15]])       # the fitted outline lands on the true corners
    truth = side_camera.project([[0.0, 0.0, 0.0], [1.15, 1.15, 0.0]])
    assert np.abs(outline - truth).max() < 1.0 and out.shape == frame.shape
    message = arena_overlay.save_profile(app, path)
    assert "saved 6 point(s)" in message and "previous file kept" in message and len(list(tmp_path.glob("live_clicks.json.bak-*"))) == 1
    saved = load_clicks(path)
    assert saved.kind == "live" and saved.image_size == (w, h) and not saved.lens.active and "side view" in saved.meta["source"]
    # next start: the saved calibration loads; a different resolution or a moved camera does not pass silently
    again, _ = arena_overlay.load_profile_clicks(args_for(points=str(points), camera=0), "camera")
    app2 = arena_overlay.OverlayApp(config, again, "camera: synthetic side view"); app2.process(frame, 0.0, source)
    assert app2.mapping.valid and "calibrated" in app2.profile_line()[0]
    smaller = cv2.resize(frame, (w // 2, h // 2)); app2.process(smaller, 0.0, source)
    assert not app2.mapping.valid and "click them again" in app2.mapping.reason and app2.quality.level == "insufficient"


def test_noisy_side_view_clicks_show_in_the_check_points_not_hidden(config, side_camera):
    pixels = reference_pixels(config, side_camera); frame = render_frame(config, side_camera)
    source = arena_overlay.ImageSource.__new__(arena_overlay.ImageSource); source._frame = frame; source.description = "synthetic"
    app = arena_overlay.OverlayApp(config, ClickSet(), "image: synthetic"); app.process(frame, 0.0, source)
    names = list(VISIBLE_FROM_THE_SIDE)
    for name in names:
        app.session.select([ref.name for ref in config.references].index(name))
        if app.session.role_of(app.session.current) != VISIBLE_FROM_THE_SIDE[name]: app.session.toggle_role()
        offset = (9.0, -7.0) if name == "flipper_edge_xmax_wall" else (0.0, 0.0)             # one careless click
        app.place_click(pixels[name][0] + offset[0], pixels[name][1] + offset[1])
    app.process(frame, 0.0, source)
    assert app.mapping.valid and app.mapping.check_stats["max_mm"] > 10.0 and app.quality.level == "weak"
    assert any("is off by" in reason for reason in app.quality.reasons)


def test_cli_fresh_calibration_ignores_saved_points(tmp_path, config, side_camera):
    frame_path = tmp_path / "side.png"; cv2.imwrite(str(frame_path), render_frame(config, side_camera))
    points = tmp_path / "clicks.json"; pixels = reference_pixels(config, side_camera)
    save_clicks(points, ClickSet({name: tuple(pixels[name]) for name in VISIBLE_FROM_THE_SIDE}, dict(VISIBLE_FROM_THE_SIDE), side_camera.image_size))
    report = tmp_path / "r.json"
    assert arena_overlay.main(["--image", str(frame_path), "--points", str(points), "--no-window", "--report", str(report)]) == 0
    loaded = json.loads(report.read_text(encoding="utf-8"))
    assert loaded["mapping"]["valid"] and loaded["profile"]["kind"] == "live" and loaded["fit_quality"]["level"] in ("ok", "weak")
    assert any("moved" in line for line in loaded["profile"]["when_to_recalibrate"])
    assert arena_overlay.main(["--image", str(frame_path), "--points", str(points), "--calibrate", "--no-window", "--report", str(report)]) == 0
    fresh = json.loads(report.read_text(encoding="utf-8"))
    assert not fresh["mapping"]["valid"] and fresh["fit_quality"]["level"] == "insufficient"
    assert load_clicks(points).points                                      # nothing is overwritten until s is pressed


def test_topdown_view_puts_floor_points_where_expected(config, side_camera):
    frame = render_frame(config, side_camera)
    mapping = fit_floor_mapping(config.references, reference_pixels(config, side_camera), side_camera.image_size)
    view = topdown_view(frame, mapping, config.arena, 200.0, 0.05)
    assert view.shape[:2] == (250, 250)
    assert np.allclose(topdown_pixel([[0.0, 0.0], [1.15, 1.15]], config.arena, 200.0, 0.05), [[10.0, 240.0], [240.0, 10.0]])


# ---- unmarked-blob tracker (synthetic blobs) ------------------------------------------------------

@pytest.fixture(scope="module")
def stage(config):
    camera = default_camera(); background = render_frame(config, camera)
    mapping = fit_floor_mapping(config.references, reference_pixels(config, camera), camera.image_size)
    return camera, background, mapping


def paint(background, camera, bots):
    """Flat coloured squares on the floor: (x, y, half_size, bgr)."""
    frame = background.copy(); H = camera.plane_homography()
    for x, y, half, colour in bots:
        corners = apply_homography(H, [[x - half, y - half], [x + half, y - half], [x + half, y + half], [x - half, y + half]])
        cv2.fillConvexPoly(frame, np.round(corners).astype(np.int32), colour)
    return frame


RED, WHITE = (40, 40, 220), (245, 245, 245)


def run_meeting(config, stage, colour_a, colour_b):
    camera, background, mapping = stage; tracker = BlobTracker(mapping, config.arena, background); log = []
    for frame_index in range(110):
        close = min(1.0, frame_index / 30.0) if frame_index < 60 else max(0.0, (100 - frame_index) / 40.0)
        ax, bx = 0.30 + 0.20 * close, 0.90 - 0.28 * close                 # touching (centres 0.12 m apart) between frames 30 and 60
        tracker.update(paint(background, camera, [(ax, 0.50, 0.055, colour_a), (bx, 0.55, 0.055, colour_b)]))
        log.append((frame_index, ax, bx, [(t.track_id, t.state, t.floor, t.identity_uncertain) for t in tracker.tracks]))
    return tracker, log


def test_tracker_reports_overlap_without_positions_and_keeps_identities(config, stage):
    tracker, log = run_meeting(config, stage, RED, WHITE)
    assert [t.track_id for t in tracker.tracks] == [1, 2]
    states = {frame: {tid: (state, floor) for tid, state, floor, _ in tracks} for frame, _, _, tracks in log}
    assert all(states[f][1][0] == TRACKED and states[f][2][0] == TRACKED for f in range(12, 22))
    for frame in range(36, 58):                                            # one shared blob: nobody gets a position
        assert states[frame][1] == (OVERLAP, None) and states[frame][2] == (OVERLAP, None)
    _, ax, bx, final = log[-1]; by_id = {tid: (state, floor, uncertain) for tid, state, floor, uncertain in final}
    first = states[15]                                                     # which id started on the left?
    left_id = min((1, 2), key=lambda tid: first[tid][1][0]); right_id = 3 - left_id
    assert by_id[left_id][0] == TRACKED and abs(by_id[left_id][1][0] - ax) < 0.03 and abs(by_id[right_id][1][0] - bx) < 0.03
    assert not by_id[left_id][2] and not by_id[right_id][2]                 # different colours: no doubt
    names = [event["event"] for event in tracker.events]
    assert names.count("overlap begins") == 2 and "overlap ends" in names and tracker.frame_report()["overlaps"] == []


def test_tracker_flags_identity_when_colour_cannot_tell(config, stage):
    tracker, _ = run_meeting(config, stage, WHITE, WHITE)
    assert len(tracker.tracks) == 2 and all(t.state == TRACKED and t.identity_uncertain for t in tracker.tracks)
    assert all(t.label.endswith("?") and "not available" in t.to_report()["heading"] for t in tracker.tracks)


def test_tracker_keeps_a_stationary_robot_and_ignores_everything_off_the_floor(config, stage):
    camera, background, mapping = stage; tracker = BlobTracker(mapping, config.arena, background)
    outside = [(1.40, 0.50, 0.06, RED), (1.07, 0.07, 0.04, WHITE), (1.08, 1.08, 0.04, RED), (0.07, 1.08, 0.04, WHITE), (0.5, -0.2, 0.06, RED)]
    for _ in range(150):                                                   # five seconds without moving
        tracker.update(paint(background, camera, [(0.30, 0.30, 0.055, RED)] + outside))
    assert len(tracker.tracks) == 1
    track = tracker.tracks[0]
    assert track.state == TRACKED and math.dist(track.floor, (0.30, 0.30)) < 0.02 and track.still_frames > 120
    assert tracker.mask[tuple(np.round(camera.project([[1.08, 1.08, 0.0]])[0][::-1]).astype(int))] == 0     # fixed flipper corner is masked
    for _ in range(3): tracker.update(background)                          # it vanishes: lost at once, no invented position
    assert track.state == LOST and track.floor is None and track.to_report()["x_m"] is None
    for _ in range(100): tracker.update(background)
    assert tracker.tracks == []


def test_playable_floor_polygon_excludes_pits_and_the_flipper_corner(config, stage):
    polygon = arena_floor_polygon(config.arena).astype(np.float32).reshape(-1, 1, 2)
    inside = lambda x, y: cv2.pointPolygonTest(polygon, (x, y), False) > 0
    assert inside(0.575, 0.575) and inside(0.05, 0.05) and inside(0.6, 1.1) and inside(1.1, 0.5)
    assert not inside(1.1, 0.05) and not inside(0.05, 1.1) and not inside(1.1, 1.1) and not inside(1.2, 0.5)
    camera, _, mapping = stage
    mask = floor_mask(mapping, config.arena, camera.image_size)
    assert 0.6 < mask.mean() / 255 * camera.image_size[0] * camera.image_size[1] / cv2.contourArea(
        np.round(camera.project(np.column_stack([arena_floor_polygon(config.arena), np.zeros(len(polygon))]))).astype(np.int32)) < 1.01


def test_cli_track_needs_a_video(tmp_path, config, side_camera):
    frame_path = tmp_path / "side.png"; cv2.imwrite(str(frame_path), render_frame(config, side_camera))
    assert arena_overlay.main(["--image", str(frame_path), "--track", "--no-window"]) == 2


# ---- the recorded footage (only when it is available) ---------------------------------------------

def test_recorded_clip_tracks_two_robots_through_contact():
    video = os.environ.get("AUTOBOTS_FOOTAGE_MP4")
    if not video or not Path(video).is_file() or not profile_points_path(HISTORICAL).is_file():
        pytest.skip("set AUTOBOTS_FOOTAGE_MP4 to the Bristol recording to run this")
    report = arena_overlay.run(arena_overlay.build_parser().parse_args(
        ["--video", video, "--profile", HISTORICAL, "--start", "433", "--duration", "5.5", "--track", "--no-window"]))
    assert report["profile"]["kind"] == "historical" and report["profile"]["approximate"] and report["mapping"]["valid"]
    events = [event["event"] for event in report["tracking"]["events"]]
    assert "overlap begins" in events and "overlap ends" in events
    last = report["frames"][-1]["blobs"]["tracks"]                         # 438.5 s: apart again after the first contact
    assert sum(track["state"] == "tracked" for track in last) == 3          # two competitors and the parked house robot
    touching = [track for frame in report["frames"] for track in frame["blobs"]["tracks"] if track["state"] == "overlap"]
    assert len(touching) > 20 and all(track["x_m"] is None for track in touching)
    assert all(track["x_m"] is None for frame in report["frames"] for track in frame["blobs"]["tracks"] if track["state"] == "lost")


def test_live_calibrate_command_end_to_end_with_a_stand_in_camera_and_window(tmp_path, monkeypatch, config, side_camera):
    """`--camera 0 --calibrate` driven through the real window loop: freeze, pick landmarks by number, click, save.

    The camera and the window are stand-ins (no real camera is opened, nothing is shown)."""
    frame = render_frame(config, side_camera); pixels = reference_pixels(config, side_camera)
    shown: list[np.ndarray] = []; state: dict = {}

    class FakeCapture:
        def __init__(self, index, backend=None): self.reads = 0
        def isOpened(self): return True
        def set(self, prop, value): return True
        def read(self): self.reads += 1; state["reads"] = self.reads; return True, frame.copy()
        def release(self): pass

    keys = {"corner_origin": "1", "pitA_rim_x0_wall": "2", "flipper_edge_ymax_wall": "4", "flipper_edge_xmax_wall": "5",
            "pitA_rim_ymax_wall": "6", "tape_centre": "8"}
    script: list = []
    for name, key in keys.items(): script += [("key", ord(key)), ("click", pixels[name])]
    script += [("key", ord("s")), ("key", ord("q"))]

    def wait_key(delay):
        state.setdefault("reads_at_first_key", state["reads"])
        kind, value = script.pop(0)
        if kind == "click":
            state["mouse"](cv2.EVENT_LBUTTONDOWN, int(round(value[0])), int(round(value[1])), 0, None); return 255
        return value

    monkeypatch.setattr(arena_overlay.cv2, "VideoCapture", FakeCapture)
    monkeypatch.setattr(arena_overlay.cv2, "namedWindow", lambda *a, **k: None)
    monkeypatch.setattr(arena_overlay.cv2, "setMouseCallback", lambda name, callback: state.__setitem__("mouse", callback))
    monkeypatch.setattr(arena_overlay.cv2, "imshow", lambda name, image: shown.append(image.copy()))
    monkeypatch.setattr(arena_overlay.cv2, "waitKey", wait_key)
    monkeypatch.setattr(arena_overlay.cv2, "getWindowProperty", lambda *a: 1.0)
    monkeypatch.setattr(arena_overlay.cv2, "destroyAllWindows", lambda: None)
    points = tmp_path / "live.json"
    report = arena_overlay.run(arena_overlay.build_parser().parse_args(["--camera", "0", "--calibrate", "--points", str(points)]))
    assert script == [] and state["reads"] == state["reads_at_first_key"]      # the picture stayed frozen while clicking
    saved = load_clicks(points)
    assert set(saved.points) == set(keys) and saved.kind == "live" and saved.image_size == side_camera.image_size
    assert saved.meta["fit_quality_when_saved"] in ("ok", "weak")
    assert report["mapping"]["valid"] and report["mapping"]["check_points"]["count"] == 2 and report["mapping"]["check_points"]["max_mm"] < 5.0
    assert report["profile"]["kind"] == "live" and len(shown) >= len(keys) and shown[-1].shape == frame.shape


def test_arena_config_equals_the_arena_floor_technical_drawing(config):
    """Numbers read off "Arena Floor", rev 1, 11/11/2025 (millimetres). The PDF is not in the repo."""
    drawing = {"side": 1150, "pit_radius": 250, "rim_from_plain_corner": 900, "rim_from_pit_corner": 250, "diagonal_leg": 335, "diagonal_angle_deg": 135}
    arena = config.arena; mm = lambda value: round(1000 * float(value))
    assert mm(arena["width_m"]) == mm(arena["height_m"]) == drawing["side"] and mm(arena["pit_radius_m"]) == drawing["pit_radius"]
    assert sorted(map(tuple, np.round(np.asarray(arena["pit_centres_m"]) * 1000).astype(int).tolist())) == [(0, 1150), (1150, 0)]
    flipper = sorted(map(tuple, np.round(np.asarray(arena["flipper_vertices_m"]) * 1000).astype(int).tolist()))
    assert flipper == [(815, 1150), (1150, 815), (1150, 1150)] and 1150 - 815 == drawing["diagonal_leg"]
    (ax, ay), (bx, by) = flipper[0], flipper[1]                              # the cut runs at 45 degrees, i.e. 135 degrees to each wall
    assert abs(math.degrees(math.atan2(by - ay, bx - ax))) == 180 - drawing["diagonal_angle_deg"]
    at = {ref.name: tuple(round(1000 * v) for v in ref.arena_xy) for ref in config.references}
    assert at["pitA_rim_x0_wall"] == (0, 900) and at["pitB_rim_y0_wall"] == (900, 0)
    assert at["pitA_rim_ymax_wall"] == (250, 1150) and at["pitB_rim_xmax_wall"] == (1150, 250)
    assert at["flipper_edge_ymax_wall"] == (815, 1150) and at["flipper_edge_xmax_wall"] == (1150, 815) and at["corner_origin"] == (0, 0)
    assert not arena["measured"]                                             # a drawing is not a measurement of the arena as built
