"""Offline checks for the read-only arena overlay / ArUco bot-pose tool.

Every image used here is rendered by control_center.tools.overlay_fixture: a SYNTHETIC scene
seen through an ideal pinhole camera with a known pose. The tolerances below prove the maths
and the plumbing; they are NOT real-camera accuracy figures.
"""

from __future__ import annotations

import ast
from dataclasses import replace
import importlib.util
import json
import math
from pathlib import Path
import sys

import cv2
import numpy as np
import pytest
import yaml

from control_center.tools import arena_overlay, overlay_fixture
from control_center.tools.overlay_core import (
    ClickSession, ClickSet, MarkerDetector, PixelModel, PoseTracker, ReferencePoint,
    apply_homography, arena_outline, derive_camera_from_homography, estimate_bot_pose,
    fit_floor_mapping, flipper_boundary, load_clicks, load_config, parse_config, pit_arcs,
    resolve_camera, save_clicks)
from control_center.tools.overlay_fixture import (
    BotInstance, MarkerPlacement, Scene, default_camera, demo_scenes, fixture_clicks,
    make_camera, reference_pixels, render_frame, write_fixture)

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def config():
    return load_config()


@pytest.fixture(scope="module")
def camera():
    return default_camera()


@pytest.fixture(scope="module")
def mapping(config, camera):
    return fit_floor_mapping(config.references, reference_pixels(config, camera),
                             camera.image_size)


def pose_for(config, camera, mapping, scene, render_config=None):
    """Render a synthetic scene, detect the markers and estimate the pose with ``config``."""
    frame = render_frame(render_config or config, camera, scene)
    detections = MarkerDetector(config.dictionary).detect_markers(frame)
    return estimate_bot_pose(detections, mapping, resolve_camera(config, mapping), config)


def position_error_mm(pose, bot: BotInstance) -> float:
    return 1000.0 * math.hypot(pose.x_m - bot.x, pose.y_m - bot.y)


def heading_error_deg(pose, bot: BotInstance) -> float:
    return abs((pose.heading_deg - math.degrees(bot.heading_rad) + 180.0) % 360.0 - 180.0)


# ---- arena geometry ---------------------------------------------------------------------

def test_placeholder_arena_matches_the_simulator_arena_block(config) -> None:
    sim_path = REPO / "config" / "sim.yaml"
    if not sim_path.exists():
        pytest.skip("config/sim.yaml not present")
    if config.arena.get("measured"):
        pytest.skip("arena geometry has been measured; it no longer has to equal the simulator")
    sim = yaml.safe_load(sim_path.read_text(encoding="utf-8"))["arena"]
    for key in ("width_m", "height_m", "pit_radius_m"):
        assert config.arena[key] == pytest.approx(sim[key])
    np.testing.assert_allclose(config.arena["pit_centres_m"], sim["pit_centres_m"])
    np.testing.assert_allclose(config.arena["flipper_vertices_m"], sim["flipper_vertices_m"])
    assert config.arena["width_m"] == pytest.approx(1.150)
    assert config.arena["height_m"] == pytest.approx(1.150)


def test_overlay_draws_fixed_pit_and_flipper_boundaries(config) -> None:
    arena = config.arena
    arcs = pit_arcs(arena)
    assert len(arcs) == 2
    expected_ends = [{(0.0, 0.90), (0.25, 1.15)}, {(1.15, 0.25), (0.90, 0.0)}]
    for arc, centre, ends in zip(arcs, arena["pit_centres_m"], expected_ends):
        radii = np.linalg.norm(arc - np.asarray(centre), axis=1)
        np.testing.assert_allclose(radii, arena["pit_radius_m"], atol=1e-9)
        assert (arc >= -1e-9).all() and (arc <= 1.15 + 1e-9).all()     # inside the arena only
        found = {tuple(np.round(arc[0], 6)), tuple(np.round(arc[-1], 6))}
        assert found == ends
        length = float(np.sum(np.linalg.norm(np.diff(arc, axis=0), axis=1)))
        assert length == pytest.approx(math.pi * arena["pit_radius_m"] / 2.0, rel=1e-3)
    flipper = flipper_boundary(arena)
    np.testing.assert_allclose(flipper, [[0.815, 1.15], [1.15, 1.15], [1.15, 0.815],
                                         [0.815, 1.15]])
    np.testing.assert_allclose(arena_outline(arena), [[0, 0], [1.15, 0], [1.15, 1.15],
                                                      [0, 1.15], [0, 0]])
    labels = [label for label, _, _ in arena_overlay.overlay_shapes(arena)]
    assert labels == ["arena boundary", "pit boundary", "pit boundary", "fixed flipper boundary"]


def test_pit_arcs_agree_with_the_simulator_geometry_helper(config) -> None:
    helper = REPO / "autobot" / "sim" / "arena.py"
    if not helper.exists():
        pytest.skip("autobot/sim/arena.py not present")
    spec = importlib.util.spec_from_file_location("_sim_arena_readonly", helper)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module                      # its dataclass needs the module entry
    try:
        spec.loader.exec_module(module)                  # numpy only; no MuJoCo import
    finally:
        sys.modules.pop(spec.name, None)
    arcs = np.vstack(pit_arcs(config.arena))
    segments = module.pit_rim_segments(dict(config.arena), 20)
    assert len(segments) == 40
    for start, end in segments:
        for point in (start, end):
            assert np.min(np.linalg.norm(arcs - point, axis=1)) < 0.0015


# ---- image -> floor mapping ---------------------------------------------------------------

def test_mapping_fit_and_held_out_check_error(config, camera, mapping) -> None:
    assert mapping.valid, mapping.reason
    fit_names = {e.name for e in mapping.errors if e.role == "fit"}
    check_names = {e.name for e in mapping.errors if e.role == "check"}
    assert fit_names == {r.name for r in config.references if r.role == "fit"}
    assert len(fit_names) == 5 and len(check_names) == 7
    assert mapping.check_stats["max_mm"] < 0.05           # exact synthetic clicks
    # The check points are genuinely held out: dropping them leaves the fit untouched.
    clicks = reference_pixels(config, camera)
    fit_only = {name: clicks[name] for name in fit_names}
    alone = fit_floor_mapping(config.references, fit_only, camera.image_size)
    np.testing.assert_allclose(alone.H / alone.H[2, 2], mapping.H / mapping.H[2, 2], atol=1e-12)
    assert alone.check_stats["count"] == 0
    assert alone.to_report()["held_out_check_available"] is False


def test_held_out_error_is_reported_in_millimetres(config, camera, mapping) -> None:
    clicks = reference_pixels(config, camera)
    # A check mark that is really 30 mm from where the config says it is.
    clicks["tape_centre"] = tuple(camera.project([[0.575 + 0.030, 0.575, 0.0]])[0])
    # Click noise on the check points only, so the fit stays exact and the reported error can
    # be compared with the true floor position of each noisy click.
    rng = np.random.default_rng(11)
    floor_from_pixel = np.linalg.inv(camera.plane_homography(0.0))
    expected = {}
    for ref in config.references:
        if ref.role != "check":
            continue
        if ref.name != "tape_centre":
            clicks[ref.name] = tuple(np.asarray(clicks[ref.name]) + rng.normal(0, 1.5, 2))
        true_floor = apply_homography(floor_from_pixel, [clicks[ref.name]])[0]
        expected[ref.name] = 1000.0 * float(np.linalg.norm(true_floor - ref.arena_xy))
    noisy = fit_floor_mapping(config.references, clicks, camera.image_size)
    assert noisy.valid
    np.testing.assert_allclose(noisy.H / noisy.H[2, 2], mapping.H / mapping.H[2, 2], atol=1e-12)
    reported = {e.name: e.error_mm for e in noisy.errors if e.role == "check"}
    assert reported["tape_centre"] == pytest.approx(30.0, abs=0.05)
    for name, value in expected.items():
        assert reported[name] == pytest.approx(value, abs=0.02)
    assert noisy.check_stats["max_mm"] == pytest.approx(30.0, abs=0.05)
    assert 0.2 < min(reported.values()) and max(reported.values()) < 31.0


def test_noisy_fit_clicks_show_up_at_the_check_points(config, camera) -> None:
    noisy = fit_floor_mapping(config.references,
                              reference_pixels(config, camera, noise_px=1.5, seed=4),
                              camera.image_size)
    assert noisy.valid
    assert 0.5 < noisy.check_stats["max_mm"] < 15.0


def test_mapping_rejects_unusable_reference_points(config, camera) -> None:
    clicks = reference_pixels(config, camera)
    fit_names = [r.name for r in config.references if r.role == "fit"]
    too_few = fit_floor_mapping(config.references, {n: clicks[n] for n in fit_names[:3]},
                                camera.image_size)
    assert not too_few.valid and "at least 4" in too_few.reason

    line = [ReferencePoint(f"l{i}", (0.2 * i, 0.2 * i), "fit", False) for i in range(5)]
    line_clicks = {r.name: tuple(camera.project([[*r.arena_xy, 0.0]])[0]) for r in line}
    collinear = fit_floor_mapping(line, line_clicks, camera.image_size)
    assert not collinear.valid and "degenerate" in collinear.reason

    swapped = [replace(r, arena_xy=(r.arena_xy[1], r.arena_xy[0])) for r in config.references]
    mirrored = fit_floor_mapping(swapped, clicks, camera.image_size)
    assert not mirrored.valid and "mirrored" in mirrored.reason

    resized = fit_floor_mapping(config.references, clicks, (640, 360),
                                clicks_image_size=camera.image_size)
    assert not resized.valid and "click them again" in resized.reason

    wrong = dict(clicks)
    wrong["corner_origin"] = (wrong["corner_origin"][0] + 220.0, wrong["corner_origin"][1])
    bad = fit_floor_mapping(config.references, wrong, camera.image_size)
    assert not bad.valid and "fit residual" in bad.reason


def test_lens_intrinsics_are_applied_when_configured(config, camera) -> None:
    distortion = np.array([-0.18, 0.05, 0.0, 0.0, 0.0])
    rvec, _ = cv2.Rodrigues(camera.R)
    tvec = -camera.R @ camera.position
    world = np.array([[*r.arena_xy, 0.0] for r in config.references])
    distorted, _ = cv2.projectPoints(world, rvec, tvec, camera.K, distortion)
    clicks = {r.name: tuple(p) for r, p in zip(config.references, distorted.reshape(-1, 2))}
    model = PixelModel(camera.K, distortion)
    points = np.array(list(clicks.values()))
    np.testing.assert_allclose(model.distort(model.undistort(points)), points, atol=1e-3)
    with_model = fit_floor_mapping(config.references, clicks, camera.image_size, model)
    without = fit_floor_mapping(config.references, clicks, camera.image_size)
    assert with_model.valid and with_model.check_stats["max_mm"] < 0.1
    assert without.check_stats["max_mm"] > 2.0            # distortion is visible when ignored
    back = with_model.floor_to_pixel([config.reference("tape_centre").arena_xy])[0]
    np.testing.assert_allclose(back, clicks["tape_centre"], atol=0.01)


# ---- camera position and raised-marker height correction ------------------------------------

def test_camera_position_is_derived_from_the_floor_mapping(config, camera, mapping) -> None:
    derived = resolve_camera(config, mapping)
    assert derived.source == "derived"
    np.testing.assert_allclose(derived.position_m, camera.position, atol=1e-3)
    assert derived.focal_px == pytest.approx(camera.K[0, 0], abs=1.0)
    measured = resolve_camera(replace(config, camera_position_m=(0.4, -0.55, 1.3)), mapping)
    assert measured.source == "measured" and measured.position_m == (0.4, -0.55, 1.3)
    assert resolve_camera(replace(config, height_correction="off"), mapping).position_m is None
    required = resolve_camera(replace(config, height_correction="measured"), mapping)
    assert required.position_m is None and "camera.position_m" in required.detail
    assert derive_camera_from_homography(np.eye(3), camera.image_size).position_m is None


BOT_POSES = [BotInstance(0.80, 0.45, math.radians(35.0)),
             BotInstance(0.15, 0.20, math.radians(-100.0)),      # near wall, close to camera
             BotInstance(0.95, 0.85, math.radians(170.0)),       # far side
             BotInstance(0.35, 0.80, math.radians(75.0)),
             BotInstance(1.02, 0.42, math.radians(-10.0))]       # right edge


@pytest.mark.parametrize("bot", BOT_POSES)
def test_raised_marker_parallax_uncorrected_versus_corrected(config, camera, mapping, bot) -> None:
    scene = Scene([bot])
    height = config.markers[0].height_m
    assert height > 0.0

    raw = pose_for(replace(config, height_correction="off"), camera, mapping, scene)
    assert raw.ok and raw.orientation == "upright"
    assert raw.position_quality == "approximate" and "APPROXIMATE" in raw.position_label
    assert raw.correction_source == "none" and raw.parallax_shift_mm is None
    horizontal = math.hypot(bot.x - camera.position[0], bot.y - camera.position[1])
    expected = 1000.0 * height * horizontal / (camera.position[2] - height)
    raw_error = position_error_mm(raw, bot)
    assert raw_error == pytest.approx(expected, abs=2.5)
    assert raw_error > 8.0                                # far larger than the corrected error
    away = np.array([bot.x, bot.y]) - camera.position[:2]
    shift = np.array([raw.x_m - bot.x, raw.y_m - bot.y])
    assert float(np.dot(shift, away)) > 0.0               # pushed away from the camera
    assert heading_error_deg(raw, bot) < 1.5              # heading survives the parallax

    derived = pose_for(config, camera, mapping, scene)
    assert derived.ok and derived.position_quality == "height-corrected"
    assert derived.correction_source == "derived" and "DERIVED" in derived.position_label
    assert position_error_mm(derived, bot) < 2.0
    assert heading_error_deg(derived, bot) < 1.5
    assert derived.parallax_shift_mm == pytest.approx(expected, abs=2.5)
    assert derived.floor_projection_xy == pytest.approx((raw.x_m, raw.y_m), abs=1e-9)

    measured_config = replace(config, camera_position_m=tuple(camera.position))
    measured = pose_for(measured_config, camera, mapping, scene)
    assert measured.correction_source == "measured"
    assert position_error_mm(measured, bot) < 2.0


def test_wrong_marker_height_or_camera_position_leaves_a_measurable_error(config, camera,
                                                                         mapping) -> None:
    bot = BotInstance(0.95, 0.85, math.radians(20.0))
    tall = parse_config({**_raw_config(), "bot": {**_raw_config()["bot"],
                                                 "marker_height_m": {"top": 0.060,
                                                                     "bottom": 0.060}}})
    horizontal = math.hypot(bot.x - camera.position[0], bot.y - camera.position[1])
    # Markers really at 60 mm, config says 60 mm: corrected.
    assert position_error_mm(pose_for(tall, camera, mapping, Scene([bot])), bot) < 2.0
    # Markers really at 60 mm, config still says 34 mm: the missing 26 mm shows as parallax.
    stale = pose_for(config, camera, mapping, Scene([bot]), render_config=tall)
    assert stale.ok
    leftover = 1000.0 * 0.026 * horizontal / (camera.position[2] - 0.060)
    assert position_error_mm(stale, bot) == pytest.approx(leftover, abs=3.0)
    # A camera position measured 10 cm wrong sideways only costs a few millimetres.
    off = replace(config, camera_position_m=(camera.position[0] + 0.10, camera.position[1],
                                             camera.position[2]))
    assert position_error_mm(pose_for(off, camera, mapping, Scene([bot])), bot) < 5.0


def test_derived_camera_tolerates_an_off_centre_principal_point(config) -> None:
    skewed = make_camera((0.40, -0.55, 1.30), (0.60, 0.42, 0.0), 930.0, principal=(665.0, 345.0))
    mapping = fit_floor_mapping(config.references, reference_pixels(config, skewed),
                                skewed.image_size)
    derived = resolve_camera(config, mapping)
    assert derived.source == "derived"
    assert np.linalg.norm(np.asarray(derived.position_m) - skewed.position) < 0.15
    bot = BotInstance(0.95, 0.85, math.radians(20.0))
    assert position_error_mm(pose_for(config, skewed, mapping, Scene([bot])), bot) < 5.0


def test_inverted_bot_is_read_from_the_bottom_markers(config, camera, mapping) -> None:
    bot = BotInstance(0.45, 0.50, math.radians(-60.0), orientation="inverted")
    pose = pose_for(config, camera, mapping, Scene([bot]))
    assert pose.ok and pose.orientation == "inverted"
    assert set(pose.used_ids) == {4, 5, 6, 7}
    assert position_error_mm(pose, bot) < 2.0 and heading_error_deg(pose, bot) < 1.5
    upright = pose_for(config, camera, mapping, Scene([replace(bot, orientation="upright")]))
    assert upright.orientation == "upright" and set(upright.used_ids) == {0, 1, 2, 3}


def test_partly_hidden_markers_still_give_a_pose(config, camera, mapping) -> None:
    bot = BotInstance(0.60, 0.70, math.radians(140.0), ids=[1, 3])
    pose = pose_for(config, camera, mapping, Scene([bot]))
    assert pose.ok and pose.used_ids == [1, 3]
    assert position_error_mm(pose, bot) < 3.0 and heading_error_deg(pose, bot) < 2.0
    single = pose_for(config, camera, mapping, Scene([replace(bot, ids=[2])]))
    assert single.ok and single.used_ids == [2]
    assert position_error_mm(single, bot) < 6.0


# ---- unknown: missing, conflicting, stale ---------------------------------------------------

def test_unknown_when_markers_are_missing_or_conflicting(config, camera, mapping) -> None:
    bot = BotInstance(0.70, 0.60, math.radians(100.0))

    empty = pose_for(config, camera, mapping, Scene())
    assert not empty.ok and "no configured bot markers" in empty.reason
    report = empty.to_report()
    assert (report["x_m"], report["y_m"], report["heading_deg"]) == ("unknown",) * 3
    assert report["orientation"] == "unknown" and report["position_quality"] == "unknown"

    both = pose_for(config, camera, mapping, Scene(
        [bot], [MarkerPlacement(5, (0.30, 0.30), 0.3, 0.040, 0.034)]))
    assert not both.ok and "top markers" in both.reason and "bottom markers" in both.reason
    assert both.orientation == "unknown" and both.x_m is None

    apart = pose_for(config, camera, mapping, Scene(
        [replace(bot, ids=[0, 3])], [MarkerPlacement(1, (0.35, 0.80), 1.0, 0.040, 0.034)]))
    assert not apart.ok and "disagree on the bot centre" in apart.reason
    assert apart.orientation == "unknown"

    twisted_spec = replace(config.markers[1], yaw_deg=25.0)
    twisted = replace(config, markers={**config.markers, 1: twisted_spec},
                      max_position_disagreement_m=1.0)
    turned = pose_for(twisted, camera, mapping, Scene([bot]), render_config=config)
    assert not turned.ok and "disagree on heading" in turned.reason

    twice = pose_for(config, camera, mapping, Scene([bot, BotInstance(0.30, 0.35, 0.4)]))
    assert not twice.ok and "more than once" in twice.reason

    oversize = replace(config, markers={k: replace(v, size_m=0.080)
                                        for k, v in config.markers.items()})
    wrong_size = pose_for(oversize, camera, mapping, Scene([bot]), render_config=config)
    assert not wrong_size.ok and "configured as 80 mm" in wrong_size.reason

    stray = pose_for(config, camera, mapping, Scene(
        [bot], [MarkerPlacement(20, (0.30, 0.30), 0.3, 0.040, 0.034)]))
    assert stray.ok and stray.ignored_ids == [20]          # unconfigured IDs never join the pose


def test_without_a_mapping_only_the_orientation_is_known(config, camera) -> None:
    unmapped = fit_floor_mapping(config.references, {}, camera.image_size)
    pose = pose_for(config, camera, unmapped, Scene([BotInstance(0.6, 0.6, 0.0)]))
    assert not pose.ok and "no valid floor mapping" in pose.reason
    assert pose.orientation == "upright" and pose.x_m is None


def test_detection_age_and_stale_pose(config, camera, mapping) -> None:
    good = pose_for(config, camera, mapping, Scene([BotInstance(0.6, 0.6, 0.0)]))
    missing = pose_for(config, camera, mapping, Scene())
    tracker = PoseTracker(max_age_s=0.25)
    never = tracker.status(5.0)
    assert not never.pose.ok and never.detection_age_s is None
    tracker.update(good, 10.0)
    fresh = tracker.status(10.1)
    assert fresh.pose.ok and fresh.detection_age_s == pytest.approx(0.1)
    stale = tracker.status(10.4)                           # no new frame arrived
    assert not stale.pose.ok and "stale" in stale.pose.reason
    assert stale.detection_age_s == pytest.approx(0.4)
    assert stale.to_report()["x_m"] == "unknown"
    assert "NOT the current pose" in stale.to_report()["last_valid_pose"]["note"]
    tracker.update(missing, 10.5)                          # a frame with no markers
    gone = tracker.status(10.5)
    assert not gone.pose.ok and "no configured bot markers" in gone.pose.reason
    assert gone.detection_age_s == pytest.approx(0.5)      # age of the last VALID detection
    tracker.update(good, 11.0)
    assert tracker.status(11.0).pose.ok and tracker.status(11.0).detection_age_s == 0.0


# ---- reference-point clicking and saving ----------------------------------------------------

def test_click_session_places_saves_and_reloads_points(config, camera, mapping, tmp_path) -> None:
    exact = reference_pixels(config, camera)
    clicks = ClickSet(image_size=camera.image_size)
    session = ClickSession(config.references, clicks)
    assert session.current.name == "corner_origin"
    assert session.click(*exact["corner_origin"]) == "corner_origin"
    session.click(1.0, 2.0)                                # a mis-click on the second point
    assert session.undo() == "pitA_rim_x0_wall"
    assert session.current.name == "pitA_rim_x0_wall"
    session.skip()                                         # e.g. hidden behind the near wall
    assert session.current.name == "pitB_rim_y0_wall"
    while session.current is not None and len(clicks.points) < len(config.references) - 1:
        session.click(*exact[session.current.name])
    assert "pitA_rim_x0_wall" not in clicks.points and len(clicks.points) == 11
    # The skipped fit point is replaced by promoting a check point to the fit set.
    session.index = [r.name for r in config.references].index("pitA_rim_ymax_wall")
    assert session.toggle_role() == "fit"
    path = tmp_path / "clicks.json"
    save_clicks(path, clicks, source="unit test")
    loaded = load_clicks(path)
    assert loaded.image_size == camera.image_size and not loaded.synthetic
    assert loaded.roles == {"pitA_rim_ymax_wall": "fit"}
    refit = fit_floor_mapping(config.references, loaded.points, camera.image_size,
                              roles=loaded.roles, clicks_image_size=loaded.image_size)
    assert refit.valid and refit.fit_stats["count"] == 5 and refit.check_stats["count"] == 6
    assert refit.check_stats["max_mm"] < 0.1               # saved to 0.01 px
    assert session.toggle_role() == "check" and clicks.roles == {}

    app = arena_overlay.OverlayApp(config, ClickSet(), "unit test")
    frame = render_frame(config, camera)
    app.process(frame, 0.0, arena_overlay.ImageSource.__new__(arena_overlay.ImageSource))
    assert not app.mapping.valid
    app.click_mode = True
    assert "CLICK corner_origin [FIT]" in app.prompt()
    for ref in config.references:
        assert app.place_click(*exact[ref.name]) == ref.name
    app.process(frame, 0.0, arena_overlay.ImageSource.__new__(arena_overlay.ImageSource))
    assert app.mapping.valid and app.clicks.image_size == camera.image_size
    assert app.mapping.check_stats["count"] == 7 and "All reference points" in app.prompt()


# ---- the command-line tool: saved frame, saved video, camera -------------------------------

def _raw_config() -> dict:
    return yaml.safe_load((REPO / "config" / "arena_overlay.yaml").read_text(encoding="utf-8"))


def _run(tmp_path, *args):
    report_path = tmp_path / "report.json"
    code = arena_overlay.main([*args, "--no-window", "--report", str(report_path)])
    assert code == 0
    return json.loads(report_path.read_text(encoding="utf-8"))


def test_cli_saved_frame_headless(tmp_path, camera, capsys) -> None:
    fixture = write_fixture(tmp_path / "fixture")
    paths, truth = fixture["paths"], fixture["truth"]["frame"]["bot"]
    overlay_path = tmp_path / "overlay.png"
    report = _run(tmp_path, "--image", paths["frame"], "--points", paths["clicks"],
                  "--save-overlay", str(overlay_path), "--compact-hud",
                  "--truth", f"{truth['x_m']},{truth['y_m']},{truth['heading_deg']}")
    assert report["read_only"] is True and report["motor_commands_sent"] is False
    assert report["synthetic_fixture_input"] is True and "SYNTHETIC" in report["synthetic_note"]
    assert report["placeholder_values_in_use"]            # shipped config is all placeholders
    assert report["mapping"]["valid"] and report["mapping"]["check_points"]["count"] == 7
    assert report["mapping"]["check_points"]["max_mm"] < 0.2
    assert report["mapping"]["held_out_check_available"] is True
    pose = report["bot_pose"]
    assert pose["state"] == "ok" and pose["orientation"] == "upright"
    assert pose["position_quality"] == "height-corrected"
    assert pose["correction_source"] == "derived" and report["camera"]["source"] == "derived"
    assert pose["detection_age_s"] == 0.0
    assert report["truth_check"]["reported_error_mm"] < 2.0
    assert report["truth_check"]["floor_projection_error_mm"] > 15.0
    assert abs(report["truth_check"]["heading_error_deg"]) < 1.5
    assert len(report["how_to_measure_position_error"]) >= 4
    printed = capsys.readouterr().out
    assert "SYNTHETIC" in printed and "PLACEHOLDER still in use" in printed

    source = cv2.imread(paths["frame"])
    overlay = cv2.imread(str(overlay_path))
    assert overlay.shape == source.shape
    # The drawn boundary, pit rim and fixed flipper edge sit on the rendered features.
    floor_points = {"boundary": ((1.15, 0.55), arena_overlay.BOUNDARY),
                    "pit": ((0.25 * math.cos(math.radians(-45)),
                             1.15 + 0.25 * math.sin(math.radians(-45))), arena_overlay.PIT),
                    "flipper": ((0.9825, 0.9825), arena_overlay.FLIPPER)}
    for name, ((x, y), colour) in floor_points.items():
        u, v = np.round(camera.project([[x, y, 0.0]])[0]).astype(int)
        patch = overlay[v - 2:v + 3, u - 2:u + 3].reshape(-1, 3).astype(int)
        closest = np.min(np.abs(patch - np.asarray(colour)).sum(axis=1))
        assert closest < 60, f"{name} overlay line is not where the rendered {name} is"


def test_cli_saved_video_headless(tmp_path) -> None:
    fixture = write_fixture(tmp_path / "fixture")
    paths, truth = fixture["paths"], fixture["truth"]["video_frames"]
    report = _run(tmp_path, "--video", paths["video"], "--points", paths["clicks"])
    assert report["source"]["kind"] == "video"
    assert report["frames_processed"] == len(truth) == len(demo_scenes())
    seen_reasons = set()
    for frame, expected in zip(report["frames"], truth):
        assert frame["state"] == expected["expect"], (frame, expected)
        if expected["expect"] == "ok":
            bot = expected["bot"]
            assert math.hypot(frame["x_m"] - bot["x_m"], frame["y_m"] - bot["y_m"]) < 0.004
            heading = (frame["heading_deg"] - bot["heading_deg"] + 180.0) % 360.0 - 180.0
            assert abs(heading) < 2.5
            assert frame["orientation"] == bot["orientation"]
            assert frame["position_quality"] == "height-corrected"
            assert frame["detection_age_s"] == 0.0
        else:
            assert frame["x_m"] == "unknown" and frame["heading_deg"] == "unknown"
            seen_reasons.add(frame["reason"].split(":")[0].split(" visible")[0])
    assert seen_reasons == {"no configured bot markers", "conflict"}
    timestamps = [frame["t_s"] for frame in report["frames"]]
    assert timestamps == pytest.approx([i / 10.0 for i in range(len(truth))])
    orientations = {frame["orientation"] for frame in report["frames"]}
    assert orientations == {"upright", "inverted", "unknown"}
    limited = _run(tmp_path, "--video", paths["video"], "--points", paths["clicks"],
                   "--max-frames", "3")
    assert limited["frames_processed"] == 3


def test_cli_labels_position_approximate_without_height_correction(tmp_path) -> None:
    fixture = write_fixture(tmp_path / "fixture")
    raw = _raw_config()
    raw["pose"]["height_correction"] = "off"
    config_path = tmp_path / "overlay_off.yaml"
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    report = _run(tmp_path, "--image", fixture["paths"]["frame"], "--points",
                  fixture["paths"]["clicks"], "--config", str(config_path),
                  "--truth", "0.80,0.45")
    pose = report["bot_pose"]
    assert pose["state"] == "ok" and pose["position_quality"] == "approximate"
    assert "APPROXIMATE" in pose["position_label"] and pose["correction_source"] == "none"
    assert report["camera"]["position_m"] is None
    assert report["truth_check"]["reported_error_mm"] > 15.0      # the uncorrected parallax

    app = arena_overlay.OverlayApp(load_config(config_path), load_clicks(
        fixture["paths"]["clicks"]), "unit test")
    source = arena_overlay.ImageSource(fixture["paths"]["frame"])
    app.process(source.read()[1], 0.0, source)
    for compact in (False, True):
        text = "\n".join(line for line, _ in app.status_lines(compact=compact))
        assert "APPROXIMATE" in text and "NOT height-corrected" in text
    full = "\n".join(line for line, _ in app.status_lines())
    assert "SYNTHETIC" in full and "PLACEHOLDER" in full and "held-out check 7 pts" in full


def test_cli_without_reference_points_says_unknown(tmp_path) -> None:
    fixture = write_fixture(tmp_path / "fixture")
    report = _run(tmp_path, "--image", fixture["paths"]["frame"], "--points",
                  str(tmp_path / "missing.json"))
    assert report["source"]["points_file_found"] is False
    assert report["mapping"]["valid"] is False
    assert report["bot_pose"]["state"] == "unknown" and report["bot_pose"]["x_m"] == "unknown"
    assert report["bot_pose"]["orientation"] == "upright"
    blank = _run(tmp_path, "--image", _blank_frame(tmp_path), "--points",
                 fixture["paths"]["clicks"])
    assert blank["bot_pose"]["state"] == "unknown"
    assert blank["bot_pose"]["detection_age_s"] is None
    assert arena_overlay.main(["--image", str(tmp_path / "nope.png"), "--no-window"]) == 2


def _blank_frame(tmp_path) -> str:
    path = tmp_path / "blank.png"
    cv2.imwrite(str(path), np.full((720, 1280, 3), 90, np.uint8))
    return str(path)


def test_cli_camera_path_with_a_fake_capture_device(tmp_path, monkeypatch, config, camera) -> None:
    """The live path is exercised with a stand-in capture object; no real camera is opened."""
    frame = render_frame(config, camera, Scene([BotInstance(0.5, 0.6, 1.0)]))
    opened: list["FakeCapture"] = []

    class FakeCapture:
        def __init__(self, index, backend=None):
            self.index, self.backend, self.props, self.released = index, backend, [], False
            self.ok = index == 0
            opened.append(self)

        def isOpened(self):
            return self.ok

        def set(self, prop, value):
            self.props.append(prop)
            return True

        def read(self):
            return True, frame.copy()

        def release(self):
            self.released = True

    monkeypatch.setattr(arena_overlay.cv2, "VideoCapture", FakeCapture)
    clicks_path = tmp_path / "clicks.json"
    save_clicks(clicks_path, fixture_clicks(config, camera))
    report = _run(tmp_path, "--camera", "0", "--points", str(clicks_path), "--max-frames", "3",
                  "--backend", "dshow", "--width", "1280", "--height", "720")
    assert report["source"]["kind"] == "camera" and report["frames_processed"] == 3
    assert report["bot_pose"]["state"] == "ok"
    assert 0.0 <= report["bot_pose"]["detection_age_s"] < 0.25
    capture = opened[0]
    assert capture.backend == cv2.CAP_DSHOW and capture.released
    # FOURCC is requested last, as in control_center/backend/camera_worker.py.
    assert capture.props == [cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT,
                             cv2.CAP_PROP_FPS, cv2.CAP_PROP_FOURCC]
    assert arena_overlay.main(["--camera", "5", "--no-window"]) == 2     # no such camera


# ---- configuration and read-only guard ------------------------------------------------------

def test_config_is_validated_and_placeholders_are_listed(config) -> None:
    assert sorted(config.markers) == list(range(8))
    assert {config.markers[i].face for i in range(4)} == {"top"}
    assert {config.markers[i].face for i in range(4, 8)} == {"bottom"}
    assert len(config.placeholders) == 4
    raw = _raw_config()
    raw["arena"]["measured"] = True
    raw["bot"]["measured"] = True
    raw["camera"]["position_m"] = [0.4, -0.55, 1.3]
    for point in raw["reference_points"]:
        point["measured"] = True
    assert parse_config(raw).placeholders == []
    for mutate, message in [
            (lambda d: d["bot"]["markers"].append(dict(d["bot"]["markers"][0])), "twice"),
            (lambda d: d["bot"].update(dictionary="DICT_NOPE"), "Unknown marker dictionary"),
            (lambda d: d["bot"]["markers"][0].update(face="side"), "top or bottom"),
            (lambda d: d["pose"].update(height_correction="maybe"), "height_correction")]:
        broken = _raw_config()
        mutate(broken)
        with pytest.raises(ValueError, match=message):
            parse_config(broken)
    apriltag = _raw_config()
    apriltag["bot"]["dictionary"] = "DICT_APRILTAG_36h11"
    assert MarkerDetector(parse_config(apriltag).dictionary).dictionary == "DICT_APRILTAG_36h11"


def test_tool_modules_never_import_the_robot_link_or_serial() -> None:
    forbidden = ("robot_link", "command_manager", "serial", "socket", "telemetry")
    for name in ("arena_overlay.py", "overlay_core.py", "overlay_fixture.py", "blob_tracker.py", "footage_reference.py"):
        tree = ast.parse((REPO / "control_center" / "tools" / name).read_text(encoding="utf-8"))
        imported: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported += [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                imported += [node.module or ""] + [alias.name for alias in node.names]
        assert not [item for item in imported if any(word in item for word in forbidden)], name


def test_fixture_frames_are_labelled_synthetic(config, camera) -> None:
    assert "SYNTHETIC" in overlay_fixture.FIXTURE_LABEL
    labelled = render_frame(config, camera)
    plain = render_frame(config, camera, label=False)
    band = labelled[-26:, :400].astype(int) - plain[-26:, :400].astype(int)
    assert np.abs(band).max() > 100                        # the label is burnt into the pixels
    assert fixture_clicks(config, camera).synthetic
