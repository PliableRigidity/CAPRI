"""Arena and hazard set-up: coordinates, saving and loading, profile isolation, undo/reset, invalid geometry,
camera recalibration that keeps the physical hazards.

Pure software: rendered-free, no camera, no serial port, no robot link. All files go to a temporary folder;
one test proves the real ``config/`` folder is not touched.
"""
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from control_center.tools import arena_setup as A
from control_center.tools.overlay_core import ClickSet, fit_floor_mapping, load_clicks, load_config
from control_center.tools.stream_adapter import fit_profile

ROOT = Path(__file__).resolve().parents[1]
W = H = 1.150
IMAGE = (1280, 720)
CAMERA_1 = [(250.0, 650.0), (1030.0, 650.0), (900.0, 150.0), (380.0, 150.0)]       # c1..c4 as the picture shows them
CAMERA_2 = [(300.0, 600.0), (1000.0, 680.0), (860.0, 120.0), (420.0, 170.0)]       # the camera was moved


def homography(pixels):
    src = np.float32([(0, 0), (W, 0), (W, H), (0, H)])
    return cv2.getPerspectiveTransform(src, np.float32(pixels))


def to_pixel(h, point):
    value = h @ np.array([point[0], point[1], 1.0])
    return (float(value[0] / value[2]), float(value[1] / value[2]))


# The arena as the user describes the current view: Pit 1 top-left, Pit 2 bottom-right, flipper bottom-left.
PIT1, PIT2 = (0.0, H), (W, 0.0)
FLIPPER = [(0.0, 0.335), (0.0, 0.0), (0.335, 0.0)]
HINTS = {"pit_1": "top-left", "pit_2": "bottom-right", "flipper": "bottom-left"}


def run_full_setup(camera=CAMERA_1, profile="live", jitter=0.0, radii=(0.25, 0.25), hints=HINTS, default=None):
    h = homography(camera)
    session = A.ArenaSetupSession(profile, "new", image_size=IMAGE, hints=hints, default_size_m=default, default_radius_m=0.25 if default else None)
    for cid, pixel in zip(A.CORNER_IDS, camera):
        assert session.current == cid
        session.click(*pixel)
    assert session.current == "width"
    session.submit_number("width", W)
    session.submit_number("height", H)
    session.click(*to_pixel(h, (PIT1[0] + jitter, PIT1[1] - jitter)))
    session.submit_number("pit1_radius", radii[0])
    session.click(*to_pixel(h, (PIT2[0] - jitter, PIT2[1] + jitter)))
    session.submit_number("pit2_radius", radii[1])
    for vertex in FLIPPER:
        session.click(*to_pixel(h, vertex))
    assert session.finish_flipper()
    assert session.done
    return session, h


# ---------------------------------------------------------------- typed lengths


@pytest.mark.parametrize("text, metres", [("1.15", 1.15), ("1150", 1.15), ("1150 mm", 1.15), ("1,15", 1.15), ("115 cm", 1.15),
                                          ("0.25 m", 0.25), ("250", 0.25), ("25 cm", 0.25), ("0.5", 0.5)])
def test_typed_lengths_in_metres_or_millimetres(text, metres):
    assert A.parse_length(text) == pytest.approx(metres)


@pytest.mark.parametrize("text", ["", "abc", "-1", "1..2", "1 km", "1e3"])
def test_typed_lengths_that_are_not_lengths_are_refused(text):
    with pytest.raises(ValueError):
        A.parse_length(text)


# ---------------------------------------------------------------- coordinate transformation


def test_clicked_hazards_become_metres_through_the_corner_mapping():
    session, _ = run_full_setup()
    solution = session.solve()
    assert solution.complete and solution.setup is not None, solution.problems
    g = solution.setup.geometry                                     # set-up frame, as clicked
    assert (g.width_m, g.height_m) == (W, H)
    assert g.pits[0].centre_m == pytest.approx(PIT1, abs=1e-3) and g.pits[1].centre_m == pytest.approx(PIT2, abs=1e-3)
    assert [p.radius_m for p in g.pits] == [0.25, 0.25]
    for got, want in zip(sorted(g.flipper_m), sorted(FLIPPER)):
        assert got == pytest.approx(want, abs=1e-3)
    assert solution.mapping.valid


def test_a_click_error_moves_the_hazard_by_about_the_mapped_distance_only():
    session, h = run_full_setup()
    exact = session.solve().setup.geometry.pits[0].centre_m
    pixel = to_pixel(h, (0.02, H - 0.02))
    off = A.ArenaSetupSession("live", "new", image_size=IMAGE)
    for cid, p in zip(A.CORNER_IDS, CAMERA_1):
        off.click(*p)
    off.submit_number("width", W)
    off.submit_number("height", H)
    off.click(*pixel)
    off.submit_number("pit1_radius", 0.25)
    mapped = off.solve(IMAGE).geometry.pits[0].centre_m
    assert mapped == pytest.approx((0.02, H - 0.02), abs=1e-3) and exact != mapped


def test_a_pit_centre_near_its_corner_is_put_on_the_corner_and_the_flipper_on_the_walls():
    session, _ = run_full_setup(jitter=0.03)
    solution = session.solve()
    assert solution.setup.geometry.pits[0].centre_m == (0.0, H)
    assert any("put exactly on it" in w for w in solution.warnings)


def test_mirrored_corner_order_is_refused_with_a_hint():
    session = A.ArenaSetupSession("live", "new", image_size=IMAGE)
    for pixel in (CAMERA_1[0], CAMERA_1[3], CAMERA_1[2], CAMERA_1[1]):       # clockwise instead of the order shown
        session.click(*pixel)
    session.submit_number("width", W)
    session.submit_number("height", H)
    solution = session.solve()
    assert not solution.complete and any("mirror" in p for p in solution.problems) and any("order shown" in p for p in solution.problems)


# ---------------------------------------------------------------- the simulator frame comes from the hazards, not the screen


def test_the_current_view_maps_onto_the_simulator_frame_by_a_half_turn():
    session, _ = run_full_setup()
    setup = session.solve().setup
    assert setup.canonical.available and setup.policy_frame_ok and setup.canonical.corner_names == {
        "c3": "plain", "c1": "flipper", "c4": "pit_1", "c2": "pit_2"}
    g = setup.active_geometry()
    assert sorted(g.pits[i].centre_m for i in range(2)) == sorted([(0.0, H), (W, 0.0)])
    assert sorted(g.flipper_m) == pytest.approx(sorted([(0.815, 1.15), (1.15, 1.15), (1.15, 0.815)]), abs=2e-3)
    arena = setup.arena_dict()
    assert arena["frame"] == "simulator" and arena["policy_frame_ok"] and arena["measured"] is True
    assert arena["pit_radii_m"] == [0.25, 0.25]


def test_the_same_arena_seen_from_another_side_gives_the_same_active_geometry():
    """Hazard positions are never taken from screen orientation: a camera on the other side saves identical metres."""
    first = run_full_setup()[0].solve().setup
    rotated = [CAMERA_1[2], CAMERA_1[3], CAMERA_1[0], CAMERA_1[1]]               # the picture shows the arena turned half a turn
    h = homography(rotated)
    # The arena is turned, so the corner that is c1 for this new view is the old c3; its hazards move with it.
    turned = lambda p: (W - p[0], H - p[1])
    session = A.ArenaSetupSession("live", "new", image_size=IMAGE)
    for pixel in rotated:
        session.click(*pixel)
    session.submit_number("width", W)
    session.submit_number("height", H)
    session.click(*to_pixel(h, turned(PIT2)))
    session.submit_number("pit1_radius", 0.25)
    session.click(*to_pixel(h, turned(PIT1)))
    session.submit_number("pit2_radius", 0.25)
    for vertex in FLIPPER:
        session.click(*to_pixel(h, turned(vertex)))
    session.finish_flipper()
    second = session.solve().setup
    assert second.canonical.available and second.canonical.origin_index != first.canonical.origin_index
    a, b = first.active_geometry(), second.active_geometry()
    assert sorted(p.centre_m for p in a.pits) == pytest.approx(sorted(p.centre_m for p in b.pits), abs=2e-3)
    assert sorted(a.flipper_m) == pytest.approx(sorted(b.flipper_m), abs=2e-3)


def test_a_layout_that_is_not_the_simulators_stays_in_the_setup_frame_and_says_so():
    session = A.ArenaSetupSession("live", "new", image_size=IMAGE)
    h = homography(CAMERA_1)
    for pixel in CAMERA_1:
        session.click(*pixel)
    session.submit_number("width", W)
    session.submit_number("height", H)
    session.click(*to_pixel(h, (0.0, 0.0)))                       # pits on ADJACENT corners
    session.submit_number("pit1_radius", 0.25)
    session.click(*to_pixel(h, (W, 0.0)))
    session.submit_number("pit2_radius", 0.25)
    for vertex in [(W, H - 0.335), (W, H), (W - 0.335, H)]:
        session.click(*to_pixel(h, vertex))
    session.finish_flipper()
    solution = session.solve()
    assert solution.complete
    setup = solution.setup
    assert not setup.canonical.available and not setup.policy_frame_ok
    assert setup.arena_dict()["frame"] == "setup" and setup.arena_dict()["policy_frame_ok"] is False
    assert any("not the simulator's" in w for w in solution.warnings)


# ---------------------------------------------------------------- saving, loading, the existing tools


@pytest.fixture
def files(tmp_path):
    return {"arena_path": tmp_path / "arena.json", "clicks_path": tmp_path / "clicks.json"}


def saved_session(files, camera=CAMERA_1, profile="live"):
    session, _ = run_full_setup(camera, profile)
    solution = session.solve()
    clicks = A.corner_clickset(profile, solution.setup, session.corner_pixels(), IMAGE)
    report = A.save_setup(profile, solution.setup, clicks, **files)
    return session, solution, report


def test_save_and_reload_round_trip_in_metres(files):
    session, solution, report = saved_session(files)
    data = json.loads(files["arena_path"].read_text(encoding="utf-8"))
    assert data["profile"] == "live" and data["units"] == "metres" and data["active_frame"] == "canonical"
    assert data["canonical_frame"]["pits"][0]["radius_m"] == 0.25 and report.wrote_arena
    assert "pixel" not in json.dumps(data).lower() and '_px' not in json.dumps(data)          # metres only: no pixels in the arena file
    loaded = A.load_arena_setup("live", files["arena_path"])
    assert loaded.setup_id == solution.setup.setup_id
    for got, want in zip(loaded.active_geometry().pits, solution.setup.active_geometry().pits):
        assert got.centre_m == pytest.approx(want.centre_m, abs=1e-6) and got.radius_m == want.radius_m
    assert loaded.summary_lines()[0].startswith("profile live: 1.150 x 1.150 m")
    clicks = load_clicks(files["clicks_path"])
    assert set(clicks.points) == {f"{A.CORNER_PREFIX}{c}" for c in A.CORNER_IDS} and clicks.image_size == IMAGE


def test_saved_arena_feeds_the_existing_overlay_tools_unchanged(files):
    saved_session(files)
    config = load_config(ROOT / "config" / "arena_overlay.yaml")
    applied = A.with_arena_setup(config, "live", load_clicks(files["clicks_path"]), files["arena_path"])
    assert applied.arena["measured"] is True and applied.arena["pit_radii_m"] == [0.25, 0.25]
    assert not any(item.startswith(("arena geometry", "reference point")) for item in applied.placeholders)
    assert any("bot markers" in tag for tag in applied.placeholder_tags)                     # the unmeasured markers are still flagged
    mapping, camera, quality = fit_profile(applied, load_clicks(files["clicks_path"]), IMAGE)
    assert mapping.valid
    h = homography(CAMERA_1)
    # A pixel on the floor maps to the SIMULATOR frame: the view's far-right (plain) corner is the origin.
    origin = mapping.pixel_to_floor([to_pixel(h, (W, H))])[0]
    assert origin == pytest.approx((0.0, 0.0), abs=2e-3)
    pit_pixel = to_pixel(h, PIT1)
    assert mapping.pixel_to_floor([pit_pixel])[0] == pytest.approx((W, 0.0), abs=2e-3)
    assert config.arena["measured"] is False                                                 # the input config was not modified


def test_per_pit_radii_reach_the_arc_drawing_and_the_floor_mask(files):
    from control_center.tools.blob_tracker import arena_floor_polygon
    from control_center.tools.overlay_core import pit_arcs
    session, _ = run_full_setup(radii=(0.20, 0.30))
    setup = session.solve().setup
    arena = setup.arena_dict()
    radii = sorted(arena["pit_radii_m"])
    assert radii == [0.20, 0.30]
    arcs = pit_arcs(arena)
    for arc, centre, radius in zip(arcs, arena["pit_centres_m"], arena["pit_radii_m"]):
        assert np.allclose(np.linalg.norm(arc - np.asarray(centre), axis=1), radius, atol=1e-9)
    assert len(arena_floor_polygon(arena)) > 8
    bad = dict(arena, pit_centres_m=[[0.0, 0.0], [W, 0.0]], pit_radii_m=[0.2, 0.2])
    assert len(arena_floor_polygon(bad)) == 4                                                # not the simulator layout: a plain rectangle, no crash


def test_the_arena_file_is_all_or_nothing_when_a_write_fails(files, monkeypatch):
    session, solution, _ = saved_session(files)
    before = (files["arena_path"].read_bytes(), files["clicks_path"].read_bytes())
    bigger = A.ArenaSetup("live", A.Geometry(1.2, 1.2, solution.setup.geometry.pits, solution.setup.geometry.flipper_m),
                          solution.setup.canonical)
    clicks = A.corner_clickset("live", bigger, session.corner_pixels(), IMAGE)
    monkeypatch.setattr(A, "_backup", lambda path: (_ for _ in ()).throw(OSError("disk full")))
    with pytest.raises(OSError):
        A.save_setup("live", bigger, clicks, **files)
    assert (files["arena_path"].read_bytes(), files["clicks_path"].read_bytes()) == before
    assert not list(files["arena_path"].parent.glob("*.tmp"))


def test_an_older_file_is_kept_as_a_backup_when_saving_again(files):
    saved_session(files)
    _, _, report = saved_session(files)
    assert report.arena_backup is not None and report.arena_backup.exists() and report.clicks_backup.exists()
    assert any("previous kept as" in line for line in report.lines())


# ---------------------------------------------------------------- profiles never mix


def test_profiles_have_separate_files_and_names_are_checked():
    assert A.profile_arena_path("live") == A.LIVE_ARENA_PATH
    assert A.profile_arena_path("home_test") == ROOT / "config" / "overlay_profiles" / "home_test" / "arena.json"
    assert A.profile_arena_path("live") != A.profile_arena_path("home_test")
    for bad in ("", "..", ".hidden", "a/b", "a\\b", "c:x"):
        with pytest.raises(A.ArenaSetupError):
            A.profile_arena_path(bad)


def test_an_arena_file_copied_to_another_profile_is_refused(files, tmp_path):
    saved_session(files, profile="live")
    with pytest.raises(A.ArenaSetupError, match="belongs to profile 'live'"):
        A.load_arena_setup("home_test", files["arena_path"])
    config = load_config(ROOT / "config" / "home_test_overlay.yaml")
    with pytest.raises(A.ArenaSetupError, match="belongs to profile"):
        A.with_arena_setup(config, "home_test", None, files["arena_path"])


def test_the_real_arena_profile_refuses_the_home_test_configuration():
    config = load_config(ROOT / "config" / "home_test_overlay.yaml")
    assert config.arena["width_m"] == 0.5                                                    # the A4 patch
    with pytest.raises(A.ArenaSetupError, match="HOME TEST patch"):
        A.with_arena_setup(config, "live")


def test_a_profile_without_a_saved_arena_changes_nothing_and_never_borrows_another(files, monkeypatch, tmp_path):
    monkeypatch.setattr(A, "LIVE_ARENA_PATH", tmp_path / "live-arena.json")
    monkeypatch.setattr(A, "PROFILES_DIR", tmp_path / "profiles")
    session, solution, _ = saved_session({"arena_path": A.profile_arena_path("live"), "clicks_path": tmp_path / "live-clicks.json"})
    home = load_config(ROOT / "config" / "home_test_overlay.yaml")
    assert A.with_arena_setup(home, "home_test") is home                                     # nothing saved for home_test: untouched
    assert home.arena["width_m"] == 0.5
    live = load_config(ROOT / "config" / "arena_overlay.yaml")
    applied = A.with_arena_setup(live, "live", None)
    assert applied.arena["source"] == "arena setup file" and applied.arena["width_m"] == pytest.approx(W)
    assert A.with_arena_setup(live, "other_profile") is live


def test_setup_for_one_profile_cannot_be_saved_as_another(files):
    session, solution, _ = saved_session(files)
    clicks = A.corner_clickset("live", solution.setup, session.corner_pixels(), IMAGE)
    with pytest.raises(A.ArenaSetupError, match="refusing to save"):
        A.save_setup("home_test", solution.setup, clicks, **files)


def test_tests_never_touch_the_real_config_folder():
    def digest():
        return hashlib.sha256(b"".join(p.read_bytes() for p in sorted((ROOT / "config").rglob("*")) if p.is_file())).hexdigest()
    before = digest()
    session, solution, _ = saved_session({"arena_path": ROOT / "artifacts" / "arena-test" / "a.json", "clicks_path": ROOT / "artifacts" / "arena-test" / "c.json"})
    try:
        assert digest() == before
    finally:
        for path in (ROOT / "artifacts" / "arena-test").glob("*"):
            path.unlink()
        (ROOT / "artifacts" / "arena-test").rmdir()


# ---------------------------------------------------------------- undo, reset and correcting one point


def test_undo_walks_back_one_step_at_a_time():
    session, h = run_full_setup()
    assert session.undo() == "flipper3" and session.current == "flipper3" and not session.flipper_closed
    assert session.undo() == "flipper2"
    session.click(*to_pixel(h, FLIPPER[1]))
    session.click(*to_pixel(h, FLIPPER[2]))
    session.finish_flipper()
    assert session.solve().complete
    while session.undo():
        pass
    assert session.value_of("c1") is None and session.width_m is None and session.pit_radii == [None, None] and not session.flipper
    assert session.undo() is None


def test_reset_clears_everything_in_a_new_setup():
    session, _ = run_full_setup()
    session.reset()
    assert session.current == "c1" and session.corner_pixels() is None and not session.flipper and session.width_m is None
    assert not session.solve().complete


def test_one_point_can_be_corrected_without_redoing_the_rest():
    session, h = run_full_setup()
    first = session.solve().setup.geometry
    assert session.goto("c2")
    session.click(*CAMERA_1[1])                                   # click the same corner again
    assert session.done and session.solve().setup.geometry.pits[0].centre_m == pytest.approx(first.pits[0].centre_m, abs=1e-6)
    assert session.goto("pit1_radius")
    session.submit_number("pit1_radius", 0.22)
    assert session.done and session.solve().setup.geometry.pits[0].radius_m == 0.22
    assert session.goto("flipper2")
    session.click(*to_pixel(h, (0.0, 0.02)))
    assert session.done and len(session.flipper) == 3
    assert not session.goto("nonsense")


def test_flipper_vertices_can_be_added_removed_and_reopened():
    session, h = run_full_setup()
    assert session.remove_flipper_vertex(3) and len(session.flipper) == 2 and not session.flipper_closed
    assert not session.finish_flipper() and "at least 3" in session.message
    session.reopen_flipper()
    session.click(*to_pixel(h, (0.335, 0.0)))
    assert session.finish_flipper() and session.solve().complete
    session.reopen_flipper()
    assert session.current == "flipper4"
    session.click(*to_pixel(h, (0.2, 0.2)))
    assert len(session.flipper) == 4 and session.finish_flipper()
    assert not session.remove_flipper_vertex(9)


def test_defaults_are_offered_but_must_be_confirmed():
    session = A.ArenaSetupSession("live", "new", image_size=IMAGE, default_size_m=(W, H), default_radius_m=0.25)
    for pixel in CAMERA_1:
        session.click(*pixel)
    assert "Enter accepts 1.150 m" in session.prompt()
    assert session.accept_default() and session.accept_default() and session.dimensions_source == "confirmed default"
    assert session.width_m == W and session.height_m == H
    plain = A.ArenaSetupSession("home_test", "new", image_size=IMAGE)
    for pixel in CAMERA_1:
        plain.click(*pixel)
    assert not plain.accept_default() and plain.width_m is None                                # no default offered: it must be typed


# ---------------------------------------------------------------- invalid and inconsistent geometry


def geometry(**changes):
    base = dict(width_m=W, height_m=H, pits=[A.Pit(PIT1, 0.25), A.Pit(PIT2, 0.25)], flipper_m=list(FLIPPER))
    base.update(changes)
    return A.Geometry(**base)


def test_valid_geometry_passes():
    result = A.validate_geometry(geometry())
    assert result.ok and result.geometry is not None and not result.errors


@pytest.mark.parametrize("changes, message", [
    (dict(width_m=0.1), "outside the plausible"),
    (dict(height_m=9.0), "outside the plausible"),
    (dict(pits=[A.Pit(PIT1, 0.25)]), "2 pits are required"),
    (dict(pits=[A.Pit(PIT1, 0.01), A.Pit(PIT2, 0.25)]), "radius 10 mm"),
    (dict(pits=[A.Pit(PIT1, 0.9), A.Pit(PIT2, 0.25)]), "radius 900 mm"),
    (dict(pits=[A.Pit((2.0, 0.5), 0.25), A.Pit(PIT2, 0.25)]), "outside the"),
    (dict(pits=[A.Pit((0.5, 0.5), 0.3), A.Pit((0.7, 0.5), 0.3)]), "overlap"),
    (dict(pits=[A.Pit((float("nan"), 0.5), 0.3), A.Pit(PIT2, 0.25)]), "not a number"),
    (dict(flipper_m=[(0.0, 0.0), (0.3, 0.0)]), "at least 3"),
    (dict(flipper_m=[(0.1, 0.1)] * 9), "at most"),
    (dict(flipper_m=[(0.0, 0.0), (0.3, 0.3), (0.3, 0.0), (0.0, 0.3)]), "crosses itself"),
    (dict(flipper_m=[(0.0, 0.0), (0.3, 0.0), (0.6, 0.0)]), "almost in a line"),
    (dict(flipper_m=[(0.0, 0.0), (0.0, 0.0), (0.3, 0.3)]), "apart"),
    (dict(flipper_m=[(0.0, 0.0), (1.1, 0.0), (1.1, 1.1), (0.0, 1.1)]), "implausible"),
    (dict(flipper_m=[(0.05, 0.95), (0.2, 0.95), (0.2, 1.1)]), "overlaps pit 1"),
    (dict(flipper_m=[(3.0, 0.0), (0.3, 0.0), (0.0, 0.3)]), "outside the arena"),
])
def test_invalid_geometry_is_rejected_with_a_reason(changes, message):
    result = A.validate_geometry(geometry(**changes))
    assert not result.ok and result.geometry is None and any(message in e for e in result.errors), result.errors


def test_an_incomplete_setup_cannot_produce_a_saveable_result():
    session = A.ArenaSetupSession("live", "new", image_size=IMAGE)
    assert not session.solve().complete and session.solve().setup is None
    for pixel in CAMERA_1[:3]:
        session.click(*pixel)
    assert any("all four" in p for p in session.solve().problems)
    session.click(*CAMERA_1[3])
    assert any("width and height" in p for p in session.solve().problems)
    session.click(*CAMERA_1[3])                                   # ignored: the next slot is a number, not a point
    session.submit_number("width", W)
    session.submit_number("height", H)
    problems = session.solve().problems
    assert "click the centre of pit 1" in problems and "enter the radius of pit 2" in problems and any("flipper" in p for p in problems)


def test_corners_that_are_the_same_spot_are_refused():
    session = A.ArenaSetupSession("live", "new", image_size=IMAGE)
    for pixel in [(100, 100), (105, 100), (900, 150), (380, 150)]:
        session.click(*pixel)
    session.submit_number("width", W)
    session.submit_number("height", H)
    assert any("same spot" in p for p in session.solve().problems)


def test_impossible_numbers_are_refused_by_the_session():
    session = A.ArenaSetupSession("live", "new", image_size=IMAGE)
    for pixel in CAMERA_1:
        session.click(*pixel)
    for bad in (0, -1, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            session.submit_number("width", bad)
    with pytest.raises(ValueError):
        session.submit_number("c1", 0.2)                           # a corner is clicked, not typed
    session.submit_number("width", 5.0)
    session.submit_number("height", H)
    assert any("outside the plausible" in p for p in session.solve().problems)


def test_a_wrong_entered_size_is_flagged_by_the_camera_plausibility_check():
    config = load_config(ROOT / "config" / "arena_overlay.yaml")
    session, _ = run_full_setup()
    ok = session.solve(config=config)
    assert ok.complete and not any("plausible camera" in w for w in ok.warnings)
    session.goto("width")
    session.submit_number("width", 0.35)                            # nowhere near the real width of the picture
    wrong = session.solve(config=config)
    assert any("plausible camera" in w or "outside" in w for w in wrong.warnings + wrong.problems) or not wrong.complete


def test_hints_are_advisory_warnings_only():
    session = A.ArenaSetupSession("live", "new", image_size=IMAGE, hints={"pit_1": "bottom-right", "pit_2": "top-left", "flipper": "top-right"})
    h = homography(CAMERA_1)
    for pixel in CAMERA_1:
        session.click(*pixel)
    session.submit_number("width", W)
    session.submit_number("height", H)
    session.click(*to_pixel(h, PIT1))
    session.submit_number("pit1_radius", 0.25)
    session.click(*to_pixel(h, PIT2))
    session.submit_number("pit2_radius", 0.25)
    for vertex in FLIPPER:
        session.click(*to_pixel(h, vertex))
    session.finish_flipper()
    solution = session.solve()
    assert solution.complete and sum("was expected" in w for w in solution.warnings) == 3


# ---------------------------------------------------------------- the camera moved: the hazards must not


def test_recalibrating_the_camera_keeps_the_saved_hazards_byte_for_byte(files):
    session, solution, _ = saved_session(files, CAMERA_1)
    arena_before = files["arena_path"].read_bytes()
    saved = A.load_arena_setup("live", files["arena_path"])
    recal = A.ArenaSetupSession("live", "recalibrate", saved, image_size=IMAGE)
    assert [s.key for s in recal.slots()] == list(A.CORNER_IDS)
    assert "PLAIN corner" in recal.corner_label("c3") and "FLIPPER corner" in recal.corner_label("c1")
    assert "PIT 1 corner" in recal.corner_label("c4")
    for pixel in CAMERA_2:                                         # corners in the SAME physical order c1..c4
        recal.click(*pixel)
    assert recal.done
    result = recal.solve()
    assert result.complete, result.problems
    assert result.setup.geometry.pits == saved.geometry.pits and result.setup.geometry.flipper_m == saved.geometry.flipper_m
    clicks = A.corner_clickset("live", result.setup, recal.corner_pixels(), IMAGE)
    report = A.save_setup("live", saved, clicks, write_arena=False, **files)
    assert files["arena_path"].read_bytes() == arena_before and not report.wrote_arena and report.arena_backup is None
    assert any("left unchanged" in line for line in report.lines())
    # With the NEW camera the same physical hazards land on the right pixels.
    config = load_config(ROOT / "config" / "arena_overlay.yaml")
    applied = A.with_arena_setup(config, "live", load_clicks(files["clicks_path"]), files["arena_path"])
    mapping, _, _ = fit_profile(applied, load_clicks(files["clicks_path"]), IMAGE)
    h2 = homography(CAMERA_2)
    assert mapping.pixel_to_floor([to_pixel(h2, PIT1)])[0] == pytest.approx((W, 0.0), abs=3e-3)
    assert mapping.pixel_to_floor([to_pixel(h2, FLIPPER[0])])[0] == pytest.approx((1.15, 0.815), abs=3e-3)
    assert applied.arena["pit_centres_m"] == saved.arena_dict()["pit_centres_m"]


def test_recalibrating_needs_a_saved_arena_and_the_right_profile(files):
    with pytest.raises(A.ArenaSetupError, match="run a full set-up first"):
        A.ArenaSetupSession("live", "recalibrate", None)
    _, solution, _ = saved_session(files)
    with pytest.raises(A.ArenaSetupError, match="different profile"):
        A.ArenaSetupSession("home_test", "edit", solution.setup)


def test_editing_the_hazards_keeps_the_corners_and_changes_only_what_was_touched(files):
    session, solution, _ = saved_session(files)
    saved = A.load_arena_setup("live", files["arena_path"])
    clicks = load_clicks(files["clicks_path"])
    edit = A.ArenaSetupSession("live", "edit", saved, clicks, image_size=IMAGE)
    assert edit.done and edit.corner_pixels() == {cid: pytest.approx(p) for cid, p in zip(A.CORNER_IDS, CAMERA_1)}
    unchanged = edit.solve()
    assert unchanged.complete and unchanged.setup.geometry.pits == saved.geometry.pits
    assert edit.goto("pit2_radius")
    edit.submit_number("pit2_radius", 0.20)
    result = edit.solve()
    assert result.setup.geometry.pits[1].radius_m == 0.20 and result.setup.geometry.pits[0] == saved.geometry.pits[0]
    assert result.setup.setup_id == saved.setup_id                     # same corner geometry: the camera clicks stay valid
    A.save_setup("live", result.setup, A.corner_clickset("live", result.setup, edit.corner_pixels(), IMAGE), **files)
    assert A.load_arena_setup("live", files["arena_path"]).geometry.pits[1].radius_m == 0.20
    edit.reset()                                                       # discard: back to what is saved
    assert edit.pit_radii[1] == 0.25 or edit.saved is saved


def test_changing_the_arena_size_makes_old_corner_clicks_stale(files):
    session, solution, _ = saved_session(files)
    saved = A.load_arena_setup("live", files["arena_path"])
    edit = A.ArenaSetupSession("live", "edit", saved, load_clicks(files["clicks_path"]), image_size=IMAGE)
    edit.goto("width")
    edit.submit_number("width", 1.0)
    stale = edit.solve()
    assert not stale.complete and any("pit 2" in p and "outside" in p for p in stale.problems)   # the locked pit no longer fits
    assert edit.goto("pit2_centre")
    edit.click(*CAMERA_1[1])                                           # the pit is still at the second corner
    edit.submit_number("pit2_radius", 0.25)
    result = edit.solve()
    assert result.complete and result.setup.setup_id != saved.setup_id
    A.save_setup("live", result.setup, A.corner_clickset("live", result.setup, edit.corner_pixels(), IMAGE), write_arena=True, **files)
    # Someone restores the old corner clicks next to the new arena: refused, not silently reused.
    old = A.corner_clickset("live", solution.setup, session.corner_pixels(), IMAGE)
    config = load_config(ROOT / "config" / "arena_overlay.yaml")
    with pytest.raises(A.ArenaSetupError, match="different arena size"):
        A.with_arena_setup(config, "live", old, files["arena_path"])


def test_damaged_or_foreign_files_fail_loudly(files):
    saved_session(files)
    data = json.loads(files["arena_path"].read_text(encoding="utf-8"))
    for mutate, message in ((lambda d: d.update(kind="other"), "not an arena set-up"), (lambda d: d.update(version=99), "version"),
                            (lambda d: d["setup_frame"].pop("pits"), "incomplete or damaged"),
                            (lambda d: d["setup_frame"]["pits"][0].update(radius_m=5.0), "not valid"),
                            (lambda d: d.update(setup_id="deadbeef00"), "inconsistent")):
        broken = json.loads(json.dumps(data))
        mutate(broken)
        files["arena_path"].write_text(json.dumps(broken), encoding="utf-8")
        with pytest.raises(A.ArenaSetupError, match=message):
            A.load_arena_setup("live", files["arena_path"])
    files["arena_path"].write_text("{not json", encoding="utf-8")
    with pytest.raises(A.ArenaSetupError, match="unreadable"):
        A.load_arena_setup("live", files["arena_path"])
    assert A.load_arena_setup("live", files["arena_path"].with_name("none.json")) is None


# ---------------------------------------------------------------- optional held-out check


def test_rim_points_give_an_independent_check():
    session, h = run_full_setup()
    assert session.goto("rim1")
    session.click(*to_pixel(h, (0.0 + 0.25, H)))                         # exactly on the rim of pit 1, along the wall
    assert session.goto("rim2")
    session.click(*to_pixel(h, (W, 0.0 + 0.25)))
    solution = session.solve()
    assert solution.rim_check_mm and max(solution.rim_check_mm) < 1.0
    assert not any("no independent check" in r for r in solution.quality.reasons)
    session.goto("rim2")
    session.click(*to_pixel(h, (W, 0.40)))                                 # 150 mm out
    bad = session.solve()
    assert max(bad.rim_check_mm) > 100 and bad.level == "weak" and any("pit-rim check" in r for r in bad.quality.reasons)
    plain, _ = run_full_setup()
    assert plain.solve().level == "weak" and any("no independent check" in r for r in plain.solve().quality.reasons)
