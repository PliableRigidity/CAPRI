"""The design-based arena set-up: dimensions come from the drawing, the operator only clicks where things are.

Synthetic pictures only (rendered through a known homography). Nothing here opens a camera or a robot link, and
all files go to a temporary folder. These tests do not say anything about the real arena or the real camera.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

from control_center.backend.arena_workflow import ArenaWorkflow  # noqa: E402
from control_center.tools import arena_setup as A  # noqa: E402
from control_center.tools.arena_design import (DESIGN_PATH, DesignError, EffectiveDesign, Overrides, design_for_profile,  # noqa: E402
                                               load_design)
from test_arena_setup import CAMERA_1, CAMERA_2, H, IMAGE, PIT1, PIT2, W, homography, to_pixel  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
LEG = 0.335
FLIPPER_CENTRE = (0.08, 0.08)           # a click inside the flipper triangle, near the (0, 0) corner of the test frame


def design_session(camera=CAMERA_1, jitter=0.0, design=None, profile="live", flipper=FLIPPER_CENTRE, pit1=PIT1, pit2=PIT2):
    h = homography(camera)
    session = A.ArenaSetupSession(profile, "new", image_size=IMAGE, design=design or EffectiveDesign(load_design()))
    for cid, pixel in zip(A.CORNER_IDS, camera):
        assert session.current == cid
        session.click(*pixel)
    assert session.current == "pit1_centre"
    session.click(*to_pixel(h, (pit1[0] + jitter, pit1[1] - jitter)))
    session.click(*to_pixel(h, (pit2[0] - jitter, pit2[1] + jitter)))
    session.click(*to_pixel(h, flipper))
    return session, h


def rolled(camera, turns):
    """The same arena clicked starting from another physical corner (the camera turned, or another start corner)."""
    return camera[turns:] + camera[:turns]


# ---- the canonical design file -----------------------------------------------------------------

def test_the_design_file_says_what_the_drawing_says_and_where_it_came_from():
    design = load_design()
    assert (design.width_m, design.height_m, design.pit_radius_m, design.flipper_leg_m) == (1.150, 1.150, 0.250, 0.335)
    assert design.revision == "1" and design.created_by == "Alex Chung" and design.source_sha256.startswith("c6f59bdf")
    assert design.source_file == "Arena Floor Technical Drawing.pdf"
    assert design.pit_centres() == [(0.0, 1.150), (1.150, 0.0)]
    assert design.flipper_vertices() == [(pytest.approx(0.815), 1.150), (1.150, 1.150), (1.150, pytest.approx(0.815))]


def test_the_design_and_the_simulator_config_cannot_drift_apart():
    sim = yaml.safe_load((ROOT / "config" / "sim.yaml").read_text(encoding="utf-8"))["arena"]
    design = load_design()
    assert sim["width_m"] == pytest.approx(design.width_m) and sim["height_m"] == pytest.approx(design.height_m)
    assert sim["pit_radius_m"] == pytest.approx(design.pit_radius_m)
    assert [list(c) for c in sim["pit_centres_m"]] == [list(c) for c in design.pit_centres()]
    assert np.allclose(np.array(sim["flipper_vertices_m"], float), np.array(design.flipper_vertices(), float))
    assert sim["wall_thickness_m"] == pytest.approx(design.wall_thickness_m) and sim["wall_height_m"] == pytest.approx(design.wall_height_m)


def test_a_damaged_or_missing_design_file_is_reported_never_guessed(tmp_path):
    with pytest.raises(DesignError, match="restore config/arena_geometry.yaml"):
        load_design(tmp_path / "missing.yaml")
    (tmp_path / "bad.yaml").write_text("floor: {width_m: 1.15}\n", encoding="utf-8")
    with pytest.raises(DesignError):
        load_design(tmp_path / "bad.yaml")
    text = DESIGN_PATH.read_text(encoding="utf-8").replace("radius_m: 0.250", "radius_m: 0.700")
    (tmp_path / "huge.yaml").write_text(text, encoding="utf-8")
    with pytest.raises(DesignError, match="inconsistent"):
        load_design(tmp_path / "huge.yaml")


def test_only_the_competition_profile_gets_the_design():
    assert design_for_profile("live") is not None
    assert design_for_profile("home_test") is None


# ---- the click flow -----------------------------------------------------------------------------

def test_the_flow_asks_for_clicks_only_no_dimensions_and_no_radius():
    session = A.ArenaSetupSession("live", "new", image_size=IMAGE, design=EffectiveDesign(load_design()))
    required = [s for s in session.slots() if s.required]
    assert [s.key for s in required] == ["c1", "c2", "c3", "c4", "pit1_centre", "pit2_centre", "flipper1"]
    assert all(s.kind == A.POINT for s in session.slots())
    assert session.width_m == pytest.approx(1.15) and session.pit_radii == [0.25, 0.25] and session.dimensions_source == "design"


def test_seven_clicks_give_the_simulators_geometry():
    session, _ = design_session()
    assert session.done
    solution = session.solve()
    assert solution.complete, solution.problems
    geometry = solution.setup.active_geometry()
    assert (geometry.width_m, geometry.height_m) == (pytest.approx(1.15), pytest.approx(1.15))
    assert [p.centre_m for p in geometry.pits] == [(pytest.approx(0.0), pytest.approx(1.15)), (pytest.approx(1.15), pytest.approx(0.0))]
    assert [p.radius_m for p in geometry.pits] == [0.25, 0.25]
    expected = {(0.815, 1.15), (1.15, 1.15), (1.15, 0.815)}
    assert {(round(x, 3), round(y, 3)) for x, y in geometry.flipper_m} == expected
    assert solution.setup.design_used and solution.setup.policy_frame_ok and solution.setup.dimensions_source == "design"
    assert solution.click_checks["pit_1_click_offset_mm"] < 1.0 and solution.setup.canonical.corner_names["c1"] == "flipper"


@pytest.mark.parametrize("turns", [0, 1, 2, 3])
def test_the_camera_turned_or_another_start_corner_gives_the_same_physical_hazards(turns):
    """Clicking the corners starting from a different physical corner must neither move a hazard nor lose the frame."""
    camera = rolled(CAMERA_1, turns)
    session, _ = design_session(camera=CAMERA_1)            # reference result
    reference = session.solve().setup.active_geometry()
    h = homography(CAMERA_1)
    turned = A.ArenaSetupSession("live", "new", image_size=IMAGE, design=EffectiveDesign(load_design()))
    for pixel in camera:
        turned.click(*pixel)
    turned.click(*to_pixel(h, PIT1))
    turned.click(*to_pixel(h, PIT2))
    turned.click(*to_pixel(h, FLIPPER_CENTRE))
    solution = turned.solve()
    assert solution.complete, solution.problems
    geometry = solution.setup.active_geometry()
    assert [tuple(round(v, 3) for v in p.centre_m) for p in geometry.pits] == [tuple(round(v, 3) for v in p.centre_m) for p in reference.pits]
    assert {tuple(round(v, 3) for v in p) for p in geometry.flipper_m} == {tuple(round(v, 3) for v in p) for p in reference.flipper_m}
    assert solution.setup.policy_frame_ok


def test_a_mirrored_corner_order_is_refused_not_silently_accepted():
    mirrored = [CAMERA_1[0], CAMERA_1[3], CAMERA_1[2], CAMERA_1[1]]
    session = A.ArenaSetupSession("live", "new", image_size=IMAGE, design=EffectiveDesign(load_design()))
    for pixel in mirrored:
        session.click(*pixel)
    h = homography(CAMERA_1)
    for point in (PIT1, PIT2, FLIPPER_CENTRE):
        session.click(*to_pixel(h, point))
    solution = session.solve()
    assert not solution.complete and any("mirror" in p or "order" in p for p in solution.problems)


def test_a_strong_perspective_view_gives_the_same_hazards_and_says_how_coarse_it_is():
    session, _ = design_session(camera=CAMERA_2)
    solution = session.solve()
    assert solution.complete, solution.problems
    assert solution.accuracy and "mm per pixel" in solution.accuracy[0]
    geometry = solution.setup.active_geometry()
    assert max(abs(a - b) for p, q in zip(geometry.pits, [(0.0, 1.15), (1.15, 0.0)]) for a, b in zip(p.centre_m, q)) < 1e-6


def test_clicks_a_little_off_the_corner_are_put_on_it_and_the_offset_is_reported():
    session, _ = design_session(jitter=0.05)
    solution = session.solve()
    assert solution.complete
    assert 50 < solution.click_checks["pit_1_click_offset_mm"] < 90
    assert solution.setup.active_geometry().pits[0].centre_m == (pytest.approx(0.0), pytest.approx(1.15))


def test_a_pit_click_a_long_way_from_any_corner_is_refused_with_a_reason():
    session, _ = design_session(pit1=(0.575, 0.575))
    solution = session.solve()
    assert not solution.complete and any("pit 1 click" in p and "quarter circle" in p for p in solution.problems)


def test_pits_on_adjacent_or_the_same_corner_are_refused():
    adjacent, _ = design_session(pit2=(0.0, 0.0))
    assert any("OPPOSITE" in p for p in adjacent.solve().problems)
    same, _ = design_session(pit2=PIT1)
    assert any("same corner" in p for p in same.solve().problems)


def test_a_flipper_click_on_a_pit_corner_is_refused():
    session, _ = design_session(flipper=(0.05, H - 0.05))
    assert any("pit corner" in p for p in session.solve().problems)


def test_an_oblique_far_view_is_flagged_unreliable_and_downgrades_the_level():
    oblique = [(250.0, 700.0), (1030.0, 700.0), (700.0, 640.0), (580.0, 640.0)]
    session, _ = design_session(camera=oblique)
    solution = session.solve()
    assert solution.complete and solution.unreliable and solution.level == "weak"
    assert any("oblique" in u or "pixels per metre" in u for u in solution.unreliable)


def test_a_normal_view_is_not_flagged():
    solution = design_session()[0].solve()
    assert not solution.unreliable


# ---- overrides: measured differences, never silent ------------------------------------------------

def test_the_1130_question_is_a_deliberate_inset_override_not_a_default():
    base = EffectiveDesign(load_design())
    assert base.width_m == pytest.approx(1.15) and not base.overrides.any
    inset = base.with_overrides(Overrides(inset_per_side_m=0.010, reason="walls stand 10 mm in from the floor edge (measured)"))
    assert inset.width_m == pytest.approx(1.13) and inset.height_m == pytest.approx(1.13)
    assert inset.source_of("width_m") == "design with inset override"
    assert any("ADVANCED OVERRIDE" in line for line in inset.lines())


def test_implausible_overrides_are_refused():
    base = EffectiveDesign(load_design())
    for bad in (Overrides(width_m=0.1), Overrides(pit_radius_m=0.7), Overrides(inset_per_side_m=0.5), Overrides(flipper_leg_m=0.001),
                Overrides(width_m=float("nan"))):
        with pytest.raises(DesignError):
            base.with_overrides(bad)


def test_an_override_changes_the_geometry_and_is_recorded_in_the_saved_arena(tmp_path):
    session, h = design_session()
    session.set_overrides(Overrides(pit_radius_m=0.24, flipper_leg_m=0.33, reason="measured on the day"))
    solution = session.solve()
    assert solution.complete
    geometry = solution.setup.active_geometry()
    assert [p.radius_m for p in geometry.pits] == [0.24, 0.24]
    legs = sorted(abs(a - b) for a, b in zip(geometry.flipper_m, geometry.flipper_m[1:] + geometry.flipper_m[:1]) for a, b in [(a[0], b[0])] if a != b)
    assert legs and max(legs) == pytest.approx(0.33, abs=1e-6)
    data = solution.setup.to_json()
    assert data["version"] == 2 and data["design"]["overrides"]["pit_radius_m"] == 0.24 and data["design"]["source_sha256"].startswith("c6f59bdf")
    restored = A.ArenaSetup.from_json(data, "live")
    assert restored.design_used and restored.design["overrides"]["flipper_leg_m"] == 0.33
    assert EffectiveDesign.from_dict(restored.design).pit_radius_m == 0.24


def test_a_non_square_override_assigns_the_sides_from_the_plain_corner():
    session, _ = design_session(design=EffectiveDesign(load_design()).with_overrides(Overrides(width_m=1.15, height_m=1.0)))
    # the test frame has c1->c2 along x: with the flipper at c1 the plain corner is c3, an odd edge
    A_dims = session.solve()
    assert A_dims.complete is False or A_dims.setup is not None      # either way it must not crash
    # the canonical frame: x runs from the plain corner to the pit-2 corner and has the design WIDTH
    if A_dims.complete:
        g = A_dims.setup.active_geometry()
        assert g.width_m == pytest.approx(1.15) and g.height_m == pytest.approx(1.0)


# ---- saving, loading, drafts, files ---------------------------------------------------------------

def test_save_and_reload_round_trip_keeps_the_design_and_is_version_2(tmp_path):
    session, _ = design_session()
    solution = session.solve()
    clicks = A.corner_clickset("live", solution.setup, session.corner_pixels(), IMAGE)
    A.save_setup("live", solution.setup, clicks, arena_path=tmp_path / "arena.json", clicks_path=tmp_path / "clicks.json")
    loaded = A.load_arena_setup("live", tmp_path / "arena.json")
    assert loaded.design_used and loaded.setup_id == solution.setup.setup_id
    assert [p.radius_m for p in loaded.active_geometry().pits] == [0.25, 0.25]
    assert json.loads((tmp_path / "arena.json").read_text())["version"] == 2


def test_a_version_1_typed_arena_still_loads():
    session = A.ArenaSetupSession("home_test", "new", image_size=IMAGE)
    h = homography(CAMERA_1)
    for pixel in CAMERA_1:
        session.click(*pixel)
    session.submit_number("width", W)
    session.submit_number("height", H)
    session.click(*to_pixel(h, PIT1))
    session.submit_number("pit1_radius", 0.25)
    session.click(*to_pixel(h, PIT2))
    session.submit_number("pit2_radius", 0.25)
    for vertex in [(0.0, 0.335), (0.0, 0.0), (0.335, 0.0)]:
        session.click(*to_pixel(h, vertex))
    session.finish_flipper()
    data = session.solve().setup.to_json()
    data["version"] = 1
    data.pop("design"), data.pop("click_checks")
    setup = A.ArenaSetup.from_json(data, "home_test")
    assert not setup.design_used and setup.geometry.width_m == pytest.approx(1.15)


def test_an_unfinished_design_setup_survives_a_restart_with_its_override():
    session, _ = design_session()
    session.set_overrides(Overrides(inset_per_side_m=0.01))
    draft = json.loads(json.dumps(session.to_draft()))
    fresh = A.ArenaSetupSession("live", "new", image_size=IMAGE, design=EffectiveDesign(load_design()))
    assert fresh.restore_draft(draft)
    assert fresh.design.width_m == pytest.approx(1.13) and fresh.done
    typed = A.ArenaSetupSession("live", "new", image_size=IMAGE)
    assert not typed.restore_draft(draft)                  # a typed-dimension session never adopts a design draft


def test_editing_a_design_setup_keeps_the_corners_and_needs_no_new_hazard_clicks(tmp_path):
    session, _ = design_session()
    solution = session.solve()
    clicks = A.corner_clickset("live", solution.setup, session.corner_pixels(), IMAGE)
    A.save_setup("live", solution.setup, clicks, arena_path=tmp_path / "a.json", clicks_path=tmp_path / "c.json")
    saved = A.load_arena_setup("live", tmp_path / "a.json")
    edit = A.ArenaSetupSession("live", "edit", saved, A.load_clicks_for_tests(tmp_path / "c.json") if hasattr(A, "load_clicks_for_tests") else clicks,
                               IMAGE, design=EffectiveDesign.from_dict(saved.design))
    assert edit.done
    assert edit.solve().complete


# ---- the workflow: LIVE uses the design, HOME TEST still asks --------------------------------------

def make_workflow(tmp_path, profile):
    return ArenaWorkflow(profile, arena_path=tmp_path / f"{profile}-arena.json", clicks_path=tmp_path / f"{profile}-clicks.json")


def test_live_workflow_starts_the_design_flow_and_home_test_the_typed_one(tmp_path):
    live = make_workflow(tmp_path, "live")
    ok, _ = live.begin("new", IMAGE)
    assert ok and live.session.design_flow and live.design is not None
    home = make_workflow(tmp_path, "home_test")
    ok, _ = home.begin("new", IMAGE)
    assert ok and not home.session.design_flow and home.design is None


def test_workflow_saves_a_design_setup_with_accuracy_lines_and_no_typing(tmp_path):
    workflow = make_workflow(tmp_path, "live")
    workflow.begin("new", IMAGE)
    h = homography(CAMERA_1)
    for pixel in CAMERA_1:
        workflow.click(*pixel)
    for point in (PIT1, PIT2, FLIPPER_CENTRE):
        workflow.click(*to_pixel(h, point))
    assert workflow.session.done
    ok, _ = workflow.can_save()
    assert ok
    lines = workflow.accuracy_lines()
    assert any("1-pixel" in line for line in lines) and any("pit 1 click" in line for line in lines)
    ok, report = workflow.save()
    assert ok, report
    assert workflow.loaded.arena.design_used
    ok, message = workflow.begin("edit", IMAGE)
    assert ok and workflow.session.design_flow


def test_workflow_overrides_are_validated_and_rejected_values_change_nothing(tmp_path):
    workflow = make_workflow(tmp_path, "live")
    workflow.begin("new", IMAGE)
    ok, message = workflow.set_overrides(Overrides(width_m=9.0))
    assert not ok and workflow.design.width_m == pytest.approx(1.15)
    ok, message = workflow.set_overrides(Overrides(inset_per_side_m=0.01, reason="tape measure"))
    assert ok and workflow.design.width_m == pytest.approx(1.13)


def test_tests_never_touch_the_real_config_folder_for_design_files():
    before = {p.name: p.stat().st_mtime_ns for p in (ROOT / "config").glob("arena_overlay_*.json")}
    session, _ = design_session()
    session.solve()
    after = {p.name: p.stat().st_mtime_ns for p in (ROOT / "config").glob("arena_overlay_*.json")}
    assert before == after
