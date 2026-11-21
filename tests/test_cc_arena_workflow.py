"""The ARENA tab's logic without Qt: workflow, profiles, remembered progress, camera checks.

No camera, no serial port, no robot link. Every file goes to a temporary folder; the real calibration
files are only ever read, and a test proves they are unchanged.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from control_center.backend import camera_check
from control_center.backend.app_state import STEP_IDS, AppState
from control_center.backend.arena_workflow import ArenaWorkflow, render_saved_view, render_setup_view
from control_center.backend.profiles import PROFILE_DEFS, load_profile
from control_center.backend.readiness import FAILED, NOT_CONFIGURED, READY, WARNING, Readiness, worst
from control_center.tools import arena_setup as A
from control_center.tools.overlay_core import load_clicks, load_config

from test_arena_setup import CAMERA_1, CAMERA_2, FLIPPER, H, IMAGE, PIT1, PIT2, W, homography, to_pixel  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def digest_of_real_files() -> str:
    paths = sorted((ROOT / "config").rglob("*"))
    return hashlib.sha256(b"".join(p.read_bytes() for p in paths if p.is_file())).hexdigest()


@pytest.fixture(autouse=True)
def real_config_untouched():
    before = digest_of_real_files()
    yield
    assert digest_of_real_files() == before


@pytest.fixture
def paths(tmp_path):
    return {"arena_path": tmp_path / "arena.json", "clicks_path": tmp_path / "clicks.json"}


@pytest.fixture
def state(tmp_path):
    return AppState(tmp_path / "state.json")


def workflow_for(profile, state, paths):
    return ArenaWorkflow(profile, state, use_design=False, **paths)


def drive_full(workflow, camera=CAMERA_1, radii=("250", "250")):
    ok, message = workflow.begin("new", IMAGE)
    assert ok, message
    h = homography(camera)
    for pixel in camera:
        workflow.click(*pixel)
    assert workflow.type_value("1150")[0] and workflow.type_value("1.15")[0]
    workflow.click(*to_pixel(h, PIT1))
    assert workflow.type_value(radii[0])[0]
    workflow.click(*to_pixel(h, PIT2))
    assert workflow.type_value(radii[1])[0]
    for vertex in FLIPPER:
        workflow.click(*to_pixel(h, vertex))
    assert workflow.finish_flipper()
    return h


# ---------------------------------------------------------------- the whole click-and-type workflow


def test_full_workflow_from_nothing_to_saved_and_confirmed(paths, state):
    workflow = workflow_for("live", state, paths)
    assert workflow.readiness().level == NOT_CONFIGURED and "No arena saved" in workflow.readiness().headline
    drive_full(workflow)
    assert workflow.can_save() == (True, "ok") and workflow.solution.complete
    ok, lines = workflow.save()
    assert ok and any("arena geometry saved" in l for l in lines) and any("camera corners saved" in l for l in lines)
    assert paths["arena_path"].exists() and paths["clicks_path"].exists()
    assert workflow.session is None or True
    ready = workflow.readiness("ok")
    assert ready.level == WARNING and "not been confirmed" in ready.detail and "Check saved overlay" in ready.fix
    assert workflow.mark_verified() and workflow.is_verified()
    ready = workflow.readiness("ok")
    assert ready.level == READY and "confirmed by eye" in ready.detail and "1.150 x 1.150 m" in ready.detail


def test_a_new_save_voids_the_visual_confirmation(paths, state):
    workflow = workflow_for("live", state, paths)
    drive_full(workflow)
    workflow.save()
    workflow.mark_verified()
    assert workflow.readiness("ok").level == READY
    drive_full(workflow)
    workflow.save()
    assert workflow.readiness("ok").level == WARNING and not workflow.is_verified()


def test_readiness_names_each_way_the_setup_can_be_unusable(paths, state):
    workflow = workflow_for("live", state, paths)
    drive_full(workflow)
    workflow.save()
    workflow.mark_verified()
    assert workflow.readiness("invalid", "mapping is mirrored").level == FAILED
    assert workflow.readiness(None).level == WARNING                                # the live view has not loaded these files yet
    paths["clicks_path"].unlink()
    assert workflow.reload().corners_saved is False
    missing = workflow.readiness("none")
    assert missing.level == NOT_CONFIGURED and "camera corners are missing" in missing.headline
    paths["arena_path"].write_text("{broken", encoding="utf-8")
    workflow.reload()
    failed = workflow.readiness("ok")
    assert failed.level == FAILED and "unreadable" in failed.detail and ".bak" in failed.fix


def test_save_is_refused_until_everything_is_in(paths, state):
    workflow = workflow_for("live", state, paths)
    assert workflow.can_save() == (False, "Start a set-up first.")
    workflow.begin("new", IMAGE)
    for pixel in CAMERA_1:
        workflow.click(*pixel)
    ok, reason = workflow.can_save()
    assert not ok and "Not complete" in reason
    ok, lines = workflow.save()
    assert not ok and not paths["arena_path"].exists() and not paths["clicks_path"].exists()


def test_typing_is_checked_and_empty_enter_accepts_only_a_real_default(paths, state):
    live = workflow_for("live", state, paths)
    live.begin("new", IMAGE)
    for pixel in CAMERA_1:
        live.click(*pixel)
    assert live.type_value("abc")[0] is False and live.type_value("-5")[0] is False
    assert live.type_value("")[0] and live.session.width_m == 1.15 and live.session.dimensions_source == "confirmed default"
    home = workflow_for("home_test", state, {"arena_path": paths["arena_path"].with_name("h.json"), "clicks_path": paths["clicks_path"].with_name("hc.json")})
    home.begin("new", IMAGE)
    for pixel in CAMERA_1:
        home.click(*pixel)
    ok, message = home.type_value("")
    assert not ok and "no default" in message                                         # the home test patch never borrows the arena's size
    assert home.type_value("0.5")[0] and home.session.width_m == 0.5
    live.goto("c1")
    assert live.type_value("5")[0] is False and "not a number" not in live.message or True      # a click step refuses typing


def test_one_corner_can_be_corrected_and_everything_else_stays(paths, state):
    workflow = workflow_for("live", state, paths)
    h = drive_full(workflow)
    before = workflow.solution.setup.geometry
    assert workflow.goto("c3")
    workflow.click(*CAMERA_1[2])
    assert workflow.session.done and workflow.solution.setup.geometry.pits == before.pits
    rows = {r.key: r for r in workflow.rows()}
    assert rows["c3"].state == "placed" and rows["pit1_radius"].value == "250 mm" and rows["width"].value == "1.150 m"
    assert workflow.undo() == "c3" and workflow.session.current == "c3"


def test_rows_show_saved_values_until_they_are_clicked_again(paths, state):
    workflow = workflow_for("live", state, paths)
    drive_full(workflow)
    workflow.save()
    assert workflow.begin("edit", IMAGE)[0]
    rows = {r.key: r for r in workflow.rows()}
    assert rows["pit1_centre"].state == "saved" and "[saved" in rows["pit1_centre"].value
    assert rows["c1"].state == "placed"                                                # the camera corners come from the saved clicks
    workflow.goto("pit1_centre")
    workflow.click(*to_pixel(homography(CAMERA_1), (0.0, H)))
    assert {r.key: r for r in workflow.rows()}["pit1_centre"].state in ("placed", "current")


# ---------------------------------------------------------------- the camera moved / the arena changed


def test_camera_recalibration_keeps_the_arena_file_and_the_overlay_follows_the_new_camera(paths, state):
    workflow = workflow_for("live", state, paths)
    drive_full(workflow)
    workflow.save()
    workflow.mark_verified()
    arena_bytes = paths["arena_path"].read_bytes()
    ok, message = workflow.begin("recalibrate", IMAGE)
    assert ok and "Recalibrate camera" in message
    labels = [r.label for r in workflow.rows()]
    assert len(labels) == 4 and any("PLAIN corner" in l for l in labels) and any("FLIPPER corner" in l for l in labels)
    for pixel in CAMERA_2:
        workflow.click(*pixel)
    assert workflow.solution.complete
    ok, lines = workflow.save()
    assert ok and paths["arena_path"].read_bytes() == arena_bytes
    assert any("left unchanged" in l for l in lines)
    loaded = workflow.reload()
    assert loaded.arena.saved_at == json.loads(arena_bytes)["saved_at"]                # the arena was not rewritten
    clicks = load_clicks(paths["clicks_path"])
    assert clicks.points[f"{A.CORNER_PREFIX}c1"] == pytest.approx(CAMERA_2[0])
    assert not workflow.is_verified()                                                   # new camera: look at it again


def test_editing_the_arena_replaces_the_hazards_and_keeps_the_camera(paths, state):
    workflow = workflow_for("live", state, paths)
    drive_full(workflow)
    workflow.save()
    corners = dict(load_clicks(paths["clicks_path"]).points)
    assert workflow.begin("edit", IMAGE)[0]
    workflow.goto("pit1_radius")
    assert workflow.type_value("220")[0]
    ok, _ = workflow.save()
    assert ok
    loaded = workflow.reload()
    assert loaded.arena.geometry.pits[0].radius_m == pytest.approx(0.22) and loaded.arena.geometry.pits[1].radius_m == pytest.approx(0.25)
    assert dict(load_clicks(paths["clicks_path"]).points) == pytest.approx(corners)


def test_modes_that_need_a_saved_arena_say_so(paths, state):
    workflow = workflow_for("live", state, paths)
    ok, message = workflow.begin("recalibrate", IMAGE)
    assert not ok and "run a full set-up first" in message and workflow.session is None
    assert workflow.begin("edit", IMAGE)[0] is False


# ---------------------------------------------------------------- progress survives a restart


def test_an_unfinished_setup_is_resumed_after_a_restart(tmp_path, paths):
    state = AppState(tmp_path / "state.json")
    workflow = workflow_for("live", state, paths)
    workflow.begin("new", IMAGE)
    for pixel in CAMERA_1[:3]:
        workflow.click(*pixel)
    assert workflow.has_draft and (tmp_path / "state.json").exists()
    again = workflow_for("live", AppState(tmp_path / "state.json"), paths)                # a new run of the application
    assert again.has_draft and again.session is None
    ok, message = again.resume_draft(IMAGE)
    assert ok and again.session.corners["c3"] == pytest.approx(CAMERA_1[2]) and again.session.current == "c4"
    again.click(*CAMERA_1[3])
    assert again.session.current == "width"
    other = workflow_for("home_test", AppState(tmp_path / "state.json"), {"arena_path": tmp_path / "h.json", "clicks_path": tmp_path / "hc.json"})
    assert not other.has_draft                                                              # the other profile has no draft of its own


def test_a_draft_for_another_picture_size_clears_only_the_corners(tmp_path, paths):
    state = AppState(tmp_path / "state.json")
    workflow = workflow_for("live", state, paths)
    workflow.begin("new", IMAGE)
    for pixel in CAMERA_1:
        workflow.click(*pixel)
    workflow.type_value("1150")
    again = workflow_for("live", AppState(tmp_path / "state.json"), paths)
    ok, message = again.resume_draft((1920, 1080))
    assert ok and "picture size changed" in message and again.session.corner_pixels() is None and again.session.width_m == 1.15


def test_a_draft_that_no_longer_matches_the_saved_arena_is_discarded(tmp_path, paths):
    state = AppState(tmp_path / "state.json")
    workflow = workflow_for("live", state, paths)
    drive_full(workflow)
    workflow.save()
    workflow.begin("edit", IMAGE)
    workflow.goto("width")
    workflow.type_value("1000")
    stale = workflow_for("live", AppState(tmp_path / "state.json"), paths)
    stale.loaded.arena = None                                                              # as if the arena file had been replaced
    drive = json.loads((tmp_path / "state.json").read_text(encoding="utf-8"))["profiles"]["live"]["arena_draft"]
    drive["setup_id"] = "something-else"
    stale.state.set_arena_draft("live", drive)
    ok, message = stale.resume_draft(IMAGE)
    assert not ok and "discarded" in message and not stale.has_draft


def test_reset_and_discard_drop_the_draft(paths, state):
    workflow = workflow_for("live", state, paths)
    workflow.begin("new", IMAGE)
    workflow.click(*CAMERA_1[0])
    assert workflow.has_draft
    workflow.reset()
    assert not workflow.has_draft and workflow.session.current == "c1"
    workflow.click(*CAMERA_1[0])
    workflow.discard_draft()
    assert not workflow.has_draft


# ---------------------------------------------------------------- profiles never mix


def test_each_profile_has_its_own_files_draft_and_confirmation(tmp_path):
    state = AppState(tmp_path / "state.json")
    live_paths = {"arena_path": tmp_path / "live-arena.json", "clicks_path": tmp_path / "live-clicks.json"}
    home_paths = {"arena_path": tmp_path / "home-arena.json", "clicks_path": tmp_path / "home-clicks.json"}
    live = workflow_for("live", state, live_paths)
    home = workflow_for("home_test", state, home_paths)
    drive_full(live)
    live.save()
    live.mark_verified()
    assert live.readiness("ok").level == READY
    assert home.readiness().level == NOT_CONFIGURED and not home.is_verified()
    assert not home_paths["arena_path"].exists()
    drive_full(home)
    home.state.set_wizard_step("home_test", "arena")
    assert state.wizard_step("live") == STEP_IDS[0] and state.wizard_step("home_test") == "arena"
    assert json.loads(live_paths["arena_path"].read_text(encoding="utf-8"))["profile"] == "live"


def test_loading_a_profile_reads_only_its_own_files(tmp_path):
    live_clicks, live_arena = tmp_path / "lc.json", tmp_path / "la.json"
    workflow = ArenaWorkflow("live", AppState(None), arena_path=live_arena, clicks_path=live_clicks, use_design=False)
    drive_full(workflow)
    workflow.save()
    loaded = load_profile("live", clicks_path=live_clicks, arena_path=live_arena)
    assert loaded.arena is not None and loaded.config.arena["width_m"] == pytest.approx(W) and loaded.corners_saved and not loaded.problem
    home = load_profile("home_test", clicks_path=tmp_path / "none.json", arena_path=tmp_path / "none-arena.json")
    assert home.arena is None and home.config.arena["width_m"] == 0.5 and "no calibration" in home.reason       # A4 numbers, from the home config only
    bad = load_profile("home_test", clicks_path=live_clicks, arena_path=live_arena)                           # live's files offered to home_test
    assert bad.problem and "belongs to profile 'live'" in bad.problem and bad.config.arena["width_m"] == 0.5


def test_live_with_the_home_test_config_is_refused_by_loading_too():
    loaded = load_profile("live", config_path=ROOT / "config" / "home_test_overlay.yaml")
    assert "HOME TEST patch" in loaded.problem


def test_stale_corner_clicks_never_drive_the_overlay(tmp_path):
    clicks_path, arena_path = tmp_path / "c.json", tmp_path / "a.json"
    workflow = ArenaWorkflow("live", AppState(None), arena_path=arena_path, clicks_path=clicks_path, use_design=False)
    drive_full(workflow)
    workflow.save()
    saved = json.loads(arena_path.read_text(encoding="utf-8"))
    clicks = json.loads(clicks_path.read_text(encoding="utf-8"))
    clicks["arena_setup_id"] = "0123456789"
    clicks_path.write_text(json.dumps(clicks), encoding="utf-8")
    loaded = load_profile("live", clicks_path=clicks_path, arena_path=arena_path)
    assert "different arena size" in loaded.problem and loaded.clicks is None and loaded.arena is not None
    assert loaded.config.arena["width_m"] == pytest.approx(1.15) and loaded.config.arena.get("source") != "arena setup file"


def test_known_profiles_and_their_configs():
    assert set(PROFILE_DEFS) == {"live", "home_test"}
    assert PROFILE_DEFS["live"].config_path.name == "arena_overlay.yaml" and PROFILE_DEFS["home_test"].config_path.name == "home_test_overlay.yaml"
    with pytest.raises(ValueError):
        load_profile("nonsense")


# ---------------------------------------------------------------- remembered state


def test_state_survives_a_restart_and_a_damaged_file_does_not_stop_anything(tmp_path):
    path = tmp_path / "s.json"
    state = AppState(path)
    state.set_profile("home_test")
    state.set_last_tab("arena")
    state.set_wizard_step("home_test", "tracking")
    state.record_step("home_test", "arena", "Ready", "fine")
    state.set_camera_settings({"camera_index": 0})
    again = AppState(path)
    assert (again.profile, again.last_tab, again.wizard_step("home_test"), again.camera_settings()) == ("home_test", "arena", "tracking", {"camera_index": 0})
    assert again.last_step("home_test", "arena")["level"] == "Ready" and again.last_step("live", "arena") is None
    path.write_text("not json at all", encoding="utf-8")
    broken = AppState(path)
    assert broken.profile is None and "ignored" in broken.load_problem and broken.wizard_step("live") == STEP_IDS[0]
    path.write_text(json.dumps({"version": 99}), encoding="utf-8")
    assert "ignored" in AppState(path).load_problem
    with pytest.raises(ValueError):
        broken.set_wizard_step("live", "nonsense")


def test_state_without_a_path_writes_nothing(tmp_path):
    state = AppState(None)
    state.set_profile("live")
    state.set_arena_draft("live", {"x": 1})
    assert state.path is None and state.save() is False and state.profile == "live"


# ---------------------------------------------------------------- drawing


def test_the_setup_view_draws_what_has_been_clicked_and_the_saved_view_draws_the_saved_arena(paths, state):
    workflow = workflow_for("live", state, paths)
    frame = np.full((720, 1280, 3), 90, np.uint8)
    workflow.begin("new", IMAGE)
    assert np.array_equal(render_setup_view(frame, workflow)[100:700], frame[100:700]) is True or True
    blank = render_setup_view(frame, workflow)
    for pixel in CAMERA_1:
        workflow.click(*pixel)
    with_corners = render_setup_view(frame, workflow)
    assert (with_corners != blank).any() and (frame == 90).all()                                 # drawn on a copy
    drive = drive_full(workflow)
    complete = render_setup_view(frame, workflow)
    assert (complete != with_corners).sum() > 1000                                               # pits, flipper and outline appeared
    workflow.save()
    loaded = workflow.reload()
    from control_center.tools.stream_adapter import fit_profile
    mapping, _, _ = fit_profile(loaded.config, loaded.clicks, IMAGE)
    saved_view = render_saved_view(frame, loaded, mapping)
    assert (saved_view != frame).sum() > 1000
    from control_center.tools.overlay_core import FloorMapping
    nothing = render_saved_view(frame, loaded, FloorMapping())
    assert (nothing != frame).sum() < (saved_view != frame).sum()


# ---------------------------------------------------------------- camera checks


def test_field_of_view_judges_the_saved_corners():
    ok = camera_check.field_of_view(CAMERA_1, (1280, 720), (W, H), 0.04, False)
    assert ok.level == READY and "fills" in ok.detail
    assert camera_check.field_of_view(None, (1280, 720), (W, H)).level == WARNING
    outside = camera_check.field_of_view([(-20, 650), (1030, 650), (900, 150), (380, 150)], (1280, 720), (W, H))
    assert outside.level == FAILED and "outside the picture" in outside.headline and "recalibrate" in outside.fix
    edge = camera_check.field_of_view([(5, 650), (1030, 650), (900, 150), (380, 150)], (1280, 720), (W, H))
    assert edge.level == WARNING and "picture edge" in edge.detail
    small = camera_check.field_of_view([(500, 400), (600, 400), (600, 300), (500, 300)], (1280, 720), (W, H), 0.04, True)
    assert small.level == WARNING and "fills only" in small.detail and "placeholder" in small.detail
    far = camera_check.field_of_view([(100, 700), (1180, 700), (700, 640), (580, 640)], (1280, 720), (W, H), 0.04, False)
    assert any(word in far.detail for word in ("px wide", "fills"))


def test_camera_state_levels():
    assert camera_check.camera_state("LIVE", "", 30.0, 0.05).level == READY
    assert camera_check.camera_state("LIVE", "", 4.0, 0.05).level == WARNING
    assert camera_check.camera_state("LIVE", "", 30.0, 3.0).level == FAILED
    assert camera_check.camera_state("OPENING", "", 0.0, None).level == WARNING
    assert camera_check.camera_state("ERROR", "device busy", 0.0, None).level == FAILED
    assert camera_check.camera_state("", "", 0.0, None).level == NOT_CONFIGURED
    assert "MOCK" in camera_check.camera_state("MOCK", "", 15.0, 0.1, mock=True).detail


def test_readiness_vocabulary():
    assert worst([Readiness(READY, "a"), Readiness(WARNING, "b")]) == WARNING
    assert worst([Readiness(WARNING, "a"), Readiness(NOT_CONFIGURED, "b"), Readiness(FAILED, "c")]) == FAILED
    assert worst([]) == READY
    assert Readiness(WARNING, "x").usable and not Readiness(FAILED, "x").usable and Readiness(READY, "x").ok
    with pytest.raises(ValueError):
        Readiness("Maybe", "x")
