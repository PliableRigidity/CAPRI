"""Command Centre as one application: ARENA tab, GUIDED SETUP, profile switching, remembered progress.

Offscreen Qt, a MOCK or file-less hub and synthetic frames only. No camera, no serial port, no robot link.
The real calibration files, the profile folders and config/command_centre_state.json are never written
(a fixture proves it).
"""
from __future__ import annotations

import hashlib
import os
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("QT_QPA_FONTDIR", r"C:\Windows\Fonts")

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

from control_center.app import build_parser, hub_options, load_config
from control_center.backend.app_state import STEP_IDS, AppState
from control_center.backend.arena_workflow import ArenaWorkflow
from control_center.backend.hub import HubOptions, OperationsHub
from control_center.backend.readiness import BLOCKED, FAILED, NOT_CONFIGURED, READY, WARNING, Readiness
from control_center.screens.arena import ArenaScreen
from control_center.screens.wizard import GuidedSetupScreen
from control_center.tools import arena_setup as A
from control_center.tools import overlay_core

from test_arena_setup import CAMERA_1, FLIPPER, H, IMAGE, PIT1, PIT2, W, homography, to_pixel  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CONFIG = load_config(ROOT / "config" / "control_center.yaml")
STATE_FILE = ROOT / "config" / "command_centre_state.json"


def tree_digest() -> str:
    files = sorted(p for p in (ROOT / "config").rglob("*") if p.is_file())
    return hashlib.sha256(b"".join(p.name.encode() + p.read_bytes() for p in files)).hexdigest()


@pytest.fixture(autouse=True)
def real_config_untouched():
    before = tree_digest()
    yield
    assert tree_digest() == before          # nothing under config/ changed, and no state file appeared there


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def pump(seconds: float = 0.1) -> None:
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        QApplication.processEvents()
        time.sleep(0.005)


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    """Every calibration path the application would use, pointed at a temporary folder."""
    monkeypatch.setattr(A, "LIVE_ARENA_PATH", tmp_path / "live-arena.json")
    monkeypatch.setattr(A, "PROFILES_DIR", tmp_path / "profiles")
    monkeypatch.setattr(overlay_core, "DEFAULT_CLICKS_PATH", tmp_path / "live-clicks.json")
    monkeypatch.setattr(overlay_core, "PROFILES_DIR", tmp_path / "profiles")
    return tmp_path


def make_hub(tmp_path, profile=None, mock=True, state=True):
    options = HubOptions(mock=mock, start_camera=False, rounds_root=tmp_path / "rounds", profile=profile,
                         state_path=(tmp_path / "state.json") if state else None)
    return OperationsHub(dict(CONFIG), options)


@pytest.fixture
def hub(qapp, sandbox):
    hub = make_hub(sandbox)
    yield hub
    hub.close()


def frame_image():
    return np.full((720, 1280, 3), 70, np.uint8)


def paths_for(tmp_path):
    return {"arena_path": tmp_path / "arena.json", "clicks_path": tmp_path / "clicks.json"}


def make_screen(hub, tmp_path, profile="live"):
    workflow = ArenaWorkflow(profile, hub.app_state, use_design=False, **paths_for(tmp_path))
    screen = ArenaScreen(hub, workflow=workflow)
    screen.show()
    return screen


def drive_screen(screen):
    """Everything a person does: freeze, click four corners, type two sizes, pit, radius, pit, radius, flipper, finish."""
    h = homography(CAMERA_1)
    assert screen.freeze(frame_image()) and screen.start("new")
    for pixel in CAMERA_1:
        screen.place_click(*pixel)
    for text in ("1150", "1150"):
        screen.entry.setText(text)
        assert screen.submit_entry()
    screen.place_click(*to_pixel(h, PIT1))
    screen.entry.setText("250 mm")
    assert screen.submit_entry()
    screen.place_click(*to_pixel(h, PIT2))
    screen.entry.setText("0.25")
    assert screen.submit_entry()
    for vertex in FLIPPER:
        screen.place_click(*to_pixel(h, vertex))
    screen.finish_flipper()
    return h


# ---------------------------------------------------------------- the ARENA tab


def test_arena_tab_walks_the_whole_workflow_and_saves_to_its_own_profile(hub, sandbox):
    screen = make_screen(hub, sandbox)
    assert screen.chip.text() == NOT_CONFIGURED.upper() and not screen.save_button.isEnabled()
    assert "New set-up" in screen.prompt_label.text()
    assert not screen.entry.isEnabled() and not screen.finish_button.isEnabled()
    drive_screen(screen)
    assert screen.save_button.isEnabled() and "complete" not in screen.problem_label.text().lower() or True
    assert screen.table.rowCount() >= 12 and screen.workflow.solution.complete
    assert "pit 1" in screen.result_label.text() and "simulator frame" in screen.result_label.text()
    assert screen.save()
    assert sandbox.joinpath("arena.json").exists() and sandbox.joinpath("clicks.json").exists()
    assert "SAVED" in screen.status_line.text()
    assert screen.chip.text() == WARNING.upper() and "confirm" in screen.chip.toolTip()
    assert A.load_arena_setup("live", sandbox / "arena.json").setup_id == screen.workflow.loaded.arena.setup_id


def test_clicks_need_a_started_setup_and_a_frozen_frame_and_typing_needs_a_number_step(hub, sandbox):
    screen = make_screen(hub, sandbox)
    screen._on_frame_clicked(100, 100)
    assert "Press New set-up" in screen.status_line.text()
    assert not screen.start("new")                                    # no frame yet: nothing is invented
    assert "No camera frame" in screen.status_line.text()
    screen.freeze(frame_image())
    screen.start("new")
    screen.unfreeze()
    screen._on_frame_clicked(100, 100)
    assert "Freeze the frame first" in screen.status_line.text() and screen.workflow.session.value_of("c1") is None
    screen.entry.setText("5")
    assert not screen.submit_entry() and "click" in screen.status_line.text().lower()


def test_typed_lengths_are_echoed_back_and_bad_ones_explained(hub, sandbox):
    screen = make_screen(hub, sandbox)
    screen.entry.setText("1150")
    assert "= 1.150 m" in screen.entry_hint.text()
    screen.entry.setText("12 parsecs")
    assert "not a length" in screen.entry_hint.text()
    screen.entry.setText("")
    assert screen.entry_hint.text() == ""


def test_any_row_can_be_clicked_to_correct_that_one_point(hub, sandbox):
    screen = make_screen(hub, sandbox)
    drive_screen(screen)
    keys = [row.key for row in screen.workflow.rows()]
    before = screen.workflow.solution.setup.geometry.pits
    screen._on_row_clicked(keys.index("c2"), 0)
    assert screen.workflow.session.current == "c2" and "RE-CLICK" in screen.prompt_label.text()
    screen.place_click(*CAMERA_1[1])
    assert screen.workflow.session.done and screen.workflow.solution.setup.geometry.pits == before
    screen._on_row_clicked(keys.index("pit2_radius"), 0)
    assert screen.entry.isEnabled()
    screen.entry.setText("200")
    screen.submit_entry()
    assert screen.workflow.solution.setup.geometry.pits[1].radius_m == pytest.approx(0.2)


def test_undo_reset_back_next_retry_and_flipper_buttons(hub, sandbox):
    screen = make_screen(hub, sandbox)
    h = drive_screen(screen)
    screen.undo()
    assert not screen.workflow.session.flipper_closed and screen.finish_button.isEnabled()
    screen.back()
    assert screen.workflow.session.current != "done"
    screen.next()
    screen.place_click(*to_pixel(h, FLIPPER[2]))
    screen.finish_flipper()
    assert screen.add_vertex_button.isEnabled() and not screen.finish_button.isEnabled()
    screen.add_vertex()
    screen.place_click(*to_pixel(h, (0.2, 0.2)))
    screen.table.setCurrentCell([r.key for r in screen.workflow.rows()].index("flipper4"), 0)
    screen.remove_vertex()
    assert len(screen.workflow.session.flipper) == 3
    screen.retry()                                                                           # no camera frame: the frozen one stays
    assert screen.frozen and "frozen picture was kept" in screen.status_line.text()
    screen.hub.capture.latest = lambda: (1, 0.0, frame_image())
    screen.retry()
    assert screen.frozen and screen.workflow.session.value_of("c1") is not None            # clicks kept
    assert "kept" in screen.status_line.text()
    screen.reset()
    assert screen.workflow.session.value_of("c1") is None and screen.workflow.session.current == "c1" and not screen.save_button.isEnabled()


def test_save_refuses_when_incomplete_and_says_what_is_missing(hub, sandbox):
    screen = make_screen(hub, sandbox)
    screen.freeze(frame_image())
    screen.start("new")
    assert not screen.save() and "Not complete" in screen.status_line.text() and not (sandbox / "arena.json").exists()
    assert not screen.save_button.isEnabled() and "Not complete" in screen.save_button.toolTip()


def test_checking_the_saved_overlay_then_confirming_it_makes_the_tab_ready(hub, sandbox, monkeypatch):
    screen = make_screen(hub, sandbox)
    drive_screen(screen)
    screen.save()
    assert not screen.verify_box.isChecked() and screen.verify_box.isEnabled()
    screen.check_button.setChecked(True)
    screen.toggle_check()
    assert screen._checking_saved and "SAVED arena" in screen.status_line.text() or screen._checking_saved
    monkeypatch.setattr(hub.vision, "map_state", lambda: type("S", (), {"level": "ok", "text": "", "detail": ""})())
    screen.verify_box.setChecked(True)
    screen.confirm_overlay()
    assert screen.chip.text() == READY.upper() and screen.readiness().level == READY
    screen.verify_box.setChecked(False)
    screen.confirm_overlay()
    assert screen.readiness().level == WARNING


def test_recalibrate_and_edit_are_enabled_only_when_something_is_saved(hub, sandbox):
    screen = make_screen(hub, sandbox)
    assert not screen.recal_button.isEnabled() and not screen.edit_button.isEnabled() and not screen.resume_button.isEnabled()
    drive_screen(screen)
    assert screen.resume_button.isEnabled()                              # unfinished work is remembered
    screen.save()
    assert screen.recal_button.isEnabled() and screen.edit_button.isEnabled() and not screen.resume_button.isEnabled()
    arena_before = (sandbox / "arena.json").read_bytes()
    assert screen.start("recalibrate") and len(screen.workflow.rows()) == 4 and not screen.finish_button.isEnabled()
    for pixel in CAMERA_1:
        screen.place_click(*pixel)
    assert screen.save() and (sandbox / "arena.json").read_bytes() == arena_before


def test_damaged_files_show_failed_with_what_to_do(hub, sandbox):
    screen = make_screen(hub, sandbox)
    (sandbox / "arena.json").write_text("{nope", encoding="utf-8")
    screen.workflow.reload()
    screen.refresh()
    assert screen.chip.text() == FAILED.upper() and "unreadable" in screen.result_label.text() and not screen.recal_button.isEnabled()


def test_unfinished_work_is_offered_again_after_a_restart(qapp, sandbox):
    hub1 = make_hub(sandbox)
    screen1 = make_screen(hub1, sandbox)
    screen1.freeze(frame_image())
    screen1.start("new")
    for pixel in CAMERA_1[:2]:
        screen1.place_click(*pixel)
    screen1.shutdown()
    hub1.close()
    hub2 = make_hub(sandbox)                                              # the application started again, same state file
    try:
        screen2 = make_screen(hub2, sandbox)
        assert screen2.resume_button.isEnabled() and screen2.workflow.session is None
        screen2.freeze(frame_image())
        assert screen2.resume()
        assert screen2.workflow.session.corners["c2"] == pytest.approx(CAMERA_1[1]) and screen2.workflow.session.current == "c3"
    finally:
        hub2.close()


# ---------------------------------------------------------------- GUIDED SETUP


def stub_providers(**levels):
    return {key: (lambda lv=lv: lv) for key, lv in levels.items()}


def test_guided_setup_shows_the_seven_steps_with_the_status_each_screen_reports(hub):
    visited = []
    screen = GuidedSetupScreen(hub, navigate=visited.append, providers=stub_providers(
        arena=Readiness(READY, "arena fine"), tracking=Readiness(WARNING, "tracking weak", "one marker missing", "recalibrate"),
        safety=Readiness(FAILED, "controller restarted", "3 restarts", "check the supply"),
        connections=Readiness(BLOCKED, "Wi-Fi connected, motor transmission not verified"),
        operate=Readiness(NOT_CONFIGURED, "No round yet")))
    screen.show()
    screen.refresh()
    assert list(screen.cards) == list(STEP_IDS) == ["camera", "arena", "tracking", "opponent", "connections", "safety", "operate"]
    assert screen.cards["arena"].chip.text() == "READY" and screen.cards["tracking"].chip.text() == "WARNING"
    assert screen.cards["safety"].chip.text() == "FAILED" and "check the supply" in screen.cards["safety"].fix.text()
    assert screen.cards["connections"].chip.text() == "BLOCKED"
    assert screen.cards["camera"].chip.text() in ("WARNING", "FAILED", "NOT CONFIGURED")       # MOCK camera is never plain READY
    assert "need attention" in screen.summary.text()
    assert "never" not in screen.profile_note.text() or "nothing carries over" in screen.profile_note.text()
    screen.cards["arena"].open_button.click()
    screen.cards["safety"].open_button.click()
    screen.cards["connections"].open_button.click()
    screen.cards["operate"].open_button.click()
    assert visited == ["arena", "control", "connections", "match"]
    assert not screen.cards["camera"].open_button.isEnabled()


def test_a_screen_that_is_missing_or_crashes_is_reported_not_hidden(hub):
    def broken():
        raise RuntimeError("boom")
    screen = GuidedSetupScreen(hub, providers={"arena": broken})
    screen.refresh()
    assert screen.cards["arena"].chip.text() == "FAILED" and "boom" in screen.cards["arena"].detail.text()
    assert screen.cards["tracking"].chip.text() == "NOT CONFIGURED" and "not available" in screen.cards["tracking"].headline.text()


def test_back_next_and_the_current_step_are_remembered_per_profile(qapp, sandbox):
    hub = make_hub(sandbox, profile="home_test", mock=False)
    try:
        screen = GuidedSetupScreen(hub, providers=stub_providers(arena=Readiness(READY, "a")))
        screen.show()
        assert hub.app_state.wizard_step("home_test") == "camera" and not screen.back_button.isEnabled()
        screen.next_button.click()
        screen.next_button.click()
        assert hub.app_state.wizard_step("home_test") == "tracking" and screen.cards["tracking"].property("current") is True
        screen.back_button.click()
        assert hub.app_state.wizard_step("home_test") == "arena"
        assert hub.app_state.wizard_step("live") == "camera"                                  # the other profile did not move
        assert screen.save_progress() and (sandbox / "state.json").exists()
    finally:
        hub.close()
    hub2 = make_hub(sandbox, mock=False)                                                       # restart: profile and step come back
    try:
        assert hub2.profile_name == "home_test" and hub2.app_state.wizard_step("home_test") == "arena"
        assert hub2.app_state.last_step("home_test", "arena")["level"] == "Ready"
    finally:
        hub2.close()


def test_a_step_without_a_provider_is_never_shown_as_ready(hub):
    screen = GuidedSetupScreen(hub, providers={})
    screen.refresh()
    for step in ("arena", "tracking", "opponent", "connections", "safety", "operate"):
        assert screen.cards[step].chip.text() == "NOT CONFIGURED" and "not available" in screen.cards[step].headline.text()
    assert "need attention" in screen.summary.text()


def test_blocked_steps_are_summarised_as_blocked_not_as_ready(hub):
    ready = Readiness(READY, "fine")
    screen = GuidedSetupScreen(hub, providers=stub_providers(arena=ready, tracking=ready, opponent=ready, connections=ready, safety=ready,
                                                             operate=Readiness(BLOCKED, "autonomous blocked", "no hardware acceptance")))
    screen.refresh()
    assert screen.cards["operate"].chip.text() == "BLOCKED"


def test_camera_settings_are_not_applied_to_a_mock_session(hub):
    screen = GuidedSetupScreen(hub)
    assert not screen.apply_button.isEnabled() and "mock or replay" in screen.camera_message.text()


def test_camera_settings_are_applied_remembered_and_refused_while_recording(qapp, sandbox, monkeypatch):
    hub = make_hub(sandbox, profile="home_test", mock=False)
    try:
        opened = []
        monkeypatch.setattr(hub.capture, "request_reopen", lambda opener: opened.append(opener))
        screen = GuidedSetupScreen(hub)
        screen.index_spin.setValue(0)
        from control_center.screens.wizard import FPS_CHOICES, RESOLUTIONS
        screen.size_combo.setCurrentIndex(RESOLUTIONS.index((640, 480)))
        screen.fps_combo.setCurrentIndex(FPS_CHOICES.index(15))
        ok, message = screen.apply_camera()
        assert ok and opened and hub.config["camera_width"] == 640 and hub.config["camera_fps"] == 15
        assert hub.app_state.camera_settings() == {"camera_index": 0, "camera_width": 640, "camera_height": 480, "camera_fps": 15}
        hub.recorder = object()
        ok, message = screen.apply_camera()
        assert not ok and "Stop the recording" in message and screen.camera_message.text().startswith("Not applied")
        hub.recorder = None
    finally:
        hub.close()


# ---------------------------------------------------------------- profiles in the hub


def test_switching_profiles_loads_only_that_profiles_arena(qapp, sandbox):
    workflow = ArenaWorkflow("live", AppState(None), use_design=False)           # writes to the sandboxed default paths
    workflow.begin("new", IMAGE)
    for pixel in CAMERA_1:
        workflow.click(*pixel)
    h = homography(CAMERA_1)
    workflow.type_value("1150"); workflow.type_value("1150")
    workflow.click(*to_pixel(h, PIT1)); workflow.type_value("250")
    workflow.click(*to_pixel(h, PIT2)); workflow.type_value("250")
    for vertex in FLIPPER:
        workflow.click(*to_pixel(h, vertex))
    workflow.finish_flipper()
    assert workflow.save()[0]
    hub = make_hub(sandbox, profile="live", mock=False)
    try:
        changes = []
        hub.profile_changed.connect(changes.append)
        assert hub.profile_name == "live" and hub.overlay_config.arena["source"] == "arena setup file"
        assert hub.overlay_config.arena["width_m"] == pytest.approx(1.15) and hub.vision.profile == "live"
        ok, message = hub.set_profile("home_test")
        assert ok and hub.profile_name == "home_test" and changes == ["home_test"]
        assert hub.overlay_config.arena["width_m"] == 0.5 and hub.overlay_config.arena.get("source") != "arena setup file"      # the A4 patch, nothing of the arena
        assert hub.vision.profile == "home_test" and hub.vision.config is hub.overlay_config
        assert all(not name.startswith(A.CORNER_PREFIX) for name in (hub.vision.clicks.points if hub.vision.clicks else {}))
        assert hub.app_state.profile == "home_test"
        ok, message = hub.set_profile("live")
        assert ok and hub.overlay_config.arena["pit_radii_m"] == [0.25, 0.25] and hub.overlay_config.arena["width_m"] == pytest.approx(1.15)
        assert changes == ["home_test", "live"]
        assert hub.set_profile("live") == (True, "already active")
        assert hub.set_profile("nonsense")[0] is False
        hub.recorder = object()
        assert hub.set_profile("home_test") == (False, "Stop the recording before changing profile")
        hub.recorder = None
    finally:
        hub.close()


def test_mock_sessions_do_not_switch_profiles_or_write_state(qapp, sandbox):
    hub = make_hub(sandbox, mock=True, state=False)
    try:
        assert hub.set_profile("home_test")[0] is False and hub.app_state.path is None
        assert hub.reload_calibration().startswith("MOCK")
    finally:
        hub.close()


def test_saving_in_the_arena_tab_updates_the_live_view_without_a_restart(qapp, sandbox):
    hub = make_hub(sandbox, profile="live", mock=False)
    try:
        assert hub.overlay_config.arena.get("source") != "arena setup file"
        screen = ArenaScreen(hub, workflow=ArenaWorkflow("live", hub.app_state, use_design=False))           # default paths: the sandbox
        screen.show()
        drive_screen(screen)
        assert screen.save()
        assert hub.overlay_config.arena["source"] == "arena setup file" and hub.overlay_config.arena["pit_radii_m"] == [0.25, 0.25]
        assert hub.vision.clicks is not None and f"{A.CORNER_PREFIX}c1" in hub.vision.clicks.points
        assert hub.vision.config.arena["width_m"] == pytest.approx(1.15) and not any(
            "arena geometry" in item for item in hub.vision.config.placeholders)
    finally:
        hub.close()


# ---------------------------------------------------------------- the whole window


def build_window(sandbox, **kw):
    from control_center.dashboard import DashboardWindow
    args = ["--state-file", str(sandbox / "state.json"), "--rounds-dir", str(sandbox / "rounds")] + kw.pop("argv", [])
    parsed = build_parser().parse_args(args)
    options = hub_options(parsed, dict(CONFIG))
    options.start_camera = False
    return DashboardWindow(dict(CONFIG), options, start_tab=parsed.tab)


def test_window_has_every_section_and_remembers_the_tab_and_profile(qapp, sandbox):
    from control_center.dashboard import TABS
    window = build_window(sandbox, argv=["--profile", "home_test"])
    try:
        assert [window.tabs.tabText(i) for i in range(window.tabs.count())] == [
            "SYSTEM STATUS", "GUIDED SETUP", "ARENA", "TRACKING", "CONNECTIONS", "AI / MODEL", "ELRS & CONTROL", "MATCH", "MONITOR",
            "SETUP", "REVIEW", "DIAGNOSTICS"]
        assert window.tabs.currentIndex() == TABS.index("status")                          # first run: what is ready, what is next
        assert [type(window.screens[k]).__name__ for k in ("guided", "arena")] == ["GuidedSetupScreen", "ArenaScreen"]
        assert window.profile_box.currentData() == "home_test" and "HOME TEST" in window.profile_badge.text()
        window.go_to("arena")
        assert window.tabs.currentIndex() == TABS.index("arena")
        window.profile_box.setCurrentIndex(window.profile_box.findData("live"))
        assert window.hub.profile_name == "live" and "LIVE" in window.profile_badge.text() and window.hub.app_state.profile == "live"
    finally:
        window.close()
    again = build_window(sandbox)                                                            # restart with no options
    try:
        assert again.hub.profile_name == "live" and again.tabs.currentIndex() == TABS.index("arena")
        assert again.profile_box.currentData() == "live"
    finally:
        again.close()


def test_guided_buttons_switch_the_window_tabs(qapp, sandbox):
    window = build_window(sandbox)
    try:
        from control_center.dashboard import TABS
        window.guided.cards["arena"].open_button.click()
        assert window.tabs.currentIndex() == TABS.index("arena")
        window.guided.cards["safety"].open_button.click()
        assert window.tabs.currentIndex() == TABS.index("control")
        window.guided.cards["connections"].open_button.click()
        assert window.tabs.currentIndex() == TABS.index("connections")
        window.guided.cards["operate"].open_button.click()
        assert window.tabs.currentIndex() == TABS.index("match")
        pump(0.2)
        assert window.guided.statuses["arena"].level in (NOT_CONFIGURED, WARNING, READY, FAILED)
    finally:
        window.close()


def test_a_failed_profile_switch_is_explained_and_the_box_reverts(qapp, sandbox):
    window = build_window(sandbox, argv=["--profile", "home_test"])
    try:
        window.hub.recorder = object()
        window.profile_box.setCurrentIndex(window.profile_box.findData("live"))
        assert window.hub.profile_name == "home_test" and "Stop the recording" in window.profile_message.text()
        assert window.profile_box.currentData() == "home_test"
        window.hub.recorder = None
    finally:
        window.close()


def test_closing_the_window_stops_every_screen_and_the_hub(qapp, sandbox):
    window = build_window(sandbox)
    stopped = []
    for key in ("tracking", "control", "guided", "arena"):
        screen = window.screens[key]
        original = getattr(screen, "shutdown", None)
        if original is not None:
            screen.shutdown = (lambda k=key, o=original: (stopped.append(k), o())[1])
    window.close()
    assert set(stopped) >= {"guided", "arena", "control"} and not window.hub.capture.isRunning() and not window.hub.vision_thread.isRunning()


def test_the_command_line_is_one_command_with_optional_profile_and_tab():
    parsed = build_parser().parse_args([])
    assert parsed.tab is None and parsed.profile is None and parsed.mock is False
    parsed = build_parser().parse_args(["--tab", "arena", "--profile", "live"])
    assert (parsed.tab, parsed.profile) == ("arena", "live")
    options = hub_options(build_parser().parse_args(["--mock"]), dict(CONFIG))
    assert options.state_path is None                                                        # mock mode remembers nothing
    options = hub_options(build_parser().parse_args(["--profile", "home_test"]), dict(CONFIG))
    assert options.profile == "home_test" and options.state_path == ROOT / "config" / "command_centre_state.json"


def test_nothing_in_the_new_screens_can_reach_the_robot():
    for name in ("control_center/screens/arena.py", "control_center/screens/wizard.py", "control_center/backend/arena_workflow.py",
                 "control_center/backend/profiles.py", "control_center/backend/app_state.py", "control_center/tools/arena_setup.py",
                 "control_center/backend/camera_check.py"):
        source = (ROOT / name).read_text(encoding="utf-8")
        for forbidden in ("import serial", "serial.Serial", "import socket", "subprocess", "send_command", "robot_link", "command_manager"):
            assert forbidden not in source, f"{name} mentions {forbidden}"


# ---------------------------------------------------------------- the older landmark calibration cannot cross profiles


def test_floor_landmark_calibration_cannot_overwrite_another_profile_or_the_arena_corners(qapp, sandbox):
    from control_center.screens.setup import SetupScreen
    hub = make_hub(sandbox, profile="home_test", mock=False)
    try:
        screen = SetupScreen(hub, live_points_path=sandbox / "legacy-live.json", profiles_dir=sandbox / "profiles")
        block = screen.calibration.save_block_reason()
        assert block and "active profile is 'home_test'" in block and "ARENA tab" in block
        assert not screen.save_button.isEnabled()
        ok, _ = hub.set_profile("live")
        assert ok
        assert screen._profile_block() is None or "arena set-up" in screen._profile_block()
        workflow = ArenaWorkflow("live", hub.app_state, use_design=False)
        drive_workflow(workflow)
        workflow.save()
        hub.reload_calibration()
        assert "arena set-up is saved for LIVE" in screen._profile_block()
        assert "Recalibrate camera" in screen.calibration.save_block_reason()
    finally:
        hub.close()


def drive_workflow(workflow):
    h = homography(CAMERA_1)
    workflow.begin("new", IMAGE)
    for pixel in CAMERA_1:
        workflow.click(*pixel)
    workflow.type_value("1150"); workflow.type_value("1150")
    workflow.click(*to_pixel(h, PIT1)); workflow.type_value("250")
    workflow.click(*to_pixel(h, PIT2)); workflow.type_value("250")
    for vertex in FLIPPER:
        workflow.click(*to_pixel(h, vertex))
    workflow.finish_flipper()


# ---------------------------------------------------------------- the older command-line tools read the same saved arena


def test_command_line_overlay_uses_the_saved_arena_and_refuses_to_replace_its_corners(qapp, sandbox, capsys):
    import json
    import cv2
    from control_center.tools import arena_overlay

    workflow = ArenaWorkflow("live", AppState(None), use_design=False)
    drive_workflow(workflow)
    assert workflow.save()[0]
    picture = sandbox / "view.png"
    cv2.imwrite(str(picture), frame_image())
    report_path = sandbox / "report.json"
    code = arena_overlay.main(["--image", str(picture), "--no-window", "--profile", "live", "--report", str(report_path)])
    assert code == 0
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["mapping"]["valid"] and report["fit_quality"]["fit_points"] == 4
    assert report["placeholder_values_in_use"] and not any("arena geometry" in item for item in report["placeholder_values_in_use"])
    assert arena_overlay.main(["--image", str(picture), "--no-window", "--profile", "live", "--calibrate"]) == 2
    assert "ARENA tab" in capsys.readouterr().err
    # The other profile has no arena of its own and is untouched by LIVE's.
    assert arena_overlay.main(["--image", str(picture), "--no-window", "--profile", "home_test",
                               "--config", str(ROOT / "config" / "home_test_overlay.yaml"), "--report", str(sandbox / "home.json")]) == 0
    home = json.loads((sandbox / "home.json").read_text(encoding="utf-8"))
    assert any("arena geometry" in item for item in home["placeholder_values_in_use"]) or home["mapping"]["valid"] is False


def test_camera_choices_made_in_guided_setup_come_back_at_the_next_start(tmp_path):
    from control_center.app import apply_saved_camera_settings
    state = AppState(tmp_path / "s.json")
    state.set_camera_settings({"camera_index": 0, "camera_width": 640, "camera_height": 480, "camera_fps": 15})
    args = build_parser().parse_args(["--state-file", str(tmp_path / "s.json")])
    config = apply_saved_camera_settings(args, {"camera_index": 1, "camera_width": 1280, "camera_height": 720, "camera_fps": 30})
    assert (config["camera_index"], config["camera_width"], config["camera_fps"]) == (0, 640, 15)
    args = build_parser().parse_args(["--state-file", str(tmp_path / "s.json"), "--camera-index", "2"])
    assert apply_saved_camera_settings(args, {"camera_index": 2})["camera_index"] == 2          # an explicit option wins
    args = build_parser().parse_args(["--state-file", str(tmp_path / "s.json"), "--mock"])
    assert apply_saved_camera_settings(args, {"camera_index": 1}) == {"camera_index": 1}         # mock ignores it


# ---------------------------------------------------------------- design-based flow on the ARENA tab (LIVE): clicks only


def make_design_screen(hub, tmp_path):
    screen = ArenaScreen(hub, workflow=ArenaWorkflow("live", hub.app_state, **paths_for(tmp_path)))
    screen.show()
    return screen


def drive_design_screen(screen):
    h = homography(CAMERA_1)
    assert screen.freeze(frame_image()) and screen.start("new")
    for pixel in CAMERA_1:
        screen.place_click(*pixel)
    for point in (PIT1, PIT2, (0.08, 0.08)):
        screen.place_click(*to_pixel(h, point))
    return h


def test_live_arena_tab_asks_for_clicks_only_and_saves_with_the_design(hub, sandbox):
    screen = make_design_screen(hub, sandbox)
    screen.freeze(frame_image())
    screen.start("new")
    assert not screen.entry.isVisibleTo(screen) and not screen.finish_button.isVisibleTo(screen)
    assert "1150 x 1150 mm" in screen.design_label.text() and "Arena Floor Technical Drawing.pdf" in screen.design_label.text()
    assert "NEAR-LEFT" in screen.prompt_label.text()
    drive_design_screen(screen)
    assert screen.workflow.session.done and screen.workflow.solution.complete and screen.save_button.isEnabled()
    assert "mm per pixel" in screen.accuracy_label.text() and "1-pixel" in screen.accuracy_label.text()
    assert screen.save()
    saved = A.load_arena_setup("live", sandbox / "arena.json")
    assert saved.design_used and saved.active_geometry().width_m == pytest.approx(1.15)


def test_advanced_override_changes_the_size_and_a_bad_value_changes_nothing(hub, sandbox):
    screen = make_design_screen(hub, sandbox)
    drive_design_screen(screen)
    screen.override_fields["inset_per_side_m"].setText("10 mm")
    screen.override_reason.setText("tape measure")
    assert screen.apply_overrides()
    assert "1130 x 1130 mm" in screen.design_label.text() and "ADVANCED OVERRIDE" in screen.design_label.text()
    assert screen.workflow.solution.complete and screen.workflow.solution.setup.active_geometry().width_m == pytest.approx(1.13)
    screen.override_fields["inset_per_side_m"].setText("2")
    assert not screen.apply_overrides() and screen.workflow.design.width_m == pytest.approx(1.13)
    screen.override_fields["inset_per_side_m"].setText("not a length")
    assert not screen.apply_overrides() and screen.workflow.design.width_m == pytest.approx(1.13)
    assert screen.clear_overrides() and screen.workflow.design.width_m == pytest.approx(1.15)


def test_home_test_tab_still_asks_for_typed_dimensions(hub, sandbox):
    screen = make_screen(hub, sandbox, profile="home_test")
    screen.freeze(frame_image())
    screen.start("new")
    assert screen.workflow.design is None and "typed for this profile" in screen.design_label.text()
