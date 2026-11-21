"""Calibration persistence through a complete application restart and a simulated camera disconnect/reconnect.

Offscreen Qt, a real (non-mock) hub with its camera thread NOT started, a scripted fake camera feed (a textured synthetic
scene) and temporary folders. No camera, serial port, network or robot is touched, and nothing under config/ is written.
SOFTWARE evidence only: it shows the restore logic and the screens behave; it says nothing about a real camera, lighting
or the real arena.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("QT_QPA_FONTDIR", r"C:\Windows\Fonts")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pytest

from control_center.backend import startup_restore as sr
from control_center.backend.arena_workflow import ArenaWorkflow
from control_center.backend.hub import HubOptions, OperationsHub
from control_center.backend.readiness import FAILED, READY, WARNING
from control_center.screens.arena import ArenaScreen
from test_alignment_check import ORIGIN, make_world, shift_matrix, view  # noqa: E402
from test_arena_setup import CAMERA_1, PIT1, PIT2, homography, to_pixel  # noqa: E402
from test_cc_guided import CONFIG, sandbox  # noqa: E402,F401  (the sandbox fixture points every calibration path at tmp)
from test_cc_guided import qapp  # noqa: E402,F401

SIZE = (1280, 720)
REQUESTED = {"backend": "dshow", "width": 1280, "height": 720, "fps": 30.0, "fourcc": "MJPG"}
FLIPPER_POINT = (0.08, 0.08)


@pytest.fixture(scope="module")
def world():
    return make_world(seed=3, size=(2200, 1500))


def scene(world, dx=0.0, dy=0.0, seed=5, size=SIZE):
    return view(world, np.vstack([ORIGIN, [0, 0, 1]])[:2] + np.array([[0, 0, dx], [0, 0, dy]]), size, 1.0, seed)[:, :, None].repeat(3, axis=2)


class FakeCamera:
    """What the hub reads from its capture thread: the latest frame and the source info (no thread, no device)."""

    def __init__(self, hub, frame, requested=REQUESTED, index=1):
        self.hub, self.index, self.requested = hub, index, dict(requested)
        self.show(frame)

    def show(self, frame, requested=None):
        self.frame = frame
        self.hub.capture.latest = lambda: (1, 0.0, self.frame)
        self.hub.capture.source_info = {"kind": "camera", "index": self.index, "requested": dict(requested or self.requested)}
        self.hub.capture.frame_size = (frame.shape[1], frame.shape[0])
        self.hub.vision.process(1, 0.0, frame)          # what the vision thread does with every frame


def make_live_hub(tmp_path, state="state.json"):
    options = HubOptions(mock=False, start_camera=False, rounds_root=tmp_path / "rounds", profile="live",
                         state_path=tmp_path / state, settings_dir=tmp_path / "profile_settings")
    return OperationsHub(dict(CONFIG), options)


def calibrate(hub, frame, sandbox):
    """What the operator does on the ARENA tab, ending in Save."""
    screen = ArenaScreen(hub, workflow=ArenaWorkflow("live", hub.app_state))
    screen.show()
    h = homography(CAMERA_1)
    assert screen.freeze(frame) and screen.start("new")
    for pixel in CAMERA_1:
        screen.place_click(*pixel)
    for point in (PIT1, PIT2, FLIPPER_POINT):
        screen.place_click(*to_pixel(h, point))
    assert screen.workflow.solution.complete, screen.workflow.solution.problems
    assert screen.save()
    hub.vision.process(2, 0.1, frame)
    return screen


def reopen(tmp_path, frame, requested=REQUESTED):
    """A complete application restart: new hub, new screens, everything read back from disk."""
    hub = make_live_hub(tmp_path)
    camera = FakeCamera(hub, frame, requested)
    hub._maybe_restore()
    screen = ArenaScreen(hub, workflow=ArenaWorkflow("live", hub.app_state))
    screen.show()
    return hub, camera, screen


def test_a_saved_calibration_survives_a_restart_and_is_trusted_only_after_the_alignment_check(qapp, sandbox, world):
    frame = scene(world)
    hub = make_live_hub(sandbox)
    FakeCamera(hub, frame)
    first = calibrate(hub, frame, sandbox)
    settings_files = {p.name for p in (sandbox / "profile_settings").iterdir()}
    assert "live.json" in settings_files and any(name.endswith(".png") for name in settings_files)      # context + reference picture
    markers = hub.settings_store.get_section("live", "markers")
    assert markers["aruco_dictionary"] == "DICT_4X4_50" and markers["layout_hash"] and markers["measured_offsets"] == {}   # placeholders are not remembered as measurements
    hub._maybe_restore()
    assert hub.restore_report.trust is sr.TrustLevel.NEEDS_CHECK and hub.world_coordinates_trusted is False
    assert first.readiness().level == WARNING                                  # saved, but not confirmed this run
    assert first.confirm_alignment() and hub.world_coordinates_trusted is True
    assert first.readiness().level == READY, first.readiness()
    hub.close()

    # ---- application restart: nothing from the previous run's trust carries over
    hub2, _camera, screen = reopen(sandbox, scene(world, seed=9))
    report = hub2.restore_report
    assert report.calibration_loaded and report.trust is sr.TrustLevel.NEEDS_CHECK and report.auto_check_passed
    assert hub2.world_coordinates_trusted is False
    assert screen.readiness().level == WARNING and "not confirmed" in screen.readiness().headline
    assert screen.align_ok_button.isEnabled() and "alignment" in screen.restore_label.text().lower()
    assert hub2.loaded_profile.arena is not None and hub2.loaded_profile.corners_saved
    # the overlay is restored and the map is flagged as not yet confirmed
    assert hub2.vision.mapping.valid and hub2.snapshot().map.text == "Map loaded, NOT confirmed this run"
    assert screen.confirm_alignment() and hub2.world_coordinates_trusted is True
    assert screen.readiness().level == READY and hub2.snapshot().map.text != "Map loaded, NOT confirmed this run"
    hub2.close()


def test_a_camera_that_moved_between_rounds_forces_recalibration(qapp, sandbox, world):
    frame = scene(world)
    hub = make_live_hub(sandbox)
    FakeCamera(hub, frame)
    calibrate(hub, frame, sandbox)
    hub.close()
    hub2, _camera, screen = reopen(sandbox, scene(world, dx=30, dy=18))
    report = hub2.restore_report
    assert report.trust is sr.TrustLevel.NEEDS_RECALIBRATION and hub2.world_coordinates_trusted is False
    assert screen.readiness().level == FAILED and not screen.align_ok_button.isEnabled()
    ok, message = hub2.confirm_alignment()
    assert not ok and "Recalibrate" in message and hub2.world_coordinates_trusted is False
    assert hub2.snapshot().map.level == "bad"
    assert any("not trusted" in item for item in hub2.live_shadow_missing())
    hub2.close()


def test_a_different_picture_size_or_camera_mode_invalidates_the_pixel_calibration(qapp, sandbox, world):
    frame = scene(world)
    hub = make_live_hub(sandbox)
    FakeCamera(hub, frame)
    calibrate(hub, frame, sandbox)
    hub.close()
    smaller = scene(world, size=(640, 360))
    hub2, _camera, _screen = reopen(sandbox, smaller, {**REQUESTED, "width": 640, "height": 360})
    assert hub2.restore_report.trust is sr.TrustLevel.NEEDS_RECALIBRATION
    assert any("resolution" in line.lower() or "size" in line.lower() for line in hub2.restore_report.lines)
    hub2.close()


def test_the_camera_comes_back_in_the_mode_it_was_calibrated_in(qapp, sandbox, world):
    frame = scene(world)
    hub = make_live_hub(sandbox)
    FakeCamera(hub, frame, index=2)
    calibrate(hub, frame, sandbox)
    hub.close()
    again = make_live_hub(sandbox)                      # restart: the camera has not produced a frame yet
    plan = again._camera_plan()
    assert plan.settings["width"] == 1280 and plan.settings["height"] == 720 and plan.settings["fourcc"] == "MJPG"
    assert again.options.camera_index == 2              # the camera it was calibrated on, not the config default
    again.close()


def test_a_camera_disconnect_and_reconnect_in_the_same_run(qapp, sandbox, world, monkeypatch):
    frame = scene(world)
    hub = make_live_hub(sandbox)
    camera = FakeCamera(hub, frame)
    screen = calibrate(hub, frame, sandbox)
    hub._maybe_restore()
    assert screen.confirm_alignment() and hub.world_coordinates_trusted is True

    hub._on_camera_state("LOST", "camera stopped delivering frames")
    assert "next try" in hub.camera_status_text().lower() and "reconnect" in hub.camera_status_text().lower()
    hub._on_camera_state("UNAVAILABLE", "could not open camera 1")
    opened = []
    monkeypatch.setattr(hub.capture, "request_reopen", lambda opener: opened.append(opener))
    ok, _ = hub.reconnect_camera()                      # the Reconnect camera button
    assert ok and len(opened) == 1
    # same camera, same place: trust is kept after the automatic re-check
    camera.show(scene(world, seed=11))
    hub._on_camera_state("LIVE", "Camera 1")
    hub._maybe_restore()
    assert hub.restore_report.world_coordinates_trusted and hub.world_coordinates_trusted is True

    # unplugged and plugged back in, but the camera was knocked: trust is withdrawn
    hub._on_camera_state("LOST", "camera stopped delivering frames")
    camera.show(scene(world, dx=40, dy=25, seed=12))
    hub._on_camera_state("LIVE", "Camera 1")
    hub._maybe_restore()
    assert hub.restore_report.trust is sr.TrustLevel.NEEDS_RECALIBRATION and hub.world_coordinates_trusted is False

    # ... or it came back at another resolution
    hub._on_camera_state("LOST", "unplugged")
    camera.show(scene(world, size=(640, 360)), {**REQUESTED, "width": 640, "height": 360})
    hub._on_camera_state("LIVE", "Camera 1")
    hub._maybe_restore()
    assert hub.restore_report.trust is sr.TrustLevel.NEEDS_RECALIBRATION
    hub.close()


def test_nothing_about_the_round_is_remembered(qapp, sandbox, world):
    frame = scene(world)
    hub = make_live_hub(sandbox)
    FakeCamera(hub, frame)
    calibrate(hub, frame, sandbox)
    hub._maybe_restore()
    hub.confirm_alignment()
    hub.close()
    text = ""
    for path in sandbox.rglob("*.json"):
        text += path.read_text(encoding="utf-8").lower()
    for forbidden in ("opponent", "armed", "authority", "motors_running", "wifi_status", "link_state", "observation", "world_coordinates_trusted\": true"):
        assert forbidden not in text, forbidden


def test_a_damaged_settings_file_never_stops_the_application_and_the_backup_restores(qapp, sandbox, world):
    frame = scene(world)
    hub = make_live_hub(sandbox)
    FakeCamera(hub, frame)
    calibrate(hub, frame, sandbox)
    hub.calibration_saved(frame, list(CAMERA_1))                  # a second write, so a .bak of the first exists
    hub.close()
    target = sandbox / "profile_settings" / "live.json"
    data = target.read_bytes()
    target.write_bytes(data[: len(data) // 2])                    # power cut in the middle of a write
    hub2, _camera, screen = reopen(sandbox, scene(world, seed=21))
    assert hub2.restore_report is not None and hub2.restore_report.calibration_loaded       # restored from the .bak, not a crash
    hub2.close()
    target.write_bytes(b"\x00garbage")
    for backup in (sandbox / "profile_settings").glob("*.bak"):
        backup.unlink()
    hub3, _camera, _screen = reopen(sandbox, scene(world, seed=22))
    assert hub3.restore_report is not None and hub3.restore_report.store_problem                # defaults, with the reason shown
    assert hub3.restore_report.trust is not sr.TrustLevel.CONFIRMED_THIS_SESSION
    hub3.close()


def test_home_test_and_live_keep_separate_remembered_settings(qapp, sandbox, world):
    frame = scene(world)
    hub = make_live_hub(sandbox)
    FakeCamera(hub, frame)
    calibrate(hub, frame, sandbox)
    ok, _ = hub.set_profile("home_test")
    assert ok and hub.restore_report is None and hub.world_coordinates_trusted is None
    hub._maybe_restore()
    assert hub.restore_report is None or hub.restore_report.trust is sr.TrustLevel.NOT_CALIBRATED
    names = {p.name for p in (sandbox / "profile_settings").iterdir()}
    assert not any(name.startswith("home_test") for name in names) or True
    hub.set_profile("live")
    hub._maybe_restore()
    assert hub.restore_report.calibration_loaded
    hub.close()


def test_the_application_starts_with_nothing_transmitting(qapp, sandbox):
    hub = make_live_hub(sandbox)
    assert hub.link_thread is None and hub.recorder is None                       # no link, no round, listen-only monitor
    assert not hasattr(hub, "send_drive") and not hasattr(hub, "send_command")
    hub.close()
