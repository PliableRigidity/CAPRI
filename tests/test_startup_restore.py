"""Startup restore: a full restart at the backend level, the trust levels, and camera disconnect / reconnect.

Software simulation only: synthetic pictures, a temporary settings directory, a fake clock, duck-typed loaded
profiles. No camera, serial port or socket is opened and nothing is written under config/. Passing these tests says the
LOGIC is right; it does not show that a real USB camera, floor or lighting behaves as the synthetic scene does.
"""
from __future__ import annotations

import dataclasses
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from control_center.backend import alignment_check as ac
from control_center.backend import camera_signature as cs
from control_center.backend import startup_restore as sr
from control_center.backend.camera_signature import CameraSignature
from control_center.backend.profile_settings import ProfileSettingsStore
from control_center.backend.startup_restore import (CameraReconnectPolicy, ConfirmationRefused, TrustLevel, apply_reconnect_verdict,
                                                    confirm_alignment, record_calibration, record_confirmation, restore_calibration)
from control_center.tools.overlay_core import ClickSet
from test_alignment_check import CORNERS, ORIGIN, compose, make_world, shift_matrix, view

NOW = datetime(2026, 11, 21, 12, 0, 0, tzinfo=timezone.utc)
SETUP_ID = "setup-1150"
SIG = CameraSignature(index=1, device_name="Logitech C525", backend="dshow", width=640, height=360, fps=30.0, fourcc="MJPG",
                      extra={"exposure": -6, "autofocus": False})
CORNER_POINTS = {f"arena_corner_c{i}": p for i, p in enumerate(CORNERS, start=1)}


@pytest.fixture(scope="module")
def world() -> np.ndarray:
    return make_world()


@pytest.fixture(scope="module")
def calibration_frame(world) -> np.ndarray:
    return cv2.cvtColor(view(world, ORIGIN, seed=1), cv2.COLOR_GRAY2BGR)


def frame_with(world, delta=None, seed=2) -> np.ndarray:
    return cv2.cvtColor(view(world, ORIGIN if delta is None else compose(delta), seed=seed), cv2.COLOR_GRAY2BGR)


def loaded_profile(name="live", points=None, size=(640, 360), setup_id=SETUP_ID, arena=True, problem="", reason="", saved_at="2026-11-20T18:00:00+00:00"):
    points = CORNER_POINTS if points is None else points
    clicks = None if points == {} else ClickSet(dict(points), {}, size, {"arena_setup_id": setup_id, "saved_at": saved_at, "profile": name})
    return SimpleNamespace(name=name, clicks=clicks, arena=SimpleNamespace(setup_id=setup_id, saved_at=saved_at) if arena else None,
                           problem=problem, reason=reason, corners_saved=bool(points))


def calibrated_store(tmp_path, world, profile="live", clock_start=NOW - timedelta(hours=18)):
    store = ProfileSettingsStore(tmp_path / "ps", clock=lambda: clock_start)
    frame = frame_with(world, seed=1)
    assert record_calibration(profile, store, SIG, CORNER_POINTS, SETUP_ID, frame, now=clock_start).ok
    return store, frame


def restore(store, frame, signature=SIG, profile="live", **profile_kw):
    return restore_calibration(profile, store, loaded_profile(profile, **profile_kw), signature, frame, now=NOW)


# ------------------------------------------------------------------------------------------ FULL RESTART

def test_full_restart_everything_comes_back_and_trust_must_be_earned_again(tmp_path, world):
    directory = tmp_path / "ps"
    wifi = {"host": "192.168.4.1", "port": 4210, "preferred_ssid": "AutoBots-Bot", "auto_connect": False, "last_endpoint_ok_at": "2026-11-20T18:01:00+00:00"}
    markers = {"aruco_dictionary": "DICT_4X4_50", "self_tag_id": 0, "layout_name": "antweight_v1", "layout_path": "config/arena_overlay.yaml",
               "layout_hash": "cafe1234", "measured_offsets": {"0": {"forward_m": 0.035, "left_m": 0.03, "yaw_deg": 0.0}}, "tag_size_m": None}

    # ---- session 1: operator calibrates, everything is saved
    store = ProfileSettingsStore(directory, clock=lambda: NOW - timedelta(hours=18))
    frame = frame_with(world, seed=1)
    assert sr.save_camera_preferences("live", store, SIG).ok
    assert store.set_section("live", "markers", markers).ok and store.set_section("live", "wifi", wifi).ok
    assert record_calibration("live", store, SIG, CORNER_POINTS, SETUP_ID, frame, now=NOW - timedelta(hours=18)).ok
    first = {name: store.get_section("live", name) for name in ("camera", "calibration_context", "markers", "wifi")}
    first_reference = store.load_reference_frame("live")
    saved_ats = {name: store.section_saved_at("live", name) for name in ("camera", "calibration_context", "markers", "wifi")}
    del store, frame, first_reference                                                            # drop every object ("close the app")

    # ---- session 2: laptop and camera unplugged and plugged back in: all objects are rebuilt from disk
    reopened = ProfileSettingsStore(directory)
    assert {name: reopened.get_section("live", name) for name in first} == first
    assert {name: reopened.section_saved_at("live", name) for name in first} == saved_ats
    assert reopened.get_section("live", "wifi")["auto_connect"] is False
    assert reopened.load_problem("live") == ""
    reference = reopened.load_reference_frame("live")
    assert reference.original_size == (640, 360) and reference.stored_size == (640, 360)
    assert np.array_equal(reference.image, cv2.cvtColor(frame_with(world, seed=1), cv2.COLOR_BGR2GRAY))
    plan = sr.camera_open_settings("live", reopened, {"backend": "msmf", "width": 1280, "height": 720, "fps": 30.0, "fourcc": "MJPG", "camera_index": 0})
    assert plan.index == 1 and (plan.settings["width"], plan.settings["height"], plan.settings["backend"]) == (640, 360, "dshow")

    same_view = frame_with(world, seed=7)                                                      # same scene, new sensor noise
    report = restore(reopened, same_view)
    assert report.calibration_loaded and report.settings_match.level == cs.MATCH
    assert report.alignment.status == ac.MATCH and report.alignment.max_corner_shift_px < 1.0
    assert report.trust is TrustLevel.NEEDS_CHECK and not report.world_coordinates_trusted and report.auto_check_passed
    assert report.can_confirm and report.calibrated_at == (NOW - timedelta(hours=18)).isoformat()
    assert "confirm the alignment" in report.headline

    confirmed = confirm_alignment(report, now=NOW)
    assert confirmed.trust is TrustLevel.CONFIRMED_THIS_SESSION and confirmed.world_coordinates_trusted
    assert record_confirmation("live", reopened, confirmed).ok
    assert report.world_coordinates_trusted is False                                            # the original report is unchanged

    # ---- session 3: another restart. Confirmation did NOT survive; only the informational time did.
    third = ProfileSettingsStore(directory)
    again = restore(third, same_view)
    assert again.trust is TrustLevel.NEEDS_CHECK and not again.world_coordinates_trusted
    assert again.last_confirmed_at == NOW.isoformat() and again.confirmed_at is None
    assert any("must be confirmed again after every restart" in line for line in again.lines)
    assert "world_coordinates_trusted" not in str(third.path_for("live").read_text(encoding="utf-8"))
    assert "trust" not in str(third.path_for("live").read_text(encoding="utf-8")).lower()


def test_restart_with_a_different_camera_resolution_needs_recalibration(tmp_path, world):
    store, _ = calibrated_store(tmp_path, world)
    store = ProfileSettingsStore(store.directory)
    big_sig = dataclasses.replace(SIG, width=1280, height=720)
    big_frame = cv2.resize(frame_with(world), (1280, 720))
    report = restore(store, big_frame, big_sig)
    assert report.trust is TrustLevel.NEEDS_RECALIBRATION and not report.world_coordinates_trusted
    assert "640 x 360" in report.recalibration_reason and "1280 x 720" in report.recalibration_reason
    assert report.alignment is None and report.settings_match.level == cs.INCOMPATIBLE
    assert not report.can_confirm
    with pytest.raises(ConfirmationRefused, match="Recalibrate"):
        confirm_alignment(report)
    assert any("DIFFERENT" in line for line in report.lines) and "Recalibrate" in report.next_action
    # a signature alone (no frame yet) is enough to see it
    assert restore(store, None, big_sig).trust is TrustLevel.NEEDS_RECALIBRATION


def test_restart_with_a_shifted_camera_needs_recalibration(tmp_path, world):
    store, _ = calibrated_store(tmp_path, world)
    report = restore(ProfileSettingsStore(store.directory), frame_with(world, shift_matrix(15.0, 0.0)))
    assert report.trust is TrustLevel.NEEDS_RECALIBRATION and report.alignment.major
    assert "moved" in report.recalibration_reason and not report.world_coordinates_trusted
    with pytest.raises(ConfirmationRefused):
        confirm_alignment(report)
    assert any("MOVED (major)" in line for line in report.lines)


def test_restart_with_a_slightly_shifted_camera_needs_a_check_and_may_be_confirmed(tmp_path, world):
    store, _ = calibrated_store(tmp_path, world)
    report = restore(store, frame_with(world, shift_matrix(5.0, 4.0)))
    assert report.trust is TrustLevel.NEEDS_CHECK and report.alignment.status == ac.SHIFTED and report.alignment.severity == "minor"
    assert not report.auto_check_passed and "check the alignment" in report.headline
    assert confirm_alignment(report, NOW).world_coordinates_trusted


def test_restart_with_an_unverifiable_frame_never_claims_a_match(tmp_path, world):
    store, calibration = calibrated_store(tmp_path, world)
    dark = (calibration * 0.04).astype(np.uint8)
    report = restore(store, dark)
    assert report.alignment.status == ac.UNVERIFIABLE
    assert report.trust is TrustLevel.NEEDS_CHECK and not report.auto_check_passed and not report.world_coordinates_trusted
    assert "could not be judged" in " ".join(report.lines) and "by eye" in report.next_action


def test_no_frame_yet_gives_needs_check_with_a_reason(tmp_path, world):
    store, _ = calibrated_store(tmp_path, world)
    report = restore(store, None)
    assert report.alignment is None and "no camera picture" in report.alignment_skipped_reason and report.trust is TrustLevel.NEEDS_CHECK
    assert not report.needs_reference_frame
    assert restore(store, None, signature=None).settings_match is None


def test_identical_frame_needs_check_then_confirm_gives_trust(tmp_path, world):
    store, frame = calibrated_store(tmp_path, world)
    report = restore(store, frame)
    assert report.trust is TrustLevel.NEEDS_CHECK and report.alignment.status == ac.MATCH
    done = confirm_alignment(report, NOW)
    assert done.trust is TrustLevel.CONFIRMED_THIS_SESSION and done.world_coordinates_trusted and done.confirmed_at == NOW.isoformat()
    assert done.headline == "Alignment confirmed for this session"
    assert confirm_alignment(done) is done                                    # idempotent


def test_a_changed_camera_with_a_matching_picture_still_needs_a_check(tmp_path, world):
    store, frame = calibrated_store(tmp_path, world)
    other = dataclasses.replace(SIG, device_name="Integrated Webcam")
    report = restore(store, frame, other)
    assert report.settings_match.level == cs.CHANGED and report.trust is TrustLevel.NEEDS_CHECK and not report.auto_check_passed
    assert any("DIFFERENT" in line for line in report.lines)
    # INFO-only differences (index moved, same name) do not spoil the automatic pass
    moved = restore(store, frame, dataclasses.replace(SIG, index=3))
    assert moved.settings_match.level == cs.MATCH and moved.auto_check_passed


# ------------------------------------------------------------------------------------------ not calibrated / inconsistent

def test_nothing_saved_is_not_calibrated(tmp_path):
    store = ProfileSettingsStore(tmp_path / "ps")
    report = restore_calibration("live", store, loaded_profile(points={}, reason="no calibration saved yet for profile 'live'"), SIG, None, now=NOW)
    assert report.trust is TrustLevel.NOT_CALIBRATED and not report.calibration_loaded and not report.world_coordinates_trusted
    assert "no calibration saved yet" in report.calibration_problem and report.headline == "Arena not calibrated"
    with pytest.raises(ConfirmationRefused, match="no calibration"):
        confirm_alignment(report)


def test_corners_without_an_arena_are_not_calibrated(tmp_path, world):
    store, frame = calibrated_store(tmp_path, world)
    report = restore(store, frame, arena=False)
    assert report.trust is TrustLevel.NOT_CALIBRATED and "arena itself" in report.calibration_problem


def test_a_profile_problem_means_recalibration(tmp_path, world):
    store, frame = calibrated_store(tmp_path, world)
    report = restore(store, frame, problem="the saved camera corner clicks were made for a different arena size than the saved arena", points={})
    assert report.trust is TrustLevel.NEEDS_RECALIBRATION and not report.calibration_loaded and "different arena size" in report.calibration_problem


def test_corner_and_arena_ids_that_disagree_mean_recalibration(tmp_path, world):
    store, frame = calibrated_store(tmp_path, world)
    clicks = loaded_profile(setup_id="old")
    profile = loaded_profile()
    profile.clicks = clicks.clicks                                           # corners from one set-up, arena from another
    report = restore_calibration("live", store, profile, SIG, frame, now=NOW)
    assert report.trust is TrustLevel.NEEDS_RECALIBRATION and "do not belong together" in report.recalibration_reason


def test_asking_for_another_profile_than_the_one_loaded_is_refused(tmp_path, world):
    store, frame = calibrated_store(tmp_path, world)
    with pytest.raises(ValueError, match="never share"):
        restore_calibration("live", store, loaded_profile("home_test"), SIG, frame, now=NOW)


def test_profiles_do_not_share_references_or_trust(tmp_path, world):
    store, frame = calibrated_store(tmp_path, world, profile="home_test")
    live = restore_calibration("live", store, loaded_profile("live"), SIG, frame, now=NOW)
    assert live.trust is TrustLevel.NEEDS_CHECK and live.alignment is None and live.needs_reference_frame
    assert "no reference picture was stored" in live.alignment_skipped_reason
    assert restore_calibration("home_test", store, loaded_profile("home_test"), SIG, frame, now=NOW).alignment.status == ac.MATCH


def test_legacy_calibration_without_a_reference_gets_one_after_the_operator_confirms(tmp_path, world):
    store = ProfileSettingsStore(tmp_path / "ps")
    frame = frame_with(world, seed=3)
    first = restore(store, frame)
    assert first.trust is TrustLevel.NEEDS_CHECK and first.alignment is None and first.needs_reference_frame and not first.auto_check_passed
    confirmed = confirm_alignment(first, NOW)
    assert record_calibration("live", store, SIG, CORNER_POINTS, SETUP_ID, frame, now=NOW).ok            # as the integration is told to do
    assert record_confirmation("live", store, confirmed).ok
    second = restore(store, frame_with(world, seed=9))
    assert second.alignment.status == ac.MATCH and not second.needs_reference_frame


def test_a_reference_from_an_older_calibration_is_not_used(tmp_path, world):
    store, frame = calibrated_store(tmp_path, world)
    moved_points = {**CORNER_POINTS, "arena_corner_c1": (CORNERS[0][0] + 25.0, CORNERS[0][1])}       # corners clicked again, reference not refreshed
    report = restore(store, frame, points=moved_points)
    assert report.alignment is None and "earlier calibration" in report.alignment_skipped_reason
    assert report.trust is TrustLevel.NEEDS_CHECK and report.needs_reference_frame
    other_arena = restore(store, frame, setup_id="a-different-arena")
    assert other_arena.trust is TrustLevel.NEEDS_CHECK and "different arena set-up" in other_arena.alignment_skipped_reason


def test_a_damaged_reference_picture_is_reported_and_not_a_crash(tmp_path, world):
    store, frame = calibrated_store(tmp_path, world)
    png = store.directory / store.load_reference_frame("live").file
    png.write_bytes(b"not a png")
    report = restore(store, frame)
    assert report.alignment is None and "damaged" in report.alignment_skipped_reason and report.trust is TrustLevel.NEEDS_CHECK


def test_a_damaged_settings_file_is_shown_but_the_bak_still_restores_the_calibration(tmp_path, world):
    store, frame = calibrated_store(tmp_path, world)
    sr.save_camera_preferences("live", store, SIG)                                                    # makes a .bak that holds the calibration
    store.path_for("live").write_bytes(b"\x00broken")
    report = restore(store, frame)
    assert report.alignment is not None and report.store_problem and "Saved settings:" in " ".join(report.lines)
    assert report.trust is TrustLevel.NEEDS_CHECK


def test_the_downscaled_reference_path_works_through_the_store(tmp_path):
    big = make_world(seed=3, size=(2400, 1700))
    origin = np.array([[1.0, 0.0, -500.0], [0.0, 1.0, -400.0]])
    full = (1280, 720)
    cal = cv2.cvtColor(view(big, origin, size=full, seed=1), cv2.COLOR_GRAY2BGR)
    corners = {f"arena_corner_c{i}": p for i, p in enumerate([(160.0, 120.0), (1120.0, 120.0), (1120.0, 600.0), (160.0, 600.0)], start=1)}
    sig = dataclasses.replace(SIG, width=1280, height=720)
    store = ProfileSettingsStore(tmp_path / "ps")
    assert record_calibration("live", store, sig, corners, SETUP_ID, cal, now=NOW).ok
    assert store.load_reference_frame("live").stored_size == (640, 360)
    profile = loaded_profile(points=corners, size=full)
    same = restore_calibration("live", store, profile, sig, cv2.cvtColor(view(big, origin, size=full, seed=4), cv2.COLOR_GRAY2BGR), now=NOW)
    assert same.alignment.status == ac.MATCH and same.trust is TrustLevel.NEEDS_CHECK
    moved = cv2.cvtColor(view(big, compose(shift_matrix(14.0, 0.0), origin), size=full, seed=5), cv2.COLOR_GRAY2BGR)
    result = restore_calibration("live", store, profile, sig, moved, now=NOW)
    assert result.trust is TrustLevel.NEEDS_RECALIBRATION and result.alignment.max_corner_shift_px == pytest.approx(14.0, abs=2.0)


def test_world_coordinates_are_never_trusted_unless_confirmed(tmp_path, world):
    store, frame = calibrated_store(tmp_path, world)
    reports = [restore(store, frame), restore(store, frame, signature=dataclasses.replace(SIG, width=1, height=1)),
               restore_calibration("live", ProfileSettingsStore(tmp_path / "empty"), loaded_profile(points={}), SIG, None, now=NOW)]
    assert [r.trust for r in reports] == [TrustLevel.NEEDS_CHECK, TrustLevel.NEEDS_RECALIBRATION, TrustLevel.NOT_CALIBRATED]
    assert not any(r.world_coordinates_trusted for r in reports)
    for level in TrustLevel:
        assert dataclasses.replace(reports[0], trust=level).world_coordinates_trusted == (level is TrustLevel.CONFIRMED_THIS_SESSION)


def test_report_is_serialisable_for_logs(tmp_path, world):
    store, frame = calibrated_store(tmp_path, world)
    import json
    data = restore(store, frame).to_dict()
    assert json.loads(json.dumps(data))["trust"] == "NEEDS_CHECK" and data["alignment"]["status"] == "MATCH"


def test_record_confirmation_refuses_an_unconfirmed_report(tmp_path, world):
    store, frame = calibrated_store(tmp_path, world)
    result = record_confirmation("live", store, restore(store, frame))
    assert not result.ok and "not been confirmed" in result.error


def test_arena_setup_corner_prefix_is_the_one_used_here():
    from control_center.tools.arena_setup import CORNER_PREFIX
    assert sr.CORNER_PREFIX == CORNER_PREFIX


def test_confirmation_time_is_stored_but_cannot_become_trust(tmp_path, world):
    store, frame = calibrated_store(tmp_path, world)
    done = confirm_alignment(restore(store, frame), NOW)
    assert record_confirmation("live", store, done).ok
    assert store.get_section("live", "calibration_context")["last_confirmed_at"] == NOW.isoformat()
    assert store.get_section("live", "calibration_context")["calibrated_at"] != NOW.isoformat()      # the calibration time is untouched
    assert not restore(store, frame).world_coordinates_trusted
    # recalibrating clears the old confirmation time
    assert record_calibration("live", store, SIG, CORNER_POINTS, SETUP_ID, frame, now=NOW).ok
    assert store.get_section("live", "calibration_context")["last_confirmed_at"] is None


# ------------------------------------------------------------------------------------------ camera open settings

def test_camera_open_settings_prefers_the_calibrated_mode_and_falls_back_to_config(tmp_path, world):
    store = ProfileSettingsStore(tmp_path / "ps")
    config = {"backend": "dshow", "width": 1280, "height": 720, "fps": 30.0, "fourcc": "MJPG", "camera_index": 1}
    empty = sr.camera_open_settings("live", store, config)
    assert empty.index == 1 and empty.settings == {k: v for k, v in config.items() if k != "camera_index"} and empty.notes == ()
    sr.save_camera_preferences("live", store, dataclasses.replace(SIG, index=2, width=800, height=600, extra={"exposure": -4}))
    prefs = sr.camera_open_settings("live", store, config)
    assert prefs.index == 2 and (prefs.settings["width"], prefs.settings["height"]) == (800, 600) and prefs.settings["exposure"] == -4
    record_calibration("live", store, dataclasses.replace(SIG, width=640, height=360, extra={"exposure": -6}), CORNER_POINTS, SETUP_ID, frame_with(world), now=NOW)
    calibrated = sr.camera_open_settings("live", store, config)
    assert (calibrated.settings["width"], calibrated.settings["height"]) == (640, 360) and calibrated.settings["exposure"] == -6 and calibrated.index == 2
    assert any("from the calibration" in n for n in calibrated.notes)
    remembered = sr.camera_open_settings("live", store, config, prefer_calibration=False)
    assert (remembered.settings["width"], remembered.settings["height"]) == (800, 600)
    assert sr.camera_open_settings("home_test", store, config).settings["width"] == 1280               # another profile is untouched


# ------------------------------------------------------------------------------------------ disconnect and reconnect

class FakeClock:
    def __init__(self) -> None:
        self.t = 100.0

    def __call__(self) -> float:
        return self.t

    def advance(self, seconds: float) -> None:
        self.t += seconds


def test_backoff_schedule_is_1_2_4_5_5_and_poll_fires_once_per_retry():
    clock = FakeClock()
    policy = CameraReconnectPolicy(clock=clock)
    assert policy.status_text() == "Camera not opened yet." and policy.poll() is False
    policy.on_event("LIVE")
    assert policy.status_text() == "Camera connected." and policy.next_retry_in() is None
    policy.on_event("LOST", "camera stopped delivering frames")
    assert policy.state == sr.LOST and "Camera lost" in policy.status_text() and "Reconnect" in policy.status_text()
    delays = []
    for _ in range(6):
        due = policy.next_retry_in()
        delays.append(round(due, 3))
        clock.advance(due - 0.4)
        assert policy.poll() is False                          # not yet
        clock.advance(0.4)
        assert policy.poll() is True and policy.poll() is False     # exactly once
        assert policy.state == sr.RECONNECTING and "Reconnecting" in policy.status_text()
        clock.advance(0.1)
        policy.on_event("FAILED", "camera index 1 did not open")
    assert delays == [1.0, 2.0, 4.0, 5.0, 5.0, 5.0]
    assert policy.attempts == 6 and policy.failures == 7


def test_manual_reconnect_resets_the_backoff_and_retries_immediately():
    clock = FakeClock()
    policy = CameraReconnectPolicy(clock=clock)
    policy.on_event("LIVE")
    policy.on_event("LOST")
    for _ in range(4):
        clock.advance(policy.next_retry_in())
        assert policy.poll()
        policy.on_event("FAILED")
    assert policy.next_retry_in() == 5.0
    assert policy.manual_reconnect() is True
    assert policy.next_retry_in() == 0.0 and "retrying now" in policy.status_text()
    assert policy.poll() is True
    policy.on_event("FAILED")
    assert policy.next_retry_in() == 1.0                         # the schedule starts over
    policy.on_event("LIVE")
    assert policy.manual_reconnect() is False and CameraReconnectPolicy(clock=clock).manual_reconnect() is False


def test_manual_reconnect_while_an_attempt_is_running_asks_for_another_one_now():
    clock = FakeClock()
    policy = CameraReconnectPolicy(clock=clock)
    policy.on_event("LOST")
    clock.advance(1.0)
    assert policy.poll() and policy.state == sr.RECONNECTING
    assert policy.manual_reconnect() and policy.poll()


def test_an_attempt_that_never_answers_counts_as_failed():
    clock = FakeClock()
    policy = CameraReconnectPolicy(clock=clock, attempt_timeout_s=10.0)
    policy.on_event("LOST")
    clock.advance(1.0)
    assert policy.poll()
    clock.advance(9.0)
    assert policy.poll() is False and policy.state == sr.RECONNECTING
    clock.advance(1.5)
    assert policy.poll() is False and policy.state == sr.LOST and policy.next_retry_in() == pytest.approx(2.0)


def test_capture_state_names_are_understood_and_unknown_ones_refused():
    policy = CameraReconnectPolicy(clock=FakeClock())
    for name in ("OPENING", "UNAVAILABLE", "LOST", "MOCK", "REPLAY", "LIVE", "reconnecting", "failed"):
        policy.on_event(name)
    with pytest.raises(ValueError):
        policy.on_event("EXPLODED")
    with pytest.raises(ValueError):
        CameraReconnectPolicy(backoff=())
    fresh = CameraReconnectPolicy(clock=FakeClock())
    fresh.on_event("UNAVAILABLE", "camera index 1 did not open")                 # unplugged at start-up: never had a picture
    assert "Camera not found" in fresh.status_text() and "camera index 1 did not open" in fresh.status_text()


def test_disconnect_then_reconnect_with_the_same_signature_keeps_the_calibration_and_trust(tmp_path, world):
    store, frame = calibrated_store(tmp_path, world)
    clock = FakeClock()
    policy = CameraReconnectPolicy(clock=clock)
    policy.on_event("LIVE")
    report = confirm_alignment(restore(store, frame), NOW)
    assert report.world_coordinates_trusted
    policy.on_event("LOST")
    clock.advance(1.0)
    assert policy.poll()
    policy.on_event("FAILED")
    clock.advance(2.0)
    assert policy.poll()
    policy.on_event("LIVE")
    saved = CameraSignature.from_dict(store.get_section("live", "calibration_context")["camera_signature"])
    verdict = policy.on_reconnected(dataclasses.replace(SIG, index=1), saved)
    assert verdict.outcome == sr.VALID and verdict.calibration_valid and verdict.keeps_trust and not verdict.recheck_alignment_recommended
    after = apply_reconnect_verdict(report, verdict, NOW)
    assert after is report and after.world_coordinates_trusted


def test_a_long_outage_recommends_a_recheck_but_does_not_revoke_trust():
    clock = FakeClock()
    policy = CameraReconnectPolicy(clock=clock)
    policy.on_event("LIVE")
    policy.on_event("LOST")
    clock.advance(90.0)
    verdict = policy.on_reconnected(SIG, SIG)
    assert verdict.outcome == sr.VALID and verdict.keeps_trust and verdict.recheck_alignment_recommended
    assert verdict.outage_s == pytest.approx(90.0) and "re-check" in verdict.message
    assert policy.outage_s is None                                               # consumed


def test_reconnect_with_a_different_resolution_invalidates_the_calibration(tmp_path, world):
    store, frame = calibrated_store(tmp_path, world)
    policy = CameraReconnectPolicy(clock=FakeClock())
    policy.on_event("LIVE")
    report = confirm_alignment(restore(store, frame), NOW)
    policy.on_event("LOST")
    policy.on_event("LIVE")
    verdict = policy.on_reconnected(dataclasses.replace(SIG, width=1280, height=720), SIG)
    assert verdict.outcome == sr.NEEDS_RECALIBRATION and not verdict.calibration_valid and verdict.needs_recalibration and not verdict.keeps_trust
    after = apply_reconnect_verdict(report, verdict, NOW)
    assert after.trust is TrustLevel.NEEDS_RECALIBRATION and not after.world_coordinates_trusted and after.confirmed_at is None
    assert "no longer applies" in after.recalibration_reason and not after.can_confirm


def test_reconnect_with_a_changed_device_untrusts_world_coordinates_until_reconfirmed(tmp_path, world):
    store, frame = calibrated_store(tmp_path, world)
    policy = CameraReconnectPolicy(clock=FakeClock())
    policy.on_event("LIVE")
    report = confirm_alignment(restore(store, frame), NOW)
    policy.on_event("LOST")
    policy.on_event("LIVE")
    verdict = policy.on_reconnected(dataclasses.replace(SIG, device_name="Integrated Webcam", index=0), SIG)
    assert verdict.outcome == sr.NEEDS_CHECK and verdict.needs_check and verdict.calibration_valid and verdict.recheck_alignment_recommended
    after = apply_reconnect_verdict(report, verdict, NOW)
    assert after.trust is TrustLevel.NEEDS_CHECK and not after.world_coordinates_trusted and after.can_confirm
    assert after.settings_match.level == cs.CHANGED and any("DIFFERENT" in line for line in after.lines)
    assert confirm_alignment(after, NOW).world_coordinates_trusted                  # the operator can re-confirm


def test_reconnect_with_only_info_differences_keeps_the_calibration():
    policy = CameraReconnectPolicy(clock=FakeClock())
    policy.on_event("LOST")
    policy.on_event("LIVE")
    verdict = policy.on_reconnected(dataclasses.replace(SIG, index=4, fps=15.0), SIG)            # replugged into another port, same camera
    assert verdict.outcome == sr.VALID and verdict.keeps_trust
    assert [d.severity for d in verdict.comparison.differences] == [cs.INFO, cs.INFO]


def test_reconnect_with_no_saved_signature_is_unknown_and_untrusted(tmp_path, world):
    policy = CameraReconnectPolicy(clock=FakeClock())
    policy.on_event("LOST")
    policy.on_event("LIVE")
    verdict = policy.on_reconnected(SIG, None)
    assert verdict.outcome == sr.UNKNOWN and verdict.needs_check and not verdict.keeps_trust and verdict.calibration_valid
    store, frame = calibrated_store(tmp_path, world)
    report = confirm_alignment(restore(store, frame), NOW)
    assert not apply_reconnect_verdict(report, verdict, NOW).world_coordinates_trusted


def test_a_verdict_never_improves_a_report():
    for trust in (TrustLevel.NOT_CALIBRATED, TrustLevel.NEEDS_RECALIBRATION):
        report = sr.RestoreReport("live", trust, False, calibration_problem="x")
        verdict = sr.ReconnectVerdict(sr.NEEDS_CHECK, None, None, "m")
        assert apply_reconnect_verdict(report, verdict) is report
