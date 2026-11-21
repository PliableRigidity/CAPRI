"""Setup screen: calibration model, single-tag check and the diagnostics panels.

Offscreen Qt, a MOCK hub and synthetic frames only: no camera, no serial port, and nothing is ever
written to config/arena_overlay_clicks.json or config/overlay_profiles/.
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtWidgets import QApplication, QLabel, QPushButton

from control_center.app import load_config
from control_center.backend import setup_tools
from control_center.backend.hub import HubOptions, OperationsHub
from control_center.backend.live_vision import VisionSample
from control_center.backend.setup_tools import (NO_IMU_TEXT, SPREAD_STATEMENT, CalibrationModel, RawLineLog,
                                                SingleTagCheck, list_profiles, marker_report)
from control_center.backend.vision import TagDetection
from control_center.screens import setup as screen_module
from control_center.screens.setup import SetupScreen, render_calibration_view
from control_center.tools.overlay_core import LIVE_PROFILE, PROFILES_DIR, BotPose, load_clicks, profile_points_path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REAL_LIVE_FILE = profile_points_path(LIVE_PROFILE)
FOUR_FIT = ["corner_origin", "pitA_rim_x0_wall", "pitB_rim_y0_wall", "flipper_edge_ymax_wall"]


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture()
def hub(qapp, tmp_path):
    hub = OperationsHub(load_config(ROOT / "config" / "control_center.yaml"),
                        HubOptions(mock=True, start_camera=False, rounds_root=tmp_path / "rounds"))
    yield hub
    hub.close()


@pytest.fixture(autouse=True)
def real_calibration_files_untouched():
    """Whatever a test does, the real live file and the real profiles folder stay byte-identical."""
    def state():
        files = [REAL_LIVE_FILE] + sorted(PROFILES_DIR.rglob("*")) + sorted(REAL_LIVE_FILE.parent.glob(REAL_LIVE_FILE.name + "*"))
        return {str(path): (path.stat().st_mtime_ns, path.stat().st_size) for path in files if path.is_file()}
    before = state()
    yield
    assert state() == before


def make_model(hub, tmp_path, mock=False) -> CalibrationModel:
    model = CalibrationModel(hub.vision.config, tmp_path / "live_clicks.json", mock=mock, profiles_dir=tmp_path / "profiles")
    model.set_frame_size(hub.mock_source.clicks().image_size)
    return model


def place(model: CalibrationModel, hub, names) -> None:
    pixels = hub.mock_source.clicks().points
    for name in names:
        assert model.select_name(name) is not None
        assert model.place(*pixels[name]) == name


def write_historical_profile(folder: Path, hub, name="historical_test_footage") -> Path:
    clicks = hub.mock_source.clicks()
    path = folder / name / "clicks.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"profile_kind": "historical", "profile": name, "image_size": list(clicks.image_size),
                                "points": {key: list(value) for key, value in clicks.points.items()}}), encoding="utf-8")
    return path


def pose_sample(index: int, tag: int, x: float, y: float, heading: float, visible=None) -> VisionSample:
    pose = BotPose(state="ok", reason="ok", x_m=x, y_m=y, heading_rad=heading, orientation="upright",
                   visible_ids=[tag] if visible is None else visible, used_ids=[tag])
    return VisionSample(index, time.monotonic(), pose, mapping_valid=True)


# ---------------------------------------------------------------- calibration model


def test_four_fit_points_give_a_valid_mapping_and_a_check_point_reports_held_out_error(hub, tmp_path):
    model = make_model(hub, tmp_path)
    assert model.quality.level == "insufficient" and not model.mapping.valid
    place(model, hub, FOUR_FIT)
    assert model.mapping.valid
    assert model.quality.level == "weak"                      # valid, but nothing independent confirms it
    assert "no check point" in model.fit_line()
    assert all(row.check_error_mm is None for row in model.rows())
    assert sum(row.fit_residual_mm is not None for row in model.rows()) == 4

    place(model, hub, ["tape_centre"])
    row = next(row for row in model.rows() if row.name == "tape_centre")
    assert row.role == "check" and row.fit_residual_mm is None
    assert row.check_error_mm is not None and row.check_error_mm < 1.0     # exact synthetic pixels
    assert model.mapping.check_stats["count"] == 1
    assert model.quality.level == "ok" and model.fit_line().startswith("FIT: OK")
    assert "Held-out CHECK error: max" in model.error_summary()


def test_a_misclicked_check_point_is_shown_as_weak_not_ok(hub, tmp_path):
    model = make_model(hub, tmp_path)
    place(model, hub, FOUR_FIT)
    u, v = hub.mock_source.clicks().points["tape_centre"]
    model.select_name("tape_centre")
    model.place(u + 40.0, v)
    assert model.mapping.valid and model.quality.level == "weak"
    assert "check point tape_centre is off by" in model.fit_line()
    assert model.save_block_reason() is None                  # weak may be saved; it is labelled weak


def test_too_few_points_are_insufficient_and_cannot_be_saved(hub, tmp_path):
    model = make_model(hub, tmp_path)
    place(model, hub, FOUR_FIT[:3])
    assert model.quality.level == "insufficient"
    assert "need at least 4 clicked fit points, have 3" in model.fit_line()
    result = model.save()
    assert not result.ok and "INSUFFICIENT" in result.message
    assert not model.target_path.exists()


def test_collinear_points_are_insufficient_and_cannot_be_saved(hub, tmp_path):
    model = make_model(hub, tmp_path)
    diagonal = ["corner_origin", "tape_1", "tape_centre", "tape_4"]       # all on the line y = x
    for name in diagonal[1:]:
        model.select_name(name)
        assert model.toggle_role() == "fit"
    place(model, hub, diagonal)
    assert not model.mapping.valid and model.quality.level == "insufficient"
    assert "collinear" in model.fit_line()
    assert not model.save().ok
    assert not model.target_path.exists()


def test_roles_toggle_per_point_and_undo_clear_skip_work(hub, tmp_path):
    model = make_model(hub, tmp_path)
    place(model, hub, FOUR_FIT + ["flipper_edge_xmax_wall"])
    index = [ref.name for ref in model.config.references].index("flipper_edge_xmax_wall")
    assert model.toggle_role(index) == "check"
    row = model.rows()[index]
    assert row.role == "check" and row.role_overridden and row.check_error_mm is not None
    assert model.undo() == "flipper_edge_xmax_wall"
    assert not model.rows()[index].placed and model.current.name == "flipper_edge_xmax_wall"
    before = model.current.name
    model.skip()
    assert model.current.name != before
    model.clear()
    assert not model.clicks.points and not model.clicks.roles and model.quality.level == "insufficient"


def test_save_writes_the_injected_path_as_live_and_backs_up_an_existing_file(hub, tmp_path):
    model = make_model(hub, tmp_path)
    place(model, hub, FOUR_FIT + ["tape_centre"])
    first = model.save(source="unit test")
    assert first.ok and first.path == tmp_path / "live_clicks.json" and first.backup is None
    saved = json.loads(first.path.read_text(encoding="utf-8"))
    assert saved["profile_kind"] == "live" and saved["profile"] == LIVE_PROFILE
    assert saved["fit_quality_when_saved"] == "ok"
    assert set(saved["points"]) == set(FOUR_FIT + ["tape_centre"])
    assert saved["image_size"] == list(model.frame_size)
    original = first.path.read_bytes()

    place(model, hub, ["tape_1"])
    second = model.save()
    assert second.ok and second.backup is not None
    assert re.fullmatch(r"live_clicks\.json\.bak-\d{8}-\d{6}(-\d+)?", second.backup.name)
    assert second.backup.read_bytes() == original
    assert "tape_1" in json.loads(first.path.read_text(encoding="utf-8"))["points"]
    third = model.save()                                      # same second: the first backup must survive
    assert third.ok and third.backup != second.backup and second.backup.read_bytes() == original


def test_saving_is_refused_in_mock_mode_for_the_real_live_path(hub):
    model = CalibrationModel(hub.vision.config, mock=True)
    assert model.targets_real_live_file
    model.set_frame_size(hub.mock_source.clicks().image_size)
    place(model, hub, FOUR_FIT + ["tape_centre"])
    assert model.quality.level == "ok"
    existed = REAL_LIVE_FILE.exists()
    result = model.save()
    assert not result.ok and "MOCK" in result.message
    assert REAL_LIVE_FILE.exists() == existed
    assert model.reload_saved()[0] is False                   # nor is the real file read over the mock picture


def test_synthetic_fixture_points_are_never_written_to_the_real_live_path(hub):
    model = CalibrationModel(hub.vision.config, mock=False, clicks=hub.mock_source.clicks())
    model.set_frame_size(hub.mock_source.clicks().image_size)
    assert model.quality.level == "ok"
    result = model.save()
    assert not result.ok and "SYNTHETIC" in result.message


def test_a_historical_profile_is_listed_but_cannot_be_loaded_or_overwritten(hub, tmp_path):
    profiles = tmp_path / "profiles"
    path = write_historical_profile(profiles, hub)
    listed = list_profiles(profiles)
    assert [info.name for info in listed] == ["historical_test_footage"]
    assert listed[0].historical
    assert listed[0].label() == "historical_test_footage — HISTORICAL footage, never used for the live camera"

    model = make_model(hub, tmp_path)
    place(model, hub, FOUR_FIT)
    points_before = dict(model.clicks.points)
    ok, reason = model.load_profile("historical_test_footage")
    assert not ok and "never loaded" in reason
    ok, reason = model.adopt(load_clicks(path))
    assert not ok and "HISTORICAL" in reason
    assert model.clicks.points == points_before and not model.clicks.historical

    # A live calibration is never saved into the profiles folder, nor over a footage file elsewhere.
    inside = CalibrationModel(hub.vision.config, path, profiles_dir=profiles)
    inside.set_frame_size(hub.mock_source.clicks().image_size)
    place(inside, hub, FOUR_FIT + ["tape_centre"])
    original = path.read_bytes()
    assert not inside.save().ok and path.read_bytes() == original
    elsewhere = tmp_path / "live_clicks.json"
    elsewhere.write_bytes(original)
    place(model, hub, ["tape_centre"])
    result = model.save()
    assert not result.ok and "HISTORICAL" in result.message and elsewhere.read_bytes() == original
    assert model.reload_saved()[0] is False                   # the footage file is not loaded as "saved live" either


def test_the_repository_footage_profile_is_listed_as_historical():
    labels = [info.label() for info in list_profiles()]
    assert "historical_bristol_2026_03_22 — HISTORICAL footage, never used for the live camera" in labels


# ---------------------------------------------------------------- single-tag check


def test_single_tag_check_reports_a_spread_between_tags_and_never_a_movement():
    check = SingleTagCheck([0, 1, 2, 3, 4, 5, 6, 7], samples_needed=20)
    assert check.add(pose_sample(0, 0, 0.5, 0.5, 0.0)) is None and not check.estimates      # not capturing yet
    check.start()
    index = 100
    for step in range(5):                                     # rejected: two tags, then an invalid pose
        check.add(pose_sample(index, 0, 0.5, 0.5, 0.0, visible=[0, 1]))
        index += 1
    assert "cover all but one" in check.status and check.progress[0] == 0
    invalid = VisionSample(index, time.monotonic(), BotPose(reason="no valid floor mapping (x)", visible_ids=[0]))
    check.add(invalid)
    assert check.progress[0] == 0 and "no valid pose" in check.status
    for step in range(20):
        index += 1
        done = check.add(pose_sample(index, 0, 0.500 + 0.0002 * (step % 2), 0.500, 0.10))
    assert done is not None and done.tag_id == 0 and done.samples == 20 and not check.capturing
    assert done.x_m == pytest.approx(0.5001) and done.heading_deg == pytest.approx(5.7296, abs=1e-3)
    assert check.spread() is None and "at least one more tag" in check.spread_text()

    check.start()
    for step in range(20):
        index += 1
        check.add(pose_sample(index, 2, 0.512, 0.505, 0.10 + 0.07))
    assert sorted(check.estimates) == [0, 2]
    spread = check.spread()
    assert spread.tags == (0, 2) and spread.position_pair == (0, 2)
    assert spread.position_mm == pytest.approx(1000.0 * ((0.512 - 0.5001) ** 2 + 0.005 ** 2) ** 0.5, abs=1e-6)
    assert spread.heading_deg == pytest.approx(4.0107, abs=1e-3)
    assert "configuration error" in spread.cause
    text = check.spread_text()
    assert "Spread between 2 tags" in text and "configuration error" in text
    for word in ("mov", "motion", "drift", "travel"):         # the difference is never described as the bot moving
        assert word not in text.lower()
        assert not any(word in field for field in vars(spread))
    assert "NOT bot movement" in SPREAD_STATEMENT and "must not be read as motion" in SPREAD_STATEMENT

    check.reset()
    assert not check.estimates and check.spread() is None and not check.capturing


def test_single_tag_capture_restarts_when_the_tag_in_view_changes():
    check = SingleTagCheck([0, 1], samples_needed=5)
    check.start()
    for index in range(3):
        check.add(pose_sample(index, 0, 0.5, 0.5, 0.0))
    check.add(pose_sample(3, 1, 0.52, 0.5, 0.0))
    assert check.progress == (1, 5, 1) and "restarted" in check.status
    for index in range(4, 8):
        done = check.add(pose_sample(index, 1, 0.52, 0.5, 0.0))
    assert done.tag_id == 1 and done.x_m == pytest.approx(0.52) and list(check.estimates) == [1]


# ---------------------------------------------------------------- panels


def fabricated_sample() -> VisionSample:
    corners = np.zeros((4, 2))
    detections = [TagDetection(tag_id, corners, (0, 0), 0.0) for tag_id in (0, 1, 1, 9)]
    pose = BotPose(reason="conflict: marker id(s) [1] detected more than once", visible_ids=[0, 1, 1], ignored_ids=[9])
    return VisionSample(7, time.monotonic(), pose, detections, mapping_valid=True, duplicate_ids=[1], unknown_ids=[9])


def test_marker_report_names_missing_duplicate_and_unknown_ids(hub):
    report = marker_report(hub.vision.config, fabricated_sample(), time.monotonic())
    assert report.value("Detected IDs") == "0, 1 (x2), 9"
    assert report.value("Configured IDs in view") == "0, 1 (top face)"
    assert report.value("Missing for the visible face").startswith("2, 3 not seen")
    assert report.value("Duplicate IDs").startswith("1 -")
    assert report.value("Unknown IDs").startswith("9 -")
    assert report.value("Pose") == "NOT VALID - conflict: marker id(s) [1] detected more than once"
    assert report.value("Position quality") == "unknown (no valid pose)"
    empty = marker_report(hub.vision.config, None)
    assert all(text for _, text, _ in empty.rows) and "No vision sample yet" in empty.text()
    stale = marker_report(hub.vision.config, fabricated_sample(), time.monotonic() + 5.0)
    assert stale.value("Vision").startswith("STALE")


def test_markers_panel_shows_missing_and_duplicate_ids_and_the_layout(hub, tmp_path):
    screen = SetupScreen(hub, live_points_path=tmp_path / "live.json", profiles_dir=tmp_path / "profiles")
    assert "No vision sample yet" in screen.markers_text()
    hub.latest_sample = fabricated_sample()
    screen.refresh()
    text = screen.markers_text()
    assert "Missing for the visible face: 2, 3 not seen" in text
    assert "Duplicate IDs: 1 - seen more than once" in text
    assert "Unknown IDs: 9 - not in the configured layout" in text
    assert "Pose: NOT VALID - conflict" in text
    table = screen.layout_table
    assert [table.horizontalHeaderItem(c).text() for c in range(table.columnCount())] == [
        "id", "face", "forward_m", "left_m", "yaw_deg", "size_m", "height_m"]
    assert table.rowCount() == len(hub.vision.config.markers)
    assert [table.item(0, c).text() for c in range(7)] == ["0", "top", "+0.035", "+0.030", "+0.0", "0.040", "0.034"]
    assert "PLACEHOLDER" in screen.placeholder_label.text()
    assert screen.single_tag_statement.text() == SPREAD_STATEMENT

    screen.capture_single_tag()
    assert screen.single_tag.capturing and screen.capture_button.text() == "Cancel capture"
    for index in range(20):
        hub.sample_ready.emit(pose_sample(index, 3, 0.4, 0.6, 1.0))
    screen.refresh()
    assert screen.single_tag_table.rowCount() == 1 and screen.single_tag_table.item(0, 0).text() == "3"
    screen.reset_single_tag()
    assert screen.single_tag_table.rowCount() == 0


def test_controller_panel_labels_the_missing_imu_stream_then_shows_raw_values(hub, tmp_path):
    screen = SetupScreen(hub, live_points_path=tmp_path / "live.json", profiles_dir=tmp_path / "profiles")
    assert NO_IMU_TEXT == "No IMU stream — the controller is not sending IMU lines (the handoff bench sketch has no IMU)"
    assert screen.imu_label.text() == NO_IMU_TEXT
    text = screen.controller_text()
    assert "no controller telemetry source connected" in text and "Not connected" in text
    assert "no ST line received" in text and "no CH line received" in text and "no HELLO line received" in text
    assert "no HELLO line received" in screen.firmware_label.text()
    assert "no lines received" in screen.raw_tail.toPlainText()

    now = time.monotonic()
    hub.monitor.feed("IMU,100,0.1,0.2,9.8,1,2,3", now)
    hub.monitor.feed("HELLO fw=unit-test-fw arm=CH5:HIGH", now)
    hub.controller_line.emit("RX", "HELLO fw=unit-test-fw arm=CH5:HIGH")
    hub.controller_line.emit("RX", "CH 1=1500 2=1500")
    screen.refresh(now)
    imu = screen.imu_label.text()
    assert "No IMU stream" not in imu and "sensor axes, raw" in imu
    assert "accel X/Y/Z (m/s^2):  +0.100   +0.200   +9.800" in imu
    assert "gyro  X/Y/Z (deg/s):  +1.000   +2.000   +3.000" in imu
    assert "rate: measuring" in imu
    assert screen.firmware_label.text().startswith("unit-test-fw")
    assert screen.raw_values["HELLO"].text() == "HELLO fw=unit-test-fw arm=CH5:HIGH"
    assert screen.raw_values["CH"].text() == "CH 1=1500 2=1500"
    assert "RX  CH 1=1500 2=1500" in screen.raw_tail.toPlainText()

    screen.refresh(now + 5.0)                                 # the stream stopped: stale values are not shown as current
    assert screen.imu_label.text().startswith(NO_IMU_TEXT) and "9.800" not in screen.imu_label.text()


def test_raw_line_log_keeps_the_newest_line_of_each_kind():
    log = RawLineLog(keep=3)
    for line in ("ST t=1 tok=1", "IMU,1,0,0,0,0,0,0", "ST t=2 tok=1", "STATUS odd", "HELLO fw=x"):
        log.add("RX", line)
    log.add("TX", "ST t=99 sent by someone else")
    assert log.latest["ST"] == "ST t=2 tok=1" and log.latest["IMU"].startswith("IMU,1") and log.count == 6
    lines = log.tail_text().splitlines()
    assert len(log.tail) == 3 and lines[2].startswith("TX  ")
    assert not any("IMU," in line for line in lines)          # a 50 Hz IMU stream must not push ST out of the list
    assert lines[3] == "(1 IMU line(s) received; they are left out of this list)"
    log.clear()
    assert log.latest_text("ST") == "no ST line received"


# ---------------------------------------------------------------- the screen


def test_screen_constructs_and_refreshes_without_a_camera(hub, tmp_path):
    screen = SetupScreen(hub, live_points_path=tmp_path / "live.json", profiles_dir=tmp_path / "profiles")
    for index in range(screen.tabs.count()):
        screen.tabs.setCurrentIndex(index)
        screen.refresh()
    assert [screen.tabs.tabText(i) for i in range(4)] == ["Camera", "Floor calibration", "Markers", "Controller && IMU"]
    assert screen.camera_values["Source"].text() == "not opened yet"
    assert screen.camera_values["Resolution"].text() == "no frame received yet"
    assert screen.camera_values["Measured fps"].text().startswith("not measured yet")
    assert screen.camera_values["State"].text().startswith("OPENING")
    assert screen.camera_index.value() == hub.options.camera_index == 1
    assert screen.profile_label.text() == "LIVE profile (your camera)"
    recalibrate = screen.recalibrate_label.text()
    assert all(word in recalibrate for word in ("moved", "re-aimed", "zoomed", "resolution changes"))
    assert any("AT FLOOR LEVEL" in label.text() for label in screen.calibration_tab.findChildren(QLabel))
    assert screen.fit_label.text().startswith("FIT: INSUFFICIENT") and not screen.save_button.isEnabled()
    assert not screen.calibration.clicks.points               # the mock's synthetic points are not adopted
    assert not screen.freeze() and not screen.frozen          # no frame to freeze, and it says so
    assert "No camera frame to freeze" in screen.calibration_message.text()

    screen.camera_index.setValue(2)
    screen.reopen_camera()                                    # the hub refuses in mock mode; its message is shown
    assert "mock or replay source" in screen.reopen_message.text() and hub.options.camera_index == 1


def test_screen_calibrates_on_a_frozen_frame_and_saves_through_the_hub(hub, tmp_path):
    target = tmp_path / "live.json"
    screen = SetupScreen(hub, live_points_path=target, profiles_dir=tmp_path / "profiles")
    screen.tabs.setCurrentWidget(screen.calibration_tab)
    frame = hub.mock_source.frame_for(0)
    pixels = hub.mock_source.clicks().points

    screen.calibration_view.clicked.emit(10.0, 10.0)          # a click on the live picture places nothing
    assert not screen.calibration.clicks.points and "Freeze a frame" in screen.calibration_message.text()

    assert screen.freeze(frame) and screen.frozen
    assert screen.calibration.frame_size == (frame.shape[1], frame.shape[0])
    names = [ref.name for ref in hub.vision.config.references]
    for name in FOUR_FIT + ["tape_centre"]:
        screen.landmark_table.selectRow(names.index(name))    # select in the list, then click the picture
        screen.calibration_view.clicked.emit(*pixels[name])
    assert screen.fit_label.text().startswith("FIT: OK") and screen.save_button.isEnabled()
    row = names.index("tape_centre")
    assert screen.landmark_table.item(row, 2).text() == "CHECK" and screen.landmark_table.item(row, 3).text() == "placed"
    assert screen.landmark_table.item(row, 4).text() == "-" and float(screen.landmark_table.item(row, 5).text()) < 1.0
    assert float(screen.landmark_table.item(0, 4).text()) < 1.0 and screen.landmark_table.item(0, 5).text() == "-"

    drawn = render_calibration_view(frame, screen.calibration)
    assert drawn is not frame and drawn.shape == frame.shape and (drawn != frame).any()
    assert (frame == hub.mock_source.frame_for(0)).all()      # the shared frame itself is never drawn on

    assert screen.save_calibration()
    assert json.loads(target.read_text(encoding="utf-8"))["profile_kind"] == "live"
    assert hub.vision.profile == LIVE_PROFILE
    assert set(hub.vision.clicks.points) == set(FOUR_FIT + ["tape_centre"])
    assert hub.vision.clicks is not screen.calibration.clicks  # the hub got its own copy
    assert "saved 5 point(s)" in screen.save_message.text()

    screen.undo_click()
    screen.undo_click()
    assert screen.fit_label.text().startswith("FIT: INSUFFICIENT") and not screen.save_button.isEnabled()
    assert len(hub.vision.clicks.points) == 5                 # editing after a save does not touch the live vision
    screen.unfreeze()
    assert not screen.frozen


def test_screen_disables_saving_in_mock_mode_and_never_offers_a_footage_profile(hub, tmp_path):
    write_historical_profile(tmp_path / "profiles", hub)
    screen = SetupScreen(hub, profiles_dir=tmp_path / "profiles")            # default target = the real live file
    assert screen.calibration.targets_real_live_file
    assert screen.freeze(hub.mock_source.frame_for(0))
    pixels = hub.mock_source.clicks().points
    for name in FOUR_FIT + ["tape_centre"]:
        screen.calibration.select_name(name)
        screen.place_click(*pixels[name])
    assert screen.fit_label.text().startswith("FIT: OK")
    assert not screen.save_button.isEnabled()
    assert "MOCK" in screen.save_message.text() and "disabled" in screen.save_message.text()
    assert not screen.save_calibration()                      # even called directly, nothing is written
    assert hub.vision.clicks.synthetic                        # and the hub's calibration is unchanged

    assert "historical_test_footage — HISTORICAL footage, never used for the live camera" in screen.profiles_label.text()
    buttons = {button.text() for button in screen.findChildren(QPushButton)}
    assert buttons == {"Reopen camera", "Freeze frame", "Unfreeze", "Swap fit / check", "Skip", "Undo", "Clear",
                       "Save live calibration", "Reload saved", "Capture (20 samples)", "Reset",
                       "Listen (read-only)", "Follow client log…", "Disconnect"}
    assert not [name for name in dir(screen) if "profile" in name and name.startswith(("load", "open", "use", "apply"))]


def test_setup_sources_never_write_to_a_port_or_build_a_drive_command():
    for module in (screen_module, setup_tools):
        source = Path(module.__file__).read_text(encoding="utf-8")
        assert ".write(" not in source
        assert not re.search(r"""["']D,""", source)           # no "D,<token>,<throttle>,<steer>" construction
        assert "import serial" not in source and "SerialLineSource" not in source
        assert "VideoCapture" not in source                   # the hub owns the camera
        assert not re.search(r"(?<![A-Za-z_])open_camera", source)      # only hub.reopen_camera is ever called
    screen_source = Path(screen_module.__file__).read_text(encoding="utf-8")
    assert "robot_link" not in screen_source and "command_manager" not in screen_source
