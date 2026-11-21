"""Review screen and trace plot. Rounds are built in tmp_path with RoundRecorder; no camera, no port."""

from __future__ import annotations

import json
import math
import os
import re
import time
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import cv2
import numpy as np
import pytest
from PySide6.QtCore import QPoint, Qt
from PySide6.QtGui import QColor
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from control_center.backend.capture import MockArenaSource
from control_center.backend.controller_link import parse_status_line, status_line
from control_center.backend.live_vision import LiveVision
from control_center.backend.round_reader import RoundData
from control_center.backend.round_recorder import RoundRecorder, new_round_folder, pose_row
from control_center.screens.review import (ReviewScreen, control_readout, event_rows, format_time, latest_round_folder,
                                           pose_readout, round_duration)
from control_center.tools.overlay_core import load_config
from control_center.widgets.trace_plot import Series, TracePlot, finite_runs

FPS = 10.0


# ---------------------------------------------------------------- building rounds


class FakeClock:
    """The recorder's clock, so the round's timeline is exactly what the test feeds it."""

    def __init__(self, t: float = 500.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t


def numbered_frame(index: int, size: tuple[int, int] = (160, 120)) -> np.ndarray:
    frame = np.full((size[1], size[0], 3), 40, np.uint8)
    cv2.putText(frame, str(index), (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)
    return frame


def record_round(folder: Path, manifest: dict, fill, duration_s: float):
    clock = FakeClock()
    recorder = RoundRecorder(folder, manifest, fps=FPS, max_queued_frames=10000, clock=clock)
    recorder.start()
    fill(recorder, recorder.t0_mono)
    clock.t = recorder.t0_mono + duration_s
    return recorder.stop()


def log_status(recorder: RoundRecorder, t_mono: float, **fields) -> None:
    """One controller ST line, parsed by the real parser, stored as docs/round-recording-format.md describes."""
    status = parse_status_line(status_line(int((t_mono - recorder.t0_mono) * 1000), **fields))
    assert status is not None
    recorder.log_control(t_mono, "controller_status", {
        "device_ms": status.device_ms, "link": status.link, "armed": status.armed, "requested": status.requested,
        "active": status.active, "pc_ok": status.pc_ok, "block": status.block, "fault": status.fault,
        "stop": status.stop, "left": status.left, "right": status.right, "pc_age_ms": status.pc_age_ms,
        "token": status.token})


def make_video_only_round(root: Path) -> Path:
    """2 s of video, no calibration, no telemetry. Frame 7 has the wrong size, so it is not in the video."""
    folder = new_round_folder(root, "video only")

    def fill(recorder: RoundRecorder, t0: float) -> None:
        for i in range(21):
            frame = numbered_frame(i, (80, 60)) if i == 7 else numbered_frame(i)
            assert recorder.add_frame(100 + i, t0 + i / FPS, frame)

    record_round(folder, {"label": "video only", "mock": False, "calibration": None,
                          "calibration_note": "no live calibration saved yet"}, fill, 2.0)
    return folder


GAP_REASON = "no configured bot markers visible"


def make_handoff_round(root: Path) -> Path:
    """6 s: RADIO, then a requested and confirmed handoff to PC, a telemetry gap, a tracking dropout."""
    folder = new_round_folder(root, "handoff")

    def fill(recorder: RoundRecorder, t0: float) -> None:
        for i in range(61):
            t = i / FPS
            recorder.add_frame(i, t0 + t, numbered_frame(i))
            if 2.0 <= t < 3.0:
                row = {"valid": 0, "x_m": "", "y_m": "", "heading_deg": "", "orientation": "unknown", "position_quality": "",
                       "used_ids": "", "visible_ids": "", "tag_switch": 0, "reason": GAP_REASON}
            else:
                row = {"valid": 1, "x_m": f"{0.30 + 0.05 * t:.4f}", "y_m": "0.5000", "heading_deg": "45.00",
                       "orientation": "upright", "position_quality": "height-corrected", "used_ids": "0 1",
                       "visible_ids": "0 1", "tag_switch": 0, "reason": ""}
            recorder.log_pose(t0 + t, i, row)
        for k in range(31):
            t = k * 0.2
            if k <= 9:
                log_status(recorder, t0 + t, armed=1, req="RADIO", act="RADIO", block="NONE", token=2, left=90, right=80)
            elif k <= 11:
                log_status(recorder, t0 + t, armed=1, req="PC", act="NONE", block="WAIT_PC_NEUTRAL", stop="HANDOFF", token=3)
            elif k <= 15:
                log_status(recorder, t0 + t, armed=1, req="PC", act="PC", pc_ok=1, block="NONE", stop="HANDOFF", token=3,
                           left=60, right=60, pc_age=10)
                recorder.log_control(t0 + t, "pc_sent", {"token": 3, "throttle": 60, "steer": 0})
            elif k >= 23:
                log_status(recorder, t0 + t, armed=0, req="PC", act="NONE", block="REARM_REQUIRED", fault="PC_STALE",
                           stop="PC_STALE", token=4)
            # k = 16..22 (3.2 s to 4.4 s): no controller line at all
        for j in range(300):
            t = j * 0.02
            recorder.log_imu(t0 + t, int(t * 1000), (0.12, -0.05, 9.807), (0.1, -0.2, 12.0 * math.sin(t)))
        recorder.log_policy(t0 + 2.5, {"mode": "SIM", "input_source": "simulator", "checkpoint": "test-checkpoint",
                                       "proposed": {"throttle": 0.4, "steer": -0.1}, "note": "display only"})
        recorder.log_event(t0 + 0.0, "source_confirmed", "Controller confirms source RADIO (armed, requested RADIO)", source="RADIO")
        recorder.mark_round_start(t0 + 0.5)
        recorder.log_event(t0 + 2.0, "mode_requested", "RadioMaster requests PC", requested="PC")
        recorder.log_event(t0 + 2.4, "source_confirmed", "Controller confirms source PC", source="PC")
        recorder.log_event(t0 + 2.5, "tracking_lost", f"Own-bot tracking lost: {GAP_REASON}", reason=GAP_REASON)
        recorder.log_event(t0 + 3.0, "tracking_reacquired", "Own-bot tracking reacquired after 1.0 s")
        recorder.log_event(t0 + 3.7, "controller_link", "Controller telemetry STALE: control source is no longer confirmed")

    manifest = {"label": "handoff", "mock": True, "mock_note": "MOCK session: synthetic, not hardware",
                "calibration": {"points_px": {"plain corner": [10.0, 20.0]}}, "arena": dict(load_config().arena)}
    record_round(folder, manifest, fill, 6.0)
    return folder


@pytest.fixture(scope="module")
def mock_material():
    """6 s of MockArenaSource frames with the live pose LiveVision gives for each (hidden from 2 s to 3 s)."""
    config = load_config()
    source = MockArenaSource(config, fps=FPS, period_s=6.0, hide=(2.0, 3.0), single_tag=(4.0, 4.6), realtime=False)
    vision = LiveVision(config, source.clicks(), profile="synthetic fixture")
    frames, rows = [], []
    for i in range(60):
        _, frame, _ = source.read()
        sample = vision.process(i, i / FPS, frame)
        frames.append(frame)
        rows.append(pose_row(sample.pose, sample.tag_switch))
    assert sum(row["valid"] for row in rows) >= 40
    return {"config": config, "frames": frames, "rows": rows, "calibration": vision.calibration_snapshot()}


def make_mock_round(root: Path, material: dict, label: str, with_live_pose: bool) -> Path:
    folder = new_round_folder(root, label)

    def fill(recorder: RoundRecorder, t0: float) -> None:
        for i, frame in enumerate(material["frames"]):
            recorder.add_frame(i, t0 + i / FPS, frame)
            if with_live_pose:
                recorder.log_pose(t0 + i / FPS, i, material["rows"][i])

    manifest = {"label": label, "mock": True, "calibration": material["calibration"], "arena": dict(material["config"].arena)}
    record_round(folder, manifest, fill, 6.0)
    return folder


# ---------------------------------------------------------------- Qt fixtures


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def screen(app):
    widget = ReviewScreen()
    widget.resize(1400, 900)
    widget.show()
    app.processEvents()
    yield widget
    widget.close()
    widget.deleteLater()
    app.processEvents()


def wait_until(app, condition, timeout_s: float = 30.0) -> bool:
    end = time.monotonic() + timeout_s
    while time.monotonic() < end:
        app.processEvents()
        if condition():
            return True
        time.sleep(0.01)
    return False


def paint(app, widget) -> None:
    """Force a full repaint: a paint-time exception would surface here."""
    app.processEvents()
    assert not widget.grab().isNull()


# ---------------------------------------------------------------- (a) video-only, uncalibrated


def test_video_only_uncalibrated_round_opens_and_labels_absent_streams(app, screen, tmp_path):
    folder = make_video_only_round(tmp_path)
    ok, message = screen.open_round(folder)
    assert ok, message
    assert screen.slider.maximum() == 2000
    assert not screen.mock_badge.isVisibleTo(screen)
    for t in (0.0, 0.35, 1.0, 1.999, 2.0, 5.0, -1.0):
        screen.seek(t)
        paint(app, screen)
    assert screen.current_time == 0.0                               # clamped
    screen.seek(1.0)
    assert screen.current_frame_row == 10
    assert re.fullmatch(r"\d\d:\d\d\.\d / \d\d:\d\d\.\d  frame \d+", screen.time_label.text())
    assert screen.time_label.text() == "00:01.0 / 00:02.0  frame 110"   # the camera's frame index, not the row

    streams = screen.streams_label.text()
    absent = streams.split("Absent:")[1]
    for name in ("own-bot pose, live", "own-bot pose, reconstructed offline", "opponent track", "IMU",
                 "controller status", "PC-sent commands", "policy"):
        assert name in absent
    assert "video" in streams.split("Absent:")[0]

    # No telemetry: said so, never a value.
    assert screen.source_value.text() == "UNKNOWN"
    assert screen.requested_value.text() == "UNKNOWN"
    assert screen.armed_value.text() == "UNKNOWN"
    assert screen.telemetry_value.text() == "No control telemetry in this round"
    assert screen.pc_sent_value.text() == "No PC-sent commands in this round"
    assert screen.policy_value.text() == "No policy stream in this round"
    assert screen.imu_value.text() == "No IMU stream in this round"
    assert screen.pose_value.text().startswith("position unknown: no own-bot track in this round")

    # No calibration: the map says so and draws nothing.
    assert screen.map is not None
    assert screen.map.note == "No calibration in this round: no camera positions"
    assert screen.map.trails == [] and screen.map.markers == []
    assert "no calibration in this round" in screen.warnings_label.text()
    assert screen.opponent_label.text() == "No opponent track in this round"
    assert "No live own-bot track in this round" in screen.map_legend.text()
    assert "No offline reconstruction for this round" in screen.map_legend.text()

    for plot in (screen.accel_plot, screen.gyro_plot):
        assert not plot.has_data() and plot.message() == "No IMU stream in this round"
    assert not screen.position_plot.has_data() and "No camera position in this round" in screen.position_plot.message()
    screen.tabs.setCurrentIndex(1)
    paint(app, screen)


def test_frame_not_in_video_shows_placeholder_and_header_counts_it(app, screen, tmp_path):
    folder = make_video_only_round(tmp_path)
    assert screen.open_round(folder)[0]
    assert "frames received 21, written 20, dropped 0 (1 of the received frames are not in the video)" in screen.summary_label.text()
    assert "write errors: 1" in screen.summary_label.text()
    assert "not written" in screen.warnings_label.text()            # the recorder's own error text is shown

    screen.seek(0.7)                                                # camera frame 107 was refused (wrong size)
    assert screen.current_frame_row is None
    assert "Frame 107 is not in the video" in screen.frame_view.placeholder
    assert screen.time_label.text().endswith("frame 107")
    paint(app, screen)
    screen.step_frame(1)
    assert screen.current_frame_row == 8 and screen.current_time == pytest.approx(0.8)
    screen.step_frame(-1)
    screen.step_frame(-1)
    assert screen.current_frame_row == 6 and screen.current_time == pytest.approx(0.6)
    screen.seek(0.0)
    screen.step_frame(-1)
    assert screen.current_time == 0.0
    screen.seek(2.0)
    screen.step_frame(1)
    assert screen.current_time == pytest.approx(2.0)


def test_incomplete_or_missing_manifest_does_not_crash(app, screen, tmp_path):
    folder = make_video_only_round(tmp_path)
    manifest_path = folder / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for key in ("duration_s", "video", "streams", "write_errors"):
        manifest.pop(key)
    manifest["complete"] = False                                    # what a crash before stop() leaves behind
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    ok, message = screen.open_round(folder)
    assert ok, message
    assert "not closed cleanly" in screen.warnings_label.text()
    assert "counted from frames.csv" in screen.summary_label.text()
    assert "write errors: unknown" in screen.summary_label.text()
    assert screen.duration == pytest.approx(2.0, abs=0.01)          # from the streams, not the manifest
    screen.seek(1.0)
    assert screen.current_frame_row == 10
    paint(app, screen)

    manifest_path.write_text("{}", encoding="utf-8")                # nothing but a manifest that exists
    assert screen.open_round(folder)[0]
    screen.seek(1.5)
    assert screen.current_frame_row == 15
    paint(app, screen)

    opened = screen.data.folder
    ok, message = screen.open_round(tmp_path / "not-a-round")
    assert not ok and "manifest.json" in message
    manifest_path.write_text("{ not json", encoding="utf-8")
    ok, message = screen.open_round(folder)
    assert not ok and "Could not open" in screen.status_label.text()
    assert screen.data.folder == opened                             # the round already open stays
    screen.seek(0.5)
    paint(app, screen)


def test_no_round_open_is_safe(app, screen):
    assert screen.data is None
    screen.seek(1.0)
    screen.step_frame(1)
    screen.jump_event(1)
    screen.toggle_play()
    assert not screen.playing
    assert screen.reconstruct_offline() is False
    assert not screen.slider.isEnabled() and not screen.reconstruct_button.isEnabled()
    paint(app, screen)


# ---------------------------------------------------------------- (b) handoff, telemetry gap, tracking dropout


def test_handoff_round_panel_shows_source_requested_armed_and_stale(app, screen, tmp_path):
    folder = make_handoff_round(tmp_path)
    ok, message = screen.open_round(folder)
    assert ok, message
    assert screen.mock_badge.isVisibleTo(screen) and "not hardware" in screen.mock_badge.text()
    assert "not hardware" in screen.frame_view.badge
    assert "Absent:" in screen.streams_label.text() and "opponent track" in screen.streams_label.text().split("Absent:")[1]

    def panel() -> tuple[str, str, str]:
        return screen.source_value.text(), screen.requested_value.text(), screen.armed_value.text()

    screen.seek(1.0)
    assert panel() == ("RADIO", "RADIO", "ARMED")
    assert screen.motors_value.text() == "L 90   R 80"
    assert "block NONE" in screen.flags_value.text()
    assert screen.pc_sent_value.text() == "none in the 1.0 s before this time"

    screen.seek(2.1)                                                # PC requested, not yet confirmed
    assert panel() == ("NONE", "PC", "ARMED")
    assert "WAIT_PC_NEUTRAL" in screen.flags_value.text()

    screen.seek(2.7)
    assert panel() == ("PC", "PC", "ARMED")
    assert screen.motors_value.text() == "L 60   R 60"
    assert screen.pc_sent_value.text().startswith("throttle 60   steer 0")
    assert screen.moment_captions["pc_sent"].text() == "PC sent (not confirmed applied)"
    assert screen.moment_captions["policy"].text() == "policy proposed (display only)"
    assert "mode SIM" in screen.policy_value.text() and "throttle 0.400" in screen.policy_value.text()

    screen.seek(3.3)                                                # 0.3 s after the last report: still fresh
    assert panel() == ("PC", "PC", "ARMED")

    screen.seek(4.0)                                                # 1.0 s after the last report
    assert panel() == ("STALE", "STALE", "STALE")
    assert screen.motors_value.text() == "STALE" and screen.flags_value.text() == "STALE"
    assert screen.telemetry_value.text().startswith("STALE")
    assert "PC" not in " ".join(panel())
    assert "mode SIM" in screen.policy_value.text() and "throttle" not in screen.policy_value.text()
    assert "proposed: none in the 1.0 s before this time" in screen.policy_value.text()

    screen.seek(5.0)
    assert panel() == ("NONE", "PC", "DISARMED")
    assert "fault PC_STALE" in screen.flags_value.text()
    assert screen.imu_value.text().startswith("accel")
    paint(app, screen)


def test_handoff_round_trail_has_gap_and_unknown_position_is_said(app, screen, tmp_path):
    folder = make_handoff_round(tmp_path)
    assert screen.open_round(folder)[0]

    screen.seek(1.0)
    assert len(screen.map.trails) == 1
    trail = screen.map.trails[0]
    assert trail.dashed is False and trail.label == "live"
    assert len(trail.runs) == 1 and len(trail.runs[0]) == 11        # only up to the selected time
    assert len(screen.map.markers) == 1 and screen.map.markers[0].hollow is False
    assert screen.map.markers[0].x == pytest.approx(0.35)
    assert screen.pose_value.text() == "x 0.350 m   y 0.500 m   heading 45.0°"
    assert screen.pose_source_value.text() == "live (estimated during the round)"
    assert screen.map_position.text().endswith("[live]")
    assert screen.map.note == ""

    screen.seek(2.5)                                                # inside the dropout
    assert screen.map.markers == []
    assert screen.pose_value.text() == f"position unknown: {GAP_REASON}"
    assert screen.map_position.text() == f"position unknown: {GAP_REASON}"
    assert len(screen.map.trails[0].runs) == 1

    screen.seek(6.0)
    runs = screen.map.trails[0].runs
    assert len(runs) == 2                                           # the dropout is a gap, not a bridge
    assert runs[0][-1][0] == pytest.approx(0.30 + 0.05 * 1.9) and runs[1][0][0] == pytest.approx(0.30 + 0.05 * 3.0)
    assert "solid" in screen.map_legend.text() and "live (estimated during the round)" in screen.map_legend.text()
    assert "No offline reconstruction for this round" in screen.map_legend.text()
    assert screen.opponent_label.text() == "No opponent track in this round"

    # Traces: IMU and camera are separate plots, the camera trace has the same gap, cursors follow.
    assert screen.accel_plot.has_data() and screen.gyro_plot.has_data()
    assert [s.name for s in screen.accel_plot.series] == ["X", "Y", "Z"]
    assert "sensor axes" in screen.accel_plot.title and "sensor axes" in screen.gyro_plot.title
    assert "PC receipt time" in screen.traces_note.text() and "not fused" in screen.traces_note.text()
    assert len(screen.position_plot.runs(0)) == 2 and screen.position_plot.series[0].dashed is False
    assert all(plot.cursor == pytest.approx(6.0) for plot in screen.plots)
    screen.accel_plot.time_clicked.emit(4.0)
    assert screen.current_time == pytest.approx(4.0) and screen.slider.value() == 4000
    assert screen.position_plot.cursor == pytest.approx(4.0)
    screen.tabs.setCurrentIndex(1)
    paint(app, screen)


def test_handoff_round_events_are_clickable_and_distinct(app, screen, tmp_path):
    folder = make_handoff_round(tmp_path)
    assert screen.open_round(folder)[0]
    view = screen.events_view
    rows = [(view.topLevelItem(i).text(0), view.topLevelItem(i).text(1)) for i in range(view.topLevelItemCount())]
    kinds = [kind for _, kind in rows]
    assert kinds == ["recording_start", "source_confirmed", "round_start", "mode_requested", "tracking_gap (derived)",
                     "source_confirmed", "tracking_lost", "tracking_reacquired", "controller_link", "recording_stop"]
    assert rows[3][0] == "00:02.0"

    def look(kind: str) -> tuple[str, bool]:
        item = view.topLevelItem(kinds.index(kind))
        return item.foreground(1).color().name(), item.font(1).bold()

    plain = look("tracking_reacquired")
    assert plain[1] is False
    distinct = {kind: look(kind) for kind in ("round_start", "mode_requested", "source_confirmed", "tracking_lost")}
    assert all(style[1] and style[0] != plain[0] for style in distinct.values())
    assert distinct["mode_requested"] == distinct["source_confirmed"]           # both are handoff steps
    assert len({distinct["round_start"][0], distinct["mode_requested"][0], distinct["tracking_lost"][0]}) == 3

    # A real click on the row moves the scrubber (and everything it drives) there.
    screen.seek(5.5)
    item = view.topLevelItem(kinds.index("mode_requested"))
    view.scrollToItem(item)
    app.processEvents()
    QTest.mouseClick(view.viewport(), Qt.MouseButton.LeftButton, pos=view.visualItemRect(item).center())
    app.processEvents()
    assert screen.current_time == pytest.approx(2.0)
    assert screen.slider.value() == 2000
    assert (screen.source_value.text(), screen.requested_value.text()) == ("NONE", "PC")
    assert screen.time_label.text().startswith("00:02.0 / 00:06.0")

    screen.jump_event(1)
    assert screen.current_time == pytest.approx(2.4)                # the confirmed handoff
    assert screen.source_value.text() == "PC"
    screen.next_event_button.click()
    assert screen.current_time == pytest.approx(2.5)
    screen.prev_event_button.click()
    screen.prev_event_button.click()
    assert screen.current_time == pytest.approx(2.0)
    screen.seek(0.0)
    screen.jump_event(-1)
    assert screen.current_time == 0.0
    screen.seek(6.0)
    screen.jump_event(1)
    assert screen.current_time == pytest.approx(6.0)
    assert view.currentItem() is view.topLevelItem(len(kinds) - 1)  # the list follows the scrubber


def test_slider_and_playback_drive_the_same_state(app, screen, tmp_path):
    folder = make_handoff_round(tmp_path)
    assert screen.open_round(folder)[0]
    screen.slider.setValue(1000)                                    # as a user drag would
    assert screen.current_time == pytest.approx(1.0)
    assert screen.source_value.text() == "RADIO" and screen.current_frame_row == 10
    screen.slider.setValue(4000)
    assert screen.source_value.text() == "STALE" and screen.current_frame_row == 40

    screen.seek(0.0)
    screen.play()
    assert screen.playing and screen.play_button.text() == "Pause"
    started = time.monotonic()
    assert wait_until(app, lambda: screen.current_time >= 0.3, 5.0)
    elapsed = time.monotonic() - started
    assert screen.current_time <= elapsed + 0.2                     # real time, not as fast as frames decode
    screen.pause()
    assert not screen.playing
    stopped_at = screen.current_time
    wait_until(app, lambda: False, 0.1)
    assert screen.current_time == stopped_at

    screen.seek(5.9)
    screen.play()
    assert wait_until(app, lambda: not screen.playing, 5.0)         # stops by itself at the end
    assert screen.current_time == pytest.approx(6.0)


def test_readout_helpers_never_report_stale_values(tmp_path):
    folder = make_handoff_round(tmp_path)
    data = RoundData(folder)
    try:
        assert round_duration(data) == pytest.approx(6.0)
        for t in np.arange(0.0, 6.01, 0.05):
            shown = control_readout(data, float(t))
            fresh = data.control_at(float(t))["freshness"] == "FRESH"
            assert (shown["source"] in ("RADIO", "PC", "NONE")) == fresh
            if not fresh:
                assert {shown[k] for k in ("source", "requested", "armed", "flags", "motors")} == {shown["freshness"]}
                assert shown["freshness"] in ("STALE", "UNKNOWN")
        assert control_readout(data, 4.0)["source"] == "STALE"
        assert pose_readout(data, 2.5)["known"] is False
        assert pose_readout(data, 1.0)["source"] == "live"
        derived = [row for row in event_rows(data) if row["derived"]]
        assert len(derived) == 1 and derived[0]["t"] == pytest.approx(2.0) and "1.0 s" in derived[0]["text"]
    finally:
        data.close()
    assert format_time(0.0) == "00:00.0" and format_time(75.26) == "01:15.2" and format_time(-3.0) == "00:00.0"


def test_open_latest_uses_the_hub_rounds_root(app, tmp_path):
    root = tmp_path / "rounds"
    assert latest_round_folder(root) is None
    hub = SimpleNamespace(options=SimpleNamespace(rounds_root=root))
    widget = ReviewScreen(hub)
    try:
        assert widget.rounds_root == root
        ok, message = widget.open_latest()
        assert not ok and "No recorded rounds" in message
        first = make_video_only_round(root)
        (root / "zz-not-a-round").mkdir()
        older = first.rename(root / "2020-01-01_00-00-00_older")
        newer = make_video_only_round(root)
        assert latest_round_folder(root) == newer != older
        ok, message = widget.open_latest()
        assert ok and widget.data.folder == newer
    finally:
        widget.close()
        widget.deleteLater()
        app.processEvents()


# ---------------------------------------------------------------- (c) offline reconstruction


def test_offline_reconstruction_adds_a_dashed_track_distinct_from_live(app, screen, tmp_path, mock_material):
    folder = make_mock_round(tmp_path, mock_material, "mock both", with_live_pose=True)
    ok, message = screen.open_round(folder)
    assert ok, message
    screen.seek(5.9)
    assert [(t.label, t.dashed) for t in screen.map.trails] == [("live", False)]
    assert "No offline reconstruction for this round" in screen.map_legend.text()
    assert not (folder / "derived" / "pose_offline.csv").exists()
    live_csv = (folder / "pose.csv").read_bytes()

    assert screen.reconstruct_offline() is True
    assert screen.reconstructing and not screen.reconstruct_button.isEnabled()
    assert screen.open_round(folder)[0] is False                    # refused while the thread runs
    assert wait_until(app, lambda: not screen.reconstructing, 60.0)
    assert "Reconstructed OFFLINE" in screen.status_label.text(), screen.status_label.text()
    assert (folder / "pose.csv").read_bytes() == live_csv           # the live track is never rewritten
    assert screen.current_time == pytest.approx(5.9)                # the reload keeps the selected time

    data = screen.data
    assert data.pose_offline.source == "offline" and int(data.pose_offline.valid.sum()) >= 40
    assert data.pose_offline_meta and "OFFLINE" in data.pose_offline_meta["source"]
    trails = screen.map.trails
    assert [(t.label, t.dashed) for t in trails] == [("live", False), ("reconstructed offline", True)]
    assert trails[0].colour != trails[1].colour
    assert len(trails[1].runs) >= 2                                 # hidden from 2 s to 3 s: a gap offline too
    legend = screen.map_legend.text()
    assert "dashed" in legend and "reconstructed offline" in legend and "solid" in legend
    assert "own-bot pose, reconstructed offline" in screen.streams_label.text().split("Absent:")[0]
    assert screen.reconstruct_button.isEnabled()

    screen.seek(1.0)                                                # both tracks have a pose here
    assert len(screen.map.markers) == 2
    live, offline = screen.map.markers
    assert (live.hollow, offline.hollow) == (False, True) and live.colour != offline.colour
    assert live.label == "● live   ○ reconstructed offline" and offline.label == ""   # one label: two would overprint
    assert math.hypot(live.x - offline.x, live.y - offline.y) < 0.03
    assert screen.pose_source_value.text() == "live (estimated during the round)"   # live is preferred when it exists
    assert screen.position_plot.series[0].dashed is False and "live" in screen.position_plot.title

    screen.seek(2.5)                                                # hidden: neither track has a position
    assert screen.map.markers == [] and screen.pose_value.text().startswith("position unknown:")
    paint(app, screen)


def test_offline_only_track_is_labelled_offline_everywhere(app, screen, tmp_path, mock_material):
    folder = make_mock_round(tmp_path, mock_material, "mock video only", with_live_pose=False)
    assert screen.open_round(folder)[0]
    screen.seek(1.0)
    assert screen.map.trails == [] and screen.map.note == "No own-bot track in this round"
    assert screen.pose_value.text() == "position unknown: no own-bot track in this round"

    assert screen.reconstruct_offline() is True
    assert wait_until(app, lambda: not screen.reconstructing, 60.0)
    assert "Reconstructed OFFLINE" in screen.status_label.text(), screen.status_label.text()
    assert not (folder / "pose.csv").exists()
    assert [(t.label, t.dashed) for t in screen.map.trails] == [("reconstructed offline", True)]
    assert screen.map.note == ""
    assert len(screen.map.markers) == 1 and screen.map.markers[0].hollow and screen.map.markers[0].label == "reconstructed offline"
    assert screen.pose_value.text().startswith("x ")
    assert screen.pose_source_value.text() == "reconstructed offline (after the round, from the video)"
    assert screen.map_position.text().endswith("[offline]")
    assert "No live own-bot track in this round" in screen.map_legend.text()
    assert "reconstructed offline" in screen.position_plot.title and "reconstructed offline" in screen.heading_plot.title
    assert all(series.dashed for series in screen.position_plot.series + screen.heading_plot.series)
    truth_x = 0.575 + 0.25 * math.cos(2.0 * math.pi * 1.0 / 6.0)   # where the mock drew the bot at t = 1.0 s
    assert screen.map.markers[0].x == pytest.approx(truth_x, abs=0.03)
    screen.tabs.setCurrentIndex(1)
    paint(app, screen)


def test_reconstruction_without_calibration_shows_the_reason(app, screen, tmp_path):
    folder = make_video_only_round(tmp_path)
    assert screen.open_round(folder)[0]
    assert screen.reconstruct_offline() is True
    assert wait_until(app, lambda: not screen.reconstructing, 30.0)
    status = screen.status_label.text()
    assert status.startswith("Offline reconstruction not done:") and "no calibration snapshot" in status
    assert not (folder / "derived" / "pose_offline.csv").exists()
    assert screen.map.trails == []
    assert screen.reconstruct_button.isEnabled()


# ---------------------------------------------------------------- (d) TracePlot


def near(colour: QColor, target: str, tolerance: int = 70) -> bool:
    wanted = QColor(target)
    return all(abs(a - b) <= tolerance for a, b in ((colour.red(), wanted.red()), (colour.green(), wanted.green()),
                                                    (colour.blue(), wanted.blue())))


def column_has(image, x: float, y: float, colour: str) -> bool:
    return any(near(image.pixelColor(int(round(x)) + dx, int(round(y)) + dy), colour)
               for dx in (-1, 0, 1) for dy in (-2, -1, 0, 1, 2))


def test_trace_plot_does_not_connect_across_nan(app):
    plot = TracePlot("Accelerometer · sensor axes", "m/s²", "No IMU stream in this round")
    plot.resize(600, 200)
    t = np.linspace(0.0, 10.0, 101)
    y = np.ones_like(t)
    y[(t > 4.0) & (t < 6.0)] = np.nan
    plot.set_series([Series("X", t, y, colour="#ff00ff")], (0.0, 10.0))
    assert plot.has_data() and plot.message() == ""
    runs = plot.runs(0)
    assert len(runs) == 2
    assert runs[0][-1][0] == pytest.approx(4.0) and runs[1][0][0] == pytest.approx(6.0)
    assert plot.y_range() == pytest.approx((0.5, 1.5))

    plot.set_cursor(8.0)
    image = plot.grab().toImage()
    line_y = plot.value_to_y(1.0)
    assert column_has(image, plot.time_to_x(2.0), line_y, "#ff00ff")
    assert column_has(image, plot.time_to_x(8.5), line_y, "#ff00ff")
    for gap_t in (4.5, 5.0, 5.5):
        assert not column_has(image, plot.time_to_x(gap_t), line_y, "#ff00ff")
    cursor_x = int(round(plot.time_to_x(8.0)))
    assert any(near(image.pixelColor(cursor_x + dx, int(plot.plot_rect().top()) + 20), "#f2f5f8", 40)
               for dx in (-1, 0, 1))                                # the cursor line
    assert plot.value_at_cursor(0) == pytest.approx(1.0)
    plot.set_cursor(5.0)
    assert plot.value_at_cursor(0) is None                          # in the gap there is no value to read


def test_trace_plot_series_rules(app):
    assert finite_runs(np.array([]), np.array([])) == []
    assert finite_runs(np.array([0.0, 1.0]), np.array([np.nan, np.nan])) == []
    t = np.array([0.0, 0.1, 0.2, 2.0, 2.1])
    assert len(finite_runs(t, np.zeros(5))) == 1
    assert [len(run) for run in finite_runs(t, np.zeros(5), max_gap_s=0.5)] == [3, 2]   # a pause in the samples is a gap
    assert [len(run) for run in finite_runs(t, np.array([0.0, np.nan, 0.0, 0.0, 0.0]))] == [1, 3]

    plot = TracePlot("Gyro", "deg/s")
    plot.resize(500, 160)
    with pytest.raises(ValueError):
        plot.set_series([("a", t, t), ("b", t, t), ("c", t, t), ("d", t, t)])
    plot.set_series([("X", t, t), ("Y", t, 2 * t), ("Z", t, -t)])
    assert len(plot.series) == 3 and len({s.colour for s in plot.series}) == 3
    assert plot.time_range() == pytest.approx((0.0, 2.1))
    assert not plot.grab().isNull()

    dense_t = np.linspace(0.0, 60.0, 30000)                         # far more samples than pixels
    plot.set_series([Series("X", dense_t, np.sin(dense_t))], (0.0, 60.0))
    assert not plot.grab().isNull()
    plot.set_series([Series("X", np.array([1.0]), np.array([3.0]))], (0.0, 2.0))   # one sample, flat range
    assert plot.y_range() == pytest.approx((2.5, 3.5))
    assert not plot.grab().isNull()


def test_trace_plot_without_data_shows_message_and_ignores_clicks(app):
    plot = TracePlot("Accelerometer", "m/s²", "No IMU stream in this round")
    plot.resize(600, 200)
    clicks: list[float] = []
    plot.time_clicked.connect(clicks.append)
    for series in ([], [Series("X", np.array([]), np.array([]))], [Series("X", np.arange(3.0), np.full(3, np.nan))]):
        plot.set_series(series, (0.0, 10.0))
        plot.set_cursor(5.0)
        assert not plot.has_data()
        assert plot.message() == "No IMU stream in this round"
        image = plot.grab().toImage()
        centre = plot.plot_rect().center()
        lit = sum(1 for dx in range(-120, 121) for dy in range(-8, 9)
                  if image.pixelColor(int(centre.x()) + dx, int(centre.y()) + dy).lightness() > 110)
        assert lit > 30                                             # the centred message was drawn
        QTest.mouseClick(plot, Qt.MouseButton.LeftButton, pos=QPoint(300, 100))
    assert clicks == []


def test_trace_plot_click_reports_time(app):
    plot = TracePlot("Camera position", "m")
    plot.resize(600, 200)
    plot.show()
    app.processEvents()
    t = np.linspace(0.0, 20.0, 201)
    plot.set_series([Series("x", t, 0.1 * t), Series("y", t, 0.5 + 0.0 * t)], (0.0, 20.0))
    clicks: list[float] = []
    plot.time_clicked.connect(clicks.append)
    rect = plot.plot_rect()
    for wanted in (0.0, 5.0, 12.5, 20.0):
        QTest.mouseClick(plot, Qt.MouseButton.LeftButton, pos=QPoint(int(round(plot.time_to_x(wanted))), int(rect.center().y())))
    assert clicks == pytest.approx([0.0, 5.0, 12.5, 20.0], abs=20.0 / rect.width())
    QTest.mouseClick(plot, Qt.MouseButton.LeftButton, pos=QPoint(2, int(rect.center().y())))    # left of the axis: clamped
    assert clicks[-1] == pytest.approx(0.0)
    plot.close()
