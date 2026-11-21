"""Command Centre: status model, controller telemetry (listen-only), capture + recording, Operate screen.

Everything here runs on mocks: a synthetic camera, scripted controller lines and temporary folders.
No real camera is opened and no serial port is touched.
"""
import ast
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6.QtCore import QCoreApplication
from PySide6.QtWidgets import QApplication, QPushButton

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from control_center.app import build_parser, hub_options, load_config  # noqa: E402
from control_center.backend import status as st  # noqa: E402
from control_center.backend.controller_link import (ClientLogSource, ControllerMonitor, ScriptedSource,  # noqa: E402
                                                    SerialLineSource, parse_imu_line, parse_status_line, status_line)
from control_center.backend.hub import HubOptions, OperationsHub  # noqa: E402
from control_center.backend.live_vision import load_live_clicks  # noqa: E402
from control_center.backend.round_reader import RoundData  # noqa: E402
from control_center.backend.round_recorder import RoundRecorder, new_round_folder, pose_row  # noqa: E402
from control_center.tools.overlay_core import BotPose  # noqa: E402

CONFIG = load_config(ROOT / "config" / "control_center.yaml")


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def pump(seconds: float) -> None:
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        QCoreApplication.processEvents()
        time.sleep(0.005)


def frame(value: int = 0, size=(160, 120)) -> np.ndarray:
    image = np.full((size[1], size[0], 3), 40, np.uint8)
    image[:, : 1 + value % size[0]] = (0, 200, 0)
    return image


# ---------------------------------------------------------------- controller telemetry

RADIO = dict(armed=1, req="RADIO", act="RADIO", block="NONE", token=2, left=90, right=90)
WAIT_PC = dict(armed=1, req="PC", act="NONE", block="WAIT_PC_NEUTRAL", stop="HANDOFF", token=3)
PC = dict(armed=1, req="PC", act="PC", pc_ok=1, block="NONE", stop="HANDOFF", token=3, left=60, right=60)


def test_status_parser_matches_the_bench_client_on_the_same_lines():
    import handoff_bench_client as bench

    for fields in (dict(), RADIO, WAIT_PC, PC, dict(link=0, block="RADIO_LOST", fault="RADIO_LOST", stop="RADIO_LOST")):
        line = status_line(1234, **fields)
        mine, theirs = parse_status_line(line), bench.parse_status(line)
        assert (mine.token, mine.link, mine.armed, mine.requested, mine.active, mine.pc_ok, mine.block, mine.fault, mine.stop,
                mine.left, mine.right) == (theirs.token, theirs.link, theirs.armed, theirs.requested, theirs.active,
                                           theirs.pc_ok, theirs.block, theirs.fault, theirs.stop, theirs.left, theirs.right)
    for bad in ("ST", "ST t=1 tok=1", status_line(1).replace("armed=0", "armed=2"), status_line(1) + " extra=1", "CH ch1=992"):
        assert parse_status_line(bad) is None and bench.parse_status(bad) is None
    sample = parse_imu_line("IMU,1500,0.12,-0.05,9.807,0.1,-0.2,12.5")
    assert sample.device_ms == 1500 and sample.accel_m_s2 == (0.12, -0.05, 9.807) and sample.gyro_deg_s == (0.1, -0.2, 12.5)
    assert parse_imu_line("IMU,1500,0.12,-0.05,nan,0.1,-0.2,12.5") is None and parse_imu_line("IMU,1,2,3") is None


def test_control_source_is_unknown_until_heard_and_stale_when_reports_stop():
    monitor = ControllerMonitor()
    unknown = st.control_display(monitor.authority(10.0))
    assert (unknown.source.text, unknown.armed.text, unknown.requested.text) == ("UNKNOWN", "UNKNOWN", "UNKNOWN")

    events = monitor.feed(status_line(1000, **RADIO), 10.0)
    assert [kind for kind, _, _ in events] == ["controller_link", "source_confirmed"]
    fresh = st.control_display(monitor.authority(10.3))
    assert (fresh.source.text, fresh.armed.text, fresh.requested.text) == ("RADIO", "ARMED", "RADIO")

    assert monitor.poll(10.5) == []
    stale_events = monitor.poll(10.7)
    assert [kind for kind, _, _ in stale_events] == ["controller_link"] and "STALE" in stale_events[0][1]
    assert monitor.poll(10.8) == []                                   # reported once, not every tick
    stale = st.control_display(monitor.authority(12.0))
    assert stale.source.text == "STALE" and stale.source.level == "bad" and stale.armed.text == "UNKNOWN"
    assert "RADIO" not in (stale.source.text, stale.requested.text)   # the old value is never shown as current
    assert "no longer confirmed" in stale.detail

    back = monitor.feed(status_line(4000, **RADIO), 13.0)
    assert [kind for kind, _, _ in back] == ["controller_link", "source_confirmed"]


def test_requested_mode_and_confirmed_source_are_separate():
    monitor = ControllerMonitor()
    monitor.feed(status_line(1000, **RADIO), 1.0)
    events = monitor.feed(status_line(1200, **WAIT_PC), 1.2)
    assert [kind for kind, _, _ in events] == ["mode_requested", "source_confirmed"]
    display = st.control_display(monitor.authority(1.3))
    assert display.requested.text == "PC" and display.source.text == "NONE"        # asked for, not confirmed
    assert display.requested.level == "warn" and "WAIT_PC_NEUTRAL" in display.detail
    events = monitor.feed(status_line(1400, **PC), 1.4)
    assert [kind for kind, _, _ in events] == ["source_confirmed"]
    assert st.control_display(monitor.authority(1.5)).source.text == "PC"
    events = monitor.feed(status_line(1600, armed=0, req="PC", act="NONE", block="REARM_REQUIRED", fault="PC_STALE", stop="PC_STALE", token=4), 1.6)
    assert [kind for kind, _, _ in events] == ["disarm", "source_confirmed", "fault"]
    # A command the PC SENT changes nothing about who has control.
    monitor.feed("D,4,80,0", 1.7, direction="TX")
    assert monitor.last_pc_command.throttle == 80 and st.control_display(monitor.authority(1.7)).source.text == "NONE"
    # Unchanged status lines produce no events (the log is not flooded).
    assert monitor.feed(status_line(1800, armed=0, req="PC", act="NONE", block="REARM_REQUIRED", fault="PC_STALE", stop="PC_STALE", token=4), 1.8) == []


def test_status_wording():
    assert st.camera_map_caption("LIVE", "Map not calibrated") == "Camera live · Map not calibrated"
    assert st.camera_map_caption("MOCK", "Map calibrated").startswith("MOCK camera")
    assert st.camera_map_caption("UNAVAILABLE", "Map calibrated") == "No camera · Map calibrated"
    assert st.camera_item("LIVE", "", 30.0, 0.02).text == "Camera live · 30 fps"
    assert st.camera_item("LIVE", "", 30.0, 2.5).level == "bad"
    assert st.own_bot_item(True, "none", False, "no valid floor mapping", [0, 1], [], None).text.startswith("No position (map not calibrated)")
    assert st.own_bot_item(True, "ok", False, "no configured bot markers visible", [], [], 2.0).text.startswith("LOST for 2.0 s")
    assert st.own_bot_item(True, "ok", True, "ok", [0, 1], [0, 1], 0.0).level == "ok"
    assert st.opponent_item(False).text == "Not available live" and "recorded footage" in st.opponent_item(False).detail
    assert st.policy_item("OFF", None, None).text == "OFF"
    assert st.policy_item("SIM", "simulator observations", "clone-x").text.startswith("SIM")
    assert st.imu_item(False, None, True).text == "No IMU stream" and st.imu_item(True, 50.2, True).text == "IMU stream live · 50 Hz"
    assert st.robot_link_item(ControllerMonitor().authority(0.0), None).text == "Not connected"
    assert st.format_elapsed(125.9) == "02:05"


def test_client_log_source_follows_new_lines_only(tmp_path):
    log = tmp_path / "bench.txt"
    log.write_text("1.0 RX " + status_line(1, **RADIO) + "\n", encoding="utf-8")
    source = ClientLogSource(log)
    assert source.read() == []                                        # old lines are not replayed as live
    with log.open("a", encoding="utf-8") as handle:
        handle.write("2.0 RX " + status_line(2, **PC) + "\n2.1 TX D,3,60,0\n2.2 RX partial")
    assert source.read() == [("RX", status_line(2, **PC)), ("TX", "D,3,60,0")]
    with log.open("a", encoding="utf-8") as handle:
        handle.write(" line\n")
    assert source.read() == [("RX", "partial line")]
    source.close()


# ---------------------------------------------------------------- recorder


def test_video_only_recording_needs_nothing_else(tmp_path):
    recorder = RoundRecorder(new_round_folder(tmp_path, "video only!"), {"label": "video only", "calibration": None}, fps=30)
    recorder.start()
    for index in range(45):
        assert recorder.add_frame(index, recorder.t0_mono + index / 30.0, frame(index))
    recorder.mark_round_start(recorder.t0_mono + 0.5)
    summary = recorder.stop()
    assert summary.frames_written == 45 and summary.frames_dropped == 0 and summary.write_errors == []
    assert summary.video_verified, summary.video_verify_detail
    assert summary.folder.name.endswith("_video-only") and "no dropped frames" in summary.headline()
    present = {name: info["present"] for name, info in summary.streams.items()}
    assert present == {"video": True, "frames": True, "imu": False, "pose": False, "opponent": False,
                       "control": False, "policy": False, "events": True}
    assert sorted(path.name for path in summary.folder.iterdir()) == ["events.jsonl", "frames.csv", "manifest.json", "video.avi"]

    data = RoundData(summary.folder)
    assert data.manifest["complete"] and data.manifest["video"]["overlays_baked_in"] is False and not data.calibrated
    assert [event["kind"] for event in data.events] == ["recording_start", "round_start", "recording_stop"]
    assert data.control_at(1.0)["source"] == "UNKNOWN" and data.best_pose_track() is None and len(data.imu_t) == 0
    shown = data.read_frame(data.frame_at(1.0))
    assert shown.shape == (120, 160, 3)
    assert abs(float(data.frame_t[data.frame_at(1.0)]) - 1.0) < 0.02
    data.close()


def test_slow_disk_drops_frames_visibly_and_never_blocks_the_caller(tmp_path):
    recorder = RoundRecorder(new_round_folder(tmp_path), {}, fps=30, max_queued_frames=2)
    original = recorder._open_video

    def slow_open(first):
        original(first)
        real = recorder._writer

        class Slow:
            def write(self, image):
                time.sleep(0.05)
                real.write(image)

            def release(self):
                real.release()

        recorder._writer = Slow()

    recorder._open_video = slow_open
    recorder.start()
    started = time.perf_counter()
    accepted = [recorder.add_frame(index, recorder.t0_mono + index / 30.0, frame(index)) for index in range(40)]
    assert time.perf_counter() - started < 0.5                        # 40 frames queued or dropped without waiting for the disk
    summary = recorder.stop()
    assert summary.frames_dropped == accepted.count(False) > 0 and summary.frames_written == accepted.count(True)
    assert "dropped" in summary.headline()
    data = RoundData(summary.folder)
    assert int((~data.frame_written).sum()) == summary.frames_dropped and len(data.frame_t) == 40
    dropped_row = int(np.flatnonzero(~data.frame_written)[0])
    assert data.read_frame(dropped_row) is None                       # a dropped frame is a gap, not a neighbour
    assert data.manifest["video"]["frames_dropped"] == summary.frames_dropped
    data.close()


def test_write_failure_is_reported_and_telemetry_continues(tmp_path):
    recorder = RoundRecorder(new_round_folder(tmp_path), {}, fps=30)
    recorder.start()
    recorder.add_frame(0, recorder.t0_mono, frame(0))
    recorder.add_frame(1, recorder.t0_mono + 0.03, frame(1, size=(320, 240)))       # wrong size: refused, not fatal
    recorder.log_imu(recorder.t0_mono + 0.04, 40, (0.1, 0.2, 9.8), (1.0, 2.0, 3.0))
    recorder.add_frame(2, recorder.t0_mono + 0.06, frame(2))
    summary = recorder.stop()
    assert summary.frames_written == 2 and len(summary.write_errors) == 1 and "320x240" in summary.write_errors[0]
    assert summary.streams["imu"]["rows"] == 1 and "write error" in summary.headline()


def test_round_left_open_by_a_crash_still_loads(tmp_path):
    recorder = RoundRecorder(new_round_folder(tmp_path), {}, fps=30)
    recorder.start()
    for index in range(10):
        recorder.add_frame(index, recorder.t0_mono + index / 30.0, frame(index))
    time.sleep(1.8)                                                   # the writer flushes every second
    data = RoundData(recorder.folder)                                 # read while the recorder still has it open
    assert data.manifest["complete"] is False and any("not closed cleanly" in text for text in data.warnings)
    assert len(data.frame_t) == 10
    data.close()
    recorder.stop()


def test_pose_gap_is_logged_as_a_gap_not_the_last_position(tmp_path):
    good = BotPose(state="ok", reason="ok", x_m=0.5, y_m=0.4, heading_rad=1.0, orientation="upright",
                   position_quality="approximate", visible_ids=[0, 1], used_ids=[0, 1])
    lost = BotPose(reason="no configured bot markers visible")
    assert pose_row(lost, False) == {"valid": 0, "x_m": "", "y_m": "", "heading_deg": "", "orientation": "unknown",
                                     "position_quality": "unknown", "used_ids": "", "visible_ids": "", "tag_switch": 0,
                                     "reason": "no configured bot markers visible"}
    recorder = RoundRecorder(new_round_folder(tmp_path), {}, fps=10)
    recorder.start()
    for index in range(20):
        t = recorder.t0_mono + index / 10.0
        recorder.add_frame(index, t, frame(index))
        recorder.log_pose(t, index, pose_row(lost if 8 <= index < 14 else good, tag_switch=index == 14))
    summary = recorder.stop()
    data = RoundData(summary.folder)
    track = data.pose_live
    assert int(track.valid.sum()) == 14 and np.isnan(track.x[8:14]).all()
    assert track.gaps() == [(pytest.approx(0.8), pytest.approx(1.4))]
    assert [len(run) for run in track.segments()] == [8, 6]           # two runs: the gap is not bridged
    assert track.at(1.0)["valid"] is False and track.at(1.0)["reason"] == "no configured bot markers visible"
    assert track.at(0.5)["valid"] is True and track.at(0.5)["x_m"] == 0.5
    data.close()


def test_a_historical_footage_profile_is_never_used_as_the_live_calibration(tmp_path):
    points = tmp_path / "clicks.json"
    points.write_text(json.dumps({"points": {"a": [1, 2]}, "image_size": [640, 360], "profile_kind": "historical"}), encoding="utf-8")
    clicks, reason = load_live_clicks(points)
    assert clicks is None and "HISTORICAL" in reason
    assert load_live_clicks(tmp_path / "missing.json") == (None, "no live calibration saved yet")
    real = ROOT / "config" / "overlay_profiles" / "historical_bristol_2026_03_22" / "clicks.json"
    if real.exists():
        assert load_live_clicks(real)[0] is None


# ---------------------------------------------------------------- the whole thing on mocks


def script():
    lines = [(0.0, "RX", "HELLO fw=MOCK-not-hardware arm=CH5:HIGH pc=CH6:HIGH pc_limit=128")]
    for start, end, fields in ((0.0, 2.0, RADIO), (2.0, 2.4, WAIT_PC), (2.4, 4.0, PC), (5.2, 9.0, PC)):   # 4.0-5.2: nothing
        t = start
        while t < end:
            lines.append((t, "RX", status_line(int(t * 1000), **fields)))
            if fields is PC:
                lines.append((t, "TX", "D,3,60,0"))
            t += 0.1
    t = 0.0
    while t < 9.0:
        lines.append((t, "RX", f"IMU,{int(t * 1000)},0.1,0.0,9.8,0.0,0.0,5.0"))
        t += 0.02
    return lines


@pytest.fixture(scope="module")
def recorded(app, tmp_path_factory):
    """One mock session: record through a RADIO -> PC handoff, a telemetry gap and a tracking dropout."""
    root = tmp_path_factory.mktemp("rounds")
    hub = OperationsHub(CONFIG, HubOptions(mock=True, rounds_root=root))
    hub.mock_source.hide = (3.0, 4.5)             # the bot disappears for 1.5 s in every 16 s
    hub.mock_source.single_tag = (99.0, 99.0)
    seen = []
    hub.event_added.connect(lambda stamp, kind, text: seen.append((kind, text)))
    finished = []
    hub.recording_finished.connect(finished.append)
    hub.start()
    pump(1.2)                                     # camera up, first poses
    assert hub.mark_round_start() == (False, "not recording: start Record Round first")
    hub.connect_controller(ScriptedSource(script(), label="MOCK test script"))
    ok, folder = hub.start_recording("mode change")
    assert ok, folder
    pump(1.0)
    marked = hub.mark_round_start()
    policy_before = dict(hub.policy)
    pump(5.5)
    snapshot = hub.snapshot()
    refused = hub.set_policy("LIVE SHADOW", "live camera", "any-checkpoint")
    sim = hub.set_policy("SIM", "simulator observations", "mock-checkpoint", proposed={"tactic": "advance"})
    hub.stop_recording()
    end = time.monotonic() + 20
    while not finished and time.monotonic() < end:
        pump(0.05)
    hub.set_policy("OFF")
    hub.close()
    assert finished, "recording did not finish"
    return {"summary": finished[0], "events": seen, "snapshot": snapshot, "marked": marked, "refused": refused, "sim": sim,
            "policy_before": policy_before, "hub": hub}


def test_recording_continues_through_a_mode_change_and_a_telemetry_gap(recorded):
    summary = recorded["summary"]
    assert summary.video_verified and summary.write_errors == [] and summary.frames_dropped == 0
    assert summary.frames_written >= 60 and 6.0 < summary.duration_s < 9.0
    data = RoundData(summary.folder)
    gaps = np.diff(data.frame_t)
    assert gaps.max() < 0.5                                            # no hole in the video at the handoff or anywhere else
    assert data.manifest["mock"] is True and data.manifest["calibration"]["synthetic_fixture"] is True
    assert {name: data.stream_present(name) for name in ("video", "imu", "pose", "control", "events", "opponent")} == {
        "video": True, "imu": True, "pose": True, "control": True, "events": True, "opponent": False}

    sources = [record["active"] for record in data.controller_status]
    assert sources[0] == "RADIO" and "NONE" in sources and sources[-1] == "PC"
    switch = next(record["t_s"] for record in data.controller_status if record["requested"] == "PC")
    confirm = next(record["t_s"] for record in data.controller_status if record["active"] == "PC")
    assert confirm > switch                                            # requested first, confirmed later
    before, waiting, after = data.control_at(switch - 0.3), data.control_at((switch + confirm) / 2), data.control_at(confirm + 0.5)
    assert (before["source"], before["requested"]) == ("RADIO", "RADIO")
    assert (waiting["source"], waiting["requested"], waiting["armed"]) == ("NONE", "PC", True)
    assert (after["source"], after["requested"]) == ("PC", "PC")
    times = np.array(data._status_t)
    widest = int(np.argmax(np.diff(times)))
    assert times[widest + 1] - times[widest] > 1.0                     # the scripted 1.2 s with no controller lines
    stale = data.control_at(float(times[widest]) + 0.9)
    assert stale["freshness"] == "STALE" and stale["source"] == "STALE" and stale["armed"] is None

    assert data.pc_sent and all(record["note"].startswith("sent by the PC client") for record in data.pc_sent)
    assert data.pc_sent_at(switch - 0.3) is None and data.pc_sent_at(confirm + 0.5)["throttle"] == 60
    assert 200 < len(data.imu_t) and np.allclose(data.imu["az_m_s2"], 9.8) and np.all(np.diff(data.imu_device_ms) > 0)
    kinds = [event["kind"] for event in data.events]
    for expected in ("recording_start", "round_start", "mode_requested", "source_confirmed", "controller_link", "recording_stop"):
        assert expected in kinds, expected
    assert kinds[0] == "recording_start" and kinds[-1] == "recording_stop" and kinds.count("recording_start") == 1
    assert data.policy and data.policy[-1]["mode"] == "SIM" and "never sent" in data.policy[-1]["note"]
    data.close()


def test_a_tracking_dropout_is_a_gap_in_the_log_and_two_events(recorded):
    data = RoundData(recorded["summary"].folder)
    track = data.pose_live
    assert len(track) >= 40 and 0 < int((~track.valid).sum()) < len(track)
    gaps = track.gaps(min_s=0.8)
    assert len(gaps) == 1 and 1.0 < gaps[0][1] - gaps[0][0] < 2.2      # the bot was hidden for 1.5 s
    assert len(track.segments()) >= 2
    inside = track.at((gaps[0][0] + gaps[0][1]) / 2)
    assert inside["valid"] is False and inside["reason"] == "no configured bot markers visible"
    kinds = [event["kind"] for event in data.events]
    assert "tracking_lost" in kinds and "tracking_reacquired" in kinds
    assert kinds.index("tracking_lost") < kinds.index("tracking_reacquired")
    assert kinds.count("tracking_lost") <= 2                           # an event per change, not per frame
    # The video kept going through the dropout.
    during = data.read_frame(data.frame_at((gaps[0][0] + gaps[0][1]) / 2))
    assert during is not None and during.shape[:2] == (540, 960)
    data.close()


def test_mark_round_start_only_adds_a_timestamp(recorded):
    ok, text = recorded["marked"]
    assert ok and text.startswith("ROUND START marked at 00:0")
    data = RoundData(recorded["summary"].folder)
    marks = [event for event in data.events if event["kind"] == "round_start"]
    assert len(marks) == 1 and 0.8 < marks[0]["t_s"] < 1.6
    assert recorded["policy_before"]["mode"] == "OFF"                  # it did not start or change anything
    data.close()


def test_live_shadow_is_refused_and_sim_is_display_only(recorded):
    ok, reason = recorded["refused"]
    assert not ok and reason.startswith("LIVE SHADOW unavailable")
    assert "live observation adapter" in reason and "opponent" in reason
    assert recorded["sim"] == (True, "ok")
    assert recorded["snapshot"].policy.text == "OFF"                    # before SIM was switched on
    assert recorded["snapshot"].opponent.text == "Not available live"
    assert recorded["snapshot"].camera_map_caption.startswith("MOCK camera")


def test_no_module_of_the_dashboard_can_write_to_the_robot():
    """There is no motor-write path: only controller_link touches a port, and it only reads."""
    package = ROOT / "control_center"
    serial_users = []
    for path in package.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        names = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names |= {alias.name.split(".")[0] for alias in node.names}
            elif isinstance(node, ast.ImportFrom) and node.module:
                names.add(node.module.split(".")[0])
        if "serial" in names:
            serial_users.append(path.relative_to(package).as_posix())
    assert serial_users == ["backend/controller_link.py"]
    link = (package / "backend" / "controller_link.py").read_text(encoding="utf-8")
    code = "\n".join(line.split("#")[0] for line in link.splitlines())
    assert ".write(" not in code and "sendall" not in code and "send_command" not in code
    assert not any(hasattr(SerialLineSource, name) for name in ("write", "send", "send_command"))
    hub_code = (package / "backend" / "hub.py").read_text(encoding="utf-8")
    assert ".write(" not in hub_code and 'f"D,' not in hub_code and "format_command" not in hub_code


# ---------------------------------------------------------------- Operate screen


@pytest.fixture
def operate(app, tmp_path):
    from control_center.screens.operate import OperateScreen

    hub = OperationsHub(CONFIG, HubOptions(mock=True, start_camera=False, rounds_root=tmp_path))
    screen = OperateScreen(hub)
    yield hub, screen
    screen._display_timer.stop()
    hub.close()


def test_operate_shows_unknown_then_confirmed_then_stale(operate):
    hub, screen = operate
    screen.show_snapshot(hub.snapshot())
    assert screen.source_label.text() == "UNKNOWN" and screen.armed_label.text() == "UNKNOWN"
    assert screen.rows["robot_link"].value.text() == "Not connected"
    assert screen.rows["opponent"].value.text() == "Not available live"
    assert screen.rows["policy"].value.text() == "OFF"

    hub.link_label = "MOCK test"
    hub.monitor.feed(status_line(1000, **WAIT_PC), time.monotonic())
    screen.show_snapshot(hub.snapshot())
    assert (screen.source_label.text(), screen.requested_label.text(), screen.armed_label.text()) == ("NONE", "PC", "ARMED")
    hub.monitor.feed(status_line(1200, **PC), time.monotonic())
    screen.show_snapshot(hub.snapshot())
    assert (screen.source_label.text(), screen.requested_label.text()) == ("PC", "PC")

    hub.monitor.status_at -= 3.0                                       # no report for 3 s
    screen.show_snapshot(hub.snapshot())
    assert screen.source_label.text() == "STALE" and screen.armed_label.text() == "UNKNOWN"
    assert screen.rows["robot_link"].value.text().startswith("Telemetry STALE")


def test_operate_has_recording_controls_and_no_arm_or_handoff_control(operate):
    hub, screen = operate
    labels = [button.text().upper() for button in screen.findChildren(QPushButton)]
    assert any("RECORD ROUND" in text for text in labels) and any("STOP RECORDING" in text for text in labels)
    assert any("MARK ROUND START" in text for text in labels)
    for forbidden in ("ARM", "DISARM", "TAKEOVER", "AUTO", "HANDOFF", "ENABLE", "DRIVE"):
        assert not any(forbidden in text.replace("MARK", "") for text in labels), forbidden
    screen.show_snapshot(hub.snapshot())
    assert not screen.record_button.isHidden() and screen.stop_button.isHidden() and screen.mark_button.isHidden()
    screen.start_recording()                                           # no camera frames in this fixture
    assert "no camera frames yet" in screen.record_note.text() and hub.recorder is None
    screen.mark_round_start()
    assert "NOT marked" in screen.record_note.text()
    assert screen.caption.text() == "Opening camera · Map calibration loaded (waiting for a frame)"


def test_entry_point_keeps_the_first_dashboard_and_reads_the_camera_index():
    args = build_parser().parse_args(["--camera-index", "3", "--controller-log", "bench.txt"])
    config = dict(CONFIG)
    assert int(CONFIG["camera_index"]) == 1                            # the C525 on the bench PC; still configurable
    config["camera_index"] = args.camera_index
    options = hub_options(args, config)
    assert options.camera_index == 3 and options.controller_log == Path("bench.txt") and not options.mock
    assert options.rounds_root == ROOT / "rounds" and options.codec == "MJPG"
    assert build_parser().parse_args(["--legacy"]).legacy
    from control_center.main_window import MainWindow                  # the first version still imports

    assert MainWindow.__name__ == "MainWindow"
    review = hub_options(build_parser().parse_args(["--review", "rounds/x"]), config)
    assert review.start_camera is False                                # reviewing a round opens no camera


def test_a_missing_camera_is_reported_retried_and_never_blocks_the_rest(app, tmp_path, monkeypatch):
    from control_center.backend import capture

    monkeypatch.setattr(capture, "RETRY_S", 0.2)
    hub = OperationsHub(CONFIG, HubOptions(mock=True, rounds_root=tmp_path))
    attempts = []

    def opener():
        attempts.append(time.monotonic())
        if len(attempts) < 3:
            raise RuntimeError("camera index 7 did not open (backend dshow); try another --camera index")
        return hub.mock_source

    hub.capture._opener = opener
    hub.start()
    pump(0.25)
    snapshot = hub.snapshot()
    assert snapshot.camera.text == "No camera" and "camera index 7 did not open" in snapshot.camera.detail
    assert snapshot.camera_map_caption.startswith("No camera · ")
    assert snapshot.control.source.text in ("NONE", "UNKNOWN")         # controller telemetry does not depend on the camera
    assert hub.start_recording("x") == (False, "no camera frames yet: recording needs video")
    pump(1.2)
    assert len(attempts) >= 3 and hub.capture.state == "MOCK"          # it kept retrying and recovered by itself
    hub.close()


def test_the_controller_state_at_the_moment_recording_starts_is_in_the_round(app, tmp_path):
    hub = OperationsHub(CONFIG, HubOptions(mock=True, rounds_root=tmp_path))
    finished = []
    hub.recording_finished.connect(finished.append)
    hub.start()
    pump(1.2)                                                          # the mock controller is already reporting
    assert hub.snapshot().control.source.text == "NONE" and hub.snapshot().control.armed.text == "DISARMED"
    assert hub.start_recording("")[0]
    pump(0.6)
    hub.stop_recording(wait=True)
    hub.close()
    data = RoundData(hub.last_summary.folder)
    at_start = data.control_at(0.0)
    assert (at_start["freshness"], at_start["source"], at_start["armed"]) == ("FRESH", "NONE", False)
    assert data.link_changes[0]["text"] == "telemetry FRESH at recording start"
    data.close()


def test_dashboard_opens_all_four_screens_and_review_mode_opens_no_camera(app, tmp_path):
    from control_center.dashboard import TAB_TITLES, TABS, DashboardWindow

    recorder = RoundRecorder(new_round_folder(tmp_path, "to review"), {"label": "to review"}, fps=30)
    recorder.start()
    for index in range(20):
        recorder.add_frame(index, recorder.t0_mono + index / 30.0, frame(index))
    folder = recorder.stop().folder

    args = build_parser().parse_args(["--review", str(folder), "--rounds-dir", str(tmp_path), "--state-file", str(tmp_path / "state.json")])
    window = DashboardWindow(dict(CONFIG), hub_options(args, dict(CONFIG)), start_tab=args.tab, review_folder=args.review)
    try:
        assert [window.tabs.tabText(i) for i in range(window.tabs.count())] == [TAB_TITLES[key] for key in TABS]
        assert [type(screen).__name__ for screen in (window.operate, window.setup, window.review, window.diagnostics)] == [
            "OperateScreen", "SetupScreen", "ReviewScreen", "DiagnosticsScreen"]          # none fell back to a placeholder
        assert window.tabs.currentIndex() == TABS.index("review")
        assert not window.hub.capture.isRunning() and not window.hub.vision_thread.isRunning()   # no camera was opened
        assert window.hub.link_thread is None                                                  # and no controller port
        pump(0.3)
        assert window.operate.source_label.text() == "UNKNOWN"
    finally:
        window.close()
