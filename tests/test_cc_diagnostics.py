"""Diagnostics screen: sim-to-real comparison and policy shadow.

Everything here runs against files and the simulator. No camera, no serial port, no hardware.
The one slow test (`test_slow_...`) imports torch and steps the simulator; set AUTOBOT_SKIP_SLOW=1 to skip it.
"""
import ast
import json
import os
import re
import shutil
import time
import zipfile
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

from control_center.app import load_config
from control_center.backend import shadow, sim_compare
from control_center.backend.hub import HubOptions, OperationsHub
from control_center.backend.round_reader import RoundData
from control_center.backend.round_recorder import RoundRecorder

ROOT = Path(__file__).resolve().parents[1]
CLONE = ROOT / "runs" / "recovery-challenger-v2-seed1" / "final_model.zip"
PPO = ROOT / "runs" / "ppo-curriculum-overnight" / "final_model.zip"
REAL_LOG = ROOT / "artifacts" / "human-vs-final-default.jsonl"
SOURCES = [ROOT / "control_center" / "backend" / "shadow.py", ROOT / "control_center" / "backend" / "sim_compare.py",
           ROOT / "control_center" / "screens" / "diagnostics.py"]


def needs(path):
    if not path.is_file():
        pytest.skip(f"{path} not present (runs/ and artifacts/ are not tracked)")


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def hub(app, tmp_path):
    hub = OperationsHub(load_config(ROOT / "config" / "control_center.yaml"),
                        HubOptions(mock=True, start_camera=False, rounds_root=tmp_path))
    yield hub
    hub.close()


# ----------------------------------------------------------------------------------------------
# Fabricated inputs


def write_zip(path, num_timesteps=0, observation=19, actions=5, metadata=None):
    data = {"num_timesteps": num_timesteps, "observation_space": {"_shape": [observation]}, "action_space": {"n": str(actions)}}
    if metadata is not None:
        data["autobot_observation"] = metadata
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("data", json.dumps(data))
    return path


def write_sim_log(path, turn="turn_left", contact_from=None, poses=False):
    """Round 1: 1.0 s advance, 0.5 s stop, 1.0 s turn. Round 2 (after a reset line): 0.2 s advance."""
    commands = {"advance": [-1, -1], "stop": [0, 0], "turn_left": [0.7, -0.7], "turn_right": [-0.7, 0.7]}
    lines = [json.dumps({"type": "reset", "time_s": 0.0})]
    x, yaw = 0.30, 0.0
    for index, tactic in enumerate(["advance"] * 50 + ["stop"] * 25 + [turn] * 50):
        record = {"type": "step", "time_s": 0.02 * (index + 1), "blue_tactic": tactic, "blue_command": commands[tactic],
                  "red_command": [0.0, 0.0], "end_reason": None}
        if contact_from is not None:
            record["opponent_contact"] = index >= contact_from
        if poses:
            x += 0.008 if tactic == "advance" else 0.0
            yaw += np.radians(1.0) if tactic == turn else 0.0
            record["blue_pose"] = [x, 0.5, yaw]
        lines.append(json.dumps(record))
    lines.append(json.dumps({"type": "reset", "time_s": 0.0}))
    lines += [json.dumps({"type": "step", "time_s": 0.02 * (i + 1), "blue_tactic": "advance", "blue_command": [-1, -1],
                          "red_command": [0.0, 0.0], "end_reason": None}) for i in range(10)]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


class Clock:
    def __init__(self, t):
        self.t = t

    def __call__(self):
        return self.t


def make_round(folder, pose=True):
    """A recorded round: ROUND START at 1.0 s, straight 0.30 m (1.0-2.0 s), 0.5 s pose gap, 90 deg left turn (2.5-3.5 s).

    The controller reports L/R matching that (forward, stop, turn left) and the PC log has the commands sent."""
    folder.mkdir(parents=True)
    clock = Clock(100.0)
    manifest = {"label": "bench", "calibration": {"points_px": [[0, 0]]} if pose else None,
                "arena": {"measured": False, "width_m": 1.15, "height_m": 1.15, "pit_radius_m": 0.25,
                          "pit_centres_m": [[0.0, 1.15], [1.15, 0.0]], "flipper_vertices_m": [[0.815, 1.15], [1.15, 1.15], [1.15, 0.815]]}}
    recorder = RoundRecorder(folder, manifest, clock=clock)
    recorder.start()
    t0 = recorder.t0_mono
    recorder.mark_round_start(t0 + 1.0)

    def log_pose(t, valid, x=0.0, y=0.0, heading=0.0):
        recorder.log_pose(t0 + t, int(round(t * 20)), {
            "valid": 1 if valid else 0, "x_m": f"{x:.4f}" if valid else "", "y_m": f"{y:.4f}" if valid else "",
            "heading_deg": f"{heading:.2f}" if valid else "", "orientation": "upright", "position_quality": "height-corrected",
            "used_ids": "0" if valid else "", "visible_ids": "0" if valid else "", "tag_switch": 0,
            "reason": "" if valid else ("no markers" if pose else "no calibration")})

    for i in range(71):                                 # 0.00 .. 3.50 s at 20 Hz
        t = round(i * 0.05, 2)
        if not pose or t < 1.0 or 2.0 < t < 2.5:
            log_pose(t, False)
        elif t <= 2.0:
            log_pose(t, True, 0.30 + 0.30 * (t - 1.0), 0.50, 0.0)
        else:
            log_pose(t, True, 0.60, 0.50, 90.0 * (t - 2.5))
    for i in range(36):                                 # controller status at 10 Hz
        t = round(i * 0.1, 1)
        left, right = (80, 80) if 1.0 <= t < 2.0 else (-60, 60) if 2.5 <= t < 3.5 else (0, 0)
        recorder.log_control(t0 + t, "controller_status", {
            "device_ms": int(t * 1000), "link": True, "armed": True, "requested": "PC", "active": "PC", "pc_ok": True,
            "block": "NONE", "fault": "NONE", "stop": "NONE", "left": left, "right": right, "pc_age_ms": 10, "token": 7})
    for i in range(176):                                # PC-sent commands at 50 Hz
        t = round(i * 0.02, 2)
        throttle, steer = (80, 0) if 1.0 <= t < 2.0 else (0, -60) if 2.5 <= t < 3.5 else (0, 0)
        recorder.log_control(t0 + t, "pc_sent", {"token": 7, "throttle": throttle, "steer": steer})
    clock.t = 104.0
    recorder.stop()
    return RoundData(folder)


# ----------------------------------------------------------------------------------------------
# Checkpoint labelling: from metadata, never from the file name


def test_supervised_clone_is_labelled_from_its_run_record():
    needs(CLONE)
    info = shadow.inspect_checkpoint(CLONE)
    assert info.kind == shadow.KIND_SUPERVISED
    assert "SUPERVISED" in info.evidence and "scripts/reactive_recovery.py" in info.evidence and "num_timesteps = 0" in info.evidence
    assert info.num_timesteps == 0 and info.observation_size == 19 and info.action_count == 5
    assert info.action_profile == "experimental5_v1" and info.observation_profile == "sim_debug"
    assert info.previous_action_encoding == "masked_zero_v1"
    assert info.tactics == ("stop", "advance", "reverse", "turn_left", "turn_right")
    needs_by_index = {item.index: item.needs for item in info.requirements}
    assert needs_by_index == {"0-2": "imu", "3-5": "imu", "6-12": "none", "13-14": "camera_pose", "15-16": "camera_pose",
                              "17-18": "opponent_tracker"}
    assert "supervised clone" in info.label


def test_ppo_run_is_labelled_from_the_timesteps_in_its_zip():
    needs(PPO)
    info = shadow.inspect_checkpoint(PPO)
    assert info.kind == shadow.KIND_PPO
    assert info.num_timesteps == 5253633 and "5,253,633" in info.evidence
    assert info.observation_size == 17 and info.observation_profile == "sim_debug"
    assert info.action_profile == "legacy5" and "inferred" in info.action_profile_source      # the run record does not name it
    assert info.previous_action_encoding == "onehot_v1" and "legacy default" in info.previous_action_encoding_source
    assert [item.needs for item in info.requirements] == ["imu", "imu", "policy", "camera_pose", "camera_pose", "opponent_tracker"]


def test_file_and_folder_names_play_no_part_in_the_label(tmp_path):
    # A zip with no record at all, in a folder and with a name that both shout "PPO".
    nameless = shadow.inspect_checkpoint(write_zip(tmp_path / "ppo-final" / "ppo_model.zip"))
    assert nameless.kind == shadow.KIND_UNKNOWN == "unknown (no training record; 0 PPO timesteps)"
    assert "no effective_config.json" in nameless.evidence
    assert nameless.observation_profile == "sim_debug" and nameless.action_profile == "experimental5_v1"   # inferred from 19 / 5
    # PPO timesteps in the zip, in a folder named like a clone.
    trained = shadow.inspect_checkpoint(write_zip(tmp_path / "supervised-clone" / "clone.zip", num_timesteps=4096))
    assert trained.kind == shadow.KIND_PPO
    # A run record that says supervised, next to a zip named like a PPO model.
    folder = tmp_path / "ppo-run"
    write_zip(folder / "ppo.zip", metadata={"observation_profile": "sim_debug", "action_profile": "experimental5_v1",
                                            "previous_action_encoding": "masked_zero_v1"})
    (folder / "effective_config.json").write_text(json.dumps({
        "training": "SUPERVISED imitation of a test controller. Not trained by PPO.", "trainer": "scripts/some_trainer.py",
        "previous_action_encoding": "masked_zero_v1"}), encoding="utf-8")
    cloned = shadow.inspect_checkpoint(folder / "ppo.zip")
    assert cloned.kind == shadow.KIND_SUPERVISED and "scripts/some_trainer.py" in cloned.evidence
    # A record that exists but says nothing about training, with 0 timesteps: still unknown.
    folder = tmp_path / "silent-run"
    write_zip(folder / "model.zip")
    (folder / "effective_config.json").write_text(json.dumps({"profile": "sim_debug"}), encoding="utf-8")
    assert shadow.inspect_checkpoint(folder / "model.zip").kind == shadow.KIND_UNKNOWN


def test_unreadable_checkpoint_and_unknown_layout(tmp_path):
    bad = tmp_path / "not_a_zip.zip"
    bad.write_text("hello", encoding="utf-8")
    with pytest.raises(shadow.CheckpointError):
        shadow.inspect_checkpoint(bad)
    with pytest.raises(shadow.CheckpointError):
        shadow.inspect_checkpoint(tmp_path / "missing.zip")
    odd = shadow.inspect_checkpoint(write_zip(tmp_path / "odd" / "model.zip", observation=23))
    assert not odd.layout_known and odd.requirements[0].needs == "unknown"
    with pytest.raises(shadow.CheckpointError):
        shadow.SimShadowSession(odd).load()


# ----------------------------------------------------------------------------------------------
# LIVE SHADOW is unavailable, and says why


def test_live_shadow_is_unavailable_with_reasons_and_the_hub_refuses_it(hub, tmp_path):
    info = shadow.inspect_checkpoint(write_zip(tmp_path / "run" / "model.zip"))
    available, missing = shadow.live_shadow_status(info, hub)
    assert available is False
    assert all(reason in missing for reason in hub.live_shadow_missing())
    assert any("live observation adapter" in reason for reason in missing)
    assert any("opponent" in reason for reason in missing)
    assert any("SIMULATOR TRUTH" in reason and "13-14, 15-16" in reason and "17-18" in reason for reason in missing)
    available, missing = shadow.live_shadow_status(None, hub)
    assert available is False and "no checkpoint is selected" in missing
    ok, message = hub.set_policy("LIVE SHADOW", "live observations", info.label)
    assert ok is False and message.startswith("LIVE SHADOW unavailable")
    assert hub.policy["mode"] == "OFF"


# ----------------------------------------------------------------------------------------------
# No motor-write path


def _without_docstrings(tree):
    strings = []
    docstrings = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and node.body:
            first = node.body[0]
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str):
                docstrings.add(id(first.value))
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in docstrings:
            strings.append(node.value)
    return strings


@pytest.mark.parametrize("source", SOURCES, ids=lambda path: path.name)
def test_diagnostics_code_has_no_serial_import_and_no_write_path(source):
    tree = ast.parse(source.read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add(node.module or "")
            imported.update(f"{node.module}.{alias.name}" for alias in node.names)
    forbidden = ("serial", "socket", "control_center.backend.controller_link", "control_center.backend.robot_link",
                 "control_center.backend.command_manager", "subprocess")
    assert not [name for name in imported if any(name == bad or name.startswith(bad + ".") for bad in forbidden)], imported
    assert not [name for name in imported if name.endswith(("controller_link", "robot_link", "command_manager"))], imported

    writers = {"write", "writelines", "write_text", "write_bytes", "writestr", "send", "sendall", "sendto", "open"}
    hub_calls = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            assert not (isinstance(node.func, ast.Name) and node.func.id == "open"), f"{source.name}:{node.lineno} opens a file"
            if isinstance(node.func, ast.Attribute):
                assert node.func.attr not in writers, f"{source.name}:{node.lineno} calls .{node.func.attr}()"
                owner = node.func.value
                if (isinstance(owner, ast.Name) and owner.id == "hub") or (isinstance(owner, ast.Attribute) and owner.attr == "hub"):
                    hub_calls.add(node.func.attr)
        if isinstance(node, (ast.Name, ast.Attribute)):
            name = node.id if isinstance(node, ast.Name) else node.attr
            assert name not in ("SerialLineSource", "Serial", "connect_controller", "connect_controller_port"), f"{source.name}: {name}"
    # The only things ever called on the hub: read what is missing, and store what to display.
    assert hub_calls <= {"set_policy", "live_shadow_missing"}, hub_calls
    # No drive command line ("D,<token>,<throttle>,<steer>") is formatted anywhere outside the explanatory docstrings.
    assert not [text for text in _without_docstrings(tree) if re.search(r"(^|\n)D,|D,\{|D,%", text)]


# ----------------------------------------------------------------------------------------------
# Simulator log


def test_sim_log_is_split_at_resets_and_gives_a_command_timeline(tmp_path):
    log = sim_compare.load_sim_log(write_sim_log(tmp_path / "play.jsonl"))
    assert [len(r.steps) for r in log.rounds] == [125, 10] and [r.split_by for r in log.rounds] == ["reset line", "reset line"]
    assert log.step_fields == sim_compare.STEP_FIELDS_WRITTEN_BY_CLI
    assert not log.has_poses and "no pose fields" in log.pose_note
    assert not log.has_contact and "cannot be detected" in log.contact_note
    segments = sim_compare.command_timeline(log.rounds[0])
    assert [(round(s.start_s, 2), round(s.end_s, 2), s.label, s.detail, s.left, s.right) for s in segments] == [
        (0.0, 1.0, "forward", "advance", -1.0, -1.0), (1.0, 1.5, "stop", "stop", 0.0, 0.0), (1.5, 2.5, "turn_left", "turn_left", 0.7, -0.7)]
    assert log.rounds[0].duration_s == pytest.approx(2.5)
    simulated = sim_compare.simulated_motion(log, log.rounds[0], 0.0, 2.5)
    assert not simulated.available and "type the simulated values" in simulated.reason


def test_sim_log_without_reset_lines_splits_where_time_restarts_and_skips_junk(tmp_path):
    path = tmp_path / "old.jsonl"
    steps = [{"type": "step", "time_s": 0.02 * (i % 5 + 1), "blue_tactic": "advance", "blue_command": [-1, -1]} for i in range(10)]
    path.write_text("\n".join([json.dumps(s) for s in steps] + ["{broken", json.dumps({"type": "other"})]) + "\n", encoding="utf-8")
    log = sim_compare.load_sim_log(path)
    assert [len(r.steps) for r in log.rounds] == [5, 5] and log.rounds[1].split_by == "simulated time restarted"
    assert log.skipped_lines == 2


def test_a_real_play_log_has_commands_but_no_poses_and_no_contact_flag():
    needs(REAL_LOG)
    log = sim_compare.load_sim_log(REAL_LOG)
    assert log.step_fields == ("type", "time_s", "blue_tactic", "blue_command", "red_command", "end_reason")
    assert len(log.rounds) == 2 and not log.has_poses and not log.has_contact
    for sim_round in log.rounds:
        segments = sim_compare.command_timeline(sim_round)
        assert segments[0].start_s == pytest.approx(0.0, abs=1e-9) and segments[-1].end_s == pytest.approx(sim_round.end_s)
        assert all(a.end_s == pytest.approx(b.start_s) for a, b in zip(segments, segments[1:]))
        assert sum(s.duration_s for s in segments) == pytest.approx(sim_round.duration_s)
        assert {s.label for s in segments} <= set(sim_compare.DRIVE_CLASSES)
        assert all(s.label == sim_compare.TACTIC_CLASS[s.detail] for s in segments)       # wheel command and tactic name agree


def test_drive_classes_follow_each_side_s_sign_convention():
    sim = lambda l, r: sim_compare.drive_class(l, r, sim_compare.SIM_FORWARD_SIGN, deadband=0.02)
    real = lambda l, r: sim_compare.drive_class(l, r, sim_compare.CONTROLLER_FORWARD_SIGN, sim_compare.CONTROLLER_SCALE)
    assert [sim(*c) for c in ((0, 0), (-1, -1), (0.65, 0.65), (0.7, -0.7), (-0.7, 0.7), (-0.55, -1.0), (-1.0, -0.55))] == [
        "stop", "forward", "reverse", "turn_left", "turn_right", "forward_left", "forward_right"]
    assert [real(*c) for c in ((0, 0), (80, 80), (-80, -80), (-60, 60), (60, -60), (40, 100), (3, -3))] == [
        "stop", "forward", "reverse", "turn_left", "turn_right", "forward_left", "stop"]
    assert sim_compare.pc_mix(80, 0) == (80, 80) and sim_compare.pc_mix(0, -60) == (-60, 60)       # steer + = right


# ----------------------------------------------------------------------------------------------
# Measured motion


def test_measured_distance_and_heading_come_from_valid_runs_only(tmp_path):
    data = make_round(tmp_path / "round")
    assert sim_compare.round_start_times(data) == [1.0]
    motion = sim_compare.measured_motion(data, 1.0, 3.5)
    assert motion.available and motion.track == "live" and "LIVE camera track" in motion.track_note
    assert motion.distance_m == pytest.approx(0.30, abs=1e-3) and motion.straight_line_m == pytest.approx(0.30, abs=1e-3)
    assert motion.heading_change_deg == pytest.approx(90.0, abs=0.05)
    assert motion.runs == 2 and motion.known_s == pytest.approx(2.0) and motion.unknown_s == pytest.approx(0.5)
    assert len(motion.path) == 2 and "UNKNOWN for 0.50 s" in motion.text()
    # The straight alone, the turn alone, and a window that starts before any pose exists.
    straight = sim_compare.measured_motion(data, 1.0, 2.0)
    assert straight.distance_m == pytest.approx(0.30, abs=1e-3) and straight.heading_change_deg == pytest.approx(0.0, abs=0.05)
    assert straight.unknown_s == pytest.approx(0.0, abs=1e-6)
    turn = sim_compare.measured_motion(data, 2.5, 3.5)
    assert turn.distance_m == pytest.approx(0.0, abs=1e-3) and turn.heading_change_deg == pytest.approx(90.0, abs=0.05)
    early = sim_compare.measured_motion(data, 0.0, 3.5)
    assert early.unknown_s == pytest.approx(1.5) and early.distance_m == pytest.approx(0.30, abs=1e-3)
    # Entirely inside the gap: there is a track, but nothing valid in the window.
    gap = sim_compare.measured_motion(data, 2.1, 2.4)
    assert not gap.available and "no valid pose inside this window" in gap.reason
    assert not sim_compare.measured_motion(data, 1.0, 3.5, track="offline").available


def test_jitter_does_not_add_up_to_distance_and_a_marker_switch_is_not_movement():
    from control_center.backend.round_reader import PoseTrack

    class Stub:
        pose_offline = PoseTrack.from_rows([], "offline")

    rng = np.random.default_rng(3)
    rows = [{"t_s": f"{i / 30:.4f}", "valid": "1", "x_m": f"{0.5 + rng.normal(0, 0.0008):.4f}", "y_m": f"{0.5 + rng.normal(0, 0.0008):.4f}",
             "heading_deg": "10.00", "tag_switch": "0"} for i in range(150)]
    Stub.pose_live = PoseTrack.from_rows(rows, "live")
    still = sim_compare.measured_motion(Stub, 0.0, 5.0)
    assert still.distance_m < 0.01                       # 5 s standing still: raw step sums would give about 0.2 m
    rows = [{"t_s": f"{i * 0.05:.2f}", "valid": "1", "x_m": f"{0.2 + 0.01 * i + (0.03 if i >= 10 else 0):.4f}", "y_m": "0.5000",
             "heading_deg": "0.00", "tag_switch": "1" if i == 10 else "0"} for i in range(21)]
    Stub.pose_live = PoseTrack.from_rows(rows, "live")
    switched = sim_compare.measured_motion(Stub, 0.0, 1.0)
    assert switched.tag_switches == 1 and switched.runs == 2
    assert switched.distance_m == pytest.approx(0.19, abs=1e-3)      # 9 + 10 steps of 1 cm; the 4 cm switch step is left out
    assert switched.unknown_s == pytest.approx(0.05)


def test_a_round_without_calibrated_pose_says_so(tmp_path):
    data = make_round(tmp_path / "round", pose=False)
    assert len(data.pose_live) > 0 and not data.pose_live.valid.any()
    motion = sim_compare.measured_motion(data, 1.0, 3.5)
    assert not motion.available and motion.reason == sim_compare.NO_POSE == "not available: no calibrated pose in this round"
    assert sim_compare.pick_pose_track(data) is None
    log = sim_compare.load_sim_log(write_sim_log(tmp_path / "play.jsonl"))
    result = sim_compare.compare(log, log.rounds[0], data, 1.0, 1.0, 3.5, sim_compare.typed_simulated_motion(0.30, 90.0))
    assert result.motion_text() == sim_compare.NO_POSE
    assert result.command_vs_controller.share == pytest.approx(1.0)      # commands can still be compared without any pose


# ----------------------------------------------------------------------------------------------
# Command agreement and motion agreement are separate


def test_commands_can_agree_while_motion_differs_and_the_other_way_round(tmp_path):
    data = make_round(tmp_path / "round")
    same = sim_compare.load_sim_log(write_sim_log(tmp_path / "same.jsonl"))
    other = sim_compare.load_sim_log(write_sim_log(tmp_path / "other.jsonl", turn="turn_right"))
    offset = sim_compare.round_start_times(data)[0]

    # Same commands, simulated robot went half as far again and turned less: commands agree, motion does not.
    result = sim_compare.compare(same, same.rounds[0], data, offset, 1.0, 3.5, sim_compare.typed_simulated_motion(0.45, 70.0))
    assert result.command_vs_controller.available and result.command_vs_controller.share == pytest.approx(1.0)
    assert result.command_vs_controller.compared_s == pytest.approx(2.5, abs=0.02) and not result.command_vs_controller.disagreements
    assert result.command_vs_pc_sent.share == pytest.approx(1.0)
    assert result.motion.distance_difference_m == pytest.approx(-0.15, abs=1e-3)
    assert result.motion.heading_difference_deg == pytest.approx(20.0, abs=0.05)
    assert "controller-reported" in result.command_text() and "PC-sent" in result.command_text()
    assert "distance" not in result.command_text() and "direction class" not in result.motion_text()
    assert "typed by the user" in result.motion_text() and "LIVE camera track" in result.motion_text()
    assert "UNKNOWN for 0.50 s" in result.motion_text()

    # Different commands (the simulator turned right), simulated motion typed equal to the measurement:
    # motion "agrees" in these two numbers while the commands did not.
    result = sim_compare.compare(other, other.rounds[0], data, offset, 1.0, 3.5, sim_compare.typed_simulated_motion(0.30, 90.0))
    assert result.command_vs_controller.share == pytest.approx(0.6, abs=0.01)
    assert [(round(a, 2), round(b, 2), s, r) for a, b, s, r in result.command_vs_controller.disagreements] == [(2.5, 3.5, "turn_right", "turn_left")]
    assert result.motion.distance_difference_m == pytest.approx(0.0, abs=1e-3)
    assert result.motion.heading_difference_deg == pytest.approx(0.0, abs=0.05)

    # A wrong offset destroys command agreement and leaves the measured motion untouched.
    shifted = sim_compare.compare(same, same.rounds[0], data, offset + 0.5, 1.0, 3.5, sim_compare.typed_simulated_motion(0.45, 70.0))
    assert shifted.command_vs_controller.share < 0.7 and shifted.command_vs_controller.outside_sim_s == pytest.approx(0.5, abs=0.02)
    assert shifted.motion.measured.distance_m == pytest.approx(0.30, abs=1e-3)

    # No typed values and no poses in the log: motion is measured only, and says the simulated side is missing.
    bare = sim_compare.compare(same, same.rounds[0], data, offset, 1.0, 3.5)
    assert bare.motion.distance_difference_m is None and "measured only" in bare.motion_text() and "no pose fields" in bare.motion_text()
    assert not bare.contact.detectable and bare.not_comparable_after_s is None and "cannot be detected" in bare.warning_text()


def test_stale_controller_telemetry_is_unknown_not_held(tmp_path):
    data = make_round(tmp_path / "round")
    data.controller_status = [r for r in data.controller_status if not 1.25 < r["t_s"] < 2.75]     # telemetry lost for 1.5 s
    log = sim_compare.load_sim_log(write_sim_log(tmp_path / "play.jsonl"))
    timeline = sim_compare.controller_timeline(data)
    assert any(a.end_s < b.start_s - 0.5 for a, b in zip(timeline, timeline[1:]))                   # a real hole in the timeline
    agreement = sim_compare.compare(log, log.rounds[0], data, 1.0, 1.0, 3.5).command_vs_controller
    assert agreement.real_unknown_s == pytest.approx(1.0, abs=0.03)      # last report at 1.2 s counts until 1.8 s; next at 2.8 s
    assert agreement.compared_s == pytest.approx(1.5, abs=0.03)


def test_everything_after_the_first_simulated_contact_is_not_comparable(tmp_path):
    data = make_round(tmp_path / "round")
    log = sim_compare.load_sim_log(write_sim_log(tmp_path / "contact.jsonl", contact_from=40))    # contact at simulated t = 0.8 s
    assert log.has_contact
    result = sim_compare.compare(log, log.rounds[0], data, 1.0, 1.0, 3.5, sim_compare.typed_simulated_motion(0.30, 90.0))
    assert result.contact.first_contact_sim_s == pytest.approx(0.8) and result.not_comparable_after_s == pytest.approx(1.8)
    assert result.command_vs_controller.compared_s == pytest.approx(0.8, abs=0.02)
    assert result.command_vs_controller.not_comparable_s == pytest.approx(1.7, abs=0.02)
    assert not result.motion.comparable and result.motion.distance_difference_m is None
    assert sim_compare.NOT_COMPARABLE in result.motion_text() and sim_compare.NOT_COMPARABLE in result.warning_text()
    assert sim_compare.NOT_COMPARABLE == "not comparable: the simulated collision did not happen in the real run"
    # A window that ends before the contact is still comparable.
    before = sim_compare.compare(log, log.rounds[0], data, 1.0, 1.0, 1.7, sim_compare.typed_simulated_motion(0.20, 0.0))
    assert before.motion.comparable and before.motion.distance_difference_m is not None


def test_simulated_motion_is_read_from_the_log_only_when_it_carries_poses(tmp_path):
    data = make_round(tmp_path / "round")
    log = sim_compare.load_sim_log(write_sim_log(tmp_path / "poses.jsonl", poses=True))
    assert log.has_poses
    result = sim_compare.compare(log, log.rounds[0], data, 1.0, 1.0, 3.5)
    assert result.motion.simulated.source == "simulator log poses"
    assert result.motion.simulated.distance_m == pytest.approx(0.392, abs=1e-6)                   # 49 steps of 8 mm inside the window
    assert result.motion.simulated.heading_change_deg == pytest.approx(50.0, abs=1e-6)
    assert result.motion.distance_difference_m == pytest.approx(0.30 - 0.392, abs=1e-3)


# ----------------------------------------------------------------------------------------------
# SIM reporting to the hub (no torch: the decisions are fabricated)


def decision(index, tactic="advance", valid=True):
    return shadow.ShadowDecision(index, 1, 0.02 * index, valid, "" if valid else "observation contains non-finite values",
                                 1 if valid else None, tactic if valid else None, (-1.0, -1.0) if valid else None,
                                 {"stop": 0.1, "advance": 0.9} if valid else None, None)


def test_runner_reports_sim_to_the_hub_at_a_limited_rate_and_off_when_stopped(hub, tmp_path):
    info = shadow.inspect_checkpoint(write_zip(tmp_path / "run" / "model.zip"))
    runner = shadow.SimShadowRunner(hub, info)          # never started: only the UI-thread side is exercised
    calls = []
    original = hub.set_policy
    hub.set_policy = lambda *args, **kwargs: (calls.append((args, kwargs)), original(*args, **kwargs))[1]
    for i in range(1, 31):
        runner._on_decision(decision(i))
    assert len(calls) == 1 and runner.decisions_seen == 30                                          # 30 decisions at once, one report
    assert hub.policy["mode"] == "SIM" and hub.policy["input"] == "simulator observations" and hub.policy["checkpoint"] == info.label
    proposed = hub.policy["proposed"]
    assert proposed["tactic"] == "advance" and proposed["wheel_command"] == [-1.0, -1.0] and proposed["observation_valid"] is True
    assert "SIMULATED robot only" in proposed["wheel_command_units"]
    json.dumps(proposed)                                # the recorder writes this as JSON
    time.sleep(shadow.HUB_REPORT_PERIOD_S + 0.05)
    runner._on_decision(decision(31, valid=False))
    assert len(calls) == 2 and hub.policy["proposed"]["tactic"] is None and hub.policy["proposed"]["observation_valid"] is False
    runner.stop()
    assert hub.policy["mode"] == "OFF" and hub.policy["proposed"] is None
    runner._on_decision(decision(32))                   # a late decision after stop is ignored
    assert hub.policy["mode"] == "OFF" and len(calls) == 3


# ----------------------------------------------------------------------------------------------
# The screen


def test_screen_constructs_and_shows_both_workflows(app, hub, tmp_path):
    from control_center.screens.diagnostics import DiagnosticsScreen

    screen = DiagnosticsScreen(hub)
    try:
        screen.resize(1400, 900)
        assert [screen.tabs.tabText(i) for i in range(screen.tabs.count())] == ["Sim-to-real", "Shadow"]
        tab, shadow_tab = screen.sim_to_real, screen.shadow
        assert not screen.grab().isNull()

        # Shadow: LIVE SHADOW is unavailable with the reasons, its button is disabled, SIM needs a checkpoint.
        assert not shadow_tab.live_button.isEnabled() and shadow_tab.live_state.text() == "UNAVAILABLE"
        assert "live observation adapter" in shadow_tab.live_missing.text() and "no checkpoint is selected" in shadow_tab.live_missing.text()
        assert not shadow_tab.sim_button.isEnabled()
        assert screen.load_checkpoint(write_zip(tmp_path / "ppo-named" / "ppo.zip"))
        assert shadow_tab.kind_label.text() == shadow.KIND_UNKNOWN and "not the file name" in shadow_tab.evidence_label.text()
        assert shadow_tab.requirements.rowCount() == 6 and shadow_tab.requirements.item(5, 3).text().startswith("an opponent tracker")
        assert shadow_tab.sim_button.isEnabled() and not shadow_tab.live_button.isEnabled()
        assert "SIMULATOR TRUTH" in shadow_tab.live_missing.text()
        assert not screen.load_checkpoint(tmp_path / "missing.zip") and not shadow_tab.sim_button.isEnabled()
        screen.tabs.setCurrentIndex(1)
        assert not screen.grab().isNull()
        screen.tabs.setCurrentIndex(0)

        # Sim-to-real: a round with no calibrated pose gets the explicit note and no map.
        assert screen.load_round(make_round(tmp_path / "no-pose", pose=False).folder)
        assert tab.map_stack.currentWidget() is tab.no_pose_note and "No calibrated pose in this round" in tab.no_pose_note.text()
        assert tab.arena_map is None and tab.table.item(0, 0).text() == "not available"
        assert "no calibrated pose in this round" in tab.motion_line.text()
        assert not screen.load_round(tmp_path / "not-a-round")

        # A round with pose: the measured path is drawn, the offset comes from the ROUND START marker.
        assert screen.load_round(make_round(tmp_path / "with-pose").folder)
        assert tab.map_stack.currentWidget() is tab.map_holder and tab.arena_map is not None and len(tab.arena_map.trails) == 2
        assert tab.offset_box.value() == pytest.approx(1.0) and tab.align_button.isEnabled()
        assert "choose a simulator log" in tab.command_line.text()
        assert screen.load_sim_log(write_sim_log(tmp_path / "play.jsonl"))
        assert tab.sim_round_box.count() == 2
        tab.set_window(1.0, 3.5)
        assert tab.comparison is not None
        assert tab.command_line.text().startswith("COMMAND agreement") and "(100%)" in tab.command_line.text()
        assert tab.motion_line.text().startswith("MOTION agreement") and "measured only" in tab.motion_line.text()
        assert tab.command_line is not tab.motion_line and "separate checks" in tab.separate_note.text()
        assert tab.table.item(0, 0).text() == "0.300" and tab.table.item(1, 0).text() == "+90.0" and tab.table.item(0, 2).text() == "—"
        assert "cannot be detected" in tab.warning_line.text() and "no pose fields" in tab.simulated_note.text()
        assert [row.title for row in tab.timeline.rows] == ["SIM blue command", "Controller L/R (reported)", "PC sent (not confirmed)",
                                                            "Camera pose valid"]
        assert [(round(b.start, 2), round(b.end, 2)) for b in tab.timeline.rows[3].bars] == [(1.0, 2.0), (2.5, 3.5)]
        assert tab.timeline.rows[0].bars[0].start == pytest.approx(1.0)                           # shifted by the offset

        # Typed simulated values fill the difference column; the command line does not change.
        before = tab.command_line.text()
        tab.sim_distance_edit.setText("0.45")
        tab.sim_heading_edit.setText("70")
        tab._refresh()
        assert tab.table.item(0, 2).text() == "-0.150" and tab.table.item(1, 2).text() == "+20.0"
        assert tab.command_line.text() == before and "typed by the user" in tab.motion_line.text()

        # A simulated contact: the warning, and the difference column says not comparable.
        assert screen.load_sim_log(write_sim_log(tmp_path / "contact.jsonl", contact_from=40))
        tab.set_window(1.0, 3.5)
        assert sim_compare.NOT_COMPARABLE in tab.warning_line.text() and tab.table.item(0, 2).text() == "not comparable"
        assert tab.timeline.blocked_after == pytest.approx(1.8)
        assert not screen.grab().isNull()
    finally:
        screen.shutdown()
        screen.deleteLater()


# ----------------------------------------------------------------------------------------------
# SLOW: really loads a checkpoint (torch + stable-baselines3 import) and steps the simulator a few decisions.


def run_runner(app, runner, timeout_s=150.0):
    seen, errors, messages = [], [], []
    runner.decision.connect(seen.append)
    runner.failed.connect(errors.append)
    runner.progress.connect(messages.append)
    runner.start()
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline and (runner.isRunning() or not runner.isFinished()):
        app.processEvents()
        time.sleep(0.01)
    assert runner.wait(5000), "the SIM runner did not finish"
    for _ in range(5):
        app.processEvents()
    return seen, errors, messages


def test_slow_checkpoint_runs_on_simulator_observations_and_only_reports_to_the_hub(app, hub):
    if os.environ.get("AUTOBOT_SKIP_SLOW"):
        pytest.skip("AUTOBOT_SKIP_SLOW is set")
    needs(CLONE)
    needs(shadow.FROZEN_CONFIG)
    needs(shadow.FROZEN_SCENARIOS)
    from autobot.sim.env import tactic_to_drive

    info = shadow.inspect_checkpoint(CLONE)
    modes = []
    hub.event_added.connect(lambda stamp, kind, text: modes.append(text) if kind == "policy" else None)
    runner = shadow.SimShadowRunner(hub, info, realtime=False, max_decisions=8)
    seen, errors, messages = run_runner(app, runner)
    assert not errors, errors
    assert len(messages) == 3 and "torch" in messages[0]
    assert [d.index for d in seen] == list(range(1, 9)) and all(d.round_index == 1 for d in seen)
    assert [d.sim_time_s for d in seen] == pytest.approx([0.02 * i for i in range(8)], abs=1e-9)
    for d in seen:
        assert d.observation_valid and d.tactic in info.tactics
        assert d.wheel_command == tuple(float(v) for v in tactic_to_drive(d.action, info.action_profile))
        assert set(d.probabilities) == set(info.tactics) and sum(d.probabilities.values()) == pytest.approx(1.0, abs=1e-4)
        assert max(d.probabilities, key=d.probabilities.get) == d.tactic                          # the greedy choice
    assert any(text.startswith("Policy SIM") and info.label in text and "simulator observations" in text for text in modes)
    assert modes[-1] == "Policy OFF" and hub.policy["mode"] == "OFF"

    # An invalid observation proposes nothing.
    session = shadow.SimShadowSession(info)
    session.load()
    first = session.step()
    assert first.observation_valid and first.sim_time_s == pytest.approx(0.0, abs=1e-9)
    session.session.env._obs = lambda: np.full(info.observation_size, np.nan, dtype=np.float32)
    broken = session.step()
    assert not broken.observation_valid and broken.tactic is None and broken.wheel_command is None
    assert "non-finite" in broken.observation_problem and broken.proposed()["tactic"] is None

    # The PPO run's layout is not recorded in full; the inferred one must actually load.
    if PPO.is_file():
        ppo = shadow.SimShadowSession(shadow.inspect_checkpoint(PPO))
        ppo.load()
        assert ppo.step().observation_valid
