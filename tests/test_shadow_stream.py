"""Observation contract, camera-stream adapter and the shadow-stream demo. Read-only vision only.

Everything runs on SYNTHETIC pictures, plain numbers or the simulator, except the footage test, which
needs the Bristol recording and is skipped unless AUTOBOTS_FOOTAGE_MP4 points at it. No real camera,
serial port or robot link is opened: camera mode is driven through a stand-in capture device and window.
None of the tolerances here are real-camera accuracy.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from control_center.tools import shadow_stream_demo as demo
from control_center.tools.blob_tracker import OVERLAP, TRACKED, Track
from control_center.tools.observation_contract import (GROUP_IMU, GROUP_OPPONENT, GROUP_OWN, GROUP_PREVIOUS,
                                                       RECOMMENDED_CHECKPOINT, SOURCE_ARUCO, SOURCE_CONSTANT, SOURCE_FLOOR,
                                                       SOURCE_TELEMETRY, ContractError, format_contract, load_contract)
from control_center.tools.overlay_core import BotPose, load_config, save_clicks
from control_center.tools.overlay_fixture import BotInstance, Scene, fixture_clicks, make_camera, render_frame
from control_center.tools.stream_adapter import (MODE_BLOCKED, CameraSample, CheckpointPolicy, ImuSample, NoTelemetry,
                                                 OpponentEstimate, OpponentResolver, RecordedTelemetry, ShadowAssembler)

REPO = Path(__file__).resolve().parents[1]
needs_checkpoint = pytest.mark.skipif(not RECOMMENDED_CHECKPOINT.is_file(), reason="the recommended checkpoint is not in runs/")
pytestmark = needs_checkpoint


@pytest.fixture(scope="module")
def contract():
    return load_contract()


# ---- the contract ---------------------------------------------------------------------------------

def test_contract_lists_every_field_of_the_recommended_checkpoint(contract):
    assert contract.observation_size == 19 and len(contract.fields) == 19 and [f.index for f in contract.fields] == list(range(19))
    assert (contract.observation_profile, contract.action_profile, contract.previous_action_encoding) == ("sim_debug", "experimental5_v1", "masked_zero_v1")
    assert contract.tactics == ("stop", "advance", "reverse", "turn_left", "turn_right")
    assert [contract.span(g) for g in (GROUP_IMU, GROUP_PREVIOUS, GROUP_OWN, GROUP_OPPONENT)] == ["0-5", "6-12", "13-16", "17-18"]
    source = {group: {f.live_source for f in contract.fields if f.group == group} for group in (GROUP_IMU, GROUP_PREVIOUS, GROUP_OWN, GROUP_OPPONENT)}
    assert source == {GROUP_IMU: {SOURCE_TELEMETRY}, GROUP_PREVIOUS: {SOURCE_CONSTANT}, GROUP_OWN: {SOURCE_ARUCO}, GROUP_OPPONENT: {SOURCE_FLOOR}}
    assert contract.required_groups() == [GROUP_IMU, GROUP_OWN, GROUP_OPPONENT]          # the masked previous tactic needs no input
    assert all("NOT AVAILABLE" in f.live_status for f in contract.fields if f.group in (GROUP_IMU, GROUP_OPPONENT))
    assert "supervised" in contract.training.lower() and "19 values in, 5 tactics out" in format_contract(contract)
    assert (contract.arena_width_m, contract.arena_height_m, contract.control_hz) == (1.15, 1.15, 50.0)


def test_contract_reproduces_the_simulator_observation_exactly(contract):
    """The field list is a description of AutoBotEnv._obs; this is what keeps the two from drifting apart."""
    from autobot.sim.env import AutoBotEnv

    env = AutoBotEnv(config_path=str(contract.sim_config_path), observation_profile=contract.observation_profile, seed=3,
                     opponent_mode="static", action_profile=contract.action_profile, previous_action_encoding=contract.previous_action_encoding)
    env.reset(seed=3)
    for step in range(60):
        observation, *_ = env.step([1, 3, 4][step // 20])
        imu, own, opponent = env.sim.imu(), env.sim.truth(0), env.sim.truth(1)
        mine = contract.assemble(gyro_rad_s=imu.gyro_rad_s, accel_m_s2=imu.accel_m_s2, own_xy_m=own["position_m"][:2],
                                 own_heading_rad=own["yaw_rad"], opponent_xy_m=opponent["position_m"][:2])
        assert np.array_equal(mine, observation)
    # Conventions the camera side relies on: "advance" moves along the heading, and a left turn is positive yaw and gyro z.
    env.reset(seed=3); env.sim.set_pose(0, 0.4, 0.4, 0.0)
    start = env.sim.truth(0)["position_m"][:2].copy()
    for _ in range(20): env.step(1)
    moved = env.sim.truth(0)["position_m"][:2] - start
    assert moved[0] > 0.1 and abs(moved[1]) < 0.02
    env.sim.set_pose(0, 0.4, 0.4, 0.0); rates = []
    for _ in range(15): env.step(3); rates.append(env.sim.imu().gyro_rad_s[2])
    assert env.sim.truth(0)["yaw_rad"] > 0.3 and np.mean(rates) > 1.0


def test_contract_refuses_to_guess_a_missing_or_inconsistent_record(tmp_path, contract):
    def checkpoint(name, data):
        path = tmp_path / name / "model.zip"; path.parent.mkdir()
        with zipfile.ZipFile(path, "w") as archive: archive.writestr("data", json.dumps(data))
        return path

    spaces = {"observation_space": {"_shape": [19]}, "action_space": {"n": "5"}}
    with pytest.raises(ContractError, match="not recorded"):                         # right sizes, no record: no inference from the size
        load_contract(checkpoint("bare", spaces), sim_config=contract.sim_config_path)
    recorded = {"schema": "autobot_observation_v1", "observation_profile": "sim_debug", "action_profile": "experimental5_v1", "previous_action_encoding": "masked_zero_v1"}
    with pytest.raises(ContractError, match="does not name the simulator config"):
        load_contract(checkpoint("noconfig", {**spaces, "autobot_observation": recorded}))
    with pytest.raises(ContractError, match="zip's spaces"):
        load_contract(checkpoint("sizes", {"observation_space": {"_shape": [13]}, "action_space": {"n": "5"}, "autobot_observation": recorded}), sim_config=contract.sim_config_path)
    ok = load_contract(checkpoint("good", {**spaces, "autobot_observation": recorded}), sim_config=contract.sim_config_path)
    assert ok.observation_size == 19 and ok.training == "not recorded"
    with pytest.raises(ContractError, match="does not exist"):
        load_contract(tmp_path / "nothing.zip")
    with pytest.raises(ContractError, match="3 values each"):
        contract.assemble(gyro_rad_s=[0, 0], accel_m_s2=[0, 0, 9.81], own_xy_m=[0.5, 0.5], own_heading_rad=0.0, opponent_xy_m=[0.6, 0.6])


# ---- which blob is the opponent --------------------------------------------------------------------

def pose(x=None, y=None, ok=True):
    if not ok:
        return BotPose(reason="no configured bot markers visible")
    return BotPose(state="ok", reason="ok", x_m=x, y_m=y, heading_rad=0.0, floor_projection_xy=(x, y), used_ids=[0])


def tracker_with(tracks, overlap_at=None):
    return SimpleNamespace(tracks=tracks, overlap_at=overlap_at or {})


def blob(track_id, x, y, state=TRACKED, group=None):
    return Track(track_id, state=state, floor=None if state != TRACKED else (x, y), last_floor=(x, y), group=group)


def test_opponent_is_known_only_when_the_tag_accounts_for_the_own_bot():
    resolver = OpponentResolver()
    two = tracker_with([blob(1, 0.30, 0.30), blob(2, 0.80, 0.70)])
    unknown = resolver.resolve(two, pose(ok=False))                       # two unmarked blobs and no tag: neither is "the opponent"
    assert not unknown.ok and unknown.xy_m is None and "not identified by a tag" in unknown.reason
    one = resolver.resolve(tracker_with([blob(1, 0.30, 0.30)]), pose(ok=False))
    assert not one.ok and "not identified" in one.reason                  # a single blob could just as well be the own bot
    known = resolver.resolve(two, pose(0.31, 0.29))
    assert known.ok and known.track_id == 2 and known.xy_m == (0.80, 0.70) and known.own_track_id == 1 and known.own_identified_by == "tag this frame"
    three = resolver.resolve(tracker_with([blob(1, 0.30, 0.30), blob(2, 0.80, 0.70), blob(3, 0.50, 0.90)]), pose(0.30, 0.30))
    assert not three.ok and three.candidates == 2 and "cannot tell which" in three.reason
    alone = resolver.resolve(tracker_with([blob(1, 0.30, 0.30)]), pose(0.30, 0.30))
    assert not alone.ok and "no separate unmarked blob" in alone.reason
    assert not resolver.resolve(None, pose(0.3, 0.3), "no startup reference picture").ok


def test_own_label_follows_one_blob_and_does_not_survive_an_overlap():
    resolver = OpponentResolver()
    apart = tracker_with([blob(1, 0.30, 0.30), blob(2, 0.80, 0.70)])
    assert resolver.resolve(apart, pose(0.30, 0.30)).ok
    covered = resolver.resolve(apart, pose(ok=False))                     # tag covered, same two blobs still tracked on their own
    assert covered.ok and covered.track_id == 2 and covered.own_identified_by == "same blob as when the tag was last read"
    merged = tracker_with([blob(1, 0.55, 0.50, OVERLAP, 1), blob(2, 0.55, 0.50, OVERLAP, 1)], {1: (0.55, 0.50)})
    touching = resolver.resolve(merged, pose(0.50, 0.47))                 # tag still read, but the blobs are one
    assert not touching.ok and touching.xy_m is None and touching.reason.startswith("overlap")
    blind = resolver.resolve(merged, pose(ok=False))
    assert not blind.ok and resolver.own_track_id is None
    after = resolver.resolve(apart, pose(ok=False))                       # apart again but no tag yet: the old label is gone
    assert not after.ok and "not identified" in after.reason
    assert resolver.resolve(apart, pose(0.80, 0.70)).track_id == 1        # the tag, not history, says which blob is ours


# ---- a decision time -------------------------------------------------------------------------------

def sample(index, t, own, opponent):
    return CameraSample(index, t, own, opponent)


def opponent_at(x, y):
    return OpponentEstimate(state="ok", reason="ok", xy_m=(x, y), track_id=2, own_track_id=1)


def imu_stream(times, yaw_rate=0.5):
    return RecordedTelemetry([ImuSample(t, (0.0, 0.0, yaw_rate), (0.0, 0.0, 9.81)) for t in times], "test IMU")


def test_assembler_blocks_on_each_missing_input_and_names_the_fields(contract):
    blocked = ShadowAssembler(contract, "test", NoTelemetry())
    first = blocked.decide(0.0)
    assert first.mode == MODE_BLOCKED and first.observation is None and first.proposed is None
    assert [entry.split(":")[0] for entry in first.blocked_by] == ["[0-5] imu", "[13-16] own_pose", "[17-18] opponent_position"]
    assert "no camera frame" in first.inputs[GROUP_OWN].reason and first.inputs[GROUP_PREVIOUS].valid
    blocked.offer(sample(0, 1.0, pose(0.40, 0.50), opponent_at(0.70, 0.60)))
    camera_ok = blocked.decide(1.01)
    assert [entry.split(":")[0] for entry in camera_ok.blocked_by] == ["[0-5] imu"] and camera_ok.observation is None
    trace = camera_ok.to_trace()
    assert trace["proposed"] is None and trace["inputs"]["own_pose"]["valid"] and trace["inputs"]["imu"]["source"] == "none"
    assert "nothing is sent" in trace["output"] and any("BLOCKED" in text for text, _ in camera_ok.ui_lines())

    full = ShadowAssembler(contract, "test", imu_stream([0.99, 1.0, 1.5]))
    full.offer(sample(0, 1.0, pose(0.40, 0.50), opponent_at(0.70, 0.60)))
    complete = full.decide(1.02)
    expected = contract.assemble(gyro_rad_s=(0, 0, 0.5), accel_m_s2=(0, 0, 9.81), own_xy_m=(0.40, 0.50), own_heading_rad=0.0, opponent_xy_m=(0.70, 0.60))
    assert complete.blocked_by == [] and np.array_equal(complete.observation, expected) and complete.proposed is None   # no policy attached
    assert complete.inputs[GROUP_IMU].age_ms == pytest.approx(20.0) and complete.inputs[GROUP_OWN].age_ms == pytest.approx(20.0)
    for own, opponent, group in ((pose(ok=False), opponent_at(0.7, 0.6), "own_pose"), (pose(0.4, 0.5), OpponentEstimate(reason="overlap"), "opponent_position")):
        full.offer(sample(1, 1.04, own, opponent))
        decision = full.decide(1.04)
        assert decision.observation is None and [entry.split("] ")[1].split(":")[0] for entry in decision.blocked_by] == [group]
    full.offer(sample(2, 1.06, pose(0.4, 0.5), opponent_at(0.7, 0.6)))
    assert full.decide(1.08).observation is not None
    stale_imu = full.decide(1.30)                                          # IMU stopped at 1.0 s; the camera frame is 240 ms old, still inside its limit
    assert [entry.split(":")[0] for entry in stale_imu.blocked_by] == ["[0-5] imu"] and "stale" in stale_imu.inputs[GROUP_IMU].reason
    stale = full.decide(1.40)                                             # no new frame for 340 ms
    assert stale.observation is None and "stale" in stale.inputs[GROUP_OWN].reason and "stale" in stale.inputs[GROUP_OPPONENT].reason
    with pytest.raises(ValueError, match="timestamp order"):
        full.offer(sample(3, 0.5, pose(0.4, 0.5), opponent_at(0.7, 0.6)))
    with pytest.raises(ValueError, match="before the newest frame"):
        full.decide(1.0)


def test_telemetry_never_hands_out_a_sample_from_the_future():
    stream = imu_stream([0.0, 0.1, 0.2])
    assert stream.latest(-0.01) is None and stream.latest(0.05).t == 0.0 and stream.latest(0.1).t == 0.1 and stream.latest(9.0).t == 0.2


# ---- the causal loop, on the synthetic fixture ------------------------------------------------------

@pytest.fixture(scope="module")
def fixture_run(contract, tmp_path_factory):
    out = tmp_path_factory.mktemp("shadow-fixture")
    policy = CheckpointPolicy(contract)
    summary = demo.run_fixture(contract, out, policy, save_video=False)
    read = lambda name: [json.loads(line) for line in (out / name).read_text(encoding="utf-8").splitlines()]
    return SimpleNamespace(summary=summary, out=out, policy=policy, frames=read("fixture_frames.jsonl"),
                           camera_only=read("fixture_trace_camera_only.jsonl"), shadow=read("fixture_trace_with_synthetic_imu.jsonl"),
                           truth=json.loads((out / "fixture_truth.json").read_text(encoding="utf-8"))["frames"])


def phase(run, text):
    return next(row for row in run.summary["phases"] if text in row["phase"])


def test_fixture_tag_dropouts_and_recovery(fixture_run):
    one = phase(fixture_run, "ONE tag covered")
    assert one["own_pose_valid_frames"] == one["frames_delivered"] and set(one["tag_sets_used"]) == {"[0, 2, 3]"}
    assert one["proposals"] == one["decisions"] > 0                        # three of four tags still give a pose
    none = phase(fixture_run, "ALL tags covered")
    assert none["own_pose_valid_frames"] == 0 and none["proposals"] == 0 and none["opponent_valid_frames"] == none["frames_delivered"]
    back = phase(fixture_run, "tags uncovered")
    assert back["own_pose_valid_frames"] == back["frames_delivered"] and back["proposals"] == back["decisions"]
    single = phase(fixture_run, "THREE tags covered")
    assert single["own_pose_valid_frames"] == single["frames_delivered"] and set(single["tag_sets_used"]) == {"[2]"}
    truth = fixture_run.summary["against_fixture_truth"]                   # arithmetic on rendered pictures, not camera accuracy
    assert truth["own_position_error_mm"]["max"] < 5.0 and truth["own_heading_error_deg"]["max"] < 3.0
    assert truth["opponent_position_error_mm"]["max"] < 60.0              # a blob centre is not height-corrected
    missing = [d for d in fixture_run.shadow if not d["inputs"]["own_pose"]["valid"] and d["camera_frame"] is not None]
    assert missing and all(d["proposed"] is None and d["observation"] is None and d["mode"] == MODE_BLOCKED for d in missing)


def test_fixture_overlap_reports_unknown_and_recovers(fixture_run):
    touching = phase(fixture_run, "TOUCHING")
    assert touching["own_pose_valid_frames"] == touching["frames_delivered"] and touching["opponent_valid_frames"] == 0 and touching["proposals"] == 0
    tracks = fixture_run.summary["blob_tracks"]
    assert tracks["ids_seen"] == [1, 2] and len(tracks["overlap_spans_s"]) == 1 and tracks["positions_reported_during_overlap_or_loss"] == 0
    assert tracks["events"]["overlap begins"] == 2 and tracks["events"]["separated"] == 2
    overlapped = [f for f in fixture_run.frames if any(t["state"] == "overlap" for t in f["tracks"])]
    assert overlapped and all(f["opponent"]["state"] == "unknown" and f["opponent"]["x_m"] is None and f["opponent"]["reason"].startswith("overlap") for f in overlapped)
    away = phase(fixture_run, "backing away")
    assert 0 < away["opponent_valid_frames"] < away["frames_delivered"]    # unknown until the blobs part, then known again
    final = phase(fixture_run, "frames return")
    assert final["opponent_valid_frames"] == final["frames_delivered"] and final["proposals"] == final["decisions"]
    last = fixture_run.frames[-1]; truth = fixture_run.truth[-1]            # the moving unmarked object is followed
    assert math.hypot(last["opponent"]["x_m"] - truth["opponent"][0], last["opponent"]["y_m"] - truth["opponent"][1]) < 0.06


def test_fixture_stall_goes_stale_instead_of_holding_the_last_frame(fixture_run):
    stalled = [f["t_s"] for f in fixture_run.truth if not f["frame_delivered"]]
    start, end = stalled[0], stalled[-1]
    during = [d for d in fixture_run.shadow if start + 0.25 + 0.021 < d["t_s"] < end]
    assert len(during) > 10 and all(d["proposed"] is None and "stale" in d["inputs"]["own_pose"]["reason"] for d in during)
    assert all(d["inputs"]["imu"]["valid"] for d in during)                 # the stand-in IMU kept arriving: only the camera is stale
    after = [d for d in fixture_run.shadow if d["t_s"] > end + 0.1]
    assert after and all(d["proposed"] is not None for d in after)


def test_fixture_never_uses_a_frame_before_it_arrived_and_only_the_labelled_run_proposes(fixture_run, contract):
    delivered = [f["t_s"] for f in fixture_run.truth if f["frame_delivered"]]
    assert [f["t_s"] for f in fixture_run.frames] == delivered and delivered == sorted(delivered)
    for trace in (fixture_run.camera_only, fixture_run.shadow):
        assert all(d["camera_frame_t_s"] <= d["t_s"] + 1e-9 for d in trace)
        assert all(d["camera_frame_t_s"] == max(t for t in delivered if t <= d["t_s"] + 1e-9) for d in trace[::7])   # always the newest frame in hand
        assert [d["t_s"] for d in trace] == pytest.approx([i / contract.control_hz for i in range(len(trace))], abs=1e-6)
    assert all(d["proposed"] is None and d["observation"] is None and d["mode"] == MODE_BLOCKED for d in fixture_run.camera_only)
    assert all("[0-5] imu" in " ".join(d["blocked_by"]) for d in fixture_run.camera_only)
    ready = [d for d in fixture_run.camera_only if len(d["blocked_by"]) == 1]
    proposals = [d for d in fixture_run.shadow if d["proposed"] is not None]
    assert len(ready) == len(proposals) > 200                               # the IMU is the only thing between those and a proposal
    assert all(d["mode"] == demo.FIXTURE_SHADOW and "SYNTHETIC" in d["input_source"] and "LIVE" not in d["mode"].upper() for d in proposals)
    assert all(d["proposed"]["tactic"] in contract.tactics and len(d["observation"]) == 19 and d["observation"][6:13] == [0.0] * 7 for d in proposals)
    assert fixture_run.summary["imu_sensitivity"]["observations"] == len(proposals)
    assert fixture_run.summary["camera_observation_vs_exact_fixture_values"]["same_tactic"] > 0.9 * len(proposals)


def test_a_longer_stream_does_not_change_what_was_decided_earlier(contract):
    """Feeding more frames afterwards must leave every earlier frame result and decision untouched."""
    config, camera = load_config(), demo.default_camera()
    states = demo.fixture_timeline()

    def run(count, latency=0.0):
        mapping, estimate, _ = demo.fit_profile(config, fixture_clicks(config, camera), camera.image_size)
        pipeline = demo.CameraPipeline(config, mapping, estimate)
        pipeline.set_reference([demo.render_fixture_frame(config, camera, None, seed=9000)], "one empty synthetic frame")
        frames = ((s.t, demo.render_fixture_frame(config, camera, s, seed=100 + i)) for i, s in enumerate(states[:count]))
        assembler = ShadowAssembler(contract, "test", demo.fixture_imu(states, contract.imu_rate_hz))
        result = demo.run_stream("t", frames, pipeline, {"a": assembler}, contract.control_hz, assumed_latency_s=latency)
        strip = lambda report: {key: value for key, value in report.items() if key != "processing_ms"}
        return [strip(s.to_report()) for s in result.samples], [d.to_trace() for d in result.decisions["a"]]

    short_frames, short_decisions = run(24)
    long_frames, long_decisions = run(40)
    assert long_frames[:24] == short_frames and long_decisions[:len(short_decisions)] == short_decisions
    _, late = run(24, latency=0.1)                                          # a frame that arrives 100 ms late is at least 100 ms old when used
    assert all(d["t_s"] - d["camera_frame_t_s"] >= 0.1 - 1e-6 for d in late) and late[0]["t_s"] == pytest.approx(0.1)


# ---- nothing can reach the robot ---------------------------------------------------------------------

def test_no_serial_port_robot_link_or_command_path_is_imported():
    code = ("import sys, control_center.tools.shadow_stream_demo, control_center.tools.stream_adapter, control_center.tools.observation_contract;"
            "print([m for m in sys.modules if m.split('.')[0] == 'serial' or m.rsplit('.', 1)[-1] in "
            "('controller_link', 'robot_link', 'command_manager', 'hub', 'round_recorder', 'shadow')])")
    result = subprocess.run([sys.executable, "-c", code], cwd=REPO, capture_output=True, text=True, timeout=300)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().splitlines()[-1] == "[]"
    for name in ("shadow_stream_demo.py", "stream_adapter.py", "observation_contract.py"):
        text = (REPO / "control_center" / "tools" / name).read_text(encoding="utf-8")
        assert "import serial" not in text and "controller_link" not in text.replace("the controller link", "") and "Serial(" not in text


# ---- camera mode, through a stand-in capture device and window ----------------------------------------

@pytest.fixture()
def home():
    config = load_config(REPO / "config" / "home_test_overlay.yaml")
    camera = make_camera((0.25, -0.35, 0.60), (0.25, 0.20, 0.0), focal_px=1000.0, image_size=(1280, 720))   # the tag is about 36 px wide
    return config, camera


def stand_in_window(monkeypatch, frames, keys):
    state = {"reads": 0, "opened": 0, "shown": []}

    class FakeCapture:
        def __init__(self, *args): state["opened"] += 1
        def isOpened(self): return True
        def set(self, prop, value): return True
        def read(self):
            index = min(state["reads"], len(frames) - 1); state["reads"] += 1
            return True, frames[index].copy()
        def release(self): pass

    monkeypatch.setattr(cv2, "VideoCapture", FakeCapture)
    monkeypatch.setattr(cv2, "namedWindow", lambda *a, **k: None)
    monkeypatch.setattr(cv2, "imshow", lambda name, image: state["shown"].append(image.shape))
    monkeypatch.setattr(cv2, "waitKey", lambda delay: keys.get(state["reads"], 255))
    monkeypatch.setattr(cv2, "getWindowProperty", lambda *a: 1.0)
    monkeypatch.setattr(cv2, "destroyAllWindows", lambda: None)
    return state


def test_home_test_config_maps_an_a4_sheet(home):
    config, camera = home
    mapping, _, quality = demo.fit_profile(config, fixture_clicks(config, camera), camera.image_size)
    assert mapping.valid and quality.level == "ok" and mapping.check_stats["count"] == 1 and mapping.check_stats["max_mm"] < 1.0
    corners = {ref.name: ref.arena_xy for ref in config.references}
    assert corners["a4_near_right"][0] - corners["a4_near_left"][0] == pytest.approx(0.297) and corners["a4_far_left"][1] - corners["a4_near_left"][1] == pytest.approx(0.210)
    four = {name: point for name, point in fixture_clicks(config, camera).points.items() if name != "a4_centre"}
    clicks = fixture_clicks(config, camera); clicks.points = four
    assert demo.fit_profile(config, clicks, camera.image_size)[2].level == "weak"      # usable, and labelled: nothing independent checks it


def test_home_calibration_command_saves_a_separate_profile_through_the_existing_overlay_tool(tmp_path, monkeypatch, home):
    """`arena_overlay --camera N --config config/home_test_overlay.yaml --calibrate`: click the four A4 corners and the centre."""
    from control_center.tools import arena_overlay
    from control_center.tools.overlay_core import load_clicks
    from control_center.tools.overlay_fixture import reference_pixels

    config, camera = home
    frame = render_frame(config, camera, Scene(), label=False); pixels = reference_pixels(config, camera)
    script = []
    for number, ref in enumerate(config.references, start=1):
        script += [("key", ord(str(number))), ("click", pixels[ref.name])]
    script += [("key", ord("s")), ("key", ord("q"))]
    state = stand_in_window(monkeypatch, [frame], {})

    def wait_key(delay):
        kind, value = script.pop(0)
        if kind == "click":
            state["mouse"](cv2.EVENT_LBUTTONDOWN, int(round(value[0])), int(round(value[1])), 0, None); return 255
        return value

    monkeypatch.setattr(cv2, "setMouseCallback", lambda name, callback: state.__setitem__("mouse", callback))
    monkeypatch.setattr(cv2, "waitKey", wait_key)
    points = tmp_path / "home_test_clicks.json"
    report = arena_overlay.run(arena_overlay.build_parser().parse_args(
        ["--camera", "0", "--config", str(REPO / "config" / "home_test_overlay.yaml"), "--calibrate", "--points", str(points)]))
    saved = load_clicks(points)
    assert script == [] and set(saved.points) == {ref.name for ref in config.references} and saved.kind == "live" and not saved.historical
    assert report["mapping"]["valid"] and report["mapping"]["check_points"]["max_mm"] < 3.0
    assert demo.profile_points_path("home_test") != demo.profile_points_path("live")     # the home profile is its own file


def test_camera_mode_end_to_end_with_a_stand_in_camera(tmp_path, monkeypatch, home, contract):
    """The loop the home test runs: reference on SPACE, a tagged and an unmarked object, then the tag covered."""
    config, camera = home
    points = tmp_path / "home_clicks.json"; save_clicks(points, fixture_clicks(config, camera), source="synthetic")
    empty = render_frame(config, camera, Scene(), label=False)

    def scene(tagged):
        frame = render_frame(config, camera, Scene([BotInstance(0.13, 0.20, math.radians(90), ids=[0])] if tagged else []), label=False) if tagged else empty.copy()
        if not tagged:
            demo._paint_box(frame, camera, (0.13, 0.20), math.radians(90), demo.PLATE_SIZE_M, 0.030, (45, 45, 45))
        demo._paint_box(frame, camera, (0.39, 0.22), 0.3, (0.07, 0.06), 0.030, demo.OPPONENT_COLOUR)
        return frame

    config_one_tag = REPO / "config" / "home_test_overlay.yaml"
    frames = [empty] * 25 + [scene(True)] * 30 + [scene(False)] * 15
    state = stand_in_window(monkeypatch, frames, {4: 32, 70: ord("q")})
    args = demo.build_parser().parse_args(["--camera", "0", "--config", str(config_one_tag), "--points", str(points), "--out", str(tmp_path / "out")])
    summary = demo.run_camera(args, contract)
    assert state["opened"] == 1 and summary["frames"] == 70 and summary["modes"] == {MODE_BLOCKED: 70}     # no IMU: never a proposal
    trace = [json.loads(line) for line in Path(summary["trace"]).read_text(encoding="utf-8").splitlines()]
    assert all(d["proposed"] is None and d["observation"] is None and not d["inputs"]["imu"]["valid"] for d in trace)
    assert "LIVE camera" in trace[0]["input_source"] and not any(d["inputs"]["opponent_position"]["valid"] for d in trace[:25])
    both = [d for d in trace[40:55]]
    assert all(d["inputs"]["own_pose"]["valid"] and d["inputs"]["opponent_position"]["valid"] for d in both)
    assert all(d["blocked_by"] == ["[0-5] imu: no IMU telemetry (none connected)"] for d in both)
    values = both[-1]["inputs"]
    assert math.hypot(values["own_pose"]["values"]["x_m"] - 0.13, values["own_pose"]["values"]["y_m"] - 0.20) < 0.01
    assert math.hypot(values["opponent_position"]["values"]["x_m"] - 0.39, values["opponent_position"]["values"]["y_m"] - 0.22) < 0.04
    covered = trace[58:]                                                   # tag covered: own pose gone at once, the other blob still followed
    assert all(not d["inputs"]["own_pose"]["valid"] and "no configured bot markers" in d["inputs"]["own_pose"]["reason"] for d in covered)
    assert all(d["inputs"]["opponent_position"]["valid"] for d in covered)
    assert state["shown"] and state["shown"][0] == (720, 1280 + 600, 3)


def test_camera_mode_refuses_a_footage_profile_before_any_camera_is_opened(tmp_path, monkeypatch, contract):
    state = stand_in_window(monkeypatch, [np.zeros((360, 640, 3), np.uint8)], {})
    args = demo.build_parser().parse_args(["--camera", "0", "--profile", demo.HISTORICAL_PROFILE, "--out", str(tmp_path)])
    with pytest.raises(ValueError, match="HISTORICAL"):
        demo.run_camera(args, contract)
    assert state["opened"] == 0 and demo.main(["--camera", "0", "--profile", demo.HISTORICAL_PROFILE, "--out", str(tmp_path)]) == 2
    assert state["opened"] == 0


def test_camera_mode_without_a_map_shows_tag_ids_only(tmp_path, monkeypatch, home, contract):
    config, camera = home
    frame = render_frame(config, camera, Scene([BotInstance(0.2, 0.2, 0.0, ids=[0])]), label=False)
    stand_in_window(monkeypatch, [frame], {5: 32, 30: ord("q")})
    args = demo.build_parser().parse_args(["--camera", "0", "--config", str(REPO / "config" / "home_test_overlay.yaml"),
                                           "--points", str(tmp_path / "missing.json"), "--out", str(tmp_path / "out")])
    summary = demo.run_camera(args, contract)
    trace = [json.loads(line) for line in Path(summary["trace"]).read_text(encoding="utf-8").splitlines()]
    assert all("no valid floor mapping" in d["inputs"]["own_pose"]["reason"] for d in trace)
    assert all("no startup reference" in d["inputs"]["opponent_position"]["reason"] for d in trace) and "reference NOT taken" in summary["last_message"]


# ---- the recorded footage (only when it is available) -------------------------------------------------

def test_footage_is_fed_causally_and_every_decision_is_blocked(tmp_path, contract):
    video = os.environ.get(demo.FOOTAGE_ENV)
    if not video or not Path(video).is_file():
        pytest.skip(f"set {demo.FOOTAGE_ENV} to the Bristol recording to run this")
    with pytest.raises(ValueError, match="strictly before"):
        demo.run_footage(Path(video), contract, tmp_path, start_s=433.0, duration_s=1.0, reference_at_s=432.5, save_video=False)
    summary = demo.run_footage(Path(video), contract, tmp_path, duration_s=6.0, save_video=False)
    assert summary["frames"] == 180 and summary["notes"]["profile"] == demo.HISTORICAL_PROFILE and "before the clip" in summary["notes"]["startup_reference"]
    assert summary["own_pose"]["valid_frames"] == 0 and summary["own_pose"]["markers_decoded_frames"] == 0    # no tags in that recording
    assert summary["opponent"]["valid_frames"] == 0 and summary["blob_tracks"]["frames_with_an_overlap"] > 10
    assert summary["blob_tracks"]["positions_reported_during_overlap_or_loss"] == 0
    block = summary["decisions"]["footage"]
    assert block["complete_observations"] == 0 and block["proposals"] == 0 and block["modes"] == {MODE_BLOCKED: block["count"]}
    assert set(block["decisions_blocked_by_field"]) == {"[0-5] imu", "[13-16] own_pose", "[17-18] opponent_position"}
    frames = [json.loads(line) for line in (tmp_path / "footage_frames.jsonl").read_text(encoding="utf-8").splitlines()]
    times = [f["t_s"] for f in frames]
    assert times == sorted(times) and times[0] >= 432.99 and all(f["opponent"]["x_m"] is None for f in frames)
