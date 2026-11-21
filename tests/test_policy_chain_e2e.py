"""END-TO-END, SIMULATED: synthetic arena frames -> tracker poses -> observation -> trained policy -> tactic -> command plan.

Evidence class: SIMULATED / FIXTURE. Every picture is rendered by overlay_fixture / track_fixture (ideal pinhole camera, flat coloured
slab for the unmarked opponent) and every IMU line is scripted in the firmware's format (gravity only, as the simulator generates
it). Nothing here is a camera, a serial port, a socket or a robot, so NOTHING here is hardware evidence; the decisions are labelled
SIMULATED and are never valid_for_control. The one LIVE-labelled check at the end exercises the labelling rule only: the test itself
declares the origin, the inputs are still synthetic.

What it proves: the chain's directions are consistent. An opponent on the robot's left (counter-clockwise side) produces a policy
tactic turn_left, whose command makes the firmware mixer drive the right motor faster than the left (counter-clockwise), and an
opponent straight ahead produces advance. What it cannot prove: that the real camera, IMU axes or wheels behave that way.
"""

import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from control_center.backend import observation_builder as ob  # noqa: E402
from control_center.backend import policy_engine as pe  # noqa: E402
from control_center.backend.command_mapping import MapperState, firmware_drive, map_tactic  # noqa: E402
from control_center.tools import live_shadow  # noqa: E402
from control_center.tools import track_fixture as tf  # noqa: E402
from control_center.tools.live_telemetry import ImuMounting, LiveTelemetry  # noqa: E402
from control_center.tools.observation_contract import RECOMMENDED_CHECKPOINT, load_contract  # noqa: E402
from control_center.tools.overlay_core import load_config  # noqa: E402
from control_center.tools.overlay_fixture import default_camera, fixture_clicks  # noqa: E402

HAS_STACK = RECOMMENDED_CHECKPOINT.is_file()
try:
    import stable_baselines3  # noqa: F401
    import torch  # noqa: F401
except Exception:                                   # pragma: no cover - environment without the RL stack
    HAS_STACK = False
pytestmark = pytest.mark.skipif(not HAS_STACK, reason="SKIPPED: stable-baselines3 / torch / the checkpoint zip are not available here")

DT = 1.0 / 30.0
RED = (40, 40, 220)
OWN = (0.575, 0.575, 0.3)                           # x, y (m, arena frame), heading (rad, counter-clockwise from +x)
SIZE = (0.11, 0.10)


def polar(bearing: float, distance: float = 0.30):
    angle = OWN[2] + bearing
    return OWN[0] + distance * math.cos(angle), OWN[1] + distance * math.sin(angle)


SCENES = {                                           # bearing of the opponent relative to the nose; + = counter-clockwise (left)
    "ahead": (0.0, "advance", 0),
    "left": (math.radians(55.0), "turn_left", -1),
    "right": (math.radians(-55.0), "turn_right", +1),
}


def imu_line(sequence: int, t: float) -> str:
    return f"IMU seq={sequence} t_us={int(t * 1e6)} ok=1 ax=0.0000 ay=0.0000 az=9.8067 gx=0.000 gy=0.000 gz=0.000 unit=m/s2,deg/s"


def run_scene(opponent_xy, with_imu=True, frames=10):
    config, camera = load_config(), default_camera()
    telemetry = LiveTelemetry(ImuMounting(np.eye(3), True, Path("fixture-mounting.yaml")))
    shadow = live_shadow.LiveShadow(load_contract(), config, fixture_clicks(config, camera), telemetry, "SYNTHETIC fixture frames + scripted IMU lines",
                                    shadow_label="FIXTURE SHADOW", multi_marker=True, click_opponent=True)
    shadow.start(camera.image_size)
    sequence, last = 0, None
    for index in range(frames):
        t = index * DT
        if with_imu:
            for step in range(4):
                sequence += 1
                telemetry.feed("RX", imu_line(sequence, t - 0.03 + 0.01 * step), t - 0.03 + 0.01 * step)
        frame = tf.render_bot(config, camera, *OWN, seed=11 + index)
        tf.paint_box(frame, camera, opponent_xy, 0.2, SIZE, 0.034, RED)
        if index == 2:
            u, v = camera.project([[*opponent_xy, 0.034]])[0]
            assert shadow.pipeline.select_opponent(frame, (float(u), float(v)), t).ok
        sample, decision = shadow.step(frame, t)
        last = (t, sample, decision)
    return telemetry, last


@pytest.mark.parametrize("scene", list(SCENES))
def test_frames_to_command_the_whole_chain_is_consistent_and_labelled_simulated(scene):
    bearing, expected_tactic, expected_steer_sign = SCENES[scene]
    opponent = polar(bearing)
    telemetry, (t, sample, assembled) = run_scene(opponent)

    # 1. tracker poses (SIMULATED frames)
    assert sample.own.ok and sample.own_track["state"] == "measured" and sample.opponent.ok
    assert (sample.own.x_m, sample.own.y_m) == pytest.approx(OWN[:2], abs=0.02)
    assert math.radians(sample.own.heading_deg) == pytest.approx(OWN[2], abs=math.radians(3.0))
    assert sample.opponent.xy_m == pytest.approx(opponent, abs=0.05)

    # 2. the existing assembler path produced a complete vector, labelled SIMULATED and not valid for control
    assert assembled.observation is not None and assembled.observation.shape == (19,) and assembled.blocked_by == []
    assert assembled.source == "SIMULATED" and assembled.valid_for_control is False
    assert set(assembled.ages_s) == {"imu", "previous_action", "own_pose", "opponent_position"}

    # 3. the builder reaches the same vector from the same objects (one schema, two code paths, no drift)
    inputs = ob.inputs_from_camera(sample, t, telemetry, origin=ob.SOURCE_SIMULATED)
    result = ob.build_observation(inputs, t)
    assert result.values is not None and np.allclose(result.values, assembled.observation, atol=1e-6)
    assert result.source == "SIMULATED" and result.valid_for_control is False and not result.missing and not result.stale

    # 4. the single shared policy -> a tactic, still diagnostic
    decision = pe.get_engine().decide(result)
    assert decision.tactic == expected_tactic, (scene, decision.tactic, decision.probabilities)
    assert decision.source == "SIMULATED" and decision.valid_for_control is False and decision.blocked_by

    # 5. tactic -> command plan: right sign, inside the cap, and the mixer turns the robot toward the opponent
    state = MapperState()
    plan = map_tactic(decision.tactic, t, decision.input_time, state=state)
    assert abs(plan.throttle) <= 60 and abs(plan.steer) <= 60
    left, right = firmware_drive(plan.intended_throttle, plan.intended_steer)
    if scene == "ahead":
        assert plan.intended_throttle > 0 and plan.intended_steer == 0 and left == right > 0
    else:
        assert plan.intended_throttle == 0 and (plan.intended_steer > 0) == (expected_steer_sign > 0) and plan.intended_steer != 0
        assert (right > left) == (scene == "left")                 # counter-clockwise (right motor faster) when the opponent is on the left
    assert plan.reason.startswith(("ok", "slew-limited"))          # the cap is not what limits it at the default scale


def test_a_live_label_without_an_imu_never_reaches_a_command():
    """The labelling rule on the same synthetic frames: declaring LIVE does not help when there is no IMU line."""
    telemetry, (t, sample, assembled) = run_scene(polar(0.0), with_imu=False)
    assert assembled.observation is None and any("imu" in entry for entry in assembled.blocked_by)
    inputs = ob.inputs_from_camera(sample, t, telemetry, origin=ob.SOURCE_LIVE, camera_origin=ob.SOURCE_LIVE, imu_origin=ob.SOURCE_LIVE)
    result = ob.build_observation(inputs, t)
    assert result.values is None and result.valid_for_control is False and result.missing == ("imu",)
    decision = pe.get_engine().decide(result)
    assert decision.tactic is None and decision.valid_for_control is False and decision.source == "LIVE"
    plan = map_tactic(decision.tactic, t, decision.input_time if decision.input_time is not None else t)
    assert (plan.throttle, plan.steer) == (0, 0) and plan.reason.startswith("no tactic")


def test_the_live_label_with_complete_fresh_inputs_is_what_marks_a_vector_valid_for_control():
    """LABELLING LOGIC ONLY: the test declares the origins; the frames and IMU lines are still synthetic."""
    telemetry, (t, sample, _) = run_scene(polar(math.radians(55.0)))
    inputs = ob.inputs_from_camera(sample, t, telemetry, origin=ob.SOURCE_LIVE, camera_origin=ob.SOURCE_LIVE, imu_origin=ob.SOURCE_LIVE)
    result = ob.build_observation(inputs, t)
    assert result.source == "LIVE" and result.valid_for_control is True and result.blocked_by == ()
    decision = pe.get_engine().decide(result)
    assert decision.tactic == "turn_left" and decision.valid_for_control is True
    stale = ob.build_observation(inputs, t + 0.30)                  # the same inputs 300 ms later: stale, so no vector at all
    assert stale.values is None and stale.valid_for_control is False and set(stale.stale) == {"imu", "own_pose", "opponent_position"}
    plan = map_tactic(decision.tactic, t + 0.30, decision.input_time)
    assert (plan.throttle, plan.steer) == (0, 0) and "stale" in plan.reason          # and the mapper refuses the old decision too


# ---------------------------------------------------------------- the Command Centre's own tracking engine + ShadowRunner


def test_tracking_engine_shadow_takes_its_source_from_the_capture_state_and_survives_inference_errors():
    """SIMULATED frames through TrackingEngine + ShadowRunner (the screen's own path, minus Qt): the camera's declared state
    (LIVE / MOCK / REPLAY, normally hub.capture.state) decides the decision's source; a failing policy blocks, it does not raise
    and it is not discarded."""
    from control_center.backend.tracking_worker import HubTelemetry, ShadowRunner, TrackingEngine
    from control_center.tools.stream_adapter import CheckpointPolicy

    config, camera = load_config(), default_camera()
    contract = load_contract()
    telemetry = HubTelemetry(ImuMounting(np.eye(3), True, Path("fixture-mounting.yaml")))
    real = CheckpointPolicy(contract)
    real.load()

    class Flaky:
        def __init__(self):
            self.fail, self.loaded = False, True

        def propose(self, observation):
            if self.fail:
                raise RuntimeError("boom")
            return real.propose(observation)

    flaky = Flaky()
    clock = {"t": 0.0}
    engine = TrackingEngine(config, fixture_clicks(config, camera), "synthetic fixture", clock=lambda: clock["t"])
    runner = ShadowRunner(contract, config, fixture_clicks(config, camera), telemetry, "synthetic frames", flaky)
    engine.set_shadow(runner)
    opponent = polar(math.radians(55.0))
    frame = tf.render_bot(config, camera, *OWN)
    tf.paint_box(frame, camera, opponent, 0.2, SIZE, 0.034, RED)
    u, v = camera.project([[*opponent, 0.034]])[0]
    box = (float(u) - 34.0, float(v) - 30.0, 68.0, 60.0)

    def step(index, state, imu=True):
        clock["t"] = index * DT
        if imu:
            telemetry.feed("RX", f"IMU,{index * 33},0.0,0.0,9.81,0.0,0.0,0.0", clock["t"])
        engine.camera_state = state
        if index == 2:
            engine.select_roi(box, frame, clock["t"])
        return engine.process(index, clock["t"], frame).shadow

    for index in range(2):
        step(index, "LIVE")
    first = step(2, "LIVE")
    live = step(3, "LIVE")
    assert live["blocked_by"] == [] and live["proposed"]["tactic"] == "turn_left"
    assert (live["source"], live["valid_for_control"]) == ("LIVE", True) and set(live["ages_ms"]) == {"imu", "previous_action", "own_pose", "opponent_position"}
    assert first["source"] == "LIVE"
    mock = step(4, "MOCK")
    assert mock["proposed"]["tactic"] == "turn_left" and (mock["source"], mock["valid_for_control"]) == ("SIMULATED", False)
    replay = step(5, "REPLAY")
    assert (replay["source"], replay["valid_for_control"]) == ("REPLAY", False)
    undeclared = step(6, None)
    assert (undeclared["source"], undeclared["valid_for_control"]) == ("SIMULATED", False)             # nothing is LIVE by default
    clock["t"] = 7 * DT + 0.5                                                                         # the wall clock moved on 0.5 s: IMU and frame are old
    no_imu = engine.process(7, 7 * DT, frame).shadow
    assert no_imu["proposed"] is None and no_imu["valid_for_control"] is False and any("imu" in e for e in no_imu["blocked_by"])

    flaky.fail = True
    failed = step(8, "LIVE")
    assert failed["proposed"] is None and failed["valid_for_control"] is False and "inference error" in failed["error"]
    assert any("boom" in entry for entry in failed["blocked_by"]) and runner.assembler.policy is flaky      # blocked, not discarded
    flaky.fail = False
    recovered = step(9, "LIVE")
    assert recovered["proposed"]["tactic"] == "turn_left" and recovered["error"] == "" and recovered["valid_for_control"] is True
