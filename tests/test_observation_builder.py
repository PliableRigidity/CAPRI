"""control_center/backend/observation_builder.py: physical inputs -> the 19-value observation, with provenance.

Evidence class: SOFTWARE / SIMULATED. No camera, port or socket is opened. Where a vector is compared with the simulator's own
observation (AutoBotEnv._obs), both sides are simulator values: that proves the builder reproduces the TRAINING-time layout and
scaling, not that any live sensor produces such values.
"""

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from control_center.backend import observation_builder as ob  # noqa: E402
from control_center.backend.observation_builder import (AgeLimits, ImuInput, ObservationInputs, OpponentInput,  # noqa: E402
                                                        OwnPoseInput, build_observation)
from control_center.tools.observation_contract import RECOMMENDED_CHECKPOINT  # noqa: E402

pytestmark = pytest.mark.skipif(not RECOMMENDED_CHECKPOINT.is_file(), reason="runs/recovery-challenger-v2-seed1/final_model.zip is not present")

W = H = 1.15                       # runs/ppo-curriculum-overnight/effective_sim.yaml arena.width_m / height_m
GYRO_RANGE, ACCEL_RANGE = 34.9, 39.2
NOW = 100.0
GYRO, ACCEL = (0.5, -1.0, 2.0), (0.3, -0.2, 9.81)
OWN = (0.40, 0.70, math.radians(90.0))
OPP = (0.90, 0.20)


def inputs(origin=ob.SOURCE_LIVE, imu=True, own=True, opp=True, imu_age=0.02, cam_age=0.05, verified=True, **extra):
    return ObservationInputs(
        origin=origin,
        own=OwnPoseInput(OWN[0], OWN[1], OWN[2], NOW - cam_age, "measured") if own else None,
        opponent=OpponentInput(OPP[0], OPP[1], NOW - cam_age, "tracked") if opp else None,
        imu=ImuInput(GYRO, ACCEL, NOW - imu_age, None, verified) if imu else None,
        **extra)


def expected_vector():
    """Hand-built from AutoBotEnv._obs (autobot/sim/env.py:133-149): gyro/range, accel/range, 7 zeros, own x, own y, sin, cos, opp x, opp y."""
    return np.array([GYRO[0] / GYRO_RANGE, GYRO[1] / GYRO_RANGE, GYRO[2] / GYRO_RANGE,
                     ACCEL[0] / ACCEL_RANGE, ACCEL[1] / ACCEL_RANGE, ACCEL[2] / ACCEL_RANGE,
                     0, 0, 0, 0, 0, 0, 0,
                     2 * OWN[0] / W - 1, 2 * OWN[1] / H - 1, math.sin(OWN[2]), math.cos(OWN[2]),
                     2 * OPP[0] / W - 1, 2 * OPP[1] / H - 1], dtype=np.float32)


# ---------------------------------------------------------------- ordering and scaling


def test_vector_matches_the_hand_built_expectation_channel_by_channel():
    result = build_observation(inputs(), NOW)
    assert result.values is not None and result.values.shape == (19,) and result.values.dtype == np.float32
    assert np.allclose(result.values, expected_vector(), atol=1e-6)
    names = ["gyro_x", "gyro_y", "gyro_z", "accel_x", "accel_y", "accel_z"] + [f"prev{i}" for i in range(7)] + \
            ["own_x", "own_y", "sin_yaw", "cos_yaw", "opp_x", "opp_y"]
    for index, name in enumerate(names):
        assert result.values[index] == pytest.approx(expected_vector()[index], abs=1e-6), (index, name)
    assert [result.groups[g].indices for g in ("imu", "previous_action", "own_pose", "opponent_position")] == ["0-5", "6-12", "13-16", "17-18"]


def test_values_are_clipped_like_the_training_environment():
    wild = ObservationInputs(origin=ob.SOURCE_LIVE, own=OwnPoseInput(0.40, 0.70, 0.0, NOW), opponent=OpponentInput(0.9, 0.2, NOW),
                             imu=ImuInput((100.0, -100.0, 0.0), (0.0, 0.0, 100.0), NOW, None, True))
    values = build_observation(wild, NOW).values
    assert values[0] == 1.0 and values[1] == -1.0 and values[5] == 1.0 and np.abs(values).max() <= 1.0


def test_builder_reproduces_the_simulator_observation_exactly():
    """SIMULATED: the training env's own observation, rebuilt from the same physical truth through the builder."""
    pytest.importorskip("mujoco", reason="mujoco is needed to read the training environment's observation")
    from autobot.sim.env import AutoBotEnv

    config = ROOT / "runs" / "ppo-curriculum-overnight" / "effective_sim.yaml"      # the config the checkpoint was trained with
    if not config.is_file():
        pytest.skip(f"{config} is not present")
    env = AutoBotEnv(str(config), observation_profile="sim_debug", action_profile="experimental5_v1",
                     previous_action_encoding="masked_zero_v1", seed=3)
    env.reset(seed=3, options={"scenario": {"player": [0.4, 0.5, 0.7], "opponent": [0.8, 0.9, -1.0], "name": "builder-check"}})
    for action in (1, 3, 4, 2, 0):
        observation, *_ = env.step(action)
        imu, own, opp = env.sim.imu(), env.sim.truth(0), env.sim.truth(1)         # the sample _obs just read (cached for 10 ms)
        simulated = ObservationInputs(
            origin=ob.SOURCE_SIMULATED,
            own=OwnPoseInput(float(own["position_m"][0]), float(own["position_m"][1]), float(own["yaw_rad"]), NOW),
            opponent=OpponentInput(float(opp["position_m"][0]), float(opp["position_m"][1]), NOW),
            imu=ImuInput(tuple(imu.gyro_rad_s), tuple(imu.accel_m_s2), NOW, None, True))
        result = build_observation(simulated, NOW)
        assert result.values is not None
        assert np.allclose(result.values, observation, atol=2e-6), (action, result.values - observation)
        assert result.source == ob.SOURCE_SIMULATED and result.valid_for_control is False


def test_the_previous_action_slots_are_constants_the_training_env_defined():
    """masked_zero_v1: the 7 previous-action channels are exactly 0.0 (env.py:137-139); no other channel is ever a constant."""
    a = build_observation(inputs(), NOW)
    b = build_observation(inputs(previous_action=3), NOW)
    assert np.array_equal(a.values[6:13], np.zeros(7, np.float32)) and np.array_equal(a.values, b.values)
    group = a.groups["previous_action"]
    assert group.status == "OK" and group.origin == ob.ORIGIN_CONSTANT and group.age_s is None
    assert "masked_zero_v1" in group.reason
    for index in list(range(0, 6)) + list(range(13, 19)):                           # every other channel responds to its input
        moved = inputs()
        if index < 3:
            g = list(GYRO); g[index] += 1.0; moved = ObservationInputs(**{**moved.__dict__, "imu": ImuInput(tuple(g), ACCEL, NOW - 0.02, None, True)})
        elif index < 6:
            a_ = list(ACCEL); a_[index - 3] += 1.0; moved = ObservationInputs(**{**moved.__dict__, "imu": ImuInput(GYRO, tuple(a_), NOW - 0.02, None, True)})
        elif index in (13, 15, 16):
            moved = ObservationInputs(**{**moved.__dict__, "own": OwnPoseInput(OWN[0] + (0.1 if index == 13 else 0), OWN[1], OWN[2] + (0.3 if index != 13 else 0), NOW - 0.05)})
        elif index == 14:
            moved = ObservationInputs(**{**moved.__dict__, "own": OwnPoseInput(OWN[0], OWN[1] + 0.1, OWN[2], NOW - 0.05)})
        else:
            moved = ObservationInputs(**{**moved.__dict__, "opponent": OpponentInput(OPP[0] + (0.1 if index == 17 else 0), OPP[1] + (0.1 if index == 18 else 0), NOW - 0.05)})
        assert build_observation(moved, NOW).values[index] != a.values[index], f"channel {index} did not respond to its input"


# ---------------------------------------------------------------- missing and stale: never a vector, never a made-up value


def test_missing_imu_blocks_everything_and_nothing_is_filled_in():
    result = build_observation(inputs(imu=False), NOW)
    assert result.values is None and result.valid_for_control is False and result.complete is False
    assert result.missing == ("imu",) and result.stale == ()
    assert result.groups["imu"].status == "MISSING" and any("imu" in line and "does not send" in line for line in result.blocked_by)
    assert result.groups["own_pose"].ok and result.groups["opponent_position"].ok and result.groups["previous_action"].ok


@pytest.mark.parametrize("missing", ["imu", "own", "opp"])
def test_each_missing_group_alone_withholds_the_whole_vector(missing):
    result = build_observation(inputs(**{missing: False}), NOW)
    assert result.values is None and not result.valid_for_control and len(result.missing) == 1


def test_an_unverified_imu_mounting_is_a_missing_imu():
    result = build_observation(inputs(verified=False), NOW)
    assert result.values is None and result.groups["imu"].status == "MISSING" and "not verified" in result.groups["imu"].reason


def test_camera_age_limit_is_025_s_and_imu_age_limit_is_010_s():
    assert (AgeLimits().camera_s, AgeLimits().imu_s) == (0.25, 0.10) == (ob.MAX_CAMERA_AGE_S, ob.MAX_IMU_AGE_S)
    assert build_observation(inputs(cam_age=0.25, imu_age=0.10), NOW).values is not None             # exactly at the limits
    camera_old = build_observation(inputs(cam_age=0.26), NOW)
    assert camera_old.values is None and set(camera_old.stale) == {"own_pose", "opponent_position"} and "ms old" in camera_old.blocked_by[0]
    imu_old = build_observation(inputs(imu_age=0.11), NOW)
    assert imu_old.values is None and imu_old.stale == ("imu",) and imu_old.groups["imu"].status == "STALE"
    assert imu_old.groups["imu"].age_s == pytest.approx(0.11) and not imu_old.valid_for_control


@pytest.mark.parametrize("state", ["uncertain", "lost", "not_selected", "ambiguous", "unknown", "TRACKED ", ""])
def test_only_a_tracked_opponent_counts_uncertain_or_lost_is_missing(state):
    stale_position = OpponentInput(OPP[0], OPP[1], NOW - 0.01, state)               # it still carries a position: it must not be used
    base = inputs()
    result = build_observation(ObservationInputs(**{**base.__dict__, "opponent": stale_position}), NOW)
    assert result.values is None and result.missing == ("opponent_position",) and not result.valid_for_control
    assert result.groups["opponent_position"].values == {}
    assert state.upper().strip() in result.groups["opponent_position"].reason.upper() or "opponent" in result.groups["opponent_position"].reason


def test_the_last_position_is_never_substituted():
    seen = build_observation(inputs(), NOW)
    lost = build_observation(ObservationInputs(**{**inputs().__dict__, "opponent": OpponentInput(None, None, NOW, "lost", "TRACKING LOST")}), NOW + 0.05)
    assert seen.values is not None and lost.values is None and lost.groups["opponent_position"].status == "MISSING"
    assert "TRACKING LOST" in lost.groups["opponent_position"].reason


def test_own_pose_states_and_bad_numbers():
    base = inputs().__dict__
    for state in ("lost", "stale", "unknown"):
        r = build_observation(ObservationInputs(**{**base, "own": OwnPoseInput(OWN[0], OWN[1], OWN[2], NOW, state)}), NOW)
        assert r.values is None and r.missing == ("own_pose",)
    assert build_observation(ObservationInputs(**{**base, "own": OwnPoseInput(OWN[0], OWN[1], OWN[2], NOW, "predicted")}), NOW).values is not None
    for bad in ((float("nan"), 0.5, 0.0), (0.5, float("inf"), 0.0), (0.5, 0.5, float("nan")), (None, 0.5, 0.0), (3.0, 0.5, 0.0), (0.5, -0.5, 0.0)):
        r = build_observation(ObservationInputs(**{**base, "own": OwnPoseInput(*bad, NOW)}), NOW)
        assert r.values is None and r.missing == ("own_pose",), bad
    for bad in ((float("nan"), 0.5), (0.5, None), (2.0, 0.5)):
        r = build_observation(ObservationInputs(**{**base, "opponent": OpponentInput(*bad, NOW)}), NOW)
        assert r.values is None and r.missing == ("opponent_position",), bad
    for bad in ((float("nan"), 0.0, 0.0), (0.0, float("inf"), 0.0)):
        r = build_observation(ObservationInputs(**{**base, "imu": ImuInput(bad, ACCEL, NOW, None, True)}), NOW)
        assert r.values is None and r.missing == ("imu",)


def test_a_timestamp_from_the_future_is_a_fault_not_a_fresh_input():
    r = build_observation(inputs(cam_age=-0.5), NOW)
    assert r.values is None and "future" in r.groups["own_pose"].reason
    assert build_observation(inputs(cam_age=-0.004), NOW).values is not None                 # within the 5 ms tolerance
    assert build_observation(ObservationInputs(**{**inputs().__dict__, "imu": ImuInput(GYRO, ACCEL, float("nan"), None, True)}), NOW).values is None


# ---------------------------------------------------------------- LIVE vs SIMULATED vs REPLAY, and valid_for_control


def test_all_live_real_fresh_inputs_are_labelled_live_and_valid_for_control():
    """Logic only: this is a declaration made by the test, not a live measurement."""
    result = build_observation(inputs(origin=ob.SOURCE_LIVE), NOW)
    assert result.source == "LIVE" and result.valid_for_control is True and result.blocked_by == ()
    assert result.ages_s()["imu"] == pytest.approx(0.02) and result.ages_s()["own_pose"] == pytest.approx(0.05)
    assert result.ages_s()["previous_action"] is None
    assert result.oldest_input_t() == pytest.approx(NOW - 0.05)


@pytest.mark.parametrize("origin,expected", [(ob.SOURCE_SIMULATED, "SIMULATED"), (ob.SOURCE_REPLAY, "REPLAY")])
def test_simulated_and_replayed_inputs_give_a_vector_that_is_diagnostic_only(origin, expected):
    result = build_observation(inputs(origin=origin), NOW)
    assert result.values is not None and result.source == expected                      # DIAGNOSTIC inference is possible ...
    assert result.valid_for_control is False                                             # ... but never valid for control
    assert any(expected in line and "diagnostic only" in line for line in result.blocked_by)


def test_the_default_origin_is_simulated_so_nothing_is_live_by_accident():
    default = ObservationInputs(own=inputs().own, opponent=inputs().opponent, imu=inputs().imu)
    result = build_observation(default, NOW)
    assert default.origin == "SIMULATED" and result.source == "SIMULATED" and not result.valid_for_control


def test_one_non_live_input_makes_the_whole_vector_non_live():
    base = inputs(origin=ob.SOURCE_LIVE).__dict__
    replayed_imu = ObservationInputs(**{**base, "imu": ImuInput(GYRO, ACCEL, NOW - 0.02, ob.SOURCE_REPLAY, True)})
    assert build_observation(replayed_imu, NOW).source == "REPLAY" and not build_observation(replayed_imu, NOW).valid_for_control
    simulated_camera = ObservationInputs(**{**base, "own": OwnPoseInput(*OWN, NOW - 0.05, "measured", origin=ob.SOURCE_SIMULATED)})
    assert build_observation(simulated_camera, NOW).source == "SIMULATED"
    unknown_label = ObservationInputs(**{**base, "imu": ImuInput(GYRO, ACCEL, NOW - 0.02, "REAL", True)})
    assert build_observation(unknown_label, NOW).source == "SIMULATED"                    # an unrecognised label is never trusted
    assert ob.combine_origins(ob.SOURCE_LIVE, ob.SOURCE_REPLAY, ob.SOURCE_SIMULATED) == "SIMULATED"
    assert ob.combine_origins(None, ob.ORIGIN_CONSTANT) == "SIMULATED" and ob.combine_origins("LIVE", None) == "LIVE"


@pytest.mark.parametrize("breakage", ["no_imu", "imu_sim", "imu_unverified", "imu_stale", "imu_nan", "extra_blocker"])
def test_never_valid_for_control_without_a_real_imu(breakage):
    base = inputs(origin=ob.SOURCE_LIVE).__dict__
    if breakage == "no_imu":
        broken = ObservationInputs(**{**base, "imu": None})
    elif breakage == "imu_sim":
        broken = ObservationInputs(**{**base, "imu": ImuInput(GYRO, ACCEL, NOW - 0.02, ob.SOURCE_SIMULATED, True)})
    elif breakage == "imu_unverified":
        broken = ObservationInputs(**{**base, "imu": ImuInput(GYRO, ACCEL, NOW - 0.02, None, False)})
    elif breakage == "imu_stale":
        broken = ObservationInputs(**{**base, "imu": ImuInput(GYRO, ACCEL, NOW - 0.5, None, True)})
    elif breakage == "imu_nan":
        broken = ObservationInputs(**{**base, "imu": ImuInput((math.nan, 0.0, 0.0), ACCEL, NOW - 0.02, None, True)})
    else:
        broken = ObservationInputs(**{**base, "extra_blocked_by": ("arena layout is not the simulator's",)})
    result = build_observation(broken, NOW)
    assert result.valid_for_control is False and result.blocked_by
    if breakage == "extra_blocker":
        assert result.values is not None and "arena layout" in result.blocked_by[-1]          # diagnostic values, but not for control
    elif breakage == "imu_sim":
        assert result.values is not None and result.source == "SIMULATED"                      # a simulated IMU: DIAGNOSTIC vector only
        assert any("no real IMU" in line for line in result.blocked_by)
    else:
        assert result.values is None


def test_there_is_no_input_that_makes_a_constant_vector_valid():
    """The only constants are the 7 previous-action slots: a caller cannot supply an IMU by supplying only constants."""
    only_pose = ObservationInputs(origin=ob.SOURCE_LIVE, own=inputs().own, opponent=inputs().opponent)
    result = build_observation(only_pose, NOW)
    assert result.values is None and not result.valid_for_control and result.groups["previous_action"].ok


def test_the_report_is_json_and_carries_source_validity_blockers_and_ages():
    result = build_observation(inputs(imu=False), NOW)
    report = json.loads(json.dumps(result.to_report()))
    assert report["source"] == "LIVE" and report["valid_for_control"] is False and report["complete"] is False
    assert report["blocked_by"] and report["missing"] == ["imu"] and set(report["ages_ms"]) == {"imu", "previous_action", "own_pose", "opponent_position"}
    assert report["observation"] is None and report["groups"]["imu"]["status"] == "MISSING"
    full = json.loads(json.dumps(build_observation(inputs(), NOW).to_report()))
    assert len(full["observation"]) == 19 and full["ages_ms"]["imu"] == pytest.approx(20.0, abs=0.1)


def test_capture_state_maps_to_a_source_and_only_live_is_live():
    assert [ob.origin_from_capture_state(s) for s in ("LIVE", "live", "REPLAY", "MOCK", "OPENING", "LOST", "UNAVAILABLE", None, "")] == \
        ["LIVE", "LIVE", "REPLAY", "SIMULATED", "SIMULATED", "SIMULATED", "SIMULATED", "SIMULATED", "SIMULATED"]
