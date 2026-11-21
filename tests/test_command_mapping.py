"""control_center/backend/command_mapping.py: tactic -> (throttle, steer). Evidence class: SOFTWARE / SIMULATED.

Nothing here talks to a robot. The firmware mixer is checked against the SKETCH TEXT (the constants and the mixer lines are
read from ELRS-Handoff-WiFi.ino and ELRS-Handoff-Bench.ino) and then simulated in Python; the turn directions are additionally
checked by running the resulting wheel drive on the MuJoCo simulator. A passing run says the software is self-consistent with
the sketch text and the simulator; it does NOT say the real wheels are wired that way (that needs the supervised motion check).
"""

import itertools
import json
import math
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from control_center.backend import command_mapping as cm  # noqa: E402
from control_center.backend.command_mapping import (LIMITS, CommandPlan, Limits, MapperState, describe_mapping,  # noqa: E402
                                                    firmware_drive, map_tactic)

WIFI_INO = ROOT / "ELRS-Handoff-WiFi" / "ELRS-Handoff-WiFi.ino"
BENCH_INO = ROOT / "ELRS-Handoff-Bench" / "ELRS-Handoff-Bench.ino"
TACTICS = ("stop", "advance", "reverse", "turn_left", "turn_right")
TRACK_M = 0.132                      # config/sim.yaml robots.autobot.wheel_spacing_m (only the SIGN of the yaw rate matters here)


def plan(tactic, limits=LIMITS, t=10.0, age=0.0, state=None):
    return map_tactic(tactic, t, t - age, limits, state)


# ---------------------------------------------------------------- conventions pinned, and tied to the sketch text


def test_sign_convention_constants_are_pinned():
    """If anyone flips a sign or swaps the motors this fails: the convention is never changed silently."""
    assert (cm.THROTTLE_SIGN, cm.STEER_SIGN, cm.LEFT_RIGHT_SWAPPED, cm.SIM_FORWARD_SIGN) == (+1, +1, False, -1.0)
    assert (cm.FIRMWARE_FULL_SCALE, cm.FIRMWARE_PC_LIMIT, cm.DEFAULT_MAX_COMMAND, cm.DEFAULT_MAX_AGE_S) == (255, 128, 60, 0.25)
    assert cm.POLICY_TACTICS == TACTICS


@pytest.mark.parametrize("sketch", [WIFI_INO, BENCH_INO], ids=lambda p: p.parent.name)
def test_constants_and_mixer_match_the_sketch_text(sketch):
    text = sketch.read_text(encoding="utf-8", errors="replace")
    assert int(re.search(r"#define\s+PC_COMMAND_LIMIT\s+(\d+)", text).group(1)) == cm.FIRMWARE_PC_LIMIT
    assert re.search(r"steer:\s+-255\.\.255, positive = right", text), "the sketch no longer says positive steer = right"
    assert re.search(r"throttle:\s+-255\.\.255, positive = forward", text)
    body = text[text.index("void driveMixed"):]
    body = body[:body.index("setRightMotor(rightSpeed)")]
    assert re.search(r"leftSpeed\s*=\s*throttle\s*\+\s*steering", re.sub(r"\s+", " ", body)), "left is no longer throttle + steer"
    assert re.search(r"rightSpeed\s*=\s*throttle\s*-\s*steering", re.sub(r"\s+", " ", body)), "right is no longer throttle - steer"
    assert "maxMagnitude > 255" in body and "* 255) / maxMagnitude" in body
    # the PC path clamps to PC_COMMAND_LIMIT and applies no deadband (the deadzone is the radio sticks' normalizeChannel only)
    clamp = "constrain(throttle, -PC_COMMAND_LIMIT, PC_COMMAND_LIMIT)"
    accept = text[text.index(clamp):]
    accept = accept[:accept.index("lastPcCommand = now;")]
    assert "DEADZONE" not in accept and "constrain(steer, -PC_COMMAND_LIMIT, PC_COMMAND_LIMIT)" in accept
    # forward on the left motor is AIN1 PWM, on the right motor BIN2 PWM: positive = forward for both (so no motor swap in software)
    assert re.search(r"LEFT FORWARD\s*\n\s*analogWrite\(AIN1, speed\)", text) and re.search(r"RIGHT FORWARD\s*\n\s*analogWrite\(BIN1, 0\);\s*\n\s*analogWrite\(BIN2, speed\)", text)


def test_default_cap_equals_the_wifi_senders_default():
    wifi_link = pytest.importorskip("control_center.backend.wifi_link", reason="wifi_link is being written concurrently")
    if not hasattr(wifi_link, "DEFAULT_MAX_COMMAND"):
        pytest.skip("wifi_link has no DEFAULT_MAX_COMMAND (yet)")
    assert wifi_link.DEFAULT_MAX_COMMAND == cm.DEFAULT_MAX_COMMAND == LIMITS.max_command


# ---------------------------------------------------------------- the firmware mixer in Python


def test_firmware_mixer_reproduces_drivemixed_including_c_truncation():
    assert firmware_drive(60, 0) == (60, 60) and firmware_drive(0, -42) == (-42, 42) and firmware_drive(0, 42) == (42, -42)
    assert firmware_drive(-39, 0) == (-39, -39)
    assert firmware_drive(500, 0) == (128, 128) and firmware_drive(0, -500) == (-128, 128)
    # 128 + 128 = 256 > 255: rescaled by 255/256 with C truncation toward zero: 255, 0
    assert firmware_drive(128, 128) == (255, 0) and firmware_drive(128, -128) == (0, 255) and firmware_drive(-128, -128) == (-255, 0)
    # one more truncation example: (100, 128) -> left 228 right -28, fine; (128, 100) -> 228, 28
    assert firmware_drive(128, 100) == (228, 28)
    for throttle, steer in itertools.product(range(-128, 129, 8), repeat=2):                  # agrees with sim_compare.pc_mix to 1 count
        from control_center.backend.sim_compare import pc_mix
        left, right = firmware_drive(throttle, steer)
        ref = pc_mix(throttle, steer)
        assert abs(left - ref[0]) <= 1 and abs(right - ref[1]) <= 1


# ---------------------------------------------------------------- the table: every action


EXPECTED_DEFAULT = {"stop": (0, 0), "advance": (60, 0), "reverse": (-39, 0), "turn_left": (0, -42), "turn_right": (0, 42)}


@pytest.mark.parametrize("tactic", TACTICS)
def test_every_action_at_the_default_limits(tactic):
    result = plan(tactic)
    assert isinstance(result, CommandPlan)
    assert (result.intended_throttle, result.intended_steer) == EXPECTED_DEFAULT[tactic]
    assert (result.throttle, result.steer) == EXPECTED_DEFAULT[tactic] and result.clamped is False
    assert result.tactic == tactic and result.reason
    assert abs(result.throttle) <= LIMITS.max_command and abs(result.steer) <= LIMITS.max_command


@pytest.mark.parametrize("index", range(5))
def test_the_policy_action_index_maps_like_its_tactic(index):
    assert plan(index) == plan(TACTICS[index])
    assert (plan(index).throttle, plan(index).steer) == EXPECTED_DEFAULT[TACTICS[index]]


def test_the_forty_percent_motion_check_scale_is_reproduced_and_is_not_the_default():
    """40 % is the SUPERVISED motion-check scale (artifacts/motion-check/sim-reference.json), not an operating limit."""
    reference = json.loads((ROOT / "artifacts" / "motion-check" / "sim-reference.json").read_text(encoding="utf-8"))
    assert reference["plan"]["scale"] == 0.4 and reference["evidence"] == "SIM"
    motion_check_scale = Limits(max_command=128, operating_scale=0.4)
    seen = {}
    for step in reference["plan"]["steps"]:
        result = plan(step["tactic"], motion_check_scale)
        assert (result.throttle, result.steer) == (step["throttle"], step["steer"]), step["tactic"]
        seen[step["tactic"]] = (result.throttle, result.steer)
    assert seen == {"stop": (0, 0), "advance": (102, 0), "turn_left": (0, -71), "turn_right": (0, 71), "reverse": (-66, 0)}
    assert max(abs(v) for pair in seen.values() for v in pair) > LIMITS.max_command            # the default is far below it
    assert LIMITS.max_command <= 60


def test_the_table_agrees_with_motion_check_build_plan():
    sys.path.insert(0, str(ROOT / "scripts"))
    motion_check = pytest.importorskip("motion_check")
    built = motion_check.build_plan("stop:0.6,advance:0.2,stop:0.6,turn_left:0.2,stop:0.6,turn_right:0.2,stop:0.6,reverse:0.2,stop:0.6", 0.4)
    for step in built.steps:
        got = plan(step.tactic, Limits(max_command=128, operating_scale=0.4))
        assert (got.throttle, got.steer) == (step.throttle, step.steer)
        assert firmware_drive(got.throttle, got.steer) == (step.left, step.right)


# ---------------------------------------------------------------- left/right verified independently against the mixer


def _wheel_speeds(result):
    """Left/right wheel surface speeds from the firmware mixer, as fractions of full scale (positive = forward)."""
    left, right = firmware_drive(result.throttle, result.steer)
    return left / 255.0, right / 255.0


@pytest.mark.parametrize("tactic,expected_forward,expected_turn", [
    ("stop", 0, 0), ("advance", +1, 0), ("reverse", -1, 0), ("turn_left", 0, +1), ("turn_right", 0, -1)])
def test_mixer_simulation_turns_the_way_the_tactic_says(tactic, expected_forward, expected_turn):
    """Working: left motor = throttle + steer, right = throttle - steer (sketch). Positive = forward for each motor. A differential
    drive's body velocity is v = (vl + vr) / 2 and its yaw rate (counter-clockwise from above, the simulator's convention,
    autobot/sim/core.py:202) is w = (vr - vl) / track. turn_left must therefore have the RIGHT motor faster than the LEFT."""
    left, right = _wheel_speeds(plan(tactic))
    v, w = (left + right) / 2, (right - left) / TRACK_M
    assert (v > 1e-9) - (v < -1e-9) == expected_forward
    assert (w > 1e-9) - (w < -1e-9) == expected_turn
    if tactic in ("turn_left", "turn_right"):
        assert abs(v) < 1e-9 and (right > left) == (tactic == "turn_left")


@pytest.mark.parametrize("tactic", TACTICS)
def test_the_mapped_command_drives_the_simulator_the_way_the_tactic_says(tactic):
    """SIMULATED: feed the wheel drive the firmware mixer would produce (converted to the simulator's negative-forward wheel units)
    to the MuJoCo model for 0.4 s from rest and read the truth pose. turn_left must increase yaw (counter-clockwise)."""
    from autobot.sim.core import AutoBotSim

    left, right = _wheel_speeds(plan(tactic, Limits(max_command=128, operating_scale=0.4)))
    sim = AutoBotSim(str(ROOT / "config" / "sim.yaml"), 0, calibration_track=True)
    start = sim.truth(0)
    yaw0 = start["yaw_rad"]
    for _ in range(20):
        truth = sim.step((cm.SIM_FORWARD_SIGN * left, cm.SIM_FORWARD_SIGN * right), (0.0, 0.0))
    dx, dy = (truth["position_m"][:2] - start["position_m"][:2])
    forward = dx * math.cos(yaw0) + dy * math.sin(yaw0)
    turned = math.degrees((truth["yaw_rad"] - yaw0 + math.pi) % (2 * math.pi) - math.pi)
    if tactic == "stop":
        assert abs(forward) < 0.02 and abs(turned) < 1.0       # settling only: no drive
    elif tactic == "advance":
        assert forward > 0.02 and abs(turned) < 5.0
    elif tactic == "reverse":
        assert forward < -0.01 and abs(turned) < 5.0
    elif tactic == "turn_left":
        assert turned > 5.0 and abs(forward) < 0.02                      # counter-clockwise from above
    else:
        assert turned < -5.0 and abs(forward) < 0.02                     # clockwise from above


# ---------------------------------------------------------------- invalid input, staleness, caps


@pytest.mark.parametrize("bad", [None, "", "banana", "ADVANCE ", float("nan"), float("inf"), -1, 5, 2.5, True, [], {}, object(), b"advance", "forward_left"])
def test_invalid_unknown_none_nan_actions_stop_with_a_reason(bad):
    result = map_tactic(bad, 5.0, 5.0)
    if isinstance(bad, str) and bad.strip().lower() in TACTICS:           # "ADVANCE " is a valid name once normalised: a name, not a guess
        assert result.tactic == "advance"
        return
    assert (result.throttle, result.steer) == (0, 0) and result.reason.startswith(("no tactic", "unknown", "invalid"))
    assert (result.intended_throttle, result.intended_steer) == (0, 0) and result.tactic in ("none", "invalid")


def test_non_finite_or_missing_times_stop():
    for now, then in ((float("nan"), 1.0), (1.0, float("nan")), (float("inf"), 1.0), (None, 1.0), (1.0, None), ("1", 1.0)):
        out = map_tactic("advance", now, then)
        assert (out.throttle, out.steer) == (0, 0) and "finite" in out.reason


def test_a_stale_decision_is_zero_and_the_limit_is_exact():
    assert plan("advance", age=0.25).throttle == 60                                  # exactly at the limit: still fresh
    stale = plan("advance", age=0.2501)
    assert (stale.throttle, stale.steer) == (0, 0) and "stale" in stale.reason and stale.intended_throttle == 60
    assert plan("turn_left", age=1.0).steer == 0
    assert plan("advance", limits=Limits(max_age_s=0.1), age=0.11).throttle == 0
    future = map_tactic("advance", 10.0, 10.2)
    assert (future.throttle, future.steer) == (0, 0) and "future" in future.reason


def test_the_hard_cap_holds_for_any_input_and_any_limits():
    """No tactic, scale, cap, age or call order can put more than max_command on either axis."""
    scales = (0.05, 0.235, 0.4, 0.5, 0.75, 1.0)
    caps = (1, 10, 39, 60, 102, 127, 128)
    inputs = list(TACTICS) + [0, 1, 2, 3, 4, None, float("nan"), "x", 99]
    for cap, scale in itertools.product(caps, scales):
        limits = Limits(max_command=cap, operating_scale=scale)
        state = MapperState()
        for i, tactic in enumerate(inputs * 3):
            for st in (None, state):
                out = map_tactic(tactic, 100.0 + 0.02 * i, 100.0 + 0.02 * i, limits, st)
                assert abs(out.throttle) <= cap and abs(out.steer) <= cap, (cap, scale, tactic, out)
                assert isinstance(out.throttle, int) and isinstance(out.steer, int)
                assert max(abs(v) for v in firmware_drive(out.throttle, out.steer)) <= 255


def test_clamped_is_set_when_the_cap_binds_and_the_command_is_still_in_shape():
    out = plan("advance", Limits(max_command=60, operating_scale=0.4))
    assert (out.intended_throttle, out.throttle, out.clamped) == (102, 60, True) and "clamped" in out.reason
    out = plan("turn_left", Limits(max_command=60, operating_scale=0.4))
    assert (out.intended_steer, out.steer, out.clamped) == (-71, -60, True)
    assert plan("advance").clamped is False


def test_limits_are_validated():
    for kwargs in ({"max_command": 0}, {"max_command": 129}, {"max_command": True}, {"max_command": float("nan")},
                   {"operating_scale": 0.0}, {"operating_scale": 1.01}, {"operating_scale": float("nan")}, {"max_age_s": 0.0},
                   {"slew_full_scale_s": -1.0}, {"reversal_neutral_s": -0.1}):
        with pytest.raises(ValueError):
            Limits(**kwargs)
    assert Limits(max_command=128).max_command == 128
    assert LIMITS.max_command <= 60 and LIMITS.max_age_s == 0.25


# ---------------------------------------------------------------- rapid changes: slew-limited, neutral-first on reversal


def run(sequence, dt=0.02, limits=LIMITS, start=100.0):
    state, out, t = MapperState(), [], start
    for tactic in sequence:
        out.append(map_tactic(tactic, t, t, limits, state))
        t += dt
    return out


def test_slew_limits_an_increase_from_rest_and_never_delays_a_stop():
    steps = run(["advance"] * 10 + ["stop"] * 2)
    throttles = [s.throttle for s in steps]
    assert throttles[0] == 8 and all(b - a <= 8 for a, b in zip(throttles, throttles[1:10])) and max(throttles) == 60
    assert throttles[:3] == [8, 16, 24] and "slew" in steps[0].reason and steps[-1].reason == "tactic stop"
    assert throttles[10] == 0                                         # the stop is immediate
    # full speed is reached no sooner than slew_full_scale_s from rest
    reached = next(i for i, v in enumerate(throttles) if v == 60)
    assert reached * 0.02 >= LIMITS.slew_full_scale_s - 0.03
    # decreasing magnitude is immediate: advance at 60 -> turn_left drops the throttle at once
    mixed = run(["advance"] * 10 + ["turn_left"])
    assert mixed[-1].throttle == 0 and mixed[-1].steer != 0 and abs(mixed[-1].steer) < 42


def test_a_direction_reversal_goes_through_neutral_first_and_holds():
    for first, second in (("advance", "reverse"), ("reverse", "advance"), ("turn_left", "turn_right"), ("turn_right", "turn_left")):
        steps = run([first] * 12 + [second] * 12)
        after = steps[12:]
        assert (after[0].throttle, after[0].steer) == (0, 0) and "neutral-first" in after[0].reason
        held = [s for s in after if "neutral-first" in s.reason]
        assert 4 <= len(held) <= 6                                    # 100 ms of 20 ms ticks
        assert all((s.throttle, s.steer) == (0, 0) for s in held)
        # it then ramps up on the new sign, from rest, and never passes straight through
        moving = [s for s in after if (s.throttle, s.steer) != (0, 0)]
        assert moving
        before = (steps[11].throttle, steps[11].steer)
        assert before != (0, 0) and abs(before[0]) + abs(before[1]) == max(abs(before[0]), abs(before[1]))
        wanted = EXPECTED_DEFAULT[second]
        for s in moving:                                              # every later command has the NEW sign and never the old one
            assert all(v == 0 or (v > 0) == (w > 0) for v, w in ((s.throttle, wanted[0]), (s.steer, wanted[1])) if w != 0)
        # between two consecutive commands the change never exceeds the neutral-first step (no sign flip in one tick)
        for a, b in zip(steps, steps[1:]):
            assert a.throttle * b.throttle >= 0 and a.steer * b.steer >= 0


def test_dithering_between_opposite_tactics_produces_mostly_stop_not_oscillation():
    steps = run(["turn_left", "turn_right"] * 30)
    sent = [s.steer for s in steps]
    assert max(sent) <= 8 and min(sent) >= -8                        # never builds up: at most the first slew step
    assert sum(1 for s in steps if (s.throttle, s.steer) == (0, 0)) >= 40


def test_after_a_stale_decision_the_next_command_ramps_from_rest():
    state = MapperState()
    for i in range(10):
        map_tactic("advance", 100.0 + 0.02 * i, 100.0 + 0.02 * i, LIMITS, state)
    assert state.last_throttle == 60
    stale = map_tactic("advance", 101.0, 100.0, LIMITS, state)
    assert (stale.throttle, stale.steer) == (0, 0)
    again = map_tactic("advance", 101.02, 101.02, LIMITS, state)
    assert 0 < again.throttle <= 8                                    # not 60: the state was reset by the stop


def test_clock_going_backwards_stops():
    state = MapperState()
    map_tactic("advance", 100.0, 100.0, LIMITS, state)
    out = map_tactic("advance", 99.0, 99.0, LIMITS, state)
    assert (out.throttle, out.steer) == (0, 0) and "backwards" in out.reason


def test_stateless_call_applies_neither_slew_nor_neutral():
    assert plan("advance").throttle == 60 and plan("reverse").throttle == -39


def test_slew_can_be_disabled_but_the_cap_still_holds():
    limits = Limits(slew_full_scale_s=0.0, reversal_neutral_s=0.0)
    steps = run(["advance", "reverse", "turn_left"], limits=limits)
    assert [(s.throttle, s.steer) for s in steps] == [(60, 0), (-39, 0), (0, -42)]


# ---------------------------------------------------------------- what the UI shows


def test_describe_mapping_is_a_table_the_ui_can_show():
    rows = describe_mapping()
    assert [r["tactic"] for r in rows] == list(TACTICS)
    by = {r["tactic"]: r for r in rows}
    assert (by["advance"]["throttle"], by["advance"]["left"], by["advance"]["right"]) == (60, 60, 60)
    assert (by["turn_left"]["steer"], by["turn_left"]["left"], by["turn_left"]["right"]) == (-42, -42, 42)
    assert (by["turn_right"]["steer"], by["turn_right"]["left"], by["turn_right"]["right"]) == (42, 42, -42)
    assert (by["reverse"]["throttle"], by["reverse"]["left"], by["reverse"]["right"]) == (-39, -39, -39)
    assert by["stop"]["left"] == by["stop"]["right"] == 0 and all(r["meaning"] and r["sim_wheels"] for r in rows)
    wide = {r["tactic"]: r for r in describe_mapping(Limits(max_command=128, operating_scale=0.4))}
    assert wide["advance"]["throttle"] == 102 and wide["turn_left"]["steer"] == -71
    clipped = {r["tactic"]: r for r in describe_mapping(Limits(max_command=60, operating_scale=0.4))}
    assert clipped["advance"]["clamped"] is True
    text = cm.format_mapping()
    assert "turn_left" in text and "NOT verified on hardware" in text
