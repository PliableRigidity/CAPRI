"""THE runtime mapping from a policy tactic to the (throttle, steer) the ESP32 firmware understands. Pure: nothing is sent.

Firmware facts this rests on (ELRS-Handoff-WiFi/ELRS-Handoff-WiFi.ino and ELRS-Handoff-Bench/ELRS-Handoff-Bench.ino):

* PC command ``D,<token>,<throttle>,<steer>[,<seq>]``, each -255..255, positive throttle = forward, positive steer = RIGHT
  (WiFi.ino:831-835). The firmware clamps both to ``PC_COMMAND_LIMIT`` = 128 (WiFi.ino:139, :940-941).
* Mixer ``driveMixed`` (WiFi.ino:717-745): left = throttle + steer, right = throttle - steer; if the larger magnitude exceeds 255
  both are rescaled by 255/max with C integer division (truncation toward zero). Positive motor value = forward (WiFi.ino:376-437).
* The PC path has NO deadband (the DEADZONE of 45 belongs to the radio sticks, WiFi.ino ``normalizeChannel``); whether a small
  command actually moves the real chassis (stiction) is UNVERIFIED and can only be learned on the robot.

The tactic table is the simulator's own (``autobot.sim.env.tactic_to_drive``, profile ``experimental5_v1`` = the checkpoint's
action profile); the simulator's wheel command is negative-forward (env.py:52), so ``firmware = SIM_FORWARD_SIGN * sim``. Both
``throttle = (L+R)/2`` and ``steer = (L-R)/2`` follow from the mixer above and match scripts/motion_check.build_plan.

The motion-check artifact (artifacts/motion-check/sim-reference.json) is at a 40 % scale (advance +102). That is a SUPERVISED
motion-check scale, not an operating limit. This module is parameterised by ``Limits``; the default is deliberately small.

API (tiny on purpose):
    map_tactic(tactic, now, decision_time, limits=LIMITS, state=None) -> CommandPlan
    LIMITS / Limits                       the limits dataclass and its default instance
    describe_mapping(limits=LIMITS)       rows for the UI: tactic, throttle, steer, left/right drive, simulator wheels
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

POLICY_ACTION_PROFILE = "experimental5_v1"        # the checkpoint's action profile (runs/recovery-challenger-v2-seed1/effective_config.json:15)
POLICY_TACTICS = ("stop", "advance", "reverse", "turn_left", "turn_right")

# ---- conventions: stated once, asserted by tests/test_command_mapping.py, never flipped silently ----------------------
THROTTLE_SIGN = +1                  # positive throttle = forward                     (WiFi.ino:834)
STEER_SIGN = +1                     # positive steer = turn RIGHT (clockwise from above) (WiFi.ino:835)
LEFT_RIGHT_SWAPPED = False          # left = throttle + steer drives the LEFT motor    (WiFi.ino:719-741)
SIM_FORWARD_SIGN = -1.0             # simulator wheel command: negative = forward      (autobot/sim/env.py:52; sim_compare.py:50)

FIRMWARE_FULL_SCALE = 255           # WiFi.ino:384 (motor value), :731 (mixer rescale)
FIRMWARE_PC_LIMIT = 128             # PC_COMMAND_LIMIT, WiFi.ino:139
DEFAULT_MAX_COMMAND = 60            # == wifi_link.DEFAULT_MAX_COMMAND (wifi_link.py:69), ~23.5 % of 255
DEFAULT_MAX_AGE_S = 0.25            # == the firmware's PC_TIMEOUT (WiFi.ino:134) and wifi_link.DEFAULT_MAX_OBSERVATION_AGE_S
DEFAULT_SLEW_FULL_SCALE_S = 0.15    # an increase in command magnitude takes at least this long from 0 to max_command
DEFAULT_REVERSAL_NEUTRAL_S = 0.10   # 0,0 held for this long when a command flips sign on either axis
NOMINAL_TICK_S = 0.02               # the sender's period (wifi_link.MIN_SEND_PERIOD_S); the assumed dt of a first call
FUTURE_TOLERANCE_S = 0.05


@dataclass(frozen=True)
class Limits:
    """Operating limits.

    ``max_command``      hard cap on |throttle| and |steer| (of 255), <= the firmware's own 128. Always enforced last.
    ``operating_scale``  fraction of the simulator's full-scale tactic command that the table is scaled to BEFORE the cap
                         (1.0 = the tactic's native wheel command is 255). Default = max_command / 255, so the table peaks at
                         exactly max_command and nothing needs clamping; a supervised motion check at 0.4 is
                         ``Limits(max_command=128, operating_scale=0.4)``.
    ``max_age_s``        a decision older than this is not acted on (0,0).
    ``slew_full_scale_s`` minimum time for |command| to grow from 0 to max_command (0 disables slewing). Decreases and stops
                         are never delayed.
    ``reversal_neutral_s`` 0,0 held when throttle or steer flips sign (forward<->reverse, left<->right).

    How the defaults were chosen: 60 is the Wi-Fi sender's own default cap (wifi_link.py:69), so the mapping never produces
    something the sender would rescale; the firmware limit is 128 and the 40 % motion-check scale is 102. At 60 every tactic is
    single-axis, so capping never changes a tactic's shape. 60/255 is a starting point only: the stiction threshold of the real
    drivetrain is UNMEASURED, and the user must raise ``max_command`` only after supervised tests.
    """

    max_command: int = DEFAULT_MAX_COMMAND
    operating_scale: float = DEFAULT_MAX_COMMAND / FIRMWARE_FULL_SCALE
    max_age_s: float = DEFAULT_MAX_AGE_S
    slew_full_scale_s: float = DEFAULT_SLEW_FULL_SCALE_S
    reversal_neutral_s: float = DEFAULT_REVERSAL_NEUTRAL_S

    def __post_init__(self) -> None:
        def number(name: str, value: Any, low: float, high: float, *, low_open: bool = False) -> None:
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) \
                    or value < low or value > high or (low_open and value == low):
                raise ValueError(f"Limits.{name} must be a finite number in {'(' if low_open else '['}{low}, {high}], got {value!r}")
        number("max_command", self.max_command, 1, FIRMWARE_PC_LIMIT)
        number("operating_scale", self.operating_scale, 0.0, 1.0, low_open=True)
        number("max_age_s", self.max_age_s, 0.0, 5.0, low_open=True)
        number("slew_full_scale_s", self.slew_full_scale_s, 0.0, 10.0)
        number("reversal_neutral_s", self.reversal_neutral_s, 0.0, 5.0)


LIMITS = Limits()


@dataclass(frozen=True)
class CommandPlan:
    tactic: str                    # the requested tactic, or "none" / "invalid"
    intended_throttle: int         # the table at operating_scale, before cap, slew and neutral-first
    intended_steer: int
    throttle: int                  # what may go on the wire: always within +/- max_command
    steer: int
    clamped: bool                  # the hard cap (max_command) reduced the intended command
    reason: str

    @property
    def is_stop(self) -> bool:
        return self.throttle == 0 and self.steer == 0


@dataclass
class MapperState:
    """Memory of the last output, for slew limiting and neutral-first. One per command path; pass the same object each call."""
    last_throttle: int = 0
    last_steer: int = 0
    last_now: float | None = None
    neutral_until: float = -math.inf


# ----------------------------------------------------------------------------------------------
# The firmware mixer (WiFi.ino acceptPcCommand + driveMixed), in Python


def _truncating_div(numerator: int, denominator: int) -> int:
    """C integer division: truncates toward zero."""
    quotient = abs(numerator) // abs(denominator)
    return quotient if (numerator >= 0) == (denominator > 0) else -quotient


def firmware_drive(throttle: int, steer: int) -> tuple[int, int]:
    """(left, right) motor values the sketch applies for a PC command, including its clamp to 128 and its 255 rescale."""
    throttle = max(-FIRMWARE_PC_LIMIT, min(FIRMWARE_PC_LIMIT, int(throttle)))
    steer = max(-FIRMWARE_PC_LIMIT, min(FIRMWARE_PC_LIMIT, int(steer)))
    left, right = throttle + steer, throttle - steer
    biggest = max(abs(left), abs(right))
    if biggest > FIRMWARE_FULL_SCALE:
        left, right = _truncating_div(left * FIRMWARE_FULL_SCALE, biggest), _truncating_div(right * FIRMWARE_FULL_SCALE, biggest)
    return left, right


# ----------------------------------------------------------------------------------------------
# The table: simulator tactic -> firmware command


_TABLE_CACHE: dict[str, tuple[float, float]] | None = None


def _round_half_away(value: float) -> int:
    return int(math.floor(abs(value) + 0.5)) * (1 if value >= 0 else -1)


def _unit_table() -> dict[str, tuple[float, float]]:
    """tactic -> (throttle, steer) as FRACTIONS of full scale, from the simulator's own tactic table. Raises if it cannot be read."""
    global _TABLE_CACHE
    if _TABLE_CACHE is None:
        from autobot.sim.env import ACTION_PROFILES, tactic_to_drive

        tactics = ACTION_PROFILES[POLICY_ACTION_PROFILE]["tactics"]
        if tuple(tactics) != POLICY_TACTICS:
            raise RuntimeError(f"the simulator's {POLICY_ACTION_PROFILE} tactics are {tactics}, not {POLICY_TACTICS}")
        table: dict[str, tuple[float, float]] = {}
        for index, name in enumerate(tactics):
            sim_left, sim_right = tactic_to_drive(index, POLICY_ACTION_PROFILE)
            left, right = SIM_FORWARD_SIGN * sim_left, SIM_FORWARD_SIGN * sim_right          # firmware sense: + = forward
            if LEFT_RIGHT_SWAPPED:
                left, right = right, left
            table[name] = (THROTTLE_SIGN * (left + right) / 2.0, STEER_SIGN * (left - right) / 2.0)
        _TABLE_CACHE = table
    return _TABLE_CACHE


def intended_command(tactic: str, limits: Limits = LIMITS) -> tuple[int, int]:
    """Table value for a tactic at ``limits.operating_scale`` (before the cap). Raises KeyError for an unknown tactic."""
    throttle, steer = _unit_table()[tactic]
    return (_round_half_away(throttle * FIRMWARE_FULL_SCALE * limits.operating_scale),
            _round_half_away(steer * FIRMWARE_FULL_SCALE * limits.operating_scale))


def _cap(throttle: int, steer: int, limits: Limits) -> tuple[int, int]:
    """Scale both axes together so the larger is at most max_command (an arc would keep its shape), then clip as a last resort."""
    cap = min(limits.max_command, FIRMWARE_PC_LIMIT)
    biggest = max(abs(throttle), abs(steer))
    if biggest > cap:
        factor = cap / biggest
        throttle, steer = _round_half_away(throttle * factor), _round_half_away(steer * factor)
    return max(-cap, min(cap, throttle)), max(-cap, min(cap, steer))


def _sign(value: int) -> int:
    return (value > 0) - (value < 0)


def _plan(tactic: str, intended: tuple[int, int], out: tuple[int, int], clamped: bool, reason: str) -> CommandPlan:
    return CommandPlan(tactic, intended[0], intended[1], int(out[0]), int(out[1]), clamped, reason)


def _name(tactic: Any) -> tuple[str | None, str]:
    """(tactic name, "") or (None, why it is not one). Accepts a name or a policy action index; nothing else."""
    if tactic is None:
        return None, "no tactic: stop"
    if isinstance(tactic, bool):
        return None, "invalid tactic (a bool): stop"
    if isinstance(tactic, str):
        key = tactic.strip().lower()
        return (key, "") if key in POLICY_TACTICS else (None, f"unknown tactic {tactic[:40]!r}: stop")
    try:
        number = float(tactic)
    except (TypeError, ValueError):
        return None, f"invalid tactic of type {type(tactic).__name__}: stop"
    if not math.isfinite(number):
        return None, "invalid tactic (NaN or infinite): stop"
    if number != int(number) or not 0 <= int(number) < len(POLICY_TACTICS):
        return None, f"invalid action index {number!r}: stop"
    return POLICY_TACTICS[int(number)], ""


def map_tactic(tactic: Any, now: float, decision_time: float, limits: Limits = LIMITS, state: MapperState | None = None) -> CommandPlan:
    """Tactic (name or action index) at ``decision_time``, acted on at ``now`` -> the throttle/steer to send.

    Rules, in order. Any failure is a stop (0,0) with the reason; nothing is ever guessed or carried over.
      1. ``now`` / ``decision_time`` not finite numbers, or the decision from the future -> stop.
      2. Stale: ``now - decision_time > limits.max_age_s`` -> stop. Pass the age of the OLDEST input
         (``PolicyDecision.input_time``) to make this the observation's age.
      3. None / unknown / NaN / non-integer action -> stop.
      4. Table value at ``operating_scale``; the hard cap ``max_command`` scales both axes together (``clamped``).
      5. With a ``state`` (the same object every call):
           * neutral-first: if either axis flips sign relative to the last non-zero output, output 0,0 for
             ``reversal_neutral_s``. Reason: a sudden reversal of a geared motor at speed is a current spike and, on this
             chassis, a violent jerk; dithering between opposite tactics therefore produces mostly stop, not oscillation.
           * slew: an increase in |throttle| or |steer| is limited to ``max_command / slew_full_scale_s`` per second
             (assumed dt = one 20 ms tick on the first call, and at most ``slew_full_scale_s`` after a long gap).
             Decreases and stop are immediate (braking is never delayed).
         Without a ``state`` the call is stateless and neither rule applies.
      6. The hard cap is applied again to the final value: no input can exceed ``max_command``.
    """
    intended = (0, 0)
    if not all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in (now, decision_time)):
        _reset(state)
        return _plan("invalid", intended, (0, 0), False, "stop: the time of the decision or of now is not a finite number")
    name, problem = _name(tactic)
    label = "none" if tactic is None else name or "invalid"
    if name is None:
        _reset(state, now)
        return _plan(label, intended, (0, 0), False, problem)
    try:
        intended = intended_command(name, limits)
    except Exception as error:                                          # the simulator's table cannot be read: never guess it
        _reset(state, now)
        return _plan(label, (0, 0), (0, 0), False, f"stop: the simulator tactic table is unavailable ({type(error).__name__})")
    age = now - decision_time
    if age > limits.max_age_s:
        _reset(state, now)
        return _plan(label, intended, (0, 0), False, f"stop: stale decision ({1000 * age:.0f} ms old, limit {1000 * limits.max_age_s:.0f} ms)")
    if age < -FUTURE_TOLERANCE_S:
        _reset(state, now)
        return _plan(label, intended, (0, 0), False, f"stop: the decision is {-1000 * age:.0f} ms in the future (clock fault)")

    target = _cap(*intended, limits)
    clamped = target != intended
    reason = "tactic stop" if name == "stop" else ("ok" if not clamped else f"clamped to max_command {limits.max_command}")
    out = target

    if state is not None:
        dt = NOMINAL_TICK_S if state.last_now is None else now - state.last_now
        if dt < 0:
            _reset(state, now)
            return _plan(label, intended, (0, 0), clamped, "stop: the clock went backwards")
        last = (state.last_throttle, state.last_steer)
        in_hold = now < state.neutral_until
        reversal = any(_sign(last[i]) * _sign(target[i]) == -1 for i in (0, 1))
        if reversal and not in_hold and limits.reversal_neutral_s > 0:
            state.neutral_until = now + limits.reversal_neutral_s
            in_hold = True
        if in_hold and target != (0, 0):
            out, reason = (0, 0), f"neutral-first: direction reversal, holding 0,0 for {1000 * limits.reversal_neutral_s:.0f} ms"
        elif limits.slew_full_scale_s > 0:
            step = limits.max_command / limits.slew_full_scale_s * min(dt, limits.slew_full_scale_s)
            slewed = []
            for current, wanted in zip(last, target):
                if _sign(wanted) == _sign(current) and abs(wanted) > abs(current):
                    slewed.append(_sign(wanted) * min(abs(wanted), _round_half_away(abs(current) + step)))
                elif _sign(current) != _sign(wanted) and abs(wanted) > 0:      # from rest (or after a hold) toward a new sign
                    slewed.append(_sign(wanted) * min(abs(wanted), max(1, _round_half_away(step))))
                else:
                    slewed.append(wanted)                                       # a decrease, or a stop: immediate
            if tuple(slewed) != target:
                out, reason = (slewed[0], slewed[1]), f"slew-limited toward {target[0]},{target[1]}"
    out = _cap(*out, limits)
    if state is not None:
        state.last_throttle, state.last_steer, state.last_now = out[0], out[1], float(now)
    return _plan(label, intended, out, clamped, reason)


def _reset(state: MapperState | None, now: float | None = None) -> None:
    """A stop for any reason also zeroes the memory: the next command ramps up from rest."""
    if state is not None:
        state.last_throttle = state.last_steer = 0
        state.last_now = now if isinstance(now, (int, float)) and math.isfinite(now) else None
        state.neutral_until = -math.inf


# ----------------------------------------------------------------------------------------------
# For the UI


_MEANING = {"stop": "both motors braked", "advance": "drive straight forward", "reverse": "drive straight backward",
            "turn_left": "spin counter-clockwise (seen from above): right wheel forward, left wheel backward",
            "turn_right": "spin clockwise (seen from above): left wheel forward, right wheel backward"}


def describe_mapping(limits: Limits = LIMITS) -> list[dict[str, Any]]:
    """One row per tactic at ``limits``: what is intended, what is finally sent at steady state, and what the mixer makes of it."""
    rows = []
    for name in POLICY_TACTICS:
        try:
            intended = intended_command(name, limits)
        except Exception as error:
            rows.append({"tactic": name, "throttle": None, "steer": None, "left": None, "right": None, "sim_wheels": None,
                         "meaning": f"UNAVAILABLE: {type(error).__name__}", "clamped": False})
            continue
        throttle, steer = _cap(*intended, limits)
        left, right = firmware_drive(throttle, steer)
        rows.append({"tactic": name, "throttle": throttle, "steer": steer, "left": left, "right": right,
                     "sim_wheels": (SIM_FORWARD_SIGN * left / FIRMWARE_FULL_SCALE, SIM_FORWARD_SIGN * right / FIRMWARE_FULL_SCALE),
                     "meaning": _MEANING[name], "clamped": (throttle, steer) != intended})
    return rows


def format_mapping(limits: Limits = LIMITS) -> str:
    lines = [f"max_command {limits.max_command}/255 ({100 * limits.max_command / FIRMWARE_FULL_SCALE:.0f} %), operating_scale "
             f"{limits.operating_scale:.3f}, firmware limit {FIRMWARE_PC_LIMIT}, max age {1000 * limits.max_age_s:.0f} ms",
             "  tactic      throttle  steer   left  right   meaning"]
    for row in describe_mapping(limits):
        lines.append(f"  {row['tactic']:<10} {row['throttle']!s:>8} {row['steer']!s:>6} {row['left']!s:>6} {row['right']!s:>6}   {row['meaning']}")
    lines.append("  convention: positive steer = RIGHT (clockwise), left = throttle + steer, right = throttle - steer; "
                 "NOT verified on hardware.")
    return "\n".join(lines)
