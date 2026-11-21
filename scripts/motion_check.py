"""Sim-to-real movement check: one short, fixed tactic sequence in the simulator and on the robot.

The question it answers is narrow: does the same commanded sequence produce movement of the same
SIGN (forward, reverse, left, right) in the simulator and on the physical robot, and by roughly how
much do distance and rotation differ? It is not an autonomy test and it runs no policy.

    python scripts\\motion_check.py plan                          print the sequence and the commands (no hardware)
    python scripts\\motion_check.py sim                           write the simulator reference trace (no hardware)
    python scripts\\motion_check.py checks --port COM5 --power .. guided stop / disarm / takeover / link-loss checks, WHEELS RAISED
    python scripts\\motion_check.py run --port COM5 --surface raised --power ..    supervised physical run
    python scripts\\motion_check.py compare artifacts\\motion-check\\<run>         simulator against what you observed

What is reused, unchanged:
  * tactics and their wheel commands: `autobot.sim.env.tactic_to_drive` (the simulator's own table);
  * the PC command format and status parsing: `scripts/handoff_bench_client.py`
    (`D,<tok>,<throttle>,<steer>` at 50 Hz, `ST` lines, the port opener that does not pulse reset);
  * the firmware's mixer and the direction classes: `control_center.backend.sim_compare`.

Safety model (nothing here replaces the RadioMaster, which stays the only thing that can arm):
  * Starting this program moves nothing. It sends zero commands until the firmware itself reports
    armed, PC selected and PC commands accepted, and then waits for you to type GO.
  * A non-zero command is only ever sent between GO and the end of the sequence.
  * It stops and never resumes on: no status for 0.5 s, disarm, MODE away from PC, any latched fault,
    radio loss, a changed token (control changed hands), the firmware reporting motor outputs the
    sequence did not command, Enter, or Ctrl+C. Stopping means zero commands, then silence; the
    firmware's own 250 ms stale-command stop is the backstop.
  * Speeds are capped at the firmware's own PC limit and each move at 1 s.
  * A floor run is refused unless `checks` passed on this firmware within the last 12 hours.
  * It never arms, never selects PC mode, never clears a fault and never re-arms: those are switches
    on the RadioMaster.

Everything this file can show without a robot is a software result. Only a run you make is hardware.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import queue
import sys
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

ROOT = Path(__file__).resolve().parents[1]
for entry in (str(ROOT), str(ROOT / "scripts")):
    if entry not in sys.path:
        sys.path.insert(0, entry)

import handoff_bench_client as client  # noqa: E402  (the tested PC client: protocol, parsing, port)

from control_center.backend.sim_compare import (  # noqa: E402  (pure functions, no hardware, no Qt)
    CONTROLLER_SCALE, SIM_FORWARD_SIGN, TACTIC_CLASS, drive_class, pc_mix)

OUT_ROOT = ROOT / "artifacts" / "motion-check"
SAFETY_RECORD = OUT_ROOT / "safety-checks.json"
SIM_CONFIG = ROOT / "config" / "sim.yaml"

ACTION_PROFILE = "arc7_v1"          # the simulator's widest tactic table; the default sequence uses its first five
DEFAULT_SEQUENCE = "stop:1.0,advance:0.6,stop:1.0,turn_left:0.4,stop:1.0,turn_right:0.4,stop:1.0,reverse:0.6,stop:1.0"
DEFAULT_SCALE = 0.40                # fraction of the simulator's full-scale tactic command
FIRMWARE_PC_LIMIT = 128             # PC_COMMAND_LIMIT in the handoff sketches (of 255); re-read from HELLO at run time
MAX_SCALE = FIRMWARE_PC_LIMIT / CONTROLLER_SCALE
MAX_MOVE_S = 1.0                    # a moving step in a normal run
MAX_CHECK_MOVE_S = 2.5              # a moving step in a wheels-raised safety check (you interrupt it by hand)
MIN_STOP_S = 0.5
MAX_TOTAL_S = 30.0
TICK_S = client.SEND_PERIOD_S       # 20 ms, the tested client's send period

READY_HOLD_S = 0.3                  # readiness must have held this long before GO is accepted
MISMATCH_GRACE_S = 0.20             # firmware-reported L/R may lag a step change by this much
ABORT_ZERO_S = 0.5                  # zero commands sent after a stop before going silent
SETTLE_S = 0.8                      # how long to keep listening after a stop, for the evidence
HOLD_ZERO_S = 20.0                  # after a finished sequence: zeros until you disarm, at most this long
CUT_TIMEOUT_S = 1.0                 # link-loss check: the firmware must have stopped by then
CUT_PASS_S = 0.5                    # ... and to pass, within this (firmware PC timeout is 250 ms)
STOP_PASS_S = 0.30                  # stop check: reported outputs at zero this soon after the stop command
SAFETY_MAX_AGE_S = 12 * 3600.0
PROBE_PERIOD_S = 0.5

WAITING, READY, RUNNING, CUT, HOLDING, DONE, ABORTED = "WAITING", "READY", "RUNNING", "CUT", "HOLDING", "DONE", "ABORTED"

SURFACES = ("raised", "floor")
POWER_CHOICES = ("usb-only", "battery-and-usb", "battery-only")
REQUIRED_CHECKS = ("stop", "disarm", "takeover", "pc_link_loss", "radio_loss")
# Where the keyboard client's wording talks about keys, say what this tool is doing instead.
BLOCK_TEXT = {"WAIT_PC_NEUTRAL": "PC selected: waiting for the firmware to accept this program's zero command",
              "DISARMED": "ARM switch is off (arm with MODE on PC when you are ready)"}


class PlanError(ValueError):
    """The requested sequence is outside the limits of this tool."""


# ---------------------------------------------------------------------------------------------- plan


@dataclass(frozen=True)
class Step:
    index: int
    tactic: str
    duration_s: float
    throttle: int               # what is sent: D,<tok>,<throttle>,<steer>
    steer: int
    left: int                   # what the firmware's mixer makes of it (its ST line reports these)
    right: int
    sim_wheels: tuple[float, float]     # the same command in simulator units (negative = forward)

    @property
    def moving(self) -> bool:
        return self.throttle != 0 or self.steer != 0

    @property
    def command_class(self) -> str:
        return drive_class(self.left, self.right, forward_sign=1.0, scale=CONTROLLER_SCALE)


@dataclass(frozen=True)
class Plan:
    steps: tuple[Step, ...]
    scale: float
    sequence_text: str

    @property
    def total_s(self) -> float:
        return sum(step.duration_s for step in self.steps)

    @property
    def max_command(self) -> int:
        return max(max(abs(step.throttle), abs(step.steer), abs(step.left), abs(step.right)) for step in self.steps)

    def to_report(self) -> dict[str, Any]:
        return {"sequence": self.sequence_text, "scale": self.scale, "total_s": round(self.total_s, 3),
                "max_command_of_255": self.max_command,
                "steps": [dict(asdict(step), sim_wheels=list(step.sim_wheels), command_class=step.command_class)
                          for step in self.steps]}


def parse_sequence(text: str) -> list[tuple[str, float]]:
    items = []
    for part in text.split(","):
        name, separator, seconds = part.strip().partition(":")
        if not separator:
            raise PlanError(f"sequence item {part!r} is not tactic:seconds")
        try:
            items.append((name.strip().lower(), float(seconds)))
        except ValueError as error:
            raise PlanError(f"sequence item {part!r}: {error}") from None
    return items


def build_plan(sequence: str = DEFAULT_SEQUENCE, scale: float = DEFAULT_SCALE, *, max_move_s: float = MAX_MOVE_S) -> Plan:
    """Tactic names -> the exact integers sent to the firmware, using the simulator's own tactic table."""
    from autobot.sim.env import ACTION_PROFILES, tactic_to_drive

    tactics = ACTION_PROFILES[ACTION_PROFILE]["tactics"]
    if not (math.isfinite(scale) and 0.0 < scale <= MAX_SCALE + 1e-9):
        raise PlanError(f"scale must be above 0 and at most {MAX_SCALE:.3f} (the firmware clamps PC commands to "
                        f"{FIRMWARE_PC_LIMIT} of 255)")
    items = parse_sequence(sequence)
    if not items:
        raise PlanError("the sequence is empty")
    steps: list[Step] = []
    for index, (name, seconds) in enumerate(items):
        if name not in tactics:
            raise PlanError(f"unknown tactic {name!r}; the simulator defines {', '.join(tactics)}")
        ticks = round(seconds / TICK_S)
        if ticks < 1 or abs(ticks * TICK_S - seconds) > 1e-6:
            raise PlanError(f"{name}:{seconds}: durations must be a positive multiple of {TICK_S * 1000:.0f} ms")
        sim_left, sim_right = tactic_to_drive(tactics.index(name), ACTION_PROFILE)
        # Simulator wheel commands are negative-forward; the firmware's are positive-forward, 255 full scale.
        left = round(SIM_FORWARD_SIGN * sim_left * scale * CONTROLLER_SCALE)
        right = round(SIM_FORWARD_SIGN * sim_right * scale * CONTROLLER_SCALE)
        throttle, steer = round((left + right) / 2), round((left - right) / 2)
        left, right = (int(round(value)) for value in pc_mix(throttle, steer))      # what the sketch's mixer will apply
        if max(abs(throttle), abs(steer)) > FIRMWARE_PC_LIMIT:
            raise PlanError(f"{name} at scale {scale} needs {max(abs(throttle), abs(steer))} of 255, above the firmware limit")
        steps.append(Step(index, name, ticks * TICK_S, throttle, steer, left, right,
                          (SIM_FORWARD_SIGN * left / CONTROLLER_SCALE, SIM_FORWARD_SIGN * right / CONTROLLER_SCALE)))
    if steps[0].moving or steps[-1].moving:
        raise PlanError("the sequence must begin and end with stop")
    for step, following in zip(steps, steps[1:]):
        if step.moving and (following.moving or following.duration_s < MIN_STOP_S - 1e-9):
            raise PlanError(f"{step.tactic} (step {step.index}) must be followed by a stop of at least {MIN_STOP_S} s")
    for step in steps:
        if step.moving and step.duration_s > max_move_s + 1e-9:
            raise PlanError(f"{step.tactic}:{step.duration_s} is longer than the {max_move_s} s allowed for one move")
        if step.moving and step.command_class != TACTIC_CLASS[step.tactic]:
            raise PlanError(f"{step.tactic} maps to a {step.command_class} command: sign conventions disagree")
    plan = Plan(tuple(steps), float(scale), sequence)
    if plan.total_s > MAX_TOTAL_S:
        raise PlanError(f"the sequence lasts {plan.total_s:.1f} s, above the {MAX_TOTAL_S:.0f} s limit")
    return plan


def format_plan(plan: Plan) -> str:
    lines = [f"Sequence ({plan.total_s:.1f} s, scale {plan.scale:.2f} of the simulator's full tactic command, "
             f"largest command {plan.max_command} of 255):",
             "  #  tactic         seconds  sent D,<tok>,thr,steer   firmware L / R   simulator wheels (neg = forward)"]
    for step in plan.steps:
        lines.append(f"  {step.index}  {step.tactic:<13} {step.duration_s:>6.2f}   thr {step.throttle:>4}  steer {step.steer:>4}"
                     f"     {step.left:>4} / {step.right:<4}     ({step.sim_wheels[0]:+.3f}, {step.sim_wheels[1]:+.3f})")
    return "\n".join(lines)


# ---------------------------------------------------------------------------------------------- simulator reference


def wrap(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def motion_class(forward_m: float, heading_change_deg: float, min_m: float = 0.010, min_deg: float = 5.0,
                 spin_drift_m: float = 0.020, curve_deg: float = 10.0) -> str:
    """Direction of a measured or simulated movement, in the same words as `sim_compare.drive_class`.

    A spin on the spot drifts a little, and a straight run wanders a little: up to `spin_drift_m` of travel
    is still a turn, and up to `curve_deg` of heading change is still straight.
    """
    turning = abs(heading_change_deg) >= min_deg
    if turning and abs(forward_m) < spin_drift_m:
        return "turn_left" if heading_change_deg > 0 else "turn_right"
    if abs(forward_m) >= min_m:
        side = "" if abs(heading_change_deg) < curve_deg else ("_left" if heading_change_deg > 0 else "_right")
        return ("forward" if forward_m > 0 else "reverse") + side
    return "stop"


def sim_reference(plan: Plan, config: Path | str = SIM_CONFIG, seed: int = 0) -> dict[str, Any]:
    """Run the plan's exact commands on the simulator's unbounded calibration floor, from rest.

    Per step, in the frame of the pose at that step's start: forward travel (m, + = nose direction),
    sideways travel (m, + = left) and heading change (deg, + = left / counter-clockwise from above).
    """
    from autobot.sim.core import AutoBotSim

    sim = AutoBotSim(str(config), seed, calibration_track=True)
    hz = float(sim.cfg["simulation"]["control_hz"])
    if abs(1.0 / hz - TICK_S) > 1e-9:
        raise PlanError(f"the simulator's control rate is {hz} Hz but commands are sent every {TICK_S * 1000:.0f} ms")
    samples: list[dict[str, float]] = []
    steps: list[dict[str, Any]] = []
    truth = sim.truth(0)
    yaw = float(truth["yaw_rad"])
    t = 0.0
    for step in plan.steps:
        x0, y0, yaw0 = float(truth["position_m"][0]), float(truth["position_m"][1]), yaw
        path = 0.0
        for _ in range(round(step.duration_s * hz)):
            before = truth["position_m"][:2].copy()
            truth = sim.step(step.sim_wheels, (0.0, 0.0))
            reason = sim.instability()
            if reason:
                raise RuntimeError(f"simulator reference invalid in step {step.index} ({step.tactic}): {reason}")
            yaw += wrap(float(truth["yaw_rad"]) - wrap(yaw))
            path += float(math.hypot(*(truth["position_m"][:2] - before)))
            t += 1.0 / hz
            samples.append({"t_s": round(t, 4), "step": step.index, "x_m": round(float(truth["position_m"][0]), 5),
                            "y_m": round(float(truth["position_m"][1]), 5), "yaw_deg": round(math.degrees(yaw), 3)})
        dx, dy = float(truth["position_m"][0]) - x0, float(truth["position_m"][1]) - y0
        forward = dx * math.cos(yaw0) + dy * math.sin(yaw0)
        left = -dx * math.sin(yaw0) + dy * math.cos(yaw0)
        turned = math.degrees(yaw - yaw0)
        steps.append({"index": step.index, "tactic": step.tactic, "duration_s": step.duration_s,
                      "wheels": list(step.sim_wheels), "forward_m": round(forward, 4), "left_m": round(left, 4),
                      "path_m": round(path, 4), "heading_change_deg": round(turned, 2),
                      "motion_class": motion_class(forward, turned)})
    return {"evidence": "SIM", "simulator": "MuJoCo calibration floor (unbounded, no walls, no opponent), from rest",
            "config": str(Path(config)), "seed": seed, "control_hz": hz,
            "robot_model": sim.robot_model_key(0),
            "note": ("mass, drive gain and friction in this config are placeholders and 390 rpm is an estimate: the "
                     "signs are meaningful, the distances are a prediction to be corrected by the physical run"),
            "plan": plan.to_report(), "steps": steps, "samples": samples}


def format_sim(reference: dict[str, Any]) -> str:
    lines = ["Simulator reference (SIM, not a hardware result):",
             "  #  tactic         seconds   forward m   left m   heading deg   reads as"]
    for step in reference["steps"]:
        lines.append(f"  {step['index']}  {step['tactic']:<13} {step['duration_s']:>6.2f}   {step['forward_m']:>+8.3f}  "
                     f"{step['left_m']:>+7.3f}   {step['heading_change_deg']:>+9.1f}     {step['motion_class']}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------------------------- the runner


class Runner:
    """The sequence state machine. No serial port, no clock, no keyboard: `tick` is called every 20 ms.

    `cut=(step_index, after_s)` turns the run into the PC-link-loss check: at that point it stops sending
    altogether, on purpose, and watches for the firmware's own stale-command stop.
    """

    def __init__(self, plan: Plan, *, sequenced: bool = False, link_label: str = "USB",
                 cut: tuple[int, float] | None = None, unix: Callable[[], float] = time.time) -> None:
        self.plan, self.cut, self.unix = plan, cut, unix
        self.link_label = link_label
        self.core = client.ClientCore(sequenced=sequenced, link_label=link_label)   # used for receiving and parsing only
        self.sequenced = sequenced
        self.sequence = 0
        self.state = WAITING
        self.events: list[dict[str, Any]] = []
        self.step_records: list[dict[str, Any]] = []
        self.abort_code: str | None = None
        self.abort_text = ""
        self.cut_result: dict[str, Any] | None = None
        self.stop_latencies: list[dict[str, Any]] = []
        self.status_at_stop: client.Status | None = None
        self._ready_since: float | None = None
        self._probed_at = float("-inf")
        self._go_at = self._ended_at = 0.0
        self._go_token: int | None = None
        self._step = -1
        self._step_started = 0.0
        self._bounds: list[tuple[float, float]] = []
        self._mismatch_since: float | None = None
        self._await_zero: dict[str, Any] | None = None
        self._last_sent: tuple[int, int] | None = None
        self._last_tx_at: float | None = None
        self._zero_until = 0.0
        self._hold_until = 0.0
        self._cut_at = 0.0
        self._last_now = 0.0
        start = 0.0
        for step in plan.steps:
            self._bounds.append((start, start + step.duration_s))
            start += step.duration_s

    # ---- receiving

    def receive(self, data: bytes, now: float) -> list[str]:
        lines = self.core.receive(data, now)
        for text in lines:
            status = client.parse_status(text)
            if status is not None:
                self._on_status(status, now)
        return lines

    def _on_status(self, status: client.Status, now: float) -> None:
        if self.state in (RUNNING, CUT) and 0 <= self._step < len(self.step_records):
            pair = [status.left, status.right]
            seen = self.step_records[self._step]["reported"]
            if not seen or seen[-1] != pair:
                seen.append(pair)
        if self._await_zero is not None and status.left == 0 and status.right == 0:
            waited = self._await_zero
            self.stop_latencies.append({"after_step": waited["after_step"], "tactic": waited["tactic"],
                                        "seconds": round(now - waited["at"], 3)})
            self._await_zero = None
        if self.state == CUT and self.cut_result is None and status.fault == "PC_STALE" and status.left == 0 and status.right == 0:
            self.cut_result = {"stopped": True, "seconds_after_last_command": round(now - self._cut_at, 3),
                               "fault": status.fault, "stop": status.stop, "block": status.block}
            self._event("firmware_stale_stop", now, **self.cut_result)

    # ---- readiness and operator actions

    def not_ready_reason(self, now: float) -> str:
        core = self.core
        if not core.fresh(now):
            return "no status from the controller (wrong port, wrong network, or the sketch is not running)"
        if not core.hello:
            return "firmware not identified yet (waiting for its HELLO line)"
        if not core.firmware_known:
            return f"firmware {core.hello.get('fw', '?')!r} is not a handoff build this tool knows"
        if core.pc_limit is None or core.pc_limit < self.plan.max_command:
            return f"firmware PC limit {core.pc_limit} is below the plan's largest command {self.plan.max_command}"
        status = core.status
        if not status.link:
            return "no radio link: switch the RadioMaster on"
        if status.fault != "NONE":
            return f"fault {status.fault} is latched: turn ARM off, then on (on the RadioMaster)"
        if status.block != "NONE":
            return f"{status.block}: {BLOCK_TEXT.get(status.block) or client.BLOCK_HELP.get(status.block, 'see the ST line')}"
        if status.requested != "PC" or status.active != "PC" or not status.pc_ok:
            return "the firmware is not accepting PC commands: set MODE to PC on the RadioMaster"
        if status.left != 0 or status.right != 0:
            return f"the firmware reports motor outputs L={status.left} R={status.right}, not zero"
        return ""

    def go(self, now: float) -> str:
        """The operator's confirmation. Returns '' when the sequence starts, else why it was refused."""
        if self.state != READY:
            return f"not ready: {self.not_ready_reason(now) or self.state}"
        if self._ready_since is None or now - self._ready_since < READY_HOLD_S:
            return "not ready for long enough yet: confirm again"
        reason = self.not_ready_reason(now)
        if reason:
            return f"not ready: {reason}"
        self.state, self._go_at, self._go_token = RUNNING, now, self.core.status.token
        self._event("go", now, token=self._go_token)
        self._start_step(0, now)
        return ""

    def abort(self, code: str, text: str, now: float) -> None:
        if self.state in (DONE, ABORTED):
            return
        was_moving = self.state in (RUNNING, CUT)
        if self.state == RUNNING:
            self._finish_step(now)
        self.abort_code, self.abort_text = code, text
        self.state, self._ended_at, self._zero_until = ABORTED, now, now + ABORT_ZERO_S
        self.status_at_stop = self.core.status
        self._event("abort", now, code=code, text=text, during_sequence=was_moving)

    def finished(self, now: float) -> bool:
        if self.state == DONE:
            return now - self._ended_at >= (SETTLE_S if self.cut is not None else 0.0)
        return self.state == ABORTED and now - self._ended_at >= SETTLE_S

    # ---- sending

    def tick(self, now: float) -> list[bytes]:
        """Everything to send this 20 ms period (possibly nothing)."""
        self._last_now = now
        core = self.core
        fresh = core.fresh(now)
        if self.state in (WAITING, READY):
            out = []
            if (not fresh or not core.hello) and now - self._probed_at >= PROBE_PERIOD_S:
                self._probed_at = now
                out.append(b"?\n")                      # a status request, never a drive command
            reason = self.not_ready_reason(now)
            if reason:
                self.state, self._ready_since = WAITING, None
            else:
                if self._ready_since is None:
                    self._ready_since = now
                self.state = READY
            if fresh:
                out.append(self._command(0, 0, now))
            return out
        if self.state == RUNNING:
            problem = self._running_problem(now, fresh)
            if problem is not None:
                self.abort(problem[0], problem[1], now)
                return [self._command(0, 0, now)] if fresh else []
            elapsed = now - self._go_at
            while self._step < len(self.plan.steps) and elapsed >= self._bounds[self._step][1] - 1e-9:
                self._finish_step(now)
                if self._step + 1 < len(self.plan.steps):
                    self._start_step(self._step + 1, now)
                else:
                    self._step += 1
            if self._step >= len(self.plan.steps):
                self.state, self._hold_until = HOLDING, now + HOLD_ZERO_S
                self._event("sequence_complete", now)
                return [self._command(0, 0, now)]
            step = self.plan.steps[self._step]
            if self.cut is not None and self._step == self.cut[0] and now - self._step_started >= self.cut[1]:
                self.state, self._cut_at = CUT, (self._last_tx_at if self._last_tx_at is not None else now)
                self._event("link_cut_on_purpose", now, step=step.index, tactic=step.tactic)
                return []
            return [self._command(step.throttle, step.steer, now)]
        if self.state == CUT:
            if self.cut_result is not None or now - self._cut_at > CUT_TIMEOUT_S:
                if self.cut_result is None:
                    status = core.status
                    self.cut_result = {"stopped": False, "seconds_after_last_command": None,
                                       "fault": status.fault, "stop": status.stop, "block": status.block,
                                       "left": status.left, "right": status.right}
                    self._event("firmware_did_not_report_a_stale_stop", now, **self.cut_result)
                self._finish_step(now)
                self.state, self._ended_at = DONE, now
                self.status_at_stop = core.status
            return []
        if self.state == HOLDING:
            if not fresh or not core.status.pc_ok or now >= self._hold_until:
                self.state, self._ended_at = DONE, now
                self.status_at_stop = core.status
                self._event("done", now, reason=("status lost" if not fresh else "PC no longer selected or armed"
                                                 if not core.status.pc_ok else "hold time over"))
                return []
            return [self._command(0, 0, now)]
        if self.state == ABORTED and fresh and now < self._zero_until:
            return [self._command(0, 0, now)]
        return []

    def _running_problem(self, now: float, fresh: bool) -> tuple[str, str] | None:
        if not fresh:
            return "STATUS_LOST", "no status line from the controller for 0.5 s"
        status = self.core.status
        if not status.link:
            return "RADIO_LOST", "the controller reports the radio link down"
        if status.fault != "NONE":
            return status.fault, f"the controller latched fault {status.fault}"
        if not status.armed:
            return "DISARMED", f"the controller reports disarmed ({status.block})"
        if status.requested != "PC" or status.active != "PC" or not status.pc_ok:
            return "NOT_PC", f"the controller is no longer taking PC commands (req={status.requested} act={status.active} block={status.block})"
        if status.token != self._go_token:
            return "TOKEN_CHANGED", "the controller's token changed: control changed hands since GO"
        step = self.plan.steps[self._step] if 0 <= self._step < len(self.plan.steps) else None
        if step is not None and (status.left, status.right) != (step.left, step.right):
            since = max(self._mismatch_since if self._mismatch_since is not None else now, self._step_started)
            self._mismatch_since = since
            if now - since > MISMATCH_GRACE_S:
                return "OUTPUT_MISMATCH", (f"the controller reports L={status.left} R={status.right} but step {step.index} "
                                           f"({step.tactic}) commands L={step.left} R={step.right}")
        else:
            self._mismatch_since = None
        return None

    def _command(self, throttle: int, steer: int, now: float) -> bytes:
        if (throttle or steer) and self.state != RUNNING:
            raise AssertionError("a non-zero command outside a confirmed sequence")
        if self._last_sent != (throttle, steer):
            self._event("tx", now, throttle=throttle, steer=steer)
            self._last_sent = (throttle, steer)
        self._last_tx_at = now
        token = self.core.status.token
        if not self.sequenced:
            return client.format_command(token, throttle, steer)
        self.sequence += 1
        return client.format_command(token, throttle, steer, self.sequence)

    def _start_step(self, index: int, now: float) -> None:
        step = self.plan.steps[index]
        self._step, self._step_started, self._mismatch_since = index, now, None
        status = self.core.status
        self.step_records.append({
            "index": step.index, "tactic": step.tactic, "planned_s": step.duration_s,
            "throttle": step.throttle, "steer": step.steer, "expected_left": step.left, "expected_right": step.right,
            "source": f"PC scripts/motion_check.py over {self.link_label}",
            "controller_active_source": status.active if status else None,
            "start_unix": round(self.unix(), 3), "start_s": round(now - self._go_at, 3), "reported": []})
        self._event("step", now, index=step.index, tactic=step.tactic, throttle=step.throttle, steer=step.steer)
        if not step.moving and index > 0 and self.plan.steps[index - 1].moving:
            self._await_zero = {"after_step": index - 1, "tactic": self.plan.steps[index - 1].tactic, "at": now}

    def _finish_step(self, now: float) -> None:
        if not self.step_records or "end_unix" in self.step_records[-1]:
            return
        record = self.step_records[-1]
        record["end_unix"] = round(self.unix(), 3)
        record["actual_s"] = round(now - self._step_started, 3)
        expected = [record["expected_left"], record["expected_right"]]
        record["controller_reported_expected_outputs"] = expected in record["reported"]

    def _event(self, name: str, now: float, **fields: Any) -> None:
        self.events.append({"t_unix": round(self.unix(), 3), "t_s": round(now, 3), "event": name, **fields})

    # ---- for the operator and the record

    def headline(self, now: float) -> str:
        if self.state == WAITING:
            return f"WAITING  nothing will move. {self.not_ready_reason(now)}"
        if self.state == READY:
            return "READY    the controller is armed and accepting PC commands, outputs at zero. Type GO + Enter to run."
        if self.state == RUNNING:
            step = self.plan.steps[self._step]
            return f"RUNNING  step {step.index} {step.tactic} for {step.duration_s:.2f} s   (Enter or Ctrl+C stops; ARM off always stops)"
        if self.state == CUT:
            return "LINK CUT ON PURPOSE  nothing is being sent; the controller should stop by itself within about 0.25 s"
        if self.state == HOLDING:
            return "COMPLETE sending zero. Switch ARM off on the RadioMaster now."
        if self.state == ABORTED:
            return f"STOPPED  {self.abort_code}: {self.abort_text}"
        return "DONE     nothing is being sent"

    def result(self) -> dict[str, Any]:
        status = self.status_at_stop or self.core.status
        final = self.core.status
        moving = [record for record in self.step_records if record["throttle"] or record["steer"]]
        return {
            "state": self.state, "completed": self.state == DONE and self.abort_code is None and self.cut is None
            and len(self.step_records) == len(self.plan.steps),
            "abort": None if self.abort_code is None else {"code": self.abort_code, "text": self.abort_text},
            "link": self.link_label, "firmware_hello": dict(self.core.hello),
            "steps": self.step_records, "stop_latencies": self.stop_latencies, "link_cut": self.cut_result,
            "controller_reported_every_moving_step": bool(moving) and all(
                record.get("controller_reported_expected_outputs") for record in moving),
            "status_when_stopped": None if status is None else asdict(status),
            "status_final": None if final is None else asdict(final),
            "events": self.events}


# ---------------------------------------------------------------------------------------------- safety checks record


def check_plans(scale: float) -> dict[str, tuple[Plan, tuple[int, float] | None]]:
    """The five wheels-raised checks. The long move is there to be interrupted by hand."""
    short = build_plan("stop:0.6,advance:0.6,stop:1.0", scale)
    long = build_plan("stop:0.6,advance:2.0,stop:0.6", scale, max_move_s=MAX_CHECK_MOVE_S)
    return {"stop": (short, None), "disarm": (long, None), "takeover": (long, None),
            "pc_link_loss": (long, (1, 0.5)), "radio_loss": (long, None)}


CHECK_INSTRUCTIONS = {
    "stop": "Motor stop: the wheels turn forward for 0.6 s and must stop when the command ends. Do nothing.",
    "disarm": "Disarm: when the wheels start turning, switch ARM OFF on the RadioMaster. They must stop at once.",
    "takeover": "Manual takeover: when the wheels start turning, switch MODE to RADIO with both sticks centred. "
                "They must stop at once, and the sticks must then drive the robot.",
    "pc_link_loss": "PC link loss: this program stops sending half a second into the move, on purpose. The wheels "
                    "must stop by themselves within about a quarter of a second.",
    "radio_loss": "Radio loss: when the wheels start turning, switch the RadioMaster OFF. They must stop within "
                  "about a quarter of a second. Switch it back on afterwards.",
}
CHECK_QUESTIONS = {
    "stop": "Did both wheels turn forward and then stop when the command ended?",
    "disarm": "Did the wheels stop at once when you switched ARM off?",
    "takeover": "Did the wheels stop at once at the MODE switch, and do the sticks drive the robot now?",
    "pc_link_loss": "Did the wheels stop by themselves within about a quarter of a second?",
    "radio_loss": "Did the wheels stop within about a quarter of a second of switching the transmitter off?",
}


def judge_check(name: str, result: dict[str, Any]) -> tuple[bool, str]:
    """What the controller's own reports say about one check. The operator's eyes are asked separately."""
    stopped = result["status_final"] or {}
    outputs_zero = stopped.get("left") == 0 and stopped.get("right") == 0
    abort = result["abort"] or {}
    if name == "stop":
        latency = next((item["seconds"] for item in result["stop_latencies"] if item["tactic"] == "advance"), None)
        if not result["completed"]:
            return False, f"the sequence did not complete ({abort.get('code', result['state'])})"
        if not result["controller_reported_every_moving_step"]:
            return False, "the controller never reported the commanded outputs"
        if latency is None or latency > STOP_PASS_S:
            return False, f"outputs not reported at zero within {STOP_PASS_S} s of the stop command (saw {latency})"
        return True, f"controller reported the commanded outputs, then L=0 R=0 {latency:.2f} s after the stop command"
    if name == "pc_link_loss":
        cut = result["link_cut"] or {}
        seconds = cut.get("seconds_after_last_command")
        if not cut.get("stopped") or seconds is None:
            return False, "the controller did not report a PC_STALE stop within 1 s of the last command"
        if seconds > CUT_PASS_S:
            return False, f"the controller took {seconds:.2f} s to report the stop (limit {CUT_PASS_S} s)"
        return True, f"controller reported fault PC_STALE with L=0 R=0, {seconds:.2f} s after the last command"
    expected = {"disarm": ("DISARMED",), "takeover": ("NOT_PC", "TOKEN_CHANGED"), "radio_loss": ("RADIO_LOST", "STATUS_LOST")}[name]
    if not abort:
        return False, "the move ran to its end: the switch was not operated during it"
    if abort.get("code") not in expected:
        return False, f"stopped for a different reason ({abort.get('code')}: {abort.get('text')})"
    if abort.get("code") == "STATUS_LOST":
        return False, "status was lost instead of a reported radio loss: the controller may have restarted"
    if not outputs_zero:
        return False, f"the controller still reports L={stopped.get('left')} R={stopped.get('right')}"
    detail = {"disarm": f"armed={int(bool(stopped.get('armed')))} stop={stopped.get('stop')}",
              "takeover": f"req={stopped.get('requested')} act={stopped.get('active')} stop={stopped.get('stop')}",
              "radio_loss": f"link={int(bool(stopped.get('link')))} fault={stopped.get('fault')}"}[name]
    ok = {"disarm": not stopped.get("armed"), "takeover": stopped.get("requested") == "RADIO",
          "radio_loss": not stopped.get("link") or stopped.get("fault") == "RADIO_LOST"}[name]
    return ok, f"controller reported {detail} with L=0 R=0"


def load_safety_records(path: Path | None = None) -> list[dict[str, Any]]:
    path = SAFETY_RECORD if path is None else path
    if not path.is_file():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    return data if isinstance(data, list) else []


def floor_permission(firmware: str | None, now_unix: float, path: Path | None = None) -> tuple[bool, str]:
    """A floor run needs every wheels-raised check passed on this firmware, recently. There is no override."""
    if not firmware:
        return False, "the firmware has not identified itself"
    candidates = [record for record in load_safety_records(path) if record.get("firmware") == firmware]
    if not candidates:
        return False, f"no wheels-raised safety checks are recorded for firmware {firmware}: run `checks` first"
    record = max(candidates, key=lambda item: item.get("finished_unix", 0.0))
    age = now_unix - float(record.get("finished_unix", 0.0))
    if age > SAFETY_MAX_AGE_S or age < -60.0:
        return False, f"the last safety checks on {firmware} are {age / 3600.0:.1f} h old (limit {SAFETY_MAX_AGE_S / 3600.0:.0f} h): rerun `checks`"
    missing = [name for name in REQUIRED_CHECKS
               if not (record.get("checks", {}).get(name, {}).get("controller_pass")
                       and record.get("checks", {}).get(name, {}).get("operator_confirmed"))]
    if missing:
        return False, f"safety checks not passed on {firmware}: {', '.join(missing)}"
    return True, f"all five wheels-raised checks passed on {firmware} {age / 60.0:.0f} min ago"


# ---------------------------------------------------------------------------------------------- comparison


OBSERVED_TEMPLATE_HEAD = """# What you SAW in this run. Fill in what you measured; leave anything you did not measure as null.
# Nothing here is assumed: an empty field is reported as NOT RECORDED.
#
# surface: {surface}
#   raised -> fill left_wheel / right_wheel with forward, reverse or still (looking at each wheel).
#   floor  -> fill direction (forward, reverse, turn_left, turn_right, none), distance_mm (how far the
#             chassis centre moved, always positive) and rotation_deg (+ = turned LEFT, counter-clockwise
#             seen from above; - = turned right). Say how you measured it (tape, video, camera trace).
measured_by: ""
notes: ""
steps:
"""


def write_observed_template(path: Path, plan: Plan, surface: str) -> None:
    lines = [OBSERVED_TEMPLATE_HEAD.format(surface=surface)]
    for step in plan.steps:
        if not step.moving:
            continue
        lines.append(f"  - index: {step.index}\n    tactic: {step.tactic}\n    left_wheel: null\n    right_wheel: null\n"
                     f"    direction: null\n    distance_mm: null\n    rotation_deg: null\n    stopped_when_command_ended: null\n")
    path.write_text("".join(lines), encoding="utf-8")


def _wheel_word(value: int) -> str:
    return "forward" if value > 0 else "reverse" if value < 0 else "still"


def compare(reference: dict[str, Any], run: dict[str, Any] | None, observed: dict[str, Any] | None) -> dict[str, Any]:
    """Simulator against controller reports against what the operator saw. Signs are judged; sizes are ratios."""
    seen = {int(item["index"]): item for item in (observed or {}).get("steps") or [] if item and item.get("index") is not None}
    ran = {int(item["index"]): item for item in ((run or {}).get("result") or {}).get("steps") or []}
    surface = (run or {}).get("surface")
    rows, mismatches, ratios = [], [], {"distance": [], "rotation": []}
    for sim_step in reference["steps"]:
        plan_step = reference["plan"]["steps"][sim_step["index"]]
        if not (plan_step["throttle"] or plan_step["steer"]):
            continue
        index, tactic = sim_step["index"], sim_step["tactic"]
        row: dict[str, Any] = {"index": index, "tactic": tactic, "command_class": plan_step["command_class"],
                               "sim_class": sim_step["motion_class"], "sim_forward_m": sim_step["forward_m"],
                               "sim_heading_change_deg": sim_step["heading_change_deg"]}
        row["sim_agrees_with_command"] = sim_step["motion_class"] == plan_step["command_class"]
        if not row["sim_agrees_with_command"]:
            mismatches.append(f"step {index} {tactic}: the simulator moved as {sim_step['motion_class']} for a {plan_step['command_class']} command")
        record = ran.get(index)
        row["controller"] = ("NOT RUN" if record is None else
                             "reported the commanded outputs" if record.get("controller_reported_expected_outputs")
                             else f"did NOT report L={plan_step['left']} R={plan_step['right']} (saw {record.get('reported')})")
        if record is not None and not record.get("controller_reported_expected_outputs"):
            mismatches.append(f"step {index} {tactic}: {row['controller']}")
        item = seen.get(index, {})
        wheels = (item.get("left_wheel"), item.get("right_wheel"))
        if all(wheels):
            expected = (_wheel_word(plan_step["left"]), _wheel_word(plan_step["right"]))
            row["wheels_observed"], row["wheels_expected"] = list(wheels), list(expected)
            row["wheels_match"] = tuple(str(word).lower() for word in wheels) == expected
            if not row["wheels_match"]:
                mismatches.append(f"step {index} {tactic}: wheels seen {wheels[0]}/{wheels[1]}, expected {expected[0]}/{expected[1]} (left/right)")
        else:
            row["wheels_match"] = None
        direction = item.get("direction")
        row["direction_observed"] = direction
        if direction:
            row["direction_match"] = str(direction).lower() == sim_step["motion_class"]
            if not row["direction_match"]:
                mismatches.append(f"step {index} {tactic}: robot moved {direction}, simulator moved {sim_step['motion_class']}")
        else:
            row["direction_match"] = None
        distance, rotation = item.get("distance_mm"), item.get("rotation_deg")
        row["distance_mm"], row["rotation_deg"] = distance, rotation
        row["distance_ratio_physical_over_sim"] = row["rotation_ratio_physical_over_sim"] = None
        if distance is not None and abs(sim_step["forward_m"]) >= 0.02 and sim_step["motion_class"] in ("forward", "reverse"):
            row["distance_ratio_physical_over_sim"] = round(float(distance) / 1000.0 / abs(sim_step["forward_m"]), 3)
            ratios["distance"].append(row["distance_ratio_physical_over_sim"])
        if rotation is not None and abs(sim_step["heading_change_deg"]) >= 5.0 and sim_step["motion_class"].startswith("turn"):
            if float(rotation) * sim_step["heading_change_deg"] < 0:
                mismatches.append(f"step {index} {tactic}: rotation sign {rotation:+} deg opposes the simulator's {sim_step['heading_change_deg']:+.1f}")
            row["rotation_ratio_physical_over_sim"] = round(abs(float(rotation)) / abs(sim_step["heading_change_deg"]), 3)
            ratios["rotation"].append(row["rotation_ratio_physical_over_sim"])
        if item.get("stopped_when_command_ended") is False:
            mismatches.append(f"step {index} {tactic}: the robot did not stop when the command ended")
        rows.append(row)

    judged = [row for row in rows if row["wheels_match"] is not None or row["direction_match"] is not None]
    if run is None:
        verdict = "SIMULATOR ONLY: no physical run in this folder"
    elif mismatches:
        verdict = "MISMATCH: see the list. Do not go further until each one is explained"
    elif len(judged) < len(rows):
        verdict = (f"INCOMPLETE: {len(rows) - len(judged)} of {len(rows)} moving steps have no observation "
                   "(fill in observed.yaml); nothing is assumed for them")
    else:
        verdict = "SIGNS AGREE on every moving step" + (" (wheel directions only: wheels were raised)" if surface == "raised" else "")
    calibration = []
    for name, unit in (("distance", "straight-line travel"), ("rotation", "rotation")):
        if ratios[name]:
            mean = sum(ratios[name]) / len(ratios[name])
            calibration.append(f"{unit}: the robot did {mean:.2f} x what the simulator predicted "
                               f"({', '.join(f'{value:.2f}' for value in ratios[name])}); a calibration difference, not a failure")
    if run is not None and not calibration and surface == "floor":
        calibration.append("no distances or rotations were recorded, so there is no calibration ratio")
    return {"verdict": verdict, "surface": surface, "mismatches": mismatches, "calibration": calibration, "rows": rows,
            "evidence": {"simulator": "SIM", "controller reports": "HARDWARE (reported by the firmware, not measured motion)"
                         if run is not None else "none", "observations": (observed or {}).get("measured_by") or "NOT RECORDED"}}


def format_comparison(report: dict[str, Any]) -> str:
    lines = [f"VERDICT: {report['verdict']}", "",
             "| # | tactic | simulator | controller | wheels seen (L/R) | robot moved | distance mm (x sim) | rotation deg (x sim) |",
             "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for row in report["rows"]:
        wheels = "NOT RECORDED" if row["wheels_match"] is None else (
            "/".join(map(str, row["wheels_observed"])) + (" ok" if row["wheels_match"] else " MISMATCH"))
        moved = "NOT RECORDED" if row["direction_match"] is None else (
            f"{row['direction_observed']}" + (" ok" if row["direction_match"] else " MISMATCH"))
        distance = "NOT RECORDED" if row["distance_mm"] is None else (
            f"{row['distance_mm']}" + (f" ({row['distance_ratio_physical_over_sim']:.2f})" if row["distance_ratio_physical_over_sim"] else ""))
        rotation = "NOT RECORDED" if row["rotation_deg"] is None else (
            f"{row['rotation_deg']}" + (f" ({row['rotation_ratio_physical_over_sim']:.2f})" if row["rotation_ratio_physical_over_sim"] else ""))
        lines.append(f"| {row['index']} | {row['tactic']} | {row['sim_class']}, {row['sim_forward_m']:+.3f} m, "
                     f"{row['sim_heading_change_deg']:+.1f} deg | {row['controller']} | {wheels} | {moved} | {distance} | {rotation} |")
    if report["mismatches"]:
        lines += ["", "Mismatches:"] + [f"- {text}" for text in report["mismatches"]]
    if report["calibration"]:
        lines += ["", "Calibration differences:"] + [f"- {text}" for text in report["calibration"]]
    lines += ["", "Evidence: " + "; ".join(f"{key} = {value}" for key, value in report["evidence"].items())]
    return "\n".join(lines)


# ---------------------------------------------------------------------------------------------- command line


class Lines:
    """Lines typed by the operator, read on a thread so the 50 Hz loop never waits for the keyboard."""

    def __init__(self, stream=None) -> None:
        self.queue: "queue.Queue[str | None]" = queue.Queue()
        self.closed = False
        thread = threading.Thread(target=self._read, args=(stream or sys.stdin,), daemon=True)
        thread.start()

    def _read(self, stream) -> None:
        for line in stream:
            self.queue.put(line.rstrip("\r\n"))
        self.queue.put(None)

    def poll(self) -> str | None:
        """A typed line, or None when nothing was typed. End of input reads as an empty line (which stops a run)."""
        try:
            line = self.queue.get_nowait()
        except queue.Empty:
            return None
        if line is None:
            self.closed = True
            return ""
        return line

    def ask(self, question: str, idle: Callable[[], None] | None = None) -> str:
        print(question, end=" ", flush=True)
        while True:
            line = self.poll()
            if line is not None:
                if self.closed:
                    raise EOFError("input closed")
                return line.strip()
            if idle is not None:
                idle()
            time.sleep(0.02)

    def yes(self, question: str, idle: Callable[[], None] | None = None) -> bool:
        return self.ask(f"{question} [y/N]", idle).lower() in ("y", "yes")


def drive(port, runner: Runner, log: client.Log, lines: Lines, show: Callable[[str], None] = print) -> None:
    """The 50 Hz loop. Returns when the runner has finished or stopped."""
    shown = ""
    next_at = time.monotonic()
    try:
        while True:
            now = time.monotonic()
            try:
                log.received(runner.receive(port.read(4096), now))
            except Exception as error:               # cable pulled, port gone: nothing more can be sent
                runner.abort("PORT_LOST", f"the link to the controller failed while reading: {error}", now)
                show(runner.headline(now))
                show("         The controller stops by itself 250 ms after the last command it received.")
                return
            typed = lines.poll()
            if typed is not None:
                if runner.state == READY and typed.strip().upper() == "GO":
                    refusal = runner.go(now)
                    if refusal:
                        show(f"         GO refused: {refusal}")
                elif runner.state in (RUNNING, CUT):
                    runner.abort("OPERATOR", "stopped from the keyboard", now)
                elif runner.state in (WAITING, READY) and typed.strip().upper() in ("Q", "QUIT"):
                    runner.abort("OPERATOR", "cancelled before starting", now)
                elif runner.state in (WAITING, READY) and typed.strip():
                    show("         (type GO to run, Q to cancel)")
            try:
                for datagram in runner.tick(now):
                    port.write(datagram)
                    log.sent(None if datagram == b"?\n" else datagram)
            except Exception as error:
                runner.abort("PORT_LOST", f"the link to the controller failed while writing: {error}", now)
                show(runner.headline(now))
                show("         The controller stops by itself 250 ms after the last command it received.")
                return
            headline = runner.headline(now)
            if headline != shown:
                show(headline)
                shown = headline
            if runner.finished(now):
                return
            next_at = max(next_at + TICK_S, now)
            time.sleep(max(0.0, next_at - time.monotonic()))
    except KeyboardInterrupt:
        now = time.monotonic()
        runner.abort("OPERATOR", "Ctrl+C", now)
        try:
            for _ in range(5):                           # a few zero commands, then silence
                for datagram in runner.tick(time.monotonic()):
                    port.write(datagram)
                time.sleep(TICK_S)
        except Exception:                                # the port is gone as well: the firmware times out by itself
            pass
        show(runner.headline(time.monotonic()))


def shown_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def sketch_hashes() -> dict[str, str]:
    """Hashes of the sketch files in this checkout. They say nothing about what is flashed on the board."""
    out = {}
    for name in ("ELRS-Handoff-Bench/ELRS-Handoff-Bench.ino", "ELRS-Handoff-WiFi/ELRS-Handoff-WiFi.ino",
                 "scripts/handoff_bench_client.py", "scripts/motion_check.py"):
        path = ROOT / name
        if path.is_file():
            out[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return out


def open_link(args):
    if args.wifi:
        print("EXPERIMENTAL LINK: Wi-Fi was unreliable on 2026-11-20 (restarts, dropouts). Wheels raised only makes sense here.")
        return client.UdpPort(args.wifi, args.udp_port), f"Wi-Fi {args.wifi}:{args.udp_port} (UDP)", True
    return client.open_port(args.port), f"USB {args.port}", False


def preflight(args, lines: Lines, surface: str) -> dict[str, Any] | None:
    """Declarations by the operator, written into the record. Returns None when the run must not start."""
    if not args.port and not args.wifi:
        print("--port COMx (USB) or --wifi is required.")
        return None
    if args.port and args.wifi:
        print("Use --port or --wifi, not both.")
        return None
    if args.port and args.power == "battery-only":
        print("--power battery-only is not possible over USB: the cable is plugged in.")
        return None
    if args.wifi and args.power != "battery-only":
        print("Over Wi-Fi the robot must be on battery with USB unplugged: use --power battery-only.")
        return None
    declared: dict[str, Any] = {"surface": surface, "power": args.power}
    print("\nThis program cannot arm the robot. You arm it, select PC mode and can stop it at any time with the RadioMaster.")
    if args.power == "battery-and-usb":
        print("\nPOWER: USB and the battery together. On 2026-11-20, with USB powering the board and the battery\n"
              "disconnected, about 4.8 V was measured on BOTH terminals of the buck converter: USB 5 V reaches the\n"
              "converter's output and passes back to its battery side. With both supplies connected they are tied\n"
              "together through the converter. Whether that is safe for this wiring has not been established.")
        declared["usb_battery_isolation_confirmed"] = lines.yes(
            "Have you confirmed how USB 5 V and the battery/regulator are isolated on this robot (diode or separate rails)?")
        if not declared["usb_battery_isolation_confirmed"]:
            print("Not started. Use --power usb-only (battery disconnected), or resolve the isolation question first.")
            return None
    if args.power == "usb-only":
        print("\nPOWER: USB only, battery disconnected. Note the measurement above: USB 5 V was seen on the converter's\n"
              "battery side, so the motor driver may be powered from USB. Treat the wheels as live.")
    if surface == "raised":
        declared["wheels_raised_confirmed"] = lines.yes("Are both wheels off the ground and free to spin, with the power cut-off in reach?")
        if not declared["wheels_raised_confirmed"]:
            print("Not started.")
            return None
    else:
        declared["floor_clear_confirmed"] = lines.yes(
            "Is the robot on a clear, flat floor with at least 0.5 m free all round, and is the RadioMaster in your hand?")
        if not declared["floor_clear_confirmed"]:
            print("Not started.")
            return None
    return declared


def command_plan(args) -> int:
    plan = build_plan(args.sequence, args.scale)
    print(format_plan(plan))
    print("\nNothing was sent anywhere. `sim` writes the simulator reference; `run` is the supervised physical run.")
    return 0


def command_sim(args) -> int:
    plan = build_plan(args.sequence, args.scale)
    reference = sim_reference(plan, args.config)
    out = Path(args.out) if args.out else OUT_ROOT / "sim-reference.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(reference, indent=1), encoding="utf-8")
    print(format_plan(plan))
    print()
    print(format_sim(reference))
    print(f"\nwritten: {out}")
    return 0


def command_run(args) -> int:
    plan = build_plan(args.sequence, args.scale)
    print(format_plan(plan))
    reference = sim_reference(plan, args.config)
    print()
    print(format_sim(reference))
    lines = Lines()
    declared = preflight(args, lines, args.surface)
    if declared is None:
        return 2
    folder = OUT_ROOT / time.strftime(f"run-%Y%m%d-%H%M%S-{args.surface}")
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "sim_reference.json").write_text(json.dumps(reference, indent=1), encoding="utf-8")
    port, label, sequenced = open_link(args)
    log = client.Log(folder / "link.txt")
    runner = Runner(plan, sequenced=sequenced, link_label=label)
    started = time.time()
    try:
        if args.surface == "floor":
            print("\nFLOOR RUN: checking the wheels-raised safety record once the firmware has identified itself...")
            deadline = time.monotonic() + 5.0
            while not runner.core.hello and time.monotonic() < deadline:
                now = time.monotonic()
                log.received(runner.receive(port.read(4096), now))
                for datagram in runner.tick(now):
                    port.write(datagram)
                time.sleep(TICK_S)
            allowed, why = floor_permission(runner.core.hello.get("fw"), time.time())
            print(("  allowed: " if allowed else "  REFUSED: ") + why)
            if not allowed:
                return 2
            declared["floor_permission"] = why
        print("\nOn the RadioMaster: ARM off, MODE = PC, then ARM on with nothing held. Then type GO here.\n")
        drive(port, runner, log, lines)
    finally:
        log.close()
        port.close()
    result = runner.result()
    record = {"evidence": "HARDWARE (controller reports and the operator's observations; not a measurement of motion by itself)",
              "started_unix": round(started, 3), "finished_unix": round(time.time(), 3), "surface": args.surface,
              "declared": declared, "files_in_checkout_sha256": sketch_hashes(), "plan": plan.to_report(), "result": result}
    (folder / "run.json").write_text(json.dumps(record, indent=1), encoding="utf-8")
    write_observed_template(folder / "observed.yaml", plan, args.surface)
    print(f"\nrecord: {folder}")
    if result["completed"]:
        print("Sequence completed. Now write what you saw into observed.yaml in that folder, then:")
        print(f"  python scripts\\motion_check.py compare {shown_path(folder)}")
        return 0
    print(f"Sequence did NOT complete: {(result['abort'] or {}).get('code')} - {(result['abort'] or {}).get('text')}")
    return 1


def command_checks(args) -> int:
    plans = check_plans(args.scale)
    lines = Lines()
    declared = preflight(args, lines, "raised")
    if declared is None:
        return 2
    folder = OUT_ROOT / time.strftime("checks-%Y%m%d-%H%M%S")
    folder.mkdir(parents=True, exist_ok=True)
    port, label, sequenced = open_link(args)
    log = client.Log(folder / "link.txt")
    record: dict[str, Any] = {"started_unix": round(time.time(), 3), "link": label, "declared": declared,
                              "scale": args.scale, "firmware": None, "checks": {}}

    def idle() -> None:
        log.received(client.ClientCore().receive(port.read(4096), time.monotonic()))

    try:
        for name in REQUIRED_CHECKS:
            plan, cut = plans[name]
            print(f"\n=== CHECK {REQUIRED_CHECKS.index(name) + 1} of {len(REQUIRED_CHECKS)}: {name} ===\n{CHECK_INSTRUCTIONS[name]}")
            print("On the RadioMaster: ARM off, MODE = PC, ARM on, nothing held. Then type GO (or Q to skip this check).\n")
            runner = Runner(plan, sequenced=sequenced, link_label=label, cut=cut)
            drive(port, runner, log, lines)
            result = runner.result()
            record["firmware"] = result["firmware_hello"].get("fw") or record["firmware"]
            skipped = (result["abort"] or {}).get("text") == "cancelled before starting"
            ok, why = (False, "skipped") if skipped else judge_check(name, result)
            print(f"  controller evidence: {'PASS' if ok else 'FAIL'} - {why}")
            confirmed = bool(ok) and lines.yes("  " + CHECK_QUESTIONS[name], idle)
            record["checks"][name] = {"controller_pass": bool(ok), "controller_evidence": why,
                                      "operator_confirmed": confirmed, "result": result}
            if name == "radio_loss":
                print("  Switch the RadioMaster back on if it is off.")
    except EOFError:
        print("\ninput closed: checks stopped")
    finally:
        log.close()
        port.close()
    record["finished_unix"] = round(time.time(), 3)
    (folder / "checks.json").write_text(json.dumps(record, indent=1), encoding="utf-8")
    summary = {key: record[key] for key in ("started_unix", "finished_unix", "link", "firmware", "scale")}
    summary["checks"] = {name: {key: value for key, value in item.items() if key != "result"} for name, item in record["checks"].items()}
    summary["detail"] = shown_path(folder)
    SAFETY_RECORD.parent.mkdir(parents=True, exist_ok=True)
    SAFETY_RECORD.write_text(json.dumps(load_safety_records() + [summary], indent=1), encoding="utf-8")
    print(f"\nrecord: {folder}")
    for name in REQUIRED_CHECKS:
        item = record["checks"].get(name)
        state = "NOT RUN" if item is None else "PASS" if item["controller_pass"] and item["operator_confirmed"] else "FAIL"
        print(f"  {name:<13} {state}" + ("" if item is None else f"   {item['controller_evidence']}"))
    allowed, why = floor_permission(record["firmware"], time.time())
    print(("\nFloor runs are now permitted: " if allowed else "\nFloor runs stay refused: ") + why)
    return 0 if allowed else 1


def command_compare(args) -> int:
    import yaml

    folder = Path(args.folder)
    reference_path = folder / "sim_reference.json" if folder.is_dir() else folder
    if not reference_path.is_file():
        print(f"no sim_reference.json in {folder}")
        return 2
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    run_path, observed_path = reference_path.parent / "run.json", reference_path.parent / "observed.yaml"
    run = json.loads(run_path.read_text(encoding="utf-8")) if run_path.is_file() else None
    observed = yaml.safe_load(observed_path.read_text(encoding="utf-8")) if observed_path.is_file() else None
    report = compare(reference, run, observed)
    text = format_comparison(report)
    print(text)
    if run is not None:
        (reference_path.parent / "comparison.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
        (reference_path.parent / "comparison.md").write_text(text + "\n", encoding="utf-8")
        print(f"\nwritten: {reference_path.parent / 'comparison.md'}")
    return 1 if report["mismatches"] else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sim-to-real movement check: a short fixed tactic sequence, simulated and (supervised) physical.")
    commands = parser.add_subparsers(dest="command", required=True)

    def sequence_options(sub) -> None:
        sub.add_argument("--sequence", default=DEFAULT_SEQUENCE, help=f"tactic:seconds, comma separated (default {DEFAULT_SEQUENCE})")
        sub.add_argument("--scale", type=float, default=DEFAULT_SCALE,
                         help=f"fraction of the simulator's full tactic command, at most {MAX_SCALE:.2f} (default {DEFAULT_SCALE})")
        sub.add_argument("--config", default=str(SIM_CONFIG), help="simulator config (default config/sim.yaml)")

    def link_options(sub) -> None:
        sub.add_argument("--port", help="USB: serial port of the ESP32-C3, e.g. COM5 (handoff_bench_client.py --list shows them)")
        sub.add_argument("--wifi", nargs="?", const=client.WIFI_HOST, metavar="ROBOT_IP", help="EXPERIMENTAL: the Wi-Fi handoff sketch over UDP")
        sub.add_argument("--udp-port", type=int, default=client.WIFI_UDP_PORT)
        sub.add_argument("--power", required=True, choices=POWER_CHOICES, help="how the robot is powered for this run (recorded)")

    sub = commands.add_parser("plan", help="print the sequence and the exact commands; no hardware")
    sequence_options(sub)
    sub.set_defaults(handler=command_plan)
    sub = commands.add_parser("sim", help="write the simulator reference trace; no hardware")
    sequence_options(sub)
    sub.add_argument("--out", help="output file (default artifacts/motion-check/sim-reference.json)")
    sub.set_defaults(handler=command_sim)
    sub = commands.add_parser("checks", help="guided stop / disarm / takeover / link-loss checks, wheels raised")
    sub.add_argument("--scale", type=float, default=DEFAULT_SCALE)
    link_options(sub)
    sub.set_defaults(handler=command_checks)
    sub = commands.add_parser("run", help="supervised physical run of the sequence (waits for you to type GO)")
    sequence_options(sub)
    link_options(sub)
    sub.add_argument("--surface", required=True, choices=SURFACES, help="raised = wheels off the ground; floor needs passed `checks`")
    sub.set_defaults(handler=command_run)
    sub = commands.add_parser("compare", help="simulator against the controller's reports and what you observed")
    sub.add_argument("folder", help="a run folder under artifacts/motion-check, or a sim reference file")
    sub.set_defaults(handler=command_compare)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(None if argv is None else list(argv))
    try:
        return args.handler(args)
    except PlanError as error:
        print(f"plan refused: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
