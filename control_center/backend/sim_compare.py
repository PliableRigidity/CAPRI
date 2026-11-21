"""Sim-to-real comparison: a simulator play log against a recorded round. Pure, no Qt.

Two questions are kept apart on purpose, because one does not answer the other:

  COMMAND agreement   were the same drive commands issued over time?
                      simulator `blue_command` / tactic   versus   what the round recorded
                      (controller-reported L/R, and separately the commands the PC client sent)
  MOTION agreement    did the robot move the same way?
                      distance travelled and heading change, camera-measured versus simulated

Matching commands say nothing about motion: the real robot can receive the simulator's exact command
sequence and still travel a different distance.

What a simulator play log contains (written by `autobot/sim/cli.py`, `preview`, human-red play):

    {"type": "reset", "time_s": 0.0}
    {"type": "step", "time_s": <sim time AFTER the step>, "blue_tactic": "advance",
     "blue_command": [left, right], "red_command": [left, right], "end_reason": null | "Pit fall (bot 1)"}

That is all: no robot poses and no contact flag. So simulated distance and heading cannot be read
from such a log (the user types them, or a future log carries `blue_pose`), and a simulated collision
cannot be detected from it. Both facts are reported rather than papered over.

Sign conventions used to compare commands (direction class only; magnitudes are in different units
and are never compared):

  simulator      wheel command -1..1, NEGATIVE = chassis forward (`autobot/sim/env.py`, tactic_to_drive)
  controller     `ST ... L= R=` -255..255, POSITIVE = forward (ELRS-Handoff-Bench sketch and README)
  PC client      `D,<tok>,<throttle>,<steer>`, positive = forward / right; left = throttle + steer,
                 right = throttle - steer (the sketch's mixer)

Those are what the sources state. That the wheels physically turn that way is not checked here.

Nothing in this module talks to hardware. It reads files.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

NO_POSE = "not available: no calibrated pose in this round"
NOT_COMPARABLE = "not comparable: the simulated collision did not happen in the real run"

SIM_FORWARD_SIGN = -1.0             # simulator: negative wheel command = forward
CONTROLLER_FORWARD_SIGN = 1.0       # controller ST line: positive L/R = forward
CONTROLLER_SCALE = 255.0
CONTROLLER_HOLD_S = 0.6             # a status older than this says nothing about now (round format doc)
PC_SENT_HOLD_S = 0.25               # the sketch's PC_TIMEOUT: an older PC command no longer counts
POSE_MAX_STEP_S = 0.25              # two valid poses further apart than this are not joined
JITTER_STEP_M = 0.005               # movement smaller than this is not added to the path length

DRIVE_CLASSES = ("stop", "forward", "reverse", "turn_left", "turn_right",
                 "forward_left", "forward_right", "reverse_left", "reverse_right")
TACTIC_CLASS = {"stop": "stop", "advance": "forward", "reverse": "reverse", "turn_left": "turn_left",
                "turn_right": "turn_right", "forward_left": "forward_left", "forward_right": "forward_right"}
STEP_FIELDS_WRITTEN_BY_CLI = ("type", "time_s", "blue_tactic", "blue_command", "red_command", "end_reason")
POSE_FIELDS = ("blue_pose",)                                   # [x_m, y_m, yaw_rad]; no current writer emits it
CONTACT_FIELDS = ("opponent_contact", "wall_contact", "contact")   # none is written by the current play loop


def wrap_deg(angle: float) -> float:
    return (angle + 180.0) % 360.0 - 180.0


# ----------------------------------------------------------------------------------------------
# Drive classes


def drive_class(left: float, right: float, forward_sign: float = 1.0, scale: float = 1.0,
                deadband: float = 0.04, ratio: float = 0.15) -> str:
    """Direction class of a two-wheel command. Magnitude is ignored beyond the deadband."""
    lf, rf = forward_sign * float(left) / scale, forward_sign * float(right) / scale
    top = max(abs(lf), abs(rf))
    if top < deadband:
        return "stop"
    forward, turn_right = (lf + rf) / 2.0, (lf - rf) / 2.0
    if abs(turn_right) <= ratio * top:
        return "forward" if forward > 0 else "reverse"
    side = "right" if turn_right > 0 else "left"
    if abs(forward) <= ratio * top:
        return f"turn_{side}"
    return f"{'forward' if forward > 0 else 'reverse'}_{side}"


def pc_mix(throttle: float, steer: float) -> tuple[float, float]:
    """The bench sketch's mixer, for classifying a PC-sent command. Not a prediction of motor output."""
    left, right = throttle + steer, throttle - steer
    top = max(abs(left), abs(right))
    if top > CONTROLLER_SCALE:
        left, right = left * CONTROLLER_SCALE / top, right * CONTROLLER_SCALE / top
    return left, right


# ----------------------------------------------------------------------------------------------
# Simulator log


@dataclass
class SimRound:
    index: int                              # 1-based, in file order
    steps: list[dict[str, Any]]
    step_s: float
    split_by: str                           # "reset line" | "simulated time restarted" | "start of file"

    @property
    def start_s(self) -> float:
        return float(self.steps[0]["time_s"]) - self.step_s

    @property
    def end_s(self) -> float:
        return float(self.steps[-1]["time_s"])

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s

    @property
    def end_reason(self) -> str | None:
        return self.steps[-1].get("end_reason")

    def describe(self) -> str:
        end = self.end_reason or "log ends (no round end recorded)"
        return f"Round {self.index}: {len(self.steps)} decisions, {self.duration_s:.2f} s, {end}"


@dataclass
class SimLog:
    path: Path
    rounds: list[SimRound]
    step_fields: tuple[str, ...]
    has_commands: bool
    has_poses: bool
    has_contact: bool
    skipped_lines: int = 0
    notes: list[str] = field(default_factory=list)

    @property
    def pose_note(self) -> str:
        if self.has_poses:
            return "this log carries simulated poses (blue_pose)"
        return ("this simulator log has no pose fields (step fields: " + ", ".join(self.step_fields) +
                "), so simulated distance and heading cannot be read from it: type the simulated values")

    @property
    def contact_note(self) -> str:
        if self.has_contact:
            return "this log carries a contact flag"
        return ("a simulated collision cannot be detected from this log: its step records carry no contact field "
                "(step fields: " + ", ".join(self.step_fields) + ")")


def load_sim_log(path: Path | str) -> SimLog:
    """Read a play log into rounds. A new round starts at every reset line, and wherever simulated
    time goes backwards (older logs have no reset lines)."""
    path = Path(path)
    groups: list[tuple[str, list[dict[str, Any]]]] = [("start of file", [])]
    fields: dict[str, None] = {}
    skipped = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            skipped += 1
            continue
        if not isinstance(record, dict):
            skipped += 1
            continue
        kind = record.get("type")
        if kind == "reset":
            groups.append(("reset line", []))
        elif kind == "step" and isinstance(record.get("time_s"), (int, float)):
            current = groups[-1][1]
            if current and record["time_s"] <= current[-1]["time_s"]:
                groups.append(("simulated time restarted", []))
            groups[-1][1].append(record)
            fields.update(dict.fromkeys(record))
        else:
            skipped += 1
    rounds = []
    for split_by, steps in groups:
        if not steps:
            continue
        times = [float(s["time_s"]) for s in steps]
        step_s = float(np.median(np.diff(times))) if len(times) > 1 else times[0]
        if not step_s > 0:
            step_s = 0.02
        rounds.append(SimRound(len(rounds) + 1, steps, step_s, split_by))
    log = SimLog(path, rounds, tuple(fields), has_commands="blue_command" in fields or "blue_tactic" in fields,
                 has_poses=any(name in fields for name in POSE_FIELDS),
                 has_contact=any(name in fields for name in CONTACT_FIELDS), skipped_lines=skipped)
    if not rounds:
        log.notes.append("no step records in this file: it is not a simulator play log")
    elif not log.has_commands:
        log.notes.append("the step records carry no blue_command or blue_tactic: there is no simulated command to compare")
    if skipped:
        log.notes.append(f"{skipped} line(s) were not step or reset records and were skipped")
    return log


# ----------------------------------------------------------------------------------------------
# Command timelines


@dataclass
class CommandSegment:
    start_s: float
    end_s: float
    label: str                              # direction class (see DRIVE_CLASSES)
    left: float | None
    right: float | None
    source: str                             # "sim" | "controller" | "pc_sent"
    detail: str = ""

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s


def command_timeline(sim_round: SimRound) -> list[CommandSegment]:
    """Run-length segments of the simulated blue command, in simulated time.

    A step record's `time_s` is the simulated time after that step, so its command covered
    [time_s - step, time_s]."""
    segments: list[CommandSegment] = []
    for step in sim_round.steps:
        command = step.get("blue_command")
        tactic = step.get("blue_tactic")
        if command is None and tactic is None:
            continue
        end = float(step["time_s"])
        start = end - sim_round.step_s
        if command is not None and len(command) == 2:
            left, right = float(command[0]), float(command[1])
            label = drive_class(left, right, SIM_FORWARD_SIGN, deadband=0.02)
        else:
            left = right = None
            label = TACTIC_CLASS.get(str(tactic), str(tactic))
        detail = str(tactic) if tactic is not None else label
        last = segments[-1] if segments else None
        if last is not None and (last.detail, last.left, last.right) == (detail, left, right) and abs(last.end_s - start) < 1e-6:
            last.end_s = end
        else:
            segments.append(CommandSegment(start, end, label, left, right, "sim", detail))
    return segments


def _held_segments(records: list[dict[str, Any]], hold_s: float, source: str, wheels) -> list[CommandSegment]:
    segments: list[CommandSegment] = []
    for i, record in enumerate(records):
        start = float(record["t_s"])
        end = start + hold_s
        if i + 1 < len(records):
            end = min(end, float(records[i + 1]["t_s"]))
        if end <= start:
            continue
        try:
            left, right, detail = wheels(record)
        except (KeyError, TypeError, ValueError):
            continue
        label = drive_class(left, right, CONTROLLER_FORWARD_SIGN, CONTROLLER_SCALE)
        last = segments[-1] if segments else None
        if last is not None and (last.left, last.right, last.detail) == (left, right, detail) and abs(last.end_s - start) < 1e-6:
            last.end_s = end
        else:
            segments.append(CommandSegment(start, end, label, left, right, source, detail))
    return segments


def controller_timeline(round_data, hold_s: float = CONTROLLER_HOLD_S) -> list[CommandSegment]:
    """Motor outputs the CONTROLLER REPORTED, in round time. Where no report is fresh there is no segment."""
    def wheels(record):
        return (float(record["left"]), float(record["right"]),
                f"act={record.get('active')} armed={int(bool(record.get('armed')))}")
    return _held_segments(round_data.controller_status, hold_s, "controller", wheels)


def pc_sent_timeline(round_data, hold_s: float = PC_SENT_HOLD_S) -> list[CommandSegment]:
    """Commands the PC client SENT, in round time. Sent is not applied."""
    def wheels(record):
        left, right = pc_mix(float(record["throttle"]), float(record["steer"]))
        return left, right, f"throttle={record['throttle']} steer={record['steer']}"
    return _held_segments(round_data.pc_sent, hold_s, "pc_sent", wheels)


def round_start_times(round_data) -> list[float]:
    """`t_s` of every operator ROUND START marker in the round."""
    return [float(event["t_s"]) for event in round_data.events if event.get("kind") == "round_start"]


def _labels_at(segments: list[CommandSegment], times: np.ndarray) -> np.ndarray:
    out = np.full(len(times), "", dtype=object)
    for segment in segments:
        out[(times >= segment.start_s) & (times < segment.end_s)] = segment.label
    return out


# ----------------------------------------------------------------------------------------------
# Command agreement


@dataclass
class CommandAgreement:
    """Did the commands match over time? Says nothing about how the robot moved."""

    real_source: str                        # human-readable name of the recorded stream
    available: bool
    reason: str = ""
    compared_s: float = 0.0                 # both sides known and comparable
    agree_s: float = 0.0
    real_unknown_s: float = 0.0             # no fresh record in the round
    outside_sim_s: float = 0.0              # window time the simulated round does not cover
    not_comparable_s: float = 0.0           # after the first simulated contact
    disagreements: list[tuple[float, float, str, str]] = field(default_factory=list)   # (start, end, sim, real), round time

    @property
    def share(self) -> float | None:
        return self.agree_s / self.compared_s if self.compared_s > 0 else None

    def text(self) -> str:
        if not self.available:
            return f"not available: {self.reason}"
        if self.compared_s <= 0:
            return (f"nothing to compare in this window: recorded command unknown for {self.real_unknown_s:.2f} s, "
                    f"simulated round does not cover {self.outside_sim_s:.2f} s")
        parts = [f"same direction class for {self.agree_s:.2f} s of {self.compared_s:.2f} s compared ({100 * self.share:.0f}%)"]
        if self.real_unknown_s > 0:
            parts.append(f"recorded command unknown for {self.real_unknown_s:.2f} s")
        if self.outside_sim_s > 0:
            parts.append(f"{self.outside_sim_s:.2f} s outside the simulated round")
        if self.not_comparable_s > 0:
            parts.append(f"{self.not_comparable_s:.2f} s not comparable (after simulated contact)")
        return "; ".join(parts)


def command_agreement(sim_segments: list[CommandSegment], real_segments: list[CommandSegment], real_source: str,
                      offset_s: float, t_start: float, t_end: float, not_comparable_after_s: float | None = None,
                      step_s: float = 0.01) -> CommandAgreement:
    """Compare direction classes on a fine grid over [t_start, t_end] (round time).

    `offset_s` is the round time at which simulated time 0 happens: t_round = t_sim + offset_s."""
    if not sim_segments:
        return CommandAgreement(real_source, False, "the simulator log has no blue command")
    if not real_segments:
        return CommandAgreement(real_source, False, f"the round has no {real_source} records")
    if not t_end > t_start:
        return CommandAgreement(real_source, False, "the window is empty")
    times = np.arange(t_start, t_end, step_s) + step_s / 2.0
    sim = _labels_at(sim_segments, times - offset_s)
    real = _labels_at(real_segments, times)
    has_sim, has_real = sim != "", real != ""
    blocked = np.zeros(len(times), bool) if not_comparable_after_s is None else times >= not_comparable_after_s
    compared = has_sim & has_real & ~blocked
    agree = compared & (sim == real)
    result = CommandAgreement(real_source, True, compared_s=float(compared.sum() * step_s), agree_s=float(agree.sum() * step_s),
                              real_unknown_s=float((~has_real).sum() * step_s), outside_sim_s=float((~has_sim).sum() * step_s),
                              not_comparable_s=float((blocked & has_sim).sum() * step_s))
    differ = compared & ~agree
    start = None
    for i in range(len(times) + 1):
        same_run = (i < len(times) and differ[i] and start is not None
                    and (sim[i], real[i]) == (sim[start], real[start]))
        if start is not None and not same_run:
            result.disagreements.append((float(times[start] - step_s / 2), float(times[i - 1] + step_s / 2), str(sim[start]), str(real[start])))
            start = None
        if i < len(times) and differ[i] and start is None:
            start = i
    return result


# ----------------------------------------------------------------------------------------------
# Measured motion (camera pose track of a recorded round)


@dataclass
class MeasuredMotion:
    available: bool
    reason: str = ""
    track: str = ""                         # "live" or "offline"
    track_note: str = ""
    t_start: float = 0.0
    t_end: float = 0.0
    distance_m: float | None = None         # path length inside valid runs, jitter below JITTER_STEP_M ignored
    straight_line_m: float | None = None    # sum over valid runs of start-to-end displacement
    heading_change_deg: float | None = None  # net change inside valid runs, + = counter-clockwise
    known_s: float = 0.0
    unknown_s: float = 0.0                  # window time with no usable pose; nothing is assumed about it
    runs: int = 0
    tag_switches: int = 0
    path: list[np.ndarray] = field(default_factory=list)       # one (n, 2) array per valid run

    def text(self) -> str:
        if not self.available:
            return self.reason
        text = (f"{self.distance_m:.3f} m travelled, heading {self.heading_change_deg:+.1f} deg, over {self.known_s:.2f} s "
                f"of valid pose in {self.runs} run(s) ({self.track_note})")
        if self.unknown_s > 0.005:
            text += f"; position UNKNOWN for {self.unknown_s:.2f} s of the window (not bridged, not counted)"
        if self.tag_switches:
            text += f"; {self.tag_switches} marker-switch step(s) excluded"
        return text


def pick_pose_track(round_data, track: str = "auto"):
    """The pose track to measure from, or None. `auto` prefers the track estimated during the round."""
    candidates = {"live": round_data.pose_live, "offline": round_data.pose_offline}
    order = ("live", "offline") if track == "auto" else (track,)
    for name in order:
        chosen = candidates.get(name)
        if chosen is not None and len(chosen) and bool(np.any(chosen.valid)):
            return chosen
    return None


def track_description(track) -> str:
    if track.source == "live":
        return "LIVE camera track, estimated during the round"
    return "OFFLINE camera track, reconstructed from the recorded video after the round"


def _path_length(points: np.ndarray, jitter_m: float) -> float:
    anchor, total = points[0], 0.0
    for point in points[1:]:
        step = float(np.hypot(*(point - anchor)))
        if step >= jitter_m:
            total += step
            anchor = point
    return total + float(np.hypot(*(points[-1] - anchor)))


def measured_motion(round_data, t_start: float, t_end: float, track: str = "auto", max_step_s: float = POSE_MAX_STEP_S,
                    jitter_m: float = JITTER_STEP_M) -> MeasuredMotion:
    """Distance travelled and net heading change inside [t_start, t_end], from valid camera poses only.

    A run is a stretch of consecutive valid poses with no invalid row, no marker switch and no time
    step longer than `max_step_s`. Distance and heading are summed inside runs. Whatever happened
    between runs is unknown and is reported as unknown time."""
    chosen = pick_pose_track(round_data, track)
    if chosen is None:
        return MeasuredMotion(False, NO_POSE, t_start=t_start, t_end=t_end)
    if not t_end > t_start:
        return MeasuredMotion(False, "not available: the window is empty", t_start=t_start, t_end=t_end)
    result = MeasuredMotion(True, track=chosen.source, track_note=track_description(chosen), t_start=t_start, t_end=t_end,
                            distance_m=0.0, straight_line_m=0.0, heading_change_deg=0.0)
    rows = np.flatnonzero((chosen.t >= t_start) & (chosen.t <= t_end))
    runs: list[list[int]] = []
    current: list[int] = []
    for i in rows:
        finite = bool(chosen.valid[i]) and math.isfinite(chosen.x[i]) and math.isfinite(chosen.y[i])
        switch = bool(chosen.tag_switch[i])
        too_far = bool(current) and chosen.t[i] - chosen.t[current[-1]] > max_step_s
        if not finite or switch or too_far:
            if current:
                runs.append(current)
                if finite and switch:
                    result.tag_switches += 1        # the step into this row is a marker-offset effect, not movement
            current = []
            if not finite:
                continue
        current.append(int(i))
    if current:
        runs.append(current)
    for run in runs:
        points = np.column_stack([chosen.x[run], chosen.y[run]])
        result.path.append(points)
        if len(run) < 2:
            continue
        result.runs += 1
        result.known_s += float(chosen.t[run[-1]] - chosen.t[run[0]])
        result.distance_m += _path_length(points, jitter_m)
        result.straight_line_m += float(np.hypot(*(points[-1] - points[0])))
        headings = chosen.heading_deg[run]
        result.heading_change_deg += float(sum(wrap_deg(b - a) for a, b in zip(headings[:-1], headings[1:])
                                               if math.isfinite(a) and math.isfinite(b)))
    result.unknown_s = max(0.0, (t_end - t_start) - result.known_s)
    if result.runs == 0:
        return MeasuredMotion(False, f"not available: no valid pose inside this window ({result.track_note})",
                              track=chosen.source, track_note=result.track_note, t_start=t_start, t_end=t_end,
                              unknown_s=t_end - t_start)
    return result


# ----------------------------------------------------------------------------------------------
# Simulated motion


@dataclass
class SimulatedMotion:
    available: bool
    source: str = ""                        # "simulator log poses" | "typed by the user"
    reason: str = ""
    distance_m: float | None = None
    heading_change_deg: float | None = None

    def text(self) -> str:
        if not self.available:
            return self.reason
        distance = "distance not given" if self.distance_m is None else f"{self.distance_m:.3f} m travelled"
        heading = "heading not given" if self.heading_change_deg is None else f"heading {self.heading_change_deg:+.1f} deg"
        return f"{distance}, {heading} ({self.source})"


def typed_simulated_motion(distance_m: float | None, heading_change_deg: float | None) -> SimulatedMotion:
    if distance_m is None and heading_change_deg is None:
        return SimulatedMotion(False, reason="not available: no simulated values typed")
    return SimulatedMotion(True, "typed by the user", distance_m=distance_m, heading_change_deg=heading_change_deg)


def simulated_motion(sim_log: SimLog, sim_round: SimRound, sim_t_start: float, sim_t_end: float) -> SimulatedMotion:
    """Simulated distance and heading change over a window of SIMULATED time, if the log carries poses."""
    if not sim_log.has_poses:
        return SimulatedMotion(False, reason="not available: " + sim_log.pose_note)
    poses = [(float(step["time_s"]), step["blue_pose"]) for step in sim_round.steps
             if sim_t_start <= float(step["time_s"]) <= sim_t_end and isinstance(step.get("blue_pose"), list)
             and len(step["blue_pose"]) >= 3]
    if len(poses) < 2:
        return SimulatedMotion(False, reason="not available: fewer than two simulated poses inside this window")
    points = np.array([[pose[0], pose[1]] for _, pose in poses], float)
    yaw = [math.degrees(float(pose[2])) for _, pose in poses]
    return SimulatedMotion(True, "simulator log poses", distance_m=_path_length(points, 0.0),
                           heading_change_deg=float(sum(wrap_deg(b - a) for a, b in zip(yaw[:-1], yaw[1:]))))


# ----------------------------------------------------------------------------------------------
# Contact


@dataclass
class ContactInfo:
    detectable: bool
    first_contact_sim_s: float | None
    note: str


def first_simulated_contact(sim_log: SimLog, sim_round: SimRound) -> ContactInfo:
    if not sim_log.has_contact:
        return ContactInfo(False, None, sim_log.contact_note)
    for step in sim_round.steps:
        if any(bool(step.get(name)) for name in CONTACT_FIELDS):
            when = float(step["time_s"]) - sim_round.step_s
            return ContactInfo(True, when, f"first simulated contact at simulated t = {when:.2f} s")
    return ContactInfo(True, None, "the log carries a contact flag and it is never set in this round")


# ----------------------------------------------------------------------------------------------
# The comparison


@dataclass
class MotionAgreement:
    """Did the robot move the same way? Separate from whether the commands matched."""

    measured: MeasuredMotion
    simulated: SimulatedMotion
    comparable: bool = True
    reason: str = ""
    distance_difference_m: float | None = None          # measured - simulated
    heading_difference_deg: float | None = None         # measured - simulated

    def text(self) -> str:
        if not self.comparable:
            return self.reason
        if not self.measured.available:
            return self.measured.reason
        if not self.simulated.available:
            return f"measured only ({self.measured.text()}); simulated {self.simulated.reason}"
        parts = []
        if self.distance_difference_m is not None:
            parts.append(f"distance: measured {self.measured.distance_m:.3f} m, simulated {self.simulated.distance_m:.3f} m, "
                         f"difference {self.distance_difference_m:+.3f} m")
        if self.heading_difference_deg is not None:
            parts.append(f"heading: measured {self.measured.heading_change_deg:+.1f} deg, simulated "
                         f"{self.simulated.heading_change_deg:+.1f} deg, difference {self.heading_difference_deg:+.1f} deg")
        text = "; ".join(parts) + f" (measured from the {self.measured.track_note}; simulated values {self.simulated.source})"
        if self.measured.unknown_s > 0.005:
            text += (f"; measured position UNKNOWN for {self.measured.unknown_s:.2f} s of the window, so the measured "
                     f"figures cover only the {self.measured.known_s:.2f} s with a valid pose, not the whole window")
        return text


def motion_agreement(measured: MeasuredMotion, simulated: SimulatedMotion, not_comparable_reason: str = "") -> MotionAgreement:
    """Measured against simulated motion. Needs no simulator log when the simulated values are typed."""
    motion = MotionAgreement(measured, simulated)
    if not_comparable_reason:
        motion.comparable, motion.reason = False, not_comparable_reason
    elif measured.available and simulated.available:
        if simulated.distance_m is not None:
            motion.distance_difference_m = measured.distance_m - simulated.distance_m
        if simulated.heading_change_deg is not None:
            motion.heading_difference_deg = measured.heading_change_deg - simulated.heading_change_deg
    return motion


@dataclass
class Comparison:
    offset_s: float                         # round time of simulated t = 0
    t_start: float                          # window, round time
    t_end: float
    command_vs_controller: CommandAgreement
    command_vs_pc_sent: CommandAgreement
    motion: MotionAgreement
    contact: ContactInfo
    not_comparable_after_s: float | None    # round time of the first simulated contact, if one is known
    notes: list[str] = field(default_factory=list)

    def command_text(self) -> str:
        return (f"vs controller-reported L/R: {self.command_vs_controller.text()}  |  "
                f"vs PC-sent commands: {self.command_vs_pc_sent.text()}")

    def motion_text(self) -> str:
        return self.motion.text()

    def warning_text(self) -> str:
        if self.not_comparable_after_s is not None:
            return (f"After round t = {self.not_comparable_after_s:.2f} s (simulated t = "
                    f"{self.not_comparable_after_s - self.offset_s:.2f} s): {NOT_COMPARABLE}.")
        return self.contact.note[:1].upper() + self.contact.note[1:] + "."


def compare(sim_log: SimLog, sim_round: SimRound, round_data, offset_s: float, t_start: float, t_end: float,
            simulated: SimulatedMotion | None = None, track: str = "auto") -> Comparison:
    """Compare one simulated round with a recorded round over [t_start, t_end] (round time).

    `offset_s` places simulated t = 0 on the round's timeline (for example the ROUND START marker).
    `simulated` supplies typed simulated motion; without it the log's poses are used if it has any."""
    contact = first_simulated_contact(sim_log, sim_round)
    blocked_after = None if contact.first_contact_sim_s is None else contact.first_contact_sim_s + offset_s
    sim_segments = command_timeline(sim_round)
    controller = command_agreement(sim_segments, controller_timeline(round_data), "controller-reported L/R",
                                   offset_s, t_start, t_end, blocked_after)
    pc_sent = command_agreement(sim_segments, pc_sent_timeline(round_data), "PC-sent command",
                                offset_s, t_start, t_end, blocked_after)
    measured = measured_motion(round_data, t_start, t_end, track)
    if simulated is None:
        simulated = simulated_motion(sim_log, sim_round, t_start - offset_s, t_end - offset_s)
    blocked_reason = ""
    if blocked_after is not None and blocked_after < t_end:
        blocked_reason = (f"{NOT_COMPARABLE}: this window runs past the first simulated contact at round t = "
                          f"{blocked_after:.2f} s; end the window before it")
    motion = motion_agreement(measured, simulated, blocked_reason)
    notes = ["Command agreement compares direction classes only (stop / forward / reverse / turn / arc); magnitudes are in "
             "different units and are not compared. It says nothing about how far or how fast the robot moved.",
             "The controller's L/R are what it REPORTED applying; PC-sent commands were sent, not confirmed applied."]
    if sim_round.end_reason:
        notes.append(f"The simulated round ended with '{sim_round.end_reason}' at simulated t = {sim_round.end_s:.2f} s.")
    if not round_start_times(round_data):
        notes.append("The round has no ROUND START marker: the time offset is whatever was typed.")
    return Comparison(offset_s, t_start, t_end, controller, pc_sent, motion, contact, blocked_after, notes)
