"""Pure helpers behind the AI / MODEL and MATCH screens. No Qt widgets, no camera, port, socket or torch import.

Everything here turns data the Command Centre already has (a tracking result, a link-service snapshot, a model's identity)
into the words, tables and booleans the two screens show, or adapts one backend to another:

* ``collect_inputs`` / ``describe_observation``   latest tracking result + IMU telemetry -> ``ObservationInputs`` -> the
  19-value table with the live status of every group (VALID / MISSING / STALE, age, source, reason).
* ``mapping_self_check``                           the SOFTWARE self-check of ``command_mapping`` (every tactic, the caps,
  invalid input -> stop, stale -> stop, neutral-first). It is never a hardware result.
* ``decision_feed``                                a ``PolicyDecision`` -> the dictionary ``ModeManager.submit`` understands.
* ``build_match_inputs``                           everything the Match screen knows -> ``match_controller.MatchInputs``.
* ``ServiceModes``                                 the ModeManager stand-in ``MatchController`` talks to. The real ModeManager
  lives on the link service's worker thread and is not thread-safe, so every request is marshalled through the service.
* ``ai_readiness`` / ``match_readiness`` / ``autonomous_readiness``   the readiness levels.

Nothing here sends anything. A decision is data; only ``command_mapping`` plus the gated Wi-Fi sender can turn it into a
command, and that needs the ModeManager's gate, which needs hardware acceptance that does not exist in this repository.
Evidence from anything in this module is SOFTWARE or SIMULATED, never HARDWARE.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

from control_center.backend import tracking_model as tm
from control_center.backend.command_mapping import (FIRMWARE_PC_LIMIT, LIMITS, POLICY_TACTICS, Limits, MapperState, describe_mapping,
                                                    firmware_drive, map_tactic)
from control_center.backend.control_modes import Mode
from control_center.backend.match_controller import MatchInputs
from control_center.backend.observation_builder import (SOURCE_LIVE, SOURCE_SIMULATED, ImuInput, ObservationInputs, ObservationResult,
                                                        OpponentInput, OwnPoseInput, build_observation, default_contract,
                                                        origin_from_capture_state)
from control_center.backend.readiness import BLOCKED, FAILED, NOT_CONFIGURED, READY, WARNING, Readiness
from control_center.tools.chassis_tracker import TrackerSettings
from control_center.tools.observation_contract import (GROUP_IMU, GROUP_OPPONENT, GROUP_ORDER, GROUP_OWN, GROUP_PREVIOUS,
                                                       RECOMMENDED_CHECKPOINT)

RESULT_STALE_S = 2.0                      # a tracking result older than this is "no result" (the tracking tab uses the same)
CAMERA_FRAME_STALE_S = 2.0
HEARTBEAT_FRESH_S = 0.5
DEFAULT_MODEL_PATH = "runs/recovery-challenger-v2-seed1/final_model.zip"

# ---- statements the screens must show plainly ------------------------------------------------------------------------

MODEL_ORIGIN_TEXT = (
    "WHAT THIS CHECKPOINT IS (finding of the policy audit): behaviour cloning followed by 4 DAgger rounds of a simple pursuit "
    "controller (turn toward the opponent, advance when within 12 degrees), saved as a Stable-Baselines3 PPO object so that "
    "PPO.load works. It was NOT trained by PPO: the value head is untouched. It was trained and evaluated only in the simulator, "
    "whose observation contains simulator truth. Every result about it so far is SIMULATOR evidence; nothing is a hardware result. "
    "(docs/policy-observation-schema.md section 1)")
MODEL_UNAUDITED_TEXT = ("This is not the audited checkpoint (runs/recovery-challenger-v2-seed1/final_model.zip). Its origin and training "
                        "method are UNKNOWN to the Command Centre: nothing is claimed about it.")
IMU_BLOCK_TEXT = ("The baseline firmware sends no IMU data, so live inference is BLOCKED. Nothing is invented: no zero, rest or "
                  "simulated IMU value is ever substituted for a real one.")
CONSTANT_TEXT = ("Channels 6-12 (previous tactic) are the constants the training environment defined: exactly 0.0 "
                 "(masked_zero_v1; autobot/sim/env.py:137-139). They are not measurements and not the Command Centre's last action.")
MOTION_CHECK_NOTE = ("The 40 % motion-check values (+102 / -71 / +71 / -66) are the SUPERVISED motion-check scale, not the operating "
                     "limit. The operating limit is max_command 60 of 255 until supervised tests say otherwise.")
DIRECTION_NOTE = ("Direction signs (positive steer = RIGHT, left = throttle + steer) are a software convention taken from the firmware "
                  "source. They must be verified physically, wheels raised, before any autonomous use. Nothing here verifies them.")
SOFTWARE_STOP_WIFI_TEXT = ("A software stop cannot work when Wi-Fi is down: the stop command travels over the same Wi-Fi link. The "
                           "robot's own backstops are the controller firmware, which stops the motors 250 ms after the last "
                           "command, and the RadioMaster (ARM switch off, or MODE to RADIO).")
NOT_TRANSMITTED_TEXT = "not transmitted"
DIAGNOSTIC_LABEL = "DIAGNOSTIC (not valid for control)"
LIVE_LABEL = "LIVE (valid for control)"

GROUP_LABELS = {GROUP_IMU: "IMU", GROUP_PREVIOUS: "Previous action", GROUP_OWN: "Own pose", GROUP_OPPONENT: "Opponent"}
STATUS_WORDS = {"OK": "VALID", "MISSING": "MISSING", "STALE": "STALE"}


# ================================================================ the observation, from the latest tracking result


def lost_after_s(tracking: Any) -> float:
    getter = getattr(tracking, "_lost_after_s", None)
    try:
        return float(getter()) if getter is not None else TrackerSettings().lost_after_s
    except Exception:
        return TrackerSettings().lost_after_s


def ensure_tracking_active(tracking: Any) -> None:
    """The TRACKING tab pauses its vision worker whenever it is hidden. The Match and AI tabs need its results, so they keep it
    resumed (idempotent; the tracking screen's own ``showEvent`` does the same). Never starts anything on a closed screen."""
    worker = getattr(tracking, "worker", None)
    if worker is None or getattr(tracking, "_closed", False):
        return
    try:
        worker.resume()
        if not worker.isRunning():
            worker.start()
    except Exception:
        pass


@dataclass(frozen=True)
class LiveInputs:
    inputs: ObservationInputs
    own: dict[str, Any] | None                 # own-robot report aged to now (None: no result)
    opponent: dict[str, Any] | None
    result_age_s: float | None
    camera_state: str
    imu_note: str                              # why the IMU is or is not usable, from the telemetry's own report
    notes: tuple[str, ...]


def collect_inputs(hub: Any, tracking: Any, now: float, *, contract: Any = None) -> LiveInputs:
    """Latest tracking result + telemetry -> ObservationInputs, stamped on the clock of the tracking result (time.monotonic).

    A result older than 2 s counts as no result; the own-robot report is aged to ``now`` exactly as the TRACKING tab does, so a
    pose from frames that stopped arriving turns STALE and then LOST, never "the last position"."""
    capture = getattr(hub, "capture", None)
    camera_state = str(getattr(capture, "state", "") or "")
    options = getattr(hub, "options", None)
    notes: list[str] = []
    result = getattr(tracking, "_result", None) if tracking is not None else None
    result_age = None
    if result is not None:
        result_age = max(0.0, now - result.done_at)
        if result_age > RESULT_STALE_S or getattr(result, "epoch", 0) != getattr(tracking, "_epoch", getattr(result, "epoch", 0)):
            notes.append(f"the newest tracking result is {result_age:.1f} s old (or from an earlier calibration): ignored")
            result = None
    own_aged = None
    own_input = opponent_input = None
    if result is not None:
        own_aged = tm.age_own_report(result.own, result_age or 0.0, float(result.limits.get("predict_hold_s", 0.5)), lost_after_s(tracking))
        if own_aged is not None:
            heading = own_aged.get("heading_deg")
            own_input = OwnPoseInput(own_aged.get("x_m"), own_aged.get("y_m"), None if heading is None else math.radians(heading),
                                     float(result.t), str(own_aged.get("state", "lost")), str(own_aged.get("reason", "") or ""))
        opp = result.opponent
        if opp is not None:
            opponent_input = OpponentInput(opp.get("x_m"), opp.get("y_m"), float(result.t), str(opp.get("state", "not_selected")),
                                           str(opp.get("reason", "") or ""))
    elif tracking is None:
        notes.append("there is no tracking screen in this session")
    else:
        notes.append("the tracking vision has produced no result yet")

    telemetry = getattr(tracking, "telemetry", None) if tracking is not None else None
    imu_input = None
    imu_note = "no controller telemetry source"
    if telemetry is not None:
        replay = bool(getattr(options, "mock", False)) or getattr(options, "video", None) is not None
        origin = SOURCE_SIMULATED if replay else getattr(telemetry, "origin", SOURCE_SIMULATED)
        try:
            reading = telemetry.latest(now)
            report = telemetry.report(now)
            imu_note = "; ".join(str(report[key]) for key in ("present_detail", "fresh_detail", "mounting_detail") if report.get(key))
            if reading is not None:
                imu_input = ImuInput(tuple(reading.gyro_rad_s), tuple(reading.accel_m_s2), float(reading.t), origin, True)
        except Exception as error:
            imu_note = f"telemetry could not be read: {type(error).__name__}: {error}"

    blocked: list[str] = []
    if getattr(hub, "world_coordinates_trusted", None) is not True:
        blocked.append("the arena calibration is not confirmed trusted this run")
    arena = getattr(getattr(tracking, "config", None), "arena", None)
    if arena is not None:
        try:
            c = contract or default_contract()
            ok, detail = tm.frame_compatibility(arena, str(getattr(hub, "profile_name", "")), (c.arena_width_m, c.arena_height_m))
            if not ok:
                blocked.append(detail)
        except Exception as error:
            blocked.append(f"the arena frame could not be compared with the checkpoint's ({type(error).__name__})")
    elif tracking is not None:
        blocked.append("the tracking screen reports no arena geometry")
    origin = origin_from_capture_state(camera_state)
    return LiveInputs(ObservationInputs(origin, own_input, opponent_input, imu_input, None, tuple(blocked)), own_aged,
                      None if result is None else result.opponent, result_age, camera_state, imu_note, tuple(notes))


@dataclass(frozen=True)
class GroupRow:
    group: str
    label: str
    indices: str
    word: str                       # VALID / MISSING / STALE
    age_ms: float | None
    source: str                     # LIVE / SIMULATED / REPLAY / CONSTANT / none
    reason: str


@dataclass(frozen=True)
class ChannelRow:
    index: int
    name: str
    group: str
    value: float | None
    constant: bool
    word: str
    age_ms: float | None
    source: str
    reason: str


@dataclass(frozen=True)
class ObservationView:
    result: ObservationResult | None
    groups: tuple[GroupRow, ...]
    channels: tuple[ChannelRow, ...]
    complete: bool                  # all 19 values exist (every group valid and fresh)
    valid_for_control: bool         # complete AND source LIVE AND a real fresh IMU AND no blocker
    source: str
    label: str                      # LIVE / DIAGNOSTIC / NO OBSERVATION
    blocked_by: tuple[str, ...]
    statement: str
    imu_text: str
    notes: tuple[str, ...] = ()
    error: str = ""

    def group(self, name: str) -> GroupRow | None:
        return next((row for row in self.groups if row.group == name), None)

    @property
    def imu_missing(self) -> bool:
        row = self.group(GROUP_IMU)
        return row is None or row.word != "VALID"


def _age_ms(age_s: float | None) -> float | None:
    return None if age_s is None else round(1000.0 * age_s, 1)


def describe_observation(result: ObservationResult | None, contract: Any, imu_note: str = "", notes: Iterable[str] = (),
                         error: str = "") -> ObservationView:
    """The per-group and per-channel table for an ObservationResult (None: nothing could be built)."""
    if result is None:
        return ObservationView(None, (), (), False, False, SOURCE_SIMULATED, "NO OBSERVATION", (error,) if error else (),
                               error or "No observation could be built.", IMU_BLOCK_TEXT, tuple(notes), error)
    groups: list[GroupRow] = []
    for name in GROUP_ORDER:
        status = result.groups.get(name)
        if status is None:
            continue
        reason = status.reason
        if name == GROUP_IMU and not status.ok:
            if status.age_s is None and status.status == "MISSING":
                reason = IMU_BLOCK_TEXT + (f" ({imu_note})" if imu_note else "")
            elif imu_note:
                reason = f"{reason} ({imu_note})"
        groups.append(GroupRow(name, GROUP_LABELS.get(name, name), status.indices, STATUS_WORDS.get(status.status, status.status),
                               _age_ms(status.age_s), status.origin or "none", reason))
    by_group = {row.group: row for row in groups}
    channels: list[ChannelRow] = []
    for spec in contract.fields:
        row = by_group.get(spec.group)
        constant = bool(row and row.source == "CONSTANT")
        value = None
        if result.values is not None:
            value = float(result.values[spec.index])
        elif constant:
            value = 0.0
        channels.append(ChannelRow(spec.index, spec.name, spec.group, value, constant, row.word if row else "MISSING",
                                   row.age_ms if row else None, row.source if row else "none", row.reason if row else ""))
    complete = result.values is not None
    label = "LIVE" if result.valid_for_control else ("DIAGNOSTIC" if complete else "NO OBSERVATION")
    if result.valid_for_control:
        statement = "A complete, fresh LIVE observation exists (valid for control)."
    elif complete:
        statement = f"A complete observation exists but its source is {result.source}: DIAGNOSTIC only, never valid for control."
    else:
        statement = ("No observation vector: " + "; ".join(f"{r.label}: {r.word}" for r in groups if r.word != "VALID")
                     + ". Nothing is invented to fill the gap.")
    return ObservationView(result, tuple(groups), tuple(channels), complete, bool(result.valid_for_control), result.source, label,
                           tuple(result.blocked_by), statement, IMU_BLOCK_TEXT, tuple(notes), error)


def observation_view(hub: Any, tracking: Any, now: float, contract: Any = None) -> tuple[ObservationView, LiveInputs | None]:
    """Never raises: a problem is an ObservationView that says so."""
    try:
        contract = contract or default_contract()
        live = collect_inputs(hub, tracking, now, contract=contract)
        result = build_observation(live.inputs, now, contract)
        return describe_observation(result, contract, live.imu_note, live.notes), live
    except Exception as error:
        return describe_observation(None, None, error=f"observation could not be built: {type(error).__name__}: {error}"), None


# ================================================================ the command mapping self-check (SOFTWARE evidence)


@dataclass(frozen=True)
class CheckLine:
    name: str
    ok: bool
    detail: str


@dataclass(frozen=True)
class MappingCheck:
    passed: bool
    lines: tuple[CheckLine, ...]
    limits: Limits
    evidence: str = "SOFTWARE self-check (no robot involved)"

    @property
    def failures(self) -> tuple[CheckLine, ...]:
        return tuple(line for line in self.lines if not line.ok)

    def summary(self) -> str:
        word = "PASS" if self.passed else "FAIL"
        return f"{word}: {sum(1 for l in self.lines if l.ok)} of {len(self.lines)} software checks. {self.evidence}. Not a hardware result."


def mapping_self_check(limits: Limits = LIMITS) -> MappingCheck:
    """Every policy action, the hard cap, invalid input -> stop, stale or future decision -> stop, neutral-first. Pure software."""
    lines: list[CheckLine] = []

    def add(name: str, ok: bool, detail: str = "") -> None:
        lines.append(CheckLine(name, bool(ok), detail))

    try:
        rows = describe_mapping(limits)
    except Exception as error:
        return MappingCheck(False, (CheckLine("mapping table", False, f"{type(error).__name__}: {error}"),), limits)
    add("the table has the five policy tactics, in order", [r["tactic"] for r in rows] == list(POLICY_TACTICS), str([r["tactic"] for r in rows]))
    unavailable = [r["tactic"] for r in rows if r["throttle"] is None]
    add("every tactic can be mapped (the simulator's tactic table is readable)", not unavailable, ", ".join(unavailable))
    if unavailable:
        return MappingCheck(False, tuple(lines), limits)
    cap = min(limits.max_command, FIRMWARE_PC_LIMIT)
    now = 100.0
    plans = {}
    for row in rows:
        plan = map_tactic(row["tactic"], now, now, limits)
        plans[row["tactic"]] = plan
        add(f"{row['tactic']}: equals the table and stays within +/-{cap}",
            (plan.throttle, plan.steer) == (row["throttle"], row["steer"]) and abs(plan.throttle) <= cap and abs(plan.steer) <= cap,
            f"throttle {plan.throttle}, steer {plan.steer}")
    p = plans
    add("stop is 0,0", (p["stop"].throttle, p["stop"].steer) == (0, 0))
    add("advance is forward (+throttle), no steer", p["advance"].throttle > 0 and p["advance"].steer == 0)
    add("reverse is backward (-throttle), no steer", p["reverse"].throttle < 0 and p["reverse"].steer == 0)
    add("turn_left has negative steer (positive steer = RIGHT), no throttle", p["turn_left"].steer < 0 and p["turn_left"].throttle == 0)
    add("turn_right has positive steer, no throttle", p["turn_right"].steer > 0 and p["turn_right"].throttle == 0)
    drive = {t: firmware_drive(p[t].throttle, p[t].steer) for t in POLICY_TACTICS}
    add("the firmware mixer drives both wheels forward for advance and backward for reverse",
        drive["advance"][0] > 0 < drive["advance"][1] and drive["reverse"][0] < 0 > drive["reverse"][1], str(drive))
    add("the mixer turns the wheels in opposite directions for the two turns",
        drive["turn_left"][0] < 0 < drive["turn_left"][1] and drive["turn_right"][0] > 0 > drive["turn_right"][1])
    bad_inputs = (None, "bogus", "advance; rm", 99, -1, 1.5, float("nan"), float("inf"), True, [], {})
    leaks = [repr(b) for b in bad_inputs if not map_tactic(b, now, now, limits).is_stop]
    add("invalid input (None, unknown text, out-of-range or non-integer index, NaN, bool, list) gives a stop", not leaks, ", ".join(leaks))
    add("a stale decision gives a stop", map_tactic("advance", now, now - limits.max_age_s - 0.5, limits).is_stop)
    add("a decision from the future gives a stop", map_tactic("advance", now, now + 5.0, limits).is_stop)
    add("a non-finite time gives a stop", map_tactic("advance", float("nan"), now, limits).is_stop)
    wide = Limits(max_command=FIRMWARE_PC_LIMIT, operating_scale=1.0)
    over = [t for t in POLICY_TACTICS if max(abs(map_tactic(t, now, now, wide).throttle), abs(map_tactic(t, now, now, wide).steer)) > FIRMWARE_PC_LIMIT]
    add(f"even at full scale the cap holds: nothing above the firmware limit {FIRMWARE_PC_LIMIT}", not over, ", ".join(over))
    state = MapperState()
    first = map_tactic("advance", now, now, limits, state)
    second = map_tactic("reverse", now + 0.02, now + 0.02, limits, state)
    add("neutral-first: a forward-to-reverse flip outputs 0,0 first", second.is_stop, f"first {first.throttle},{first.steer}; then {second.throttle},{second.steer}")
    ok = all(line.ok for line in lines)
    return MappingCheck(ok, tuple(lines), limits)


# ================================================================ decisions -> the ModeManager


def decision_feed(decision: Any, result: ObservationResult | None, *, allow_tactic: bool = True) -> dict[str, Any]:
    """The dictionary ``ModeManager.submit`` reads: ``proposed`` (or None), per-group ``inputs`` with ages, the vector.

    ``allow_tactic=False`` drops the proposal (the manager then plans a stop): used in AUTONOMOUS for any decision that is not
    ``valid_for_control``. Nothing about the decision is invented: a blocked decision has ``proposed`` None."""
    inputs: dict[str, Any] = {}
    if result is not None:
        for name, status in result.groups.items():
            inputs[name] = {"valid": bool(status.ok), "age_ms": _age_ms(status.age_s), "reason": status.reason, "source": status.origin or ""}
    proposed = decision.proposal() if (decision is not None and allow_tactic) else None
    observation = None if decision is None or decision.observation is None else list(decision.observation)
    return {"proposed": proposed, "inputs": inputs, "observation": observation,
            "blocked_by": list(getattr(decision, "blocked_by", ()) or ()), "source": getattr(decision, "source", "SIMULATED"),
            "valid_for_control": bool(getattr(decision, "valid_for_control", False)) and allow_tactic,
            "error": getattr(decision, "error", ""), "policy_loaded": True,
            "mode": "DIAGNOSTIC" if not getattr(decision, "valid_for_control", False) else "LIVE",
            "output": "none: a decision is data; only the mode manager's gated path can send anything"}


# ================================================================ the ModeManager stand-in used by MatchController


class ServiceModes:
    """What ``MatchController`` needs of a ModeManager (``mode``, ``request``, ``submit``), over a ``LinkService``.

    The ModeManager belongs to the link service's worker thread, so nothing here touches it: requests are queued through the
    service's public methods and ``mode`` is read from the newest published snapshot (at most ~100 ms old). AI SHADOW and
    MANUAL are requested without waiting (both are safe states). AUTONOMOUS is the only request whose answer is needed
    synchronously (``MatchController.start_round`` decides a downgrade on it): it waits up to ``wait_s`` for the service's own
    verdict and treats silence as a refusal. With no service the mode is local and AUTONOMOUS is always refused."""

    def __init__(self, service: Any = None, *, wait_s: float = 1.0, drive: Callable[[], None] | None = None,
                 sleep: Callable[[float], None] = time.sleep, clock: Callable[[], float] = time.monotonic) -> None:
        self.service = service
        self.wait_s, self._drive, self._sleep, self._clock = wait_s, drive, sleep, clock
        self._local = Mode.MANUAL
        self.refusals: list[str] = []

    def snapshot(self) -> Any:
        try:
            return None if self.service is None else self.service.latest()
        except Exception:
            return None

    @property
    def mode(self) -> Mode:
        snap = self.snapshot()
        return self._local if snap is None else snap.mode.mode

    def request(self, mode: Mode, now: float | None = None, confirmation: Any = None) -> tuple[bool, list[str]]:
        service = self.service
        if service is None:
            if mode is Mode.AUTONOMOUS:
                return False, ["there is no link service in this session: AUTONOMOUS cannot be entered"]
            self._local = mode
            return True, []
        if mode is Mode.MANUAL:
            service.return_to_manual()
            return True, []
        if mode is Mode.AI_SHADOW:
            service.request_mode(Mode.AI_SHADOW)
            return True, []
        if not service.is_running() and self._drive is None:
            return False, ["the link service is not running, so AUTONOMOUS cannot be requested"]
        before = getattr(self.snapshot(), "last_request", None)
        service.request_mode(Mode.AUTONOMOUS, confirmation)
        end = self._clock() + self.wait_s
        while True:
            if self._drive is not None:
                self._drive()
            answer = getattr(self.snapshot(), "last_request", None)
            if answer is not None and answer is not before and answer.what == "mode AUTONOMOUS":
                reasons = [part for part in str(answer.text).split("; ") if part]
                return bool(answer.ok), ([] if answer.ok else (reasons or ["refused by the mode manager"]))
            if self._clock() >= end:
                return False, ["the link service did not answer the AUTONOMOUS request in time: treated as refused"]
            self._sleep(0.01)

    def submit(self, decision: Any, now: float | None = None, *, policy_loaded: bool | None = None, mock: bool | None = None,
               observation_at: float | None = None) -> None:
        if self.service is not None:
            self.service.submit_decision(decision, policy_loaded=policy_loaded, mock=mock, observation_at=observation_at)


# ================================================================ MatchInputs


def _gate_text(items: Iterable[Any]) -> list[str]:
    return [f"{item.name}: {item.reason}" for item in items if not item.ok]


def build_match_inputs(*, camera_state: str, camera_detail: str = "", camera_age_s: float | None, arena_trusted: bool | None,
                       arena_detail: str, own: dict[str, Any] | None, opponent: dict[str, Any] | None,
                       observation: ObservationView | None, model_loaded: bool, model_detail: str, mapping_validated: bool,
                       mapping_detail: str, snapshot: Any, controller_up: bool | None, controller_detail: str,
                       inference_error: str = "") -> MatchInputs:
    """Assemble the controller's inputs from what the screens know. Every field defaults to the refusing value."""
    camera_ok = camera_state in ("LIVE", "MOCK", "REPLAY") and camera_age_s is not None and camera_age_s <= CAMERA_FRAME_STALE_S
    kind = {"LIVE": "", "MOCK": " (MOCK camera: SIMULATED)", "REPLAY": " (REPLAY: not a live camera)"}.get(camera_state, "")
    cam_detail = (f"camera {camera_state}{kind}, newest frame {1000 * camera_age_s:.0f} ms old" if camera_ok else
                  f"camera {camera_state or 'unknown'}" + (f": {camera_detail}" if camera_detail else "") +
                  ("" if camera_age_s is None else f", newest frame {camera_age_s:.1f} s old") + ", not delivering frames")
    own_state = None if own is None else own.get("state")
    own_ok = own_state in ("measured", "predicted")
    opp_state = None if opponent is None else opponent.get("state")
    opp_ok = opp_state == "tracked"
    obs_ok = bool(observation and observation.valid_for_control)
    if observation is None:
        obs_detail = "no observation built"
    elif obs_ok:
        obs_detail = "complete, fresh and LIVE"
    else:
        obs_detail = "; ".join(observation.blocked_by[:4]) or observation.statement
    health = getattr(snapshot, "health", None)
    wifi_ok, wifi_detail = False, "no link service"
    if health is not None:
        state = getattr(health.state, "value", str(health.state))
        wifi_ok = (state == "CONNECTED" and health.heartbeat_age_s is not None and health.heartbeat_age_s < HEARTBEAT_FRESH_S
                   and health.protocol_ok is True and not getattr(snapshot, "simulated", False))
        wifi_detail = (f"link {state}" + ("" if health.heartbeat_age_s is None else f", heartbeat {1000 * health.heartbeat_age_s:.0f} ms old")
                       + (", SIMULATED link: not the robot" if getattr(snapshot, "simulated", False) else "")
                       + ("" if health.protocol_ok is True else f"; firmware/protocol not confirmed ({health.protocol_note})"))
    gate_open = bool(snapshot is not None and snapshot.mode.gate_open)
    closed = [] if snapshot is None else _gate_text(snapshot.gate)
    acceptance = None if snapshot is None else snapshot.acceptance
    safety_ok = bool(acceptance is not None and acceptance.complete and not snapshot.simulated)
    safety_detail = ("every acceptance step PASS from HARDWARE evidence" if safety_ok else
                     "; ".join(getattr(acceptance, "missing", ()) or ()) or "the hardware acceptance record is not complete (none exists yet)"
                     if acceptance is not None else "no acceptance record")
    return MatchInputs(
        camera_ok=camera_ok, camera_detail=cam_detail,
        arena_trusted=arena_trusted is True, arena_detail=arena_detail,
        own_valid=own_ok, own_detail=("own robot " + str(own_state).upper() if own is not None else "own robot not tracked (no result)")
        + ("" if own is None or own_ok else f": {own.get('reason', '')}"),
        opponent_valid=opp_ok, opponent_detail=("opponent TRACKED (image-based)" if opp_ok else
                                                 "opponent not selected" if opponent is None or opp_state == "not_selected" else
                                                 f"opponent {str(opp_state).upper()}: {opponent.get('reason', '')}"),
        observation_complete=obs_ok, observation_detail=obs_detail,
        model_loaded=model_loaded, model_detail=model_detail,
        mapping_validated=mapping_validated, mapping_detail=mapping_detail,
        wifi_ready=wifi_ok, wifi_detail=wifi_detail,
        authority_ok=gate_open, authority_detail=("every gate is open" if gate_open else "gate closed: " + "; ".join(closed or ["no link service"])),
        safety_done=safety_ok, safety_detail=safety_detail,
        manual_available=bool(controller_up), manual_detail=controller_detail,
        inference_error=inference_error)


# ================================================================ readiness


def ai_readiness(*, state: str, info: Any, error: str, view: ObservationView | None, loaded_at_text: str = "") -> Readiness:
    """NOT_CONFIGURED not loaded; FAILED incompatible or load error; WARNING loaded but required live inputs missing;
    READY only with a loaded model AND a complete, fresh, LIVE observation. BLOCKED is never used by this screen."""
    if state == "failed" or error:
        return Readiness(FAILED, "The model failed to load", error or "; ".join(getattr(info, "problems", ()) or ()), "Check the checkpoint path and library versions below.")
    if info is not None and not getattr(info, "compatible", True) and getattr(info, "problems", ()):
        return Readiness(FAILED, "The checkpoint is not compatible with this build", "; ".join(info.problems), "Choose the recommended checkpoint.")
    if state == "loading":
        return Readiness(NOT_CONFIGURED, "The model is loading", "Loading runs on a background thread (up to about a minute).", "Wait for the load to finish.")
    if state != "loaded":
        return Readiness(NOT_CONFIGURED, "The model is not loaded", "No model is in memory. AI SHADOW needs one.", "Press Load model on the AI / MODEL tab.")
    if view is not None and view.valid_for_control and view.complete:
        return Readiness(READY, "Model loaded and a complete LIVE observation is available", view.statement)
    missing = [f"{row.label}: {row.word}" for row in (view.groups if view else ()) if row.word != "VALID"]
    detail = "; ".join(missing) or (view.statement if view else "no observation")
    if view is not None and view.imu_missing:
        detail = "IMU unavailable: " + IMU_BLOCK_TEXT + " " + detail
    return Readiness(WARNING, "Model loaded, but the live inputs it needs are missing", detail,
                     "AI SHADOW can run for display; no decision can be valid for control until a LIVE IMU and every other input exist.")


def match_readiness(inputs: MatchInputs) -> Readiness:
    """From the shadow prerequisites alone: can a MANUAL or AI SHADOW round be started?"""
    if not inputs.model_loaded or inputs.inference_error:
        return Readiness(NOT_CONFIGURED, "AI model not loaded: only MANUAL rounds can start", inputs.inference_error or inputs.model_detail,
                         "Load the model on the AI / MODEL tab.")
    if not inputs.camera_ok:
        return Readiness(WARNING, "Camera not operational: only MANUAL rounds can start", inputs.camera_detail, "Check the camera.")
    return Readiness(READY, "MANUAL and AI SHADOW rounds can start",
                     "AUTONOMOUS is separate: it needs the ModeManager gate (hardware acceptance) and every prerequisite.")


def autonomous_readiness_for(gate_open: bool, closed_reasons: list[str], missing: list[str]) -> Readiness:
    if not gate_open:
        return Readiness(BLOCKED, "Autonomous drive is blocked", "Gate closed: " + "; ".join(closed_reasons or ["no link service"]),
                         "Complete the hardware acceptance workflow on the CONNECTIONS tab, with the robot, step by step.")
    if missing:
        return Readiness(WARNING, "The gate is open but prerequisites are missing", "; ".join(missing), "Fix the failed prerequisites on the MATCH tab.")
    return Readiness(READY, "Autonomous drive may be requested", "The gate is open and every prerequisite holds. The operator still confirms deliberately.")


# ================================================================ session log reading (the Match tab's log panel)


def tail_events(path: Path | str, count: int = 60, max_bytes: int = 96_000) -> list[dict[str, Any]]:
    """The last ``count`` events of a JSONL log, reading only the end of the file. A damaged line is skipped."""
    try:
        with open(path, "rb") as handle:
            handle.seek(0, 2)
            size = handle.tell()
            handle.seek(max(0, size - max_bytes))
            data = handle.read()
    except OSError:
        return []
    lines = data.decode("utf-8", errors="replace").splitlines()
    if size > max_bytes and lines:
        lines = lines[1:]                                 # the first line may be cut in half
    events: list[dict[str, Any]] = []
    for line in lines[-(count * 2):]:
        try:
            item = json.loads(line)
        except ValueError:
            continue
        if isinstance(item, dict):
            events.append(item)
    return events[-count:]


def sessions_with_rounds(root: Path | str | None, limit: int = 30) -> list[tuple[Path, list[str]]]:
    """[(session folder, its round log file names)], newest first."""
    from control_center.backend.session_log import DEFAULT_ROOT, list_sessions

    out = []
    for folder in list_sessions(root or DEFAULT_ROOT)[:limit]:
        out.append((folder, sorted(p.name for p in folder.glob("round-*.jsonl"))))
    return out


def format_elapsed(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    return f"{int(seconds // 60):02d}:{seconds % 60:04.1f}"
