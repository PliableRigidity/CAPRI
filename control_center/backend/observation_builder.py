"""Physical inputs -> the checkpoint's 19-value observation, with provenance. Pure: no camera, port, socket, torch or Qt.

``build_observation(inputs, now)`` is the one rule set for "may this vector be built, and may it be used":

* it never returns values unless EVERY group is present and fresh (camera <= 0.25 s, IMU <= 0.10 s);
* the opponent group needs a TRACKED opponent; UNCERTAIN / LOST / not selected means the group is MISSING,
  never "the last position";
* the only values filled in without a measurement are the ones the training environment itself defined as
  constants (the previous-tactic slots under ``masked_zero_v1``: always 0.0, autobot/sim/env.py:137-139);
* ``source`` is LIVE only when every present input is declared LIVE; anything simulated or replayed makes the
  vector DIAGNOSTIC: values may exist, ``valid_for_control`` is False and no command may be derived from it;
* ``valid_for_control`` additionally needs a real, fresh, mounting-verified IMU. The baseline bench firmware
  sends none, so on it nothing here can ever be valid for control.

The vector itself is assembled by ``ObservationContract.assemble`` (the single place where physical values
become channels, mirroring AutoBotEnv._obs); this module decides whether it is allowed to be called.
See docs/policy-observation-schema.md for the channel table and the file:line behind each statement.

Evidence: everything this module produces from tests is SOFTWARE/SIMULATED. Nothing here is a hardware result.
"""

from __future__ import annotations

import functools
import math
from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

from control_center.tools.observation_contract import (GROUP_IMU, GROUP_OPPONENT, GROUP_OWN, GROUP_PREVIOUS, GROUP_ORDER,
                                                       RECOMMENDED_CHECKPOINT, ObservationContract, load_contract)

SOURCE_LIVE, SOURCE_SIMULATED, SOURCE_REPLAY = "LIVE", "SIMULATED", "REPLAY"
SOURCES = (SOURCE_LIVE, SOURCE_SIMULATED, SOURCE_REPLAY)
ORIGIN_CONSTANT = "CONSTANT"                     # a value the training environment defined as a constant

STATUS_OK, STATUS_MISSING, STATUS_STALE = "OK", "MISSING", "STALE"

MAX_CAMERA_AGE_S = 0.25                          # own pose and opponent (both come from one camera frame)
MAX_IMU_AGE_S = 0.10
ARENA_MARGIN_M = 0.10                            # a pose this far outside the arena rectangle is a fault, not a position
FUTURE_TOLERANCE_S = 0.005                       # an input stamped later than ``now`` by more than this is refused
OWN_ACCEPTED_STATES = ("measured", "predicted")  # chassis tracker states (control_center/tools/chassis_tracker.py)
OPPONENT_ACCEPTED_STATE = "tracked"              # roi_tracker TRACKED; UNCERTAIN / LOST / not_selected are not accepted


# ----------------------------------------------------------------------------------------------
# Inputs


@dataclass(frozen=True)
class OwnPoseInput:
    """Own chassis-centre pose in the ARENA frame (origin at the plain corner, +x/+y along the walls), radians CCW from +x."""
    x_m: float | None
    y_m: float | None
    heading_rad: float | None
    t: float                                     # time of the camera frame the pose rests on (same clock as ``now``)
    state: str = "measured"                      # measured | predicted | anything else = not usable
    reason: str = ""                             # why it is not usable, when it is not
    origin: str | None = None                    # None = inherit ObservationInputs.origin


@dataclass(frozen=True)
class OpponentInput:
    """Opponent chassis-centre position in the ARENA frame, metres."""
    x_m: float | None
    y_m: float | None
    t: float
    state: str = "tracked"                       # only "tracked" is accepted
    reason: str = ""
    origin: str | None = None


@dataclass(frozen=True)
class ImuInput:
    """Body-frame IMU in rad/s and m/s^2 (x nose, y left, z up), i.e. AFTER the verified mounting transform."""
    gyro_rad_s: tuple[float, float, float]
    accel_m_s2: tuple[float, float, float]
    t: float
    origin: str | None = None
    mounting_verified: bool = False


@dataclass(frozen=True)
class ObservationInputs:
    """Everything known at one decision time. A group left as None is MISSING. ``origin`` is the pipeline's declared source."""
    origin: str = SOURCE_SIMULATED               # safe default: nothing is LIVE unless the caller says so
    own: OwnPoseInput | None = None
    opponent: OpponentInput | None = None
    imu: ImuInput | None = None
    previous_action: int | None = None           # only read when the checkpoint encodes it (onehot_v1)
    extra_blocked_by: tuple[str, ...] = ()       # caller-known reasons the vector must not drive (e.g. wrong arena frame)


@dataclass(frozen=True)
class AgeLimits:
    camera_s: float = MAX_CAMERA_AGE_S
    imu_s: float = MAX_IMU_AGE_S


# ----------------------------------------------------------------------------------------------
# Result


@dataclass(frozen=True)
class GroupStatus:
    group: str
    indices: str
    status: str                                  # OK | MISSING | STALE
    age_s: float | None
    reason: str
    origin: str | None                           # LIVE / SIMULATED / REPLAY / CONSTANT / None when nothing arrived
    values: Mapping[str, float] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.status == STATUS_OK

    def to_report(self) -> dict[str, Any]:
        return {"indices": self.indices, "status": self.status, "age_ms": None if self.age_s is None else round(1000 * self.age_s, 1),
                "reason": self.reason, "origin": self.origin, "values": {k: round(float(v), 5) for k, v in self.values.items()}}


@dataclass(frozen=True)
class ObservationResult:
    values: np.ndarray | None                    # the 19 values, or None when ANY group is missing or stale
    source: str                                  # LIVE | SIMULATED | REPLAY
    valid_for_control: bool
    blocked_by: tuple[str, ...]
    groups: Mapping[str, GroupStatus]
    missing: tuple[str, ...]
    stale: tuple[str, ...]
    contract_id: str
    built_at: float

    @property
    def complete(self) -> bool:
        return self.values is not None

    def ages_s(self) -> dict[str, float | None]:
        return {name: status.age_s for name, status in self.groups.items()}

    def oldest_input_t(self) -> float | None:
        """Time of the oldest measured input in the vector (constants have none), or None if there are no measured inputs."""
        ages = [status.age_s for status in self.groups.values() if status.age_s is not None]
        return None if not ages else self.built_at - max(ages)

    def to_report(self) -> dict[str, Any]:
        return {"source": self.source, "valid_for_control": self.valid_for_control, "complete": self.complete,
                "blocked_by": list(self.blocked_by), "missing": list(self.missing), "stale": list(self.stale),
                "contract": self.contract_id, "built_at": round(self.built_at, 6),
                "ages_ms": {k: None if v is None else round(1000 * v, 1) for k, v in self.ages_s().items()},
                "groups": {name: status.to_report() for name, status in self.groups.items()},
                "observation": None if self.values is None else [round(float(v), 5) for v in self.values]}


# ----------------------------------------------------------------------------------------------
# Provenance (shared with stream_adapter.ShadowAssembler so there is one rule)


def combine_origins(*origins: str | None) -> str:
    """Most conservative label: any SIMULATED makes the whole thing SIMULATED, else any REPLAY, else LIVE.

    Unknown or missing labels count as SIMULATED; CONSTANT and None are ignored (they carry no provenance)."""
    seen = {origin for origin in origins if origin not in (None, ORIGIN_CONSTANT)}
    if not seen or not seen <= set(SOURCES) or SOURCE_SIMULATED in seen:
        return SOURCE_SIMULATED
    return SOURCE_REPLAY if SOURCE_REPLAY in seen else SOURCE_LIVE


def origin_from_capture_state(state: str | None) -> str:
    """The hub capture session's state (control_center/backend/capture.py:142 'OPENING, LIVE, MOCK, REPLAY, LOST, UNAVAILABLE')."""
    return {"LIVE": SOURCE_LIVE, "REPLAY": SOURCE_REPLAY}.get(str(state or "").upper(), SOURCE_SIMULATED)


def control_validity(source: str, imu_origin: str | None, imu_ok: bool, all_groups_ok: bool, extra_blocked_by=()) -> tuple[bool, list[str]]:
    """May a command ever be derived from this vector? Returns (valid, reasons it is not)."""
    why: list[str] = []
    if source != SOURCE_LIVE:
        why.append(f"source is {source}, not LIVE: diagnostic only, never valid for control")
    if not imu_ok or imu_origin != SOURCE_LIVE:
        why.append("no real IMU: the IMU group is not a fresh LIVE measurement")
    if not all_groups_ok:
        why.append("the observation is incomplete")
    why += [str(item) for item in extra_blocked_by]
    return not why, why


# ----------------------------------------------------------------------------------------------
# The builder


@functools.lru_cache(maxsize=4)
def _cached_contract(path: str) -> ObservationContract:
    return load_contract(path)


def default_contract() -> ObservationContract:
    return _cached_contract(str(RECOMMENDED_CHECKPOINT))


def _finite(values) -> bool:
    try:
        return bool(np.isfinite(np.asarray(values, float)).all())
    except (TypeError, ValueError):
        return False


def _age(t: float, now: float) -> tuple[float | None, str]:
    if not isinstance(t, (int, float)) or isinstance(t, bool) or not math.isfinite(t):
        return None, "its timestamp is not a number"
    if t - now > FUTURE_TOLERANCE_S:
        return None, f"its timestamp is {1000 * (t - now):.0f} ms in the future of the decision time (clock fault)"
    return max(0.0, now - t), ""


def _inside_arena(x: float, y: float, contract: ObservationContract) -> bool:
    return (-ARENA_MARGIN_M <= x <= contract.arena_width_m + ARENA_MARGIN_M
            and -ARENA_MARGIN_M <= y <= contract.arena_height_m + ARENA_MARGIN_M)


def build_observation(inputs: ObservationInputs, now: float, contract: ObservationContract | None = None,
                      limits: AgeLimits = AgeLimits()) -> ObservationResult:
    """Decide, per group, whether it is usable at ``now``; assemble the vector only if every group is."""
    c = contract or default_contract()
    groups: dict[str, GroupStatus] = {}

    def add(group: str, status: str, age: float | None, reason: str, origin: str | None, values: Mapping[str, float] | None = None) -> None:
        groups[group] = GroupStatus(group, c.span(group), status, age, reason, origin, dict(values or {}))

    # ---- IMU (controller telemetry; nothing else may fill it) ----
    imu = inputs.imu
    if imu is None:
        add(GROUP_IMU, STATUS_MISSING, None, "no IMU sample: the controller firmware does not send one, or the link is down", None)
    else:
        age, problem = _age(imu.t, now)
        origin = imu.origin or inputs.origin
        if problem:
            add(GROUP_IMU, STATUS_MISSING, None, problem, origin)
        elif not imu.mounting_verified:
            add(GROUP_IMU, STATUS_MISSING, age, "the IMU mounting transform is not verified (config/imu_mounting.yaml imu.verified is false): "
                "body-frame values cannot be trusted", origin)
        elif not (_finite(imu.gyro_rad_s) and _finite(imu.accel_m_s2) and np.asarray(imu.gyro_rad_s).shape == (3,)
                  and np.asarray(imu.accel_m_s2).shape == (3,)):
            add(GROUP_IMU, STATUS_MISSING, age, "IMU values are not three finite numbers each", origin)
        elif age > limits.imu_s:
            add(GROUP_IMU, STATUS_STALE, age, f"the newest IMU sample is {1000 * age:.0f} ms old (limit {1000 * limits.imu_s:.0f} ms)", origin)
        else:
            add(GROUP_IMU, STATUS_OK, age, "ok", origin, {"gyro_z_rad_s": imu.gyro_rad_s[2], "accel_z_m_s2": imu.accel_m_s2[2]})

    # ---- previous tactic: a constant under masked_zero_v1, otherwise the policy's own last action ----
    if c.indices(GROUP_PREVIOUS):
        if c.previous_action_encoding == "masked_zero_v1":
            add(GROUP_PREVIOUS, STATUS_OK, None, "constant 0.0 (masked_zero_v1: the policy never saw its previous tactic in training)",
                ORIGIN_CONSTANT)
        else:
            previous = inputs.previous_action
            if previous is None or isinstance(previous, bool) or not isinstance(previous, (int, np.integer)) \
                    or not 0 <= int(previous) < c.previous_action_slots:
                add(GROUP_PREVIOUS, STATUS_MISSING, None, "this checkpoint reads the tactic it last APPLIED and none is known", None)
            else:
                add(GROUP_PREVIOUS, STATUS_OK, None, "the policy's own last applied tactic", inputs.origin)

    # ---- own pose and opponent (one camera frame each) ----
    own = inputs.own
    if own is None:
        add(GROUP_OWN, STATUS_MISSING, None, "no own pose", None)
    else:
        age, problem = _age(own.t, now)
        origin = own.origin or inputs.origin
        if problem:
            add(GROUP_OWN, STATUS_MISSING, None, problem, origin)
        elif own.state not in OWN_ACCEPTED_STATES:
            add(GROUP_OWN, STATUS_MISSING, age, own.reason or f"own pose state is {own.state!r}, not measured/predicted", origin)
        elif own.x_m is None or own.y_m is None or own.heading_rad is None or not _finite([own.x_m, own.y_m, own.heading_rad]):
            add(GROUP_OWN, STATUS_MISSING, age, "own pose has a missing or non-finite value", origin)
        elif not _inside_arena(own.x_m, own.y_m, c):
            add(GROUP_OWN, STATUS_MISSING, age, f"own pose ({own.x_m:.2f}, {own.y_m:.2f}) m is outside the arena by more than "
                f"{ARENA_MARGIN_M * 100:.0f} cm: wrong frame or a bad detection", origin)
        elif age > limits.camera_s:
            add(GROUP_OWN, STATUS_STALE, age, f"the newest camera frame is {1000 * age:.0f} ms old (limit {1000 * limits.camera_s:.0f} ms)", origin)
        else:
            add(GROUP_OWN, STATUS_OK, age, own.state, origin, {"x_m": own.x_m, "y_m": own.y_m, "heading_deg": math.degrees(own.heading_rad)})

    opponent = inputs.opponent
    if opponent is None:
        add(GROUP_OPPONENT, STATUS_MISSING, None, "no opponent track (none selected)", None)
    else:
        age, problem = _age(opponent.t, now)
        origin = opponent.origin or inputs.origin
        if problem:
            add(GROUP_OPPONENT, STATUS_MISSING, None, problem, origin)
        elif opponent.state != OPPONENT_ACCEPTED_STATE:
            add(GROUP_OPPONENT, STATUS_MISSING, age, opponent.reason or f"opponent track is {opponent.state.upper()}: its position is not used, "
                "and the last position is never substituted", origin)
        elif opponent.x_m is None or opponent.y_m is None or not _finite([opponent.x_m, opponent.y_m]):
            add(GROUP_OPPONENT, STATUS_MISSING, age, "opponent position has a missing or non-finite value", origin)
        elif not _inside_arena(opponent.x_m, opponent.y_m, c):
            add(GROUP_OPPONENT, STATUS_MISSING, age, f"opponent position ({opponent.x_m:.2f}, {opponent.y_m:.2f}) m is outside the arena by more "
                f"than {ARENA_MARGIN_M * 100:.0f} cm", origin)
        elif age > limits.camera_s:
            add(GROUP_OPPONENT, STATUS_STALE, age, f"the newest camera frame is {1000 * age:.0f} ms old (limit {1000 * limits.camera_s:.0f} ms)", origin)
        else:
            add(GROUP_OPPONENT, STATUS_OK, age, "tracked", origin, {"x_m": opponent.x_m, "y_m": opponent.y_m})

    ordered = {group: groups[group] for group in GROUP_ORDER if group in groups}
    missing = tuple(g for g, s in ordered.items() if s.status == STATUS_MISSING)
    stale = tuple(g for g, s in ordered.items() if s.status == STATUS_STALE)
    blocked = [f"[{s.indices}] {g}: {s.reason}" for g, s in ordered.items() if not s.ok]
    all_ok = not blocked

    values = None
    if all_ok:
        own_v, opp_v, imu_i = ordered[GROUP_OWN].values, ordered[GROUP_OPPONENT].values, inputs.imu
        values = c.assemble(gyro_rad_s=imu_i.gyro_rad_s, accel_m_s2=imu_i.accel_m_s2, own_xy_m=(own_v["x_m"], own_v["y_m"]),
                            own_heading_rad=math.radians(own_v["heading_deg"]), opponent_xy_m=(opp_v["x_m"], opp_v["y_m"]),
                            previous_action=inputs.previous_action)

    present = [s.origin for s in ordered.values() if s.status == STATUS_OK and s.origin != ORIGIN_CONSTANT]
    source = combine_origins(inputs.origin, *present)
    imu_status = ordered[GROUP_IMU]
    valid, why = control_validity(source, imu_status.origin, imu_status.ok, all_ok, inputs.extra_blocked_by)
    # An incomplete vector is already explained group by group; add only the provenance reasons not already implied.
    blocked_by = tuple(blocked + [item for item in why if item != "the observation is incomplete" and item not in blocked])
    return ObservationResult(values, source, valid and values is not None, blocked_by, ordered, missing, stale, c.identifier, float(now))


# ----------------------------------------------------------------------------------------------
# Adapter from the existing camera/telemetry objects (stream_adapter.CameraSample, a TelemetrySource)


def inputs_from_camera(sample, decision_t: float, telemetry, *, origin: str, camera_origin: str | None = None,
                       imu_origin: str | None = None, previous_action: int | None = None,
                       extra_blocked_by: tuple[str, ...] = ()) -> ObservationInputs:
    """CameraSample (+ telemetry) -> ObservationInputs, reading the pose exactly as ``ShadowAssembler`` does.

    ``origin`` is the pipeline's declared source; ``camera_origin`` / ``imu_origin`` refine it per input. The IMU is read
    with ``telemetry.latest(decision_t)``; its mounting verification is read from ``telemetry.mounting.verified`` (absent = False)."""
    from control_center.tools.stream_adapter import ShadowAssembler

    own = opponent = None
    if sample is not None:
        pose, _ = ShadowAssembler._own_at(sample, decision_t)
        track = sample.own_track or {}
        state = "predicted" if track.get("state") == "predicted" else "measured"
        if pose.ok:
            own = OwnPoseInput(pose.x_m, pose.y_m, math.radians(pose.heading_deg), sample.t, state, origin=camera_origin)
        else:
            own = OwnPoseInput(None, None, None, sample.t, "lost", pose.reason, origin=camera_origin)
        est = sample.opponent
        if est.ok:
            opponent = OpponentInput(est.xy_m[0], est.xy_m[1], sample.t, "tracked", origin=camera_origin)
        else:
            state = str((sample.opponent_track or {}).get("state") or "unknown").lower()
            opponent = OpponentInput(None, None, sample.t, state if state != "tracked" else "unknown", est.reason, origin=camera_origin)
    imu_input = None
    reading = telemetry.latest(decision_t) if telemetry is not None else None
    if reading is not None:
        verified = bool(getattr(getattr(telemetry, "mounting", None), "verified", False))
        imu_input = ImuInput(tuple(reading.gyro_rad_s), tuple(reading.accel_m_s2), reading.t,
                             imu_origin or getattr(telemetry, "origin", None), verified)
    return ObservationInputs(origin, own, opponent, imu_input, previous_action, extra_blocked_by)
