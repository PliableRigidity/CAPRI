"""What a trained checkpoint expects to be fed, field by field, and where each value could come from live.

Read-only. Nothing here loads torch, opens a port or a camera, or talks to the robot. The contract is
read from the checkpoint's own metadata (the zip's ``data`` record and the run's
``effective_config.json``) and from the simulator configuration that run was trained with; nothing
is inferred from a file name and nothing is assumed when the record is missing.

The field list mirrors ``autobot.sim.env.AutoBotEnv._obs`` (tests/test_shadow_stream.py checks the two
against each other on the real simulator, so a change there is noticed here).

    python -m control_center.tools.observation_contract runs/recovery-challenger-v2-seed1/final_model.zip
"""

from __future__ import annotations

import argparse
import json
import math
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
RECOMMENDED_CHECKPOINT = REPO_ROOT / "runs" / "recovery-challenger-v2-seed1" / "final_model.zip"

# Where a value can come from on the real system. One of these per field, never a mixture.
SOURCE_ARUCO = "ArUco markers on the own bot + floor map"
SOURCE_FLOOR = "mapped floor position of an unmarked blob"
SOURCE_TELEMETRY = "other telemetry (IMU over the controller link)"
SOURCE_CONSTANT = "constant (no input needed)"
SOURCE_POLICY = "the policy's own last decision (no sensor)"

GROUP_IMU, GROUP_PREVIOUS, GROUP_OWN, GROUP_OPPONENT = "imu", "previous_action", "own_pose", "opponent_position"
GROUP_ORDER = (GROUP_IMU, GROUP_PREVIOUS, GROUP_OWN, GROUP_OPPONENT)

GRAVITY_M_S2 = 9.81                     # the simulated accelerometer's only signal (see FieldSpec notes)


class ContractError(ValueError):
    """The checkpoint does not record what it expects, or its record is inconsistent."""


@dataclass(frozen=True)
class FieldSpec:
    index: int
    name: str
    group: str
    units: str                          # the value as the policy sees it
    convention: str
    simulator_source: str               # what filled it during training
    live_source: str                    # one of the SOURCE_* strings
    live_status: str                    # what exists for it in this repository today
    required_input: bool                # False for values that need no measurement


@dataclass
class ObservationContract:
    checkpoint: Path
    identifier: str
    training: str
    observation_size: int
    action_count: int
    observation_profile: str
    action_profile: str
    previous_action_encoding: str
    recorded_in: dict[str, str]
    tactics: tuple[str, ...]
    previous_action_slots: int
    sim_config_path: Path
    arena_width_m: float
    arena_height_m: float
    gyro_range_rad_s: float
    accel_range_m_s2: float
    imu_rate_hz: float
    imu_mounting_euler_deg: tuple[float, float, float]
    control_hz: float
    fields: list[FieldSpec] = field(default_factory=list)

    def indices(self, group: str) -> list[int]:
        return [spec.index for spec in self.fields if spec.group == group]

    def span(self, group: str) -> str:
        indices = self.indices(group)
        return "-" if not indices else str(indices[0]) if len(indices) == 1 else f"{indices[0]}-{indices[-1]}"

    def required_groups(self) -> list[str]:
        return [group for group in GROUP_ORDER if any(spec.required_input for spec in self.fields if spec.group == group)]

    # ---- the one place where physical values become the observation vector ----

    def assemble(self, *, gyro_rad_s: Any, accel_m_s2: Any, own_xy_m: Any, own_heading_rad: float, opponent_xy_m: Any,
                 previous_action: int | None = None) -> np.ndarray:
        """Physical values in the contract's units and frames -> the vector the policy reads.

        Every argument is required: a caller without a value must not call this (see ``missing_groups``)."""
        gyro, accel = np.asarray(gyro_rad_s, float).ravel(), np.asarray(accel_m_s2, float).ravel()
        own, opponent = np.asarray(own_xy_m, float).ravel(), np.asarray(opponent_xy_m, float).ravel()
        if gyro.shape != (3,) or accel.shape != (3,) or own.shape != (2,) or opponent.shape != (2,):
            raise ContractError("gyro and accel need 3 values each; own and opponent position need 2 each")
        values = list(gyro / self.gyro_range_rad_s) + list(accel / self.accel_range_m_s2)
        if self.previous_action_encoding == "masked_zero_v1":
            values += [0.0] * self.previous_action_slots
        else:
            if previous_action is None or not 0 <= int(previous_action) < self.previous_action_slots:
                raise ContractError("this checkpoint reads its previous tactic; pass previous_action")
            values += list(np.eye(self.previous_action_slots)[int(previous_action)])
        if self.observation_profile == "sim_debug":
            w, h = self.arena_width_m, self.arena_height_m
            values += [2 * own[0] / w - 1, 2 * own[1] / h - 1, math.sin(own_heading_rad), math.cos(own_heading_rad),
                       2 * opponent[0] / w - 1, 2 * opponent[1] / h - 1]
        vector = np.clip(np.asarray(values, float), -1.0, 1.0).astype(np.float32)
        if vector.shape != (self.observation_size,) or not np.isfinite(vector).all():
            raise ContractError(f"assembled {vector.shape[0]} values, the checkpoint expects {self.observation_size}")
        return vector

    def to_report(self) -> dict[str, Any]:
        return {"checkpoint": self.identifier, "training": self.training, "observation_size": self.observation_size,
                "action_count": self.action_count, "tactics": list(self.tactics),
                "observation_profile": self.observation_profile, "action_profile": self.action_profile,
                "previous_action_encoding": self.previous_action_encoding, "recorded_in": self.recorded_in,
                "simulator_config": str(self.sim_config_path), "arena_m": [self.arena_width_m, self.arena_height_m],
                "gyro_range_rad_s": self.gyro_range_rad_s, "accel_range_m_s2": self.accel_range_m_s2,
                "imu_rate_hz": self.imu_rate_hz, "imu_mounting_euler_deg": list(self.imu_mounting_euler_deg),
                "control_hz": self.control_hz,
                "fields": [{"index": s.index, "name": s.name, "group": s.group, "units": s.units, "convention": s.convention,
                            "simulator_source": s.simulator_source, "live_source": s.live_source,
                            "live_status": s.live_status, "required_input": s.required_input} for s in self.fields]}


# ----------------------------------------------------------------------------------------------
# Reading the record


def checkpoint_file(path: Path | str) -> Path:
    path = Path(path)
    return path if path.is_file() or path.suffix == ".zip" else path.with_name(path.name + ".zip")


def _zip_data(file: Path) -> dict[str, Any]:
    try:
        with zipfile.ZipFile(file) as archive:
            data = json.loads(archive.read("data"))
    except FileNotFoundError:
        raise ContractError(f"{file} does not exist") from None
    except (OSError, zipfile.BadZipFile, KeyError, json.JSONDecodeError) as error:
        raise ContractError(f"{file} is not a Stable-Baselines3 checkpoint zip ({type(error).__name__}: {error})") from None
    if not isinstance(data, dict):
        raise ContractError(f"{file}: the zip's data record is not an object")
    return data


def _sidecar(file: Path) -> tuple[Path | None, dict[str, Any]]:
    for directory in (file.parent, file.parent.parent):
        candidate = directory / "effective_config.json"
        if candidate.is_file():
            try:
                record = json.loads(candidate.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as error:
                raise ContractError(f"{candidate} is unreadable: {error}") from None
            return candidate, record if isinstance(record, dict) else {}
    return None, {}


def load_contract(checkpoint: Path | str = RECOMMENDED_CHECKPOINT, sim_config: Path | str | None = None) -> ObservationContract:
    """The checkpoint's observation contract. Raises ContractError rather than guessing a missing record."""
    from autobot.sim.env import ACTION_PROFILES, IMU_OBSERVATION_CHANNELS, PREVIOUS_ACTION_ENCODINGS

    file = checkpoint_file(checkpoint)
    data = _zip_data(file)
    sidecar_path, sidecar = _sidecar(file)
    embedded = data.get("autobot_observation") if isinstance(data.get("autobot_observation"), dict) else {}
    recorded_in: dict[str, str] = {}

    def recorded(key: str, embedded_key: str, sidecar_key: str) -> str:
        a, b = embedded.get(embedded_key), sidecar.get(sidecar_key)
        if a and b and a != b:
            raise ContractError(f"{file}: checkpoint metadata says {key} = {a} but {sidecar_path} says {b}")
        if not (a or b):
            raise ContractError(f"{file}: {key} is not recorded in the checkpoint or in an effective_config.json beside it; "
                                "refusing to infer it from the vector size")
        recorded_in[key] = "checkpoint metadata" + (" and run effective_config.json" if a and b else "") if a else "run effective_config.json"
        return str(a or b)

    profile = recorded("observation_profile", "observation_profile", "profile")
    action_profile = recorded("action_profile", "action_profile", "action_profile")
    encoding = recorded("previous_action_encoding", "previous_action_encoding", "previous_action_encoding")
    if profile not in ("hardware_plausible", "sim_debug"):
        raise ContractError(f"{file}: unknown observation profile {profile!r}")
    if action_profile not in ACTION_PROFILES:
        raise ContractError(f"{file}: unknown action profile {action_profile!r}")
    if encoding not in PREVIOUS_ACTION_ENCODINGS:
        raise ContractError(f"{file}: unknown previous-action encoding {encoding!r}")

    shape = (data.get("observation_space") or {}).get("_shape") or []
    try:
        observation_size = int(shape[0]) if len(shape) == 1 else -1
        action_count = int((data.get("action_space") or {}).get("n"))
    except (TypeError, ValueError):
        raise ContractError(f"{file}: the zip does not record its observation and action sizes") from None
    slots = int(ACTION_PROFILES[action_profile]["previous_action_slots"])
    tactics = tuple(ACTION_PROFILES[action_profile]["tactics"])
    expected = IMU_OBSERVATION_CHANNELS + slots + (6 if profile == "sim_debug" else 0)
    if observation_size != expected or action_count != len(tactics):
        raise ContractError(f"{file}: recorded layout ({profile}, {action_profile}) means {expected} values and {len(tactics)} "
                            f"actions, but the zip's spaces are {observation_size} values and {action_count} actions")

    config_path = Path(sim_config) if sim_config else Path(str(sidecar.get("config_path", "")))
    if sim_config is None and not sidecar.get("config_path"):
        raise ContractError(f"{file}: the run record does not name the simulator config it was trained with; pass sim_config")
    if not config_path.is_absolute():
        config_path = REPO_ROOT / config_path
    if not config_path.is_file():
        raise ContractError(f"simulator config {config_path} (named by the run record) is missing")
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    arena, imu = cfg["arena"], cfg["imu"]
    try:
        identifier = file.resolve().relative_to((REPO_ROOT / "runs").resolve()).as_posix()
    except ValueError:
        identifier = f"{file.parent.name}/{file.name}"
    contract = ObservationContract(
        checkpoint=file, identifier=identifier, training=" ".join(str(sidecar.get("training", "not recorded")).split()),
        observation_size=observation_size, action_count=action_count, observation_profile=profile,
        action_profile=action_profile, previous_action_encoding=encoding, recorded_in=recorded_in, tactics=tactics,
        previous_action_slots=slots, sim_config_path=config_path, arena_width_m=float(arena["width_m"]),
        arena_height_m=float(arena["height_m"]), gyro_range_rad_s=float(imu["gyro_range_rad_s"]),
        accel_range_m_s2=float(imu["accel_range_m_s2"]), imu_rate_hz=float(imu["sample_rate_hz"]),
        imu_mounting_euler_deg=tuple(float(v) for v in imu["mounting_euler_deg"]),
        control_hz=float(cfg["simulation"]["control_hz"]))
    contract.fields = _fields(contract)
    return contract


def _fields(c: ObservationContract) -> list[FieldSpec]:
    w, h = c.arena_width_m, c.arena_height_m
    frame = ("arena-fixed frame of config/arena_overlay.yaml and the simulator: origin at the plain corner, "
             "+x and +y along its two walls, flipper corner at (W, H)")
    body = ("robot body frame of the simulator: x toward the nose, y to the robot's left, z up; "
            f"mounting rotation {list(c.imu_mounting_euler_deg)} deg")
    imu_status = ("NOT AVAILABLE. The controller link defines an 'IMU,ms,ax,ay,az,gx,gy,gz' line (m/s^2 and deg/s, raw "
                  "sensor axes) but the bench sketch does not send it, the sensor's mounting orientation is not recorded, "
                  "and nothing converts it to rad/s in the body frame")
    fields: list[FieldSpec] = []
    for i, axis in enumerate("xyz"):
        fields.append(FieldSpec(
            i, f"gyro_{axis}", GROUP_IMU, f"angular rate about body {axis} in rad/s, divided by {c.gyro_range_rad_s:g}, clipped to -1..1",
            body, f"simulated IMU: true body rate + Gaussian noise, sampled at {c.imu_rate_hz:g} Hz",
            SOURCE_TELEMETRY, imu_status, True))
    for i, axis in enumerate("xyz"):
        fields.append(FieldSpec(
            3 + i, f"accel_{axis}", GROUP_IMU, f"m/s^2 along body {axis}, divided by {c.accel_range_m_s2:g}, clipped to -1..1",
            body + f". At rest and upright this is (0, 0, +{GRAVITY_M_S2})",
            "simulated IMU: gravity's reaction rotated into the body frame + noise. It contains NO acceleration from "
            "driving, turning or impacts, which a real accelerometer does measure", SOURCE_TELEMETRY, imu_status, True))
    for slot in range(c.previous_action_slots):
        masked = c.previous_action_encoding == "masked_zero_v1"
        fields.append(FieldSpec(
            6 + slot, f"previous_tactic_{slot}", GROUP_PREVIOUS,
            "always 0.0" if masked else "1.0 in the slot of the last applied tactic, else 0.0", "no frame",
            "constant 0.0 (masked_zero_v1)" if masked else "the policy's own last applied tactic",
            SOURCE_CONSTANT if masked else SOURCE_POLICY,
            "available: needs no input" if masked else "the runner must remember the last tactic", not masked))
    if c.observation_profile == "sim_debug":
        base = 6 + c.previous_action_slots
        pose_status = ("camera path exists (control_center/tools/overlay_core.py: marker corners -> floor map -> rigid fit) "
                       "but is tested ONLY on the synthetic fixture; marker dictionary, size, height and offsets are "
                       "placeholders, and the offsets must be measured from the chassis box centre for this to match")
        truth = "simulator truth (privileged): exact, no noise, no delay, no dropouts"
        fields += [
            FieldSpec(base, "own_x", GROUP_OWN, f"2*x/{w:g} - 1 with x in metres, clipped to -1..1",
                      frame + ". The point is the centre of the chassis collision box", truth, SOURCE_ARUCO, pose_status, True),
            FieldSpec(base + 1, "own_y", GROUP_OWN, f"2*y/{h:g} - 1 with y in metres, clipped to -1..1",
                      frame + ". The point is the centre of the chassis collision box", truth, SOURCE_ARUCO, pose_status, True),
            FieldSpec(base + 2, "sin_own_yaw", GROUP_OWN, "sine of the heading (no scaling)",
                      "heading = direction the nose points, counter-clockwise from arena +x seen from above", truth,
                      SOURCE_ARUCO, pose_status, True),
            FieldSpec(base + 3, "cos_own_yaw", GROUP_OWN, "cosine of the heading (no scaling)",
                      "heading = direction the nose points, counter-clockwise from arena +x seen from above", truth,
                      SOURCE_ARUCO, pose_status, True),
        ]
        opponent_status = ("NOT AVAILABLE LIVE. An unmarked-blob tracker exists (control_center/tools/blob_tracker.py) and "
                           "gives a blob centre on the floor plane, but nothing decides which blob is the opponent, it gives "
                           "no position while two robots touch, and it needs an empty-floor picture")
        fields += [
            FieldSpec(base + 4, "opponent_x", GROUP_OPPONENT, f"2*x/{w:g} - 1 with x in metres, clipped to -1..1",
                      frame + ". The point is the centre of the opponent's chassis box", truth, SOURCE_FLOOR, opponent_status, True),
            FieldSpec(base + 5, "opponent_y", GROUP_OPPONENT, f"2*y/{h:g} - 1 with y in metres, clipped to -1..1",
                      frame + ". The point is the centre of the opponent's chassis box", truth, SOURCE_FLOOR, opponent_status, True),
        ]
    return fields


def format_contract(contract: ObservationContract, width: int = 110) -> str:
    c = contract
    lines = [f"Checkpoint        {c.identifier}",
             f"Training          {c.training}",
             f"Layout            {c.observation_size} values in, {c.action_count} tactics out {list(c.tactics)}",
             f"Recorded as       observation_profile={c.observation_profile} ({c.recorded_in['observation_profile']}), "
             f"action_profile={c.action_profile}, previous_action_encoding={c.previous_action_encoding}",
             f"Simulator config  {c.sim_config_path}",
             f"Rates             control {c.control_hz:g} Hz, IMU {c.imu_rate_hz:g} Hz", ""]
    for group in GROUP_ORDER:
        specs = [s for s in c.fields if s.group == group]
        if not specs:
            continue
        lines.append(f"[{c.span(group)}] {group}  <-  {specs[0].live_source}")
        for spec in specs if group != GROUP_PREVIOUS else specs[:1]:
            name = spec.name if group != GROUP_PREVIOUS else f"previous_tactic_0..{len(specs) - 1}"
            lines.append(f"    {spec.index if group != GROUP_PREVIOUS else c.span(group):>5}  {name}: {spec.units}")
        for convention in dict.fromkeys(spec.convention for spec in specs):
            lines.append(f"    frame      {convention}")
        lines.append(f"    simulator  {specs[-1].simulator_source}")
        lines.append(f"    live now   {specs[-1].live_status}")
        lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("checkpoint", nargs="?", default=str(RECOMMENDED_CHECKPOINT))
    parser.add_argument("--json", action="store_true", help="print the contract as JSON")
    args = parser.parse_args(argv)
    try:
        contract = load_contract(args.checkpoint)
    except ContractError as error:
        print(f"contract error: {error}")
        return 2
    print(json.dumps(contract.to_report(), indent=2) if args.json else format_contract(contract))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
