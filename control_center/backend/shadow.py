"""Policy "shadow" diagnostics: what a checkpoint is, what it needs, and what it proposes in the SIMULATOR.

Three policy states exist on the dashboard:

  OFF           no policy is running
  SIM           a checkpoint reads SIMULATOR observations and drives the SIMULATED robot only
  LIVE SHADOW   a checkpoint would read live observations and only display what it proposes

LIVE SHADOW is not available: this repository has no adapter that turns the live camera pose and
IMU stream into a policy observation, and no live opponent tracker. `live_shadow_status` says exactly
what is missing. Simulator truth is never substituted and labelled live.

This module is display-only by construction. It never opens a port, has no drive-command formatter
and holds no reference to anything that can reach the motors: the only thing it calls on the hub is
`set_policy`, which stores what to show. A model running in the simulator is not evidence that it can
run from live observations.
"""

from __future__ import annotations

import json
import time
import zipfile
from argparse import Namespace
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PySide6.QtCore import QThread, Signal

ROOT = Path(__file__).resolve().parents[2]
RUNS_DIR = ROOT / "runs"

# The frozen evaluation setup, the same one scripts/live_round_diagnostic.py `make_session` uses.
# Not config/sim.yaml: that file is the working copy and has moved on since the checkpoints were made.
FROZEN_CONFIG = RUNS_DIR / "ppo-curriculum-overnight" / "effective_sim.yaml"
FROZEN_SCENARIOS = RUNS_DIR / "ppo-curriculum-overnight" / "evaluation_scenarios.json"
SCENARIO_INDEX = 2                      # default_spawn
SEED = 10007
OPPONENT_MODE = "static"
OPPONENT_CHASSIS = "last_year"

KIND_SUPERVISED = "supervised clone"
KIND_PPO = "PPO"
KIND_UNKNOWN = "unknown (no training record; 0 PPO timesteps)"

SIM_INPUT_SOURCE = "simulator observations"
HUB_REPORT_PERIOD_S = 0.2               # the dashboard is told at most 5 times a second
IMU_CHANNELS = 6
SIM_TRUTH_CHANNELS = 6


class CheckpointError(ValueError):
    """The file is not a readable checkpoint, or it cannot be run in the frozen simulator."""


@dataclass(frozen=True)
class ObservationField:
    index: str                          # position(s) in the observation vector, e.g. "0-2"
    name: str
    in_simulator: str                   # where the simulator gets it
    live_source: str                    # where it would have to come from on the real robot
    needs: str                          # "imu" | "camera_pose" | "opponent_tracker" | "policy" | "none" | "unknown"


@dataclass
class CheckpointInfo:
    path: Path
    identifier: str                     # run folder / file name: says which file, not what kind of model
    kind: str                           # KIND_SUPERVISED | KIND_PPO | KIND_UNKNOWN, from metadata only
    evidence: str
    num_timesteps: int | None
    observation_size: int | None
    action_count: int | None
    action_profile: str | None
    action_profile_source: str
    tactics: tuple[str, ...]
    previous_action_encoding: str | None
    previous_action_encoding_source: str
    observation_profile: str | None
    observation_profile_source: str
    requirements: list[ObservationField] = field(default_factory=list)
    sidecar: Path | None = None
    warnings: list[str] = field(default_factory=list)

    @property
    def label(self) -> str:
        return f"{self.identifier} [{self.kind}]"

    @property
    def layout_known(self) -> bool:
        return self.observation_profile is not None and self.action_profile is not None


# ----------------------------------------------------------------------------------------------
# Reading a checkpoint's metadata (no torch, nothing is loaded)


def _checkpoint_file(path: Path | str) -> Path:
    path = Path(path)
    return path if path.is_file() or path.suffix == ".zip" else path.with_name(path.name + ".zip")


def _read_zip_data(file: Path) -> dict[str, Any]:
    try:
        with zipfile.ZipFile(file) as archive:
            data = json.loads(archive.read("data"))
    except FileNotFoundError:
        raise CheckpointError(f"{file} does not exist") from None
    except (OSError, zipfile.BadZipFile, KeyError, json.JSONDecodeError) as error:
        raise CheckpointError(f"{file} is not a Stable-Baselines3 checkpoint zip ({type(error).__name__}: {error})") from None
    if not isinstance(data, dict):
        raise CheckpointError(f"{file}: the zip's data record is not an object")
    return data


def _sidecar(file: Path) -> tuple[Path | None, dict[str, Any]]:
    """effective_config.json of the run folder that owns the checkpoint (same search as the simulator CLI)."""
    for directory in (file.parent, file.parent.parent):
        candidate = directory / "effective_config.json"
        if candidate.is_file():
            try:
                record = json.loads(candidate.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                return candidate, {}
            return candidate, record if isinstance(record, dict) else {}
    return None, {}


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _short(text: Any, limit: int = 220) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[:limit - 1] + "…"


def _identifier(file: Path) -> str:
    try:
        return file.resolve().relative_to(RUNS_DIR.resolve()).as_posix()
    except ValueError:
        return f"{file.parent.name}/{file.name}"


def _classify(num_timesteps: int | None, sidecar_path: Path | None, sidecar: dict[str, Any], file: Path) -> tuple[str, str, list[str]]:
    """Label from metadata only: the sidecar's training record and the zip's PPO timestep count."""
    warnings: list[str] = []
    training = sidecar.get("training")
    trainer = sidecar.get("trainer")
    steps = "no num_timesteps" if num_timesteps is None else f"num_timesteps = {num_timesteps:,}"
    where = None if sidecar_path is None else f"{sidecar_path.parent.name}/{sidecar_path.name}"
    extras = []
    if trainer:
        extras.append(f"trainer = {trainer}")
    record = file.parent / "training.json"
    if record.is_file():
        try:
            fitted = json.loads(record.read_text(encoding="utf-8"))
            if isinstance(fitted, dict) and "fitted_decisions" in fitted:
                extras.append(f"training.json records fitted_decisions = {_short(fitted['fitted_decisions'], 60)}")
        except (OSError, json.JSONDecodeError):
            pass
    says_supervised = isinstance(training, str) and "supervised" in training.lower()
    if says_supervised:
        evidence = f'{where} says training = "{_short(training)}"; ' + "; ".join(extras + [f"the zip records {steps}"])
        if num_timesteps:
            warnings.append(f"the run record says supervised, but the zip also records {num_timesteps:,} PPO timesteps: "
                            "it may have been trained further by PPO after cloning")
        return KIND_SUPERVISED, evidence, warnings
    if num_timesteps is not None and num_timesteps > 0:
        if where is None:
            evidence = f"the zip records {steps}; there is no effective_config.json next to it, so nothing says it is supervised"
        else:
            evidence = f"the zip records {steps}; {where} has no 'training' text saying it is supervised"
        if extras:
            evidence += "; " + "; ".join(extras)
        if sidecar.get("initialisation") is not None:
            evidence += f"; the run record's initialisation = {_short(sidecar['initialisation'], 160)}"
        return KIND_PPO, evidence, warnings
    if where is None:
        evidence = f"the zip records {steps} and there is no effective_config.json next to it"
    else:
        evidence = f"the zip records {steps} and {where} has no 'training' text"
        if extras:
            evidence += " (" + "; ".join(extras) + ")"
    return KIND_UNKNOWN, evidence, warnings


def _layout(embedded: dict[str, Any], sidecar: dict[str, Any], observation_size: int | None,
            action_count: int | None) -> tuple[str | None, str, str | None, str, list[str]]:
    """(observation profile, its source, action profile, its source, warnings)."""
    from autobot.sim.env import ACTION_PROFILES

    warnings: list[str] = []
    candidates = [(profile, name) for profile, extra in (("hardware_plausible", 0), ("sim_debug", SIM_TRUTH_CHANNELS))
                  for name, spec in ACTION_PROFILES.items()
                  if IMU_CHANNELS + int(spec["previous_action_slots"]) + extra == observation_size
                  and len(spec["tactics"]) == action_count]

    def recorded(key_embedded: str, key_sidecar: str) -> tuple[str | None, str]:
        if embedded.get(key_embedded):
            return str(embedded[key_embedded]), "checkpoint metadata"
        if sidecar.get(key_sidecar):
            return str(sidecar[key_sidecar]), "run effective_config.json"
        return None, ""

    profile, profile_source = recorded("observation_profile", "profile")
    action, action_source = recorded("action_profile", "action_profile")
    if action is not None and action not in ACTION_PROFILES:
        warnings.append(f"recorded action profile {action!r} is not one the simulator knows")
        action, action_source = None, ""
    fitting = [c for c in candidates if profile in (None, c[0]) and action in (None, c[1])]
    inferred = f"inferred from the observation size ({observation_size}) and action count ({action_count}); not recorded"
    if profile is not None and action is not None:
        if not fitting:
            warnings.append(f"recorded layout ({profile}, {action}) does not match the zip's spaces "
                            f"({observation_size} observation values, {action_count} actions)")
    elif len(fitting) == 1:
        if profile is None:
            profile, profile_source = fitting[0][0], inferred
        if action is None:
            action, action_source = fitting[0][1], inferred
    else:
        undetermined = "could not be determined: not recorded, and the sizes fit " + (
            "no known layout" if not fitting else f"{len(fitting)} layouts")
        if profile is None:
            profile_source = undetermined
        if action is None:
            action_source = undetermined
    return profile, profile_source, action, action_source, warnings


def observation_requirements(observation_profile: str | None, action_profile: str | None, previous_action_encoding: str | None,
                             observation_size: int | None) -> list[ObservationField]:
    """What each observation value is, and where it would have to come from on the real robot."""
    if observation_profile is None or action_profile is None:
        last = "?" if observation_size is None else str(observation_size - 1)
        return [ObservationField(f"0-{last}", "layout unknown", "unknown", "cannot be listed: the layout is not recorded", "unknown")]
    from autobot.sim.env import ACTION_PROFILES

    slots = int(ACTION_PROFILES[action_profile]["previous_action_slots"])
    fields = [
        ObservationField("0-2", "gyro x, y, z ÷ gyro range", "simulated IMU (noisy, sensor frame)",
                         "IMU stream from the controller: angular rate in rad/s, in the same axes and scaling as the simulated IMU", "imu"),
        ObservationField("3-5", "accelerometer x, y, z ÷ accel range", "simulated IMU (noisy, sensor frame)",
                         "IMU stream from the controller: acceleration in m/s², in the same axes and scaling as the simulated IMU", "imu"),
    ]
    span = f"{IMU_CHANNELS}-{IMU_CHANNELS + slots - 1}"
    if previous_action_encoding == "masked_zero_v1":
        fields.append(ObservationField(span, f"previous tactic, masked ({slots} slots held at 0.0)", "constant 0.0",
                                       "nothing: these values are always 0.0 for this checkpoint", "none"))
    else:
        fields.append(ObservationField(span, f"previous tactic, one-hot ({slots} slots)", "the policy's own last applied tactic",
                                       "the policy itself: whatever runs it must remember its last tactic (no sensor)", "policy"))
    if observation_profile == "sim_debug":
        base = IMU_CHANNELS + slots
        fields += [
            ObservationField(f"{base}-{base + 1}", "own x, own y (scaled to -1..1 across the arena)", "simulator truth (privileged)",
                             "calibrated camera pose of the own robot", "camera_pose"),
            ObservationField(f"{base + 2}-{base + 3}", "sin(own yaw), cos(own yaw)", "simulator truth (privileged)",
                             "calibrated camera heading of the own robot", "camera_pose"),
            ObservationField(f"{base + 4}-{base + 5}", "opponent x, opponent y (scaled to -1..1)", "simulator truth (privileged)",
                             "an opponent tracker (none runs live)", "opponent_tracker"),
        ]
    return fields


def inspect_checkpoint(path: Path | str) -> CheckpointInfo:
    """Describe a checkpoint from its metadata. The file name plays no part in the label."""
    file = _checkpoint_file(path)
    data = _read_zip_data(file)
    from autobot.sim import cli
    from autobot.sim.env import ACTION_PROFILES

    num_timesteps = _int_or_none(cli.checkpoint_ppo_settings(file)["num_timesteps"])
    shape = (data.get("observation_space") or {}).get("_shape") or []
    observation_size = _int_or_none(shape[0]) if len(shape) == 1 else None
    action_count = _int_or_none((data.get("action_space") or {}).get("n"))
    sidecar_path, sidecar = _sidecar(file)
    embedded = cli.checkpoint_observation_metadata(file) or {}

    kind, evidence, warnings = _classify(num_timesteps, sidecar_path, sidecar, file)
    profile, profile_source, action, action_source, layout_warnings = _layout(embedded, sidecar, observation_size, action_count)
    warnings += layout_warnings

    try:
        encoding = cli.resolve_previous_action_encoding(file)
        encoding_source = ("checkpoint metadata" if embedded.get("previous_action_encoding")
                           else "run effective_config.json" if cli.run_previous_action_encoding(file)
                           else "legacy default: the checkpoint predates versioning and carries no record")
    except ValueError as error:
        encoding, encoding_source = None, f"could not be determined: {error}"
        warnings.append(str(error))

    tactics = tuple(ACTION_PROFILES[action]["tactics"]) if action in ACTION_PROFILES else ()
    return CheckpointInfo(
        path=file, identifier=_identifier(file), kind=kind, evidence=evidence, num_timesteps=num_timesteps,
        observation_size=observation_size, action_count=action_count, action_profile=action, action_profile_source=action_source,
        tactics=tactics, previous_action_encoding=encoding, previous_action_encoding_source=encoding_source,
        observation_profile=profile, observation_profile_source=profile_source,
        requirements=observation_requirements(profile, action, encoding, observation_size), sidecar=sidecar_path, warnings=warnings)


# ----------------------------------------------------------------------------------------------
# LIVE SHADOW availability


def live_shadow_status(info: CheckpointInfo | None, hub) -> tuple[bool, list[str]]:
    """(available, missing inputs). Built on the hub's own list plus what this checkpoint needs.

    In this repository it always comes back unavailable: there is no live observation adapter."""
    missing = list(hub.live_shadow_missing())
    if info is None:
        missing.append("no checkpoint is selected")
    elif not info.layout_known:
        missing.append(f"this checkpoint's observation layout could not be determined from its metadata "
                       f"({info.observation_size} values), so its live inputs cannot even be listed")
    else:
        by_need: dict[str, list[str]] = {}
        for item in info.requirements:
            by_need.setdefault(item.needs, []).append(item.index)
        size = info.observation_size
        if "camera_pose" in by_need or "opponent_tracker" in by_need:
            pose = ", ".join(by_need.get("camera_pose", [])) or "none"
            opponent = ", ".join(by_need.get("opponent_tracker", [])) or "none"
            missing.append(f"this checkpoint ({size} values, {info.observation_profile}) was only ever fed SIMULATOR TRUTH for "
                           f"own pose (values {pose}) and opponent position (values {opponent}); nothing builds those from the camera")
        if "imu" in by_need:
            missing.append(f"this checkpoint reads values {', '.join(by_need['imu'])} as gyro (rad/s) and accelerometer (m/s²) "
                           "in the simulated IMU's axes and scaling; the live stream is raw sensor axes in deg/s with no "
                           "mounting orientation recorded, and nothing converts it")
        if "policy" in by_need:
            missing.append(f"this checkpoint reads its own previous tactic (values {', '.join(by_need['policy'])}); "
                           "nothing keeps that for a live run")
    missing = list(dict.fromkeys(missing))
    return not missing, missing


# ----------------------------------------------------------------------------------------------
# SIM: the checkpoint on simulator observations, driving the simulated robot only


@dataclass(frozen=True)
class ShadowDecision:
    index: int                              # decisions since SIM started
    round_index: int                        # simulated rounds since SIM started (1-based)
    sim_time_s: float                       # simulated time at which the observation was read
    observation_valid: bool
    observation_problem: str
    action: int | None
    tactic: str | None                      # None when the observation was not valid: nothing is proposed
    wheel_command: tuple[float, float] | None   # what the tactic decodes to in SIMULATOR wheel units (-1..1, negative = forward)
    probabilities: dict[str, float] | None
    round_end: str | None                   # set on the decision that ended a simulated round

    def proposed(self) -> dict[str, Any]:
        """Display data for the dashboard. Not a command: nothing reads this to drive anything."""
        return {"tactic": self.tactic,
                "wheel_command": None if self.wheel_command is None else [float(v) for v in self.wheel_command],
                "wheel_command_units": "simulator wheel command -1..1 (negative = forward); applied to the SIMULATED robot only",
                "sim_time_s": round(float(self.sim_time_s), 3), "observation_valid": bool(self.observation_valid),
                "observation_problem": self.observation_problem or None,
                "probabilities": None if self.probabilities is None else {k: round(float(v), 4) for k, v in self.probabilities.items()},
                "simulated_round": int(self.round_index)}


class SimShadowSession:
    """A checkpoint stepping the frozen simulator with its own actions. Plain Python; `load()` is slow."""

    def __init__(self, info: CheckpointInfo, config: Path = FROZEN_CONFIG, scenarios: Path = FROZEN_SCENARIOS,
                 scenario_index: int = SCENARIO_INDEX, seed: int = SEED) -> None:
        self.info = info
        self.config, self.scenarios, self.scenario_index, self.seed = Path(config), Path(scenarios), scenario_index, seed
        self.session = None
        self.step_s = 0.02
        self.decisions = 0
        self.round_index = 1
        self._round_over = False
        self._probabilities_work = True
        self._tactic_to_drive = None

    def load(self, progress: Callable[[str], None] = lambda text: None) -> None:
        if not self.info.layout_known:
            raise CheckpointError("the checkpoint's observation layout is not recorded, so the simulator cannot be set up for it")
        for needed in (self.config, self.scenarios):
            if not needed.is_file():
                raise CheckpointError(f"frozen simulator file missing: {needed}")
        progress("Importing the simulator and torch (about 15 s; up to two minutes the first time after a restart)…")
        started = time.monotonic()
        from autobot.sim import cli
        from autobot.sim.env import tactic_to_drive
        import torch  # noqa: F401 - the slow import, done here so the next message is true

        progress(f"Libraries imported in {time.monotonic() - started:.0f} s. Loading the checkpoint into the frozen simulator…")
        # The same construction as scripts/live_round_diagnostic.py make_session, with the observation
        # and action profile taken from this checkpoint's metadata instead of being fixed.
        self.session = cli.ReplaySession(Namespace(
            config=str(self.config), seed=self.seed, model=str(self.info.path), profile=self.info.observation_profile,
            action_profile=self.info.action_profile, eval_scenarios=str(self.scenarios), scenario_index=self.scenario_index,
            opponent=OPPONENT_MODE, opponent_chassis=OPPONENT_CHASSIS, reward_profile="outcome_only", tactic_hold_ms=0.0))
        self._tactic_to_drive = tactic_to_drive
        self.step_s = 1.0 / float(self.session.sim.cfg["simulation"]["control_hz"])
        progress(f"Loaded in {time.monotonic() - started:.0f} s. Running on simulator observations.")

    def _probabilities(self, observation: np.ndarray) -> dict[str, float] | None:
        if not self._probabilities_work:
            return None
        try:
            import torch
            policy = self.session.model.policy
            with torch.no_grad():
                tensor, _ = policy.obs_to_tensor(observation)
                values = policy.get_distribution(tensor).distribution.probs[0].cpu().numpy()
            return {name: float(value) for name, value in zip(self.session.env.tactics, values)}
        except Exception:
            self._probabilities_work = False        # probabilities are a nicety; the decision does not depend on them
            return None

    def step(self) -> ShadowDecision:
        """One decision: read the simulator observation, ask the policy, apply its action to the SIMULATED robot."""
        if self.session is None:
            raise CheckpointError("load() has not been called")
        session = self.session
        if self._round_over:
            session.reset()
            self.round_index += 1
            self._round_over = False
        env, model = session.env, session.model
        sim_time = float(session.sim.data.time)
        observation = np.asarray(env._obs())
        expected = tuple(model.observation_space.shape)
        problem = ""
        if observation.shape != expected:
            problem = f"observation has shape {observation.shape}, the checkpoint expects {expected}"
        elif not bool(np.all(np.isfinite(observation))):
            problem = "observation contains non-finite values"
        action = tactic = wheels = probabilities = None
        if not problem:
            action = int(model.predict(observation, deterministic=True)[0])
            tactic = str(env.tactics[action])
            wheels = tuple(float(v) for v in self._tactic_to_drive(action, env.action_profile))
            probabilities = self._probabilities(observation)
        # With no valid observation nothing is proposed; the simulated robot is told to stop for that step.
        done, _ = session.step(forced_action=0 if action is None else action)
        self.decisions += 1
        if done:
            self._round_over = True
        return ShadowDecision(self.decisions, self.round_index, sim_time, not problem, problem, action, tactic, wheels,
                              probabilities, session.end_reason if done else None)


class SimShadowRunner(QThread):
    """Runs a `SimShadowSession` off the UI thread and tells the hub what to display.

    The slow part (torch import, checkpoint load) happens in `run()`. Hub calls are made from the
    thread that owns this object (the UI thread), through queued signals, at most 5 times a second.
    """

    progress = Signal(str)
    loaded = Signal()
    decision = Signal(object)               # ShadowDecision, one per simulator decision
    failed = Signal(str)

    def __init__(self, hub, info: CheckpointInfo, parent=None, realtime: bool = True, max_decisions: int | None = None) -> None:
        super().__init__(parent)
        self.hub = hub
        self.info = info
        self.realtime = realtime
        self.max_decisions = max_decisions
        self.latest: ShadowDecision | None = None
        self.decisions_seen = 0
        self.error: str | None = None
        self._running = True
        self._accepting = True
        self._reported = False
        self._last_report = 0.0
        self.decision.connect(self._on_decision)
        self.failed.connect(self._on_failed)
        self.finished.connect(self._report_off)

    def stop(self) -> None:
        """Ask the loop to end and tell the dashboard the policy is OFF. Call from the UI thread."""
        self._running = False
        self._accepting = False
        self._report_off()

    def run(self) -> None:
        try:
            session = SimShadowSession(self.info)
            session.load(self.progress.emit)
        except Exception as error:
            self.failed.emit(f"{type(error).__name__}: {error}")
            return
        if not self._running:
            return
        self.loaded.emit()
        reference, done = time.monotonic(), 0
        while self._running:
            try:
                decision = session.step()
            except Exception as error:
                self.failed.emit(f"{type(error).__name__}: {error}")
                return
            self.decision.emit(decision)
            done += 1
            if self.max_decisions is not None and done >= self.max_decisions:
                return
            if self.realtime:
                wait = reference + done * session.step_s - time.monotonic()
                if wait > 0:
                    time.sleep(wait)
                elif wait < -0.5:                   # fell behind: do not try to catch up
                    reference, done = time.monotonic(), 0

    # ---- UI-thread side ----

    def _on_decision(self, decision: ShadowDecision) -> None:
        if not self._accepting:
            return
        self.latest = decision
        self.decisions_seen += 1
        now = time.monotonic()
        if self._reported and now - self._last_report < HUB_REPORT_PERIOD_S:
            return
        self._last_report = now
        ok, _ = self.hub.set_policy("SIM", SIM_INPUT_SOURCE, self.info.label, proposed=decision.proposed())
        self._reported = self._reported or ok

    def _on_failed(self, message: str) -> None:
        self.error = message
        self._accepting = False

    def _report_off(self) -> None:
        self._accepting = False
        if self._reported:
            self._reported = False
            self.hub.set_policy("OFF", reason="SIM stopped" if self.error is None else f"SIM stopped: {self.error}")
