"""Gymnasium tactic-selection environment over the shared MuJoCo simulation."""
from __future__ import annotations
import math
import gymnasium as gym
import numpy as np
import mujoco
from gymnasium import spaces
from .core import AutoBotSim
from .scenarios import load_scenario_file, sample_curriculum, validate_curriculum

TACTICS = ("stop", "advance", "reverse", "turn_left", "turn_right")
ARC_TACTICS = TACTICS + ("forward_left","forward_right")
ACTION_PROFILES = {
    "legacy5":{"tactics":TACTICS,"previous_action_slots":5},
    "experimental5_v1":{"tactics":TACTICS,"previous_action_slots":7},
    "arc7_v1":{"tactics":ARC_TACTICS,"previous_action_slots":7},
}
OPPONENT_MODES = ("scripted","static","faster_follow","stop_and_go","direction_changes")
# Versioned previous-action observation encoding. Both versions keep the same observation shape
# and channel layout, so a space/shape check can NOT tell them apart: the choice is persisted in
# run metadata and inside every checkpoint (see cli.checkpoint_observation_metadata).
#   onehot_v1      - one-hot of the last applied tactic (the only behaviour before versioning).
#   masked_zero_v1 - the same channels held at exactly 0.0 at reset and on every step.
PREVIOUS_ACTION_ENCODINGS = ("onehot_v1","masked_zero_v1")
LEGACY_PREVIOUS_ACTION_ENCODING = "onehot_v1"
IMU_OBSERVATION_CHANNELS = 6


def previous_action_slice(action_profile="legacy5") -> slice:
    """Observation channels reserved for the previous tactic (identical in every encoding)."""
    slots=int(ACTION_PROFILES[action_profile]["previous_action_slots"])
    return slice(IMU_OBSERVATION_CHANNELS,IMU_OBSERVATION_CHANNELS+slots)


def manual_drive_command(direction: str) -> tuple[float,float]:
    """Keyboard drive mapping, kept testable outside the Pygame event loop."""
    return {"forward":(-1.0,-1.0),"reverse":(0.65,0.65),
            "left":(0.7,-0.7),"right":(-0.7,0.7),"stop":(0.0,0.0)}[direction]


def arcade_drive(forward=False, reverse=False, left=False, right=False) -> tuple[float,float]:
    """Continuous human throttle/steering mixer; supports driving on a curve."""
    throttle=-1.0 if forward else 0.65 if reverse else 0.0
    steering=0.7 if left else -0.7 if right else 0.0
    if throttle and steering: steering*=0.55
    return tuple(float(value) for value in np.clip([throttle+steering,throttle-steering],-1,1))


def tactic_to_drive(action: int, action_profile="legacy5") -> tuple[float, float]:
    """Deterministic low-level controller shared by training, tests, and replay."""
    # Wheel joint axes make negative wheel velocity geometric chassis-forward.
    commands=((0,0), (-1,-1), (0.65,0.65), (0.7,-0.7), (-0.7,0.7), (-0.55,-1.0), (-1.0,-0.55))
    profile=ACTION_PROFILES[action_profile]
    if not 0<=int(action)<len(profile["tactics"]): raise ValueError(f"Action {action} invalid for {action_profile}")
    return commands[int(action)]


def scripted_opponent_drive(sim: AutoBotSim) -> tuple[float,float]:
    """Privileged scripted baseline for manual play, never part of policy observations."""
    own=sim.truth(1); target=sim.truth(0)
    delta=target["position_m"][:2]-own["position_m"][:2]
    cfg=sim.cfg["opponent"]
    if float(np.linalg.norm(delta)) <= cfg["contact_yield_distance_m"]:
        return 0.0,0.0
    desired=math.atan2(delta[1],delta[0])
    error=math.atan2(math.sin(desired-own["yaw_rad"]),math.cos(desired-own["yaw_rad"]))
    if abs(error) <= math.radians(cfg["align_tolerance_deg"]):
        # Wheel convention in this model uses negative velocity for geometric +x-forward.
        speed=-float(cfg["forward_command"]); return speed,speed
    turn=float(cfg["turn_command"])
    return (turn,-turn) if error > 0 else (-turn,turn)


def opponent_drive(sim: AutoBotSim, mode: str) -> tuple[float,float]:
    """Named diagnostic opponents; `scripted` is the preserved original baseline."""
    if mode=="static": return 0.0,0.0
    base=scripted_opponent_drive(sim)
    if mode=="scripted": return base
    if mode=="faster_follow":
        if base==(0.0,0.0): return base
        scale=1.75
        return tuple(float(np.clip(command*scale,-1,1)) for command in base)
    if mode=="stop_and_go":
        return base if int(sim.data.time//0.75)%2==0 else (0.0,0.0)
    if mode=="direction_changes":
        phase=int(sim.data.time//1.0)%4
        if phase==1: return (0.55,-0.55)
        if phase==3: return (-0.35,-0.35)
        return base
    raise ValueError(f"Unknown opponent mode: {mode}")


class AutoBotEnv(gym.Env):
    metadata = {"render_modes": []}
    def __init__(self, config_path="config/sim.yaml", observation_profile="hardware_plausible", seed=None,
                 opponent_model=None, opponent_mode="scripted", evaluation_scenarios=None,
                 training_curriculum=None, reward_profile="outcome_only", curriculum_start_episode=0,
                 collision_overlay=False, action_profile="legacy5",
                 previous_action_encoding=LEGACY_PREVIOUS_ACTION_ENCODING):
        super().__init__()
        if observation_profile not in ("hardware_plausible", "sim_debug"):
            raise ValueError("observation_profile must be hardware_plausible or sim_debug")
        self.profile = observation_profile
        if opponent_mode not in OPPONENT_MODES:
            raise ValueError(f"opponent_mode must be one of {OPPONENT_MODES}")
        self.opponent_mode=opponent_mode
        if action_profile not in ACTION_PROFILES: raise ValueError(f"Unknown action profile: {action_profile}")
        self.action_profile=action_profile; self.tactics=ACTION_PROFILES[action_profile]["tactics"]
        self.previous_action_slots=int(ACTION_PROFILES[action_profile]["previous_action_slots"])
        if previous_action_encoding not in PREVIOUS_ACTION_ENCODINGS:
            raise ValueError(f"previous_action_encoding must be one of {PREVIOUS_ACTION_ENCODINGS}")
        self.previous_action_encoding=previous_action_encoding
        self.sim = AutoBotSim(config_path, seed, opponent_model=opponent_model,
                              collision_overlay=collision_overlay)
        profiles=self.sim.cfg["training"].get("reward_profiles") or {"outcome_only":self.sim.cfg["training"]["rewards"]}
        if reward_profile not in profiles: raise ValueError(f"Unknown reward profile: {reward_profile}")
        self.reward_profile=reward_profile
        self.action_space = spaces.Discrete(len(self.tactics))
        # Hardware profile: gyro xyz, accelerometer xyz, previous tactic one-hot.
        size = 6+self.previous_action_slots+(0 if observation_profile=="hardware_plausible" else 6)
        self.observation_space = spaces.Box(-1.0, 1.0, shape=(size,), dtype=np.float32)
        self.steps = 0; self.last_action = 0
        self.evaluation_scenarios=list(evaluation_scenarios or [])
        curriculum=load_scenario_file(training_curriculum) if isinstance(training_curriculum,(str,bytes)) else training_curriculum
        self.training_curriculum=validate_curriculum(curriculum) if curriculum else None
        self.curriculum_start_episode=int(curriculum_start_episode)
        self.curriculum_episode=self.curriculum_start_episode
        self.current_stage="none"; self.current_group="default"; self.current_scenario="default_spawn"
        self._scenario_index=0
        self.episode_done = False
        self.max_steps = round(self.sim.cfg["simulation"]["episode_seconds"] * self.sim.cfg["simulation"]["control_hz"])

    def _obs(self):
        imu = self.sim.imu()
        ic = self.sim.cfg["imu"]
        base = list(imu.gyro_rad_s/ic["gyro_range_rad_s"]) + list(imu.accel_m_s2/ic["accel_range_m_s2"])
        if self.previous_action_encoding=="masked_zero_v1":
            # Layout-preserving mask: the policy never sees its previous tactic.
            base += [0.0]*self.previous_action_slots
        else:
            base += list(np.eye(self.previous_action_slots, dtype=np.float32)[self.last_action])
        if self.profile == "sim_debug":
            own, opp = self.sim.truth(0), self.sim.truth(1)
            w, h = self.sim.cfg["arena"]["width_m"], self.sim.cfg["arena"]["height_m"]
            # Explicit simulation-only privileged truth, never present in standard profile.
            base += [2*own["position_m"][0]/w-1, 2*own["position_m"][1]/h-1,
                     math.sin(own["yaw_rad"]), math.cos(own["yaw_rad"]),
                     2*opp["position_m"][0]/w-1, 2*opp["position_m"][1]/h-1]
        return np.clip(base, -1, 1).astype(np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.sim.rng = np.random.default_rng(seed)
            self.curriculum_episode=self.curriculum_start_episode
        self.sim.reset(); self.steps = 0; self.last_action = 0; self.episode_done = False
        scenario=(options or {}).get("scenario")
        if scenario is None and self.training_curriculum:
            # Episode-indexed RNG makes the curriculum reproducible across a clean resume.
            curriculum_seed=int(self.training_curriculum.get("seed",0))+self.curriculum_episode
            scenario,self.current_stage,self.current_group=sample_curriculum(
                self.training_curriculum,self.curriculum_episode,np.random.default_rng(curriculum_seed))
            self.curriculum_episode += 1
        if scenario is None and self.evaluation_scenarios:
            scenario=self.evaluation_scenarios[self._scenario_index % len(self.evaluation_scenarios)]
            self._scenario_index += 1
        if scenario is not None:
            for bot,key in ((0,"player"),(1,"opponent")):
                pose=scenario[key]
                self.sim.set_pose(bot,float(pose[0]),float(pose[1]),float(pose[2]))
        if scenario is None: self.current_stage,self.current_group="none","default"
        self.current_scenario=scenario.get("name","unnamed") if scenario else "default_spawn"
        return self._obs(), {"observation_profile": self.profile,"reward_profile":self.reward_profile,
                             "previous_action_encoding":self.previous_action_encoding,
                             "scenario": scenario.get("name") if scenario else "default_spawn",
                             "curriculum_stage":self.current_stage,"scenario_group":self.current_group}

    def _wall_contact(self, bot=0):
        for index in range(self.sim.data.ncon):
            contact=self.sim.data.contact[index]
            names=[mujoco.mj_id2name(self.sim.model,mujoco.mjtObj.mjOBJ_GEOM,int(geom)) or ""
                   for geom in (contact.geom1,contact.geom2)]
            if any(name.startswith("wall_") for name in names) and any(name.startswith(f"bot{bot}_") for name in names):
                return True
        return False

    def _robots_contact(self):
        for index in range(self.sim.data.ncon):
            contact=self.sim.data.contact[index]
            names=[mujoco.mj_id2name(self.sim.model,mujoco.mjtObj.mjOBJ_GEOM,int(geom)) or ""
                   for geom in (contact.geom1,contact.geom2)]
            if any(name.startswith("bot0_") for name in names) and any(name.startswith("bot1_") for name in names):
                return True
        return False

    def step(self, action, opponent_drive_override=None):
        if self.episode_done: raise RuntimeError("step() called after episode end; call reset()")
        self.last_action = int(action); self.steps += 1
        own0=self.sim.truth(0)["position_m"][:2].copy()
        red_drive=opponent_drive(self.sim,self.opponent_mode) if opponent_drive_override is None else opponent_drive_override
        self.sim.step(tactic_to_drive(action,self.action_profile),red_drive)
        own, opp = self.sim.truth(0), self.sim.truth(1)
        distance = np.linalg.norm(own["position_m"][:2]-opp["position_m"][:2])
        # First comparison reward is deliberately sparse and outcome-only. No
        # displacement, distance, action-label, or wall-contact shaping is applied.
        terms={"own_fall":0.0,"opponent_fall":0.0,"simultaneous_fall":0.0,"timeout":0.0}
        reward = 0.0
        event=self.sim.arena_event()
        terminated = bool(event and event.match_terminal)
        truncated = self.steps >= self.max_steps or bool(event and event.simulation_bug)
        profiles=self.sim.cfg["training"].get("reward_profiles")
        rewards=profiles[self.reward_profile] if profiles else self.sim.cfg["training"]["rewards"]
        if event is not None and event.reason == "Pit fall" and event.bot == 0:
            terms["own_fall"]=float(rewards["own_pit_fall"])
        elif event is not None and event.reason == "Pit fall" and event.bot == 1:
            terms["opponent_fall"]=float(rewards["opponent_pit_fall"])
        elif event is not None and event.reason == "Simultaneous pit fall":
            terms["simultaneous_fall"]=float(rewards["simultaneous_pit_fall"])
        reward=float(sum(terms.values()))
        self.episode_done=terminated or truncated
        wall_contact=self._wall_contact(0)
        stationary_wall=bool(wall_contact and action!=0 and np.linalg.norm(own["position_m"][:2]-own0)<1e-4)
        return self._obs(), reward, terminated, truncated, {"tactic": self.tactics[self.last_action], "distance_to_opponent_m": float(distance),
                "round_end_reason": event.reason if event else ("Timeout" if truncated else None),
                "round_end_bot": event.bot if event else None,"simulation_bug": bool(event and event.simulation_bug),
                "reward_terms":terms,"wall_contact":wall_contact,"stationary_wall_contact":stationary_wall,
                "curriculum_stage":self.current_stage,"scenario_group":self.current_group,
                "scenario":self.current_scenario,
                "blue_drive":tactic_to_drive(action,self.action_profile),"opponent_drive":tuple(red_drive),
                "opponent_contact":self._robots_contact()}


class ScriptedBaseline:
    def predict(self, observation, deterministic=True):
        return np.array(1), None


class StopBaseline:
    def predict(self, observation, deterministic=True):
        return np.array(0), None
