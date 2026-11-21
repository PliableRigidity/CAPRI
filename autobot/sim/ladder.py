"""Heading ladder: near-pit starts whose heading offset grows rung by rung. Simulation-only.

Geometry is the push sweep of scripts/ordinary_start_diagnostic.py: red sits `distance` from a pit centre on
the arena diagonal and faces blue; blue is 0.18 m behind red on the line through the pit, with its heading
rotated `heading_offset` off that line. At offset 0 a plain advance wins; the ladder asks whether a policy can
learn to turn first as the offset grows.

Nothing here changes the frozen environment: LadderEnv only chooses which start pose each training episode
gets. Physics, observations, actions and the outcome-only reward are those of AutoBotEnv.
"""
from __future__ import annotations

import math
from collections import Counter

import numpy as np

from .env import AutoBotEnv
from .scenarios import validate_pose_scenario

LADDER_SCHEMA = "heading_ladder_v1"
STANDOFF_M = 0.18
# Pit centre and the unit direction from it towards the arena centre (both pits are on a diagonal).
PITS = {"top_left": ((0.0, 1.15), (1.0, -1.0)), "bottom_right": ((1.15, 0.0), (-1.0, 1.0))}
SPLITS = ("train", "validation", "holdout")
TURN_TACTICS = ("turn_left", "turn_right")
# A win "involves an initial turn" when blue rotated towards the push line by at least this share of the
# starting heading offset before it first touched red.
INITIAL_TURN_SHARE = 0.5


def wrap(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


def ladder_start(pit, distance_m, heading_offset_deg, split, rung=None):
    """One start: red `distance_m` from the pit, blue behind it, blue's heading rotated by the offset."""
    centre, inward = PITS[pit]
    centre = np.asarray(centre, dtype=float); inward = np.asarray(inward, dtype=float) / math.sqrt(2)
    toward_pit = math.atan2(-inward[1], -inward[0])
    red = centre + inward * distance_m; blue = red + inward * STANDOFF_M
    yaw = toward_pit + math.radians(heading_offset_deg)
    start = {"name": f"ladder_{split}_{pit}_d{distance_m:.2f}_h{heading_offset_deg:+g}", "group": f"ladder_{split}",
             "pit": pit, "distance_m": float(distance_m), "heading_offset_deg": float(heading_offset_deg),
             "player": [round(float(blue[0]), 6), round(float(blue[1]), 6), round(wrap(yaw), 6)],
             # Red faces blue, as it does once the scripted opponent has turned to chase.
             "opponent": [round(float(red[0]), 6), round(float(red[1]), 6), round(wrap(toward_pit + math.pi), 6)]}
    if rung is not None: start["rung"] = int(rung)
    return start


def build_ladder(rung_offsets_deg, *, train_distances_m=(0.34, 0.46, 0.58), validation_distances_m=(0.40, 0.52),
                 holdout_distances_m=(0.37, 0.43, 0.55, 0.64), holdout_offsets_deg=None, warmup_offset_deg=None,
                 seed=7, review_fraction=0.4, pits=None):
    """Train / validation / hold-out starts for a ladder with the given absolute heading offsets.

    train       both pits x train distances x both signs at the rung offset, minus two held-out combinations
    validation  both pits x validation distances x both signs at the rung offset (gates the next rung)
    holdout     (a) the two combinations per rung left out of training, one on each pit, and
                (b) offsets between the rungs at distances used by neither training nor validation,
                    on both pits and in both directions. Never used for training, gating or selection.

    `warmup_offset_deg` adds a first rung that a plain advance already wins. It lets the policy learn to push in
    this geometry before any turn is needed; it is gated on win rate only and never counts as a cleared rung.

    `pits` restricts training and validation to those pits (default: both). The hold-out always covers both,
    so with one training pit the other pit is a pure transfer test.
    """
    pits = list(PITS) if pits is None else list(pits)
    if not pits or any(pit not in PITS for pit in pits): raise ValueError(f"pits must be chosen from {list(PITS)}")
    gated = [float(value) for value in rung_offsets_deg]
    offsets = ([float(warmup_offset_deg)] if warmup_offset_deg is not None else []) + gated
    if offsets != sorted(set(offsets)) or offsets[0] <= 0: raise ValueError("rung offsets must be positive and increasing")
    if holdout_offsets_deg is None:
        holdout_offsets_deg = [round((low + high) / 2, 1) for low, high in zip(gated, gated[1:])]
    rungs = []; holdout = []
    for index, offset in enumerate(offsets):
        sign = 1 if index % 2 == 0 else -1
        distances = list(train_distances_m)
        left_out = {("top_left", distances[index % len(distances)], sign),
                    ("bottom_right", distances[(index + 1) % len(distances)], -sign)}
        train = []
        for pit in PITS:
            for distance in distances:
                for direction in (1, -1):
                    if (pit, distance, direction) in left_out:
                        holdout.append(dict(ladder_start(pit, distance, direction * offset, "holdout"),
                                            holdout_kind="combination", nearest_rung=index))
                    elif pit in pits: train.append(ladder_start(pit, distance, direction * offset, "train", index))
        validation = [ladder_start(pit, distance, direction * offset, "validation", index)
                      for pit in pits for distance in validation_distances_m for direction in (1, -1)]
        rungs.append({"index": index, "abs_heading_offset_deg": offset,
                      "warmup": bool(warmup_offset_deg is not None and index == 0), "train": train, "validation": validation})
    for offset in holdout_offsets_deg:
        nearest = max((index for index, value in enumerate(offsets) if value <= offset), default=0)
        for pit in PITS:
            for distance in holdout_distances_m:
                for direction in (1, -1):
                    holdout.append(dict(ladder_start(pit, distance, direction * offset, "holdout"),
                                        holdout_kind="level", nearest_rung=nearest))
    return validate_ladder({"schema": LADDER_SCHEMA, "seed": int(seed), "standoff_m": STANDOFF_M,
                            "review_fraction": float(review_fraction), "training_pits": pits, "rung_offsets_deg": offsets,
                            "train_distances_m": list(train_distances_m), "validation_distances_m": list(validation_distances_m),
                            "holdout_distances_m": list(holdout_distances_m), "holdout_offsets_deg": list(holdout_offsets_deg),
                            "rungs": rungs, "holdout": holdout})


def validate_ladder(ladder):
    if not isinstance(ladder, dict) or ladder.get("schema") != LADDER_SCHEMA or not ladder.get("rungs"):
        raise ValueError(f"Not a {LADDER_SCHEMA} ladder")
    if not 0.0 <= float(ladder.get("review_fraction", 0.0)) < 1.0: raise ValueError("review_fraction must be in [0, 1)")
    previous = 0.0; seen = {}
    for index, rung in enumerate(ladder["rungs"]):
        if rung.get("index") != index: raise ValueError("Ladder rung indices must count from 0")
        if not rung["abs_heading_offset_deg"] > previous: raise ValueError("Ladder rung offsets must increase")
        previous = rung["abs_heading_offset_deg"]
        if not rung.get("train") or not rung.get("validation"): raise ValueError("Every rung needs train and validation starts")
        for split in ("train", "validation"):
            for start in rung[split]:
                validate_pose_scenario(start)
                if abs(abs(start["heading_offset_deg"]) - rung["abs_heading_offset_deg"]) > 1e-9:
                    raise ValueError(f"{start['name']} is not at its rung's heading offset")
                seen.setdefault(_pose_key(start), []).append(split)
    for start in ladder.get("holdout", []):
        validate_pose_scenario(start); seen.setdefault(_pose_key(start), []).append("holdout")
    shared = {key: splits for key, splits in seen.items() if len(set(splits)) > 1}
    if shared: raise ValueError(f"Starts appear in more than one split: {sorted(shared)[:3]}")
    return ladder


def _pose_key(start):
    return tuple(round(float(value), 4) for value in (*start["player"], *start["opponent"]))


def sample_ladder_start(ladder, frontier, rng):
    """(rung sampled, start). Mostly the frontier rung; `review_fraction` of episodes revisit a cleared rung."""
    frontier = int(frontier); rung = frontier
    if frontier > 0 and rng.random() < float(ladder.get("review_fraction", 0.0)): rung = int(rng.integers(frontier))
    starts = ladder["rungs"][rung]["train"]
    return rung, starts[int(rng.integers(len(starts)))]


class LadderEnv(AutoBotEnv):
    """AutoBotEnv whose training starts come from the current ladder rung. Everything else is unchanged."""

    def __init__(self, ladder, *args, start_rung=0, ladder_stream=0, **kwargs):
        if kwargs.get("training_curriculum") is not None: raise ValueError("LadderEnv replaces the training curriculum")
        super().__init__(*args, **kwargs)
        self.ladder = validate_ladder(ladder); self.ladder_stream = int(ladder_stream)
        self.rung = 0; self.set_rung(start_rung); self.ladder_episode = 0; self.sampled_rung = None

    def set_rung(self, rung):
        if not 0 <= int(rung) < len(self.ladder["rungs"]): raise ValueError(f"No ladder rung {rung}")
        self.rung = int(rung)

    def reset(self, *, seed=None, options=None):
        options = dict(options or {}); sampled = None
        if options.get("scenario") is None:
            # Episode-indexed RNG: the start order depends only on (ladder seed, stream, episode, frontier).
            rng = np.random.default_rng([int(self.ladder["seed"]), self.ladder_stream, self.ladder_episode])
            sampled, options["scenario"] = sample_ladder_start(self.ladder, self.rung, rng)
            self.ladder_episode += 1
        observation, info = super().reset(seed=seed, options=options)
        self.sampled_rung = sampled
        if sampled is not None:
            self.current_stage = f"rung_{self.rung}"; self.current_group = f"rung_{sampled}"
            info.update(curriculum_stage=self.current_stage, scenario_group=self.current_group)
        else: self.current_stage, self.current_group = "none", options["scenario"].get("group", "ungrouped")
        return observation, info


def gate_passes(policy_wins, advance_wins, episodes, *, min_win_rate=0.75, min_margin=0.25, warmup=False):
    """Does the policy beat Always-Advance on a rung's validation starts by a clear margin?

    A warm-up rung is one a plain advance already wins, so only the win rate is required there.
    """
    if episodes <= 0: return False
    enough = policy_wins >= math.ceil(min_win_rate * episodes - 1e-9)
    ahead = warmup or policy_wins - advance_wins >= max(1, math.ceil(min_margin * episodes - 1e-9))
    return bool(enough and ahead)


class RungGate:
    """A rung is cleared only after `required` consecutive validations pass (one lucky validation is not enough)."""

    def __init__(self, required=2):
        self.required = int(required); self.streak = 0

    def update(self, passed):
        self.streak = self.streak + 1 if passed else 0
        if self.streak >= self.required:
            self.streak = 0
            return True
        return False


def push_line_error(sim, pit):
    """Signed angle (rad) from the red-to-pit direction to blue's heading. 0 means blue faces along the push line."""
    red = sim.truth(1)["position_m"][:2]
    return wrap(sim.truth(0)["yaw_rad"] - math.atan2(pit[1] - red[1], pit[0] - red[0]))


def ladder_rollout(model, env, scenario, seed, deterministic=True):
    """One reproducible episode on an existing environment, with what blue did before it first touched red."""
    if not deterministic:
        import torch
        torch.manual_seed(int(seed))
    observation, _ = env.reset(seed=int(seed), options={"scenario": scenario})
    if hasattr(model, "bind"): model.bind(env)     # privileged reference controller reads simulator truth
    sim = env.sim; arena = sim.cfg["arena"]; pits = np.asarray(arena["pit_centres_m"], dtype=float)
    red_start = sim.truth(1)["position_m"][:2].copy()
    pit = pits[int(np.argmin(np.linalg.norm(pits - red_start, axis=1)))]
    start_error = push_line_error(sim, pit); toward_sign = -1.0 if start_error > 0 else 1.0
    yaw = sim.truth(0)["yaw_rad"]; turned_toward = 0.0
    actions = []; first_contact = None; before = None; info = {}; red_closest = float("inf")
    while True:
        action = int(model.predict(observation, deterministic=deterministic)[0])
        observation, _, terminated, truncated, info = env.step(action); actions.append(action)
        now = sim.truth(0)["yaw_rad"]; turned_toward += toward_sign * wrap(now - yaw); yaw = now
        red_closest = min(red_closest, float(np.linalg.norm(pit - sim.truth(1)["position_m"][:2]) - arena["pit_radius_m"]))
        if first_contact is None and info["opponent_contact"]:
            first_contact = len(actions)
            before = {"turned_toward": turned_toward, "error": push_line_error(sim, pit)}
        if terminated or truncated: break
    if before is None: before = {"turned_toward": turned_toward, "error": None}
    reason = info.get("round_end_reason")
    if reason == "Pit fall": outcome = "own_falls" if info["round_end_bot"] == 0 else "wins"
    elif reason == "Simultaneous pit fall": outcome = "draws"
    elif reason == "Timeout": outcome = "timeouts"
    else: outcome = "simulation_bugs"
    tactics = [env.tactics[action] for action in actions]; pre = tactics[:first_contact] if first_contact else tactics
    control_hz = sim.cfg["simulation"]["control_hz"]
    return {"scenario": scenario["name"], "group": scenario.get("group", "ungrouped"), "seed": int(seed), "outcome": outcome,
            "steps": len(actions), "simulated_seconds": len(actions) / control_hz,
            "first_contact_step": first_contact,
            "heading_error_start_deg": math.degrees(start_error),
            "heading_error_at_first_contact_deg": None if before["error"] is None else math.degrees(before["error"]),
            "turned_toward_push_line_before_contact_deg": math.degrees(before["turned_toward"]),
            "turn_decisions_before_contact": sum(tactic in TURN_TACTICS for tactic in pre),
            "decisions_before_contact": len(pre), "first_tactic": tactics[0],
            "initial_turn": bool(abs(start_error) > math.radians(3)
                                 and before["turned_toward"] >= INITIAL_TURN_SHARE * abs(start_error)),
            "tactic_switches_per_s": sum(a != b for a, b in zip(actions, actions[1:])) / max(len(actions) / control_hz, 1e-9),
            "opponent_min_pit_edge_distance_m": red_closest,
            "action_counts": dict(Counter(tactics)), "deterministic": bool(deterministic),
            **{key: scenario[key] for key in ("pit", "distance_m", "heading_offset_deg", "rung", "holdout_kind", "nearest_rung")
               if key in scenario}}


def summarise_episodes(episodes):
    counts = Counter(episode["outcome"] for episode in episodes)
    wins = [episode for episode in episodes if episode["outcome"] == "wins"]
    return {"episodes": len(episodes), "wins": counts["wins"], "own_falls": counts["own_falls"], "timeouts": counts["timeouts"],
            "draws": counts["draws"], "simulation_bugs": counts["simulation_bugs"],
            "wins_with_initial_turn": sum(episode["initial_turn"] for episode in wins),
            "wins_with_a_turn_before_contact": sum(episode["turn_decisions_before_contact"] > 0 for episode in wins),
            "median_win_seconds": float(np.median([episode["simulated_seconds"] for episode in wins])) if wins else None,
            "median_turn_decisions_before_contact_in_wins":
                float(np.median([episode["turn_decisions_before_contact"] for episode in wins])) if wins else None}


def run_starts(model, env, starts, seeds, deterministic=True):
    """Every start once per seed (the seed changes only the IMU noise the policy sees)."""
    return [ladder_rollout(model, env, start, int(seed) + index, deterministic)
            for seed in seeds for index, start in enumerate(starts)]


class GreedyActor:
    """NumPy copy of a Stable-Baselines3 MlpPolicy actor, for fast greedy rollouts in evaluation.

    `model.predict` costs about three times an environment step; this is the same argmax over the same
    weights without the tensor round-trip. Greedy only: it cannot sample, and it is never used for training.
    """

    def __init__(self, model):
        import torch
        policy = model.policy
        layers = [module for module in policy.mlp_extractor.policy_net if isinstance(module, torch.nn.Linear)]
        activations = [module for module in policy.mlp_extractor.policy_net if not isinstance(module, torch.nn.Linear)]
        if not all(isinstance(module, torch.nn.Tanh) for module in activations): raise ValueError("GreedyActor supports tanh actors only")
        self.hidden = [(layer.weight.detach().numpy().astype(np.float64), layer.bias.detach().numpy().astype(np.float64)) for layer in layers]
        self.output = (policy.action_net.weight.detach().numpy().astype(np.float64), policy.action_net.bias.detach().numpy().astype(np.float64))

    def logits(self, observation):
        value = np.asarray(observation, dtype=np.float64).reshape(-1)
        for weight, bias in self.hidden: value = np.tanh(weight @ value + bias)
        return self.output[0] @ value + self.output[1]

    def predict(self, observation, deterministic=True):
        if not deterministic: raise ValueError("GreedyActor is greedy only")
        return np.array(int(np.argmax(self.logits(observation)))), None
