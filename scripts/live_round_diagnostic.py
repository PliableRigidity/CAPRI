"""Why did blue spin in a recorded live round? Offline diagnostic of a human-driven play log. Trains nothing.

    python scripts/live_round_diagnostic.py all

Replays the newest timestamped `reference-normal-spawn-*.jsonl` input log (red = the recorded human commands,
replayed open loop from the same start and seed) against three blues: the frozen combined clone, the scripted
demonstration teacher (`FlankController`) and Always-Advance. Then:

  replay   the recorded rounds, the spin segment, teacher labels on the clone's own states
  cases    three matched cases from the normal spawn: red still, red moves to arena -x, red moves to arena +x
  probe    blue held fixed at the saved state just before the spin, red moved over every valid position
  play     the real viewer loop (headless, synthetic keys): does red's live pose reach blue's observation?
  figures  PNG pictures of the above
  case     write the regression case (config/regression_cases/)

Red's recorded commands were a person's reactions to the clone. Against any other blue they are a fixed input
replay, not a live opponent, so the teacher and Always-Advance rows say what those controllers do against this
exact input, not how they would fare against the person.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import re
import sys
from argparse import Namespace
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(Path(__file__).resolve().parent)]

CONFIG = "runs/ppo-curriculum-overnight/effective_sim.yaml"
MODEL = "runs/combined-clone-balanced-dagger-seed2/final_model.zip"
SCENARIOS = "runs/ppo-curriculum-overnight/evaluation_scenarios.json"
SCENARIO_INDEX = 2
SEED = 10007
LIVE = ROOT / "runs/overnight-2026-11-17/live"
OUT = ROOT / "runs/live-round-diagnostic-2026-11-18"
CASE = ROOT / "config/regression_cases/live_spin_2026_11_18.json"
STEP_S = 0.02
NOSE_ON_DEG = 12.0            # the teacher's own "aligned, now advance" tolerance
SPIN_DEG = 720.0              # a spin: this much rotation in one direction without one advance decision
GRID_M = 0.05
TACTICS = ("stop", "advance", "reverse", "turn_left", "turn_right")


def wrap(angle): return math.atan2(math.sin(angle), math.cos(angle))


def rle(values): return [[value, len(list(group))] for value, group in itertools.groupby(values)]


def unrle(pairs): return [value for value, count in pairs for _ in range(count)]


# --------------------------------------------------------------------------------------------------------------
# The log
def newest_log(folder=LIVE, stem="reference-normal-spawn"):
    """Newest session log by the start time in its name (YYYYMMDD-HHMMSS, optional -N), then by modification time."""
    found = []
    for path in Path(folder).glob(f"{stem}-*.jsonl"):
        match = re.fullmatch(rf"{re.escape(stem)}-(\d{{8}}-\d{{6}})(?:-(\d+))?", path.stem)
        if match: found.append((match.group(1), int(match.group(2) or 1), path.stat().st_mtime, path))
    if not found: raise SystemExit(f"No timestamped {stem}-*.jsonl log in {folder}")
    return sorted(found)[-1][3], [str(item[3].relative_to(ROOT)).replace("\\", "/") for item in sorted(found)]


def load_rounds(path):
    """Split a play log at its reset lines. Each round is the list of step records."""
    rounds = [[]]
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        if record["type"] == "reset": rounds.append([])
        elif record["type"] == "step": rounds[-1].append(record)
    return [steps for steps in rounds if steps]


# --------------------------------------------------------------------------------------------------------------
# Blues
def make_session(model=MODEL, config=CONFIG):
    """The same checkpoint/environment construction `cli play` uses (ReplaySession), without the window."""
    from autobot.sim import cli
    return cli.ReplaySession(Namespace(config=config, seed=SEED, model=model, profile="sim_debug", action_profile="experimental5_v1",
                                       eval_scenarios=SCENARIOS, scenario_index=SCENARIO_INDEX, opponent="static", opponent_chassis="last_year",
                                       reward_profile="outcome_only", tactic_hold_ms=0.0))


class Clone:
    name = "clone"

    def __init__(self, model):
        from autobot.sim.ladder import GreedyActor
        self.model = model; self.actor = GreedyActor(model)

    def bind(self, env): pass

    def act(self, observation): return int(self.model.predict(observation, deterministic=True)[0])

    def probabilities(self, observation):
        logits = self.actor.logits(observation); logits = logits - logits.max(); p = np.exp(logits); return p / p.sum()

    def batch_actions(self, observations):
        value = np.asarray(observations, dtype=np.float64)
        for weight, bias in self.actor.hidden: value = np.tanh(value @ weight.T + bias)
        return np.argmax(value @ self.actor.output[0].T + self.actor.output[1], axis=1)


class FastClone(Clone):
    """Same weights and the same argmax without the tensor round trip (checked against Clone in stage_replay)."""
    name = "clone"

    def act(self, observation): return int(np.argmax(self.actor.logits(observation)))


class Teacher:
    """The scripted demonstration teacher (privileged FlankController). It reads simulator truth, not the observation."""
    name = "teacher"

    def __init__(self):
        from ordinary_start_diagnostic import FlankController
        self.controller = FlankController()

    def bind(self, env): self.controller.bind(env)

    def act(self, observation=None): return int(self.controller.predict(observation)[0])


class Stop:
    """Control for attribution: does red end up in a pit with blue doing nothing?"""
    name = "always_stop"

    def bind(self, env): pass

    def act(self, observation=None): return 0


def teacher_target(blue_xy, red_xy, arena):
    """What the teacher is steering at (a copy of FlankController's geometry, checked against it in stage_replay)."""
    from ordinary_start_diagnostic import FlankController as F
    b = np.asarray(blue_xy, dtype=float); r = np.asarray(red_xy, dtype=float); pits = np.asarray(arena["pit_centres_m"], dtype=float)
    pit = pits[int(np.argmin(np.linalg.norm(pits - r, axis=1)))]; push = (pit - r) / max(np.linalg.norm(pit - r), 1e-9); margin = 0.09
    behind = np.clip(r - push * F.BEHIND_M, margin, [arena["width_m"] - margin, arena["height_m"] - margin])
    to_red = r - b; lined_up = float(np.dot(to_red / max(np.linalg.norm(to_red), 1e-9), push))
    if np.linalg.norm(behind - b) < F.ARRIVE_M or (lined_up > math.cos(math.radians(15)) and np.linalg.norm(to_red) < 0.30): return "push", b + push
    path = behind - b; length = max(np.linalg.norm(path), 1e-9); along = float(np.dot(r - b, path / length))
    closest = b + path / length * min(max(along, 0.0), length)
    if 0.0 < along < length and np.linalg.norm(r - closest) < F.CLEAR_M:
        normal = np.array([-push[1], push[0]]); side = normal if np.dot(b - r, normal) >= 0 else -normal
        return "go_round_waypoint", np.clip(r + side * (F.CLEAR_M + 0.08), margin, [arena["width_m"] - margin, arena["height_m"] - margin])
    return "behind_red", behind


def teacher_stall(trace, arena, last=250):
    """Where the teacher was steering over its last `last` decisions, and how far from that point it sat."""
    from ordinary_start_diagnostic import FlankController as F
    kinds = []; gaps = []; agree = 0; n = len(trace["actions"])
    for index in range(n):
        p = trace["poses"][index]; kind, target = teacher_target(p[:2], p[3:5], arena)
        error = wrap(math.atan2(target[1] - p[1], target[0] - p[0]) - p[2])
        agree += (1 if abs(error) <= F.ALIGN_RAD else 3 if error > 0 else 4) == trace["actions"][index]
        if index >= n - last: kinds.append(kind); gaps.append(float(np.linalg.norm(target - p[:2])) if kind != "push" else None)
    numeric = [g for g in gaps if g is not None]
    return {"copy_of_teacher_geometry_matches_its_actions": f"{agree}/{n}", "window_decisions": len(kinds), "steering_at": dict(Counter(kinds)),
            "median_distance_to_that_point_m": round(float(np.median(numeric)), 3) if numeric else None,
            "tactics": dict(Counter(TACTICS[a] for a in trace["actions"][-last:])),
            "tactic_switches": int(np.sum(trace["actions"][-last:][1:] != trace["actions"][-last:][:-1])),
            "distance_to_red_m": [round(float(v), 3) for v in (np.linalg.norm(trace["poses"][-last:, :2] - trace["poses"][-last:, 3:5], axis=1).min(),
                                                             np.linalg.norm(trace["poses"][-last:, :2] - trace["poses"][-last:, 3:5], axis=1).max())]}


class Advance:
    name = "always_advance"

    def bind(self, env): pass

    def act(self, observation=None): return 1


# --------------------------------------------------------------------------------------------------------------
# Rollouts
def snapshot(env):
    """A full copy of the simulator state. (The simulator reports poses from the kinematics of the previous
    physics substep, so restoring positions and velocities alone and recomputing would shift them slightly.)"""
    import mujoco
    sim = env.sim; data = mujoco.MjData(sim.model); mujoco.mj_copyData(data, sim.model, sim.data)
    return {"data": data, "steps": env.steps, "last_action": env.last_action, "rng": sim.rng.bit_generator.state, "imu": (sim._last_imu_time, sim._last_imu)}


def restore(env, saved):
    import mujoco
    sim = env.sim; mujoco.mj_copyData(sim.data, sim.model, saved["data"])
    env.steps = saved["steps"]; env.last_action = saved["last_action"]; env.episode_done = False
    sim.rng.bit_generator.state = saved["rng"]; sim._last_imu_time, sim._last_imu = saved["imu"]


def pose(sim):
    blue, red = sim.truth(0), sim.truth(1)
    return [float(blue["position_m"][0]), float(blue["position_m"][1]), float(blue["yaw_rad"]),
            float(red["position_m"][0]), float(red["position_m"][1]), float(red["yaw_rad"])]


def rollout(env, controller, red, *, start=None, shadow=None, probabilities=None, keep=(), max_steps=None):
    """One round. `red` is a list of wheel commands (open loop) or a callable(env, step) -> command.

    Every array is indexed by decision: row i is the state blue saw when it made decision i. `poses` has one
    more row (the state after the last step)."""
    if start is None: env.reset(seed=SEED); env.sim.set_drive(0, 0, 0); env.sim.set_drive(1, 0, 0)
    else: restore(env, start)
    controller.bind(env)
    if shadow is not None: shadow.bind(env)
    first = env.steps; limit = (len(red) if not callable(red) else env.max_steps - first) if max_steps is None else max_steps
    trace = {"first_step": first, "actions": [], "poses": [pose(env.sim)], "observations": [], "contact": [], "red_commands": [],
             "shadow": [], "probabilities": [], "snapshots": {}, "end_reason": None, "end_bot": None}
    for index in range(limit):
        step = first + index
        if step in keep: trace["snapshots"][step] = snapshot(env)
        observation = env._obs()                      # same sim time as the policy's own read, so the same IMU sample
        if shadow is not None: trace["shadow"].append(shadow.act(observation))
        if probabilities is not None: trace["probabilities"].append(probabilities.probabilities(observation))
        action = controller.act(observation)
        command = tuple(red(env, step)) if callable(red) else tuple(red[index])
        _, _, terminated, truncated, info = env.step(action, command)
        trace["actions"].append(action); trace["observations"].append(observation); trace["contact"].append(bool(info["opponent_contact"]))
        trace.setdefault("wall", []).append(bool(info["wall_contact"]))
        trace["red_commands"].append(command); trace["poses"].append(pose(env.sim))
        if terminated or truncated:
            trace["end_reason"] = info["round_end_reason"]; trace["end_bot"] = info["round_end_bot"]; break
    for key in ("actions", "poses", "observations", "contact", "shadow", "probabilities"): trace[key] = np.asarray(trace[key])
    return trace


def outcome(trace):
    reason, bot = trace["end_reason"], trace["end_bot"]
    if reason == "Pit fall": return "blue falls into a pit" if bot == 0 else "red falls into a pit"
    if reason == "Timeout": return "timeout"
    return "recording ends" if reason is None else str(reason)


def bearings(trace):
    """Signed angle (deg) from blue's nose to red at every decision. 0 means the nose points at red; + is to blue's left."""
    p = trace["poses"][:len(trace["actions"])]
    return np.degrees([wrap(math.atan2(row[4] - row[1], row[3] - row[0]) - row[2]) for row in p])


def window_metrics(trace, a=0, b=None):
    """What blue did over decisions [a, b) of this trace (indices relative to the trace)."""
    count = len(trace["actions"]); b = count if b is None else min(b, count); a = min(a, b)
    if b <= a: return {"decisions": 0}
    actions = trace["actions"][a:b]; poses = trace["poses"][a:b + 1]; bearing = bearings(trace)[a:b]
    turn = np.degrees([wrap(y1 - y0) for y0, y1 in zip(poses[:-1, 2], poses[1:, 2])])
    path = float(np.linalg.norm(np.diff(poses[:, :2], axis=0), axis=1).sum()); centre = poses[:, :2].mean(axis=0)
    distance = np.linalg.norm(poses[:, :2] - poses[:, 3:5], axis=1); runs = rle([TACTICS[action] for action in actions])
    nose = np.abs(bearing) <= NOSE_ON_DEG; passes = int(np.sum(nose[1:] & ~nose[:-1]) + int(nose[0]))
    advance_while_on = int(np.sum(nose & (actions == 1)))
    return {"decisions": int(b - a), "seconds": round((b - a) * STEP_S, 2), "first_time_s": round((trace["first_step"] + a) * STEP_S, 2),
            "tactic_counts": dict(Counter(TACTICS[action] for action in actions)),
            "tactic_switches": int(np.sum(actions[1:] != actions[:-1])), "longest_run": max(runs, key=lambda item: item[1]),
            "net_rotation_deg": round(float(turn.sum()), 1), "total_rotation_deg": round(float(np.abs(turn).sum()), 1),
            "revolutions": round(abs(float(turn.sum())) / 360, 2), "turn_rate_deg_s": round(float(turn.sum()) / ((b - a) * STEP_S), 1),
            "path_length_m": round(path, 3), "net_displacement_m": round(float(np.linalg.norm(poses[-1, :2] - poses[0, :2])), 3),
            "largest_distance_from_mean_position_m": round(float(np.linalg.norm(poses[:, :2] - centre, axis=1).max()), 3),
            "distance_to_red_start_m": round(float(distance[0]), 3), "distance_to_red_end_m": round(float(distance[-1]), 3),
            "distance_to_red_min_m": round(float(distance.min()), 3), "contact_decisions": int(trace["contact"][a:b].sum()),
            "nose_on_red_decisions": int(nose.sum()), "nose_passes_over_red": passes, "advance_decisions_with_nose_on_red": advance_while_on,
            "smallest_bearing_to_red_deg": round(float(np.abs(bearing).min()), 1)}


def longest_one_way_rotation(trace):
    """Largest rotation in one direction made without a single advance decision in between: (degrees, first, last decision)."""
    poses = trace["poses"]; actions = trace["actions"]; best = (0.0, 0, 0); total = 0.0; start = 0
    for index, action in enumerate(actions):
        delta = math.degrees(wrap(poses[index + 1, 2] - poses[index, 2]))
        if action == 1 or (total and delta and math.copysign(1, delta) != math.copysign(1, total) and abs(delta) > 0.5):
            total = 0.0; start = index + (action == 1)
            if action == 1: continue
        total += delta
        if abs(total) > abs(best[0]): best = (total, start, index + 1)
    return {"rotation_deg": round(best[0], 1), "first_decision": int(best[1] + trace["first_step"]), "end_decision": int(best[2] + trace["first_step"])}


def spin_segment(trace):
    """The longest run of one turn tactic, if it turns blue by at least SPIN_DEG. None if there is no spin."""
    runs = rle([int(action) for action in trace["actions"]]); position = 0; best = None
    for action, count in runs:
        if action in (3, 4) and (best is None or count > best[1]): best = (position, count, action)
        position += count
    if best is None: return None
    a, count, action = best; rotation = sum(math.degrees(wrap(trace["poses"][i + 1, 2] - trace["poses"][i, 2])) for i in range(a, a + count))
    if abs(rotation) < SPIN_DEG: return None
    return {"first_decision": int(a + trace["first_step"]), "end_decision": int(a + count + trace["first_step"]), "tactic": TACTICS[action]}


def passes_case(trace, after=0):
    """Regression rule for a blue replayed against the recorded red input: it must not spin and must get to red."""
    one_way = longest_one_way_rotation(trace); a = max(0, after - trace["first_step"]); later = window_metrics(trace, a)
    reached = later.get("contact_decisions", 0) > 0 or outcome(trace) == "red falls into a pit"
    return {"no_spin": abs(one_way["rotation_deg"]) < SPIN_DEG, "reaches_red_after_spin_onset": bool(reached),
            "no_own_fall": outcome(trace) != "blue falls into a pit",
            "passes": bool(abs(one_way["rotation_deg"]) < SPIN_DEG and reached and outcome(trace) != "blue falls into a pit"),
            "longest_one_way_rotation": one_way}


def brief(trace):
    full = window_metrics(trace)
    return {"outcome": outcome(trace), "decisions": int(len(trace["actions"])), "end_time_s": round((trace["first_step"] + len(trace["actions"])) * STEP_S, 2),
            **{key: full[key] for key in ("tactic_counts", "tactic_switches", "longest_run", "revolutions", "path_length_m", "contact_decisions",
                                          "nose_on_red_decisions", "distance_to_red_min_m", "distance_to_red_end_m")},
            "longest_one_way_rotation": longest_one_way_rotation(trace)}


# --------------------------------------------------------------------------------------------------------------
# Valid red positions
def valid_red_positions(env, blue_xy, spacing=GRID_M):
    """Grid of red centres that are on the floor, clear of walls, pits, the flipper and blue (geometric pre-filter)."""
    arena = env.sim.cfg["arena"]; w, h = arena["width_m"], arena["height_m"]; margin = 0.10; points = []
    flipper = np.asarray(arena["flipper_vertices_m"], dtype=float)
    def in_flipper(p):
        a, b, c = flipper; s = [(b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]), (c[0] - b[0]) * (p[1] - b[1]) - (c[1] - b[1]) * (p[0] - b[0]),
                                (a[0] - c[0]) * (p[1] - c[1]) - (a[1] - c[1]) * (p[0] - c[0])]
        return all(v <= 0 for v in s) or all(v >= 0 for v in s)
    for x in np.arange(margin, w - margin + 1e-9, spacing):
        for y in np.arange(margin, h - margin + 1e-9, spacing):
            p = np.array([x, y])
            if any(np.linalg.norm(p - np.asarray(c)) < arena["pit_radius_m"] + margin for c in arena["pit_centres_m"]): continue
            if in_flipper(p + np.array([margin, margin]) * 0.75) or in_flipper(p): continue
            if np.linalg.norm(p - np.asarray(blue_xy)) < 0.20: continue
            points.append((round(float(x), 4), round(float(y), 4)))
    return points


def place_red(env, saved, xy, red_yaw=None):
    """Blue exactly as saved; red moved to `xy` at rest. Returns False if the simulator reports the state is not valid."""
    restore(env, saved); sim = env.sim
    sim.set_pose(1, float(xy[0]), float(xy[1]), sim.truth(1)["yaw_rad"] if red_yaw is None else red_yaw)
    sim.rng = np.random.default_rng(SEED); sim._last_imu_time, sim._last_imu = -math.inf, None      # identical IMU noise at every position
    return not env._robots_contact() and not env._wall_contact(1) and sim.arena_event() is None


# --------------------------------------------------------------------------------------------------------------
def stage_replay(out):
    log, logs = newest_log(); rounds = load_rounds(log); session = make_session(); env = session.env
    clone, teacher, advance, stop = Clone(session.model), Teacher(), Advance(), Stop()
    report = {"log": str(log.relative_to(ROOT)).replace("\\", "/"), "log_sha256": hashlib.sha256(log.read_bytes()).hexdigest(),
              "timestamped_logs_found": logs, "checkpoint": MODEL, "config": CONFIG, "seed": SEED, "scenario": "default_spawn (index 2)",
              "rounds": []}
    saved = {}
    for number, steps in enumerate(rounds, 1):
        red = [tuple(step["red_command"]) for step in steps]; recorded = [step["blue_tactic"] for step in steps]
        trace = rollout(env, clone, red, shadow=teacher, probabilities=clone, keep=set(range(len(red))))
        tactics = [TACTICS[action] for action in trace["actions"]]
        fast = FastClone(session.model); assert all(fast.act(o) == a for o, a in zip(trace["observations"], trace["actions"])), "numpy actor disagrees with the checkpoint"
        entry = {"round": number, "decisions_recorded": len(steps), "recorded_end": steps[-1]["end_reason"],
                 "clone_tactics_reproduced": int(sum(a == b for a, b in zip(tactics, recorded))),
                 "red_command_runs": len(rle(red)), "red_idle_share": round(sum(c == (0.0, 0.0) for c in red) / len(red), 3)}
        spin = spin_segment(trace); entry["spin"] = spin
        entry["controllers"] = {"clone": {**brief(trace), "case": passes_case(trace, spin["first_decision"] if spin else 0)}}
        traces = {"clone": trace}
        for other in (teacher, advance, stop):
            traces[other.name] = rollout(env, other, red, shadow=clone if other is teacher else None)
            entry["controllers"][other.name] = {**brief(traces[other.name]), "case": passes_case(traces[other.name], spin["first_decision"] if spin else 0)}
        # Where the clone and the teacher part ways.
        shadow = trace["shadow"]; differ = np.flatnonzero(shadow != trace["actions"])
        entry["teacher_labels_on_clone_states"] = {
            "first_disagreement_decision": int(differ[0]) if len(differ) else None,
            "agreement_share": round(float(np.mean(shadow == trace["actions"])), 3),
            "first_sustained_disagreement_decision": first_sustained(shadow != trace["actions"], 10)}
        entry["teacher_last_5_s"] = teacher_stall(traces["teacher"], env.sim.cfg["arena"])
        closed = traces["teacher"]; n = min(len(closed["actions"]), len(trace["actions"]))
        apart = np.flatnonzero(closed["actions"][:n] != trace["actions"][:n])
        entry["teacher_closed_loop_first_different_action"] = int(apart[0]) if len(apart) else None
        if spin:
            a, b = spin["first_decision"], spin["end_decision"]
            entry["spin"].update(window_metrics(trace, a, b))
            entry["spin"]["teacher_labels"] = dict(Counter(TACTICS[label] for label in shadow[a:b]))
            entry["spin"]["teacher_says_advance_decisions"] = int(np.sum(shadow[a:b] == 1))
            entry["spin"]["red_idle_share"] = round(sum(c == (0.0, 0.0) for c in red[a:b]) / (b - a), 3)
            entry["spin"]["red_position_range_m"] = [[round(float(v), 3) for v in trace["poses"][a:b, 3:5].min(axis=0)],
                                                     [round(float(v), 3) for v in trace["poses"][a:b, 3:5].max(axis=0)]]
            entry["spin"]["clone_turn_right_probability"] = [round(float(trace["probabilities"][a:b, 4].min()), 4), round(float(trace["probabilities"][a:b, 4].mean()), 4)]
            entry["before_spin"] = window_metrics(trace, 0, a)
            # Phases before the spin: the opening advance, then whatever came between it and the spin.
            opening = next((i for i, action in enumerate(trace["actions"]) if i > 5 and action != 1), a)
            entry["phases"] = {"opening": window_metrics(trace, 0, opening), "between_opening_and_spin": window_metrics(trace, opening, a)}
            entry["phases"]["between_opening_and_spin"]["teacher_labels"] = dict(Counter(TACTICS[label] for label in shadow[opening:a]))
            entry["phases"]["between_opening_and_spin"]["agreement_share"] = round(float(np.mean(shadow[opening:a] == trace["actions"][opening:a])), 3)
            entry["phases"]["between_opening_and_spin"]["yaw_range_deg"] = [round(math.degrees(float(v)), 1) for v in
                                                                            (trace["poses"][opening:a, 2].min(), trace["poses"][opening:a, 2].max())]
            # The teacher put in the clone's place just before the spin, against the rest of the recording.
            resumed = rollout(env, teacher, red[a:], start=trace["snapshots"][a])
            again = rollout(env, clone, red[a:], start=trace["snapshots"][a])
            entry["restore_check_clone_identical"] = bool(np.array_equal(again["actions"], trace["actions"][a:]))
            entry["teacher_from_pre_spin_state"] = {**brief(resumed), "case": passes_case(resumed, a),
                                                    "first_decisions": rle([TACTICS[x] for x in resumed["actions"][:200]])[:12]}
            entry["teacher_from_pre_spin_state"]["last_5_s"] = teacher_stall(resumed, env.sim.cfg["arena"])
            entry["teacher_from_pre_spin_state"]["distance_to_red_every_second_m"] = [
                round(float(np.linalg.norm(resumed["poses"][i, :2] - resumed["poses"][i, 3:5])), 3) for i in range(0, len(resumed["actions"]), 50)]
            traces["teacher_from_pre_spin_state"] = resumed
            saved[number] = {"snapshot": trace["snapshots"][a], "decision": a, "observation": trace["observations"][a]}
            # Red sensitivity along the round: does any valid red position change the clone's choice?
            entry["red_sensitivity"] = red_sensitivity(env, clone, trace)
            entry["first_failure_to_respond"] = first_failure(trace, entry)
        entry["timeline"] = timeline(trace); entry["red_timeline"] = red_timeline(trace)
        entry["decision_log"] = decision_log(out, env, trace, steps, spin, number)
        np.savez_compressed(out / f"round{number}-traces.npz", **{f"{name}_{key}": value[key] for name, value in traces.items()
                                                                  for key in ("actions", "poses", "contact", "shadow", "probabilities", "observations") if len(value[key])})
        report["rounds"].append(entry)
    (out / "replay.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    return report, session, saved


def free_turn_rate(env, tactic=4, seconds=2.0):
    """Yaw rate (deg/s) blue reaches turning on the spot with nothing near it: what a turn tactic does when it is executed."""
    env.reset(seed=SEED); env.sim.set_pose(0, 0.575, 0.575, 0.0); env.sim.set_pose(1, 0.2, 0.2, 0.0); yaws = [env.sim.truth(0)["yaw_rad"]]
    for _ in range(int(seconds / STEP_S)):
        env.step(tactic, (0.0, 0.0)); yaws.append(env.sim.truth(0)["yaw_rad"])
    rate = np.degrees([wrap(b - a) for a, b in zip(yaws[:-1], yaws[1:])]) / STEP_S
    return {"steady_deg_s": round(float(rate[len(rate) // 2:].mean()), 1), "first_0_2_s_deg_s": round(float(rate[:10].mean()), 1)}


def decision_log(out, env, trace, recorded, spin, number):
    """One CSV row per decision of the round: what the clone saw, chose and commanded, and what the simulator then did."""
    import csv
    from autobot.sim.env import tactic_to_drive
    poses = trace["poses"]; actions = trace["actions"]; n = len(actions); bearing = bearings(trace)
    turn = np.degrees([wrap(poses[i + 1, 2] - poses[i, 2]) for i in range(n)]); moved = np.linalg.norm(np.diff(poses[:, :2], axis=0), axis=1)
    a = spin["first_decision"] if spin else 0; path = out / f"round{number}-decision-log.csv"; mismatched = 0
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["decision", "time_s", "phase", "red_cmd_left", "red_cmd_right", "p_stop", "p_advance", "p_reverse", "p_turn_left", "p_turn_right",
                         "selected_tactic", "applied_tactic", "recorded_tactic", "blue_cmd_left", "blue_cmd_right", "recorded_blue_cmd", "teacher_label",
                         "blue_x_m", "blue_y_m", "blue_yaw_deg", "yaw_rate_deg_s", "heading_change_since_round_start_deg", "heading_change_since_spin_start_deg",
                         "step_displacement_mm", "displacement_from_spin_start_m", "red_x_m", "red_y_m", "bearing_to_red_deg", "distance_to_red_m",
                         "touching_red", "touching_wall"])
        for i in range(n):
            command = tactic_to_drive(int(actions[i]), "experimental5_v1"); logged = recorded[i]
            mismatched += tuple(logged["blue_command"]) != tuple(command) or logged["blue_tactic"] != TACTICS[actions[i]]
            writer.writerow([i, round(i * STEP_S, 2), "spin" if spin and i >= a else "before_spin", *trace["red_commands"][i],
                             *[round(float(v), 4) for v in trace["probabilities"][i]], TACTICS[actions[i]], TACTICS[actions[i]], logged["blue_tactic"],
                             *command, json.dumps(logged["blue_command"]), TACTICS[trace["shadow"][i]],
                             round(float(poses[i, 0]), 4), round(float(poses[i, 1]), 4), round(math.degrees(poses[i, 2]), 1), round(float(turn[i]) / STEP_S, 1),
                             round(float(turn[:i].sum()), 1), round(float(turn[a:i].sum()), 1) if i >= a else "", round(float(moved[i]) * 1000, 2),
                             round(float(np.linalg.norm(poses[i, :2] - poses[a, :2])), 4) if i >= a else "", round(float(poses[i, 3]), 4), round(float(poses[i, 4]), 4),
                             round(float(bearing[i]), 1), round(float(np.linalg.norm(poses[i, :2] - poses[i, 3:5])), 4), int(trace["contact"][i]), int(trace["wall"][i])])
    free = free_turn_rate(env); rates = turn / STEP_S; turning = np.isin(actions, (3, 4))
    result = {"file": str(path.relative_to(ROOT)).replace("\\", "/"), "rows": n, "rows_where_recorded_tactic_or_wheel_command_differs_from_replay": int(mismatched),
              "tactic_hold": "0 ms, so the applied tactic is the selected tactic at every decision",
              "free_turn_rate_of_a_turn_tactic": free, "touching_wall_decisions": int(np.sum(trace["wall"])),
              "advance_decisions_and_distance_moved_m": [int(np.sum(actions == 1)), round(float(moved[actions == 1].sum()), 3)]}
    if spin:
        result["spin_mean_yaw_rate_deg_s"] = round(float(rates[a:].mean()), 1)
        result["spin_yaw_rate_range_deg_s"] = [round(float(rates[a + 25:].min()), 1), round(float(rates[a + 25:].max()), 1)]
        result["spin_executed_share_of_free_turn_rate"] = round(float(rates[a + 25:].mean()) / free["steady_deg_s"], 3)
        result["before_spin_turn_decisions_mean_abs_yaw_rate_deg_s"] = round(float(np.abs(rates[:a][turning[:a]]).mean()), 1)
        result["spin_touching_red_or_wall_decisions"] = int(np.sum(trace["contact"][a:]) + np.sum(trace["wall"][a:]))
    return result


def red_timeline(trace):
    """Red's recorded input as runs of one command, with where red was and how far from blue."""
    rows = []; position = 0
    for command, count in rle(trace["red_commands"]):
        p, q = trace["poses"][position], trace["poses"][position + count]
        forward, turning = -(command[0] + command[1]) / 2, (command[0] - command[1]) / 2
        kind = "idle" if command == (0.0, 0.0) else "forward" if forward > 0.5 and abs(turning) < 0.05 else "reverse" if forward < -0.3 and abs(turning) < 0.05 \
            else "turn on the spot" if abs(forward) < 0.05 else "forward curve" if forward > 0 else "reverse curve"
        rows.append({"from_s": round(position * STEP_S, 2), "to_s": round((position + count) * STEP_S, 2), "red": kind, "command": list(command),
                     "red_from": [round(float(v), 3) for v in p[3:5]], "red_to": [round(float(v), 3) for v in q[3:5]],
                     "distance_to_blue_m": [round(float(np.linalg.norm(p[:2] - p[3:5])), 3), round(float(np.linalg.norm(q[:2] - q[3:5])), 3)],
                     "blue_tactics_meanwhile": dict(Counter(TACTICS[x] for x in trace["actions"][position:position + count]))})
        position += count
    return rows


def first_sustained(mask, length):
    run = 0
    for index, value in enumerate(mask):
        run = run + 1 if value else 0
        if run >= length: return int(index - length + 1)
    return None


def timeline(trace, every=25):
    b = bearings(trace); rows = []
    for a in range(0, len(trace["actions"]), every):
        chunk = trace["actions"][a:a + every]; p = trace["poses"][a]
        rows.append({"time_s": round((trace["first_step"] + a) * STEP_S, 2), "blue": [round(float(p[0]), 3), round(float(p[1]), 3), round(math.degrees(p[2]), 1)],
                     "red": [round(float(p[3]), 3), round(float(p[4]), 3)], "bearing_to_red_deg": round(float(b[a]), 1),
                     "distance_m": round(float(np.linalg.norm(p[:2] - p[3:5])), 3), "tactics": dict(Counter(TACTICS[x] for x in chunk)),
                     "switches": int(np.sum(chunk[1:] != chunk[:-1])), "red_commands": dict(Counter(str(c) for c in trace["red_commands"][a:a + every]))})
    return rows


def red_sensitivity(env, clone, trace):
    """At every decision, swap only red's two observation channels for every valid red position and re-ask the clone.

    `same_share` is the share of valid red positions for which the clone would have chosen the tactic it chose."""
    arena = env.sim.cfg["arena"]; w, h = arena["width_m"], arena["height_m"]; shares = []; distinct = []
    for index, observation in enumerate(trace["observations"]):
        points = valid_red_positions(env, trace["poses"][index, :2], spacing=0.10)
        batch = np.repeat(observation[None, :].astype(np.float64), len(points), axis=0)
        batch[:, -2] = [2 * p[0] / w - 1 for p in points]; batch[:, -1] = [2 * p[1] / h - 1 for p in points]
        chosen = clone.batch_actions(batch); shares.append(float(np.mean(chosen == trace["actions"][index]))); distinct.append(len(set(chosen.tolist())))
    shares = np.asarray(shares); blind = shares >= 0.999
    return {"positions_per_decision": "0.10 m grid of valid red positions (about 70)",
            "first_decision_where_no_red_position_changes_the_choice": int(np.flatnonzero(blind)[0]) if blind.any() else None,
            "first_sustained_half_second_of_that": first_sustained(blind, 25),
            "share_of_decisions_where_no_red_position_changes_the_choice": round(float(blind.mean()), 3),
            "mean_same_share_by_second": [round(float(shares[a:a + 50].mean()), 3) for a in range(0, len(shares), 50)],
            "blind_share_by_second": [round(float(blind[a:a + 50].mean()), 3) for a in range(0, len(blind), 50)]}


def first_failure(trace, entry):
    """Behavioural markers of blue not following red, in order of time."""
    b = bearings(trace); actions = trace["actions"]; n = len(actions)
    # Turning away: red is clearly to one side (more than 30 deg) and blue commands a turn to the other side.
    away = np.array([(abs(b[i]) > 30) and ((actions[i] == 3 and b[i] < 0) or (actions[i] == 4 and b[i] > 0)) for i in range(n)])
    # Red behind blue (more than 90 deg off the nose) for a sustained second.
    behind = np.abs(b) > 90
    return {"first_turn_away_from_red_decision": int(np.flatnonzero(away)[0]) if away.any() else None,
            "first_sustained_turn_away_decision (0.1 s)": first_sustained(away, 5),
            "first_decision_red_is_behind_blue_for_1_s": first_sustained(behind, 50),
            "first_red_movement_decision": next((i for i, c in enumerate(trace["red_commands"]) if c != (0.0, 0.0)), None)}


# --------------------------------------------------------------------------------------------------------------
CURVE_S = 1.0


def red_script(kind):
    """Matched red behaviours from the normal spawn. Fixed key presses, never a reaction to blue.

    still: no command. left / right: the forward curve a person drives with W+D / W+A, held for CURVE_S, then
    released. `left` is the left of the viewer window (smaller x). Red spawns facing the bottom of the window,
    so window-left is red's own right, hence W+D."""
    from autobot.sim.env import arcade_drive
    if kind == "still": return lambda env, step: (0.0, 0.0)
    held = arcade_drive(forward=True, right=True) if kind == "left" else arcade_drive(forward=True, left=True)
    return lambda env, step: held if step < round(CURVE_S / STEP_S) else (0.0, 0.0)


def stage_cases(out, session):
    env = session.env; clone, teacher = Clone(session.model), Teacher(); report = {"cases": {}}; traces = {}
    for kind in ("still", "left", "right"):
        trace = rollout(env, clone, red_script(kind), shadow=teacher, probabilities=clone); traces[kind] = trace
        taught = rollout(env, teacher, red_script(kind)); traces[f"teacher_{kind}"] = taught
        pushed = rollout(env, Advance(), red_script(kind))
        contact = np.flatnonzero(trace["contact"]); first_contact = int(contact[0]) if len(contact) else None
        before = slice(0, first_contact if first_contact is not None else len(trace["actions"]))
        red_moving = [i for i, c in enumerate(trace["red_commands"]) if c != (0.0, 0.0)]
        entry = {"red_commands": [[list(c), n] for c, n in rle(trace["red_commands"])][:6],
                 "red_position_at_0_5_1_2_s": [[round(float(v), 3) for v in trace["poses"][min(i, len(trace["poses"]) - 1), 3:5]] for i in (25, 50, 100)],
                 "red_position_in_blue_observation_at_0_5_1_2_s": [[round(float(v), 4) for v in trace["observations"][i, -2:]] for i in (25, 50, 100)],
                 "red_stops_moving_at_s": round((red_moving[-1] + 1) * STEP_S, 2) if red_moving else 0.0,
                 "first_contact_decision": first_contact, "first_contact_time_s": None if first_contact is None else round(first_contact * STEP_S, 2),
                 "tactics_before_contact": rle([TACTICS[a] for a in trace["actions"][before]])[:20],
                 "tactic_counts_before_contact": dict(Counter(TACTICS[a] for a in trace["actions"][before])),
                 "blue_path_every_quarter_second": [[round(float(p[0]), 3), round(float(p[1]), 3), round(math.degrees(float(p[2])), 1)]
                                                    for p in trace["poses"][0:126:12]],
                 "probabilities_stop_advance_reverse_left_right": {f"{i * STEP_S:.2f}s": [round(float(v), 3) for v in trace["probabilities"][i]]
                                                                   for i in (0, 10, 20, 30, 40, 50, 60, 75, 100) if i < len(trace["probabilities"])},
                 "bearing_to_red_deg": {f"{i * STEP_S:.2f}s": round(float(bearings(trace)[i]), 1) for i in (0, 25, 50, 75, 100, 150) if i < len(trace["actions"])},
                 "clone": brief(trace), "teacher": brief(taught), "always_advance": brief(pushed),
                 "teacher_last_5_s": teacher_stall(taught, env.sim.cfg["arena"]),
                 "clone_first_3_s": window_metrics(trace, 0, 150), "teacher_first_3_s": window_metrics(taught, 0, 150),
                 "red_displacement_m": [round(float(v), 3) for v in trace["poses"][-1, 3:5] - trace["poses"][0, 3:5]],
                 "teacher_labels_agreement_share": round(float(np.mean(trace["shadow"] == trace["actions"])), 3),
                 "first_teacher_disagreement_decision": int(np.flatnonzero(trace["shadow"] != trace["actions"])[0]) if np.any(trace["shadow"] != trace["actions"]) else None,
                 "first_sustained_teacher_disagreement_decision": first_sustained(trace["shadow"] != trace["actions"], 10)}
        report["cases"][kind] = entry
    still = traces["still"]
    for kind in ("left", "right"):
        trace = traces[kind]; n = min(len(trace["actions"]), len(still["actions"]))
        differ = np.flatnonzero(trace["actions"][:n] != still["actions"][:n])
        gap = np.abs(trace["probabilities"][:n] - still["probabilities"][:n]).max(axis=1)
        red_gap = np.abs(trace["observations"][:n, -2:] - still["observations"][:n, -2:]).max(axis=1)
        report["cases"][kind]["versus_still"] = {
            "first_decision_red_observation_differs": int(np.flatnonzero(red_gap > 1e-6)[0]) if np.any(red_gap > 1e-6) else None,
            "first_decision_probabilities_differ_by_0.01": int(np.flatnonzero(gap > 0.01)[0]) if np.any(gap > 0.01) else None,
            "first_decision_tactic_differs": int(differ[0]) if len(differ) else None,
            "blue_position_gap_at_1_s_m": round(float(np.linalg.norm(trace["poses"][50, :2] - still["poses"][50, :2])), 3) if n > 50 else None,
            "blue_position_gap_at_2_s_m": round(float(np.linalg.norm(trace["poses"][100, :2] - still["poses"][100, :2])), 3) if n > 100 else None}
    n = min(len(traces["left"]["actions"]), len(traces["right"]["actions"]))
    differ = np.flatnonzero(traces["left"]["actions"][:n] != traces["right"]["actions"][:n])
    report["left_versus_right_first_decision_tactic_differs"] = int(differ[0]) if len(differ) else None
    assert traces["left"]["poses"][60, 3] < traces["still"]["poses"][60, 3] < traces["right"]["poses"][60, 3], "left/right are named by arena x"
    # The still case is the exact counterfactual for the recorded spin round until red first moves in it.
    recorded = np.load(out / "round2-traces.npz"); n = min(len(recorded["clone_actions"]), len(still["actions"]))
    gap = np.abs(recorded["clone_probabilities"][:n] - still["probabilities"][:n]).max(axis=1)
    differ = np.flatnonzero(recorded["clone_actions"][:n] != still["actions"][:n])
    report["recorded_round_2_versus_still"] = {
        "first_decision_probabilities_differ_at_all": int(np.flatnonzero(gap > 1e-9)[0]) if np.any(gap > 1e-9) else None,
        "first_decision_probabilities_differ_by_0.01": int(np.flatnonzero(gap > 0.01)[0]) if np.any(gap > 0.01) else None,
        "first_decision_tactic_differs": int(differ[0]) if len(differ) else None,
        "tactics_there": [TACTICS[int(recorded["clone_actions"][differ[0]])], TACTICS[int(still["actions"][differ[0]])]] if len(differ) else None}
    np.savez_compressed(out / "matched-cases-traces.npz", **{f"{name}_{key}": value[key] for name, value in traces.items()
                                                             for key in ("actions", "poses", "contact", "probabilities") if len(value[key])})
    (out / "matched-cases.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    return report


# --------------------------------------------------------------------------------------------------------------
def stage_probe(out, session, saved, seconds=10.0):
    """Blue held exactly as it was at the decision that starts the spin; red moved over every valid position."""
    env = session.env; clone, teacher = FastClone(session.model), Teacher(); item = saved[max(saved)]; snap = item["snapshot"]
    restore(env, snap); blue = pose(env.sim); points = valid_red_positions(env, blue[:2]); rows = []; reference = None
    arena = env.sim.cfg["arena"]
    for xy in points:
        if not place_red(env, snap, xy): continue
        observation = env._obs(); teacher.bind(env)
        if reference is None: reference = observation.copy()
        changed = np.flatnonzero(np.abs(observation - reference) > 1e-7).tolist()
        probabilities = clone.probabilities(observation); choice = int(np.argmax(probabilities)); label = teacher.act()
        bearing = math.degrees(wrap(math.atan2(xy[1] - blue[1], xy[0] - blue[0]) - blue[2]))
        row = {"red": list(xy), "bearing_deg": round(bearing, 1), "distance_m": round(float(np.hypot(xy[0] - blue[0], xy[1] - blue[1])), 3),
               "clone": TACTICS[choice], "clone_probabilities": [round(float(v), 3) for v in probabilities], "teacher": TACTICS[label],
               "observation_channels_changed": changed,
               "nearest_pit": "top_left" if np.hypot(xy[0], xy[1] - arena["height_m"]) < np.hypot(xy[0] - arena["width_m"], xy[1]) else "bottom_right"}
        # What each controller then does from here with red left at rest.
        for controller in (clone, teacher):
            place_red(env, snap, xy); trace = rollout(env, controller, lambda e, s: (0.0, 0.0), start=None if False else snapshot(env), max_steps=int(seconds / STEP_S))
            verdict = passes_case(trace); one_way = abs(verdict["longest_one_way_rotation"]["rotation_deg"])
            row[f"{controller.name}_closed_loop"] = {"outcome": outcome(trace), "spins": one_way >= SPIN_DEG, "one_way_rotation_deg": round(one_way, 1),
                                                     "touches_red": bool(trace["contact"].any()), "nose_on_red_decisions": window_metrics(trace)["nose_on_red_decisions"],
                                                     "advance_decisions": int(np.sum(trace["actions"] == 1)),
                                                     "final_distance_to_red_m": round(float(np.linalg.norm(trace["poses"][-1, :2] - trace["poses"][-1, 3:5])), 3)}
            if controller is teacher and trace["end_reason"] is None:
                stall = teacher_stall(trace, arena, last=100); steering = stall["steering_at"]
                row["teacher_closed_loop"]["parked_on_its_waypoint"] = bool(steering.get("go_round_waypoint", 0) >= 90 and (stall["median_distance_to_that_point_m"] or 1) < 0.04)
                row["teacher_closed_loop"]["last_2_s"] = stall
        rows.append(row)
    n = len(rows); same = sum(r["clone"] == r["teacher"] for r in rows)
    def tally(key): return dict(Counter(r[key] for r in rows))
    def loop(name, field): return int(sum(bool(r[f"{name}_closed_loop"][field]) for r in rows))
    channels = sorted({c for r in rows for c in r["observation_channels_changed"]})
    summary = {"saved_state": {"round": max(saved), "decision": item["decision"], "time_s": round(item["decision"] * STEP_S, 2),
                               "blue_pose_x_y_yaw_deg": [round(blue[0], 3), round(blue[1], 3), round(math.degrees(blue[2]), 1)],
                               "red_recorded_position": [round(blue[3], 3), round(blue[4], 3)]},
               "valid_red_positions": n, "grid_spacing_m": GRID_M, "observation_channels_that_changed": channels,
               "clone_choice": tally("clone"), "teacher_choice": tally("teacher"), "same_choice": same, "same_choice_share": round(same / n, 3),
               "by_nearest_pit": {pit: {"positions": sum(r["nearest_pit"] == pit for r in rows),
                                        "same_choice": sum(r["nearest_pit"] == pit and r["clone"] == r["teacher"] for r in rows),
                                        "clone_choice": dict(Counter(r["clone"] for r in rows if r["nearest_pit"] == pit)),
                                        "teacher_choice": dict(Counter(r["teacher"] for r in rows if r["nearest_pit"] == pit)),
                                        "clone_spins": sum(r["nearest_pit"] == pit and r["clone_closed_loop"]["spins"] for r in rows),
                                        "teacher_spins": sum(r["nearest_pit"] == pit and r["teacher_closed_loop"]["spins"] for r in rows),
                                        "clone_touches_red": sum(r["nearest_pit"] == pit and r["clone_closed_loop"]["touches_red"] for r in rows),
                                        "teacher_touches_red": sum(r["nearest_pit"] == pit and r["teacher_closed_loop"]["touches_red"] for r in rows)}
                                  for pit in ("top_left", "bottom_right")},
               f"closed_loop_{seconds:g}s_red_at_rest": {name: {"spins": loop(name, "spins"), "touches_red": loop(name, "touches_red"),
                                                                 "outcomes": dict(Counter(r[f"{name}_closed_loop"]["outcome"] for r in rows))}
                                                          for name in ("clone", "teacher")},
               "teacher_parked_on_its_waypoint_without_touching_red": int(sum(bool(r["teacher_closed_loop"].get("parked_on_its_waypoint")) and not r["teacher_closed_loop"]["touches_red"] for r in rows)),
               "teacher_never_touches_red": int(sum(not r["teacher_closed_loop"]["touches_red"] for r in rows)),
               "clone_never_touches_red": int(sum(not r["clone_closed_loop"]["touches_red"] for r in rows))}
    # Heading sweep where the spin happened: blue at rest at its spin position, red where it sat during the spin.
    restore(env, snap); sweep = []
    spin_red = saved[max(saved)].get("spin_red", blue[3:5])
    for degrees in range(-180, 180, 10):
        restore(env, snap); env.sim.set_pose(1, spin_red[0], spin_red[1], blue[5]); env.sim.set_pose(0, blue[0], blue[1], math.radians(degrees))
        env.sim.rng = np.random.default_rng(SEED); env.sim._last_imu_time, env.sim._last_imu = -math.inf, None
        observation = env._obs(); teacher.bind(env); p = clone.probabilities(observation)
        sweep.append({"blue_yaw_deg": degrees, "bearing_to_red_deg": round(math.degrees(wrap(math.atan2(spin_red[1] - blue[1], spin_red[0] - blue[0]) - math.radians(degrees))), 1),
                      "clone": TACTICS[int(np.argmax(p))], "clone_probabilities": [round(float(v), 3) for v in p], "teacher": TACTICS[teacher.act()]})
    summary["heading_sweep_at_rest"] = {"red_position": [round(float(v), 3) for v in spin_red], "clone_choice": dict(Counter(r["clone"] for r in sweep)),
                                        "teacher_choice": dict(Counter(r["teacher"] for r in sweep)),
                                        "same_choice": sum(r["clone"] == r["teacher"] for r in sweep), "headings": len(sweep)}
    (out / "red-position-probe.json").write_text(json.dumps({"summary": summary, "positions": rows, "heading_sweep": sweep}, indent=1), encoding="utf-8")
    return summary, rows, sweep


# --------------------------------------------------------------------------------------------------------------
def stage_play(out):
    """Run the shipped viewer loop headlessly with synthetic keys and capture what blue's policy was actually given."""
    import verify_play_mode as play
    from autobot.sim import cli
    captured = []; full = []; holder = {}; original = cli.load_checkpoint

    class Recorder:
        def __init__(self, model): self.model = model
        def __getattr__(self, name): return getattr(self.model, name)
        def predict(self, observation, *args, **kwargs):
            sim = holder["harness"].session.sim; arena = sim.cfg["arena"]; blue, red = sim.truth(0), sim.truth(1)
            expected = [2 * blue["position_m"][0] / arena["width_m"] - 1, 2 * blue["position_m"][1] / arena["height_m"] - 1, math.sin(blue["yaw_rad"]),
                        math.cos(blue["yaw_rad"]), 2 * red["position_m"][0] / arena["width_m"] - 1, 2 * red["position_m"][1] / arena["height_m"] - 1]
            action, state = self.model.predict(observation, *args, **kwargs)
            imu = sim.imu(); ranges = sim.cfg["imu"]            # the cached sample of this control step, so no extra noise is drawn
            whole = list(imu.gyro_rad_s / ranges["gyro_range_rad_s"]) + list(imu.accel_m_s2 / ranges["accel_range_m_s2"]) + [0.0] * 7 + expected
            full.append((np.asarray(observation, dtype=np.float64).reshape(-1), np.clip(np.asarray(whole, dtype=np.float64), -1, 1)))
            captured.append({"time_s": float(sim.data.time), "observation_pose_channels": [float(v) for v in np.asarray(observation).reshape(-1)[-6:]],
                             "simulator_truth": [float(v) for v in expected], "red_xy": [float(v) for v in red["position_m"][:2]], "action": int(action)})
            return action, state

    def loader(*args, **kwargs):
        model, encoding = original(*args, **kwargs); return Recorder(model), encoding

    # Red: still for 0.4 s, D (turn on the spot) 0.5 s, W 0.6 s, released, then W+A curve, then released.
    schedule = [(2, [], True), (2, ["space"], True), (20, [], True), (25, ["d"], True), (30, ["w"], True), (25, [], True), (40, ["w", "a"], True), (60, [], True)]
    for old in out.glob("play-observation-check-*.jsonl"): old.unlink()          # this script's own earlier synthetic logs
    harness = play.Harness(schedule); holder["harness"] = harness; cli.load_checkpoint = loader
    try: harness.run(play.arguments("play", MODEL, SCENARIOS, SCENARIO_INDEX, SEED, "--input-log", str(out / "play-observation-check.jsonl")))
    finally: cli.load_checkpoint = original
    error = np.abs(np.asarray([c["observation_pose_channels"] for c in captured]) - np.asarray([c["simulator_truth"] for c in captured]))
    red = np.asarray([c["red_xy"] for c in captured]); channel = np.asarray([c["observation_pose_channels"][4:] for c in captured])
    log = play.read_log(harness.log); steps = [r for r in log if r["type"] == "step"]
    report = {"viewer_loop": "cli.preview via scripts/verify_play_mode.Harness (SDL dummy driver, synthetic keys)", "policy_calls": len(captured),
              "logged_steps": len(steps), "largest_gap_between_observation_and_simulator_truth": float(error.max()),
              "red_channels_largest_gap": float(error[:, 4:].max()),
              "red_moved_m": round(float(np.linalg.norm(red.max(axis=0) - red.min(axis=0))), 3),
              "red_observation_channels_distinct_values": int(len({tuple(np.round(c, 6)) for c in channel})),
              "decisions_where_red_channels_changed_from_the_previous_decision": int(np.sum(np.abs(np.diff(channel, axis=0)).max(axis=1) > 1e-7)),
              "red_commands_logged": [[c, n] for c, n in rle([tuple(s["red_command"]) for s in steps])],
              "samples": [captured[i] for i in range(0, len(captured), 25)],
              "observation_layout": "19 channels: gyro xyz, accel xyz, 7 previous-action slots (held at 0), blue x, blue y, sin(yaw), cos(yaw), red x, red y. Red's heading and velocity are not in the observation.",
              "input_tensor": tensor_check(np.asarray([o for o, _ in full]), np.asarray([e for _, e in full])),
              "session_log": str(harness.log.relative_to(ROOT)).replace("\\", "/")}
    (out / "play-observation-check.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    return report


def tensor_check(seen, rebuilt):
    """The tensor handed to the clone in play mode against (a) an independent rebuild from simulator state and
    (b) the layout and ranges of the demonstrations the clone was fitted on."""
    from autobot.sim import cli
    fields = cli.observation_fields("sim_debug", "experimental5_v1", "masked_zero_v1"); meta = cli.checkpoint_observation_metadata(MODEL)
    report = {"shape": list(seen.shape), "dtype_given_to_policy": "float32",
              "feature_order": fields if isinstance(fields, list) else str(fields), "checkpoint_metadata": meta,
              "scaling": "gyro / 34.9 rad/s, accel / 39.2 m/s^2, positions 2*x/1.15-1 and 2*y/1.15-1 (arena frame, origin at one corner), heading as sin and cos of yaw",
              "largest_gap_to_independent_rebuild_all_19_channels": float(np.abs(seen - rebuilt).max()),
              "largest_gap_by_block": {"imu_0_5": float(np.abs(seen[:, :6] - rebuilt[:, :6]).max()), "previous_action_6_12": float(np.abs(seen[:, 6:13]).max()),
                                       "blue_pose_13_16": float(np.abs(seen[:, 13:17] - rebuilt[:, 13:17]).max()), "red_position_17_18": float(np.abs(seen[:, 17:] - rebuilt[:, 17:]).max())},
              "previous_action_channels_all_zero": bool(np.all(seen[:, 6:13] == 0.0))}
    demos = ROOT / "runs/combined-clone-data/demonstrations.npz"
    if demos.is_file():
        data = np.load(demos); train = data["observations"][data["source"] < 2]
        report["demonstrations"] = {"file": str(demos.relative_to(ROOT)).replace("\\", "/"), "decisions": int(len(train)), "channels": int(train.shape[1]),
                                    "previous_action_channels_all_zero": bool(np.all(train[:, 6:13] == 0.0)),
                                    "channel_min": [round(float(v), 3) for v in train.min(axis=0)], "channel_max": [round(float(v), 3) for v in train.max(axis=0)],
                                    "play_values_outside_the_fitted_range_by_channel": [int(v) for v in ((seen < train.min(axis=0) - 1e-6) | (seen > train.max(axis=0) + 1e-6)).sum(axis=0)]}
    return report


def stage_coverage(out):
    """Were the states of the recorded spin round in the demonstrations the clone was fitted on?

    Distance from each state to its nearest demonstration state over the six pose channels (blue x, y, sin, cos of
    heading, red x, y; normalised units, where 0.1 is about 6 cm or 6 degrees). The yardstick is the same distance
    for held-out demonstrations from validation starts. DAgger added student-driven states from the same starts
    during fitting; those were not saved, so this measures the saved controller-driven demonstrations only."""
    demos = ROOT / "runs/combined-clone-data/demonstrations.npz"
    if not demos.is_file(): return {"skipped": "demonstrations not present"}
    data = np.load(demos); pose = data["observations"][:, 13:].astype(np.float64); source = data["source"]
    train = pose[source < 2]; held = pose[source == 2]

    def nearest(points):
        result = np.empty(len(points))
        for start in range(0, len(points), 256):
            chunk = points[start:start + 256]; result[start:start + 256] = np.sqrt(((chunk[:, None, :] - train[None, :, :]) ** 2).sum(axis=2)).min(axis=1)
        return result

    def describe(values): return {"decisions": int(len(values)), "median": round(float(np.median(values)), 3), "p90": round(float(np.percentile(values, 90)), 3),
                                  "max": round(float(values.max()), 3)}
    yardstick = nearest(held[:: max(1, len(held) // 3000)]); limit = float(np.percentile(yardstick, 99)); report = {
        "fitted_demonstration_decisions": int(len(train)), "yardstick_held_out_validation_demonstrations": describe(yardstick),
        "far_means": f"farther than {limit:.3f}, the 99th percentile of the yardstick"}
    red = data["observations"][source < 2][:, 17:]; blue = data["observations"][source < 2][:, 13:15]
    nearer_bottom_right = np.hypot(red[:, 0] - 1, red[:, 1] + 1) < np.hypot(red[:, 0] + 1, red[:, 1] - 1)
    report["fitted_decisions_with_red_nearer_the_bottom_right_pit"] = round(float(nearer_bottom_right.mean()), 3)
    report["fitted_decisions_with_red_more_than_0.45_m_from_blue"] = round(float((np.hypot(*(red - blue).T) * 0.575 > 0.45).mean()), 3)
    for number in (1, 2):
        path = out / f"round{number}-traces.npz"
        if not path.is_file(): continue
        observations = np.load(path)["clone_observations"][:, 13:].astype(np.float64); distance = nearest(observations)
        entry = {"all": describe(distance), "far_share": round(float((distance > limit).mean()), 3),
                 "far_share_by_second": [round(float((distance[a:a + 50] > limit).mean()), 2) for a in range(0, len(distance), 50)],
                 "first_decision_far_for_half_a_second": first_sustained(distance > limit, 25)}
        report[f"round_{number}"] = entry
    (out / "demonstration-coverage.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    return report


# --------------------------------------------------------------------------------------------------------------
def write_case(replay):
    """The recorded match as a regression case: red's commands, what the frozen clone did, and the rule to pass."""
    log = ROOT / replay["log"]; rounds = load_rounds(log); spin_round = next(r for r in replay["rounds"] if r.get("spin"))
    case = {"schema": "live_round_regression_v1", "name": "live_spin_2026_11_18",
            "description": "A person drove red against the frozen combined clone from the normal spawn. In round 2 red drove past blue; blue "
                           "dithered between turn_left and turn_right, then held turn_right to the end of the round without ever advancing.",
            "source_log": replay["log"], "source_log_sha256": replay["log_sha256"], "recorded_with": MODEL, "config": CONFIG, "seed": SEED,
            "profile": "sim_debug", "action_profile": "experimental5_v1", "previous_action_encoding": "masked_zero_v1",
            "scenarios": SCENARIOS, "scenario_index": SCENARIO_INDEX,
            "start": {"player": [0.575, 0.18, math.pi / 2], "opponent": [0.575, 0.97, -math.pi / 2]},
            "red_is": "the recorded wheel commands replayed open loop; red does not react to a different blue",
            "rounds": [{"round": n, "red_commands": [[list(map(float, c)), k] for c, k in rle([tuple(s["red_command"]) for s in steps])],
                        "recorded_blue_tactics": rle([s["blue_tactic"] for s in steps]), "recorded_end": steps[-1]["end_reason"]}
                       for n, steps in enumerate(rounds, 1)],
            "spin": {"round": spin_round["round"], **{k: spin_round["spin"][k] for k in ("first_decision", "end_decision", "tactic", "revolutions",
                                                                                         "nose_passes_over_red", "advance_decisions_with_nose_on_red")}},
            "pass_rule": {"applies_to_round": spin_round["round"],
                          "no_spin": f"never {SPIN_DEG:g} degrees of rotation in one direction without an advance decision",
                          "reaches_red": f"touches red (or red falls) at or after decision {spin_round['spin']['first_decision']}",
                          "no_own_fall": "blue does not fall into a pit"},
            "known_results": {name: {"passes": spin_round["controllers"][name]["case"]["passes"], "outcome": spin_round["controllers"][name]["outcome"]}
                              for name in ("clone", "teacher", "always_advance", "always_stop")}}
    CASE.parent.mkdir(parents=True, exist_ok=True); CASE.write_text(json.dumps(case, indent=1), encoding="utf-8")
    return case


def stage_replay_file(out, models=None):
    """Red's commands of the spin round alone, in the play log format, so `play --red-input-replay` shows that round.

    (Replaying the whole session log would stop at the end of round 1.) Each model is then run against it through
    the real viewer loop, headless, and what blue did is compared with the recording."""
    import verify_play_mode as play
    case = load_case(); item = next(r for r in case["rounds"] if r["round"] == case["spin"]["round"]); path = out / "spin-round-red-commands.jsonl"
    path.write_text("".join(json.dumps({"type": "step", "time_s": round((i + 1) * STEP_S, 2), "red_command": list(c)}) + "\n" for i, c in enumerate(item["red"])), encoding="utf-8")
    report = {"red_commands": str(path.relative_to(ROOT)).replace("\\", "/"), "decisions": len(item["red"]), "red_is": "the recorded human input of the spin round, replayed open loop", "models": {}}
    for name, model in (models or {"reference": MODEL}).items():
        for old in out.glob(f"spin-round-replayed-{name}-*.jsonl"): old.unlink()         # this script's own earlier headless logs
        harness = play.Harness([(2, [], True), (2, ["space"], True)], driver=lambda session: set(), max_frames=1200).run(
            play.arguments("play", str(model), SCENARIOS, SCENARIO_INDEX, SEED, "--red-input-replay", str(path), "--input-log", str(out / f"spin-round-replayed-{name}.jsonl")))
        steps = [r for r in play.read_log(harness.log) if r["type"] == "step"]; tactics = [r["blue_tactic"] for r in steps]
        longest = max(rle(tactics), key=lambda pair: pair[1]) if tactics else None
        report["models"][name] = {"model": str(model), "decisions": len(steps), "end": steps[-1]["end_reason"] if steps else None,
                                  "end_time_s": round(steps[-1]["time_s"], 2) if steps else None, "longest_run_of_one_tactic": longest,
                                  "hud_says_recording": any("red=recording" in line for line in harness.hud),
                                  "blue_tactics_equal_to_the_recorded_round": sum(a == b for a, b in zip(tactics, item["blue"])) if name == "reference" else None,
                                  "command": "python -m autobot.sim.cli " + " ".join(play.arguments("play", str(model).replace("\\", "/"), SCENARIOS, SCENARIO_INDEX, SEED,
                                             "--red-input-replay", str(path.relative_to(ROOT)).replace("\\", "/"), "--input-log",
                                             str((out / f"spin-round-replayed-{name}.jsonl").relative_to(ROOT)).replace("\\", "/"))),
                                  "headless_log": str(harness.log.relative_to(ROOT)).replace("\\", "/")}
    (out / "spin-round-replay.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    return report


def load_case(path=CASE):
    case = json.loads(Path(path).read_text(encoding="utf-8"))
    for item in case["rounds"]:
        item["red"] = [tuple(c) for c in unrle(item["red_commands"])]; item["blue"] = unrle(item["recorded_blue_tactics"])
    return case


def run_case(case, controller, session, round_number=None):
    """Replay one round of a case against a controller. Returns the trace and the pass/fail verdict."""
    number = case["pass_rule"]["applies_to_round"] if round_number is None else round_number
    item = next(r for r in case["rounds"] if r["round"] == number); trace = rollout(session.env, controller, item["red"])
    return trace, passes_case(trace, case["spin"]["first_decision"])


# --------------------------------------------------------------------------------------------------------------
def stage_figures(out, replay, probe_rows, sweep):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Polygon
    import yaml
    arena = yaml.safe_load((ROOT / CONFIG).read_text(encoding="utf-8"))["arena"]; colours = {"stop": "#888888", "advance": "#2ca02c", "reverse": "#9467bd", "turn_left": "#1f77b4", "turn_right": "#d62728"}

    def draw_arena(axis, title):
        axis.set_xlim(0, arena["width_m"]); axis.set_ylim(0, arena["height_m"]); axis.set_aspect("equal"); axis.set_title(title, fontsize=10)
        for centre in arena["pit_centres_m"]: axis.add_patch(Circle(centre, arena["pit_radius_m"], color="#f4c7c3", ec="#c0392b"))
        axis.add_patch(Polygon(arena["flipper_vertices_m"], color="#f2d48a")); axis.set_xlabel("x (m)"); axis.set_ylabel("y (m)")

    spin_round = next(r for r in replay["rounds"] if r.get("spin")); data = np.load(out / f"round{spin_round['round']}-traces.npz")
    poses, actions = data["clone_poses"], data["clone_actions"]; a = spin_round["spin"]["first_decision"]
    figure, axes = plt.subplots(1, 3, figsize=(17, 5.6))
    draw_arena(axes[0], f"Round {spin_round['round']}: paths (clone = blue line, red = recorded human)")
    axes[0].plot(poses[:a, 0], poses[:a, 1], color="#1f77b4", lw=1.5, label="blue before the spin"); axes[0].plot(poses[a:, 0], poses[a:, 1], color="#d62728", lw=1.0, label="blue during the spin")
    axes[0].plot(poses[:, 3], poses[:, 4], color="#444444", lw=1.5, ls="--", label="red")
    for t in (0, 50, 100, 200, 300, a, 500):
        axes[0].annotate(f"{t * STEP_S:.0f}s", poses[t, 3:5], fontsize=7, color="#444444"); axes[0].annotate(f"{t * STEP_S:.0f}s", poses[t, :2], fontsize=7, color="#1f77b4")
    axes[0].legend(fontsize=7, loc="lower right")
    time = np.arange(len(actions)) * STEP_S; bearing = np.degrees([wrap(math.atan2(p[4] - p[1], p[3] - p[0]) - p[2]) for p in poses[:-1]])
    axes[1].scatter(time, bearing, c=[colours[TACTICS[x]] for x in actions], s=4); axes[1].axhspan(-NOSE_ON_DEG, NOSE_ON_DEG, color="#2ca02c", alpha=0.15)
    axes[1].axvline(a * STEP_S, color="k", lw=0.8, ls=":"); axes[1].set_title("Angle from blue's nose to red (0 = pointing at red), coloured by blue's tactic", fontsize=9)
    axes[1].set_xlabel("time (s)"); axes[1].set_ylabel("degrees"); axes[1].set_ylim(-185, 185)
    for name, colour in colours.items():
        if name in ("advance", "turn_left", "turn_right"): axes[1].scatter([], [], c=colour, s=12, label=name)
    axes[1].legend(fontsize=7, loc="upper right")
    shadow = data["clone_shadow"]; axes[2].set_title("Tactic each decision: clone, and the teacher's label on the same state", fontsize=9)
    axes[2].scatter(time, np.full(len(actions), 1.0), c=[colours[TACTICS[x]] for x in actions], s=6, marker="|")
    axes[2].scatter(time, np.full(len(actions), 0.0), c=[colours[TACTICS[x]] for x in shadow], s=6, marker="|"); axes[2].set_yticks([0, 1]); axes[2].set_yticklabels(["teacher", "clone"])
    axes[2].set_ylim(-0.6, 1.6); axes[2].set_xlabel("time (s)"); axes[2].axvline(a * STEP_S, color="k", lw=0.8, ls=":")
    figure.tight_layout(); figure.savefig(out / "round-spin.png", dpi=130); plt.close(figure)

    figure, axes = plt.subplots(1, 4, figsize=(21, 5.6)); blue = poses[a]
    panels = (("Clone's choice by red position", lambda r: colours[r["clone"]]), ("Teacher's choice by red position", lambda r: colours[r["teacher"]]),
              ("Clone, next 10 s, red at rest", lambda r: "#d62728" if r["clone_closed_loop"]["spins"] else "#2ca02c" if r["clone_closed_loop"]["touches_red"] else "#bbbbbb"),
              ("Teacher, next 10 s, red at rest", lambda r: "#d62728" if r["teacher_closed_loop"]["spins"] else "#2ca02c" if r["teacher_closed_loop"]["touches_red"] else "#bbbbbb"))
    for axis, (title, colour) in zip(axes, panels):
        draw_arena(axis, title); axis.scatter([r["red"][0] for r in probe_rows], [r["red"][1] for r in probe_rows], c=[colour(r) for r in probe_rows], s=60, marker="s")
        axis.arrow(blue[0], blue[1], 0.09 * math.cos(blue[2]), 0.09 * math.sin(blue[2]), width=0.012, color="k"); axis.plot(blue[3], blue[4], "x", color="k", ms=9)
    for name in ("advance", "turn_left", "turn_right"): axes[0].scatter([], [], c=colours[name], s=30, marker="s", label=name)
    axes[0].legend(fontsize=7, loc="lower left")
    for label, colour in (("spins", "#d62728"), ("touches red", "#2ca02c"), ("neither", "#bbbbbb")): axes[2].scatter([], [], c=colour, s=30, marker="s", label=label)
    axes[2].legend(fontsize=7, loc="lower left")
    figure.suptitle(f"Blue held at its pose just before the spin (arrow); each square is one red position; x = where red really was", fontsize=10)
    figure.tight_layout(); figure.savefig(out / "red-position-probe.png", dpi=130); plt.close(figure)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("stage", choices=["replay", "cases", "probe", "play", "coverage", "figures", "case", "replayfile", "all"])
    parser.add_argument("--model", action="append", default=[], metavar="NAME=CHECKPOINT", help="replayfile: also run this checkpoint against the spin round")
    args = parser.parse_args()
    if args.stage == "replayfile":
        OUT.mkdir(parents=True, exist_ok=True); models = {"reference": MODEL, **dict(item.split("=", 1) for item in args.model)}
        print(json.dumps(stage_replay_file(OUT, models), indent=1)); return
    OUT.mkdir(parents=True, exist_ok=True); replay, session, saved = stage_replay(OUT)
    print(json.dumps({k: v for k, v in replay.items() if k != "rounds"}, indent=1))
    for entry in replay["rounds"]: print(json.dumps({k: v for k, v in entry.items() if k != "timeline"}, indent=1))
    if args.stage in ("cases", "all"): print(json.dumps(stage_cases(OUT, session), indent=1))
    rows = sweep = None
    if args.stage == "figures":
        stored = json.loads((OUT / "red-position-probe.json").read_text(encoding="utf-8")); rows, sweep = stored["positions"], stored["heading_sweep"]
    if args.stage in ("probe", "all"):
        spin_round = next(r for r in replay["rounds"] if r.get("spin")); data = np.load(OUT / f"round{spin_round['round']}-traces.npz")
        saved[max(saved)]["spin_red"] = [float(v) for v in data["clone_poses"][-1, 3:5]]
        summary, rows, sweep = stage_probe(OUT, session, saved); print(json.dumps(summary, indent=1))
    if args.stage in ("play", "all"): print(json.dumps({k: v for k, v in stage_play(OUT).items() if k != "samples"}, indent=1))
    if args.stage in ("coverage", "all"): print(json.dumps(stage_coverage(OUT), indent=1))
    if args.stage in ("case", "all"): print(json.dumps(write_case(replay)["known_results"], indent=1))
    if args.stage in ("figures", "all"): stage_figures(OUT, replay, rows, sweep)


if __name__ == "__main__": main()
