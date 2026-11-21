"""Does a simple reactive baseline recover where the clone and its teacher do not? Simulation-only; no PPO.

    python scripts/reactive_recovery.py make        # fresh normal starts x three red behaviours (written once)
    python scripts/reactive_recovery.py baseline    # the reactive baseline against the reference clone, the teacher and controls
    python scripts/reactive_recovery.py collect     # (only if the gate passed) demonstrations from the reactive baseline
    python scripts/reactive_recovery.py train       # one challenger recipe, three seeds, balanced cloning + DAgger
    python scripts/reactive_recovery.py validate    # choose the seed on validation cases only
    python scripts/reactive_recovery.py evaluate    # final comparison on test cases

Second pass (added after the first challenger was evaluated, see V2):
    python scripts/reactive_recovery.py make-anywhere       # both robots placed anywhere on the floor (written once)
    python scripts/reactive_recovery.py anywhere            # does the first challenger hold up from any geometry?
    python scripts/reactive_recovery.py collect --version v2 ; train --version v2 ; validate --version v2 ; evaluate --version v2

Background (runs/live-round-diagnostic-2026-11-18): in a live round the frozen combined clone held turn_right for
597 decisions. The scripted teacher it was cloned from (FlankController) does not spin on the same red input but
never reaches red either: it parks on its own go-round waypoint. So the teacher is not cloned further here.

The reactive baseline is the simplest thing that follows red: turn toward red's current bearing, advance when
the nose is within 12 degrees of it. It reads only what the clone's observation contains (blue's position and
heading, red's position). The gate (GATE) and the recommendation rule (RECOMMEND) were fixed before any result.

Red behaviours for the fresh cases are scripted (stationary, moving away, changing sides). They are not a person;
results on them say nothing about a live driver. The human recording is used only as a check, never for fitting
or for choosing a checkpoint.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path[:0] = [str(Path(__file__).resolve().parent), str(Path(__file__).resolve().parents[1])]
import clone_finetune as pilot  # noqa: E402
import combined_clone as combined  # noqa: E402
import controller_imitation as imitation  # noqa: E402
import heading_ladder as experiment  # noqa: E402
import live_round_diagnostic as live  # noqa: E402
import opponent_pool as pool_tools  # noqa: E402

ROOT = experiment.ROOT
RUN = ROOT / "runs/reactive-recovery-2026-11-18"
CASES_PATH = ROOT / "config/recovery_cases_v1.json"
REFERENCE = pool_tools.REFERENCE
ANYWHERE_PATH = ROOT / "config/recovery_anywhere_v1.json"
ANYWHERE_SEED = 181119
ANYWHERE_STARTS = {"train": 60, "validation": 20, "test": 60}
ANYWHERE_SEED_BASE = {"train": 360_000, "validation": 370_000, "test": 380_000}
CASES_SEED = 181118
STARTS_PER_ARCHETYPE = {"train": 8, "validation": 4, "test": 10}
SEED_BASE = {"train": 330_000, "validation": 340_000, "test": 350_000}
RED_KINDS = ("stationary", "moving_away", "changing_sides")
RED_RANGES = {"stationary": {},
              "moving_away": {"trigger_m": (0.30, 0.50), "speed": (0.7, 1.0), "hold_s": (0.5, 1.2), "steer_gain": (1.5, 3.0)},
              "changing_sides": {"speed": (0.7, 1.0), "flank_m": (0.16, 0.26), "beyond_m": (0.20, 0.40), "pause_s": (1.0, 3.5),
                                 "start_delay_s": (0.2, 1.0), "steer_gain": (1.5, 3.0)}}
SEEDS = (0, 1, 2)
STEP_S = 0.02
WINDOW = 50                    # one second of decisions
SPIN_DEG = 240.0               # spinning: at least this much net rotation within one second, staying within 8 cm
STALL_M = 0.03                 # stalled: moved less than this in one second, without touching red and without spinning
OUTCOMES = ("wins", "own_falls", "timeouts", "red_self_falls", "draws", "simulation_bugs")
ORDINARY_DEMO_SEED = imitation.DATA_SEED + 500_000
DAGGER_SEED = imitation.DATA_SEED + 600_000
POLICIES = {"reference": {"kind": "checkpoint", "path": str(REFERENCE)}, "teacher": {"kind": "turn_then_advance_controller"},
            "reactive": {"kind": "pursuit"}, "always_advance": {"kind": "always_advance"}, "always_stop": {"kind": "always_stop"}}
# Fixed before the baseline was run.
GATE = {"failing_case": "on the recorded spin round, from the start and from the saved state just before the spin, the baseline does not spin "
                        "(never 720 degrees one way without an advance), touches red, and does not fall",
        "fresh_variants": "on the fresh train and validation cases (red stationary, moving away, changing sides): touches red in at least 90% of "
                          "episodes, falls in on at most 5%, and spends less than half the teacher's time spinning or stalled",
        "drill": "the turning drill can be preserved: on turn-needed drill validation starts it wins at least 90% of what the teacher wins, at each pit",
        "train_if": "all three hold; otherwise keep the diagnostic and train nothing",
        "reach_share": 0.90, "max_fall_share": 0.05, "stall_ratio": 0.5, "drill_share": 0.90}
SELECT = {"data": "validation cases only: fresh validation cases, pool validation cases, drill and ordinary validation starts",
          "eligible": "wins at least 90% of the turn-needed drill validation episodes at each pit",
          "score": "blue-caused wins minus own falls over the fresh, pool and ordinary validation cases; ties go to less time spinning or stalled",
          "drill_share": 0.90}
RECOMMEND = {"applies_to": "the selected challenger against the reference clone, on test cases only",
             "keeps_drill": "at least 90% of the reference clone's turn-needed drill wins at each pit",
             "responds": "on fresh test cases: touches red in more episodes than the reference and spends less time spinning or stalled",
             "moves": "on fresh test cases: before first contact it covers ground at least as fast as the reference does (distance travelled before "
                      "first contact divided by the time before first contact, the whole episode when there is none), so spinning was not swapped for standing still",
             "amended": "The `moves` check was reworded at 01:40 on 18 November, after the gate run on train and validation cases and before training "
                        "finished or any test case was run. Its first wording compared distance travelled before first contact, which would have scored a "
                        "direct approach (0.6 m, then contact) below wandering for 20 s without ever touching red (1.2 m).",
             "does_not_fall": "own falls on at most 5% of episodes of each test set, or no more often than the reference there",
             "wins": "blue-caused wins over the fresh test cases plus the harder-opponent test pool are at least the reference's",
             "verdict": "recommend the challenger only if all five hold; otherwise the reference stays",
             "drill_share": 0.90, "max_fall_share": 0.05}


# Fixed at 02:30 on 18 November, after the first challenger's test results and before any anywhere case was run.
# Reason: put at the saved state just before the recorded spin (blue upper right, red 0.5 m away lower left),
# every seed of the first challenger failed to reach red. Its training starts all had blue at the bottom.
V2 = {"question": "is that an isolated state, or does the first challenger fail from geometries its starts never covered?",
      "measure": "the first challenger on anywhere train and validation cases (both robots placed anywhere on clear floor; never a test case)",
      "train_second_challenger_if": "it touches red in fewer than 90% of those episodes, or spends more than 2 s per episode spinning or stalled, "
                                    "while the reactive baseline itself does neither",
      "second_challenger": "same recipe and teacher, one more pool: reactive-baseline demonstrations and DAgger on the anywhere train cases",
      "choice": "between the first challenger and the best second-challenger seed, on validation cases only: must keep the drill; higher blue-caused "
                "wins minus own falls over the fresh, pool, ordinary and anywhere validation cases; ties go to less time spinning or stalled",
      "added_checks_for_the_recommendation": "from the saved pre-spin state the challenger must not spin and must touch red; on anywhere test cases it "
                                             "must touch red more often than the reference and spend less time spinning or stalled. These only make the rule stricter.",
      "reach_share": 0.90, "max_idle_s": 2.0}


def wrap(angle): return math.atan2(math.sin(angle), math.cos(angle))


# ---------------------------------------------------------------- the reactive baseline

class PursuitController:
    """Turn toward red's current bearing; advance when the nose is within ALIGN_RAD of it. Nothing else.

    It uses only the sim_debug observation the clone gets: blue's position (channels 13, 14), sin and cos of blue's
    heading (15, 16) and red's position (17, 18). The arena is square, so the normalised positions keep directions.
    """
    ALIGN_RAD = math.radians(12)

    def bind(self, env): pass

    @classmethod
    def bearing(cls, observation):
        o = np.asarray(observation, dtype=np.float64).reshape(-1)
        return wrap(math.atan2(o[18] - o[14], o[17] - o[13]) - math.atan2(o[15], o[16]))

    def predict(self, observation, deterministic=True):
        error = self.bearing(observation)
        if abs(error) <= self.ALIGN_RAD: return np.array(1), None          # advance
        return np.array(3 if error > 0 else 4), None                         # turn_left / turn_right

    def labels(self, observations):
        o = np.asarray(observations, dtype=np.float64)
        error = np.arctan2(o[:, 18] - o[:, 14], o[:, 17] - o[:, 13]) - np.arctan2(o[:, 15], o[:, 16])
        error = np.arctan2(np.sin(error), np.cos(error))
        return np.where(np.abs(error) <= self.ALIGN_RAD, 1, np.where(error > 0, 3, 4)).astype(np.int64)


def load_policy(spec):
    if spec["kind"] == "pursuit": return PursuitController()
    return pilot.load_actor(spec)[0]


# ---------------------------------------------------------------- scripted reds for the fresh cases

class RecoveryRed:
    """Scripted red for the fresh cases. Seeded and replayable; it reads simulator truth but blue never sees its settings.

    stationary      never moves
    moving_away     sits still until blue comes within `trigger_m`, then drives directly away (forwards or in
                    reverse, whichever needs less turning) for `hold_s`, and waits again
    changing_sides  after `start_delay_s` drives past blue to a point on blue's far side, waits `pause_s`, then
                    crosses back on the other flank, and so on
    """

    def __init__(self, kind, parameters, seed):
        if kind not in RED_KINDS: raise ValueError(f"Unknown red behaviour: {kind}")
        self.kind = kind; self.p = dict(parameters); self.rng = np.random.default_rng(int(seed)); self.state = {}

    @staticmethod
    def _safe(point, arena, margin=0.12):
        """Pull a goal back onto clear floor: inside the walls, away from both pits and the flipper corner."""
        centre = np.array([arena["width_m"] / 2, arena["height_m"] / 2]); p = np.asarray(point, dtype=float)
        for _ in range(40):
            inside = margin <= p[0] <= arena["width_m"] - margin and margin <= p[1] <= arena["height_m"] - margin
            clear = min(math.dist(p, pit) for pit in arena["pit_centres_m"]) >= arena["pit_radius_m"] + margin
            flipper = p[0] + p[1] > arena["width_m"] + arena["flipper_vertices_m"][0][0] - margin
            if inside and clear and not flipper: return p
            p = p + (centre - p) * 0.1
        return centre

    @staticmethod
    def _pit_ahead(sim, red, forward):
        arena = sim.cfg["arena"]; yaw = red["yaw_rad"]; direction = math.copysign(1.0, forward)
        ahead = red["position_m"][:2] + direction * 0.09 * np.array([math.cos(yaw), math.sin(yaw)])
        return min(math.dist(ahead, pit) for pit in arena["pit_centres_m"]) - arena["pit_radius_m"] < 0.03

    def _steer(self, sim, red, heading, speed, allow_reverse):
        from autobot.sim.opponents import mix
        error = wrap(heading - red["yaw_rad"]); forward = speed
        if allow_reverse and abs(error) > math.pi / 2: error = wrap(heading - red["yaw_rad"] - math.pi); forward = -0.65 * speed
        forward *= max(0.3, math.cos(error)); turn = float(np.clip(self.p["steer_gain"] * error, -1.0, 1.0)) * 0.8
        if self._pit_ahead(sim, red, forward): return None
        return mix(forward, turn)

    def drive(self, sim):
        if self.kind == "stationary": return (0.0, 0.0)
        red = sim.truth(1); blue = sim.truth(0); now = float(sim.data.time); own = red["position_m"][:2]; other = blue["position_m"][:2]
        state = self.state; arena = sim.cfg["arena"]
        if self.kind == "moving_away":
            if now >= state.get("until", -1.0):
                if math.dist(own, other) >= self.p["trigger_m"]: return (0.0, 0.0)
                state["until"] = now + self.p["hold_s"]
            command = self._steer(sim, red, math.atan2(own[1] - other[1], own[0] - other[0]), self.p["speed"], True)
            if command is None: state["until"] = now; return (0.0, 0.0)                    # a pit is in the way: stop running
            return command
        if not state: state.update(phase="wait", until=self.p["start_delay_s"], side=int(self.rng.choice([-1, 1])))
        if state["phase"] in ("wait", "pause"):
            if now < state["until"]: return (0.0, 0.0)
            along = other - own; along = along / max(float(np.linalg.norm(along)), 1e-9); across = np.array([-along[1], along[0]])
            state.update(phase="cross", goal=self._safe(other + along * self.p["beyond_m"] + state["side"] * self.p["flank_m"] * across, arena), until=now + 3.0)
        goal = state["goal"]; command = None
        if math.dist(own, goal) > 0.07 and now < state["until"]:
            command = self._steer(sim, red, math.atan2(goal[1] - own[1], goal[0] - own[0]), self.p["speed"], False)
        if command is None:
            state.update(phase="pause", until=now + self.p["pause_s"], side=-state["side"]); return (0.0, 0.0)
        return command


def resolve_red(kind, rng):
    return {name: round(float(rng.uniform(low, high)), 4) for name, (low, high) in RED_RANGES[kind].items()}


# ---------------------------------------------------------------- cases

def protected_starts():
    """Every start a new case must not repeat: held-out and validation starts of earlier experiments, the opponent
    pool, earlier demonstration starts, and the exact frozen normal starts (the human recording's spawn)."""
    from autobot.sim.scenarios import load_scenario_file
    ladder = experiment.load_ladder(); _, confirmation = pilot.load_mix(); starts = combined.load_starts()
    pool = json.loads(pool_tools.POOL_PATH.read_text(encoding="utf-8"))
    frozen = [dict(item) for item in load_scenario_file(ROOT / live.SCENARIOS) if item.get("group") == "normal"]
    pool_starts = list({case["start"]["name"]: case["start"] for split in pool["splits"].values() for case in split}.values())
    return pilot.heldout_starts(ladder) + confirmation + starts["train"] + starts["validation"] + pool_starts + frozen


def build_cases(arena):
    from autobot.sim.scenarios import load_scenario_file
    archetypes = load_scenario_file(ROOT / "runs/ppo-curriculum-overnight/training_curriculum.json")["groups"]["ordinary"]
    rng = np.random.default_rng(CASES_SEED); forbidden = protected_starts(); splits = {}; made = []
    for split, count in STARTS_PER_ARCHETYPE.items():
        starts = pilot.jittered(archetypes, count, rng, f"recovery_{split}", f"recovery_{split}", forbidden + made, arena); made += starts; cases = []
        for kind in RED_KINDS:
            for start in starts:
                index = len(cases)
                cases.append({"name": f"{split}_{kind}_{start['name']}", "split": split, "kind": kind, "family": start["derived_from"],
                              "seed": SEED_BASE[split] + index, "start": start,
                              "red": {"kind": kind, "parameters": resolve_red(kind, rng), "driver_seed": 900_000 + SEED_BASE[split] + index}})
        splits[split] = cases
    cases = {"schema": "recovery_cases_v1", "seed": CASES_SEED, "config": experiment.relative(experiment.CONFIG),
             "provenance": f"seeded offsets (+-{pilot.JITTER_M} m, +-{pilot.JITTER_RAD} rad) of the four ordinary starts of the frozen training curriculum; "
                           "none is within the same-case tolerance of an earlier held-out, validation, demonstration or pool start, or of a frozen normal start",
             "red_behaviours": {kind: RED_RANGES[kind] for kind in RED_KINDS}, "red_is": "scripted (RecoveryRed), not a person",
             "splits": splits}
    check_cases(cases, forbidden)
    return cases


def check_cases(cases, forbidden):
    starts = {split: list({case["start"]["name"]: case["start"] for case in items}.values()) for split, items in cases["splits"].items()}
    everything = [start for items in starts.values() for start in items]
    clashes = [(a["name"], b["name"]) for a in everything for b in forbidden if pilot.same_case(a, b)]
    clashes += [(a["name"], b["name"]) for i, a in enumerate(everything) for b in everything[i + 1:] if pilot.same_case(a, b)]
    if clashes: raise ValueError(f"Recovery starts repeat protected or each other's cases: {clashes[:3]}")
    names = [case["name"] for items in cases["splits"].values() for case in items]
    if len(names) != len(set(names)): raise ValueError("Duplicate case names")


def load_cases():
    cases = json.loads(CASES_PATH.read_text(encoding="utf-8")); check_cases(cases, protected_starts()); return cases


def make(args):
    from autobot.sim.core import load_config
    if CASES_PATH.exists(): raise SystemExit(f"{CASES_PATH} already exists; not overwriting.")
    cases = build_cases(load_config(str(experiment.CONFIG))["arena"])
    CASES_PATH.write_text(json.dumps(cases, indent=1), encoding="utf-8")
    print(CASES_PATH, {split: len(items) for split, items in cases["splits"].items()})


def build_anywhere(arena):
    """Both robots anywhere on clear floor, any headings, at least 0.30 m apart; the three red behaviours on each start."""
    rng = np.random.default_rng(ANYWHERE_SEED); forbidden = protected_starts() + [case["start"] for items in load_cases()["splits"].values() for case in items]
    splits = {}; made = []
    for split, count in ANYWHERE_STARTS.items():
        starts = []
        while len(starts) < count:
            pose = lambda: [round(float(rng.uniform(0.10, arena["width_m"] - 0.10)), 4), round(float(rng.uniform(0.10, arena["height_m"] - 0.10)), 4),
                            round(float(rng.uniform(-math.pi, math.pi)), 4)]
            start = {"name": f"anywhere_{split}_{len(starts)}", "group": f"anywhere_{split}", "derived_from": "anywhere", "player": pose(), "opponent": pose()}
            if not pilot.placeable(start, arena) or any(pilot.same_case(start, other) for other in forbidden + made + starts): continue
            starts.append(start)
        made += starts; cases = []
        for kind in RED_KINDS:
            for start in starts:
                index = len(cases)
                cases.append({"name": f"{split}_{kind}_{start['name']}", "split": split, "kind": kind, "family": "anywhere", "seed": ANYWHERE_SEED_BASE[split] + index,
                              "start": start, "red": {"kind": kind, "parameters": resolve_red(kind, rng), "driver_seed": 900_000 + ANYWHERE_SEED_BASE[split] + index}})
        splits[split] = cases
    cases = {"schema": "recovery_anywhere_v1", "seed": ANYWHERE_SEED, "config": experiment.relative(experiment.CONFIG),
             "provenance": "both robots placed uniformly on clear floor (0.10 m from walls, pits and the flipper corner, at least 0.30 m apart) with uniform headings",
             "red_behaviours": {kind: RED_RANGES[kind] for kind in RED_KINDS}, "red_is": "scripted (RecoveryRed), not a person", "splits": splits}
    check_cases(cases, forbidden)
    return cases


def load_anywhere():
    return json.loads(ANYWHERE_PATH.read_text(encoding="utf-8"))


def make_anywhere(args):
    from autobot.sim.core import load_config
    if ANYWHERE_PATH.exists(): raise SystemExit(f"{ANYWHERE_PATH} already exists; not overwriting.")
    cases = build_anywhere(load_config(str(experiment.CONFIG))["arena"])
    ANYWHERE_PATH.write_text(json.dumps(cases, indent=1), encoding="utf-8")
    print(ANYWHERE_PATH, {split: len(items) for split, items in cases["splits"].items()})


def start_cases(starts, seeds, group):
    """Scripted-red cases for a list of starts, with the episode seeds of `ladder.run_starts` (seed + start index)."""
    return [{"name": f"{start['name']}@{seed}", "seed": int(seed) + index, "start": start, "set_group": group}
            for seed in seeds for index, start in enumerate(starts)]


def validation_sets(baseline, anywhere=False):
    """Cases used for the gate (`baseline`: fresh and pool train+validation) or for choosing a challenger seed (validation only)."""
    ladder = experiment.load_ladder(); cases = load_cases(); pool = pool_tools.load_pool(); splits = ("train", "validation") if baseline else ("validation",)
    return {**({"anywhere": load_anywhere()["splits"]["validation"]} if anywhere else {}),
            "fresh": [case for split in splits for case in cases["splits"][split]],
            "pool": [case for split in splits for case in pool["splits"][split]],
            "drill": start_cases([start for rung in ladder["rungs"] for start in rung["validation"]], [combined.VALIDATION_SEED], "drill_validation"),
            "ordinary": start_cases(combined.load_starts()["validation"], [combined.VALIDATION_SEED], "ordinary_validation")}


def test_sets(anywhere=False):
    ladder = experiment.load_ladder(); _, confirmation = pilot.load_mix()
    return {**({"anywhere": load_anywhere()["splits"]["test"]} if anywhere else {}),
            "fresh": load_cases()["splits"]["test"], "pool": pool_tools.load_pool()["splits"]["test"],
            "drill": start_cases(ladder["holdout"], pilot.FINAL_SEEDS, "drill_holdout"),
            "ordinary": start_cases(experiment.transfer_scenarios(), pilot.FINAL_SEEDS, "earlier_ordinary")
                        + start_cases(confirmation, pilot.FINAL_SEEDS, "new_confirmation")}


# ---------------------------------------------------------------- running one case

def trace_metrics(poses, actions, contact):
    """What blue did in one episode. `poses` rows are blue x, y, yaw, red x, y, yaw; one more row than decisions."""
    poses = np.asarray(poses, dtype=float); actions = np.asarray(actions); contact = np.asarray(contact, dtype=bool); n = len(actions)
    turn = np.degrees(np.arctan2(np.sin(np.diff(poses[:, 2])), np.cos(np.diff(poses[:, 2])))); step = np.linalg.norm(np.diff(poses[:, :2], axis=0), axis=1)
    distance = np.linalg.norm(poses[:, :2] - poses[:, 3:5], axis=1); touched = np.flatnonzero(contact); first = int(touched[0]) if len(touched) else None
    spinning = stalled = 0
    if n >= WINDOW:
        rotation = np.concatenate([[0.0], np.cumsum(turn)]); net = np.abs(rotation[WINDOW:] - rotation[:-WINDOW])
        moved = np.linalg.norm(poses[WINDOW:, :2] - poses[:-WINDOW, :2], axis=1)
        touching = np.concatenate([[0], np.cumsum(contact)]); any_contact = (touching[WINDOW:] - touching[:-WINDOW]) > 0
        spin = (net >= SPIN_DEG) & (moved < 0.08); stall = ~spin & (moved < STALL_M) & ~any_contact
        spinning, stalled = int(spin.sum()), int(stall.sum())
    before = slice(0, first if first is not None else n)
    return {"decisions": int(n), "first_contact_s": None if first is None else round(first * STEP_S, 2), "reaches_red": first is not None,
            "pre_contact_path_m": round(float(step[before].sum()), 3), "pre_contact_gap_closed_m": round(float(distance[0] - distance[:(first if first is not None else n) + 1].min()), 3),
            "spinning_s": round(spinning * STEP_S, 2), "stalled_s": round(stalled * STEP_S, 2), "contact_s": round(int(contact.sum()) * STEP_S, 2),
            "path_m": round(float(step.sum()), 3), "closest_m": round(float(distance.min()), 3), "final_distance_m": round(float(distance[-1]), 3),
            "tactic_switches": int(np.sum(actions[1:] != actions[:-1])), "advance_share": round(float(np.mean(actions == 1)), 3)}


def run_case(policy, env, case, driver=None, cap=None, keep=False):
    """One reproducible episode. Red is the frozen scripted opponent, a pool style, a RecoveryRed, or recorded commands.

    With `driver`, that policy drives instead (for DAgger) and the observations it saw are returned."""
    from autobot.sim.opponents import end_outcome
    opponent = case if case.get("style") not in (None, "scripted") else None
    observation, _ = env.reset(seed=int(case["seed"]), options={"scenario": case["start"], "opponent": opponent})
    for item in (policy, driver):
        if item is not None and hasattr(item, "bind"): item.bind(env)
    red = RecoveryRed(case["red"]["kind"], case["red"]["parameters"], case["red"]["driver_seed"]) if case.get("red") else None
    sim = env.sim; poses = [live.pose(sim)]; actions = []; contact = []; observations = []; info = {}; ended = False
    while True:
        action = int((driver or policy).predict(observation, deterministic=True)[0])
        if keep or driver is not None: observations.append(observation)
        observation, _, terminated, truncated, info = env.step(action, None if red is None else red.drive(sim))
        actions.append(action); contact.append(bool(info["opponent_contact"])); poses.append(live.pose(sim))
        if terminated or truncated: ended = True; break
        if cap and len(actions) >= cap: break
    touched = np.flatnonzero(contact); recent = len(touched) > 0 and len(actions) - 1 - int(touched[-1]) < round(0.5 / STEP_S)
    outcome = end_outcome(info.get("round_end_reason"), info.get("round_end_bot"), recent) if ended else "cut"
    start = case["start"]
    record = {"case": case["name"], "seed": int(case["seed"]), "outcome": outcome, "simulated_seconds": round(len(actions) * STEP_S, 2),
              "kind": case.get("kind"), "style": case.get("style"), "family": case.get("family") or start.get("derived_from"), "split": case.get("split"),
              "group": case.get("set_group") or start.get("group"), **{key: start[key] for key in ("pit", "heading_offset_deg") if key in start},
              **trace_metrics(poses, actions, contact)}
    if keep or driver is not None:
        record["observations"] = np.asarray(observations, dtype=np.float32); record["actions"] = np.asarray(actions)
    if keep: record["poses"] = np.asarray(poses)
    return record


def _chunk(task):
    spec, cases = task
    policy = load_policy(spec); env = pool_tools.make_env()
    return [run_case(policy, env, case) for case in cases]


def run_cases(spec, cases, workers, pool=None):
    chunks = [cases[index::workers] for index in range(workers) if cases[index::workers]]
    if pool is None:
        with ProcessPoolExecutor(max_workers=len(chunks)) as own: parts = list(own.map(_chunk, [(spec, chunk) for chunk in chunks]))
    else: parts = list(pool.map(_chunk, [(spec, chunk) for chunk in chunks]))
    by_name = {episode["case"]: episode for part in parts for episode in part}
    return [by_name[case["name"]] for case in cases]


def run_policies(folder, sets, policies, workers):
    """{policy: {set: [episodes]}}, cached per policy and set. One worker pool serves every policy and set."""
    results = {name: {} for name in policies}; todo = []
    for name in policies:
        for set_name in sets:
            path = RUN / folder / f"{set_name}_{name}.json"
            if path.is_file(): results[name][set_name] = json.loads(path.read_text(encoding="utf-8"))
            else: todo.append((name, set_name, path))
    if todo:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for name, set_name, path in todo:
                started = time.perf_counter(); episodes = run_cases(policies[name], sets[set_name], workers, pool)
                path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(episodes), encoding="utf-8"); results[name][set_name] = episodes
                print(f"{folder} {set_name:9s} {name:16s} {len(episodes):4d} episodes in {time.perf_counter() - started:5.0f}s", flush=True)
    return credited({name: {set_name: results[name][set_name] for set_name in sets} for name in policies})


def credited(results):
    """A win on a case where red also falls against a blue that never moves is red's own doing, not a win."""
    adjusted = {}
    for name, sets in results.items():
        adjusted[name] = {}
        for set_name, episodes in sets.items():
            alone = {e["case"] for e in results.get("always_stop", {}).get(set_name, []) if e["outcome"] in ("wins", "red_self_falls")}
            adjusted[name][set_name] = [dict(e, outcome="red_self_falls" if e["outcome"] == "wins" and e["case"] in alone else e["outcome"]) for e in episodes]
    return adjusted


def cell(episodes):
    counts = Counter(e["outcome"] for e in episodes); n = max(len(episodes), 1); firsts = [e["first_contact_s"] for e in episodes if e["first_contact_s"] is not None]
    mean = lambda key: round(float(np.mean([e[key] for e in episodes])), 2) if episodes else None
    return {"episodes": len(episodes), **{key: counts[key] for key in OUTCOMES}, "reaches_red": sum(e["reaches_red"] for e in episodes),
            "reach_share": round(sum(e["reaches_red"] for e in episodes) / n, 3), "median_first_contact_s": round(float(np.median(firsts)), 2) if firsts else None,
            "mean_pre_contact_path_m": mean("pre_contact_path_m"),
            "pre_contact_speed_m_s": round(sum(e["pre_contact_path_m"] for e in episodes)
                                           / max(sum(e["simulated_seconds"] if e["first_contact_s"] is None else e["first_contact_s"] for e in episodes), 1e-9), 3) if episodes else None,
            "mean_spinning_s": mean("spinning_s"), "mean_stalled_s": mean("stalled_s"),
            "mean_contact_s": mean("contact_s"), "mean_seconds": mean("simulated_seconds")}


def tables(sets, first_gated):
    """Per-set tables for one policy."""
    table = {}
    for part in ("fresh", "anywhere"):
        if part in sets: table[part] = {"all": cell(sets[part]), **{kind: cell([e for e in sets[part] if e["kind"] == kind]) for kind in RED_KINDS}}
    if "pool" in sets:
        styles = list(dict.fromkeys(e["style"] for e in sets["pool"]))
        table["pool"] = {"all": cell(sets["pool"]), "harder": cell([e for e in sets["pool"] if e["style"] != "scripted"]),
                         **{style: cell([e for e in sets["pool"] if e["style"] == style]) for style in styles}}
    if "drill" in sets:
        turn = [e for e in sets["drill"] if abs(e["heading_offset_deg"]) >= first_gated]
        table["drill_turn_needed"] = {pit: cell([e for e in turn if e["pit"] == pit]) for pit in pilot.PIT_NAMES}
    if "ordinary" in sets:
        groups = list(dict.fromkeys(e["group"] for e in sets["ordinary"]))
        table["ordinary"] = {"all": cell(sets["ordinary"]), **({group: cell([e for e in sets["ordinary"] if e["group"] == group]) for group in groups} if len(groups) > 1 else {})}
    return table


def first_gated_offset():
    return min(rung["abs_heading_offset_deg"] for rung in experiment.load_ladder()["rungs"] if not rung["warmup"])


def show(c): return f"{c['wins']}/{c['own_falls']}/{c['timeouts']}" + (f" (+{c['simulation_bugs']} wall breach)" if c["simulation_bugs"] else "")


def print_tables(summary):
    print("wins / own falls / timeouts ; reach = episodes touching red ; spin, stall = mean seconds per episode")
    for name, table in summary.items():
        parts = []
        if "fresh" in table: parts += [f"{kind} {show(table['fresh'][kind])} reach {table['fresh'][kind]['reaches_red']}/{table['fresh'][kind]['episodes']} "
                                       f"spin {table['fresh'][kind]['mean_spinning_s']} stall {table['fresh'][kind]['mean_stalled_s']}" for kind in RED_KINDS]
        if "anywhere" in table:
            c = table["anywhere"]["all"]; parts.append(f"anywhere {show(c)} reach {c['reaches_red']}/{c['episodes']} spin {c['mean_spinning_s']} stall {c['mean_stalled_s']}")
        if "pool" in table: parts.append(f"pool harder {show(table['pool']['harder'])} of {table['pool']['harder']['episodes']}")
        if "drill_turn_needed" in table: parts.append("drill TL " + show(table["drill_turn_needed"]["top_left"]) + " BR " + show(table["drill_turn_needed"]["bottom_right"]))
        if "ordinary" in table: parts.append(f"ordinary {show(table['ordinary']['all'])} of {table['ordinary']['all']['episodes']}")
        print(f"{name:16s} " + " | ".join(parts))


# ---------------------------------------------------------------- the recorded round and the exact normal spawn

class Adapter:
    """A `.predict` policy as a controller for live_round_diagnostic.rollout."""

    def __init__(self, policy): self.policy = policy

    def bind(self, env):
        if hasattr(self.policy, "bind"): self.policy.bind(env)

    def act(self, observation): return int(self.policy.predict(observation, deterministic=True)[0])


def recorded_checks(controllers, session):
    """Each controller against the recorded spin round (open-loop red input), from the start and from the pre-spin
    state, and against red still / moving left / moving right from the exact normal spawn."""
    case = live.load_case(); spin_round = next(item for item in case["rounds"] if item["round"] == case["spin"]["round"]); a = case["spin"]["first_decision"]
    env = session.env; reference = live.rollout(env, live.Clone(session.model), spin_round["red"], keep={a}); saved = reference["snapshots"][a]; report = {}
    for name, controller in controllers.items():
        entry = {}
        for label, trace in (("recorded_round_from_start", live.rollout(env, controller, spin_round["red"])),
                             ("recorded_round_from_pre_spin_state", live.rollout(env, controller, spin_round["red"][a:], start=saved))):
            verdict = live.passes_case(trace, a)
            entry[label] = {"outcome": live.outcome(trace), **{key: verdict[key] for key in ("no_spin", "reaches_red_after_spin_onset", "no_own_fall", "passes")},
                            "longest_one_way_rotation_deg": verdict["longest_one_way_rotation"]["rotation_deg"],
                            **trace_metrics(trace["poses"], trace["actions"], trace["contact"])}
        for kind in ("still", "left", "right"):
            trace = live.rollout(env, controller, live.red_script(kind))
            entry[f"normal_spawn_red_{kind}"] = {"outcome": live.outcome(trace), **trace_metrics(trace["poses"], trace["actions"], trace["contact"])}
        report[name] = entry
    return report


def live_controllers(session, extra=None):
    from autobot.sim import cli
    controllers = {"reference": live.Clone(session.model), "teacher": live.Teacher(), "reactive": Adapter(PursuitController()),
                   "always_advance": live.Advance(), "always_stop": live.Stop()}
    for name, path in (extra or {}).items(): controllers[name] = live.Clone(cli.load_checkpoint(str(path), device="cpu")[0])
    return controllers


# ---------------------------------------------------------------- the gate

def gate_decision(summary, recorded):
    reactive, teacher = summary["reactive"], summary["teacher"]; fresh = reactive["fresh"]["all"]; taught = teacher["fresh"]["all"]
    idle = lambda c: (c["mean_spinning_s"] or 0) + (c["mean_stalled_s"] or 0)
    failing = all(recorded["reactive"][key]["passes"] for key in ("recorded_round_from_start", "recorded_round_from_pre_spin_state"))
    variants = {"reach_share": fresh["reach_share"], "own_fall_share": round(fresh["own_falls"] / fresh["episodes"], 3),
                "spinning_or_stalled_s": round(idle(fresh), 2), "teacher_spinning_or_stalled_s": round(idle(taught), 2)}
    variants["passes"] = bool(variants["reach_share"] >= GATE["reach_share"] and variants["own_fall_share"] <= GATE["max_fall_share"]
                              and variants["spinning_or_stalled_s"] < GATE["stall_ratio"] * variants["teacher_spinning_or_stalled_s"])
    drill = {pit: {"reactive_wins": reactive["drill_turn_needed"][pit]["wins"], "teacher_wins": teacher["drill_turn_needed"][pit]["wins"],
                   "episodes": teacher["drill_turn_needed"][pit]["episodes"]} for pit in pilot.PIT_NAMES}
    drill_ok = all(entry["reactive_wins"] >= GATE["drill_share"] * entry["teacher_wins"] and entry["reactive_wins"] > 0 for entry in drill.values())
    return {"rule": GATE, "failing_case": {"passes": bool(failing), **{key: recorded["reactive"][key] for key in ("recorded_round_from_start", "recorded_round_from_pre_spin_state")}},
            "fresh_variants": variants, "drill": {"passes": bool(drill_ok), **drill}, "train_challenger": bool(failing and variants["passes"] and drill_ok)}


def baseline(args):
    sets = validation_sets(baseline=True); results = run_policies("baseline", sets, POLICIES, args.workers); gated = first_gated_offset()
    summary = {name: tables(result, gated) for name, result in results.items()}
    session = live.make_session(); recorded = recorded_checks(live_controllers(session), session)
    decision = gate_decision(summary, recorded)
    output = {"purpose": "Gate for the reactive baseline, on the recorded round and on train and validation cases only (no test case).",
              "reactive_baseline": PursuitController.__doc__.strip(), "tables": summary, "recorded_round_and_normal_spawn": recorded, "decision": decision,
              "metrics": {"spinning_s": f"seconds in which the last second held at least {SPIN_DEG:g} degrees of net rotation within 8 cm",
                          "stalled_s": f"seconds in which blue moved under {STALL_M * 100:g} cm in the last second without touching red and without spinning",
                          "wins": "red falls within half a second of contact, and does not also fall against a blue that never moves"}}
    (RUN / "baseline.json").write_text(json.dumps(output, indent=1), encoding="utf-8")
    print_tables(summary)
    for name, entry in recorded.items():
        print(f"{name:16s} " + " | ".join(f"{key.replace('recorded_round_', 'rec ').replace('normal_spawn_red_', '')}: {value['outcome']}, reach {value['reaches_red']}, "
                                          f"spin {value['spinning_s']} stall {value['stalled_s']}" for key, value in entry.items()))
    print(json.dumps({key: (value if not isinstance(value, dict) else {k: v for k, v in value.items() if not isinstance(v, dict) or k in pilot.PIT_NAMES})
                      for key, value in decision.items() if key != "rule"}, indent=1))


def first_challenger():
    selection = json.loads((RUN / "validation.json").read_text(encoding="utf-8"))
    return selection["selected"], {"kind": "checkpoint", "path": str(ROOT / selection["selected_checkpoint"])}


def anywhere(args):
    """V2 question: the first challenger from any geometry, on anywhere train and validation cases."""
    cases = load_anywhere(); sets = {"anywhere": cases["splits"]["train"] + cases["splits"]["validation"]}; _, first = first_challenger()
    results = run_policies("anywhere", sets, {"challenger": first, **POLICIES}, args.workers)
    summary = {name: tables(result, 0) for name, result in results.items()}; idle = lambda c: (c["mean_spinning_s"] or 0) + (c["mean_stalled_s"] or 0)
    mine, base = summary["challenger"]["anywhere"]["all"], summary["reactive"]["anywhere"]["all"]
    fails = mine["reach_share"] < V2["reach_share"] or idle(mine) > V2["max_idle_s"]; baseline_ok = base["reach_share"] >= V2["reach_share"] and idle(base) <= V2["max_idle_s"]
    output = {"rule": V2, "tables": summary, "first_challenger": {"reach_share": mine["reach_share"], "spinning_or_stalled_s": round(idle(mine), 2)},
              "reactive_baseline": {"reach_share": base["reach_share"], "spinning_or_stalled_s": round(idle(base), 2)},
              "train_second_challenger": bool(fails and baseline_ok)}
    (RUN / "anywhere.json").write_text(json.dumps(output, indent=1), encoding="utf-8")
    print_tables(summary); print(json.dumps({key: value for key, value in output.items() if key not in ("rule", "tables")}, indent=1))


# ---------------------------------------------------------------- challenger: demonstrations and fitting

def challenger_directory(seed, version="v1"): return ROOT / (f"runs/recovery-challenger-seed{seed}" if version == "v1" else f"runs/recovery-challenger-{version}-seed{seed}")


def demonstration_plan(version="v1"):
    """Cases the reactive baseline demonstrates on and the student later drives (DAgger). No validation or test case."""
    if version == "v2": return demonstration_plan() + [("anywhere", case, combined.ORDINARY_STEP_CAP) for case in load_anywhere()["splits"]["train"]]
    starts = combined.load_starts()["train"]; cases = load_cases()["splits"]["train"]
    plan = [("ordinary", {"name": f"reactive_demo_{start['name']}", "seed": ORDINARY_DEMO_SEED + index, "start": start}, combined.ORDINARY_STEP_CAP)
            for index, start in enumerate(starts)]
    plan += [("fresh", case, combined.ORDINARY_STEP_CAP) for case in cases]
    return plan


def _collect_chunk(chunk):
    env = pool_tools.make_env(); expert = PursuitController(); rows = []
    for position, source, case in chunk:
        record = run_case(expert, env, case, keep=True)
        rows.append((position, source, case["name"], record["observations"], record["actions"], record["outcome"]))
    return rows


def collect_anywhere(args):
    """Second challenger only: reactive-baseline demonstrations on the anywhere train cases."""
    data = RUN / "data"
    if (data / "anywhere-demonstrations.npz").exists(): raise SystemExit("Anywhere demonstrations already collected; not overwriting.")
    if not json.loads((RUN / "anywhere.json").read_text(encoding="utf-8"))["train_second_challenger"]: raise SystemExit("The V2 rule did not call for a second challenger.")
    tasks = [(position, "anywhere", case) for position, case in enumerate(load_anywhere()["splits"]["train"])]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        rows = sorted((row for chunk in pool.map(_collect_chunk, [tasks[index::args.workers] for index in range(args.workers)]) for row in chunk), key=lambda row: row[0])
    labels = np.concatenate([row[4] for row in rows]); tactics = combined.experiment_tactics()
    summary = {"episodes": len(rows), "decisions": int(len(labels)), "outcomes": dict(Counter(row[5] for row in rows)),
               "label_counts": {tactics[key]: int(value) for key, value in sorted(Counter(labels.tolist()).items())}}
    np.savez_compressed(data / "anywhere-demonstrations.npz", observations=np.concatenate([row[3] for row in rows]), labels=labels.astype(np.int8))
    (data / "anywhere-demonstrations.json").write_text(json.dumps({"teacher": "reactive_recovery.PursuitController", "summary": summary}, indent=1), encoding="utf-8")
    print(json.dumps(summary, indent=1))


def collect(args):
    if args.version == "v2": return collect_anywhere(args)
    data = RUN / "data"
    if (data / "demonstrations.npz").exists(): raise SystemExit("Demonstrations already collected; not overwriting.")
    decision = json.loads((RUN / "baseline.json").read_text(encoding="utf-8"))["decision"]
    if not decision["train_challenger"]: raise SystemExit("The gate did not pass; nothing is collected or trained.")
    plan = demonstration_plan(); tasks = [(position, source, case) for position, (source, case, _) in enumerate(plan)]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        rows = sorted((row for chunk in pool.map(_collect_chunk, [tasks[index::args.workers] for index in range(args.workers)]) for row in chunk), key=lambda row: row[0])
    expert = PursuitController(); original = combined.load_demonstrations(); tactics = combined.experiment_tactics(); summary = {}
    for source in ("ordinary", "fresh"):
        part = [row for row in rows if row[1] == source]; labels = np.concatenate([row[4] for row in part])
        summary[source] = {"episodes": len(part), "decisions": int(len(labels)), "outcomes": dict(Counter(row[5] for row in part)),
                           "label_counts": {tactics[key]: int(value) for key, value in sorted(Counter(labels.tolist()).items())}}
    # The original drill examples are kept exactly as they are (labelled by the old teacher). How often would the
    # reactive baseline have chosen the same tactic on those states? Same question for the old ordinary examples.
    for name in ("drill", "ordinary"):
        observations, labels = original[name]
        summary[f"original_{name}_examples"] = {"decisions": int(len(labels)), "reactive_baseline_agrees": round(float(np.mean(expert.labels(observations) == labels)), 3),
                                                "used_for_fitting": name == "drill"}
    data.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(data / "demonstrations.npz", observations=np.concatenate([row[3] for row in rows]), labels=np.concatenate([row[4] for row in rows]).astype(np.int8),
                        source=np.concatenate([np.full(len(row[4]), ("ordinary", "fresh").index(row[1]), dtype=np.int8) for row in rows]))
    (data / "demonstrations.json").write_text(json.dumps({"teacher": "reactive_recovery.PursuitController (reads only the observation)", "summary": summary,
                                                           "episodes": [{"source": row[1], "case": row[2], "decisions": int(len(row[4])), "outcome": row[5]} for row in rows]}, indent=1),
                                               encoding="utf-8")
    print(json.dumps(summary, indent=1))


def load_pools(version="v1"):
    """{pool: (observations, labels)}: the original drill examples untouched, plus the reactive baseline's demonstrations."""
    if version == "v2":
        extra = np.load(RUN / "data" / "anywhere-demonstrations.npz")
        return {**load_pools(), "anywhere": (extra["observations"], extra["labels"].astype(np.int64))}
    drill = combined.load_demonstrations()["drill"]; data = np.load(RUN / "data" / "demonstrations.npz")
    pools = {"drill": drill}
    for index, source in enumerate(("ordinary", "fresh")):
        pools[source] = (data["observations"][data["source"] == index], data["labels"][data["source"] == index].astype(np.int64))
    return pools


def _train_one(task):
    version, seed = task
    import torch
    torch.set_num_threads(1)
    from autobot.sim.ladder import GreedyActor
    out = challenger_directory(seed, version); out.mkdir(parents=True, exist_ok=True); started = time.perf_counter()
    expert = PursuitController(); model = imitation.build_student(seed, None); untouched = pilot.flat(pilot.critic_modules(model.policy))
    pools = {name: ([observations], [labels]) for name, (observations, labels) in load_pools(version).items()}
    joined = lambda: {name: (np.concatenate(observations), np.concatenate(labels)) for name, (observations, labels) in pools.items()}
    history = [{"stage": "behaviour_cloning", "gradient_steps": combined.BC_STEPS, "fit": combined.fit(model, joined(), combined.BC_STEPS, True, seed)}]
    env = pool_tools.make_env(); ladder = experiment.load_ladder()
    plan = [("drill", {"name": start["name"], "seed": 0, "start": start}, combined.DRILL_STEP_CAP) for start in combined.drill_starts(ladder)] + demonstration_plan(version)
    for iteration in range(1, combined.DAGGER_ITERATIONS + 1):
        driver = GreedyActor(model); added = Counter(); disagreed = Counter(); outcomes = Counter()
        for index, (source, case, cap) in enumerate(plan):
            record = run_case(expert, env, dict(case, seed=DAGGER_SEED + 10_000 * seed + 1000 * iteration + index), driver=driver, cap=cap)
            labels = expert.labels(record["observations"])               # the reactive baseline labels every state the student reached
            pools[source][0].append(record["observations"]); pools[source][1].append(labels)
            added[source] += len(labels); disagreed[source] += int((record["actions"] != labels).sum()); outcomes[f"{source}:{record['outcome']}"] += 1
        history.append({"stage": f"dagger_{iteration}", "gradient_steps": combined.DAGGER_STEPS, "student_driven_decisions_added": dict(added),
                        "student_disagreed_while_driving": {source: round(disagreed[source] / max(added[source], 1), 4) for source in added},
                        "student_driven_episode_outcomes": dict(sorted(outcomes.items())),
                        "fit": combined.fit(model, joined(), combined.DAGGER_STEPS, True, seed + iteration)})
    if not np.array_equal(untouched, pilot.flat(pilot.critic_modules(model.policy))): raise RuntimeError("Fitting changed the value head")
    (out / "effective_config.json").write_text(json.dumps({
        "trainer": "scripts/reactive_recovery.py",
        "training": "SUPERVISED imitation of a simple reactive baseline (turn toward red, advance when aligned), with the original turning-drill "
                    "examples kept. Not trained by PPO. Simulation-only: the observation contains simulator truth.",
        "recipe": "balanced_dagger over the pools " + ", ".join(pools) + " (drill = the original drill examples)", "version": version, "seed": seed, "architecture": "mlp_64_64",
        "batch": combined.BATCH, "learning_rate": combined.LEARNING_RATE, "profile": experiment.PROFILE, "action_profile": experiment.ACTION_PROFILE,
        "previous_action_encoding": experiment.ENCODING, "reward_profile": "not used", "config_path": str(experiment.CONFIG.resolve()),
        "cases": [experiment.relative(CASES_PATH)] + ([experiment.relative(ANYWHERE_PATH)] if version == "v2" else []),
        "demonstrations": experiment.relative(RUN / "data" / "demonstrations.npz"),
        "not_used_for_fitting": "the human recording, validation cases, test cases, and the old teacher's ordinary-start examples"}, indent=1), encoding="utf-8")
    model.save(out / "final_model")
    record = {"version": version, "seed": seed, "path": experiment.relative(out / "final_model.zip"), "history": history,
              "fitted_decisions": {name: int(sum(len(labels) for labels in pool[1])) for name, pool in pools.items()},
              "value_head_untouched": True, "wall_seconds": time.perf_counter() - started}
    (out / "training.json").write_text(json.dumps(record, indent=1), encoding="utf-8")
    return seed, record


def train(args):
    for seed in args.seeds:
        directory = challenger_directory(seed, args.version)
        if directory.exists() and any(directory.iterdir()): raise SystemExit(f"Refusing to overwrite {directory}")
    with ProcessPoolExecutor(max_workers=len(args.seeds)) as pool:
        for seed, record in pool.map(_train_one, [(args.version, seed) for seed in args.seeds]):
            last = record["history"][-1]
            print(f"seed {seed}: {record['wall_seconds']:.0f}s  " + "  ".join(f"{name} agreement {part['agreement']:.3f}" for name, part in last["fit"].items())
                  + f"  disagreed while driving {last['student_disagreed_while_driving']}", flush=True)


def challengers(version="v1"):
    prefix = "challenger" if version == "v1" else f"challenger_{version}"
    return {f"{prefix}_seed{seed}": {"kind": "checkpoint", "path": str(challenger_directory(seed, version) / "final_model.zip")} for seed in SEEDS
            if (challenger_directory(seed, version) / "final_model.zip").is_file()}


def select_seed(summary):
    parts = lambda table: [part for part in ("fresh", "pool", "ordinary", "anywhere") if part in table]
    idle = lambda table: sum((table[part]["all"]["mean_spinning_s"] or 0) + (table[part]["all"]["mean_stalled_s"] or 0) for part in parts(table))
    details = {}
    for name, table in summary.items():
        if not name.startswith("challenger"): continue
        eligible = all(cellv["wins"] >= SELECT["drill_share"] * cellv["episodes"] for cellv in table["drill_turn_needed"].values())
        score = sum(table[part]["all"]["wins"] - table[part]["all"]["own_falls"] for part in parts(table))
        details[name] = {"eligible": bool(eligible), "score": int(score), "spinning_or_stalled_s": round(idle(table), 2)}
    candidates = [name for name in details if details[name]["eligible"]] or list(details)
    return max(sorted(candidates), key=lambda name: (details[name]["score"], -details[name]["spinning_or_stalled_s"])), details


def validate_v2(args):
    """Choose between the first challenger and the second-challenger seeds, on validation cases (now including anywhere validation)."""
    name, first = first_challenger(); sets = validation_sets(baseline=False, anywhere=True); policies = {name: first, **challengers("v2"), **POLICIES}
    results = run_policies("validation", sets, policies, args.workers); gated = first_gated_offset()
    summary = {key: tables(result, gated) for key, result in results.items()}; selected, details = select_seed(summary)
    output = {"purpose": "Choose between the first challenger and the second-challenger seeds on validation cases, before any anywhere test case is run.",
              "rule": V2, "tables": summary, "selection": details, "selected": selected, "selected_checkpoint": experiment.relative(policies[selected]["path"]),
              "training": {key: json.loads((Path(spec["path"]).parent / "training.json").read_text(encoding="utf-8")) for key, spec in challengers("v2").items()}}
    (RUN / "validation-v2.json").write_text(json.dumps(output, indent=1), encoding="utf-8")
    print_tables(summary); print("selection", details, "->", selected)


def validate(args):
    if args.version == "v2": return validate_v2(args)
    sets = validation_sets(baseline=False); policies = {**challengers(), **POLICIES}
    results = run_policies("validation", sets, policies, args.workers); gated = first_gated_offset()
    summary = {name: tables(result, gated) for name, result in results.items()}; selected, details = select_seed(summary)
    output = {"purpose": "Choose the challenger seed on validation cases, before any test case is run.", "rule": SELECT, "tables": summary,
              "selection": details, "selected": selected, "selected_checkpoint": experiment.relative(policies[selected]["path"]),
              "training": {name: json.loads((Path(spec["path"]).parent / "training.json").read_text(encoding="utf-8")) for name, spec in challengers().items()}}
    (RUN / "validation.json").write_text(json.dumps(output, indent=1), encoding="utf-8")
    print_tables(summary); print("selection", details, "->", selected)


# ---------------------------------------------------------------- final comparison

def recommend(candidate, reference, recorded=None):
    idle = lambda c: (c["mean_spinning_s"] or 0) + (c["mean_stalled_s"] or 0); fresh, base = candidate["fresh"]["all"], reference["fresh"]["all"]
    checks = {f"keeps_drill_{pit}": candidate["drill_turn_needed"][pit]["wins"] >= RECOMMEND["drill_share"] * reference["drill_turn_needed"][pit]["wins"]
              and candidate["drill_turn_needed"][pit]["wins"] > 0 for pit in pilot.PIT_NAMES}
    checks["responds_reaches_red_more"] = fresh["reaches_red"] > base["reaches_red"]
    checks["responds_less_spinning_or_stalled"] = idle(fresh) < idle(base)
    checks["moves_before_contact"] = fresh["pre_contact_speed_m_s"] >= base["pre_contact_speed_m_s"]
    for part in ("fresh", "pool", "ordinary"):
        own, other = candidate[part]["all"], reference[part]["all"]
        checks[f"does_not_fall_{part}"] = own["own_falls"] <= max(RECOMMEND["max_fall_share"] * own["episodes"], other["own_falls"])
    wins = lambda table: table["fresh"]["all"]["wins"] + table["pool"]["harder"]["wins"]
    checks["wins_fresh_plus_harder_pool"] = wins(candidate) >= wins(reference)
    if "anywhere" in candidate:                                   # added by V2
        own, other = candidate["anywhere"]["all"], reference["anywhere"]["all"]
        checks["responds_anywhere_reaches_red_more"] = own["reaches_red"] > other["reaches_red"]
        checks["responds_anywhere_less_spinning_or_stalled"] = idle(own) < idle(other)
        checks["does_not_fall_anywhere"] = own["own_falls"] <= max(RECOMMEND["max_fall_share"] * own["episodes"], other["own_falls"])
    if recorded is not None: checks["recovers_from_the_saved_pre_spin_state"] = bool(recorded["recorded_round_from_pre_spin_state"]["passes"])
    return {"checks": {key: bool(value) for key, value in checks.items()}, "recommend_challenger": bool(all(checks.values())),
            "wins_fresh_plus_harder_pool": {"challenger": wins(candidate), "reference": wins(reference)},
            "ordinary_wins_reported_separately": {"challenger": candidate["ordinary"]["all"]["wins"], "reference": reference["ordinary"]["all"]["wins"]}}


def evaluate(args):
    v2 = args.version == "v2"; first_name, first = first_challenger()
    selection = json.loads((RUN / ("validation-v2.json" if v2 else "validation.json")).read_text(encoding="utf-8")); selected = selection["selected"]
    # The first challenger keeps the name its test results were cached under; a second challenger gets its own.
    final = "challenger" if selected == first_name else "challenger_v2"
    policies = {"challenger": first, **({"challenger_v2": challengers("v2")[selected]} if final == "challenger_v2" else {}), **POLICIES}; sets = test_sets(anywhere=v2)
    results = run_policies("test", sets, policies, args.workers); gated = first_gated_offset()
    summary = {name: tables(result, gated) for name, result in results.items()}
    session = live.make_session()
    recorded = recorded_checks(live_controllers(session, {name: spec["path"] for name, spec in policies.items() if name.startswith("challenger")}), session)
    verdict = recommend(summary[final], summary["reference"], recorded[final] if v2 else None)
    if v2: verdict["first_challenger_by_the_same_checks"] = recommend(summary["challenger"], summary["reference"], recorded["challenger"])["checks"]
    policies["challenger_final"] = policies[final]
    output = {"purpose": "Final comparison on test cases. Scripted reds and a fixed recorded input; none of this is a live human result.",
              "fixed": {"config": experiment.relative(experiment.CONFIG), "profile": experiment.PROFILE, "action_profile": experiment.ACTION_PROFILE,
                        "previous_action_encoding": experiment.ENCODING, "evaluation": "greedy", "seeds": list(pilot.FINAL_SEEDS)},
              "selected_challenger": selected, "challenger_checkpoint": experiment.relative(policies["challenger_final"]["path"]),
              "first_challenger_checkpoint": experiment.relative(first["path"]), "reference_checkpoint": experiment.relative(REFERENCE),
              "rule": RECOMMEND, "second_pass": V2 if v2 else None, "tables": summary, "recorded_round_and_normal_spawn": recorded, "verdict": verdict,
              "recommended_checkpoint": experiment.relative(policies["challenger_final"]["path"] if verdict["recommend_challenger"] else REFERENCE)}
    Path(args.output).write_text(json.dumps(output, indent=1), encoding="utf-8")
    print_tables(summary)
    for name, entry in recorded.items():
        print(f"{name:16s} " + " | ".join(f"{key.replace('recorded_round_', 'rec ').replace('normal_spawn_red_', '')}: {value['outcome']}, reach {value['reaches_red']}, "
                                          f"spin {value['spinning_s']} stall {value['stalled_s']}" for key, value in entry.items()))
    print(json.dumps(verdict, indent=1))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("make"); sub.add_parser("make-anywhere")
    for name in ("baseline", "anywhere", "collect", "validate", "evaluate"):
        item = sub.add_parser(name); item.add_argument("--workers", type=int, default=14); item.add_argument("--version", choices=["v1", "v2"], default="v1")
        if name == "evaluate": item.add_argument("--output")
    t = sub.add_parser("train"); t.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS)); t.add_argument("--version", choices=["v1", "v2"], default="v1")
    args = parser.parse_args(); RUN.mkdir(parents=True, exist_ok=True)
    if args.command == "evaluate" and not args.output:
        args.output = str(ROOT / ("artifacts/reactive-recovery-comparison.json" if args.version == "v2" else "artifacts/reactive-recovery-comparison-v1.json"))
    {"make": make, "make-anywhere": make_anywhere, "baseline": baseline, "anywhere": anywhere, "collect": collect, "train": train, "validate": validate,
     "evaluate": evaluate}[args.command](args)


if __name__ == "__main__":
    main()
