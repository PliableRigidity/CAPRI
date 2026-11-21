"""Why are ordinary (normal) starts not being won? Offline diagnostic; trains nothing.

    python scripts/ordinary_start_diagnostic.py --seeds 23

Part 1  ordinary_starts: what each policy does from the four ordinary training starts and the four frozen
        normal starts. Does blue fail to approach, make contact and stall, or move red without finishing?
Part 2  push_sweep: a matched scenario sweep. Red is placed `distance` from a pit, blue 0.18 m behind it on the
        line through the pit, with blue's heading rotated by `heading_offset`. Offset 0 is the geometry of the
        verified positive controls and near-pit curriculum starts, at increasing range.
Part 3  flank_controller: a hand-written privileged controller restricted to the same five tactics. It is a
        feasibility check for the 20 s horizon, not a policy candidate.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import previous_action_mask_evaluation as base  # noqa: E402

ROOT = base.ROOT
FROZEN = base.FROZEN
CONFIG = base.CONFIG
EVAL_SEED = 50017
STANDOFF_M = 0.18
PITS = {"top_left": ([0.0, 1.15], [1.0, -1.0]), "bottom_right": ([1.15, 0.0], [-1.0, 1.0])}
SWEEP_DISTANCES_M = (0.32, 0.45, 0.60, 0.80)
SWEEP_HEADING_OFFSETS_DEG = (0, 10, 20, 30, 45, 60, 90)
SWEEP_REPEATS = 3
ORDINARY_REPEATS = 6


def ordinary_scenarios():
    from autobot.sim.scenarios import load_scenario_file
    curriculum = load_scenario_file(FROZEN / "training_curriculum.json")
    training = [dict(item, group="ordinary_training") for item in curriculum["groups"]["ordinary"]]
    frozen = [dict(item, group="frozen_normal") for item in base.scenarios() if item["group"] == "normal"]
    return training + frozen


def sweep_scenarios():
    cases = []
    for pit_name, (centre, inward) in PITS.items():
        centre = np.asarray(centre, dtype=float); inward = np.asarray(inward, dtype=float) / math.sqrt(2)
        toward_pit = math.atan2(-inward[1], -inward[0])
        for distance in SWEEP_DISTANCES_M:
            red = centre + inward * distance; blue = red + inward * STANDOFF_M
            for offset in SWEEP_HEADING_OFFSETS_DEG:
                for sign in ((1,) if offset == 0 else (1, -1)):
                    yaw = toward_pit + sign * math.radians(offset)
                    cases.append({"name": f"sweep_{pit_name}_d{distance:.2f}_h{sign * offset:+d}", "group": "push_sweep",
                                  "pit": pit_name, "distance_m": distance, "heading_offset_deg": sign * offset,
                                  # Red faces blue, as it does once the scripted opponent has turned to chase.
                                  "player": [float(blue[0]), float(blue[1]), yaw],
                                  "opponent": [float(red[0]), float(red[1]), toward_pit + math.pi]})
    return cases


class FlankController:
    """Privileged feasibility probe: get red between blue and a pit, face the pit through red, then advance.

    Uses only the five existing tactics at 50 Hz and reads simulator truth, so a win shows that start is
    winnable inside the 20 s horizon with this action set. It is crude and unoptimised (a more elaborate
    second attempt did worse), so a timeout says nothing. Not deployable; never trained on or used as reward.
    """
    ALIGN_RAD = math.radians(12)      # advance when heading error is inside this
    BEHIND_M = 0.20                   # stand-off behind red on the pit line
    ARRIVE_M = 0.06
    CLEAR_M = 0.19                    # keep this far from red while circling

    def __init__(self): self.env = None

    def bind(self, env): self.env = env

    def predict(self, observation, deterministic=True):
        sim = self.env.sim; blue = sim.truth(0); red = sim.truth(1)
        b = blue["position_m"][:2]; r = red["position_m"][:2]; yaw = blue["yaw_rad"]
        arena = sim.cfg["arena"]; pits = np.asarray(arena["pit_centres_m"], dtype=float)
        pit = pits[int(np.argmin(np.linalg.norm(pits - r, axis=1)))]
        push = (pit - r) / max(np.linalg.norm(pit - r), 1e-9)
        behind = r - push * self.BEHIND_M
        margin = 0.09
        behind = np.clip(behind, margin, [arena["width_m"] - margin, arena["height_m"] - margin])
        to_red = r - b; lined_up = float(np.dot(to_red / max(np.linalg.norm(to_red), 1e-9), push))
        if np.linalg.norm(behind - b) < self.ARRIVE_M or (lined_up > math.cos(math.radians(15)) and np.linalg.norm(to_red) < 0.30):
            target = b + push                                   # in position: face the pit through red and drive
        else:
            target = behind
            # Do not drive through red on the way: go round it on the side that is nearer to the goal.
            path = target - b; length = max(np.linalg.norm(path), 1e-9); along = float(np.dot(r - b, path / length))
            closest = b + path / length * min(max(along, 0.0), length)
            if 0.0 < along < length and np.linalg.norm(r - closest) < self.CLEAR_M:
                normal = np.array([-push[1], push[0]])
                side = normal if np.dot(b - r, normal) >= 0 else -normal
                target = np.clip(r + side * (self.CLEAR_M + 0.08), margin, [arena["width_m"] - margin, arena["height_m"] - margin])
        desired = math.atan2(target[1] - b[1], target[0] - b[0])
        error = math.atan2(math.sin(desired - yaw), math.cos(desired - yaw))
        if abs(error) <= self.ALIGN_RAD: return np.array(1), None          # advance
        return np.array(3 if error > 0 else 4), None                         # turn_left / turn_right


def load_policy(spec):
    from autobot.sim.cli import load_checkpoint
    from autobot.sim.env import LEGACY_PREVIOUS_ACTION_ENCODING, ScriptedBaseline, StopBaseline
    if spec["kind"] == "checkpoint":
        model, encoding = load_checkpoint(spec["path"], device="cpu")
        return model, encoding
    model = {"stop": StopBaseline, "advance": ScriptedBaseline, "flank": FlankController}[spec["kind"]]()
    return model, LEGACY_PREVIOUS_ACTION_ENCODING


def rollout(model, scenario, seed, common, **options):
    """rollout_episode, plus binding for the privileged controller (which needs the environment handle)."""
    from autobot.sim.diagnostics import rollout_episode
    bind = model.bind if isinstance(model, FlankController) else None
    return rollout_episode(model, CONFIG, scenario, seed=seed, on_env=bind, **common, **options)


def evaluate(task):
    name, spec, ordinary, sweep = task
    import torch
    torch.set_num_threads(1)
    model, encoding = load_policy(spec)
    common = dict(profile="sim_debug", action_profile=spec["action_profile"], previous_action_encoding=encoding)
    result = {"kind": spec["kind"], "path": str(Path(spec["path"]).relative_to(ROOT)) if "path" in spec else None,
              "ordinary_starts": [], "ordinary_starts_sampled": [], "push_sweep": []}
    for index, scenario in enumerate(ordinary):
        for repeat in range(ORDINARY_REPEATS):
            episode = rollout(model, scenario, EVAL_SEED + 100 * index + repeat, common, keep_trace=repeat == 0)
            result["ordinary_starts"].append(dict(episode, repeat=repeat))
            if spec["kind"] == "checkpoint":
                # Sampled tactics, as during training (the greedy policy is what gets evaluated, not what explores).
                sampled = rollout(model, scenario, EVAL_SEED + 100 * index + repeat, common, deterministic=False)
                result["ordinary_starts_sampled"].append(dict(sampled, repeat=repeat))
    if spec.get("sweep", True):
        for index, scenario in enumerate(sweep):
            for repeat in range(SWEEP_REPEATS):
                episode = rollout(model, scenario, EVAL_SEED + 10_000 + 10 * index + repeat, common)
                result["push_sweep"].append(dict(episode, repeat=repeat, pit=scenario["pit"], distance_m=scenario["distance_m"],
                                                 heading_offset_deg=scenario["heading_offset_deg"]))
    return name, result


def outcome_counts(episodes):
    counts = Counter(episode["outcome"] for episode in episodes)
    classes = Counter(episode["timeout_class"] for episode in episodes if episode["timeout_class"])
    wins = [episode["simulated_seconds"] for episode in episodes if episode["outcome"] == "wins"]
    return {"episodes": len(episodes), "wins": counts["wins"], "own_falls": counts["own_falls"], "timeouts": counts["timeouts"],
            "timeout_classes": dict(classes), "made_contact": sum(e["time_to_first_contact_s"] is not None for e in episodes),
            "median_time_to_win_s": float(np.median(wins)) if wins else None,
            "max_time_to_win_s": float(max(wins)) if wins else None,
            "mean_opponent_min_pit_edge_distance_m": float(np.mean([e["opponent_min_pit_edge_distance_m"] for e in episodes])),
            "mean_opponent_displacement_m": float(np.mean([e["opponent_displacement_m"] for e in episodes]))}


def summarise(results):
    summary = {}
    for name, result in results.items():
        entry = {"ordinary_starts": {}, "ordinary_starts_sampled": {}, "push_sweep": {}}
        for key in ("ordinary_starts", "ordinary_starts_sampled"):
            for group in ("ordinary_training", "frozen_normal"):
                episodes = [e for e in result[key] if e["group"] == group]
                if episodes: entry[key][group] = outcome_counts(episodes)
            by_scenario = {}
            for episode in result[key]: by_scenario.setdefault(episode["scenario"], []).append(episode)
            entry[key]["by_scenario"] = {scenario: outcome_counts(episodes) for scenario, episodes in by_scenario.items()}
        sweep = result["push_sweep"]
        if sweep:
            entry["push_sweep"]["wins_by_abs_heading_offset_deg"] = {
                str(offset): {"wins": sum(e["outcome"] == "wins" for e in sweep if abs(e["heading_offset_deg"]) == offset),
                              "episodes": sum(abs(e["heading_offset_deg"]) == offset for e in sweep)} for offset in SWEEP_HEADING_OFFSETS_DEG}
            entry["push_sweep"]["wins_by_distance_m"] = {
                f"{distance:.2f}": {"wins": sum(e["outcome"] == "wins" for e in sweep if e["distance_m"] == distance),
                                    "episodes": sum(e["distance_m"] == distance for e in sweep)} for distance in SWEEP_DISTANCES_M}
            entry["push_sweep"]["wins_by_distance_and_abs_heading_offset"] = {
                f"{distance:.2f}": {str(offset): f"{sum(e['outcome'] == 'wins' for e in sweep if e['distance_m'] == distance and abs(e['heading_offset_deg']) == offset)}"
                                                 f"/{sum(e['distance_m'] == distance and abs(e['heading_offset_deg']) == offset for e in sweep)}"
                                    for offset in SWEEP_HEADING_OFFSETS_DEG} for distance in SWEEP_DISTANCES_M}
            # Cells relative to config/training_curriculum_alignment_v1.json (see scripts/make_alignment_curriculum.py).
            cells = {"aligned_0_10deg": lambda e: abs(e["heading_offset_deg"]) <= 10,
                     "misaligned_in_alignment_v1_training": lambda e: e["distance_m"] <= 0.60 and abs(e["heading_offset_deg"]) in (20, 45, 90),
                     "misaligned_heldout_offsets_30_60deg": lambda e: e["distance_m"] <= 0.60 and abs(e["heading_offset_deg"]) in (30, 60),
                     "misaligned_heldout_range_0p80m": lambda e: e["distance_m"] == 0.80 and abs(e["heading_offset_deg"]) >= 20}
            entry["push_sweep"]["wins_by_cell_group"] = {
                label: {"wins": sum(e["outcome"] == "wins" for e in sweep if member(e)), "own_falls": sum(e["outcome"] == "own_falls" for e in sweep if member(e)),
                        "episodes": sum(1 for e in sweep if member(e))} for label, member in cells.items()}
            entry["push_sweep"]["own_falls"] = sum(e["outcome"] == "own_falls" for e in sweep)
            entry["push_sweep"]["episodes"] = len(sweep)
        summary[name] = entry
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="*", default=[23])
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--output", default=str(ROOT / "artifacts/ordinary-start-diagnostic.json"))
    parser.add_argument("--reuse-from", help="earlier output of this script; rollouts for the same checkpoint files are reused")
    parser.add_argument("--extra", action="append", default=[], metavar="NAME=CHECKPOINT",
                        help="additional checkpoint to include (action profile and encoding are read from the file)")
    args = parser.parse_args()
    specs = {"always_stop": {"kind": "stop", "action_profile": "legacy5"},
             "always_advance": {"kind": "advance", "action_profile": "legacy5"},
             "flank_controller_privileged": {"kind": "flank", "action_profile": "legacy5"}}
    candidates = {"overnight_final": (FROZEN / "final_model.zip", "legacy5")}
    for seed in args.seeds:
        for tag in ("0p99", "0p999"):
            candidates[f"gamma_{tag}_seed{seed}_final"] = (ROOT / f"runs/gamma-{tag}-masked-seed{seed}-1m/final_model.zip", "experimental5_v1")
    from autobot.sim.cli import checkpoint_observation_metadata
    for item in args.extra:
        name, path = item.split("=", 1); path = (ROOT / path) if not Path(path).is_absolute() else Path(path)
        metadata = checkpoint_observation_metadata(path) if path.is_file() else None
        candidates[name] = (path, (metadata or {}).get("action_profile", "legacy5"))
    missing = []
    for name, (path, profile) in candidates.items():
        if path.is_file(): specs[name] = {"kind": "checkpoint", "path": path, "action_profile": profile}
        else: missing.append(str(path.relative_to(ROOT)))
    ordinary = ordinary_scenarios(); sweep = sweep_scenarios()
    saved = {}
    if args.reuse_from and Path(args.reuse_from).is_file():
        previous = json.loads(Path(args.reuse_from).read_text(encoding="utf-8"))
        if previous["scenarios"]["ordinary"] == ordinary and previous["eval_seed"] == EVAL_SEED:
            relative = lambda spec: str(Path(spec["path"]).relative_to(ROOT)) if "path" in spec else None
            saved = {name: result for name, result in previous["results"].items()
                     if name in specs and result["kind"] == specs[name]["kind"] and result["path"] == relative(specs[name])}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        fresh = dict(pool.map(evaluate, [(name, spec, ordinary, sweep) for name, spec in specs.items() if name not in saved]))
    results = {name: saved.get(name) or fresh[name] for name in specs}
    print(f"evaluated {len(fresh)} policies, reused {len(saved)}")
    output = {
        "purpose": "Diagnose why ordinary starts are not won. Offline; nothing is trained or tuned.",
        "config": str(CONFIG.relative_to(ROOT)), "opponent": "scripted last_year", "eval_seed": EVAL_SEED,
        "definitions": {
            "ordinary_training": "the four `ordinary` starts of the frozen training curriculum",
            "frozen_normal": "the four held-out normal starts of the frozen evaluation suite",
            "ordinary_starts_sampled": "checkpoints only: tactics sampled from the policy, as in training",
            "push_sweep": f"red `distance` from a pit centre on the diagonal, facing blue; blue {STANDOFF_M} m behind red on the pit line, "
                          f"heading rotated by `heading_offset`; {SWEEP_REPEATS} repeats per case",
            "flank_controller_privileged": "hand-written, reads simulator truth, five tactics only; a feasibility check, not a policy candidate",
            "timeout_classes": "never_reached_red / contact_stall (red moved under 5 cm) / moved_red_no_finish",
            "opponent_min_pit_edge_distance_m": "closest red's centre came to a pit opening (negative = over it)"},
        "scenarios": {"ordinary": ordinary, "push_sweep_cases": len(sweep),
                      "push_sweep_distances_m": SWEEP_DISTANCES_M, "push_sweep_heading_offsets_deg": SWEEP_HEADING_OFFSETS_DEG},
        "missing_artifacts": missing, "summary": summarise(results), "results": results,
    }
    Path(args.output).write_text(json.dumps(output, indent=1), encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
