"""How often does undirected exploration stumble on a win, and does holding each random tactic longer help?

    python scripts/exploration_timescale_probe.py

A uniformly random policy over the five tactics is run with each choice held for 1, 5, 10 or 25 decisions
(20, 100, 200, 500 ms). PPO has to bootstrap from wins that exploration produces by chance, so these rates are
the raw material available on each start group. Offline; nothing is trained and no checkpoint is involved.
This is a diagnostic of exploration only: it does not test a policy trained with a tactic hold.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ordinary_start_diagnostic as diagnostic  # noqa: E402

ROOT = diagnostic.ROOT
CONFIG = diagnostic.CONFIG
HOLDS = (1, 5, 10, 25)
SEED = 70017
REPEATS = {"near_pit_training": 10, "near_pit_misaligned": 6, "ordinary_training": 25, "frozen_normal": 25}


class RandomHold:
    """Uniform random tactic, re-drawn every `hold` decisions."""
    def __init__(self, hold, seed): self.hold = hold; self.rng = np.random.default_rng(seed); self.count = 0; self.action = 0

    def predict(self, observation, deterministic=True):
        if self.count % self.hold == 0: self.action = int(self.rng.integers(5))
        self.count += 1
        return np.array(self.action), None


def scenarios():
    from autobot.sim.scenarios import load_scenario_file
    curriculum = load_scenario_file(diagnostic.FROZEN / "training_curriculum.json")
    near = [dict(item, group="near_pit_training") for name in ("near_pit_easy", "near_pit_offset") for item in curriculum["groups"][name]]
    misaligned = [dict(case, group="near_pit_misaligned") for case in diagnostic.sweep_scenarios()
                  if case["distance_m"] <= 0.60 and abs(case["heading_offset_deg"]) in (20, 45, 90)]
    return near + misaligned + diagnostic.ordinary_scenarios()


def run(task):
    hold, index, scenario = task
    from autobot.sim.diagnostics import rollout_episode
    episodes = []
    for repeat in range(REPEATS[scenario["group"]]):
        seed = SEED + 1000 * index + repeat
        episode = rollout_episode(RandomHold(hold, seed), CONFIG, scenario, seed=seed, profile="sim_debug",
                                  action_profile="legacy5", previous_action_encoding="onehot_v1")
        episodes.append({key: episode[key] for key in ("scenario", "group", "outcome", "steps", "time_to_first_contact_s",
                                                       "opponent_min_pit_edge_distance_m", "opponent_displacement_m", "switches_per_s")}
                        | {"hold_decisions": hold, "abs_heading_offset_deg": abs(scenario.get("heading_offset_deg", 0))})
    return episodes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--output", default=str(ROOT / "artifacts/exploration-timescale-probe.json"))
    args = parser.parse_args()
    cases = scenarios()
    tasks = [(hold, index, scenario) for hold in HOLDS for index, scenario in enumerate(cases)]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        episodes = [episode for batch in pool.map(run, tasks, chunksize=4) for episode in batch]
    def cell(selected):
        counts = Counter(e["outcome"] for e in selected); wins = [e["steps"] for e in selected if e["outcome"] == "wins"]
        return {"episodes": len(selected), "wins": counts["wins"], "own_falls": counts["own_falls"], "timeouts": counts["timeouts"],
                "win_rate": counts["wins"] / max(len(selected), 1), "own_fall_rate": counts["own_falls"] / max(len(selected), 1),
                "made_contact": sum(e["time_to_first_contact_s"] is not None for e in selected),
                "median_win_decisions": float(np.median(wins)) if wins else None}
    summary = {}
    for hold in HOLDS:
        chosen = [e for e in episodes if e["hold_decisions"] == hold]
        entry = {group: cell([e for e in chosen if e["group"] == group]) for group in REPEATS}
        entry["near_pit_misaligned_by_abs_offset_deg"] = {str(offset): cell([e for e in chosen if e["group"] == "near_pit_misaligned"
                                                                             and e["abs_heading_offset_deg"] == offset]) for offset in (20, 45, 90)}
        summary[f"hold_{hold}_decisions_{hold * 20}ms"] = entry
    output = {"purpose": "chance win rate of a uniformly random five-tactic policy versus how long each random tactic is held",
              "config": str(CONFIG.relative_to(ROOT)), "opponent": "scripted last_year", "seed": SEED, "holds_decisions": HOLDS,
              "repeats_per_scenario": REPEATS, "scope": "exploration only; says nothing about a policy trained with a hold",
              "summary": summary, "episodes": episodes}
    Path(args.output).write_text(json.dumps(output, indent=1), encoding="utf-8")
    print(args.output)
    for name, entry in summary.items():
        print(name, {group: f"{entry[group]['wins']}/{entry[group]['episodes']} (falls {entry[group]['own_falls']})" for group in REPEATS})


if __name__ == "__main__":
    main()
