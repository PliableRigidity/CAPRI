"""Matched evaluation of the discount-factor experiment (gamma 0.99 vs 0.999, both with masked previous action).

    python scripts/gamma_comparison_evaluation.py --seeds 23

Three things are kept apart on purpose:

* training events, counted by curriculum stage and start group with their denominators;
* the frozen suite (2 easy + 4 normal + 4 intermediate starts), evaluated at every periodic checkpoint so
  "never learned" can be told from "learned briefly, then regressed";
* a fresh set of normal-start offsets that was not used for training, in-training selection, or choosing
  a checkpoint here. A checkpoint chosen on the frozen suite is confirmed on these, not on the suite itself.
"""
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import previous_action_mask_evaluation as base  # noqa: E402  (shared scenario/recording helpers)

ROOT = base.ROOT
FROZEN = base.FROZEN
CONFIG = base.CONFIG
PRIMARY_EVAL_SEED = base.PRIMARY_EVAL_SEED
EXTRA_EVAL_SEEDS = base.EXTRA_EVAL_SEEDS
HUMAN_SEED = base.HUMAN_SEED
OUTCOMES = base.OUTCOMES
ARMS = {"gamma_0p99": ("0p99", 0.99), "gamma_0p999": ("0p999", 0.999)}
CONFIRMATION_FILE = ROOT / "config/normal_start_confirmation_scenarios.json"
CONFIRMATION_SEED = 90017
PERIODIC_STEPS = tuple(range(50_000, 1_000_001, 50_000))


def run_directory(arm, seed):
    return ROOT / f"runs/gamma-{ARMS[arm][0]}-masked-seed{seed}-1m"


def confirmation_scenarios():
    """Twelve fresh normal starts: three seeded offsets of each frozen normal start (written once, then reused)."""
    if CONFIRMATION_FILE.is_file():
        return json.loads(CONFIRMATION_FILE.read_text(encoding="utf-8"))
    rng = np.random.default_rng(CONFIRMATION_SEED)
    normal = [item for item in base.scenarios() if item["group"] == "normal"]
    fresh = []
    for item in normal:
        for index in range(3):
            pose = lambda values: [round(float(values[0] + rng.uniform(-0.06, 0.06)), 4),
                                   round(float(values[1] + rng.uniform(-0.06, 0.06)), 4),
                                   round(float(values[2] + rng.uniform(-0.25, 0.25)), 4)]
            fresh.append({"name": f"confirm_{item['name']}_{index}", "group": "normal_confirmation",
                          "derived_from": item["name"], "player": pose(item["player"]), "opponent": pose(item["opponent"])})
    CONFIRMATION_FILE.write_text(json.dumps(fresh, indent=1), encoding="utf-8")
    return fresh


def policy_specifications(seeds):
    specs = {"always_stop": {"kind": "stop", "action_profile": "legacy5"},
             "always_advance": {"kind": "advance", "action_profile": "legacy5"}}
    missing = []
    overnight = FROZEN / "final_model.zip"
    if overnight.is_file(): specs["overnight_final"] = {"kind": "checkpoint", "path": overnight, "action_profile": "legacy5"}
    else: missing.append(str(overnight.relative_to(ROOT)))
    for seed in seeds:
        for arm in ARMS:
            run = run_directory(arm, seed)
            candidates = {f"{steps // 1000}k": run / "checkpoints" / f"rl_model_{steps}_steps.zip" for steps in PERIODIC_STEPS}
            candidates["final"] = run / "final_model.zip"; candidates["best"] = run / "best" / "best_model.zip"
            for which, path in candidates.items():
                if not path.is_file(): missing.append(str(path.relative_to(ROOT))); continue
                specs[f"{arm}_seed{seed}_{which}"] = {
                    "kind": "checkpoint", "path": path, "action_profile": "experimental5_v1", "arm": arm,
                    "training_seed": seed, "checkpoint": which, "expected_encoding": "masked_zero_v1",
                    "expected_gamma": ARMS[arm][1]}
    return specs, missing


def evaluate_policy(task):
    """Worker: every case for one policy, in its own process."""
    name, spec, suite, confirmation, rounds = task
    import torch
    torch.set_num_threads(1)
    from autobot.sim.cli import checkpoint_ppo_settings, load_checkpoint
    from autobot.sim.diagnostics import rollout_episode
    from autobot.sim.env import LEGACY_PREVIOUS_ACTION_ENCODING, ScriptedBaseline, StopBaseline
    gamma = None
    if spec["kind"] == "checkpoint":
        model, encoding = load_checkpoint(spec["path"], spec.get("expected_encoding"), device="cpu")
        gamma = checkpoint_ppo_settings(spec["path"])["gamma"]
        if "expected_gamma" in spec and abs(gamma - spec["expected_gamma"]) > 1e-12:
            raise ValueError(f"{spec['path']} records gamma={gamma}, expected {spec['expected_gamma']}")
    else:
        model = StopBaseline() if spec["kind"] == "stop" else ScriptedBaseline()
        encoding = LEGACY_PREVIOUS_ACTION_ENCODING
    common = dict(profile="sim_debug", action_profile=spec["action_profile"], previous_action_encoding=encoding)
    result = {"action_profile": spec["action_profile"], "previous_action_encoding": encoding, "gamma": gamma,
              **{key: spec[key] for key in ("arm", "training_seed", "checkpoint") if key in spec},
              "path": str(Path(spec["path"]).relative_to(ROOT)) if "path" in spec else None,
              "cases": {}, "recorded_human": {}}
    reference = spec["kind"] != "checkpoint" or spec.get("checkpoint") in (None, "final", "best")
    for label, scenarios in (("suite", suite), ("confirmation", confirmation)):
        for eval_seed in (PRIMARY_EVAL_SEED, *EXTRA_EVAL_SEEDS):
            # Same per-episode seeding as cli.evaluate_model: seed + episode index.
            result["cases"][f"{label}:scripted@{eval_seed}"] = [
                rollout_episode(model, CONFIG, scenario, seed=eval_seed + index, opponent_mode="scripted", **common)
                for index, scenario in enumerate(scenarios)]
    if reference:
        for opponent in base.OPPONENTS[1:]:
            result["cases"][f"suite:{opponent}@{PRIMARY_EVAL_SEED}"] = [
                rollout_episode(model, CONFIG, scenario, seed=PRIMARY_EVAL_SEED + index, opponent_mode=opponent, **common)
                for index, scenario in enumerate(suite)]
    default_spawn = next(item for item in suite if item["name"] == "default_spawn")
    for round_name, recording in rounds.items():
        episode = rollout_episode(model, CONFIG, default_spawn, seed=HUMAN_SEED, opponent_mode="static",
                                  red_commands=recording["red_commands"], **common)
        if name == "overnight_final":
            tactics = base._replay_tactics(model, default_spawn, recording["red_commands"], common)
            episode["recorded_blue_tactic_mismatches"] = sum(a != b for a, b in zip(tactics, recording["blue_tactics"]))
        result["recorded_human"][round_name] = episode
    return name, result


def credit_wins(results):
    """A win counts for the policy only if Always-Stop does not also win the identical case."""
    stop = results["always_stop"]
    for policy in results.values():
        for case, episodes in policy["cases"].items():
            for index, episode in enumerate(episodes):
                if episode["outcome"] != "wins": continue
                episode["contact_before_fall"] = episode["time_to_first_contact_s"] is not None
                episode["matched_stop_outcome"] = stop["cases"][case][index]["outcome"] if case in stop["cases"] else None
                episode["win_credited"] = episode["matched_stop_outcome"] != "wins"
        for round_name, episode in policy["recorded_human"].items():
            if episode["outcome"] != "wins": continue
            episode["matched_stop_outcome"] = stop["recorded_human"][round_name]["outcome"]
            episode["win_credited"] = episode["matched_stop_outcome"] != "wins"


def group_summary(policy, label, group, pooled=True):
    cases = [episodes for case, episodes in policy["cases"].items()
             if case.startswith(f"{label}:scripted@") and (pooled or case.endswith(f"@{PRIMARY_EVAL_SEED}"))]
    episodes = [episode for case in cases for episode in case if episode["group"] == group]
    if not episodes: return None
    summary = base.summarise(episodes)
    classes = Counter(episode["timeout_class"] for episode in episodes if episode["timeout_class"])
    summary["timeout_classes"] = dict(classes)
    summary["mean_opponent_min_pit_edge_distance_m"] = sum(e["opponent_min_pit_edge_distance_m"] for e in episodes) / len(episodes)
    return summary


def training_summary(run):
    metrics = run / "metrics.jsonl"
    if not metrics.is_file(): return None
    rows = [json.loads(line) for line in metrics.read_text(encoding="utf-8").splitlines()]
    episodes = [row for row in rows if row["type"] == "train_episode"]
    frozen = json.loads((run / "effective_config.json").read_text(encoding="utf-8"))
    table = {}
    for row in episodes:
        cell = table.setdefault(row["stage"], {}).setdefault(row["scenario_group"], Counter()); cell[row["outcome"]] += 1
    by_stage = {stage: {group: {"episodes": sum(cell.values()), **{k: cell[k] for k in ("win", "own_fall", "timeout", "draw", "simulation_bug") if cell[k]}}
                        for group, cell in sorted(groups.items())} for stage, groups in table.items()}
    mix = {stage: {group: {"episodes": cell["episodes"], "share": cell["episodes"] / sum(c["episodes"] for c in groups.values())}
                   for group, cell in groups.items()} for stage, groups in by_stage.items()}
    configured = {stage["name"]: {group: weight / sum(stage["group_weights"].values()) for group, weight in stage["group_weights"].items()}
                  for stage in frozen["training_curriculum"]["stages"]}
    ordinary = [row for row in episodes if row["scenario_group"] == "ordinary"]
    buckets = {}
    for row in ordinary:
        bucket = buckets.setdefault(row["timesteps"] // 100_000, Counter()); bucket[row["outcome"]] += 1
        bucket["contact"] += row.get("first_contact_step") is not None
    per_scenario = {}
    for row in ordinary:
        per_scenario.setdefault(row.get("scenario"), Counter())[row["outcome"]] += 1
    mixed_ordinary = [row for row in ordinary if row["stage"] == "mixed"]
    contact_steps = [row["first_contact_step"] for row in mixed_ordinary if row.get("first_contact_step") is not None]
    complete = next((row for row in rows if row["type"] == "run_complete"), None)
    return {
        "run_directory": str(run.relative_to(ROOT)), "seed": frozen["seed"], "gamma": frozen.get("gamma"),
        "ppo_hyperparameters": frozen.get("ppo_hyperparameters"), "previous_action_encoding": frozen.get("previous_action_encoding"),
        "action_profile": frozen["action_profile"], "profile": frozen["profile"], "reward_profile": frozen["reward_profile"],
        "episode_seconds": frozen["config"]["simulation"]["episode_seconds"],
        "ending_timesteps": complete and complete["ending_timesteps"],
        "environment_steps_per_second": complete and complete["environment_steps_per_second"],
        "training_episodes": len(episodes),
        "curriculum_stage_changes": [row for row in rows if row["type"] == "curriculum_stage_change"],
        "events_by_stage_and_start_group": by_stage,
        "scenario_mix_by_stage": {"observed": mix, "configured_weights": configured},
        "ordinary_start_events_per_100k_steps": [
            {"from_timesteps": index * 100_000, "episodes": sum(bucket[k] for k in ("win", "own_fall", "timeout", "draw", "simulation_bug")),
             "win": bucket["win"], "own_fall": bucket["own_fall"], "timeout": bucket["timeout"], "made_contact": bucket["contact"]}
            for index, bucket in sorted(buckets.items())],
        "ordinary_start_events_by_scenario": {name: dict(counter) for name, counter in sorted(per_scenario.items(), key=lambda item: str(item[0]))},
        "ordinary_start_win_lengths_steps": [row["steps"] for row in ordinary if row["outcome"] == "win"],
        "mixed_stage_ordinary_starts": {
            "episodes": len(mixed_ordinary), "made_contact": len(contact_steps),
            "median_first_contact_step": statistics.median(contact_steps) if contact_steps else None,
            **{key: sum(row["outcome"] == key for row in mixed_ordinary) for key in ("win", "own_fall", "timeout")}},
        "in_training_evaluations": [{"timesteps": row["timesteps"], "selection_score": row["selection_score"],
                                     "groups": row["metrics"]["groups"]} for row in rows if row["type"] == "evaluation"]}


def checkpoint_steps(which):
    return 1_000_000 if which == "final" else int(which[:-1]) * 1000


def trajectory_and_selection(results, seeds):
    """Per-checkpoint outcomes, plus one checkpoint per run chosen on the frozen suite only."""
    trajectories, selection = {}, {}
    for seed in seeds:
        for arm in ARMS:
            rows = []
            for which in [f"{steps // 1000}k" for steps in PERIODIC_STEPS]:
                name = f"{arm}_seed{seed}_{which}"
                if name not in results: continue
                policy = results[name]
                row = {"checkpoint": which, "timesteps": checkpoint_steps(which)}
                for group in ("normal", "easy", "intermediate"):
                    s = group_summary(policy, "suite", group)
                    row[group] = {key: s[key] for key in ("episodes", "wins", "credited_wins", "own_falls", "timeouts", "mean_switches_per_s",
                                                         "mean_pre_contact_net_displacement_m", "mean_stationary_time_s",
                                                         "mean_stationary_wall_time_s", "never_reached_red", "timeout_classes")}
                rows.append(row)
            if not rows: continue
            trajectories[f"{arm}_seed{seed}"] = rows
            # Two rules, both applied to the frozen suite only. `net_outcome` is the training objective
            # (+1 opponent fall, -1 own fall); `wins_first` is the rule the in-training selector uses, kept
            # for comparison because it happily picks a checkpoint that mostly falls into pits itself.
            rules = {
                "net_outcome": ("max (credited normal wins - own falls) on the frozen suite (3 eval seeds), then most wins, then latest",
                                lambda row: (row["normal"]["credited_wins"] - row["normal"]["own_falls"], row["normal"]["credited_wins"], row["timesteps"])),
                "wins_first": ("max credited normal wins on the frozen suite (3 eval seeds), then fewest own falls, fewest timeouts, latest",
                               lambda row: (row["normal"]["credited_wins"], -row["normal"]["own_falls"], -row["normal"]["timeouts"], row["timesteps"]))}
            selection[f"{arm}_seed{seed}"] = {}
            for rule, (description, key) in rules.items():
                chosen = max(rows, key=key)
                name = f"{arm}_seed{seed}_{chosen['checkpoint']}"
                selection[f"{arm}_seed{seed}"][rule] = {
                    "rule": description, "selected_checkpoint": chosen["checkpoint"], "policy": name, "path": results[name]["path"],
                    "frozen_suite_normal_starts_used_for_selection": chosen["normal"],
                    "fresh_confirmation_normal_starts": group_summary(results[name], "confirmation", "normal_confirmation")}
    return trajectories, selection


def headline(results, selection, seeds):
    keys = ("episodes", "wins", "credited_wins", "own_falls", "timeouts", "mean_switches_per_s", "mean_pre_contact_net_displacement_m",
            "mean_stationary_time_s", "mean_stationary_wall_time_s", "never_reached_red", "timeout_classes",
            "mean_opponent_min_pit_edge_distance_m")
    pick = lambda summary: None if summary is None else {key: summary[key] for key in keys}
    def block(name):
        policy = results[name]
        normal = [e for e in policy["cases"][f"suite:scripted@{PRIMARY_EVAL_SEED}"] if e["group"] == "normal"]
        constant = {tactic for e in normal for tactic, count in e["action_counts"].items() if count >= 0.99 * e["steps"]}
        single = constant.pop() if len(constant) == 1 and all(max(e["action_counts"].values()) >= 0.99 * e["steps"] for e in normal) else None
        return {"frozen_normal": pick(group_summary(policy, "suite", "normal")),
                "frozen_easy": pick(group_summary(policy, "suite", "easy")),
                "frozen_intermediate": pick(group_summary(policy, "suite", "intermediate")),
                "fresh_confirmation_normal": pick(group_summary(policy, "confirmation", "normal_confirmation")),
                "single_tactic_on_every_frozen_normal_start": single,
                "recorded_human": {round_name: {key: episode.get(key) for key in (
                    "outcome", "win_credited", "matched_stop_outcome", "steps", "switches_per_s", "time_to_first_contact_s",
                    "pre_contact_net_displacement_m", "stationary_time_s", "stationary_wall_time_s")}
                    for round_name, episode in policy["recorded_human"].items()}}
    output = {"references": {name: block(name) for name in ("always_stop", "always_advance", "overnight_final") if name in results}}
    for seed in seeds:
        entry = {}
        for arm in ARMS:
            if f"{arm}_seed{seed}_final" not in results: continue
            entry[arm] = {"final": block(f"{arm}_seed{seed}_final")}
            if f"{arm}_seed{seed}_best" in results: entry[arm]["best_by_in_training_selection"] = block(f"{arm}_seed{seed}_best")
            for rule, chosen in (selection.get(f"{arm}_seed{seed}") or {}).items():
                entry[arm][f"selected_by_{rule}"] = {"checkpoint": chosen["selected_checkpoint"], **block(chosen["policy"])}
        output[f"seed_{seed}"] = entry
    return output


def matched_configuration(seed):
    paths = {arm: run_directory(arm, seed) / "effective_config.json" for arm in ARMS}
    if not all(path.is_file() for path in paths.values()): return None
    configs = {arm: json.loads(path.read_text(encoding="utf-8")) for arm, path in paths.items()}
    a, b = configs["gamma_0p99"], configs["gamma_0p999"]
    differing = sorted(key for key in set(a) | set(b) if a.get(key) != b.get(key))
    ppo = sorted(key for key in a["ppo_hyperparameters"] if a["ppo_hyperparameters"][key] != b["ppo_hyperparameters"][key])
    return {"differing_effective_config_keys": differing, "differing_ppo_hyperparameters": ppo,
            "matched": differing == ["gamma", "ppo_hyperparameters"] and ppo == ["gamma", "rollout_buffer_gamma"]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[23])
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--output", default=str(ROOT / "artifacts/gamma-comparison.json"))
    parser.add_argument("--resummarise", action="store_true", help="recompute summaries from rollouts already saved in --output")
    parser.add_argument("--reuse", action="store_true",
                        help="keep saved rollouts for baselines and periodic checkpoints (immutable files); evaluate only what is new")
    args = parser.parse_args()
    suite = base.scenarios(); confirmation = confirmation_scenarios(); rounds = base.recorded_rounds()
    specs, missing = policy_specifications(args.seeds)
    if args.resummarise:
        results = json.loads(Path(args.output).read_text(encoding="utf-8"))["results"]
        uncovered = sorted(set(specs) - set(results))
        if uncovered: print(f"note: {len(uncovered)} policies on disk are not in the saved rollouts (run without --resummarise to add them)")
    else:
        saved = {}
        if args.reuse and Path(args.output).is_file():
            previous = json.loads(Path(args.output).read_text(encoding="utf-8"))
            # final/best files are rewritten while a run is still training, so those are always re-evaluated.
            if previous["scenarios"] == {"frozen_suite": suite, "fresh_confirmation": confirmation}:
                saved = {name: result for name, result in previous["results"].items()
                         if name in specs and specs[name].get("checkpoint") not in ("final", "best")}
        tasks = [(name, spec, suite, confirmation, rounds) for name, spec in specs.items() if name not in saved]
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            fresh = dict(pool.map(evaluate_policy, tasks))
        results = {name: saved.get(name) or fresh[name] for name in specs}
        print(f"evaluated {len(fresh)} policies, reused {len(saved)}")
    credit_wins(results)
    trajectories, selection = trajectory_and_selection(results, args.seeds)
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    output = {
        "experiment": "discount factor: gamma 0.99 vs 0.999; experimental5_v1, masked_zero_v1, sim_debug, outcome_only, 20 s horizon",
        "git_head": revision, "config": str(CONFIG.relative_to(ROOT)), "training_seeds": args.seeds,
        "eval_seeds": [PRIMARY_EVAL_SEED, *EXTRA_EVAL_SEEDS],
        "definitions": {
            "frozen suite": "2 easy + 4 normal held-out starts from the overnight run, plus 4 intermediate diagnostic starts",
            "fresh confirmation": f"{len(confirmation)} normal starts offset from the frozen ones (seed {CONFIRMATION_SEED}); never used to train or select",
            "credited win": "opponent pit fall where Always-Stop does not also win the identical case",
            "switches_per_s": "tactic changes between consecutive 50 Hz decisions / simulated seconds",
            "pre_contact_net_displacement_m": "straight-line blue travel from spawn to first robot contact (or episode end)",
            "stationary_time_s": "time with under 10 mm net blue travel over the trailing 0.5 s",
            "stationary_wall_time_s": "time touching a wall with a non-stop tactic and under 0.1 mm travel in the step",
            "timeout_classes": "never_reached_red / contact_stall (red moved under 5 cm) / moved_red_no_finish",
            "opponent_min_pit_edge_distance_m": "closest red's centre came to a pit opening (0 = on the edge, negative = over it)",
            "pooled": "frozen-suite and confirmation numbers pool the three evaluation seeds unless stated"},
        "human_recording": {"available": bool(rounds), "file": str(base.RECORDING.relative_to(ROOT)), "seed": HUMAN_SEED,
                            "rounds": {name: {"steps": len(item["red_commands"]), "recorded_end_reason": item["recorded_end_reason"]}
                                       for name, item in rounds.items()},
                            "scope": "A fixed input trace replayed open-loop. Not evidence about live human play."},
        "scenarios": {"frozen_suite": suite, "fresh_confirmation": confirmation},
        "missing_artifacts": missing,
        "matched_configuration": {str(seed): matched_configuration(seed) for seed in args.seeds},
        "training": {f"{arm}_seed{seed}": training_summary(run_directory(arm, seed)) for seed in args.seeds for arm in ARMS},
        "checkpoint_trajectory_frozen_suite": trajectories,
        "checkpoint_selection": selection,
        "headline": headline(results, selection, args.seeds),
        "results": results,
    }
    Path(args.output).write_text(json.dumps(output, indent=1), encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
