"""Matched evaluation of the previous-action feedback experiment (one-hot vs masked-to-zero).

Every policy is run on identical cases: the frozen overnight scenario suite (2 easy + 4 normal),
the four intermediate diagnostic starts, and the recorded human red inputs replayed open-loop.
Normal starts are the headline; easy and intermediate starts are diagnostics only.

    python scripts/previous_action_mask_evaluation.py --seeds 17 23 31
"""
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FROZEN = ROOT / "runs/ppo-curriculum-overnight"
CONFIG = FROZEN / "effective_sim.yaml"
RECORDING = ROOT / "artifacts/human-vs-final-default.jsonl"
PRIMARY_EVAL_SEED = 20017          # same cases as artifacts/action-space-comparison.json
EXTRA_EVAL_SEEDS = (30017, 40017)  # scripted opponent only: IMU-noise robustness of the outcome counts
HUMAN_SEED = 10007                 # seed the human round was recorded with
OPPONENTS = ("scripted", "static", "faster_follow", "stop_and_go", "direction_changes")
OUTCOMES = ("wins", "own_falls", "draws", "timeouts", "simulation_bugs")
ARMS = {"onehot": "onehot_v1", "masked": "masked_zero_v1"}
CHECKPOINTS = ("final", "best", "25k", "50k", "75k")


def scenarios():
    from autobot.sim.scenarios import load_scenario_file
    original = load_scenario_file(FROZEN / "evaluation_scenarios.json")
    diagnostic = load_scenario_file(ROOT / "config/diagnostic_scenarios.json")
    return original + [dict(item, group="intermediate") for item in diagnostic if item["group"].startswith("intermediate")]


def recorded_rounds():
    """Recorded red wheel commands, one list per round (the log restarts time at each reset)."""
    if not RECORDING.is_file(): return {}
    rows = [json.loads(line) for line in RECORDING.read_text(encoding="utf-8").splitlines()]
    rows = [row for row in rows if row.get("type") == "step"]
    rounds, current = [], [rows[0]]
    for previous, row in zip(rows, rows[1:]):
        if row["time_s"] < previous["time_s"]: rounds.append(current); current = []
        current.append(row)
    rounds.append(current)
    return {f"round_{index + 1}": {"red_commands": [row["red_command"] for row in rows_],
                                   "blue_tactics": [row["blue_tactic"] for row in rows_],
                                   "recorded_end_reason": rows_[-1]["end_reason"]}
            for index, rows_ in enumerate(rounds)}


def checkpoint_path(run, which):
    if which == "final": return run / "final_model.zip"
    if which == "best": return run / "best" / "best_model.zip"
    return run / "checkpoints" / f"rl_model_{int(which[:-1]) * 1000}_steps.zip"


def policy_specifications(seeds):
    specs = {"always_stop": {"kind": "stop", "action_profile": "legacy5"},
             "always_advance": {"kind": "advance", "action_profile": "legacy5"},
             "overnight_final": {"kind": "checkpoint", "path": FROZEN / "final_model.zip", "action_profile": "legacy5"}}
    missing = []
    for seed in seeds:
        for arm, encoding in ARMS.items():
            run = ROOT / f"runs/prevact-{arm}-seed{seed}-100k"
            for which in CHECKPOINTS:
                path = checkpoint_path(run, which)
                name = f"{arm}_seed{seed}_{which}"
                if not path.is_file(): missing.append(str(path.relative_to(ROOT))); continue
                specs[name] = {"kind": "checkpoint", "path": path, "action_profile": "experimental5_v1", "arm": arm,
                               "training_seed": seed, "checkpoint": which, "expected_encoding": encoding}
    if not specs["overnight_final"]["path"].is_file():
        missing.append("runs/ppo-curriculum-overnight/final_model.zip"); del specs["overnight_final"]
    return specs, missing


def evaluate_policy(task):
    """Worker: every case for one policy. Runs in its own process."""
    name, spec, suite, rounds, full = task
    from autobot.sim.cli import load_checkpoint
    from autobot.sim.diagnostics import rollout_episode
    from autobot.sim.env import LEGACY_PREVIOUS_ACTION_ENCODING, ScriptedBaseline, StopBaseline
    if spec["kind"] == "checkpoint":
        # The encoding is restored from the checkpoint itself and must match the arm label.
        model, encoding = load_checkpoint(spec["path"], spec.get("expected_encoding"), device="cpu")
    else:
        model = StopBaseline() if spec["kind"] == "stop" else ScriptedBaseline()
        encoding = LEGACY_PREVIOUS_ACTION_ENCODING
    common = dict(profile="sim_debug", action_profile=spec["action_profile"], previous_action_encoding=encoding)
    default_spawn = next(item for item in suite if item["name"] == "default_spawn")
    result = {"action_profile": spec["action_profile"], "previous_action_encoding": encoding,
              **{key: spec[key] for key in ("arm", "training_seed", "checkpoint") if key in spec},
              "path": str(Path(spec["path"]).relative_to(ROOT)) if "path" in spec else None,
              "scripted_cases": {}, "recorded_human": {}}
    cases = [("scripted", PRIMARY_EVAL_SEED)]
    if full: cases += [("scripted", seed) for seed in EXTRA_EVAL_SEEDS] + [(mode, PRIMARY_EVAL_SEED) for mode in OPPONENTS[1:]]
    for opponent, eval_seed in cases:
        # Same per-episode seeding as cli.evaluate_model: seed + episode index.
        episodes = [rollout_episode(model, CONFIG, scenario, seed=eval_seed + index, opponent_mode=opponent, **common)
                    for index, scenario in enumerate(suite)]
        result["scripted_cases"][f"{opponent}@{eval_seed}"] = episodes
    for round_name, recording in rounds.items():
        episode = rollout_episode(model, CONFIG, default_spawn, seed=HUMAN_SEED, opponent_mode="static",
                                  red_commands=recording["red_commands"], **common)
        if name == "overnight_final":
            # This checkpoint drove blue when the round was recorded: it must reproduce exactly.
            env_check = _replay_tactics(model, default_spawn, recording["red_commands"], common)
            episode["recorded_blue_tactic_mismatches"] = sum(a != b for a, b in zip(env_check, recording["blue_tactics"]))
        result["recorded_human"][round_name] = episode
    return name, result


def _replay_tactics(model, scenario, commands, common):
    from autobot.sim.env import AutoBotEnv
    env = AutoBotEnv(str(CONFIG), common["profile"], HUMAN_SEED, opponent_model="last_year", opponent_mode="static",
                     action_profile=common["action_profile"], previous_action_encoding=common["previous_action_encoding"])
    obs, _ = env.reset(seed=HUMAN_SEED, options={"scenario": scenario}); tactics = []
    for command in commands:
        action = int(model.predict(obs, deterministic=True)[0]); tactics.append(env.tactics[action])
        obs, _, terminated, truncated, _ = env.step(action, opponent_drive_override=tuple(command))
        if terminated or truncated: break
    return tactics


def summarise(episodes):
    """Counts plus per-episode means for one group of episodes."""
    mean = lambda key: sum(episode[key] for episode in episodes) / len(episodes)
    outcome = Counter(episode["outcome"] for episode in episodes)
    return {"episodes": len(episodes), **{key: outcome[key] for key in OUTCOMES},
            "credited_wins": sum(episode.get("win_credited", False) for episode in episodes),
            "mean_switches_per_s": mean("switches_per_s"),
            "mean_period_2_cycle_fraction": mean("period_2_cycle_fraction"),
            "mean_pre_contact_net_displacement_m": mean("pre_contact_net_displacement_m"),
            "mean_stationary_time_s": mean("stationary_time_s"),
            "mean_pre_contact_stationary_time_s": mean("pre_contact_stationary_time_s"),
            "mean_stationary_wall_time_s": mean("stationary_wall_time_s"),
            "never_reached_red": sum(episode["time_to_first_contact_s"] is None for episode in episodes),
            "mean_greedy_probability": (sum(episode["decision_confidence"]["mean_greedy_probability"] for episode in episodes)
                                        / len(episodes)) if "decision_confidence" in episodes[0] else None,
            "mean_fraction_near_tied_decisions": (sum(
                episode["decision_confidence"]["fraction_of_decisions_with_top2_margin_below_0.10"] for episode in episodes)
                / len(episodes)) if "decision_confidence" in episodes[0] else None}


def credit_wins(results):
    """A win only counts for the learned policy if doing nothing does not also win that exact case."""
    stop = results.get("always_stop", {}).get("scripted_cases", {})
    for policy in results.values():
        for case, episodes in policy["scripted_cases"].items():
            for index, episode in enumerate(episodes):
                if episode["outcome"] != "wins": continue
                stop_outcome = stop[case][index]["outcome"] if case in stop else None
                episode["contact_before_fall"] = episode["time_to_first_contact_s"] is not None
                episode["matched_stop_outcome"] = stop_outcome
                episode["win_credited"] = stop_outcome != "wins"


def training_summary(run):
    metrics = run / "metrics.jsonl"
    if not metrics.is_file(): return None
    rows = [json.loads(line) for line in metrics.read_text(encoding="utf-8").splitlines()]
    episodes = [row for row in rows if row["type"] == "train_episode"]
    by_stage = Counter(f"{row['stage']}:{row['outcome']}" for row in episodes)
    by_group = Counter(f"{row['scenario_group']}:{row['outcome']}" for row in episodes)
    complete = next((row for row in rows if row["type"] == "run_complete"), None)
    ordinary_win_steps = [sum(row["actions"].values()) for row in episodes
                          if row["scenario_group"] == "ordinary" and row["outcome"] == "win"]
    frozen = json.loads((run / "effective_config.json").read_text(encoding="utf-8"))
    return {"run_directory": str(run.relative_to(ROOT)), "previous_action_encoding": frozen.get("previous_action_encoding"),
            "seed": frozen["seed"], "profile": frozen["profile"], "action_profile": frozen["action_profile"],
            "episode_seconds": frozen["config"]["simulation"]["episode_seconds"],
            "ending_timesteps": complete and complete["ending_timesteps"], "training_episodes": len(episodes),
            "training_outcomes_by_stage": dict(sorted(by_stage.items())),
            "training_outcomes_by_scenario_group": dict(sorted(by_group.items())),
            "ordinary_start_training_win_lengths_steps": ordinary_win_steps,
            "in_training_evaluations": [{"timesteps": row["timesteps"], "selection_score": row["selection_score"],
                                         "groups": row["metrics"]["groups"]} for row in rows if row["type"] == "evaluation"]}


def matched_configuration(seed):
    """Everything except the encoding (and the fields derived from it) must be identical between arms."""
    configs = {arm: json.loads((ROOT / f"runs/prevact-{arm}-seed{seed}-100k/effective_config.json").read_text(encoding="utf-8"))
               for arm in ARMS}
    differing = sorted(key for key in set(configs["onehot"]) | set(configs["masked"])
                       if configs["onehot"].get(key) != configs["masked"].get(key))
    return {"differing_effective_config_keys": differing,
            "matched": differing == ["observation_fields", "previous_action_encoding"]}


def headline(results, summary, seeds):
    """Per-seed final-checkpoint comparison on normal starts, so no average can hide a regression."""
    primary = f"scripted@{PRIMARY_EVAL_SEED}"
    keys = ("mean_switches_per_s", "mean_pre_contact_net_displacement_m", "mean_stationary_time_s",
            "mean_stationary_wall_time_s", "wins", "own_falls", "timeouts")
    normal = lambda name: [episode for episode in results[name]["scripted_cases"][primary] if episode["group"] == "normal"]
    advance = [episode["action_counts"] for episode in normal("always_advance")]
    output = {}
    for seed in seeds:
        entry = {}
        for arm in ARMS:
            name = f"{arm}_seed{seed}_final"
            if name not in results: continue
            episodes = normal(name)
            tactics = {tactic for episode in episodes for tactic, count in episode["action_counts"].items()
                       if count >= 0.99 * episode["steps"]}
            switch_rates = [summary[f"{arm}_seed{seed}_{which}"]["scripted_opponent_primary_eval_seed"]["normal"]["mean_switches_per_s"]
                            for which in ("25k", "50k", "75k", "final") if f"{arm}_seed{seed}_{which}" in summary]
            entry[arm] = {
                "final_normal_starts_primary_eval_seed": {key: summary[name]["scripted_opponent_primary_eval_seed"]["normal"][key] for key in keys},
                "final_normal_starts_pooled_3_eval_seeds": {key: summary[name]["normal_starts_pooled_over_eval_seeds"][key] for key in keys},
                "single_tactic_on_every_normal_start": tactics.pop() if len(tactics) == 1 and all(
                    max(episode["action_counts"].values()) >= 0.99 * episode["steps"] for episode in episodes) else None,
                "action_counts_identical_to_always_advance": [episode["action_counts"] for episode in episodes] == advance,
                "mean_switches_per_s_at_25k_50k_75k_100k": switch_rates,
                "recorded_human_switches_per_s": {round_name: episode["switches_per_s"]
                                                  for round_name, episode in results[name]["recorded_human"].items()}}
        if len(entry) == 2:
            entry["masked_minus_onehot_final"] = {key: entry["masked"]["final_normal_starts_primary_eval_seed"][key]
                                                  - entry["onehot"]["final_normal_starts_primary_eval_seed"][key] for key in keys}
        output[str(seed)] = entry
    return output


def overnight_ordinary_start_training():
    """Context for the next test: how ordinary-start wins evolved in the 5.25M-step overnight run."""
    metrics = FROZEN / "metrics.jsonl"
    if not metrics.is_file(): return None
    buckets = {}; lengths = []
    for line in metrics.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row.get("type") != "train_episode" or row.get("scenario_group") != "ordinary": continue
        bucket = buckets.setdefault(row["timesteps"] // 500000, Counter()); bucket[row["outcome"]] += 1
        if row["outcome"] == "win": lengths.append(sum(row["actions"].values()))
    lengths.sort()
    median = statistics.median(lengths) if lengths else None
    return {"ordinary_episodes": sum(sum(bucket.values()) for bucket in buckets.values()), "ordinary_wins": len(lengths),
            "ordinary_win_lengths_steps": lengths, "median_win_length_steps": median, "ppo_gamma": 0.99,
            "gamma_to_the_median_win_length": None if median is None else 0.99 ** median,
            "per_500k_steps": [{"from_timesteps": index * 500000, "episodes": sum(bucket.values()), **dict(bucket)}
                               for index, bucket in sorted(buckets.items())]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[17])
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--output", default=str(ROOT / "artifacts/previous-action-mask-comparison.json"))
    parser.add_argument("--resummarise", action="store_true",
                        help="recompute the summaries from the rollouts already saved in --output (no new rollouts)")
    args = parser.parse_args()
    suite = scenarios(); rounds = recorded_rounds()
    specs, missing = policy_specifications(args.seeds)
    if args.resummarise:
        results = json.loads(Path(args.output).read_text(encoding="utf-8"))["results"]
        assert set(results) == set(specs), "saved rollouts do not cover the requested policies"
    else:
        # Intermediate checkpoints only need the headline cases; final/best and references get everything.
        tasks = [(name, spec, suite, rounds, spec.get("checkpoint", "final") in ("final", "best")) for name, spec in specs.items()]
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            results = dict(pool.map(evaluate_policy, tasks))
    credit_wins(results)
    primary = f"scripted@{PRIMARY_EVAL_SEED}"
    summary = {}
    for name, policy in results.items():
        by_group = {}
        for group in ("normal", "easy", "intermediate"):
            by_group[group] = summarise([episode for episode in policy["scripted_cases"][primary] if episode["group"] == group])
        pooled = [episode for case, episodes in policy["scripted_cases"].items() if case.startswith("scripted@")
                  for episode in episodes if episode["group"] == "normal"]
        summary[name] = {"scripted_opponent_primary_eval_seed": by_group,
                         "normal_starts_pooled_over_eval_seeds": summarise(pooled) if len(pooled) > 4 else None,
                         "recorded_human": {round_name: {key: episode[key] for key in (
                             "outcome", "steps", "switches_per_s", "period_2_cycle_fraction", "time_to_first_contact_s",
                             "pre_contact_net_displacement_m", "stationary_time_s", "pre_contact_stationary_time_s",
                             "stationary_wall_time_s")} for round_name, episode in policy["recorded_human"].items()}}
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    output = {
        "experiment": "previous-action feedback: onehot_v1 vs masked_zero_v1, fresh five-action PPO, matched seeds",
        "git_head": revision, "config": str(CONFIG.relative_to(ROOT)),
        "training_seeds": args.seeds, "primary_eval_seed": PRIMARY_EVAL_SEED, "extra_eval_seeds": list(EXTRA_EVAL_SEEDS),
        "human_recording": {"available": bool(rounds), "file": str(RECORDING.relative_to(ROOT)), "seed": HUMAN_SEED,
                            "rounds": {name: {"steps": len(item["red_commands"]), "recorded_end_reason": item["recorded_end_reason"]}
                                       for name, item in rounds.items()},
                            "scope": "Recorded red inputs replayed open-loop against each policy. Not a claim about live "
                                     "human play: a person reacts to what blue does."},
        "definitions": {
            "switches_per_s": "tactic changes between consecutive 50 Hz decisions / simulated seconds",
            "period_2_cycle_fraction": "share of decisions inside A,B,A,B alternations at least 4 decisions long",
            "pre_contact_net_displacement_m": "straight-line blue travel from spawn to first robot contact (or episode end)",
            "stationary_time_s": "time with under 10 mm net blue travel over the trailing 0.5 s (stopped, jittering in place, or stalled)",
            "stationary_wall_time_s": "time touching a wall with a non-stop tactic and under 0.1 mm travel in the step (env flag)",
            "win_credited": "opponent pit fall where always_stop does NOT also win the identical case",
            "headline": "group 'normal' vs scripted opponent; 'easy' and 'intermediate' are diagnostics"},
        "scenarios": suite, "missing_artifacts": missing,
        "matched_configuration": {str(seed): matched_configuration(seed) for seed in args.seeds},
        "training": {f"{arm}_seed{seed}": training_summary(ROOT / f"runs/prevact-{arm}-seed{seed}-100k")
                     for seed in args.seeds for arm in ARMS},
        "per_seed_headline": headline(results, summary, args.seeds),
        "overnight_ordinary_start_training": overnight_ordinary_start_training(),
        "summary": summary, "results": results,
    }
    Path(args.output).write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
