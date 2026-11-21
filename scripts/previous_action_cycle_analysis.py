"""Look for repeated action cycles at nearly identical physical states in the recorded human match.

Three kinds of evidence are kept strictly apart in the output:

* ``observed``            - what the checkpoint actually did on the recorded red inputs.
* ``pair_counterfactual`` - previous-action substitutions restricted to values that were
                            really observed at a nearly identical physical state.
* ``closed_loop_counterfactual`` - the same recorded red inputs with the previous-action
                            channels overridden before the policy sees them. Those inputs are
                            outside what an unmasked checkpoint was trained on.

Nothing here trains or modifies a checkpoint.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from autobot.sim.diagnostics import STATIONARY_NET_DISPLACEMENT_M, STATIONARY_WINDOW_S, mobility_metrics
from autobot.sim.env import AutoBotEnv
from autobot.sim.scenarios import load_scenario_file

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "runs/ppo-curriculum-overnight/effective_sim.yaml"
SCENARIOS = ROOT / "runs/ppo-curriculum-overnight/evaluation_scenarios.json"
RECORDING = ROOT / "artifacts/human-vs-final-default.jsonl"
HUMAN_SEED = 10007
IMU_CHANNELS = 6
# "Nearly identical physical state": both chassis within 1 mm and 0.5 degrees.
POSITION_TOLERANCE_M = 0.001
YAW_TOLERANCE_RAD = math.radians(0.5)


def recorded_round():
    """Second recorded round: the one Codex's matched diagnosis used (913 steps)."""
    rows = [json.loads(line) for line in RECORDING.read_text(encoding="utf-8").splitlines()]
    rows = [row for row in rows if row.get("type") == "step"]
    split = next(index for index in range(1, len(rows)) if rows[index]["time_s"] < rows[index - 1]["time_s"])
    return rows[split:]


def angle_delta(a, b):
    return abs(math.atan2(math.sin(a - b), math.cos(a - b)))


def rollout(model, action_profile, rows, previous_action_override=None):
    """Replay recorded red inputs. ``previous_action_override`` edits only what the policy sees."""
    scenario = next(item for item in load_scenario_file(SCENARIOS) if item["name"] == "default_spawn")
    env = AutoBotEnv(str(CONFIG), "sim_debug", HUMAN_SEED, opponent_model="last_year", opponent_mode="static",
                     reward_profile="outcome_only", action_profile=action_profile)
    slots = env.previous_action_slots
    obs, _ = env.reset(seed=HUMAN_SEED, options={"scenario": scenario})
    steps = []
    info = {}
    for row in rows:
        seen = obs.copy()
        if previous_action_override is not None:
            seen[IMU_CHANNELS:IMU_CHANNELS + slots] = 0.0
            if previous_action_override != "zero":
                seen[IMU_CHANNELS + int(previous_action_override)] = 1.0
        action = int(model.predict(seen, deterministic=True)[0])
        blue, red = env.sim.truth(0), env.sim.truth(1)
        steps.append({"obs": obs.copy(), "action": action, "previous_action": int(env.last_action),
                      "blue": np.array([*blue["position_m"][:2], blue["yaw_rad"]]),
                      "red": np.array([*red["position_m"][:2], red["yaw_rad"]])})
        obs, _, terminated, truncated, info = env.step(action, opponent_drive_override=tuple(row["red_command"]))
        steps[-1].update(contact=bool(info["opponent_contact"]), wall=bool(info["stationary_wall_contact"]),
                         blue_after=env.sim.truth(0)["position_m"][:2].copy())
        if terminated or truncated:
            break
    return env, steps, info


def same_physical_state(a, b):
    return (np.linalg.norm(a["blue"][:2] - b["blue"][:2]) <= POSITION_TOLERANCE_M
            and np.linalg.norm(a["red"][:2] - b["red"][:2]) <= POSITION_TOLERANCE_M
            and angle_delta(a["blue"][2], b["blue"][2]) <= YAW_TOLERANCE_RAD
            and angle_delta(a["red"][2], b["red"][2]) <= YAW_TOLERANCE_RAD)


def mobility(env, steps):
    """Same metric definitions as the matched evaluation (autobot.sim.diagnostics)."""
    result = mobility_metrics(steps[0]["blue"][:2], [step["blue_after"] for step in steps], [step["action"] for step in steps],
                              [step["contact"] for step in steps], [step["wall"] for step in steps],
                              env.sim.cfg["simulation"]["control_hz"], env.tactics)
    for key in ("tactic_switches", "total_path_m", "contact_time_s", "period_2_cycle_fraction"): result.pop(key)
    return result


def cycle_runs(actions, period):
    """Maximal runs with a[t]==a[t-period], at least two full cycles, and a genuine alternation."""
    runs, start = [], None
    for index in range(period, len(actions) + 1):
        periodic = index < len(actions) and actions[index] == actions[index - period]
        if periodic and start is None:
            start = index - period
        if not periodic and start is not None:
            window = actions[start:index]
            minimal = all(any(window[k] != window[k - shorter] for k in range(shorter, len(window)))
                          for shorter in range(1, period))
            if len(window) >= 2 * period and len(set(window)) > 1 and minimal:
                runs.append((start, index))
            start = None
    return runs


def observed_analysis(env, steps):
    tactics = env.tactics
    slots = env.previous_action_slots
    actions = [step["action"] for step in steps]
    cycles = {}
    for period in (2, 3, 4):
        runs = cycle_runs(actions, period)
        lengths = [end - begin for begin, end in runs]
        patterns = Counter("->".join(tactics[a] for a in actions[begin:begin + period]) for begin, _ in runs)
        cycles[f"period_{period}"] = {
            "runs": len(runs), "steps_inside_runs": int(sum(lengths)),
            "fraction_of_steps": sum(lengths) / len(actions),
            "longest_run_steps": max(lengths, default=0), "median_run_steps": float(np.median(lengths)) if lengths else 0.0,
            "patterns_by_run_count": dict(patterns.most_common(6))}
    # Period-2 recurrence: the bot is back at (almost) the same physical state two decisions
    # later and repeats the same action; one decision later it is (almost) there too and flips.
    recurrence = []
    for begin, end in cycle_runs(actions, 2):
        for index in range(begin + 2, end):
            recurrence.append((float(np.linalg.norm(steps[index]["blue"][:2] - steps[index - 2]["blue"][:2])),
                               angle_delta(steps[index]["blue"][2], steps[index - 2]["blue"][2])))
    recurrence = np.array(recurrence) if recurrence else np.zeros((0, 2))
    # Adjacent decisions at a nearly identical physical state.
    adjacent = [(index, index + 1) for index in range(len(steps) - 1) if same_physical_state(steps[index], steps[index + 1])]
    flipped = [(i, j) for i, j in adjacent if actions[i] != actions[j]]
    non_action = lambda step: np.r_[step["obs"][:IMU_CHANNELS], step["obs"][IMU_CHANNELS + slots:]]
    flip_gap = np.array([np.abs(non_action(steps[i]) - non_action(steps[j])).max() for i, j in flipped]) if flipped else np.zeros(0)
    # Natural experiment over every pair of visited states, adjacent or not.
    table = Counter()
    for i in range(len(steps)):
        for j in range(i + 1, len(steps)):
            if same_physical_state(steps[i], steps[j]):
                table[(steps[i]["previous_action"] == steps[j]["previous_action"], actions[i] == actions[j])] += 1
    same_prev = table[(True, True)] + table[(True, False)]
    diff_prev = table[(False, True)] + table[(False, False)]
    return {
        "cycles": cycles,
        "period_2_same_phase_recurrence": {
            "samples": int(len(recurrence)),
            "median_position_gap_m": float(np.median(recurrence[:, 0])) if len(recurrence) else None,
            "p95_position_gap_m": float(np.percentile(recurrence[:, 0], 95)) if len(recurrence) else None,
            "median_yaw_gap_rad": float(np.median(recurrence[:, 1])) if len(recurrence) else None,
            "p95_yaw_gap_rad": float(np.percentile(recurrence[:, 1], 95)) if len(recurrence) else None},
        "adjacent_decisions_at_nearly_identical_state": {
            "pairs": len(adjacent), "pairs_with_different_action": len(flipped),
            "fraction_with_different_action": len(flipped) / max(len(adjacent), 1),
            "non_action_observation_gap_when_flipped": {
                "median_max_abs": float(np.median(flip_gap)) if len(flip_gap) else None,
                "p95_max_abs": float(np.percentile(flip_gap, 95)) if len(flip_gap) else None},
            "flip_transitions": dict(Counter(f"{tactics[actions[i]]}->{tactics[actions[j]]}" for i, j in flipped).most_common(8))},
        "all_pairs_at_nearly_identical_state": {
            "same_previous_action": {"pairs": same_prev, "different_chosen_action": table[(True, False)],
                                     "fraction_different": table[(True, False)] / max(same_prev, 1)},
            "different_previous_action": {"pairs": diff_prev, "different_chosen_action": table[(False, False)],
                                          "fraction_different": table[(False, False)] / max(diff_prev, 1)}},
    }, flipped


def pair_counterfactual(model, env, steps, flipped):
    """Swap ONLY the previous-action channels between two observed, nearly identical states."""
    slots = env.previous_action_slots
    span = slice(IMU_CHANNELS, IMU_CHANNELS + slots)
    follows_previous = keeps_own = other = 0
    for i, j in flipped:
        for own, partner in ((i, j), (j, i)):
            swapped = steps[own]["obs"].copy(); swapped[span] = steps[partner]["obs"][span]
            response = int(model.predict(swapped, deterministic=True)[0])
            if response == steps[partner]["action"]: follows_previous += 1
            elif response == steps[own]["action"]: keeps_own += 1
            else: other += 1
    total = max(follows_previous + keeps_own + other, 1)
    return {"substitutions": follows_previous + keeps_own + other,
            "action_follows_swapped_previous_action": follows_previous,
            "action_unchanged": keeps_own, "other_action": other,
            "fraction_following_previous_action": follows_previous / total,
            "note": "Each substituted previous action was observed at a state within the stated tolerance."}


def analyse(name, checkpoint, action_profile, rows):
    model = PPO.load(checkpoint, device="cpu")
    env, steps, info = rollout(model, action_profile, rows)
    recorded = [row["blue_tactic"] for row in rows[:len(steps)]]
    mismatches = sum(env.tactics[step["action"]] != tactic for step, tactic in zip(steps, recorded))
    observed, flipped = observed_analysis(env, steps)
    closed_loop = {}
    for label, override in [("zero", "zero")] + [(f"held_{tactic}", index) for index, tactic in enumerate(env.tactics)]:
        other_env, other_steps, other_info = rollout(model, action_profile, rows, override)
        closed_loop[label] = dict(mobility(other_env, other_steps), end_reason=other_info.get("round_end_reason"),
                                  end_bot=other_info.get("round_end_bot"))
    return {
        "checkpoint": str(Path(checkpoint).relative_to(ROOT)), "action_profile": action_profile,
        "recorded_blue_tactic_mismatches": mismatches if name == "overnight_final" else None,
        "observed": dict(mobility(env, steps), end_reason=info.get("round_end_reason"), end_bot=info.get("round_end_bot"), **observed),
        "pair_counterfactual": pair_counterfactual(model, env, steps, flipped),
        "closed_loop_counterfactual": closed_loop,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(ROOT / "artifacts/previous-action-cycle-analysis.json"))
    args = parser.parse_args()
    rows = recorded_round()
    checkpoints = {
        "overnight_final": (ROOT / "runs/ppo-curriculum-overnight/final_model.zip", "legacy5"),
        "fresh5_best": (ROOT / "runs/action-compare-5-seed17-100k/best/best_model.zip", "experimental5_v1"),
        "fresh5_final": (ROOT / "runs/action-compare-5-seed17-100k/final_model.zip", "experimental5_v1"),
    }
    output = {
        "recording": str(RECORDING.relative_to(ROOT)), "recorded_round": "second", "recorded_steps": len(rows),
        "config": str(CONFIG.relative_to(ROOT)), "seed": HUMAN_SEED,
        "scope": "Recorded red inputs replayed open-loop. This is not evidence about live human play: "
                 "a human would react differently to a different blue policy.",
        "nearly_identical_state": {"position_tolerance_m": POSITION_TOLERANCE_M, "yaw_tolerance_rad": YAW_TOLERANCE_RAD,
                                   "applies_to": "blue and red chassis"},
        "stationary_definition": {"window_s": STATIONARY_WINDOW_S, "net_displacement_below_m": STATIONARY_NET_DISPLACEMENT_M},
        "results": {name: analyse(name, path, profile, rows) for name, (path, profile) in checkpoints.items() if path.exists()},
        "missing_checkpoints": [name for name, (path, _) in checkpoints.items() if not path.exists()],
    }
    Path(args.output).write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
