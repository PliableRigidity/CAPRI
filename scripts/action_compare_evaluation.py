"""Controlled evaluation for the five-vs-seven action-space experiment."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from autobot.sim.cli import evaluate_model
from autobot.sim.env import AutoBotEnv, OPPONENT_MODES, ScriptedBaseline, StopBaseline
from autobot.sim.scenarios import load_scenario_file


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "runs/ppo-curriculum-overnight/effective_sim.yaml"
EVAL_SEED = 20017
HUMAN_SEED = 10007


def comparison_scenarios():
    original = load_scenario_file(ROOT / "runs/ppo-curriculum-overnight/evaluation_scenarios.json")
    diagnostic = load_scenario_file(ROOT / "config/diagnostic_scenarios.json")
    intermediate = [dict(item, group="intermediate") for item in diagnostic if item["group"].startswith("intermediate")]
    return original + intermediate


def actual_human_commands():
    rows = [json.loads(line) for line in (ROOT / "artifacts/human-vs-final-default.jsonl").read_text().splitlines()]
    split = next(index for index in range(1, len(rows)) if rows[index]["time_s"] < rows[index - 1]["time_s"])
    return [row["red_command"] for row in rows[split:]]


def human_regression(model, action_profile, scenario, commands):
    env = AutoBotEnv(str(CONFIG), "sim_debug", HUMAN_SEED, opponent_model="last_year",
                     opponent_mode="static", reward_profile="outcome_only", action_profile=action_profile)
    obs, _ = env.reset(seed=HUMAN_SEED, options={"scenario": scenario})
    blue_start = env.sim.truth(0)["position_m"][:2].copy()
    blue_previous = blue_start.copy()
    first_contact = None
    pre_contact_path = 0.0
    pre_contact_position = None
    previous_action = None
    switches = stationary = 0
    actions = Counter()
    info = {}
    terminated = truncated = False
    for red_command in commands:
        action = int(model.predict(obs, deterministic=True)[0])
        actions[env.tactics[action]] += 1
        obs, _, terminated, truncated, info = env.step(action, opponent_drive_override=red_command)
        blue_position = env.sim.truth(0)["position_m"][:2].copy()
        if first_contact is None:
            pre_contact_path += float(np.linalg.norm(blue_position - blue_previous))
        blue_previous = blue_position
        if first_contact is None and info["opponent_contact"]:
            first_contact = float(env.sim.data.time)
            pre_contact_position = blue_position.copy()
        switches += int(previous_action is not None and action != previous_action)
        previous_action = action
        stationary += int(info["stationary_wall_contact"])
        if terminated or truncated:
            break
    if info.get("round_end_reason") == "Pit fall":
        outcome = "own_falls" if info["round_end_bot"] == 0 else "wins"
    elif info.get("round_end_reason") == "Simultaneous pit fall":
        outcome = "draws"
    elif truncated:
        outcome = "timeouts"
    else:
        outcome = "recording_end"
    final_position = env.sim.truth(0)["position_m"][:2]
    return {
        "outcome": outcome,
        "steps_consumed": sum(actions.values()),
        "simulated_seconds": float(env.sim.data.time),
        "time_to_first_contact_s": first_contact,
        "switches_per_s": switches / max(float(env.sim.data.time), 1e-9),
        "stationary_time_s": stationary / env.sim.cfg["simulation"]["control_hz"],
        "pre_contact_net_displacement_m": float(np.linalg.norm(
            (pre_contact_position if pre_contact_position is not None else final_position) - blue_start)),
        "pre_contact_path_m": pre_contact_path,
        "action_counts": dict(actions),
        "end_reason": info.get("round_end_reason"),
    }


def main():
    scenarios = comparison_scenarios()
    policies = {
        "always_stop": (StopBaseline(), "legacy5"),
        "always_advance": (ScriptedBaseline(), "legacy5"),
        "overnight_final": (PPO.load(ROOT / "runs/ppo-curriculum-overnight/final_model.zip", device="cpu"), "legacy5"),
        "fresh5_best": (PPO.load(ROOT / "runs/action-compare-5-seed17-100k/best/best_model.zip", device="cpu"), "experimental5_v1"),
        "fresh5_final": (PPO.load(ROOT / "runs/action-compare-5-seed17-100k/final_model.zip", device="cpu"), "experimental5_v1"),
        "fresh7_best": (PPO.load(ROOT / "runs/action-compare-7-seed17-100k/best/best_model.zip", device="cpu"), "arc7_v1"),
        "fresh7_final": (PPO.load(ROOT / "runs/action-compare-7-seed17-100k/final_model.zip", device="cpu"), "arc7_v1"),
    }
    results = {}
    for name, (policy, action_profile) in policies.items():
        scripted = {}
        for behavior in OPPONENT_MODES:
            scripted[behavior] = evaluate_model(policy, str(CONFIG), "sim_debug", EVAL_SEED, behavior,
                                                "last_year", scenarios, "outcome_only", action_profile)
        results[name] = {"action_profile": action_profile, "scripted": scripted}
    default_spawn = next(item for item in scenarios if item["name"] == "default_spawn")
    commands = actual_human_commands()
    for name, (policy, action_profile) in policies.items():
        results[name]["recorded_human"] = human_regression(policy, action_profile, default_spawn, commands)
    output = {
        "training_seed": 17,
        "evaluation_seed": EVAL_SEED,
        "human_recording_seed": HUMAN_SEED,
        "config": str(CONFIG),
        "scenarios": scenarios,
        "human_command_steps": len(commands),
        "results": results,
    }
    path = ROOT / "artifacts/action-space-comparison.json"
    path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()
