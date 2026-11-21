"""Test deterministic policy sensitivity to only the previous-action observation."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from autobot.sim.env import ACTION_PROFILES, AutoBotEnv
from autobot.sim.scenarios import load_scenario_file


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "runs/ppo-curriculum-overnight/effective_sim.yaml"
SEED = 20017


def inspect(name, checkpoint, action_profile):
    model = PPO.load(checkpoint, device="cpu")
    scenario = next(item for item in load_scenario_file(
        ROOT / "runs/ppo-curriculum-overnight/evaluation_scenarios.json") if item["name"] == "default_spawn")
    env = AutoBotEnv(str(CONFIG), "sim_debug", SEED, opponent_model="last_year",
                     opponent_mode="scripted", reward_profile="outcome_only", action_profile=action_profile)
    obs, _ = env.reset(seed=SEED, options={"scenario": scenario})
    slots = ACTION_PROFILES[action_profile]["previous_action_slots"]
    tactics = ACTION_PROFILES[action_profile]["tactics"]
    examined = sensitive = changed_from_actual = 0
    unique_histogram = Counter()
    response_by_slot = {"zero": Counter(), **{tactic: Counter() for tactic in tactics}}
    while env.sim.data.time < 5.0:
        actual = int(model.predict(obs, deterministic=True)[0])
        variants = []
        zero = obs.copy(); zero[6:6 + slots] = 0.0
        zero_action = int(model.predict(zero, deterministic=True)[0])
        variants.append(zero_action); response_by_slot["zero"][tactics[zero_action]] += 1
        for slot, previous_name in enumerate(tactics):
            changed = obs.copy(); changed[6:6 + slots] = 0.0; changed[6 + slot] = 1.0
            response = int(model.predict(changed, deterministic=True)[0])
            variants.append(response); response_by_slot[previous_name][tactics[response]] += 1
        unique = len(set(variants)); unique_histogram[unique] += 1
        sensitive += int(unique > 1)
        changed_from_actual += sum(item != actual for item in variants)
        examined += 1
        obs, _, terminated, truncated, _ = env.step(actual)
        if terminated or truncated:
            break
    return {
        "checkpoint": str(checkpoint), "action_profile": action_profile, "states": examined,
        "states_where_previous_action_changes_output": sensitive,
        "sensitive_state_fraction": sensitive / examined,
        "variant_outputs_different_from_actual_fraction": changed_from_actual / (examined * (len(tactics) + 1)),
        "unique_action_count_histogram": dict(unique_histogram),
        "response_by_forced_previous_action": {key: dict(value) for key, value in response_by_slot.items()},
    }


def main():
    specifications = {
        "overnight_final": (ROOT / "runs/ppo-curriculum-overnight/final_model.zip", "legacy5"),
        "fresh5_best": (ROOT / "runs/action-compare-5-seed17-100k/best/best_model.zip", "experimental5_v1"),
        "fresh5_final": (ROOT / "runs/action-compare-5-seed17-100k/final_model.zip", "experimental5_v1"),
        "fresh7_best": (ROOT / "runs/action-compare-7-seed17-100k/best/best_model.zip", "arc7_v1"),
        "fresh7_final": (ROOT / "runs/action-compare-7-seed17-100k/final_model.zip", "arc7_v1"),
    }
    output = {name: inspect(name, *specification) for name, specification in specifications.items()}
    path = ROOT / "artifacts/previous-action-sensitivity.json"
    path.write_text(json.dumps({"seed": SEED, "window_s": 5.0, "results": output}, indent=2), encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()
