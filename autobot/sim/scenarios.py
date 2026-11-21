"""Seeded training curricula and fixed evaluation scenario loading."""
from __future__ import annotations
import json
from pathlib import Path


def load_scenario_file(path):
    if not path: return None
    return json.loads(Path(path).read_text(encoding="utf-8"))


def validate_pose_scenario(scenario):
    if not all(key in scenario for key in ("name","player","opponent")):
        raise ValueError("Each scenario needs name, player, and opponent")
    if len(scenario["player"])!=3 or len(scenario["opponent"])!=3:
        raise ValueError("Scenario poses are [x, y, yaw]")


def validate_evaluation_scenarios(scenarios):
    if not isinstance(scenarios,list) or not scenarios:
        raise ValueError("Evaluation scenarios must be a non-empty JSON list")
    for scenario in scenarios: validate_pose_scenario(scenario)
    return scenarios


def validate_curriculum(curriculum):
    if not isinstance(curriculum,dict) or not curriculum.get("stages") or not curriculum.get("groups"):
        raise ValueError("Curriculum needs non-empty stages and groups")
    previous=-1
    for stage in curriculum["stages"]:
        until=stage.get("until_episode")
        if until is not None and until<=previous: raise ValueError("Curriculum until_episode values must increase")
        if until is not None: previous=until
        weights=stage.get("group_weights",{})
        if not weights or any(group not in curriculum["groups"] for group in weights):
            raise ValueError("Curriculum stage references an unknown or empty group")
        if sum(float(weight) for weight in weights.values())<=0: raise ValueError("Curriculum group weights must be positive")
    for scenarios in curriculum["groups"].values():
        if not scenarios: raise ValueError("Curriculum groups cannot be empty")
        for scenario in scenarios: validate_pose_scenario(scenario)
    return curriculum


def curriculum_stage(curriculum, episode_index):
    for stage in curriculum["stages"]:
        until=stage.get("until_episode")
        if until is None or episode_index < int(until): return stage
    return curriculum["stages"][-1]


def sample_curriculum(curriculum, episode_index, rng):
    stage=curriculum_stage(curriculum,episode_index)
    groups=list(stage["group_weights"]); weights=[float(stage["group_weights"][group]) for group in groups]
    group=str(rng.choice(groups,p=[weight/sum(weights) for weight in weights]))
    scenarios=curriculum["groups"][group]
    return scenarios[int(rng.integers(len(scenarios)))],stage["name"],group
