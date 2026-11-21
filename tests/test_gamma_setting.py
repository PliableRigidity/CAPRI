"""Configurable PPO discount: validation, persistence, resume protection, and curriculum stage logging."""
import json
import math
from pathlib import Path

import pytest
import yaml

from autobot.sim.cli import (LEGACY_GAMMA, checkpoint_observation_metadata, checkpoint_ppo_settings, load_checkpoint,
                             main, resolve_gamma, validate_gamma)


@pytest.fixture(autouse=True)
def _restore_torch_threads():
    import torch
    before = torch.get_num_threads()
    yield
    torch.set_num_threads(before)


def _setup(tmp_path, episode_seconds=0.4):
    config = yaml.safe_load(Path("config/sim.yaml").read_text(encoding="utf-8"))
    config["simulation"]["episode_seconds"] = episode_seconds
    config_path = tmp_path / "sim.yaml"; config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    scenarios = tmp_path / "eval.json"
    scenarios.write_text(json.dumps([{"name": "default_spawn", "group": "normal",
                                      "player": [0.575, 0.18, 1.5707963], "opponent": [0.575, 0.97, -1.5707963]}]), encoding="utf-8")
    return str(config_path), str(scenarios)


def _train(config, scenarios, output, *extra, timesteps=64):
    main(["--config", config, "--seed", "5", "train", "--timesteps", str(timesteps), "--output", str(output),
          "--profile", "sim_debug", "--action-profile", "experimental5_v1", "--previous-action-encoding", "masked_zero_v1",
          "--eval-scenarios", scenarios, "--checkpoint-freq", "32", "--eval-freq", "32", "--latest-freq", "32", *extra])


def _rows(run):
    return [json.loads(line) for line in (Path(run) / "metrics.jsonl").read_text(encoding="utf-8").splitlines()]


def _frozen(run):
    return json.loads((Path(run) / "effective_config.json").read_text(encoding="utf-8"))


def test_gamma_validation_accepts_only_finite_values_in_zero_one():
    assert LEGACY_GAMMA == 0.99
    assert [validate_gamma(value) for value in (0.99, "0.999", 1, 1e-6)] == [0.99, 0.999, 1.0, 1e-6]
    for bad in (0, 0.0, -0.5, 1.0000001, 2, math.nan, math.inf, -math.inf, "fast", None, True, [0.9]):
        with pytest.raises(ValueError, match="gamma must be"):
            validate_gamma(bad)
    assert resolve_gamma() == 0.99 and resolve_gamma(0.999) == 0.999


def test_invalid_gamma_is_rejected_before_anything_is_written(tmp_path, capsys):
    config, scenarios = _setup(tmp_path)
    for bad in ("1.5", "0", "nan", "-0.1", "abc"):
        with pytest.raises(SystemExit) as failure:
            _train(config, scenarios, tmp_path / "bad", "--gamma", bad)
        assert failure.value.code == 2 and "gamma must be" in capsys.readouterr().err
    assert not (tmp_path / "bad").exists()


def test_gamma_is_used_by_ppo_recorded_everywhere_and_protected_on_resume(tmp_path):
    config, scenarios = _setup(tmp_path)
    default = tmp_path / "default"; high = tmp_path / "high"
    _train(config, scenarios, default)
    _train(config, scenarios, high, "--gamma", "0.999", "--torch-threads", "1")

    # Legacy default is unchanged, and the high-gamma run differs from it in gamma only.
    assert _frozen(default)["gamma"] == 0.99 and _frozen(high)["gamma"] == 0.999
    base, changed = _frozen(default)["ppo_hyperparameters"], _frozen(high)["ppo_hyperparameters"]
    assert base["gamma"] == base["rollout_buffer_gamma"] == 0.99
    assert changed["gamma"] == changed["rollout_buffer_gamma"] == 0.999 and changed["torch_threads"] == 1
    same = set(base) - {"gamma", "rollout_buffer_gamma", "torch_threads"}
    assert {key: base[key] for key in same} == {key: changed[key] for key in same}
    assert base["gae_lambda"] == 0.95 and base["n_steps"] == 32 and base["batch_size"] == 32
    differing = {key for key in set(_frozen(default)) | set(_frozen(high)) if _frozen(default).get(key) != _frozen(high).get(key)}
    assert differing == {"gamma", "ppo_hyperparameters"}

    # Every checkpoint of the run carries the discount, the observation encoding and the action profile.
    checkpoints = [high / "final_model.zip", high / "latest_model.zip", high / "best" / "best_model.zip",
                   *sorted((high / "checkpoints").glob("*.zip"))]
    assert len(checkpoints) >= 4
    for checkpoint in checkpoints:
        assert checkpoint_ppo_settings(checkpoint)["gamma"] == 0.999
        observation = checkpoint_observation_metadata(checkpoint)
        assert observation["previous_action_encoding"] == "masked_zero_v1" and observation["action_profile"] == "experimental5_v1"
    model, encoding = load_checkpoint(high / "final_model.zip", device="cpu")
    assert model.gamma == 0.999 and encoding == "masked_zero_v1"
    assert checkpoint_ppo_settings(default / "final_model.zip")["gamma"] == 0.99
    start = _rows(high)[0]
    assert start["type"] == "run_start" and start["ppo_hyperparameters"]["gamma"] == 0.999 and start["resumed_from"] is None

    # Resume keeps the checkpoint's discount when none is requested...
    resumed = tmp_path / "resumed"
    _train(config, scenarios, resumed, "--resume", str(high / "final_model.zip"))
    assert _frozen(resumed)["gamma"] == 0.999 and _frozen(resumed)["ppo_hyperparameters"]["rollout_buffer_gamma"] == 0.999
    assert checkpoint_ppo_settings(resumed / "final_model.zip")["gamma"] == 0.999
    # ...accepts the same value stated explicitly, in place...
    _train(config, scenarios, high, "--resume", str(high / "latest_model.zip"), "--gamma", "0.999")
    assert [row["ppo_hyperparameters"]["gamma"] for row in _rows(high) if row["type"] == "run_start"] == [0.999, 0.999]
    # ...and refuses a different one in either direction, without creating a run.
    for checkpoint, requested, recorded in ((high / "final_model.zip", "0.99", "0.999"), (default / "final_model.zip", "0.999", "0.99")):
        with pytest.raises(SystemExit, match=f"Resume configuration mismatch.*trained with gamma={recorded}; refusing to continue it with gamma={requested}"):
            _train(config, scenarios, tmp_path / "conflict", "--resume", str(checkpoint), "--gamma", requested)
    assert not (tmp_path / "conflict").exists()
    with pytest.raises(ValueError, match="refusing to continue"):
        resolve_gamma(0.99, high / "final_model.zip")
    assert resolve_gamma(None, high / "final_model.zip") == 0.999


def test_resume_refuses_a_run_directory_frozen_with_another_gamma(tmp_path):
    """Covers the case where the checkpoint and the directory's frozen config disagree."""
    config, scenarios = _setup(tmp_path)
    high = tmp_path / "high"; default = tmp_path / "default"
    _train(config, scenarios, high, "--gamma", "0.999"); _train(config, scenarios, default)
    with pytest.raises(SystemExit, match="Resume configuration mismatch: gamma"):
        _train(config, scenarios, default, "--resume", str(high / "final_model.zip"))


def test_curriculum_stage_changes_and_scenario_mix_are_logged_from_sampled_episodes(tmp_path):
    config, scenarios = _setup(tmp_path, episode_seconds=0.1)   # 5 decisions per episode
    curriculum = json.loads(Path("config/training_curriculum.json").read_text(encoding="utf-8"))
    curriculum["stages"][0]["until_episode"] = 3; curriculum["stages"][1]["until_episode"] = 7
    path = tmp_path / "curriculum.json"; path.write_text(json.dumps(curriculum), encoding="utf-8")
    run = tmp_path / "run"
    _train(config, scenarios, run, "--training-curriculum", str(path), "--gamma", "0.999")
    rows = _rows(run)
    episodes = [row for row in rows if row["type"] == "train_episode"]
    changes = [row for row in rows if row["type"] == "curriculum_stage_change"]
    assert [row["episode_index"] for row in episodes] == list(range(len(episodes))) and len(episodes) >= 10
    assert [(row["from"], row["to"], row["episode_index"]) for row in changes] == [
        (None, "near_pit", 0), ("near_pit", "widening", 3), ("widening", "mixed", 7)]
    assert [row["stage"] for row in episodes[:8]] == ["near_pit"] * 3 + ["widening"] * 4 + ["mixed"]
    assert changes[1]["first_episode_started_at_timesteps"] == sum(row["steps"] for row in episodes[:3])
    for row in episodes:
        assert row["steps"] == sum(row["actions"].values()) and row["scenario"].startswith("train_")
        assert row["scenario_group"] in curriculum["groups"]
    assert {row["scenario_group"] for row in episodes if row["stage"] == "near_pit"} <= {"near_pit_easy", "near_pit_offset"}
    # A resume continues the episode numbering and does not re-announce the stage it is already in.
    _train(config, scenarios, run, "--resume", str(run / "latest_model.zip"), "--training-curriculum", str(path))
    after = _rows(run)
    assert [row["episode_index"] for row in after if row["type"] == "train_episode"][len(episodes)] == len(episodes)
    assert len([row for row in after if row["type"] == "curriculum_stage_change"]) == 3
