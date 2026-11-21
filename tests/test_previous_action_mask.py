"""Versioned previous-action observation encoding: mask, persistence, and legacy compatibility."""
import json
import shutil
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest
import yaml

from autobot.sim import cli
from autobot.sim.cli import (ReplaySession, checkpoint_observation_metadata, evaluate_model, load_checkpoint,
                             main, resolve_previous_action_encoding)
from autobot.sim.env import (ACTION_PROFILES, LEGACY_PREVIOUS_ACTION_ENCODING, PREVIOUS_ACTION_ENCODINGS,
                             AutoBotEnv, previous_action_slice)

MASKED = "masked_zero_v1"
ONEHOT = "onehot_v1"
LEGACY_RUN = Path("runs/ppo-curriculum-overnight")
RECORDING = Path("artifacts/human-vs-final-default.jsonl")


def _pair(profile, action_profile, seed=91):
    make = lambda encoding: AutoBotEnv(observation_profile=profile, seed=seed, opponent_mode="scripted",
                                       action_profile=action_profile, previous_action_encoding=encoding)
    return make(ONEHOT), make(MASKED)


def test_legacy_encoding_is_the_default_and_unknown_encodings_are_rejected():
    assert LEGACY_PREVIOUS_ACTION_ENCODING == ONEHOT and PREVIOUS_ACTION_ENCODINGS == (ONEHOT, MASKED)
    env = AutoBotEnv(seed=3)
    assert env.previous_action_encoding == ONEHOT
    obs, info = env.reset(seed=3)
    assert info["previous_action_encoding"] == ONEHOT
    assert obs[previous_action_slice("legacy5")].tolist() == [1.0, 0.0, 0.0, 0.0, 0.0]
    with pytest.raises(ValueError):
        AutoBotEnv(seed=3, previous_action_encoding="masked_zero_v2")


@pytest.mark.parametrize("profile", ["hardware_plausible", "sim_debug"])
@pytest.mark.parametrize("action_profile", ["legacy5", "experimental5_v1", "arc7_v1"])
def test_mask_zeroes_only_previous_action_channels_at_reset_and_every_step(profile, action_profile):
    onehot, masked = _pair(profile, action_profile)
    span = previous_action_slice(action_profile)
    slots = ACTION_PROFILES[action_profile]["previous_action_slots"]
    other = np.ones(onehot.observation_space.shape[0], dtype=bool); other[span] = False
    # Same shape, bounds and layout: a space check cannot distinguish the two encodings.
    assert onehot.observation_space == masked.observation_space
    plain, _ = onehot.reset(seed=91); hidden, info = masked.reset(seed=91)
    assert info["previous_action_encoding"] == MASKED
    assert plain.shape == hidden.shape and plain.dtype == hidden.dtype
    assert not hidden[span].any() and plain[span].tolist() == np.eye(slots)[0].tolist()
    assert np.array_equal(plain[other], hidden[other])
    rng = np.random.default_rng(5)
    for action in [int(rng.integers(onehot.action_space.n)) for _ in range(40)]:
        plain = onehot.step(action)[0]; hidden = masked.step(action)[0]
        assert hidden[span].tolist() == [0.0] * slots
        assert plain[span].tolist() == np.eye(slots)[action].tolist()
        # Every non-action channel (IMU incl. its noise draw, and privileged truth) is bit-identical.
        assert np.array_equal(plain[other], hidden[other])
        for bot in (0, 1):
            assert np.array_equal(onehot.sim.truth(bot)["position_m"], masked.sim.truth(bot)["position_m"])
    # The mask also holds after a mid-episode reset.
    assert not masked.reset(seed=92)[0][span].any()


def _tiny_setup(tmp_path):
    config = yaml.safe_load(Path("config/sim.yaml").read_text(encoding="utf-8"))
    config["simulation"]["episode_seconds"] = 0.4
    config_path = tmp_path / "sim.yaml"; config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    scenarios = tmp_path / "eval.json"
    scenarios.write_text(json.dumps([{"name": "default_spawn", "group": "normal",
                                      "player": [0.575, 0.18, 1.5707963], "opponent": [0.575, 0.97, -1.5707963]}]), encoding="utf-8")
    return str(config_path), str(scenarios)


def _train(config, scenarios, output, *extra):
    main(["--config", config, "--seed", "5", "train", "--timesteps", "64", "--output", str(output), "--profile", "sim_debug",
          "--eval-scenarios", scenarios, "--checkpoint-freq", "32", "--eval-freq", "32", "--latest-freq", "32", *extra])


def _replay_args(config, scenarios, model, **extra):
    return Namespace(config=config, profile="sim_debug", seed=5, opponent_chassis="last_year", opponent="scripted",
                     eval_scenarios=scenarios, scenario_index=0, reward_profile="outcome_only", model=str(model),
                     action_profile="legacy5", tactic_hold_ms=0.0, **extra)


def test_masked_run_persists_setting_and_replay_evaluate_resume_restore_it(tmp_path, monkeypatch):
    config, scenarios = _tiny_setup(tmp_path)
    run = tmp_path / "masked"
    _train(config, scenarios, run, "--previous-action-encoding", MASKED)
    frozen = json.loads((run / "effective_config.json").read_text(encoding="utf-8"))
    assert frozen["previous_action_encoding"] == MASKED
    assert "previous_tactic_masked_zero[5]" in frozen["observation_fields"]
    checkpoints = [run / "final_model.zip", run / "latest_model.zip", run / "best" / "best_model.zip",
                   *sorted((run / "checkpoints").glob("*.zip"))]
    assert len(checkpoints) >= 4
    for checkpoint in checkpoints:
        assert checkpoint_observation_metadata(checkpoint)["previous_action_encoding"] == MASKED
        assert resolve_previous_action_encoding(checkpoint) == MASKED
    # The setting travels with the file: no run directory, no sidecar config.
    moved = tmp_path / "elsewhere" / "copied.zip"; moved.parent.mkdir(); shutil.copy(run / "final_model.zip", moved)
    assert resolve_previous_action_encoding(moved) == MASKED
    assert resolve_previous_action_encoding(str(moved)[:-4]) == MASKED  # SB3-style path without .zip

    # Replay restores the mask without being told, and applies it on every step.
    session = ReplaySession(_replay_args(config, scenarios, moved))
    assert session.previous_action_encoding == MASKED and session.env.previous_action_encoding == MASKED
    span = previous_action_slice("legacy5")
    for _ in range(5):
        assert not session.env._obs()[span].any()
        session.step()
    # Same observation space, so only the recorded setting can catch this mismatch.
    assert session.env.observation_space == AutoBotEnv(config, "sim_debug").observation_space
    with pytest.raises(ValueError, match="refusing to run it with onehot_v1"):
        ReplaySession(_replay_args(config, scenarios, moved, previous_action_encoding=ONEHOT))

    # Evaluation of a loaded checkpoint builds its environments with the restored setting.
    seen = []
    class Spy(AutoBotEnv):
        def __init__(self, *args, **kwargs):
            seen.append(kwargs.get("previous_action_encoding")); super().__init__(*args, **kwargs)
    monkeypatch.setattr(cli, "AutoBotEnv", Spy)
    model, encoding = load_checkpoint(moved, device="cpu")
    evaluate_model(model, config, "sim_debug", 5, "scripted", "last_year", json.loads(Path(scenarios).read_text()))
    assert encoding == MASKED and seen == [MASKED]
    monkeypatch.undo()

    # Resume into a fresh directory restores the mask from the checkpoint alone.
    resumed = tmp_path / "resumed"
    _train(config, scenarios, resumed, "--resume", str(moved))
    assert json.loads((resumed / "effective_config.json").read_text(encoding="utf-8"))["previous_action_encoding"] == MASKED
    assert checkpoint_observation_metadata(resumed / "final_model.zip")["previous_action_encoding"] == MASKED
    # In-place resume passes the frozen-config check; a conflicting request is refused both ways.
    _train(config, scenarios, run, "--resume", str(run / "latest_model.zip"))
    with pytest.raises(SystemExit, match="Resume configuration mismatch"):
        _train(config, scenarios, tmp_path / "conflict", "--resume", str(moved), "--previous-action-encoding", ONEHOT)
    assert not (tmp_path / "conflict").exists()


def test_unmasked_and_unversioned_checkpoints_keep_legacy_behaviour(tmp_path):
    from stable_baselines3 import PPO
    config, scenarios = _tiny_setup(tmp_path)
    run = tmp_path / "onehot"
    _train(config, scenarios, run)
    assert json.loads((run / "effective_config.json").read_text(encoding="utf-8"))["previous_action_encoding"] == ONEHOT
    assert "previous_tactic_one_hot[5]" in json.loads((run / "effective_config.json").read_text(encoding="utf-8"))["observation_fields"]
    assert checkpoint_observation_metadata(run / "final_model.zip")["previous_action_encoding"] == ONEHOT
    with pytest.raises(ValueError, match="refusing to run it with masked_zero_v1"):
        ReplaySession(_replay_args(config, scenarios, run / "final_model.zip", previous_action_encoding=MASKED))
    with pytest.raises(SystemExit, match="Resume configuration mismatch"):
        _train(config, scenarios, tmp_path / "conflict", "--resume", str(run / "final_model.zip"),
               "--previous-action-encoding", MASKED)

    # A checkpoint written before versioning: no embedded record and no sidecar key.
    bare = tmp_path / "unversioned" / "old_model.zip"; bare.parent.mkdir()
    PPO("MlpPolicy", AutoBotEnv(config, "sim_debug"), device="cpu", seed=1, n_steps=8, batch_size=8).save(bare)
    (bare.parent / "effective_config.json").write_text(json.dumps({"profile": "sim_debug"}), encoding="utf-8")
    assert checkpoint_observation_metadata(bare) is None
    assert resolve_previous_action_encoding(bare) == ONEHOT
    with pytest.raises(ValueError, match="predates versioning"):
        resolve_previous_action_encoding(bare, MASKED)
    session = ReplaySession(_replay_args(config, scenarios, bare))
    assert session.previous_action_encoding == ONEHOT
    session.step(forced_action=3)
    assert session.env._obs()[previous_action_slice("legacy5")].tolist() == [0.0, 0.0, 0.0, 1.0, 0.0]
    # Resuming it in place (legacy frozen config without the key) stays one-hot.
    _train(config, scenarios, tmp_path / "old-resumed", "--resume", str(bare))
    assert checkpoint_observation_metadata(tmp_path / "old-resumed" / "final_model.zip")["previous_action_encoding"] == ONEHOT


@pytest.mark.skipif(not ((LEGACY_RUN / "final_model.zip").is_file() and RECORDING.is_file()),
                    reason="local legacy checkpoint or recorded human round not present (runs/ is untracked)")
def test_legacy_overnight_checkpoint_replays_the_recorded_round_exactly():
    """The real pre-versioning checkpoint must reproduce the blue tactics logged by `play`."""
    checkpoint = LEGACY_RUN / "final_model.zip"
    assert checkpoint_observation_metadata(checkpoint) is None
    assert resolve_previous_action_encoding(checkpoint) == ONEHOT
    with pytest.raises(ValueError):
        resolve_previous_action_encoding(checkpoint, MASKED)
    rows = [json.loads(line) for line in RECORDING.read_text(encoding="utf-8").splitlines()]
    split = next(index for index in range(1, len(rows)) if rows[index]["time_s"] < rows[index - 1]["time_s"])
    recorded = rows[split:]
    span = previous_action_slice("legacy5")

    def replay():
        session = ReplaySession(Namespace(
            config=str(LEGACY_RUN / "effective_sim.yaml"), profile="sim_debug", seed=10007, opponent_chassis="last_year",
            opponent="static", eval_scenarios=str(LEGACY_RUN / "evaluation_scenarios.json"), scenario_index=2,
            reward_profile="outcome_only", model=str(checkpoint), action_profile="legacy5", tactic_hold_ms=0.0))
        assert session.previous_action_encoding == ONEHOT
        tactics, observations, previous = [], [], "stop"
        for row in recorded:
            observation = session.env._obs()
            assert observation[span].tolist() == np.eye(5)[session.env.tactics.index(previous)].tolist()
            observations.append(observation)
            done, info = session.step(tuple(row["red_command"]))
            assert info["blue_drive"] == tuple(row["blue_command"])
            tactics.append(session.current_tactic); previous = session.current_tactic
            if done: break
        pose = np.r_[session.sim.truth(0)["position_m"], session.sim.truth(0)["yaw_rad"]]
        return tactics, np.asarray(observations), pose, session.end_reason

    tactics, observations, pose, end_reason = replay()
    assert len(tactics) == len(recorded) == 913
    assert tactics == [row["blue_tactic"] for row in recorded]
    assert end_reason == recorded[-1]["end_reason"] == "Pit fall (bot 0)"
    # Fingerprint captured from the code before the encoding was versioned.
    assert np.allclose(pose, [0.9473934121805401, 0.07408563329506997, -0.020860371446633115, 2.4041383022385383], atol=1e-9)
    again_tactics, again_observations, again_pose, _ = replay()
    assert again_tactics == tactics and np.array_equal(again_observations, observations) and np.array_equal(again_pose, pose)


def test_mobility_metrics_separate_jitter_in_place_from_travel_and_wall_stall():
    from autobot.sim.diagnostics import mobility_metrics, period_two_runs
    tactics = ("stop", "advance", "reverse", "turn_left", "turn_right")
    assert period_two_runs([3, 4, 3, 4, 3, 1, 1, 1]) == [(0, 5)]
    assert period_two_runs([1, 1, 1, 1]) == [] and period_two_runs([3, 4, 3]) == []
    steps = 100
    # Jitter in place: alternating turns, centre wobbling by 1 mm, no contact.
    wobble = [[0.001 * (index % 2), 0.0] for index in range(steps)]
    jitter = mobility_metrics([0.0, 0.0], wobble, [3, 4] * (steps // 2), [False] * steps, [False] * steps, 50, tactics)
    assert jitter["switches_per_s"] == pytest.approx(49.5) and jitter["period_2_cycle_fraction"] == 1.0
    assert jitter["stationary_time_s"] == pytest.approx(2.0) and jitter["stationary_wall_time_s"] == 0.0
    assert jitter["pre_contact_net_displacement_m"] <= 0.001 and jitter["time_to_first_contact_s"] is None
    # Straight drive at 0.5 m/s with contact from decision 50: never stationary before contact.
    drive = [[0.0, 0.01 * (index + 1)] for index in range(50)] + [[0.0, 0.5]] * 50
    travel = mobility_metrics([0.0, 0.0], drive, [1] * steps, [False] * 49 + [True] * 51, [False] * 50 + [True] * 50, 50, tactics)
    assert travel["switches_per_s"] == 0.0 and travel["period_2_cycle_fraction"] == 0.0
    assert travel["time_to_first_contact_s"] == pytest.approx(1.0)
    assert travel["pre_contact_net_displacement_m"] == pytest.approx(0.5) and travel["pre_contact_path_m"] == pytest.approx(0.5)
    assert travel["pre_contact_stationary_time_s"] == 0.0
    # Parked from decision 49; the trailing 0.5 s window is fully parked from decision 74 (26 decisions).
    assert travel["stationary_time_s"] == pytest.approx(0.52) and travel["stationary_wall_time_s"] == pytest.approx(1.0)
