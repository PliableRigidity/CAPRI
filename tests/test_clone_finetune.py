"""Clone fine-tuning pilot: scenario hygiene, actor import, critic warm-up, checkpoint schedule and the recommendation rule."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import clone_finetune as pilot  # noqa: E402
import heading_ladder as experiment  # noqa: E402


@pytest.fixture(scope="module")
def short_config(tmp_path_factory):
    """The frozen config with a 2 s horizon, so a few hundred training steps contain whole episodes."""
    config = yaml.safe_load(pilot.CONFIG.read_text(encoding="utf-8")); config["simulation"]["episode_seconds"] = 2.0
    path = tmp_path_factory.mktemp("pilot") / "sim.yaml"; path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path


@pytest.fixture(scope="module")
def mix():
    return pilot.load_mix()


def table(turn_tl, turn_br, new_wins, new_falls, fresh_wins, new_episodes=144):
    cell = lambda wins, falls=0, episodes=162: {"episodes": episodes, "wins": wins, "own_falls": falls, "timeouts": episodes - wins - falls}
    return {"turn_needed": {"all": cell(turn_tl + turn_br, episodes=324), "top_left": cell(turn_tl), "bottom_right": cell(turn_br)},
            "ordinary": {"new_confirmation": cell(new_wins, new_falls, new_episodes), "fresh_normal": cell(fresh_wins, episodes=36)}}


def test_mix_is_half_turn_needed_half_ordinary_and_skips_the_warm_up(mix):
    curriculum, confirmation = mix; ladder = experiment.load_ladder()
    assert curriculum["stages"] == [{"name": "mix", "until_episode": None, "group_weights": {"turn_needed": 0.5, "ordinary": 0.5}}]
    gated = [start for rung in ladder["rungs"] if not rung["warmup"] for start in rung["train"]]
    turn = curriculum["groups"]["turn_needed"]
    assert [start["name"] for start in turn] == [start["name"] for start in gated] and len(turn) == 60
    assert min(abs(start["heading_offset_deg"]) for start in turn) == 16 and {start["pit"] for start in turn} == set(pilot.PIT_NAMES)
    assert len(curriculum["groups"]["ordinary"]) == 40 and len(confirmation) == 48
    assert len({tuple(start["player"] + start["opponent"]) for start in curriculum["groups"]["ordinary"] + confirmation}) == 88


def test_no_training_start_repeats_a_held_out_case(mix):
    curriculum, confirmation = mix; held = pilot.heldout_starts(experiment.load_ladder())
    assert len(held) == 56 + 110 + 20
    pilot.check_disjoint(curriculum, held + confirmation)                      # does not raise
    for leaked in (held[60], held[-1], confirmation[0]):                         # a hold-out ladder start, a fresh normal start, a confirmation start
        nudged = dict(leaked, player=[leaked["player"][0] + 0.01, leaked["player"][1], leaked["player"][2] + 0.02])
        bad = dict(curriculum, groups=dict(curriculum["groups"], ordinary=curriculum["groups"]["ordinary"] + [nudged]))
        with pytest.raises(ValueError, match="held-out"): pilot.check_disjoint(bad, held + confirmation)
    far = dict(held[-1], player=[held[-1]["player"][0] + 0.03, *held[-1]["player"][1:]])
    assert not pilot.same_case(far, held[-1]) and pilot.same_case(held[-1], held[-1])


def test_generated_starts_are_reproducible_and_placeable(mix):
    from autobot.sim.core import load_config
    arena = load_config(str(pilot.CONFIG))["arena"]
    rebuilt, confirmation = pilot.build_mix(experiment.load_ladder(), arena)
    assert rebuilt == mix[0] and confirmation == mix[1]
    assert all(pilot.placeable(start, arena) for start in rebuilt["groups"]["ordinary"] + confirmation)
    in_pit = {"player": [0.1, 1.05, 0.0], "opponent": [0.6, 0.6, 0.0]}; touching = {"player": [0.5, 0.5, 0.0], "opponent": [0.6, 0.6, 0.0]}
    assert not pilot.placeable(in_pit, arena) and not pilot.placeable(touching, arena)


def test_checkpoints_are_indexed_by_actor_updates_and_every_arm_gets_the_same_steps():
    tuned = pilot.checkpoint_steps("finetune"); cold = pilot.checkpoint_steps("finetune_cold_critic")
    assert tuned[0] == 0 and tuned[pilot.WARMUP_STEPS] == 0 and tuned[pilot.WARMUP_STEPS + 256] == 1          # warm-up makes no actor update
    assert tuned[pilot.TOTAL_STEPS] == 1200 and cold[pilot.TOTAL_STEPS] == 1360 and cold[256] == 1
    assert pilot.checkpoint_steps("scratch") == cold and all(step % 256 == 0 for step in list(tuned) + list(cold))
    assert sorted(set(tuned.values())) == sorted(pilot.ACTOR_UPDATES)
    assert pilot.monitored("scratch", 80, 20480) and not pilot.monitored("scratch", 4, 1024) and pilot.monitored("finetune", 4, 1024)


def test_import_copies_only_the_actor_and_keeps_the_scratch_critic(short_config, mix, monkeypatch):
    monkeypatch.setattr(pilot, "CONFIG", short_config)
    clone = pilot.build_model("scratch", 5, mix[0])                                # stands in for the behaviour-cloned file
    scratch = pilot.build_model("scratch", 17, mix[0]); tuned = pilot.build_model("finetune", 17, mix[0], clone)
    observations = np.random.default_rng(0).uniform(-1, 1, size=(200, 19)).astype(np.float32)
    reference = pilot.action_probabilities(clone, observations)
    check = pilot.probability_check(tuned, reference, observations)
    assert check["identical"] and check["max_abs_probability_difference"] == 0.0 and check["same_greedy_tactic"] == 1.0
    assert not pilot.probability_check(scratch, reference, observations)["identical"]
    assert np.array_equal(pilot.flat(pilot.critic_modules(tuned.policy)), pilot.flat(pilot.critic_modules(scratch.policy)))
    assert not np.array_equal(pilot.flat(pilot.critic_modules(tuned.policy)), pilot.flat(pilot.critic_modules(clone.policy)))
    assert len(tuned.policy.optimizer.state_dict()["state"]) == 0                 # Adam has taken no step
    # Same seed and mix give the same start order whatever the arm.
    order = lambda model: [model.env.reset() is not None and model.env.envs[0].unwrapped.current_scenario for _ in range(12)]
    assert order(tuned) == order(scratch) and len(set(order(tuned))) > 3


def test_critic_warm_up_leaves_the_actor_alone_then_ppo_updates_it(short_config, mix, monkeypatch, tmp_path):
    from autobot.sim.cli import checkpoint_observation_metadata, load_checkpoint
    monkeypatch.setattr(pilot, "CONFIG", short_config)
    observations = np.random.default_rng(1).uniform(-1, 1, size=(50, 19)).astype(np.float32)
    monkeypatch.setattr(pilot, "saved_observations", lambda: (observations, {}))
    clone = pilot.build_model("scratch", 5, mix[0]); clone.save(tmp_path / "clone")
    out = tmp_path / "run"
    pilot.train(argparse.Namespace(arm="finetune", seed=17, clone=str(tmp_path / "clone.zip"), output=str(out), total_steps=1024, warmup_steps=512))
    rows = [json.loads(line) for line in (out / "metrics.jsonl").read_text(encoding="utf-8").splitlines()]
    warm = next(row for row in rows if row["type"] == "critic_warmup_complete")
    assert warm["timesteps"] == 512 and warm["actor_unchanged_by_warmup"]
    assert [row["timesteps"] for row in rows if row["type"] == "checkpoint"] == [0, 512, 1024]
    updates = [row for row in rows if row["type"] == "update"]
    assert [row["actor_frozen"] for row in updates] == [True, True, False] and all("explained_variance" in row for row in updates)
    assert updates[0]["approx_kl"] == 0.0 and updates[0]["clip_fraction"] == 0.0     # frozen actor: the policy did not move
    episodes = [row for row in rows if row["type"] == "train_episode"]
    assert episodes and {row["group"] for row in episodes} <= {"turn_needed", "ordinary"}
    load = lambda step: load_checkpoint(out / "checkpoints" / f"step_{step:07d}.zip", device="cpu")[0].policy
    actor = lambda policy: pilot.flat(pilot.actor_modules(policy)); critic = lambda policy: pilot.flat(pilot.critic_modules(policy))
    start, warmed, final = load(0), load(512), load(1024)
    assert np.array_equal(actor(start), actor(clone.policy)) and np.array_equal(actor(warmed), actor(clone.policy))
    assert not np.array_equal(critic(warmed), critic(start)) and not np.array_equal(actor(final), actor(clone.policy))
    metadata = checkpoint_observation_metadata(out / "final_model.zip")
    assert metadata["previous_action_encoding"] == "masked_zero_v1" and metadata["action_profile"] == "experimental5_v1"
    config = json.loads((out / "effective_config.json").read_text(encoding="utf-8"))
    assert config["gamma"] == 0.99 and config["reward_profile"] == "outcome_only" and config["import_check"]["identical"]
    assert config["ppo_hyperparameters"]["ent_coef"] == 0.0 and config["critic_warmup_timesteps"] == 512
    with pytest.raises(SystemExit, match="Refusing to overwrite"):
        pilot.train(argparse.Namespace(arm="finetune", seed=17, clone=str(tmp_path / "clone.zip"), output=str(out), total_steps=1024, warmup_steps=512))


def test_scratch_and_cold_critic_arms_update_the_actor_from_the_first_rollout(short_config, mix, monkeypatch, tmp_path):
    monkeypatch.setattr(pilot, "CONFIG", short_config)
    pilot.train(argparse.Namespace(arm="scratch", seed=23, clone=None, output=str(tmp_path / "scratch"), total_steps=512, warmup_steps=256))
    rows = [json.loads(line) for line in (tmp_path / "scratch" / "metrics.jsonl").read_text(encoding="utf-8").splitlines()]
    assert not any(row["type"] == "critic_warmup_complete" for row in rows)
    assert [row["actor_frozen"] for row in rows if row["type"] == "update"] == [False]
    assert json.loads((tmp_path / "scratch" / "effective_config.json").read_text(encoding="utf-8"))["critic_warmup_timesteps"] == 0


def test_recommendation_needs_the_turn_skill_at_both_pits_and_a_clear_ordinary_gain():
    clone = table(158, 156, 0, 100, 0); advance = table(0, 0, 12, 0, 3)
    good = table(150, 150, 40, 20, 8)
    assert pilot.recommendation(good, clone, advance)["recommend"]
    one_pit = pilot.recommendation(table(160, 100, 40, 20, 8), clone, advance)
    assert not one_pit["recommend"] and not one_pit["checks"]["keeps_turning_bottom_right"] and one_pit["checks"]["keeps_turning_top_left"]
    small_gain = pilot.recommendation(table(150, 150, 20, 20, 8), clone, advance)      # +8 over Always-Advance is under 10% of 144
    assert not small_gain["recommend"] and small_gain["checks"]["beats_clone_on_new_confirmation"] and not small_gain["checks"]["beats_advance_on_new_confirmation"]
    assert not pilot.recommendation(table(150, 150, 40, 120, 8), clone, advance)["recommend"]     # more own falls than the clone
    assert not pilot.recommendation(table(150, 150, 40, 20, 3), clone, advance)["recommend"]      # no gain on the earlier fresh starts
    assert not pilot.recommendation(table(0, 0, 40, 20, 8), table(0, 0, 0, 100, 0), advance)["recommend"]   # nothing to preserve is not "kept"
