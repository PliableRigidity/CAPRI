"""Heading ladder: start geometry, split hygiene, rung gating, and that LadderEnv leaves the environment alone."""
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import ordinary_start_diagnostic as diagnostic  # noqa: E402

from autobot.sim import ladder as hl
from autobot.sim.env import AutoBotEnv, ScriptedBaseline

RUNGS = (16, 24, 34, 45, 60, 90)
COMMON = dict(opponent_model="last_year", opponent_mode="scripted", reward_profile="outcome_only",
              action_profile="experimental5_v1", previous_action_encoding="masked_zero_v1")
LADDER_FILE = Path("config/heading_ladder_v1.json")


@pytest.fixture(scope="module")
def ladder():
    return hl.build_ladder(RUNGS, warmup_offset_deg=10, holdout_offsets_deg=[20, 29, 39, 52, 75, 120])


@pytest.fixture(scope="module")
def short_config(tmp_path_factory):
    """config/sim.yaml with a 4 s horizon, so a failed push does not cost a 120 s episode."""
    config = yaml.safe_load(Path("config/sim.yaml").read_text(encoding="utf-8")); config["simulation"]["episode_seconds"] = 4.0
    path = tmp_path_factory.mktemp("ladder") / "sim.yaml"; path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return str(path)


class TurnThenAdvance:
    """Test double: a fixed number of turn decisions, then advance."""
    def __init__(self, turn_action, decisions): self.turn_action = turn_action; self.decisions = decisions; self.count = 0

    def predict(self, observation, deterministic=True):
        self.count += 1
        return np.array(self.turn_action if self.count <= self.decisions else 1), None


def all_starts(ladder):
    return [start for rung in ladder["rungs"] for split in ("train", "validation") for start in rung[split]] + ladder["holdout"]


def test_ladder_starts_put_red_between_blue_and_the_pit_with_the_stated_offset(ladder):
    assert {name: (tuple(centre), tuple(inward)) for name, (centre, inward) in diagnostic.PITS.items()} == \
           {name: (tuple(centre), tuple(inward)) for name, (centre, inward) in hl.PITS.items()}
    assert hl.STANDOFF_M == diagnostic.STANDOFF_M
    for start in all_starts(ladder):
        pit = np.asarray(hl.PITS[start["pit"]][0]); blue = np.asarray(start["player"][:2]); red = np.asarray(start["opponent"][:2])
        assert np.linalg.norm(red - pit) == pytest.approx(start["distance_m"], abs=1e-5)
        assert np.linalg.norm(blue - red) == pytest.approx(hl.STANDOFF_M, abs=1e-5)
        line = math.atan2(pit[1] - red[1], pit[0] - red[0])
        assert math.atan2(red[1] - blue[1], red[0] - blue[0]) == pytest.approx(line, abs=1e-4)
        assert math.degrees(hl.wrap(start["player"][2] - line)) == pytest.approx(start["heading_offset_deg"], abs=1e-3)
        assert abs(hl.wrap(start["opponent"][2] - line)) == pytest.approx(math.pi, abs=1e-4)      # red faces blue
        assert 0.05 < blue[0] < 1.10 and 0.05 < blue[1] < 1.10


def test_splits_are_disjoint_and_the_holdout_covers_both_pits_new_distances_and_new_angles(ladder):
    assert [rung["abs_heading_offset_deg"] for rung in ladder["rungs"]] == [10, *RUNGS]
    assert [rung["warmup"] for rung in ladder["rungs"]] == [True] + [False] * len(RUNGS)
    train = [start for rung in ladder["rungs"] for start in rung["train"]]
    validation = [start for rung in ladder["rungs"] for start in rung["validation"]]
    key = lambda start: (start["pit"], start["distance_m"], start["heading_offset_deg"])
    assert not {key(s) for s in train} & {key(s) for s in validation}
    assert not ({key(s) for s in train} | {key(s) for s in validation}) & {key(s) for s in ladder["holdout"]}
    for rung in ladder["rungs"]:
        assert len(rung["train"]) == 10 and len(rung["validation"]) == 8
        assert {s["pit"] for s in rung["train"]} == set(hl.PITS) == {s["pit"] for s in rung["validation"]}
        assert {math.copysign(1, s["heading_offset_deg"]) for s in rung["train"]} == {1.0, -1.0}
        assert not {s["distance_m"] for s in rung["train"]} & {s["distance_m"] for s in rung["validation"]}
        # Two combinations of trained levels are left out of training for this rung, one on each pit.
        left_out = [s for s in ladder["holdout"] if s["holdout_kind"] == "combination" and abs(s["heading_offset_deg"]) == rung["abs_heading_offset_deg"]]
        assert sorted(s["pit"] for s in left_out) == ["bottom_right", "top_left"]
        assert all(s["distance_m"] in ladder["train_distances_m"] for s in left_out)
        assert left_out[0]["heading_offset_deg"] == -left_out[1]["heading_offset_deg"] and left_out[0]["distance_m"] != left_out[1]["distance_m"]
    level = [s for s in ladder["holdout"] if s["holdout_kind"] == "level"]
    seen_distances = set(ladder["train_distances_m"]) | set(ladder["validation_distances_m"])
    seen_offsets = {rung["abs_heading_offset_deg"] for rung in ladder["rungs"]}
    assert {s["pit"] for s in level} == set(hl.PITS)
    assert not {s["distance_m"] for s in level} & seen_distances and not {abs(s["heading_offset_deg"]) for s in level} & seen_offsets
    assert len(level) == 6 * 2 * 4 * 2 and {math.copysign(1, s["heading_offset_deg"]) for s in level} == {1.0, -1.0}


def test_validate_ladder_rejects_shared_starts_and_unordered_rungs(ladder):
    shared = json.loads(json.dumps(ladder)); shared["holdout"].append(dict(shared["rungs"][2]["train"][0]))
    with pytest.raises(ValueError, match="more than one split"): hl.validate_ladder(shared)
    with pytest.raises(ValueError, match="increasing"): hl.build_ladder([24, 16])
    wrong = json.loads(json.dumps(ladder)); wrong["rungs"][1]["train"][0]["heading_offset_deg"] = 45.0
    with pytest.raises(ValueError, match="rung's heading offset"): hl.validate_ladder(wrong)


def test_sampling_stays_at_or_below_the_frontier_and_reviews_cleared_rungs(ladder):
    assert all(hl.sample_ladder_start(ladder, 0, np.random.default_rng(seed))[0] == 0 for seed in range(50))
    draws = [hl.sample_ladder_start(ladder, 3, np.random.default_rng(seed)) for seed in range(2000)]
    rungs = [rung for rung, _ in draws]
    assert set(rungs) == {0, 1, 2, 3}
    assert rungs.count(3) / len(rungs) == pytest.approx(1 - ladder["review_fraction"], abs=0.04)
    assert all(start in ladder["rungs"][rung]["train"] for rung, start in draws)
    assert hl.sample_ladder_start(ladder, 3, np.random.default_rng(5)) == hl.sample_ladder_start(ladder, 3, np.random.default_rng(5))


def test_gate_needs_a_clear_margin_over_always_advance_and_consecutive_passes():
    assert hl.gate_passes(12, 0, 16) and not hl.gate_passes(11, 0, 16)            # 75% of the validation episodes
    assert not hl.gate_passes(14, 12, 16) and hl.gate_passes(16, 12, 16)          # and 25% of them more than Always-Advance
    assert not hl.gate_passes(16, 16, 16) and hl.gate_passes(16, 16, 16, warmup=True)
    assert not hl.gate_passes(0, 0, 0)
    gate = hl.RungGate(required=2)
    assert [gate.update(passed) for passed in (True, False, True, True, True)] == [False, False, False, True, False]


def test_ladder_env_only_chooses_the_start(ladder, short_config):
    env = hl.LadderEnv(ladder, short_config, "sim_debug", ladder_stream=3, **COMMON)
    with pytest.raises(ValueError): hl.LadderEnv(ladder, short_config, "sim_debug", training_curriculum={"stages": []}, **COMMON)
    observation, info = env.reset(seed=11)
    assert info["curriculum_stage"] == "rung_0" and info["scenario_group"] == "rung_0"
    start = next(s for s in ladder["rungs"][0]["train"] if s["name"] == info["scenario"])
    plain = AutoBotEnv(short_config, "sim_debug", **COMMON)
    reference, _ = plain.reset(seed=11, options={"scenario": start})
    assert np.array_equal(observation, reference) and env.observation_space == plain.observation_space
    for action in (3, 3, 1, 1, 1, 4, 1, 2, 0, 1) * 3:
        a = env.step(action); b = plain.step(action)
        assert np.array_equal(a[0], b[0]) and a[1:4] == b[1:4]
        assert a[4]["scenario"] == start["name"] and a[4]["scenario_group"] == "rung_0"
    env.set_rung(4)
    sampled = {env.reset()[1]["scenario_group"] for _ in range(40)}
    assert "rung_4" in sampled and sampled <= {f"rung_{index}" for index in range(5)}
    assert env.reset()[1]["curriculum_stage"] == "rung_4"
    with pytest.raises(ValueError): env.set_rung(len(ladder["rungs"]))
    # An explicit scenario (validation, replay) bypasses the ladder and does not consume its episode counter.
    count = env.ladder_episode; _, info = env.reset(options={"scenario": ladder["holdout"][0]})
    assert info["scenario"] == ladder["holdout"][0]["name"] and env.ladder_episode == count


def test_rollout_reports_whether_a_win_involved_an_initial_turn(ladder, short_config):
    env = AutoBotEnv(short_config, "sim_debug", 0, **COMMON)
    warmup = next(s for s in ladder["rungs"][0]["validation"] if s["pit"] == "top_left" and s["heading_offset_deg"] > 0)
    crooked = next(s for s in ladder["rungs"][2]["validation"] if s["pit"] == "top_left" and s["heading_offset_deg"] > 0 and s["distance_m"] == warmup["distance_m"])
    straight = hl.ladder_rollout(ScriptedBaseline(), env, warmup, seed=5)
    assert straight["outcome"] == "wins" and not straight["initial_turn"] and straight["turn_decisions_before_contact"] == 0
    assert straight["heading_error_start_deg"] == pytest.approx(10, abs=0.01) and straight["first_tactic"] == "advance"
    missed = hl.ladder_rollout(ScriptedBaseline(), env, crooked, seed=5)
    assert missed["outcome"] != "wins" and not missed["initial_turn"] and missed["heading_error_start_deg"] == pytest.approx(24, abs=0.01)
    turned = hl.ladder_rollout(TurnThenAdvance(4, 5), env, crooked, seed=5)        # +24 deg is anticlockwise of the line: turn right
    assert turned["outcome"] == "wins" and turned["initial_turn"] and turned["turn_decisions_before_contact"] == 5
    assert turned["turned_toward_push_line_before_contact_deg"] > 12 and abs(turned["heading_error_at_first_contact_deg"]) < 12
    wrong_way = hl.ladder_rollout(TurnThenAdvance(3, 5), env, crooked, seed=5)
    assert wrong_way["outcome"] != "wins" and not wrong_way["initial_turn"] and wrong_way["turned_toward_push_line_before_contact_deg"] < 0
    # Reusing one environment gives the same episode as a fresh one.
    fresh = hl.ladder_rollout(TurnThenAdvance(4, 5), AutoBotEnv(short_config, "sim_debug", 0, **COMMON), crooked, seed=5)
    assert fresh == turned
    summary = hl.summarise_episodes([straight, missed, turned, wrong_way])
    assert (summary["episodes"], summary["wins"], summary["wins_with_initial_turn"]) == (4, 2, 1)


def test_greedy_actor_matches_the_model_it_copies(short_config):
    from stable_baselines3 import PPO
    env = AutoBotEnv(short_config, "sim_debug", 0, **COMMON)
    for kwargs in ({}, {"policy_kwargs": {"net_arch": []}}):
        model = PPO("MlpPolicy", env, device="cpu", seed=4, n_steps=8, batch_size=8, **kwargs); fast = hl.GreedyActor(model)
        observations = np.random.default_rng(1).uniform(-1, 1, size=(300, env.observation_space.shape[0])).astype(np.float32)
        assert [int(fast.predict(o)[0]) for o in observations] == [int(model.predict(o, deterministic=True)[0]) for o in observations]
    with pytest.raises(ValueError): fast.predict(observations[0], deterministic=False)


def test_written_ladder_was_verified_to_separate_advance_from_the_controller():
    if not LADDER_FILE.is_file(): pytest.skip("heading ladder has not been generated")
    written = hl.validate_ladder(json.loads(LADDER_FILE.read_text(encoding="utf-8")))
    assert written["rungs"][0]["warmup"] and all(rung["separation_verified"] for rung in written["rungs"])
    for rung in written["rungs"]:
        for split in ("validation", "train"):
            advance = rung["reference"][split]["always_advance"]; controller = rung["reference"][split]["turn_then_advance_controller"]
            assert controller["wins"] >= 0.9 * controller["episodes"]
            assert advance["wins"] == (advance["episodes"] if rung["warmup"] else 0)
    flat = json.loads(LADDER_FILE.with_name(LADDER_FILE.stem + "_scenarios.json").read_text(encoding="utf-8"))
    assert flat == [start for rung in written["rungs"] for start in rung["validation"]] + written["holdout"]


def test_single_pit_ladder_trains_on_one_pit_and_keeps_the_two_pit_holdout(ladder):
    one = hl.build_ladder(RUNGS, warmup_offset_deg=10, holdout_offsets_deg=[20, 29, 39, 52, 75, 120], pits=["bottom_right"])
    assert one["training_pits"] == ["bottom_right"] and ladder["training_pits"] == ["top_left", "bottom_right"]
    assert {s["pit"] for rung in one["rungs"] for s in rung["train"] + rung["validation"]} == {"bottom_right"}
    assert one["holdout"] == ladder["holdout"]
    for single, both in zip(one["rungs"], ladder["rungs"]):
        assert len(single["train"]) == 5 and len(single["validation"]) == 4
        assert all(s in both["train"] for s in single["train"]) and all(s in both["validation"] for s in single["validation"])
    with pytest.raises(ValueError): hl.build_ladder(RUNGS, pits=["middle"])


def test_the_two_pits_are_half_turn_images_with_negated_pose_channels(short_config):
    """The fact behind the one-pit policies: the same situation at the other pit is the negated pose observation."""
    from autobot.sim.env import previous_action_slice
    pose = slice(previous_action_slice("experimental5_v1").stop, None)
    envs = {pit: AutoBotEnv(short_config, "sim_debug", 0, **COMMON) for pit in hl.PITS}
    observations = {pit: env.reset(seed=1, options={"scenario": hl.ladder_start(pit, 0.46, 24, "probe")})[0] for pit, env in envs.items()}
    assert np.allclose(observations["top_left"][pose], -observations["bottom_right"][pose], atol=1e-4)
    assert np.abs(observations["top_left"][pose]).min() > 0.05 and not np.any(observations["top_left"][6:pose.start])
    # The same tactics then win at both pits: it really is the same situation.
    outcomes = {pit: hl.ladder_rollout(TurnThenAdvance(4, 5), env, hl.ladder_start(pit, 0.46, 24, "probe"), seed=1)["outcome"] for pit, env in envs.items()}
    assert outcomes == {"top_left": "wins", "bottom_right": "wins"}


def test_imitation_recorder_labels_every_state_with_the_controller_choice(short_config, monkeypatch):
    import controller_imitation as imitation
    monkeypatch.setattr(imitation, "MIN_GRADIENT_STEPS", 40)
    env = AutoBotEnv(short_config, "sim_debug", 0, **COMMON); recorder = imitation.Recorder(env)
    start = hl.ladder_start("top_left", 0.46, 24, "probe")
    observations, labels, taken = recorder.episode(None, start, seed=3)
    assert np.array_equal(labels, taken) and observations.shape == (len(labels), 19) and len(labels) < 150      # the controller wins quickly
    assert labels[0] == 4 and set(labels.tolist()) <= {1, 3, 4} and not np.any(observations[:, 6:13])
    capped = recorder.episode(ScriptedBaseline(), start, seed=3, step_cap=20)
    assert len(capped[1]) == 20 and set(capped[2].tolist()) == {1} and capped[1][0] == 4                     # driver advances, labels stay the controller's
    from stable_baselines3 import PPO
    model = PPO("MlpPolicy", env, device="cpu", seed=2, n_steps=8, batch_size=8)
    value_before = [parameter.detach().clone() for parameter in model.policy.value_net.parameters()]
    before = imitation.agreement(model, observations, labels, env.tactics)["agreement"]
    fitted = imitation.fit(model, observations, labels, epochs=1, seed=0)
    assert fitted["agreement"] >= max(before, 0.9)
    assert all(np.array_equal(a.numpy(), b.detach().numpy()) for a, b in zip(value_before, model.policy.value_net.parameters()))
