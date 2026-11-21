"""Harder red opponents: the original modes stay untouched, styles are seeded and distinct, outcomes are attributed fairly."""
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import heading_ladder as experiment  # noqa: E402
import opponent_pool as pool_script  # noqa: E402

from autobot.sim import opponents as op
from autobot.sim.env import AutoBotEnv, ScriptedBaseline, StopBaseline, scripted_opponent_drive

COMMON = dict(opponent_model="last_year", opponent_mode="scripted", reward_profile="outcome_only",
              action_profile="experimental5_v1", previous_action_encoding="masked_zero_v1")
START = {"name": "probe", "player": [0.575, 0.18, 1.570796], "opponent": [0.575, 0.97, -1.570796]}


@pytest.fixture(scope="module")
def short_config(tmp_path_factory):
    config = yaml.safe_load(experiment.CONFIG.read_text(encoding="utf-8")); config["simulation"]["episode_seconds"] = 6.0
    path = tmp_path_factory.mktemp("opponents") / "sim.yaml"; path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return str(path)


def case(style, combo="low_low", seed=5, start=START):
    return {"name": f"probe_{style}", "style": style, "family": "default_spawn", "combo": combo, "seed": 400 + seed, "driver_seed": 900 + seed,
            "start": start, "parameters": op.resolve_parameters(style, combo, np.random.default_rng(seed))}


class TurnLeft:
    def predict(self, observation, deterministic=True): return np.array(3), None


def test_without_an_opponent_the_styled_env_is_the_original_env(short_config):
    plain = AutoBotEnv(short_config, "sim_debug", 0, **COMMON); styled = op.StyledOpponentEnv(short_config, "sim_debug", 0, **COMMON)
    first, _ = plain.reset(seed=11, options={"scenario": START}); second, info = styled.reset(seed=11, options={"scenario": START})
    assert np.array_equal(first, second) and info["resolved_opponent"] is None and styled.red_driver is None
    for step in range(120):
        expected = tuple(scripted_opponent_drive(styled.sim))                               # what the original mode commands from this state
        a = plain.step(step % 5); b = styled.step(step % 5)
        assert np.array_equal(a[0], b[0]) and a[4]["opponent_drive"] == b[4]["opponent_drive"] == expected
    again, _ = styled.reset(seed=11, options={"scenario": START, "opponent": {"style": "scripted"}})
    assert np.array_equal(again, first) and styled.red_driver is None


def test_parameters_are_seeded_stay_in_range_and_follow_their_combination():
    for style, spec in op.SPECS.items():
        for combo in op.COMBOS:
            first = op.resolve_parameters(style, combo, np.random.default_rng(3)); again = op.resolve_parameters(style, combo, np.random.default_rng(3))
            assert first == again != op.resolve_parameters(style, combo, np.random.default_rng(4))
            assert all(low <= first[name] <= high for name, (low, high) in spec["ranges"].items()) and op.combo_of(style, first) == combo
    assert op.resolve_parameters("curved_approach", "low_low", np.random.default_rng(0))["side"] in (-1, 1)
    with pytest.raises(ValueError): op.resolve_parameters("teleport", "low_low", np.random.default_rng(0))
    with pytest.raises(ValueError): op.resolve_parameters("stop_and_go", "middle", np.random.default_rng(0))


def test_wheel_mixing_matches_the_simulator_convention_and_its_limits():
    assert op.mix(1.0, 0.0) == (-1.0, -1.0) and op.mix(0.0, 0.7) == (0.7, -0.7)          # as tactic_to_drive: advance, turn_left
    left, right = op.mix(1.0, 0.6)
    assert max(abs(left), abs(right)) == 1.0 and right < left < 0                         # forward left curve, scaled into the limit
    assert op.unmix(op.mix(0.5, 0.2)) == pytest.approx((0.5, 0.2)) and op.unmix((-1.0, -1.0)) == (1.0, 0.0)


def test_every_style_is_reproducible_stays_inside_the_drive_limits_and_never_reaches_the_observation(short_config):
    env = op.StyledOpponentEnv(short_config, "sim_debug", 0, **COMMON)
    scripted_reset, _ = env.reset(seed=405, options={"scenario": START})
    for style in op.ALL_STYLES:
        probe = case(style)
        first = op.case_rollout(TurnLeft(), env, probe, keep_commands=True); again = op.case_rollout(TurnLeft(), env, probe, keep_commands=True)
        assert first["red_commands"] == again["red_commands"] and first["outcome"] == again["outcome"]
        assert all(-1.0 <= value <= 1.0 for command in first["red_commands"] for value in command)
        other = op.case_rollout(TurnLeft(), env, case(style, seed=6), keep_commands=True)
        assert other["red_commands"] != first["red_commands"]
        observation, info = env.reset(seed=405, options={"scenario": START, "opponent": probe})
        assert observation.shape == (19,) and np.array_equal(observation, scripted_reset)          # same start and seed: nothing about red's style is visible
        assert info["resolved_opponent"]["parameters"] == probe["parameters"]


def test_persistent_follower_keeps_driving_where_scripted_yields(short_config):
    env = op.StyledOpponentEnv(short_config, "sim_debug", 0, **COMMON)
    close = {"name": "close", "player": [0.575, 0.50, 1.570796], "opponent": [0.575, 0.62, -1.570796]}       # 12 cm apart, facing each other
    env.reset(seed=1, options={"scenario": close})
    assert scripted_opponent_drive(env.sim) == (0.0, 0.0)                                                    # inside the old yield distance
    follower = case("persistent_follower", "high_high"); env.reset(seed=1, options={"scenario": close, "opponent": follower})
    command = env.red_driver.drive(env.sim)
    assert command == pytest.approx((-follower["parameters"]["speed"],) * 2) and follower["parameters"]["speed"] > 0.775 > 0.4
    result = op.case_rollout(StopBaseline(), env, dict(follower, start=close))
    assert result["red_driving"]["close_range_stopped_share"] == 0.0 and result["red_driving"]["close_range_driving_forward_share"] > 0.8


def test_styles_have_their_own_driving_patterns(short_config):
    env = op.StyledOpponentEnv(short_config, "sim_debug", 0, **COMMON)
    pattern = {style: op.case_rollout(TurnLeft(), env, case(style, "high_high"))["red_driving"] for style in op.ALL_STYLES}
    scripted = op.case_rollout(TurnLeft(), env, {"name": "s", "style": "scripted", "seed": 405, "start": START})["red_driving"]
    assert scripted["steering_while_moving_share"] == 0.0 and scripted["mean_forward_command_when_driving_forward"] == pytest.approx(0.4)
    assert pattern["persistent_follower"]["mean_forward_command_when_driving_forward"] > 0.6
    assert pattern["curved_approach"]["steering_while_moving_share"] > 0.5 > pattern["persistent_follower"]["steering_while_moving_share"]
    paced = op.case_rollout(TurnLeft(), env, case("stop_and_go", "high_low"))["red_driving"]           # short holds: several segments in 6 s
    assert paced["stopped_share"] + paced["reversing_share"] > 0.05 and pattern["persistent_follower"]["stopped_share"] == 0.0
    assert pattern["persistent_follower"]["reversing_share"] == 0.0
    assert pattern["pit_pusher"]["steering_while_moving_share"] > 0.5
    left = case("curved_approach"); right = dict(left, parameters=dict(left["parameters"], side=-left["parameters"]["side"]))
    a = op.case_rollout(StopBaseline(), env, left, keep_commands=True); b = op.case_rollout(StopBaseline(), env, right, keep_commands=True)
    turn = lambda episode: np.mean([op.unmix(command)[1] for command in episode["red_commands"][:60]])
    assert turn(a) * turn(b) < 0                                                                              # the two sides steer opposite ways


def test_stop_and_go_pauses_backs_away_and_resumes(short_config):
    env = op.StyledOpponentEnv(short_config, "sim_debug", 0, **COMMON); probe = case("stop_and_go", "high_low")
    env.reset(seed=1, options={"scenario": START, "opponent": probe}); kinds = []
    for _ in range(290):
        env.step(0); kinds.append(env.red_driver.state["kind"])
    runs = [kinds[0]] + [kind for previous, kind in zip(kinds, kinds[1:]) if kind != previous]
    assert {"go", "pause"} <= set(runs) and all(a != b or a == "go" for a, b in zip(runs, runs[1:]))
    assert all(kind == "go" for previous, kind in zip(runs, runs[1:]) if previous in ("pause", "back"))    # it always resumes
    shortest = min(sum(1 for _ in group) for group in _groups(kinds)[:-1])
    assert shortest * 0.02 >= 0.5 * probe["parameters"]["move_hold_s"] - 0.021                              # no single-tick twitches


def _groups(values):
    groups = [[values[0]]]
    for value in values[1:]:
        if value == groups[-1][-1]: groups[-1].append(value)
        else: groups.append([value])
    return groups


def test_reactive_changes_waits_for_its_delay_and_holds_each_manoeuvre(short_config):
    env = op.StyledOpponentEnv(short_config, "sim_debug", 0, **COMMON); probe = case("reactive_changes", "high_high")
    delay = probe["parameters"]["reaction_delay_s"]; hold = probe["parameters"]["move_hold_s"]
    env.reset(seed=1, options={"scenario": START, "opponent": probe}); rows = []
    for _ in range(290):
        env.step(3)                                                                                          # blue keeps turning on the spot
        state = env.red_driver.state; rows.append((float(env.sim.data.time), state["mode"], state["pending"], state["commit_until"]))
    assert rows[0][1] == "follow" and rows[int(0.7 * delay / 0.02) - 2][1] == "follow"                    # nothing happens before the delay is up
    started = [(time, mode, commit) for (_, before, _, _), (time, mode, _, commit) in zip(rows, rows[1:]) if mode != before and mode != "follow"]
    assert started and {mode for _, mode, _ in started} <= {"sidestep", "retreat", "charge", "cut_in"}
    assert started[0][0] >= 0.7 * delay - 0.021
    for (time, _, commit), (next_time, _, _) in zip(started, started[1:]):
        assert commit - time >= 0.8 * hold - 0.021 and next_time >= commit - 0.021                         # committed for the whole hold
    pending = [(time, due) for time, _, due, _ in rows if due is not None]
    assert all(0.7 * delay - 0.041 <= due - time <= 1.3 * delay + 0.021 for time, due in pending[:1])


def test_red_does_not_drive_itself_into_a_pit(short_config):
    env = op.StyledOpponentEnv(short_config, "sim_debug", 0, **COMMON)
    follower = case("persistent_follower"); follower["parameters"]["pit_margin_m"] = 0.03
    rim = {"name": "rim", "player": [0.575, 0.18, 1.570796], "opponent": [0.25, 0.90, 2.356194]}          # red faces the top-left pit from its rim
    env.reset(seed=1, options={"scenario": rim, "opponent": follower}); driver = env.red_driver; red = env.sim.truth(1)
    forward, turn = driver._guard(env.sim, red, 0.8, 0.0)
    assert forward == 0.0 and abs(turn) == pytest.approx(0.6)                                                # turns towards the centre instead of advancing
    assert driver._guard(env.sim, red, -0.8, 0.1) == (-0.8, 0.1) and driver._guard(env.sim, red, 0.0, 0.5) == (0.0, 0.5)   # backing away from it is fine
    away = {"name": "rim_back", "player": [0.575, 0.18, 1.570796], "opponent": [0.25, 0.90, -0.785398]}   # red's back is to the pit
    env.reset(seed=1, options={"scenario": away, "opponent": follower}); red = env.sim.truth(1)
    assert env.red_driver._guard(env.sim, red, -0.5, 0.0)[0] == 0.0 and env.red_driver._guard(env.sim, red, 0.5, 0.0) == (0.5, 0.0)
    careless = dict(follower, parameters=dict(follower["parameters"], pit_margin_m=-0.5))
    env.reset(seed=1, options={"scenario": rim, "opponent": careless})
    assert env.red_driver._guard(env.sim, env.sim.truth(1), 0.8, 0.0) == (0.8, 0.0)                         # the margin is a per-case parameter


def test_a_style_can_be_sampled_at_reset_and_the_draw_is_saved(short_config):
    assert op.sample_opponent(7, 3) == op.sample_opponent(7, 3) != op.sample_opponent(7, 4)
    draws = [op.sample_opponent(7, episode) for episode in range(60)]
    assert {draw["style"] for draw in draws} == set(op.STYLES) and {draw["combo"] for draw in draws} == set(op.SEEN_COMBOS)
    env = op.StyledOpponentEnv(short_config, "sim_debug", 0, opponent_sampler={"seed": 7}, **COMMON)
    infos = [env.reset(seed=1, options={"scenario": START})[1]["resolved_opponent"] for _ in range(3)]
    assert infos == [{key: op.sample_opponent(7, episode)[key] for key in ("style", "combo", "parameters", "driver_seed")} for episode in range(3)]
    replay = env.reset(seed=1, options={"scenario": START, "opponent": infos[1]})[1]["resolved_opponent"]
    assert replay == infos[1] and env.opponent_episode == 3                                                  # an explicit opponent does not consume a draw


def test_a_red_fall_without_recent_contact_is_a_self_fall_not_a_win():
    assert op.end_outcome("Pit fall", 1, True) == "wins" and op.end_outcome("Pit fall", 1, False) == "red_self_falls"
    assert op.end_outcome("Pit fall", 0, True) == op.end_outcome("Pit fall", 0, False) == "own_falls"
    assert op.end_outcome("Timeout", None, True) == "timeouts" and op.end_outcome("Simultaneous pit fall", None, True) == "draws"
    assert op.end_outcome("Simulation bug: outer-wall breach", 1, False) == "simulation_bugs"
    episodes = {"always_stop": [{"case": "a", "outcome": "red_self_falls"}, {"case": "b", "outcome": "timeouts"}],
                "blue": [{"case": "a", "outcome": "wins"}, {"case": "b", "outcome": "wins"}]}
    adjusted = pool_script.credited(episodes)["blue"]
    assert [episode["outcome"] for episode in adjusted] == ["red_self_falls", "wins"]                       # the matched Always-Stop rule
    assert [episode["red_also_falls_against_always_stop"] for episode in adjusted] == [True, False]


def test_pool_splits_share_no_start_and_keep_the_held_out_pieces_in_the_test_split():
    from autobot.sim.core import load_config
    pool = pool_script.load_pool()
    assert pool_script.build_pool(load_config(str(experiment.CONFIG))["arena"]) == pool
    cases = pool["splits"]; assert {split: len(rows) for split, rows in cases.items()} == {"train": 192, "validation": 80, "test": 240}
    names = {split: {row["start"]["name"] for row in rows} for split, rows in cases.items()}
    assert not names["train"] & names["validation"] and not names["train"] & names["test"] and not names["validation"] & names["test"]
    combos = {split: {row["combo"] for row in rows if row["style"] != "scripted"} for split, rows in cases.items()}
    assert combos["train"] == combos["validation"] == set(op.SEEN_COMBOS) and op.UNSEEN_COMBO in combos["test"]
    styles = {split: {row["style"] for row in rows} for split, rows in cases.items()}
    assert "pit_pusher" in styles["test"] and "pit_pusher" not in styles["train"] | styles["validation"] and "scripted" not in styles["train"]
    test = [row for row in cases["test"] if row["style"] != "scripted"]
    assert sum(row["combo"] == op.UNSEEN_COMBO for row in test) == len(test) // 2
    assert len({row["seed"] for rows in cases.values() for row in rows}) == 512
    for row in cases["test"][:3]:                                                                            # same starts for every style of a split
        assert {other["start"]["name"] for other in cases["test"] if other["style"] == row["style"]} == names["test"]
    leaked = dict(pool, splits=dict(cases, validation=cases["validation"] + [dict(cases["test"][50], split="validation")]))
    with pytest.raises(ValueError): pool_script.check_pool(leaked)


def test_teacher_rule_needs_a_clear_gain_on_two_styles_without_extra_own_falls():
    def summary(teacher, reference):
        cell = lambda wins, falls: {"episodes": 64, "wins": wins, "own_falls": falls, "timeouts": 64 - wins - falls, "red_self_falls": 0}
        return {"all": {"teacher": {"by_style": {style: cell(*teacher[style]) for style in op.STYLES}},
                        "reference": {"by_style": {style: cell(*reference[style]) for style in op.STYLES}}}}
    flat = {style: (5, 0) for style in op.STYLES}
    assert not pool_script.teacher_decision(summary(flat, flat))["train"]
    one = dict(flat, stop_and_go=(14, 0))
    assert pool_script.teacher_decision(summary(one, flat))["viable_styles"] == ["stop_and_go"] and not pool_script.teacher_decision(summary(one, flat))["train"]
    two = dict(one, reactive_changes=(12, 1))
    assert pool_script.teacher_decision(summary(two, flat))["train"]
    falls = dict(one, reactive_changes=(12, 9))                                                              # 9 of 64 is over 10%
    assert pool_script.teacher_decision(summary(falls, flat))["viable_styles"] == ["stop_and_go"]
    assert not pool_script.teacher_decision(summary(dict(flat, stop_and_go=(11, 0)), flat))["viable_styles"]  # +6 of 64 is under 10%
