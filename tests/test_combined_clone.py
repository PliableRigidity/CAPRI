"""Combined behaviour cloning: start hygiene, demonstrations, batch sampling, fitting, and the two fixed rules."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import clone_finetune as pilot  # noqa: E402
import combined_clone as combined  # noqa: E402
import controller_imitation as imitation  # noqa: E402
import heading_ladder as experiment  # noqa: E402
import ordinary_start_diagnostic as diagnostic  # noqa: E402

from autobot.sim import ladder as hl
from autobot.sim.env import AutoBotEnv, ScriptedBaseline

COMMON = dict(opponent_model="last_year", opponent_mode="scripted", reward_profile="outcome_only",
              action_profile="experimental5_v1", previous_action_encoding="masked_zero_v1")


@pytest.fixture(scope="module")
def short_config(tmp_path_factory):
    config = yaml.safe_load(experiment.CONFIG.read_text(encoding="utf-8")); config["simulation"]["episode_seconds"] = 2.0
    path = tmp_path_factory.mktemp("combined") / "sim.yaml"; path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return str(path)


def cell(wins, falls=0, episodes=100):
    return {"episodes": episodes, "wins": wins, "own_falls": falls, "timeouts": episodes - wins - falls}


def heldout(turn_tl, turn_br, earlier, new):
    """Held-out table from (wins, own falls) pairs for the two ordinary sets."""
    return {"turn_needed": {"top_left": cell(turn_tl, episodes=162), "bottom_right": cell(turn_br, episodes=162)},
            "ordinary": {"earlier_ordinary": cell(*earlier, episodes=60), "new_confirmation": cell(*new, episodes=144)}}


def test_new_ordinary_starts_are_reproducible_and_repeat_no_held_out_case():
    from autobot.sim.core import load_config
    starts = combined.load_starts(); ladder = experiment.load_ladder(); arena = load_config(str(experiment.CONFIG))["arena"]
    assert combined.build_starts(ladder, arena) == starts
    assert len(starts["train"]) == 120 and len(starts["validation"]) == 40
    assert all(pilot.placeable(start, arena) for start in starts["train"] + starts["validation"])
    _, confirmation = pilot.load_mix(); held = pilot.heldout_starts(ladder) + confirmation
    assert len(held) == 56 + 110 + 20 + 48
    for leaked in (held[60], held[170], confirmation[0], starts["validation"][3]):       # ladder hold-out, earlier ordinary, confirmation, validation
        nudged = dict(leaked, player=[leaked["player"][0] + 0.01, leaked["player"][1], leaked["player"][2] + 0.02])
        with pytest.raises(ValueError, match="held-out"): combined.check_starts(dict(starts, train=starts["train"] + [nudged]), held)
    with pytest.raises(ValueError, match="held-out"): combined.check_starts(dict(starts, validation=starts["validation"] + [held[-1]]), held)


def test_demonstration_plan_keeps_the_original_drill_starts_and_seeds():
    ladder = experiment.load_ladder(); starts = combined.load_starts(); plan = combined.demonstration_plan(ladder, starts)
    drill = [item for item in plan if item[0] == "drill"]
    assert [start["name"] for _, start, _ in drill] == [start["name"] for rung in ladder["rungs"] for start in rung["train"]] and len(drill) == 70
    assert [seed for _, _, seed in drill] == [imitation.DATA_SEED + index for index in range(70)]       # controller_imitation iteration 0
    assert Counter_of(plan) == {"drill": 70, "ordinary": 120, "ordinary_validation": 40}
    assert len({seed for _, _, seed in plan}) == len(plan)
    held_names = {start["name"] for start in pilot.heldout_starts(ladder) + pilot.load_mix()[1]}
    assert not held_names & {start["name"] for source, start, _ in plan if source == "ordinary"}


def Counter_of(plan):
    counts = {}
    for source, _, _ in plan: counts[source] = counts.get(source, 0) + 1
    return counts


def test_demonstrate_matches_the_original_recorder_and_reports_how_the_episode_ended(short_config):
    env = AutoBotEnv(short_config, "sim_debug", 0, **COMMON); expert = diagnostic.FlankController(); expert.bind(env)
    start = hl.ladder_start("top_left", 0.46, 24, "probe")
    observations, labels, taken, outcome = combined.demonstrate(env, expert, start, seed=3)
    reference = imitation.Recorder(env).episode(None, start, 3)
    assert np.array_equal(observations, reference[0]) and np.array_equal(labels, reference[1]) and np.array_equal(taken, labels)
    assert outcome == "wins" and labels[0] == 4
    driven = combined.demonstrate(env, expert, start, seed=3, driver=ScriptedBaseline(), step_cap=20)
    assert len(driven[1]) == 20 and set(driven[2].tolist()) == {1} and driven[1][0] == 4 and driven[3] == "cut"   # labels stay the controller's
    ordinary = combined.load_starts()["train"][0]
    assert combined.demonstrate(env, expert, ordinary, seed=5)[3] == "timeouts"                                    # 2 s horizon here


def test_balanced_batches_take_half_from_each_pool_and_uniform_batches_follow_pool_size():
    import torch
    pools = [(torch.zeros(50, 19), torch.zeros(50, dtype=torch.long)), (torch.zeros(5000, 19), torch.zeros(5000, dtype=torch.long))]
    generator = torch.Generator().manual_seed(0)
    balanced = combined.sample_batch(pools, True, generator)
    assert [len(index) for index in balanced] == [128, 128] and int(balanced[0].max()) < 50 and int(balanced[1].max()) < 5000
    counts = np.sum([[len(index) for index in combined.sample_batch(pools, False, generator)] for _ in range(200)], axis=0)
    assert counts.sum() == 200 * combined.BATCH and 0.005 < counts[0] / counts.sum() < 0.02          # 50 of 5,050 decisions
    assert all(int(index.min()) >= 0 and int(index.max()) < size for index, size in zip(combined.sample_batch(pools, False, generator), (50, 5000)))


def test_fit_learns_both_pools_and_never_touches_the_value_head(short_config):
    from stable_baselines3 import PPO
    env = AutoBotEnv(short_config, "sim_debug", 0, **COMMON); expert = diagnostic.FlankController(); expert.bind(env)
    drill = combined.demonstrate(env, expert, hl.ladder_start("top_left", 0.46, 24, "probe"), seed=3)
    ordinary = combined.demonstrate(env, expert, combined.load_starts()["train"][0], seed=5)
    pools = {"drill": drill[:2], "ordinary": ordinary[:2]}
    model = PPO("MlpPolicy", env, device="cpu", seed=2, n_steps=8, batch_size=8)
    critic = pilot.flat(pilot.critic_modules(model.policy)); actor = pilot.flat(pilot.actor_modules(model.policy))
    before = {name: imitation.agreement(model, *pool, env.tactics)["agreement"] for name, pool in pools.items()}
    report = combined.fit(model, pools, steps=1500, balanced=True, seed=0)
    assert set(report) == {"drill", "ordinary"} and all(report[name]["agreement"] >= max(before[name], 0.9) for name in report)
    assert report["drill"]["decisions"] == len(drill[1]) and report["ordinary"]["decisions"] == len(ordinary[1])
    assert np.array_equal(critic, pilot.flat(pilot.critic_modules(model.policy)))
    assert not np.array_equal(actor, pilot.flat(pilot.actor_modules(model.policy)))


def test_recipe_selection_needs_the_drill_at_both_pits_then_takes_the_best_ordinary_score():
    table = lambda tl, br, wins, falls: {"turn_needed": {"top_left": cell(tl), "bottom_right": cell(br)}, "ordinary": cell(wins, falls)}
    selected, details = combined.select_recipe({"concat": table(95, 60, 40, 0), "balanced": table(95, 92, 20, 2), "balanced_dagger": table(96, 97, 30, 15)})
    assert selected == "balanced" and not details["concat"]["eligible"] and details["balanced"]["score"] == 18 and details["balanced_dagger"]["score"] == 15
    assert combined.select_recipe({"concat": table(95, 95, 20, 2), "balanced": table(95, 92, 20, 2)})[0] == "concat"        # tie: the simpler recipe
    assert combined.select_recipe({"concat": table(10, 10, 5, 0), "balanced": table(20, 20, 9, 1)})[0] == "balanced"       # none eligible: best score
    pooled = combined.pooled([table(30, 31, 5, 1), table(32, 33, 7, 0)])
    assert pooled["turn_needed"]["top_left"]["wins"] == 62 and pooled["ordinary"] == {"episodes": 200, "wins": 12, "own_falls": 1, "timeouts": 187}
    assert combined.demo_seed({0: table(90, 90, 5, 1), 1: table(80, 80, 6, 1), 2: table(95, 95, 5, 0)}) == 2               # score 5 beats 4; ties by drill wins
    assert combined.demo_seed({0: table(90, 91, 5, 0), 2: table(95, 95, 5, 0)}) == 2


def test_keep_rule_needs_the_drill_ordinary_wins_and_few_own_falls():
    drill_only = heldout(160, 156, (0, 51), (0, 141)); advance = heldout(0, 0, (3, 0), (0, 3)); controller = heldout(162, 162, (24, 0), (36, 0))
    assert combined.keep_checks(heldout(150, 150, (14, 1), (20, 5)), drill_only, advance, controller)["passes"]
    lost_pit = combined.keep_checks(heldout(150, 100, (14, 1), (20, 5)), drill_only, advance, controller)
    assert not lost_pit["passes"] and not lost_pit["checks"]["keeps_drill_bottom_right"] and lost_pit["checks"]["keeps_drill_top_left"]
    few_wins = combined.keep_checks(heldout(150, 150, (14, 1), (10, 5)), drill_only, advance, controller)                # 10 is under half of 36
    assert not few_wins["passes"] and few_wins["checks"]["beats_advance_new_confirmation"] and not few_wins["checks"]["half_of_controller_new_confirmation"]
    falls = combined.keep_checks(heldout(150, 150, (14, 4), (20, 5)), drill_only, advance, controller)                   # 4 of 60 is over 5%
    assert not falls["passes"] and not falls["checks"]["few_own_falls_earlier_ordinary"]
    assert not combined.keep_checks(heldout(150, 150, (3, 0), (20, 5)), drill_only, advance, controller)["checks"]["beats_advance_earlier_ordinary"]


def test_written_demonstrations_match_their_summary():
    path = combined.DATA_DIR / "demonstrations.npz"
    if not path.is_file(): pytest.skip("demonstrations not collected (runs/ is not tracked)")
    data = combined.load_demonstrations(); summary = json.loads((combined.DATA_DIR / "demonstrations.json").read_text(encoding="utf-8"))["summary"]
    assert {source: len(labels) for source, (_, labels) in data.items()} == {source: summary[source]["decisions"] for source in combined.SOURCES}
    assert summary["drill"]["reproduces_the_original_drill_demonstrations"] and data["drill"][0].shape[1] == 19
    assert not np.any(data["ordinary"][0][:, 6:13])                                  # previous-action channels are masked
    assert summary["ordinary"]["outcomes"].get("own_falls", 0) == 0                 # the teacher never falls in
