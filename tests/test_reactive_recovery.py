"""The reactive baseline, the scripted reds of the fresh cases, and the spin/stall measures. Trains nothing."""
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import reactive_recovery as recovery  # noqa: E402


def observation(blue, yaw_deg, red):
    o = np.zeros(19, dtype=np.float32); yaw = math.radians(yaw_deg)
    o[13:19] = [2 * blue[0] / 1.15 - 1, 2 * blue[1] / 1.15 - 1, math.sin(yaw), math.cos(yaw), 2 * red[0] / 1.15 - 1, 2 * red[1] / 1.15 - 1]
    return o


def test_reactive_baseline_turns_toward_red_and_advances_when_aligned():
    policy = recovery.PursuitController(); act = lambda *args: int(policy.predict(observation(*args))[0])
    assert act((0.5, 0.2), 90, (0.5, 0.9)) == 1                       # red dead ahead: advance
    assert act((0.5, 0.2), 90, (0.2, 0.5)) == 3                       # red to the left: turn_left
    assert act((0.5, 0.2), 90, (0.8, 0.5)) == 4                       # red to the right: turn_right
    assert act((0.5, 0.9), 90, (0.5, 0.2)) in (3, 4)                  # red directly behind: it turns, never advances away
    assert act((0.5, 0.2), 90 - 11, (0.5, 0.9)) == 1 and act((0.5, 0.2), 90 - 13, (0.5, 0.9)) == 3      # the 12 degree window
    states = np.stack([observation((0.3, 0.3), yaw, (0.8, 0.7)) for yaw in range(-180, 180, 7)])
    assert policy.labels(states).tolist() == [int(policy.predict(state)[0]) for state in states]
    assert set(policy.labels(states).tolist()) == {1, 3, 4}             # every heading has an answer and one of them is advance


def test_reactive_baseline_reads_only_the_observation():
    policy = recovery.PursuitController(); base = observation((0.4, 0.4), 30, (0.9, 0.5)); noisy = base.copy()
    noisy[:13] = np.random.default_rng(0).uniform(-1, 1, 13)             # IMU and previous-action channels do not matter
    assert int(policy.predict(noisy)[0]) == int(policy.predict(base)[0])
    moved = base.copy(); moved[17:19] = observation((0.4, 0.4), 30, (0.1, 0.9))[17:19]
    assert int(policy.predict(moved)[0]) != int(policy.predict(base)[0])  # moving red changes the choice


def synthetic(actions, yaw_step_deg=0.0, advance_m=0.0, contact=False):
    n = len(actions); poses = np.zeros((n + 1, 6)); poses[:, 3] = 0.6; yaw = 0.0; x = 0.0
    for index, action in enumerate(actions):
        if action in (3, 4): yaw += math.radians(yaw_step_deg if action == 3 else -yaw_step_deg)
        if action == 1: x += advance_m
        poses[index + 1, 0] = x; poses[index + 1, 2] = math.atan2(math.sin(yaw), math.cos(yaw))
    return poses, np.asarray(actions), np.full(n, contact)


def test_spin_and_stall_measures():
    spin = recovery.trace_metrics(*synthetic([4] * 500, 5.4))
    assert spin["spinning_s"] == pytest.approx(9.02, abs=0.05) and spin["stalled_s"] == 0.0 and not spin["reaches_red"]
    jitter = recovery.trace_metrics(*synthetic(([3] * 8 + [4] * 8) * 30, 5.4))
    assert jitter["spinning_s"] == 0.0 and jitter["stalled_s"] == pytest.approx(480 * 0.02 - 0.98, abs=0.05)
    still = recovery.trace_metrics(*synthetic([0] * 200))
    assert still["stalled_s"] > 2.9 and still["pre_contact_path_m"] == 0.0            # standing still counts as stalled
    driving = recovery.trace_metrics(*synthetic([1] * 200, advance_m=0.004))
    assert driving["spinning_s"] == 0.0 and driving["stalled_s"] == 0.0 and driving["path_m"] == pytest.approx(0.8, abs=0.01)
    pushing = recovery.trace_metrics(*synthetic([1] * 200, advance_m=0.0, contact=True))
    assert pushing["stalled_s"] == 0.0 and pushing["contact_s"] == 4.0 and pushing["reaches_red"] and pushing["first_contact_s"] == 0.0
    turning_once = recovery.trace_metrics(*synthetic([4] * 34 + [1] * 200, 5.4, 0.004))
    assert turning_once["spinning_s"] == 0.0                              # one 180 degree turn is not a spin


@pytest.fixture(scope="module")
def cases():
    if not recovery.CASES_PATH.is_file(): pytest.skip("recovery cases not present")
    return recovery.load_cases()                                          # also checks them against every protected start


def test_fresh_cases_are_split_cleanly_and_repeat_nothing_protected(cases):
    import clone_finetune as pilot
    splits = cases["splits"]; assert {name: len(items) for name, items in splits.items()} == {"train": 96, "validation": 48, "test": 120}
    for items in splits.values(): assert Counter(case["kind"] for case in items).keys() == set(recovery.RED_KINDS)
    starts = {name: {case["start"]["name"]: case["start"] for case in items} for name, items in splits.items()}
    assert not set(starts["train"]) & set(starts["validation"]) and not set(starts["train"]) & set(starts["test"]) and not set(starts["validation"]) & set(starts["test"])
    spawn = {"player": [0.575, 0.18, math.pi / 2], "opponent": [0.575, 0.97, -math.pi / 2]}       # the human recording's start
    assert not any(pilot.same_case(start, spawn) for part in starts.values() for start in part.values())
    seeds = [case["seed"] for items in splits.values() for case in items]; assert len(seeds) == len(set(seeds))
    plan = recovery.demonstration_plan(); used = {case["name"] for _, case, _ in plan}
    assert not used & {case["name"] for split in ("validation", "test") for case in splits[split]}   # fitting never touches validation or test cases



def test_scripted_reds_are_reproducible_and_do_what_their_names_say(cases):
    import opponent_pool as pool_tools
    env = pool_tools.make_env(); stop = recovery.load_policy({"kind": "always_stop"}); chase = recovery.PursuitController()
    pick = lambda kind: next(case for case in cases["splits"]["train"] if case["kind"] == kind and case["family"] == "train_default")
    still = recovery.run_case(stop, env, pick("stationary"), keep=True, cap=150)
    assert np.linalg.norm(still["poses"][-1, 3:5] - still["poses"][0, 3:5]) < 0.02
    away = [recovery.run_case(chase, env, pick("moving_away"), keep=True, cap=120) for _ in range(2)]
    assert np.array_equal(away[0]["poses"], away[1]["poses"])                                      # same seed, same episode
    waiting = recovery.run_case(stop, env, pick("moving_away"), keep=True, cap=150)
    assert np.linalg.norm(waiting["poses"][-1, 3:5] - waiting["poses"][0, 3:5]) < 0.02              # it only runs once blue comes close
    assert np.linalg.norm(away[0]["poses"][:, 3:5] - away[0]["poses"][0, 3:5], axis=1).max() > 0.05
    sides = recovery.run_case(stop, env, pick("changing_sides"), keep=True, cap=400); p = sides["poses"]
    assert p[:, 4].min() < p[0, 1] + 0.15                                                           # red ends up past blue, on blue's far side


def table(fresh=(10, 0, 48, 2.0, 6.0, 0.2), pool=(20, 2), ordinary=(9, 0), drill=(24, 24)):
    def cell(episodes, wins, own_falls, reaches=None, spin=0.0, stall=0.0, speed=0.3):
        return {"episodes": episodes, "wins": wins, "own_falls": own_falls, "timeouts": episodes - wins - own_falls, "red_self_falls": 0, "draws": 0,
                "simulation_bugs": 0, "reaches_red": episodes if reaches is None else reaches, "mean_spinning_s": spin, "mean_stalled_s": stall,
                "pre_contact_speed_m_s": speed, "mean_pre_contact_path_m": 0.6}
    wins, falls, reaches, spin, stall, speed = fresh
    return {"fresh": {"all": cell(48, wins, falls, reaches, spin, stall, speed)}, "pool": {"all": cell(80, pool[0], pool[1]), "harder": cell(64, pool[0], pool[1])},
            "ordinary": {"all": cell(40, *ordinary)}, "drill_turn_needed": {"top_left": cell(24, drill[0], 0), "bottom_right": cell(24, drill[1], 0)}}


def test_recommendation_rule_needs_every_check():
    reference = table(fresh=(2, 4, 20, 1.0, 12.0, 0.1))
    better = recovery.recommend(table(), reference)
    assert better["recommend_challenger"] and all(better["checks"].values())
    lost_drill = recovery.recommend(table(drill=(24, 20)), reference)
    assert not lost_drill["checks"]["keeps_drill_bottom_right"] and not lost_drill["recommend_challenger"]
    stands_still = recovery.recommend(table(fresh=(10, 0, 48, 0.0, 0.5, 0.05)), reference)      # less spin, but slower to cover ground
    assert not stands_still["checks"]["moves_before_contact"] and not stands_still["recommend_challenger"]
    falls = recovery.recommend(table(fresh=(10, 9, 48, 0.0, 0.5, 0.3)), reference)
    assert not falls["checks"]["does_not_fall_fresh"] and not falls["recommend_challenger"]
    fewer_wins = recovery.recommend(table(fresh=(1, 0, 48, 0.0, 0.5, 0.3), pool=(5, 0)), table(fresh=(2, 0, 20, 1.0, 12.0, 0.1)))
    assert not fewer_wins["checks"]["wins_fresh_plus_harder_pool"] and not fewer_wins["recommend_challenger"]
    reaches_no_more = recovery.recommend(table(fresh=(10, 0, 20, 0.0, 0.5, 0.3)), reference)
    assert not reaches_no_more["checks"]["responds_reaches_red_more"]


def test_seed_selection_prefers_eligible_then_score_then_less_idle_time():
    summary = {"challenger_seed0": table(fresh=(10, 0, 48, 0.0, 0.2, 0.3)), "challenger_seed1": table(fresh=(12, 0, 48, 0.0, 0.2, 0.3), drill=(24, 10)),
               "challenger_seed2": table(fresh=(10, 0, 48, 0.0, 0.1, 0.3)), "reference": table(fresh=(40, 0, 48, 0.0, 0.0, 0.3))}
    selected, details = recovery.select_seed(summary)
    assert selected == "challenger_seed2" and not details["challenger_seed1"]["eligible"] and "reference" not in details


def test_anywhere_cases_cover_the_floor_and_stay_out_of_fitting_splits():
    import clone_finetune as pilot
    if not recovery.ANYWHERE_PATH.is_file(): pytest.skip("anywhere cases not present")
    cases = recovery.load_anywhere(); splits = cases["splits"]
    assert {name: len(items) for name, items in splits.items()} == {"train": 180, "validation": 60, "test": 180}
    starts = {name: {case["start"]["name"]: case["start"] for case in items} for name, items in splits.items()}
    everything = [start for part in starts.values() for start in part.values()]
    assert not any(pilot.same_case(a, b) for i, a in enumerate(everything) for b in everything[i + 1:])
    blue_y = np.array([start["player"][1] for start in everything]); assert blue_y.min() < 0.3 and blue_y.max() > 0.85      # blue is not always at the bottom
    used = {case["name"] for _, case, _ in recovery.demonstration_plan("v2")}
    assert {case["name"] for case in splits["train"]} <= used
    assert not used & {case["name"] for split in ("validation", "test") for case in splits[split]}
    assert recovery.recommend.__defaults__ == (None,)                 # the saved-state check is only applied when a recording result is passed


def test_added_checks_only_make_the_recommendation_stricter():
    reference = table(fresh=(2, 4, 20, 1.0, 12.0, 0.1)); candidate = table()
    reference["anywhere"] = {"all": dict(reference["fresh"]["all"])}; candidate["anywhere"] = {"all": dict(candidate["fresh"]["all"])}
    passes = {"recorded_round_from_pre_spin_state": {"passes": True}}; stalls = {"recorded_round_from_pre_spin_state": {"passes": False}}
    assert recovery.recommend(candidate, reference, passes)["recommend_challenger"]
    verdict = recovery.recommend(candidate, reference, stalls)
    assert not verdict["checks"]["recovers_from_the_saved_pre_spin_state"] and not verdict["recommend_challenger"]
    worse = table(); worse["anywhere"] = {"all": dict(reference["anywhere"]["all"])}
    assert not recovery.recommend(worse, reference, passes)["recommend_challenger"]
