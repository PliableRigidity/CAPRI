"""Regression case: the live round of 2026-11-18 in which the frozen combined clone spun on the spot.

A person drove red; `config/regression_cases/live_spin_2026_11_18.json` holds red's recorded wheel commands and
what blue did. Red is replayed open loop, so for any blue other than the recorded one this is a fixed input
replay, not a live opponent.

The rule a blue has to meet on the spin round (scripts/live_round_diagnostic.passes_case): no spin, reach red,
do not fall. The frozen clone and the scripted demonstration teacher are both KNOWN to fail it; those tests pin
what was measured, so a change in either shows up here. Nothing in this file trains anything.
"""
import hashlib
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import live_round_diagnostic as live  # noqa: E402

REFERENCE = Path(live.MODEL)


@pytest.fixture(scope="module")
def case():
    if not live.CASE.is_file(): pytest.skip(f"{live.CASE} not present")
    return live.load_case()


@pytest.fixture(scope="module")
def session():
    if not REFERENCE.is_file() or not Path(live.CONFIG).is_file(): pytest.skip("frozen checkpoint or config not present (runs/ is not tracked)")
    return live.make_session()


def test_case_file_is_complete_and_matches_the_log_it_came_from(case):
    assert case["schema"] == "live_round_regression_v1" and case["seed"] == 10007 and case["scenario_index"] == 2
    assert [len(item["red"]) for item in case["rounds"]] == [1000, 1000] == [len(item["blue"]) for item in case["rounds"]]
    assert case["spin"]["round"] == 2 and case["spin"]["tactic"] == "turn_right" and case["pass_rule"]["applies_to_round"] == 2
    log = live.ROOT / case["source_log"]
    if log.is_file():                                              # runs/ is not tracked; the case file is the durable copy
        assert hashlib.sha256(log.read_bytes()).hexdigest() == case["source_log_sha256"]
        for item, steps in zip(case["rounds"], live.load_rounds(log)):
            assert item["red"] == [tuple(step["red_command"]) for step in steps] and item["blue"] == [step["blue_tactic"] for step in steps]


def test_newest_log_is_chosen_by_the_time_in_its_name(tmp_path):
    for name in ("reference-normal-spawn-20261118-000023.jsonl", "reference-normal-spawn-20261119-140509.jsonl",
                 "reference-normal-spawn-20261119-140509-2.jsonl", "reference-normal-spawn.jsonl", "reference-player-right-20261120-000000.jsonl"):
        (tmp_path / name).write_text("", encoding="utf-8")
    live.ROOT, root = tmp_path, live.ROOT
    try: newest, found = live.newest_log(tmp_path)
    finally: live.ROOT = root
    assert newest.name == "reference-normal-spawn-20261119-140509-2.jsonl" and len(found) == 3


def test_spin_rule_on_synthetic_traces():
    def trace(actions, yaw_step_deg, contact=False):
        n = len(actions); yaw = np.concatenate([[0.0], np.cumsum(np.radians([abs(yaw_step_deg) if a == 3 else -abs(yaw_step_deg) if a == 4 else 0.0 for a in actions]))])
        poses = np.zeros((n + 1, 6)); poses[:, 2] = np.arctan2(np.sin(yaw), np.cos(yaw)); poses[:, 3] = 0.5
        return {"first_step": 0, "actions": np.asarray(actions), "poses": poses, "contact": np.full(n, contact), "end_reason": "Timeout", "end_bot": None}
    spinning = live.passes_case(trace([4] * 300, -5.0))            # 1500 degrees one way, never advancing
    assert not spinning["no_spin"] and not spinning["passes"] and spinning["longest_one_way_rotation"]["rotation_deg"] == pytest.approx(-1500, abs=1)
    weaving = live.passes_case(trace(([3] * 10 + [4] * 10) * 15, 5.0))
    assert weaving["no_spin"] and not weaving["reaches_red_after_spin_onset"] and not weaving["passes"]      # no spin, but never gets to red
    turning_then_pushing = live.passes_case(trace(([4] * 100 + [1] * 5) * 3, -5.0, contact=True))
    assert turning_then_pushing["no_spin"] and turning_then_pushing["passes"]                                # advancing resets the count


def test_frozen_clone_reproduces_the_recorded_rounds_exactly(case, session):
    clone = live.Clone(session.model)
    for item in case["rounds"]:
        trace = live.rollout(session.env, clone, item["red"])
        assert [live.TACTICS[action] for action in trace["actions"]] == item["blue"] and live.outcome(trace) == "timeout"


def test_frozen_clone_is_known_to_spin_in_round_two(case, session):
    trace, verdict = live.run_case(case, live.Clone(session.model), session)
    spin = live.spin_segment(trace)
    assert spin == {"first_decision": 403, "end_decision": 1000, "tactic": "turn_right"}
    held = live.window_metrics(trace, 403, 1000)
    assert held["tactic_counts"] == {"turn_right": 597} and held["revolutions"] == pytest.approx(8.9, abs=0.1)
    assert held["nose_passes_over_red"] == 9 and held["advance_decisions_with_nose_on_red"] == 0 and held["contact_decisions"] == 0
    assert held["net_displacement_m"] < 0.05                         # it turned on the spot
    assert not verdict["no_spin"] and not verdict["reaches_red_after_spin_onset"] and not verdict["passes"]


def test_teacher_does_not_spin_but_is_known_not_to_reach_red(case, session):
    trace, verdict = live.run_case(case, live.Teacher(), session)
    assert verdict["no_spin"] and verdict["no_own_fall"] and not verdict["reaches_red_after_spin_onset"] and not verdict["passes"]
    stall = live.teacher_stall(trace, session.env.sim.cfg["arena"])
    assert stall["copy_of_teacher_geometry_matches_its_actions"] == "1000/1000"
    assert stall["steering_at"] == {"go_round_waypoint": 250} and stall["median_distance_to_that_point_m"] < 0.04    # parked on its own waypoint
    assert not trace["contact"].any()


def test_restoring_the_saved_pre_spin_state_continues_identically(case, session):
    clone = live.Clone(session.model); red = case["rounds"][1]["red"]
    full = live.rollout(session.env, clone, red, keep={403})
    resumed = live.rollout(session.env, clone, red[403:], start=full["snapshots"][403])
    assert np.array_equal(resumed["actions"], full["actions"][403:]) and np.array_equal(resumed["poses"], full["poses"][403:])
