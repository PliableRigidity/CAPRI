"""Observation freshness and completeness (control_center/backend/ai_observation.py). Pure data in, plain words out.

SOFTWARE TESTS ONLY: the inputs are hand-built stand-ins for a Decision / ShadowRunner dictionary / TrackingResult.
Nothing here is physical verification.
"""

from __future__ import annotations

from types import SimpleNamespace

from control_center.backend.ai_observation import IMU_ABSENT_REASON, observation_status


def entry(valid=True, age_ms=20.0, reason="ok", source="stream"):
    return SimpleNamespace(valid=valid, age_ms=age_ms, reason=reason, source=source)


def decision(**overrides):
    inputs = {"imu": entry(source="controller IMU"), "previous_action": entry(age_ms=None, source="constant 0.0 (masked)"),
              "own_pose": entry(age_ms=30.0, source="ArUco ids [1, 2]"), "opponent_position": entry(age_ms=30.0, source="blob T3")}
    inputs.update(overrides)
    observation = [0.0] * 19 if all(e.valid for e in inputs.values()) else None
    return SimpleNamespace(inputs=inputs, observation=observation, blocked_by=[], proposed=None)


def test_a_complete_fresh_observation():
    status = observation_status(decision())
    assert status.complete and not status.stale and status.missing == [] and status.fresh_known
    assert set(status.groups) == {"imu", "previous", "own_pose", "opponent"}
    assert all(group.valid for group in status.groups.values())
    assert status.groups["own_pose"].age_ms == 30.0 and status.observation_size == 19
    assert "COMPLETE" in status.statement and "19 values" in status.statement
    assert status.tracking_ok


def test_an_absent_imu_is_missing_with_the_plain_reason_and_nothing_is_invented():
    absent = entry(valid=False, age_ms=None, reason="no IMU telemetry (RecordedTelemetry)", source="none")
    status = observation_status(decision(imu=absent))
    assert not status.complete and not status.groups["imu"].valid and status.groups["imu"].age_ms is None
    assert status.groups["imu"].reason == IMU_ABSENT_REASON
    assert status.groups["imu"].reason == "the baseline firmware sends no IMU data: nothing is invented"
    assert status.missing == [f"IMU: {IMU_ABSENT_REASON}"] and "INCOMPLETE" in status.statement
    assert "IMU" in status.statement and "Nothing is invented" in status.statement
    assert status.groups["own_pose"].valid                                      # the other groups are reported as they are


def test_a_stale_imu_and_a_stale_camera_are_stale_not_missing():
    stale_imu = observation_status(decision(imu=entry(valid=False, age_ms=150.0, reason="stale: the newest IMU sample is 150 ms old (limit 100 ms)")))
    assert stale_imu.stale and not stale_imu.complete
    assert stale_imu.groups["imu"].stale and "stale" in stale_imu.groups["imu"].reason and stale_imu.groups["imu"].age_ms == 150.0
    assert "STALE" in stale_imu.statement
    # limits are applied even when a source claims valid
    over = observation_status(decision(imu=entry(age_ms=101.0)))
    assert over.stale and not over.groups["imu"].valid
    old_camera = observation_status(decision(own_pose=entry(age_ms=300.0), opponent_position=entry(age_ms=300.0)))
    assert old_camera.stale and not old_camera.complete
    assert not old_camera.groups["own_pose"].valid and not old_camera.groups["opponent"].valid
    edge = observation_status(decision(own_pose=entry(age_ms=250.0)))
    assert edge.complete                                                        # exactly at the limit is still fresh


def test_a_decision_ages_while_nobody_is_looking():
    fresh = observation_status(decision(), elapsed_s=0.0)
    older = observation_status(decision(), elapsed_s=0.2)                       # IMU 20 ms -> 220 ms: beyond 100 ms
    assert fresh.complete and older.stale and not older.groups["imu"].valid
    assert older.groups["own_pose"].valid and older.groups["own_pose"].age_ms == 230.0
    much_older = observation_status(decision(), elapsed_s=0.3)
    assert not much_older.groups["own_pose"].valid


def test_the_opponent_unknown_or_own_pose_lost_blocks_completeness_and_tracking():
    status = observation_status(decision(opponent_position=entry(valid=False, age_ms=30.0, reason="no unmarked blob selected")))
    assert not status.complete and not status.tracking_ok and "opponent" in status.tracking_reason
    assert status.missing == ["opponent: no unmarked blob selected"]
    status = observation_status(decision(own_pose=entry(valid=False, age_ms=30.0, reason="no marker visible")))
    assert not status.tracking_ok and "own pose" in status.tracking_reason


def test_shadow_runner_dictionary_without_ages_is_valid_as_reported_but_freshness_is_not_confirmed():
    runner = {"proposed": None, "blocked_by": [], "mode": "x", "policy_loaded": True, "error": "",
              "inputs": {"imu": {"valid": True, "reason": "ok", "source": "imu"},
                         "previous_action": {"valid": True, "reason": "ok", "source": "masked"},
                         "own_pose": {"valid": True, "reason": "ok", "source": "aruco"},
                         "opponent_position": {"valid": True, "reason": "ok", "source": "blob"}}}
    status = observation_status(runner)
    assert status.complete and not status.fresh_known and all(g.age_ms is None for g in status.groups.values()
                                                              if g.group != "previous")
    assert "NOT confirmed" in status.statement
    runner["inputs"]["imu"] = {"valid": False, "reason": "no IMU telemetry (NoTelemetry)", "source": "none"}
    status = observation_status(runner)
    assert not status.complete and status.groups["imu"].reason == IMU_ABSENT_REASON


def test_tracking_result_states_decide_tracking_ok():
    shadow = {"inputs": decision().inputs, "observation": [0.0] * 19, "error": ""}
    ok = observation_status(SimpleNamespace(shadow=shadow, own={"state": "predicted", "age_s": 0.05},
                                            opponent={"state": "tracked"}))
    assert ok.tracking_ok and ok.own_state == "predicted" and ok.opponent_state == "tracked"
    for own, opponent, word in (({"state": "stale"}, {"state": "tracked"}, "own pose"),
                                ({"state": "measured"}, {"state": "overlap"}, "opponent"),
                                ({"state": "lost"}, {"state": "lost"}, "own pose")):
        status = observation_status(SimpleNamespace(shadow=shadow, own=own, opponent=opponent))
        assert not status.tracking_ok and word in status.tracking_reason


def test_nothing_yet_is_all_missing():
    status = observation_status(None)
    assert not status.complete and not status.tracking_ok and "No observation yet" in status.statement
    assert set(status.missing) >= {f"IMU: {IMU_ABSENT_REASON}"} and status.groups["previous"].valid
    assert all(not status.groups[g].valid for g in ("imu", "own_pose", "opponent"))
    empty = observation_status(SimpleNamespace(shadow=None, own=None, opponent=None))
    assert not empty.complete


def test_a_vector_that_was_not_assembled_is_not_complete():
    status = observation_status(SimpleNamespace(inputs=decision().inputs, observation=None, blocked_by=[], proposed=None))
    assert not status.complete and any("not assembled" in note for note in status.notes)
