"""Session and round logs: organised, crash-tolerant, honest about execution. Software only."""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

from control_center.backend import session_log as sl


class Clock:
    def __init__(self):
        self.t = 100.0

    def __call__(self):
        return self.t


def make(tmp_path, clock=None):
    clock = clock or Clock()
    wall = datetime(2026, 11, 21, 9, 0, 0, tzinfo=timezone.utc)
    return sl.SessionLog(tmp_path, clock=clock, wall_clock=lambda: wall + timedelta(seconds=clock.t - 100)), clock


def test_one_folder_per_session_and_one_file_per_round(tmp_path):
    log, clock = make(tmp_path)
    log.log("camera", "LIVE")
    assert log.start_round("AI SHADOW") == 1
    log.log("decision", "advance", tactic="advance")
    log.stop_round("operator pressed Stop")
    assert log.start_round("MANUAL") == 2
    log.stop_round("operator pressed Stop")
    log.close()
    assert log.folder.parent == tmp_path and (log.folder / "session.jsonl").exists()
    assert (log.folder / "round-001.jsonl").exists() and (log.folder / "round-002.jsonl").exists()
    first = [e["event"] for e in sl.read_events(log.folder / "round-001.jsonl")]
    assert first == ["start", "advance", "stop"]
    assert sl.list_sessions(tmp_path) == [log.folder]


def test_a_second_session_in_the_same_second_gets_its_own_folder(tmp_path):
    a, _ = make(tmp_path)
    b, _ = make(tmp_path)
    assert a.folder != b.folder


def test_transmissions_never_claim_physical_execution(tmp_path):
    log, _ = make(tmp_path)
    log.command({"tactic": "advance", "intended_throttle": 40, "intended_steer": 0, "transmitted_throttle": 40, "transmitted_steer": 0,
                 "sent": True, "why_not": ""}, mode="AUTONOMOUS", authority="AI")
    log.command({"tactic": "advance", "intended_throttle": 40, "intended_steer": 0, "transmitted_throttle": 0, "transmitted_steer": 0,
                 "sent": False, "why_not": "AI SHADOW never transmits"}, mode="AI_SHADOW", authority="RADIO")
    events = list(sl.read_events(log.folder / "session.jsonl"))
    transmits = [e for e in events if e["kind"] == "transmit"]
    assert [t["sent"] for t in transmits] == [True, False]
    assert all(t["physical_execution"] == sl.NO_EVIDENCE for t in transmits)
    assert not any("executed" in json.dumps(e).lower().replace("physical_execution", "") for e in events)
    plans = [e for e in events if e["kind"] == "command_plan"]
    assert plans[0]["intended_throttle"] == 40 and plans[1]["transmitted_throttle"] == 0


def test_high_rate_kinds_are_sampled_but_decisions_and_faults_never_are(tmp_path):
    log, clock = make(tmp_path)
    for _ in range(50):
        log.log("observation", "frame", valid=True)
        log.log("decision", "stop")
        log.fault("camera lost")
        clock.t += 0.01
    kinds = [e["kind"] for e in sl.read_events(log.folder / "session.jsonl")]
    assert kinds.count("decision") == 50 and kinds.count("safety") == 50 and kinds.count("observation") < 5 and log.dropped > 40


def test_a_crash_mid_line_loses_only_that_line(tmp_path):
    log, _ = make(tmp_path)
    log.log("note", "one")
    log.log("note", "two")
    path = log.folder / "session.jsonl"
    with open(path, "a", encoding="utf-8") as handle:
        handle.write('{"kind": "note", "event": "thr')          # power cut in the middle of a line
    events = [e["event"] for e in sl.read_events(path)]
    assert events[-2:] == ["one", "two"] and "thr" not in events


def test_closing_in_the_middle_of_a_round_ends_the_round_and_is_idempotent(tmp_path):
    log, _ = make(tmp_path)
    log.start_round("MANUAL")
    log.close()
    log.close()
    assert not log.log("note", "after close")
    events = list(sl.read_events(log.folder / "round-001.jsonl"))
    assert events[-1]["event"] == "stop" and "session closed" in events[-1]["reason"]
    assert [e["event"] for e in sl.read_events(log.folder / "session.jsonl")][-1] == "stop"


def test_a_write_failure_never_raises_into_the_application(tmp_path, monkeypatch):
    log, _ = make(tmp_path)
    monkeypatch.setattr(log, "_open", lambda name: (_ for _ in ()).throw(OSError("disk full")))
    assert log.log("note", "x") is False and log.errors == 1


def test_mode_changes_and_refusals_are_logged_with_reasons(tmp_path):
    log, _ = make(tmp_path)
    log.mode_change("MANUAL", "AUTONOMOUS", False, ["no hardware acceptance record"])
    event = [e for e in sl.read_events(log.folder / "session.jsonl") if e["kind"] == "mode"][0]
    assert "REFUSED" in event["event"] and event["reasons"] == ["no hardware acceptance record"]
    assert "mode" in sl.describe(event)
