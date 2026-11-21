"""The Wi-Fi acceptance record (control_center/backend/acceptance.py).

SOFTWARE TESTS ONLY. Records written here are SYNTHETIC test data in a temporary file: they exercise the rules and are not
evidence about any robot. The default record (config/wifi_acceptance.json) is never written.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from control_center.backend import acceptance as acc  # noqa: E402
from control_center.backend.acceptance import (BLOCKED, FAIL, NOT_TESTED, PASS, Acceptance, Evidence, StabilityMeter)  # noqa: E402
from control_center.backend.wifi_link import LinkState  # noqa: E402

FW = "autobot-handoff-wifi-1"
SKETCH = "ab" * 32
NOW = 1_700_000_000.0
READINGS = dict(pack_rest_v=5.4, converter_out_idle_v=5.0, converter_out_wheels_v=4.9, minutes_without_restart=6)
CAPS = (acc.CAP_MOTORS_DISABLED_MODE,)


class Clock:
    def __init__(self, t: float = NOW) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t


def make(tmp_path, profile="default", clock=None) -> Acceptance:
    return Acceptance(tmp_path / "wifi_acceptance.json", profile, unix_clock=clock or Clock(),
                      motion_check_path=tmp_path / "no-safety-record.json")


def hw(a: Acceptance, step, status=PASS, check=None, fw=FW, sketch=SKETCH, data=None, link_real=True, confirmed=True, now=None,
       caps=CAPS):
    return a.record_result(step, status, Evidence.HARDWARE, check=check, produced_by="test operator", firmware_id=fw,
                           sketch_sha256=sketch, data=data, now_unix=now, link_real=link_real, operator_confirmed=confirmed,
                           capabilities=caps)


def pass_everything(a: Acceptance, now=None, caps=CAPS) -> None:
    hw(a, "baseline", data=READINGS, link_real=None, now=now, caps=caps)
    hw(a, "wifi", now=now, caps=caps)
    hw(a, "stability", now=now, caps=caps)
    hw(a, "reception", now=now, caps=caps)
    for check in acc.SUBCHECKS:
        hw(a, "transitions", check=check, now=now, caps=caps)
    hw(a, "supervised", now=now, caps=caps)


def status_of(summary, key):
    for step in summary.steps:
        if step.key == key:
            return step.status
        for child in step.subchecks:
            if child.key == key:
                return child.status
    raise KeyError(key)


def summary(a: Acceptance, **kwargs):
    kwargs.setdefault("capabilities", CAPS)
    return a.summary(FW, SKETCH, NOW, **kwargs)


# ---------------------------------------------------------------- the starting state

def test_nothing_is_tested_at_first_and_autonomous_is_blocked_with_a_list(tmp_path):
    a = make(tmp_path)
    s = a.summary(None, None, NOW)
    assert [v.number for v in s.steps] == [1, 2, 3, 4, 5, 6, 7]
    assert not s.complete and s.steps[-1].status == BLOCKED and "missing" in s.steps[-1].reason
    assert s.steps[0].status == NOT_TESTED and s.steps[1].status == NOT_TESTED
    assert s.steps[3].status == BLOCKED and "motors-disabled" in s.steps[3].reason     # no firmware reports it
    assert s.steps[5].status == BLOCKED and "steps 1 to 5" in s.steps[5].reason
    requirement = a.required_for_autonomous(FW, SKETCH, NOW)
    assert not requirement.complete and not requirement.takeover_pass and not requirement.disarm_pass
    assert any("step 1" in m for m in requirement.missing)


def test_the_repository_default_record_never_completes_without_a_firmware_id():
    a = Acceptance()                                      # reads config/wifi_acceptance.json if it exists; never writes here
    assert not a.required_for_autonomous(None, None, NOW).complete
    assert not a.required_for_autonomous(None, None, NOW, CAPS).complete
    assert acc.DEFAULT_PATH.name == "wifi_acceptance.json" and acc.DEFAULT_PATH.parent.name == "config"


def test_evidence_words_are_plain():
    assert Evidence.HARDWARE == "HARDWARE" and Evidence.OPERATOR_REPORTED == "OPERATOR-REPORTED"
    assert "SIMULATED" in Evidence.describe(Evidence.MOCK) and "not physical verification" in Evidence.describe(Evidence.SIM)
    assert "real robot" in Evidence.describe(Evidence.HARDWARE)
    assert Evidence.NOTE_SIMULATED == "simulated: not physical verification"


# ---------------------------------------------------------------- evidence rules

@pytest.mark.parametrize("kind", [Evidence.MOCK, Evidence.SIM, Evidence.OPERATOR_REPORTED])
def test_non_hardware_evidence_can_never_pass_and_is_stored_as_not_tested(tmp_path, kind):
    a = make(tmp_path)
    for step, check in (("baseline", None), ("wifi", None), ("stability", None), ("transitions", "takeover")):
        outcome = a.record_result(step, PASS, kind, check=check, produced_by="software", firmware_id=FW, link_real=True,
                                  operator_confirmed=True, data=READINGS)
        assert outcome.accepted is False and outcome.status == NOT_TESTED
        assert outcome.stored.status == NOT_TESTED and outcome.stored.evidence == kind
        assert ("simulated" in outcome.note) or ("operator-reported" in outcome.note)
    s = summary(a)
    assert status_of(s, "baseline") == NOT_TESTED and status_of(s, "wifi") == NOT_TESTED
    assert not s.complete
    # a simulated FAIL is not a physical failure either: it does not re-block anything
    a.record_result("wifi", FAIL, kind, produced_by="software", firmware_id=FW)
    assert status_of(summary(a), "stability") == NOT_TESTED


def test_a_mock_record_never_replaces_hardware_evidence_but_is_kept_in_history(tmp_path):
    a = make(tmp_path)
    assert hw(a, "wifi").accepted
    a.record_result("wifi", FAIL, Evidence.MOCK, produced_by="software", firmware_id=FW)
    assert status_of(summary(a), "wifi") == PASS
    data = json.loads((tmp_path / "wifi_acceptance.json").read_text())
    events = [e["event"] for e in data["profiles"]["default"]["history"]]
    assert "non-hardware record, not counted" in events


def test_hardware_pass_needs_an_operator_a_real_link_and_the_readings(tmp_path):
    a = make(tmp_path)
    refused = hw(a, "wifi", confirmed=False)
    assert not refused.accepted and "operator" in refused.note and refused.status == NOT_TESTED
    refused = hw(a, "wifi", link_real=False)
    assert not refused.accepted and "REAL" in refused.note
    refused = hw(a, "wifi", link_real=None)
    assert not refused.accepted
    refused = hw(a, "baseline", data={}, link_real=None)
    assert not refused.accepted and "PWR-1" in refused.note
    refused = hw(a, "baseline", data=dict(READINGS, minutes_without_restart=2), link_real=None)
    assert not refused.accepted and "minutes" in refused.note
    refused = a.record_result("wifi", PASS, Evidence.HARDWARE, produced_by="  ", firmware_id=FW, link_real=True,
                              operator_confirmed=True)
    assert not refused.accepted
    assert summary(a).steps[0].status == NOT_TESTED
    assert hw(a, "baseline", data=READINGS, link_real=None).accepted                    # step 1 does not need the link
    assert hw(a, "wifi").accepted


def test_unknown_steps_checks_and_derived_steps_are_programming_errors(tmp_path):
    a = make(tmp_path)
    with pytest.raises(ValueError):
        a.record_result("autonomous", PASS, Evidence.HARDWARE)          # derived: never set
    with pytest.raises(ValueError):
        a.record_result("nonsense", PASS, Evidence.HARDWARE)
    with pytest.raises(ValueError):
        a.record_result("transitions", PASS, Evidence.HARDWARE)         # needs check=
    with pytest.raises(ValueError):
        a.record_result("wifi", PASS, Evidence.HARDWARE, check="takeover")
    with pytest.raises(ValueError):
        a.record_result("wifi", "MAYBE", Evidence.HARDWARE)
    with pytest.raises(ValueError):
        a.record_result("wifi", PASS, "FROM-MEMORY")


# ---------------------------------------------------------------- order, blocking and failure

def test_step_4_stays_blocked_unless_the_firmware_reports_the_feature(tmp_path):
    a = make(tmp_path)
    blocked = hw(a, "reception", caps=())
    assert not blocked.accepted and "motors-disabled" in blocked.note
    assert status_of(a.summary(FW, SKETCH, NOW), "reception") == BLOCKED
    assert hw(a, "reception", caps=CAPS).accepted
    assert status_of(summary(a), "reception") == PASS
    assert status_of(a.summary(FW, SKETCH, NOW, capabilities=()), "reception") == BLOCKED


def test_step_6_is_blocked_until_steps_1_to_5_pass(tmp_path):
    a = make(tmp_path)
    refused = hw(a, "supervised")
    assert not refused.accepted and "steps 1 to 5" in refused.note
    hw(a, "baseline", data=READINGS, link_real=None)
    hw(a, "wifi")
    hw(a, "stability")
    hw(a, "reception")
    for check in list(acc.SUBCHECKS)[:-1]:
        hw(a, "transitions", check=check)
    assert status_of(summary(a), "supervised") == BLOCKED
    assert status_of(summary(a), "transitions") == NOT_TESTED
    hw(a, "transitions", check="reconnect_rearm")
    assert status_of(summary(a), "transitions") == PASS and status_of(summary(a), "supervised") == NOT_TESTED
    assert hw(a, "supervised").accepted
    s = summary(a)
    assert s.complete and s.steps[-1].status == PASS and s.steps[-1].evidence == Evidence.HARDWARE


def test_a_fail_reblocks_later_steps_and_clears_their_results(tmp_path):
    a = make(tmp_path)
    pass_everything(a)
    assert summary(a).complete
    failed = hw(a, "wifi", status=FAIL)
    assert failed.accepted and failed.status == FAIL
    s = summary(a)
    assert status_of(s, "wifi") == FAIL
    for later in ("stability", "reception", "transitions", "supervised"):
        assert status_of(s, later) == BLOCKED, later
    assert status_of(s, "baseline") == PASS and not s.complete and s.steps[-1].status == BLOCKED
    assert not hw(a, "stability").accepted                                 # cannot be recorded while step 2 stands failed
    assert hw(a, "wifi").accepted                                          # passing step 2 again unblocks...
    s = summary(a)
    assert status_of(s, "stability") == NOT_TESTED and status_of(s, "transitions.takeover") == NOT_TESTED   # ...but they are redone
    history = json.loads((tmp_path / "wifi_acceptance.json").read_text())["profiles"]["default"]["history"]
    assert any(str(e.get("event", "")).startswith("invalidated by the failure of step 2") for e in history)


def test_a_failing_subcheck_fails_step_5_and_blocks_step_6(tmp_path):
    a = make(tmp_path)
    pass_everything(a)
    hw(a, "transitions", check="pc_link_loss", status=FAIL)
    s = summary(a)
    assert status_of(s, "transitions") == FAIL and status_of(s, "transitions.pc_link_loss") == FAIL
    assert status_of(s, "supervised") == BLOCKED
    req = a.required_for_autonomous(FW, SKETCH, NOW, CAPS)
    assert not req.complete


def test_takeover_and_disarm_flags_feed_the_gate(tmp_path):
    a = make(tmp_path)
    req = a.required_for_autonomous(FW, SKETCH, NOW, CAPS)
    assert not req.takeover_pass and not req.disarm_pass
    hw(a, "baseline", data=READINGS, link_real=None)
    hw(a, "wifi")
    hw(a, "stability")
    hw(a, "reception")
    hw(a, "transitions", check="takeover")
    hw(a, "transitions", check="arm_off")
    req = a.required_for_autonomous(FW, SKETCH, NOW, CAPS)
    assert req.takeover_pass and req.disarm_pass and not req.complete


# ---------------------------------------------------------------- staleness

def test_results_are_stale_after_a_firmware_or_hash_change_and_after_12_hours(tmp_path):
    a = make(tmp_path)
    pass_everything(a, now=NOW)
    assert a.summary(FW, SKETCH, NOW, CAPS).complete
    other = a.summary("autobot-handoff-bench-1", SKETCH, NOW, CAPS)
    assert not other.complete and "stale: recorded on firmware" in " ".join(v.reason for v in other.steps)
    changed = a.summary(FW, "cd" * 32, NOW, CAPS)
    assert not changed.complete and status_of(changed, "wifi") == NOT_TESTED
    assert "sketch hash changed" in [v for v in changed.steps if v.key == "wifi"][0].reason
    # 12 h: only the stability and transition steps expire by time
    later = a.summary(FW, SKETCH, NOW + 12 * 3600 + 1, CAPS)
    assert status_of(later, "stability") == NOT_TESTED and "stale" in [v for v in later.steps if v.key == "stability"][0].reason
    assert status_of(later, "transitions.takeover") == NOT_TESTED
    assert status_of(later, "baseline") == PASS and status_of(later, "wifi") == PASS
    assert not later.complete
    assert a.summary(FW, SKETCH, NOW + 12 * 3600 - 1, CAPS).complete


def test_a_result_recorded_without_a_firmware_id_never_satisfies_autonomous(tmp_path):
    complete = make(tmp_path / "with-firmware")
    pass_everything(complete)
    assert complete.required_for_autonomous(FW, SKETCH, NOW, CAPS).complete            # control: the same records, with an id
    nofw = make(tmp_path / "without-firmware")
    for step in ("baseline", "wifi", "stability", "reception"):
        hw(nofw, step, fw=None, sketch=None, data=READINGS)
    for check in acc.SUBCHECKS:
        hw(nofw, "transitions", check=check, fw=None, sketch=None)
    hw(nofw, "supervised", fw=None, sketch=None)
    assert not nofw.required_for_autonomous(FW, SKETCH, NOW, CAPS).complete
    assert not nofw.required_for_autonomous(None, None, NOW, CAPS).complete


# ---------------------------------------------------------------- persistence

def test_records_persist_per_profile_and_are_written_atomically(tmp_path):
    a = make(tmp_path)
    pass_everything(a)
    again = make(tmp_path)
    assert again.summary(FW, SKETCH, NOW, CAPS).complete
    other = make(tmp_path, profile="competition")
    assert not other.summary(FW, SKETCH, NOW, CAPS).complete                         # profiles do not share results
    hw(other, "wifi")
    assert make(tmp_path).summary(FW, SKETCH, NOW, CAPS).complete                    # saving one profile keeps the other
    assert status_of(make(tmp_path, profile="competition").summary(FW, SKETCH, NOW, CAPS), "wifi") == PASS
    assert not list(tmp_path.glob("*.tmp"))                                           # nothing half-written is left
    document = json.loads((tmp_path / "wifi_acceptance.json").read_text())
    assert document["version"] == 1 and set(document["profiles"]) == {"default", "competition"}


def test_a_damaged_file_is_ignored_with_a_message_and_kept_aside(tmp_path):
    path = tmp_path / "wifi_acceptance.json"
    path.write_text("{ this is not json", encoding="utf-8")
    a = make(tmp_path)
    assert "damaged" in a.load_message and path.name in a.load_message
    assert all(v.status != PASS for v in a.summary(FW, SKETCH, NOW, CAPS).steps)
    hw(a, "wifi")
    assert (tmp_path / "wifi_acceptance.json.damaged").read_text() == "{ this is not json"
    assert status_of(make(tmp_path).summary(FW, SKETCH, NOW, CAPS), "wifi") == PASS
    path.write_text('["not", "an", "object"]', encoding="utf-8")
    assert "damaged" in make(tmp_path).load_message


def test_a_hand_edited_pass_without_hardware_evidence_is_not_believed(tmp_path):
    a = make(tmp_path)
    hw(a, "wifi")
    path = tmp_path / "wifi_acceptance.json"
    document = json.loads(path.read_text())
    document["profiles"]["default"]["results"]["wifi"]["evidence"] = "MOCK"
    document["profiles"]["default"]["results"]["bogus.entry"] = {"key": "bogus.entry", "status": "PASS", "evidence": "HARDWARE",
                                                                 "t_unix": 1}
    path.write_text(json.dumps(document), encoding="utf-8")
    b = make(tmp_path)
    assert status_of(b.summary(FW, SKETCH, NOW, CAPS), "wifi") == NOT_TESTED
    assert "ignored" in b.load_message


# ---------------------------------------------------------------- the stability meter

def health(state=LinkState.CONNECTED, age=0.1, restarts=0, real=True):
    return SimpleNamespace(state=state, heartbeat_age_s=age, restarts=restarts, real=real)


def run_meter(meter, seconds=60.0, step=0.5, **kwargs):
    t = 100.0
    meter.start(health(**kwargs), t)
    while meter.state == StabilityMeter.RUNNING and t < 100.0 + seconds + 1:
        t += step
        meter.feed(health(**kwargs), t)
    return meter


def test_a_real_60_second_window_passes_and_records_hardware(tmp_path):
    meter = run_meter(StabilityMeter())
    assert meter.state == StabilityMeter.PASSED and meter.evidence == Evidence.HARDWARE and "no restart" in meter.reason
    a = make(tmp_path)
    outcome = a.record_stability(meter, firmware_id=FW, sketch_sha256=SKETCH, operator_confirmed=True, now_unix=NOW)
    assert outcome.accepted and status_of(summary(a), "stability") == PASS
    assert a._results["stability"].status == PASS and a._results["stability"].data["window_s"] == 60.0


def test_a_mock_link_window_is_counted_as_mock_and_never_passes(tmp_path):
    meter = run_meter(StabilityMeter(), real=False)
    assert meter.state == StabilityMeter.PASSED and meter.evidence == Evidence.MOCK
    a = make(tmp_path)
    outcome = a.record_stability(meter, firmware_id=FW, operator_confirmed=True, now_unix=NOW)
    assert not outcome.accepted and outcome.status == NOT_TESTED and "simulated" in outcome.note
    assert "stability" in a._results and a._results["stability"].status == NOT_TESTED


@pytest.mark.parametrize("kwargs, reason", [(dict(state=LinkState.DEGRADED), "DEGRADED"), (dict(age=0.6), "heartbeat"),
                                            (dict(age=None), "no heartbeat")])
def test_the_window_fails_on_a_bad_sample(kwargs, reason):
    meter = StabilityMeter()
    meter.start(health(), 0.0)
    meter.feed(health(**kwargs), 1.0)
    assert meter.state == StabilityMeter.FAILED and reason in meter.reason


def test_the_window_fails_on_a_restart_or_a_gap_and_a_running_window_records_nothing(tmp_path):
    meter = StabilityMeter()
    meter.start(health(restarts=0), 0.0)
    meter.feed(health(restarts=1), 1.0)
    assert meter.state == StabilityMeter.FAILED and "restarted" in meter.reason
    meter = StabilityMeter()
    meter.start(health(), 0.0)
    meter.feed(health(), 5.0)                                                  # not fed for 5 s: not measured
    assert meter.state == StabilityMeter.FAILED and "continuously" in meter.reason
    running = StabilityMeter()
    running.start(health(), 0.0)
    a = make(tmp_path)
    assert not a.record_stability(running, firmware_id=FW).accepted and "stability" not in a._results
    failed = StabilityMeter()
    failed.start(health(), 0.0)
    failed.feed(health(age=0.9), 1.0)
    assert a.record_stability(failed, firmware_id=FW, operator_confirmed=False).status == FAIL


# ---------------------------------------------------------------- motion_check's record

def test_motion_check_record_is_reported_but_never_becomes_a_pass(tmp_path):
    a = make(tmp_path)
    info = a.summary(FW, SKETCH, NOW).motion_check
    assert info is not None and info.record_exists is False and "never converted" in info.text
    record = [{"started_unix": NOW - 600, "finished_unix": NOW - 300, "link": "USB COM5", "firmware": FW,
               "checks": {name: {"controller_pass": True, "operator_confirmed": True} for name in
                          ("stop", "disarm", "takeover", "pc_link_loss", "radio_loss")}}]
    (tmp_path / "no-safety-record.json").write_text(json.dumps(record), encoding="utf-8")
    info = a.summary(FW, SKETCH, NOW).motion_check
    assert info.record_exists and info.link == "USB COM5" and "takeover" in info.passed and info.permitted
    assert status_of(a.summary(FW, SKETCH, NOW, CAPS), "transitions.takeover") == NOT_TESTED
