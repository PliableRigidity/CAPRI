"""ELRS & Control: the pure model (control_status) and the screen.

Mocks only: scripted controller lines, a temporary safety record, a patched subprocess.Popen. No serial port,
no camera and no socket is opened, no process is started, and nothing is sent to a robot.
"""

from __future__ import annotations

import ast
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
if os.path.isdir(r"C:\Windows\Fonts"):
    os.environ.setdefault("QT_QPA_FONTDIR", r"C:\Windows\Fonts")

import pytest
from PySide6.QtCore import QCoreApplication
from PySide6.QtWidgets import QApplication, QCheckBox, QLabel, QPushButton

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from control_center.app import load_config  # noqa: E402
from control_center.backend import control_status as cs  # noqa: E402
from control_center.backend.control_status import (BASELINE_FIRMWARE, NOT_SENT_TEXT, WIFI_FIRMWARE, ControlInputs,  # noqa: E402
                                                   ControlTracker)
from control_center.backend.controller_link import ControllerMonitor, ScriptedSource, status_line  # noqa: E402
from control_center.backend.hub import HubOptions, OperationsHub  # noqa: E402
from control_center.backend.readiness import FAILED, NOT_CONFIGURED, READY, WARNING  # noqa: E402
from control_center.screens import control as control_module  # noqa: E402
from control_center.screens.control import ControlScreen  # noqa: E402

import handoff_bench_client as client  # noqa: E402
import motion_check as mc  # noqa: E402

HELLO = f"HELLO fw={BASELINE_FIRMWARE} arm=CH5:HIGH pc=CH6:HIGH pc_limit=128"
RADIO = dict(armed=1, req="RADIO", act="RADIO", block="NONE", token=2, left=90, right=90)
DISARMED = dict(armed=0, req="RADIO", act="NONE", block="DISARMED")
PC_ACTIVE = dict(armed=1, req="PC", act="PC", pc_ok=1, block="NONE", stop="HANDOFF", token=3, pc_age=10)
LOST = dict(link=0, armed=0, req="NONE", act="NONE", block="RADIO_LOST", fault="RADIO_LOST", stop="RADIO_LOST")


def ch_line(**overrides) -> str:
    values = dict(rc_age=0, frames=2689, crc=0, len=0, lq=98, ch1=992, ch2=992, ch5=1792, ch6=191, ch7=992, ch8=992, ch9=992,
                  ch10=992, arm="CH5:1792:ARM", mode="CH6:191:RADIO", steer=0, thr=0, pc_bad=0)
    values.update(overrides)
    return "CH " + " ".join(f"{key}={value}" for key, value in values.items())


class Rig:
    """A monitor + tracker driven by explicit times, the way the screen drives them."""

    def __init__(self, label="COM5 (listen-only)", mock=False, policy=None, error=None, missing=None) -> None:
        self.monitor = ControllerMonitor()
        self.tracker = ControlTracker()
        self.tracker.observe(self.monitor, 0.0)         # baseline of this monitor
        self.label, self.mock, self.policy, self.error, self.missing = label, mock, policy or {}, error, missing

    def feed(self, line: str, now: float) -> None:
        self.monitor.feed(line, now)
        self.tracker.observe(self.monitor, now)

    def inputs(self) -> ControlInputs:
        return ControlInputs(self.monitor, self.label, self.error, self.mock, self.policy, self.missing)

    def status(self, now: float, record_path=None):
        return cs.build_control_status(self.inputs(), self.tracker, now, 1_000_000.0, record_path)

    def readiness(self, now: float):
        self.tracker.observe(self.monitor, now)
        return cs.control_readiness(self.inputs(), self.tracker, now)

    def authority(self, now: float):
        return cs.authority_view(self.inputs(), now)

    def handoff(self, now: float):
        return cs.handoff_state(self.inputs(), now)


def live_rig(now=1.0, st=None, ch=None, hello=True, **kwargs) -> Rig:
    rig = Rig(**kwargs)
    if hello:
        rig.feed(HELLO, now)
    rig.feed(ch_line(**(ch or {})), now)
    rig.feed(status_line(5000, **(st or RADIO)), now)
    return rig


def item(items, label):
    return next(i for i in items if i.label == label)


def write_record(path: Path, firmware=BASELINE_FIRMWARE, age_s=600.0, now_unix=1_000_000.0, failing=(), unconfirmed=()) -> Path:
    checks = {name: {"controller_pass": name not in failing, "operator_confirmed": name not in failing and name not in unconfirmed,
                     "controller_evidence": f"{name} evidence"} for name in mc.REQUIRED_CHECKS}
    path.write_text(json.dumps([{"firmware": firmware, "finished_unix": now_unix - age_s, "checks": checks}]), encoding="utf-8")
    return path


ALL_CONFIRMED = {key: True for key in cs.AUTONOMOUS_CONFIRMATIONS}


# ================================================================ pure model: authority, ELRS, firmware


def test_known_firmware_ids_match_the_tested_client():
    assert set(cs.KNOWN_FIRMWARE) == set(client.FIRMWARE_IDS)
    assert cs.PORT_PATTERN.match("COM5") and not cs.PORT_PATTERN.match("COM5 && calc")


def test_scripted_source_replay_drives_the_model_like_the_hub_does():
    clock = {"t": 0.0}
    script = [(0.0, "RX", HELLO), (0.0, "RX", ch_line()), (0.0, "RX", status_line(100, **RADIO)),
              (0.2, "RX", status_line(300, **RADIO))]
    source = ScriptedSource(script, clock=lambda: clock["t"])
    rig = Rig(label=source.label, mock=True)
    for t in (0.0, 0.25):
        clock["t"] = t
        for direction, line in source.read():
            rig.feed(line, 10.0 + t) if direction == "RX" else None
    view = rig.authority(10.3)
    assert view.banner.text == "MOCK · MANUAL" and "MOCK" in view.controller_line.source
    assert rig.status(10.3).mock


def test_banner_shows_manual_not_driving_and_pc_commands_without_a_pc_mode():
    rig = live_rig(st=RADIO)
    manual = rig.authority(1.1)
    assert manual.banner.text == "MANUAL" and manual.banner.level == "ok"
    assert "source=RADIO" in manual.controller_line.text and "requested=RADIO" in manual.controller_line.text
    assert [m.label for m in manual.modes] == ["MANUAL", "AUTONOMOUS", "SHADOW"]
    assert manual.modes[1].text.startswith("DISABLED") and manual.modes[2].text.startswith("off")
    assert "without this computer" in manual.manual_note

    idle = live_rig(st=DISARMED).authority(1.1)
    assert idle.banner.text == "NOT DRIVING" and "DISARMED" in idle.banner.detail and "selected" in idle.modes[0].text

    pc = live_rig(st=PC_ACTIVE).authority(1.1)
    assert pc.banner.text == "PC-ORIGINATED COMMANDS ACTIVE" and pc.banner.level == "warn"
    assert "AUTONOMOUS mode, which is disabled" in pc.banner.detail
    assert pc.modes[0].level == "warn" and pc.modes[1].text.startswith("DISABLED")


def test_shadow_is_only_ever_display_only_or_off_and_never_a_driving_state():
    for mode, word in (("LIVE SHADOW", "ACTIVE"), ("SIM", "ACTIVE"), ("OFF", "off")):
        shadow = live_rig(policy={"mode": mode}).authority(1.1).modes[2]
        assert shadow.text.startswith(word) or word == "off"
        assert "display only" in shadow.text or shadow.text.startswith("off")
        assert "non-actuating" in shadow.text
    assert live_rig(policy={"mode": "LIVE SHADOW"}).authority(1.1).modes[1].text.startswith("DISABLED")


def test_unknown_and_stale_are_never_shown_as_a_source():
    none = Rig(label=None).authority(1.0)
    assert none.banner.text == "UNKNOWN" and not none.controller_line.available
    waiting = Rig().authority(1.0)
    assert waiting.banner.text == "UNKNOWN" and "NOT AVAILABLE" in waiting.controller_line.text
    rig = live_rig()
    stale = rig.authority(5.0)
    assert stale.banner.text == "STALE" and stale.banner.level == "bad"
    assert "RADIO" not in stale.banner.text and stale.controller_line.text.startswith("NOT AVAILABLE")


def test_elrs_items_from_the_ch_and_st_lines_with_source_and_age():
    rig = live_rig(now=1.0, ch=dict(lq=97, rc_age=12, frames=500, crc=1, len=0))
    items = cs.elrs_items(rig.inputs(), rig.tracker, 1.2)
    assert item(items, "ELRS link (ST link)").text.startswith("UP")
    lq = item(items, "Link quality (CH lq)")
    assert lq.text == "97 %" and lq.source == "controller CH line" and lq.age_s == pytest.approx(0.2)
    assert item(items, "Newest RC frame age (CH rc_age)").text == "12 ms"
    counters = item(items, "RC frame counters (CH)")
    assert "500 good" in counters.text and "1 bad CRC" in counters.text and counters.level == "warn"
    assert item(items, "Receiver state (derived)").text.startswith("delivering valid RC frames")
    assert "reports no state" in item(items, "Receiver state (derived)").source
    assert item(items, "ARM switch as read (CH arm)").text == "CH5:1792:ARM"
    assert "no radio loss or fault" in item(items, "Failsafe / radio loss (ST fault, stop, block)").text


def test_rssi_tx_power_battery_and_temperature_are_never_numbers():
    items = cs.elrs_items(live_rig().inputs(), live_rig().tracker, 1.1)
    for label in ("RSSI", "TX power", "Battery voltage", "Receiver temperature"):
        found = item(items, label)
        assert found.text == NOT_SENT_TEXT and not found.available
        assert not any(ch.isdigit() for ch in found.text)
    assert NOT_SENT_TEXT == "Not available: the firmware does not send it"


def test_lq_minus_one_and_missing_ch_are_not_available_with_a_reason():
    rig = live_rig(ch=dict(lq=-1, rc_age=-1))
    items = cs.elrs_items(rig.inputs(), rig.tracker, 1.1)
    assert item(items, "Link quality (CH lq)").text.startswith("NOT AVAILABLE: the receiver has not sent link statistics")
    assert item(items, "Newest RC frame age (CH rc_age)").text.startswith("NOT AVAILABLE: no valid RC frame since boot")
    # no CH line at all
    rig = Rig()
    rig.feed(status_line(5000, **RADIO), 1.0)
    items = cs.elrs_items(rig.inputs(), rig.tracker, 1.1)
    assert item(items, "Link quality (CH lq)").text == "NOT AVAILABLE: no CH line received"
    assert item(items, "Receiver state (derived)").text.startswith("link qualified per ST; frame age not confirmed")


def test_a_ch_line_older_than_this_window_has_an_unknown_age_and_a_stale_one_is_not_shown():
    monitor = ControllerMonitor()
    monitor.feed(ch_line(), 0.5)                       # arrived before the tracker looked
    monitor.feed(status_line(5000, **RADIO), 0.5)
    tracker = ControlTracker()
    tracker.observe(monitor, 1.0)
    inputs = ControlInputs(monitor, "COM5 (listen-only)")
    unknown = cs.elrs_items(inputs, tracker, 1.1)
    assert "age is unknown" in item(unknown, "Link quality (CH lq)").text
    monitor.feed(ch_line(lq=90), 2.0)
    monitor.feed(status_line(6000, **RADIO), 2.0)
    tracker.observe(monitor, 2.0)
    assert item(cs.elrs_items(inputs, tracker, 2.1), "Link quality (CH lq)").text == "90 %"
    monitor.feed(status_line(7000, **RADIO), 9.9)
    late = cs.elrs_items(inputs, tracker, 10.0)
    assert "old" in item(late, "Link quality (CH lq)").text and "90" not in item(late, "Link quality (CH lq)").text


def test_radio_loss_and_latched_fault_come_from_the_st_fields():
    rig = live_rig(st=LOST)
    items = cs.elrs_items(rig.inputs(), rig.tracker, 1.1)
    failsafe = item(items, "Failsafe / radio loss (ST fault, stop, block)")
    assert failsafe.level == "bad" and failsafe.text.startswith("RADIO LOST") and "ARM off, then on" in failsafe.text
    assert item(items, "ELRS link (ST link)").text.startswith("DOWN")
    assert item(items, "Receiver state (derived)").text == "radio link lost (block=RADIO_LOST)"
    stale = dict(armed=0, req="PC", act="NONE", block="REARM_REQUIRED", fault="PC_STALE", stop="PC_STALE")
    pc_fault = cs.elrs_items(live_rig(st=stale).inputs(), live_rig(st=stale).tracker, 1.1)
    assert "FAULT LATCHED" in item(pc_fault, "Failsafe / radio loss (ST fault, stop, block)").text
    history = dict(DISARMED, stop="DISARM")
    text = item(cs.elrs_items(live_rig(st=history).inputs(), live_rig(st=history).tracker, 1.1),
                "Failsafe / radio loss (ST fault, stop, block)").text
    assert "history, not a current fault" in text


def test_everything_is_not_available_when_telemetry_is_stale_or_absent():
    rig = live_rig()
    for items in (cs.elrs_items(rig.inputs(), rig.tracker, 9.0), cs.elrs_items(Rig(label=None).inputs(), Rig().tracker, 1.0)):
        for label in ("ELRS link (ST link)", "Receiver state (derived)", "Failsafe / radio loss (ST fault, stop, block)"):
            assert item(items, label).text.startswith("NOT AVAILABLE:") and not item(items, label).available


def test_firmware_identity_uptime_and_line_counters():
    rig = live_rig(now=1.0)
    rig.feed("garbage line", 1.0)
    rig.feed("ST t=1 tok=1", 1.0)                      # malformed ST: counted, not parsed
    items = cs.firmware_items(rig.inputs(), rig.tracker, 1.3)
    assert item(items, "Firmware id (HELLO fw)").text == BASELINE_FIRMWARE
    assert "known build" in item(items, "Is it a build this repository knows?").text
    assert item(items, "Uptime (ST t=)").text == "5.0 s"
    assert item(items, "Time since the last status line").text == "0.3 s"
    assert item(items, "Unrecognised or malformed lines").text == "2 of 5 lines"
    assert "none seen" in item(items, "Restarts").text and "not 'none happened'" in item(items, "Restarts").detail
    stale = item(cs.firmware_items(rig.inputs(), rig.tracker, 5.0), "Uptime (ST t=)")
    assert "NOT current" in stale.text and stale.level == "warn"


def test_unidentified_and_unknown_firmware_are_labelled_as_such():
    rig = live_rig(hello=False)
    items = cs.firmware_items(rig.inputs(), rig.tracker, 1.1)
    assert item(items, "Firmware id (HELLO fw)").text.startswith("NOT AVAILABLE: no HELLO line received")
    assert item(items, "Is it a build this repository knows?").text.startswith("NOT AVAILABLE")
    rig.feed("HELLO fw=mystery-9 pc_limit=128", 1.2)
    unknown = item(cs.firmware_items(rig.inputs(), rig.tracker, 1.3), "Is it a build this repository knows?")
    assert "NOT a build this repository knows" in unknown.text and unknown.level == "warn"
    rig.feed(f"HELLO fw={WIFI_FIRMWARE}", 1.4)
    wifi = item(cs.firmware_items(rig.inputs(), rig.tracker, 1.5), "Is it a build this repository knows?")
    assert "route D" in wifi.text and wifi.level == "warn"


def test_power_is_never_a_measurement():
    power = cs.power_item()
    assert power.text == ("no measurement: the firmware has no voltage sensing; use the multimeter checks PWR-1/PWR-2 in the playbook")
    assert not power.available and "USB and the battery together" in power.detail


def test_restarts_are_counted_when_uptime_goes_backwards_and_fail_readiness():
    rig = live_rig(now=1.0)
    assert rig.tracker.restarts == 0
    rig.feed(status_line(6200, **RADIO), 2.0)
    rig.feed(status_line(300, **RADIO), 3.0)           # fell from 6.2 s to 0.3 s
    assert rig.tracker.restarts == 1
    rig.feed(status_line(9000, **RADIO), 4.0)
    rig.feed(status_line(120, **RADIO), 5.0)
    assert rig.tracker.restarts == 2
    text = item(cs.firmware_items(rig.inputs(), rig.tracker, 5.1), "Restarts")
    assert text.text.startswith("controller restarted 2 times this session") and text.level == "bad"
    assert "9.0 s to 0.1 s" in text.text
    rig.feed(ch_line(), 5.0)
    verdict = rig.readiness(5.2)
    assert verdict.level == FAILED and "restarted 2 times" in verdict.headline and "PWR-1" in verdict.fix
    rig.feed(status_line(120000, **RADIO), 1000.0)      # long after: still shown, no longer FAILED
    rig.feed(ch_line(), 1000.0)
    later = rig.readiness(1000.1)
    assert later.level == WARNING and "restarted 2 times earlier" in later.headline
    assert item(cs.firmware_items(rig.inputs(), rig.tracker, 1000.1), "Restarts").level == "warn"


def test_a_steady_clock_a_repeated_line_or_a_reconnect_is_not_a_restart():
    rig = live_rig(now=1.0)
    for t, ms in ((2.0, 5200), (2.1, 5200), (2.2, 5250)):
        rig.feed(status_line(ms, **RADIO), t)
    assert rig.tracker.restarts == 0
    # the hub makes a new monitor for every source: its first line is a new baseline, not a restart
    rig.monitor = ControllerMonitor()
    rig.tracker.observe(rig.monitor, 3.0)
    rig.feed(status_line(4000, **RADIO), 3.1)
    assert rig.tracker.restarts == 0 and rig.tracker.source_changes == 1
    assert "Telemetry source changes" in [i.label for i in cs.firmware_items(rig.inputs(), rig.tracker, 3.2)]
    rig.feed(status_line(5, **RADIO), 3.3)              # but going backwards on the same connection is
    assert rig.tracker.restarts == 1


# ================================================================ pure model: communication, handoff, transports


def test_communication_items():
    rig = live_rig(st=dict(PC_ACTIVE, pc_age=40))
    items = cs.communication_items(rig.inputs(), 1.1)
    assert item(items, "Telemetry source").text == "COM5 (listen-only)"
    assert item(items, "PC command age (ST pc_age)").text == "40 ms"
    late = live_rig(st=dict(PC_ACTIVE, pc_age=400))
    assert item(cs.communication_items(late.inputs(), 1.1), "PC command age (ST pc_age)").level == "bad"
    none = cs.communication_items(Rig(label=None).inputs(), 1.0)
    assert none[0].text.startswith("NOT AVAILABLE: no controller telemetry source connected")
    errored = cs.communication_items(Rig(error="could not open COM5").inputs(), 1.0)
    assert item(errored, "Telemetry source error").level == "bad"
    rig.monitor.feed("D,3,60,0", 1.05, "TX")
    assert item(cs.communication_items(rig.inputs(), 1.1), "PC drive commands seen sent (client log)").level == "warn"


def test_handoff_single_source_and_stopped_states():
    manual = live_rig(st=RADIO).handoff(1.1)
    assert manual.phase == "MANUAL" and manual.source == "RADIO" and not manual.conflicts
    pc = live_rig(st=PC_ACTIVE).handoff(1.1)
    assert pc.phase == "PC_COMMANDS" and pc.source == "PC" and "does not start them" in pc.headline
    waiting = live_rig(st=dict(armed=1, req="PC", act="NONE", block="WAIT_PC_NEUTRAL", stop="HANDOFF")).handoff(1.1)
    assert waiting.phase == "STOPPED_WAITING_NEUTRAL" and waiting.source is None and "Both sources stopped" in waiting.headline
    assert "asks for PC" in waiting.transition
    idle = live_rig(st=DISARMED).handoff(1.1)
    assert idle.phase == "STOPPED_IDLE" and idle.source is None
    failsafe = live_rig(st=LOST).handoff(1.1)
    assert failsafe.phase == "STOPPED_FAILSAFE" and "ARM off, then on" in failsafe.headline
    unconfirmed = live_rig().handoff(9.0)
    assert unconfirmed.phase == "UNCONFIRMED" and unconfirmed.source is None
    assert Rig(label=None).handoff(1.0).phase == "UNCONFIRMED"


def test_handoff_rules_describe_the_firmware_and_the_way_back_to_manual():
    state = live_rig().handoff(1.1)
    text = " ".join(state.rules)
    for needle in ("Only one source", "both stopped, neutral confirmed", "250 ms", "0.6 s", "ARM off, then on", "display rule, not a motor rule"):
        assert needle in text, needle
    assert state.return_to_manual == ("Return to MANUAL is always available: set MODE to RADIO, or switch ARM off, on the RadioMaster. "
                                      "Neither depends on this computer.")


@pytest.mark.parametrize("status, needle", [
    (dict(RADIO, pc_age=120), "a PC client is sending commands"),
    (dict(armed=1, req="RADIO", act="PC", pc_ok=1, block="NONE", pc_age=5), "asks for RADIO"),
    (dict(armed=1, req="PC", act="RADIO", block="NONE", pc_age=-1), "asks for PC"),
    (dict(PC_ACTIVE, pc_age=-1), "no PC command since boot"),
    (dict(RADIO, link=0), "radio link down"),
])
def test_conflicting_sources_are_flagged(status, needle):
    rig = live_rig(st=status)
    state = rig.handoff(1.1)
    assert state.phase == "CONFLICT" and state.source is None and state.level == "bad"
    assert any(needle in c for c in state.conflicts), state.conflicts
    assert rig.readiness(1.1).level in (WARNING, FAILED)   # never Ready while two sources could be live


def test_a_pc_client_log_seen_while_radio_is_selected_is_a_conflict():
    rig = live_rig(st=RADIO)
    rig.monitor.feed("D,2,60,0", 1.0, "TX")
    state = rig.handoff(1.3)
    assert state.phase == "CONFLICT" and any("client log shows a drive command" in c for c in state.conflicts)
    rig.feed(status_line(5400, **RADIO), 3.0)
    assert rig.handoff(3.1).phase == "MANUAL"


def test_transport_inventory_keeps_each_state_as_documented():
    rows = {r.route: r for r in cs.transport_inventory(live_rig().inputs(), 1.1)}
    assert list(rows) == ["RADIO", "A", "B", "C", "D", "AI SHADOW"]
    assert rows["RADIO"].evidence == "HARDWARE" and "not logged" in rows["RADIO"].detail.lower() and "reported" in rows["RADIO"].detail.lower()
    assert rows["A"].evidence == "HARDWARE" and "tethered" in rows["A"].state and "no recorded hardware result" in rows["A"].detail
    assert rows["B"].evidence == "PROPOSED" and "output-only" in rows["B"].detail and "trainer input" in rows["B"].detail
    assert rows["C"].state == "Rejected for now" and rows["C"].evidence == "REJECTED"
    assert "Experimental" in rows["D"].state and "restart" in rows["D"].detail and "not the default" in rows["D"].state
    assert "motors never driven by the AI" in rows["AI SHADOW"].state
    assert rows["AI SHADOW"].evidence not in ("HARDWARE", "PROPOSED")
    assert rows["A"].seen_now == "telemetry listener: COM5 (listen-only)" and "link=1" in rows["RADIO"].seen_now
    mock = {r.route: r for r in cs.transport_inventory(Rig(label="MOCK controller replay", mock=True).inputs(), 1.0)}
    assert mock["RADIO"].seen_now == "MOCK telemetry, not hardware" and "MOCK session" in mock["RADIO"].detail
    shadow = {r.route: r for r in cs.transport_inventory(Rig(missing=["no live tracker"]).inputs(), 1.0)}["AI SHADOW"]
    assert "BLOCKED" in shadow.detail and "no live tracker" in shadow.detail


# ================================================================ pure model: readiness


def test_readiness_not_configured_without_a_controller_source():
    verdict = Rig(label=None).readiness(1.0)
    assert verdict.level == NOT_CONFIGURED and "SETUP > Controller & IMU" in verdict.fix


def test_readiness_ready_with_fresh_qualified_link_and_consistent_authority():
    for st in (RADIO, PC_ACTIVE, DISARMED):
        rig = live_rig(st=st)
        verdict = rig.readiness(1.1)
        assert verdict.level == READY, (st, verdict)
    verdict = live_rig(st=RADIO).readiness(1.1)
    assert "RADIO has control" in verdict.headline and BASELINE_FIRMWARE in verdict.detail
    assert "not identified" in live_rig(hello=False).readiness(1.1).detail


def test_readiness_warning_when_stale_waiting_partial_or_switch_unclear():
    assert live_rig().readiness(5.0).level == WARNING
    assert "stale" in live_rig().readiness(5.0).headline
    waiting = Rig()
    assert waiting.readiness(1.0).level == WARNING and "first controller status" in waiting.readiness(1.0).headline
    partial = Rig()
    partial.feed(status_line(5000, **RADIO), 1.0)
    assert partial.readiness(1.1).level == WARNING and "Partial telemetry" in partial.readiness(1.1).headline
    for block, fix in (("ARM_ON_AT_BOOT", "ARM off, then on"), ("MODE_SWITCH_INVALID", "MODE switch")):
        verdict = live_rig(st=dict(armed=0, req="NONE", act="NONE", block=block)).readiness(1.1)
        assert verdict.level == WARNING and fix in verdict.fix
    assert live_rig(ch=dict(rc_age=-1)).readiness(1.1).level == WARNING


def test_readiness_failed_on_fault_radio_lost_or_source_error():
    fault = live_rig(st=dict(armed=0, req="PC", act="NONE", block="REARM_REQUIRED", fault="PC_STALE", stop="PC_STALE")).readiness(1.1)
    assert fault.level == FAILED and "PC_STALE" in fault.headline and "ARM off, then on" in fault.fix
    lost = live_rig(st=LOST).readiness(1.1)
    assert lost.level == FAILED and lost.headline == "Radio lost"
    nothing = live_rig(st=dict(link=0, armed=0, req="NONE", act="NONE", block="NO_RADIO")).readiness(1.1)
    assert nothing.level == FAILED and "No qualified radio link" in nothing.headline
    assert Rig(error="could not open COM5").readiness(1.0).level == FAILED


def test_mock_telemetry_is_labelled_and_never_ready():
    verdict = live_rig(label="MOCK controller replay", mock=True).readiness(1.1)
    assert verdict.level == WARNING and verdict.headline.startswith("MOCK · ") and "never Ready" in verdict.detail
    failed = live_rig(label="MOCK controller replay", mock=True, st=LOST).readiness(1.1)
    assert failed.level == FAILED and failed.headline.startswith("MOCK · ")


# ================================================================ pure model: the gates


def gate(tmp_path, **overrides):
    kwargs = dict(firmware=BASELINE_FIRMWARE, now_unix=1_000_000.0, record_path=write_record(tmp_path / "safety.json"),
                  command_path_verified=True, observation_verified=True, confirmations=ALL_CONFIRMED)
    kwargs.update(overrides)
    return cs.autonomous_gate(**kwargs)


def test_the_autonomous_gate_is_closed_for_the_current_repo_state(tmp_path):
    assert cs.COMMAND_PATH_VERIFIED is False and cs.OBSERVATION_PIPELINE_VERIFIED is False
    assert cs.AUTONOMOUS_ENABLEABLE_IN_THIS_BUILD is False
    record = write_record(tmp_path / "safety.json")                # even with all five checks fresh on the connected firmware
    allowed, reasons = cs.autonomous_gate(firmware=BASELINE_FIRMWARE, now_unix=1_000_000.0, record_path=record,
                                          confirmations=ALL_CONFIRMED)
    assert allowed is False
    joined = "\n".join(reasons)
    assert "(a)" in joined and "USB tether is demonstrated for the normal path only" in joined and "Wi-Fi was unstable" in joined
    assert "PROPOSED and unimplemented" in joined and "(d)" in joined and "live shadow is BLOCKED" in joined
    assert "(c)" not in joined and "(b)" not in joined and "(e)" not in joined
    assert cs.autonomous_enable_offered(True) is False


def test_the_autonomous_gate_logic_with_every_prerequisite_met_and_each_one_missing(tmp_path):
    allowed, reasons = gate(tmp_path)
    assert allowed is True and reasons == []
    cases = {
        "(a)": dict(command_path_verified=False),
        "(b)": dict(record_path=write_record(tmp_path / "b.json", failing=("takeover",))),
        "(c)": dict(record_path=write_record(tmp_path / "c.json", failing=("stop",))),
        "(c) old": dict(record_path=write_record(tmp_path / "old.json", age_s=13 * 3600)),
        "(c) other firmware": dict(record_path=write_record(tmp_path / "fw.json", firmware="autobot-handoff-wifi-1")),
        "(d)": dict(observation_verified=False),
        "(e)": dict(confirmations={**ALL_CONFIRMED, "power_cutoff": False}),
        "(e) none": dict(confirmations=None),
        "no firmware": dict(firmware=None),
        "mock": dict(mock=True),
    }
    for name, change in cases.items():
        allowed, reasons = gate(tmp_path, **change)
        assert allowed is False and reasons, name
    assert any(r.startswith("(b)") for r in gate(tmp_path, **cases["(b)"])[1])
    assert any(r.startswith("(c)") for r in gate(tmp_path, **cases["(c) old"])[1])
    assert any("MOCK" in r for r in gate(tmp_path, mock=True)[1])
    assert "power cut-off" in " ".join(gate(tmp_path, **cases["(e)"])[1])
    # a passing gate in the pure function still does not make the GUI offer the mode
    assert cs.autonomous_enable_offered(gate(tmp_path)[0]) is False


def test_unconfirmed_operator_answer_in_the_safety_record_keeps_the_gate_closed(tmp_path):
    record = write_record(tmp_path / "s.json", unconfirmed=("takeover",))
    allowed, reasons = gate(tmp_path, record_path=record)
    assert not allowed and any("operator did not confirm" in r for r in reasons)


def test_control_status_carries_an_always_closed_autonomous_box(tmp_path):
    record = write_record(tmp_path / "safety.json", now_unix=1_000_000.0)
    rig = live_rig(st=RADIO)
    box = rig.status(1.1, record).autonomous
    assert box.allowed is False and box.enable_offered is False
    assert [p.key for p in box.prerequisites] == ["command_path", "takeover", "safety_checks", "observation", "confirmations"]
    assert [p.met for p in box.prerequisites] == [False, True, True, False, False]
    assert len(box.reasons) == 3
    assert "Missing now" not in "".join(box.reasons)
    missing = live_rig(missing=["opponent position: there is no live opponent tracker"]).status(1.1, record).autonomous
    assert "Missing now: opponent position" in " ".join(missing.reasons)


def test_safety_rows_report_state_and_age_from_the_record(tmp_path):
    record = write_record(tmp_path / "s.json", failing=("radio_loss",), unconfirmed=("stop",))
    rows, note = cs.safety_rows(BASELINE_FIRMWARE, 1_000_000.0, record)
    states = {r.name: r.state for r in rows}
    assert states == {"stop": "FAIL", "disarm": "PASS", "takeover": "PASS", "pc_link_loss": "PASS", "radio_loss": "FAIL"}
    assert rows[1].age_text == "0.2 h ago" and "refused" in note and "radio_loss" in note
    all_pass, note = cs.safety_rows(BASELINE_FIRMWARE, 1_000_000.0, write_record(tmp_path / "ok.json"))
    assert {r.state for r in all_pass} == {"PASS"} and note.startswith("FLOOR RUN PERMITTED")
    none, note = cs.safety_rows(BASELINE_FIRMWARE, 1_000_000.0, tmp_path / "missing.json")
    assert {r.state for r in none} == {"NOT RUN"} and "no wheels-raised safety checks are recorded" in note
    unknown, note = cs.safety_rows(None, 1_000_000.0, record)
    assert {r.state for r in unknown} == {"NOT RUN"} and "not identified itself" in note


def test_plan_and_simulator_reference_are_computed_in_process_without_hardware():
    plan, reference = cs.compute_plan_and_reference()
    rows = cs.motion_plan_rows(plan, reference)
    assert len(rows) == len(plan.steps) and rows[1][1] == "advance"
    assert reference["evidence"] == "SIM" and rows[1][8] != "-"
    assert cs.motion_plan_rows(plan, None)[1][6] == "-"


# ================================================================ pure model: console command lines


def test_the_console_command_lines_match_motion_check_s_argparse():
    parser = mc.build_parser()
    checks = cs.checks_command("COM5", "usb-only")
    assert checks[0] == sys.executable and checks[1].endswith("motion_check.py") and Path(checks[1]) == ROOT / "scripts" / "motion_check.py"
    args = parser.parse_args(checks[2:])
    assert (args.command, args.port, args.power, args.wifi) == ("checks", "COM5", "usb-only", None)
    floor = cs.floor_command("COM6", "battery-and-usb")
    args = parser.parse_args(floor[2:])
    assert (args.command, args.port, args.surface, args.power) == ("run", "COM6", "floor", "battery-and-usb")
    assert all("--wifi" not in command for command in (checks, floor))


def test_port_power_and_port_conflict_validation():
    assert cs.validate_port("COM5") is None and cs.validate_port(" com12 ") is None
    for bad in ("", "  ", "COM", "COM5; calc", "192.168.4.1", "COM5 COM6"):
        assert cs.validate_port(bad)
    assert cs.power_problem("usb-only") is None and cs.power_problem("battery-and-usb") is None
    assert "not possible over USB" in cs.power_problem("battery-only")
    assert cs.power_problem(None) and cs.power_problem("solar")
    assert cs.port_held_by_hub("COM5 (listen-only)", None, "com5")
    assert not cs.port_held_by_hub("COM50 (listen-only)", None, "COM5")
    assert not cs.port_held_by_hub("COM5 (listen-only)", "port gone", "COM5")
    assert not cs.port_held_by_hub("client log x.txt", None, "COM5") and not cs.port_held_by_hub(None, None, "COM5")


def blockers(kind="checks", **overrides):
    ticks = cs.FLOOR_CONFIRMATIONS if kind == "floor" else cs.CHECKS_CONFIRMATIONS
    kwargs = dict(port="COM5", power="usb-only", confirmations={key: True for key in ticks}, link_label=None, link_error=None,
                  mock=False, process_running=False, firmware=BASELINE_FIRMWARE, now_unix=1_000_000.0)
    kwargs.update(overrides)
    return cs.launch_blockers(kind, **kwargs)


def test_launch_blockers_each_requirement_keeps_the_button_disabled(tmp_path):
    assert blockers() == []
    assert any("select the controller's COM port" in r for r in blockers(port=""))
    assert any("select how the robot is powered" in r for r in blockers(power=None))
    assert any("battery-only" in r for r in blockers(power="battery-only"))
    assert any("tick every confirmation" in r and "wheels" in r for r in blockers(confirmations={}))
    assert any("listening on COM5" in r for r in blockers(link_label="COM5 (listen-only)"))
    assert any("MOCK" in r for r in blockers(mock=True))
    assert any("still running" in r for r in blockers(process_running=True))
    missing_floor = blockers("floor", record_path=tmp_path / "none.json")
    assert any("floor run refused" in r and "no wheels-raised safety checks" in r for r in missing_floor)
    ok_floor = blockers("floor", record_path=write_record(tmp_path / "ok.json"))
    assert ok_floor == []
    assert any("refused" in r for r in blockers("floor", record_path=write_record(tmp_path / "old.json", age_s=13 * 3600)))
    assert any("firmware has not identified itself" in r for r in blockers("floor", firmware=None, record_path=tmp_path / "ok.json"))


# ================================================================ the screen


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def make_hub(qapp, tmp_path, mock):
    return OperationsHub(load_config(ROOT / "config" / "control_center.yaml"),
                         HubOptions(mock=mock, start_camera=False, rounds_root=tmp_path / "rounds"))


@pytest.fixture()
def hub(qapp, tmp_path):
    hub = make_hub(qapp, tmp_path, True)
    yield hub
    hub.close()


@pytest.fixture()
def live_hub(qapp, tmp_path):
    hub = make_hub(qapp, tmp_path, False)
    yield hub
    hub.close()


class FakePopen:
    instances: list["FakePopen"] = []

    def __init__(self, args, **kwargs):
        self.args, self.kwargs = list(args), kwargs
        self.pid = 4242
        self.code = None
        self.terminated = 0
        FakePopen.instances.append(self)

    def poll(self):
        return self.code

    def terminate(self):
        self.terminated += 1
        self.code = 1

    def wait(self, timeout=None):
        return self.code

    def kill(self):
        self.code = -9


@pytest.fixture()
def fake_popen(monkeypatch):
    FakePopen.instances = []
    monkeypatch.setattr(control_module.subprocess, "Popen", FakePopen)
    return FakePopen


def pump(seconds=0.1):
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        QCoreApplication.processEvents()
        time.sleep(0.005)


def make_screen(hub, tmp_path, **kwargs):
    kwargs.setdefault("record_path", tmp_path / "safety.json")
    kwargs.setdefault("reference_runner", lambda: (_ for _ in ()).throw(RuntimeError("not computed in this test")))
    return ControlScreen(hub, **kwargs)


def feed_live(hub, line_status=RADIO, firmware=True):
    now = time.monotonic()
    if firmware:
        hub.monitor.feed(HELLO, now)
    hub.monitor.feed(ch_line(), now)
    hub.monitor.feed(status_line(5000, **line_status), now)


def all_texts(widget):
    texts = [label.text() for label in widget.findChildren(QLabel)] + [b.text() for b in widget.findChildren(QPushButton)]
    texts += [box.text() for box in widget.findChildren(QCheckBox)]
    for table in widget.findChildren(control_module.QTableWidget):
        texts += [table.item(r, c).text() for r in range(table.rowCount()) for c in range(table.columnCount()) if table.item(r, c)]
    return texts


def test_screen_constructs_in_mock_mode_with_labelled_sections(qapp, hub, tmp_path, fake_popen):
    screen = make_screen(hub, tmp_path)
    try:
        assert control_module.TITLE == "ELRS & CONTROL"
        texts = "\n".join(all_texts(screen))
        assert "MOCK SESSION" in texts and screen.mock_badge.isHidden() is False
        for needle in ("OPERATING MODE", "ELRS LINK AND RECEIVER", "FIRMWARE AND RESTARTS", "TRANSPORT INVENTORY", "MANUAL  ·  RADIOMASTER",
                       "AUTONOMOUS MODE  ·  DISABLED", "MOTION CHECK"):
            assert needle in texts, needle
        assert "The RadioMaster works without this computer. If this window or the PC crashes, manual ELRS driving is unaffected: " \
               "the Command Centre is listen-only on the controller link" in texts
        assert "MODE to RADIO, or switch ARM off" in texts
        assert "UNKNOWN" in screen.banner.text()
        assert screen.readiness().level == NOT_CONFIGURED and "SETUP > Controller & IMU" in screen.readiness().fix
        assert screen.transport_table.rowCount() == 6
        assert screen.elrs_table.rowCount() == 12 and NOT_SENT_TEXT in texts
        assert "no measurement: the firmware has no voltage sensing" in texts
        assert screen.autonomous_button.isEnabled() is False and "not available in this build" in screen.autonomous_button.text()
        assert FakePopen.instances == []
    finally:
        screen.shutdown()


def test_there_are_no_pc_control_or_ai_mode_boxes_and_no_keyboard_drive_button(qapp, hub, tmp_path):
    screen = make_screen(hub, tmp_path)
    try:
        texts = "\n".join(all_texts(screen))
        for forbidden in ("PC CONTROL", "keyboard drive", "Open the supervised keyboard", "AI MODE", "AI CONTROL"):
            assert forbidden not in texts, forbidden
        buttons = [b.text() for b in screen.findChildren(QPushButton)]
        assert sorted(buttons) == sorted(["Enable AUTONOMOUS (not available in this build)", "Compute plan and simulator reference",
                                          "Open supervised safety checks in a console", "Open supervised FLOOR run in a console",
                                          "Stop the console"])
        assert [m.text().split("\n")[0] for m in screen.mode_chips] == ["MANUAL", "AUTONOMOUS", "SHADOW"]
    finally:
        screen.shutdown()


def test_the_screen_shows_live_telemetry_with_the_controllers_own_words(qapp, hub, tmp_path, fake_popen):
    hub.link_label = "MOCK controller replay"
    screen = make_screen(hub, tmp_path)
    try:
        screen.refresh()                                   # baseline, then lines arrive
        feed_live(hub)
        screen.refresh()
        assert screen.banner.text() == "MOCK · MANUAL"
        assert "source=RADIO requested=RADIO" in screen.controller_line.text()
        assert screen.handoff_headline.text().startswith("Single command source: RADIO")
        assert screen.readiness_chip.text() == "WARNING" and "MOCK" in screen.readiness_text.text()
        assert screen.mode_chips[1].text().startswith("AUTONOMOUS\nDISABLED")
        hub.policy = {"mode": "LIVE SHADOW", "input": "x", "checkpoint": "c", "reason": "", "proposed": None}
        screen.refresh()
        assert "ACTIVE" in screen.mode_chips[2].text() and "non-actuating" in screen.mode_chips[2].text()
        assert screen.autonomous_button.isEnabled() is False
        assert FakePopen.instances == []
    finally:
        screen.shutdown()


def test_the_autonomous_box_lists_every_unmet_prerequisite_and_cannot_be_enabled(qapp, hub, tmp_path, monkeypatch):
    screen = make_screen(hub, tmp_path)
    try:
        rows = [[screen.prereq_table.item(r, c).text() for c in range(3)] for r in range(screen.prereq_table.rowCount())]
        assert [r[0][:3] for r in rows] == ["(a)", "(b)", "(c)", "(d)", "(e)"]
        assert all(r[1] == "UNMET" for r in rows)
        assert "USB tether is demonstrated for the normal path only" in rows[0][2]
        # even a gate that passed could not enable the button in this build
        monkeypatch.setattr(cs, "COMMAND_PATH_VERIFIED", True)
        monkeypatch.setattr(cs, "OBSERVATION_PIPELINE_VERIFIED", True)
        screen.refresh()
        assert screen.autonomous_button.isEnabled() is False
    finally:
        screen.shutdown()


def test_nothing_launches_on_construction_refresh_readiness_or_a_finished_stage(qapp, live_hub, tmp_path, fake_popen):
    live_hub.link_label = "COM7 (listen-only)"
    write_record(tmp_path / "safety.json")
    screen = make_screen(live_hub, tmp_path)
    try:
        for _ in range(3):
            feed_live(live_hub)
            screen.refresh()
            screen.readiness()
        screen.compute_reference()                         # the worker finishing (here: failing) must not start anything
        pump(0.3)
        screen._on_reference(None, None, "")
        assert FakePopen.instances == []
        assert screen.process_label.text().startswith("No console started")
        assert not screen.checks_button.isEnabled() and not screen.floor_button.isEnabled()
    finally:
        screen.shutdown()


def tick_checks(screen):
    for box in screen.check_boxes.values():
        box.setChecked(True)


def test_console_launch_button_is_disabled_until_port_power_and_every_confirmation(qapp, live_hub, tmp_path, fake_popen):
    screen = make_screen(live_hub, tmp_path)
    try:
        assert not screen.checks_button.isEnabled()
        assert "select the controller's COM port" in screen.checks_reasons.text()
        screen.port_edit.setText("COM5")
        assert not screen.checks_button.isEnabled() and "select how the robot is powered" in screen.checks_reasons.text()
        screen.power_combo.setCurrentIndex(screen.power_combo.findData("usb-only"))
        assert not screen.checks_button.isEnabled() and "tick every confirmation" in screen.checks_reasons.text()
        boxes = list(screen.check_boxes.values())
        for box in boxes[:-1]:
            box.setChecked(True)
        assert not screen.checks_button.isEnabled()
        boxes[-1].setChecked(True)
        assert screen.checks_button.isEnabled() and "Nothing starts until you press" in screen.checks_reasons.text()
        assert FakePopen.instances == []                   # enabled is not launched
        screen.power_combo.setCurrentIndex(screen.power_combo.findData("battery-only"))
        assert not screen.checks_button.isEnabled() and "not possible over USB" in screen.checks_reasons.text()
    finally:
        screen.shutdown()


def test_launching_the_checks_console_uses_motion_check_in_a_new_console_and_tracks_it(qapp, live_hub, tmp_path, fake_popen):
    screen = make_screen(live_hub, tmp_path)
    try:
        screen.port_edit.setText("COM5")
        screen.power_combo.setCurrentIndex(screen.power_combo.findData("battery-and-usb"))
        tick_checks(screen)
        screen.checks_button.click()
        assert len(FakePopen.instances) == 1
        process = FakePopen.instances[0]
        assert process.args == [sys.executable, str(ROOT / "scripts" / "motion_check.py"), "checks", "--port", "COM5",
                                "--power", "battery-and-usb"]
        assert process.kwargs["creationflags"] == getattr(control_module.subprocess, "CREATE_NEW_CONSOLE", 0)
        assert process.kwargs["cwd"] == str(ROOT) and "shell" not in process.kwargs
        assert screen.process_label.text().startswith("RUNNING (checks) pid 4242")
        assert screen.stop_button.isEnabled() and not screen.checks_button.isEnabled() and not any(b.isChecked() for b in screen.check_boxes.values())
        assert screen.launch("checks") is False and len(FakePopen.instances) == 1      # one console at a time
        process.code = 0                                   # the console exits on its own
        screen.refresh()
        assert screen.process_label.text().startswith("EXITED (checks) with code 0")
        assert not screen.stop_button.isEnabled() and len(FakePopen.instances) == 1
    finally:
        screen.shutdown()


def test_stop_and_shutdown_terminate_the_console_and_shutdown_is_idempotent(qapp, live_hub, tmp_path, fake_popen):
    screen = make_screen(live_hub, tmp_path)
    screen.port_edit.setText("COM5")
    screen.power_combo.setCurrentIndex(screen.power_combo.findData("usb-only"))
    tick_checks(screen)
    assert screen.launch("checks")
    process = FakePopen.instances[0]
    screen.stop_process()
    assert process.terminated == 1 and "STOPPED (checks)" in screen.process_label.text() and "250 ms" in screen.process_label.text()
    tick_checks(screen)
    assert screen.launch("checks") and len(FakePopen.instances) == 2
    second = FakePopen.instances[1]
    screen.shutdown()
    assert second.terminated == 1 and not screen._timer.isActive()
    screen.shutdown()                                      # again: no error, no second terminate
    assert second.terminated == 1
    screen.refresh()                                       # a late timer tick after shutdown does nothing
    assert len(FakePopen.instances) == 2


def test_launch_refuses_when_the_hub_listener_holds_the_same_port(qapp, live_hub, tmp_path, fake_popen):
    live_hub.link_label = "COM5 (listen-only)"
    screen = make_screen(live_hub, tmp_path)
    try:
        screen.port_edit.setText("COM5")
        screen.power_combo.setCurrentIndex(screen.power_combo.findData("usb-only"))
        tick_checks(screen)
        assert not screen.checks_button.isEnabled()
        assert "listening on COM5" in screen.checks_reasons.text() and "Disconnect" in screen.checks_reasons.text()
        assert screen.launch("checks") is False and FakePopen.instances == []
        assert screen.process_label.text().startswith("Not started: this Command Centre is listening on COM5")
        screen.port_edit.setText("COM6")                   # another port is free
        assert screen.checks_button.isEnabled()
    finally:
        screen.shutdown()


def test_mock_screen_never_launches_a_console(qapp, hub, tmp_path, fake_popen):
    screen = make_screen(hub, tmp_path)
    try:
        screen.port_edit.setText("COM5")
        screen.power_combo.setCurrentIndex(screen.power_combo.findData("usb-only"))
        tick_checks(screen)
        assert not screen.checks_button.isEnabled() and "MOCK session" in screen.checks_reasons.text()
        assert screen.launch("checks") is False and FakePopen.instances == []
    finally:
        screen.shutdown()


def test_floor_run_is_only_offered_with_a_passing_safety_record_and_its_own_confirmations(qapp, live_hub, tmp_path, fake_popen):
    live_hub.monitor.feed(HELLO, time.monotonic())
    screen = make_screen(live_hub, tmp_path, unix_clock=lambda: 1_000_000.0)
    try:
        screen.port_edit.setText("COM5")
        screen.power_combo.setCurrentIndex(screen.power_combo.findData("usb-only"))
        for box in screen.floor_boxes.values():
            box.setChecked(True)
        assert not screen.floor_button.isEnabled()
        assert "floor run refused" in screen.floor_reasons.text() and "no wheels-raised safety checks" in screen.floor_reasons.text()
        assert "no override" in "\n".join(all_texts(screen))
        write_record(tmp_path / "safety.json", failing=("disarm",))
        screen.refresh()
        assert not screen.floor_button.isEnabled() and "disarm" in screen.floor_reasons.text()
        write_record(tmp_path / "safety.json")
        screen.refresh()
        assert screen.floor_button.isEnabled()
        assert FakePopen.instances == []
        screen.floor_button.click()
        assert FakePopen.instances[0].args[2:] == ["run", "--port", "COM5", "--surface", "floor", "--power", "usb-only"]
    finally:
        screen.shutdown()


def test_safety_record_state_and_age_are_shown(qapp, live_hub, tmp_path):
    write_record(tmp_path / "safety.json", failing=("radio_loss",))
    live_hub.monitor.feed(HELLO, time.monotonic())
    screen = make_screen(live_hub, tmp_path, unix_clock=lambda: 1_000_000.0)
    try:
        rows = {screen.safety_table.item(r, 0).text(): screen.safety_table.item(r, 1).text() for r in range(5)}
        assert rows == {"stop": "PASS", "disarm": "PASS", "takeover": "PASS", "pc_link_loss": "PASS", "radio_loss": "FAIL"}
        assert "Floor runs refused" in screen.safety_note.text()
    finally:
        screen.shutdown()


def test_plan_and_reference_are_computed_off_the_ui_thread_and_shown(qapp, hub, tmp_path):
    import threading

    seen = {}
    plan = SimpleNamespace(steps=[SimpleNamespace(index=0, tactic="stop", duration_s=1.0, throttle=0, steer=0, left=0, right=0),
                                  SimpleNamespace(index=1, tactic="advance", duration_s=0.6, throttle=50, steer=0, left=50, right=50)])
    reference = {"steps": [{"index": 0, "forward_m": 0.0, "heading_change_deg": 0.0, "motion_class": "stop"},
                           {"index": 1, "forward_m": 0.12, "heading_change_deg": 0.4, "motion_class": "forward"}], "note": "placeholder physics"}

    def runner():
        seen["thread"] = threading.current_thread()
        return plan, reference

    screen = make_screen(hub, tmp_path, reference_runner=runner)
    try:
        assert screen.plan_table.rowCount() == 0 and screen.reference_status.text() == "Not computed yet."
        screen.compute_reference()
        assert not screen.reference_button.isEnabled()      # running
        deadline = time.monotonic() + 5
        while screen.plan_table.rowCount() == 0 and time.monotonic() < deadline:
            pump(0.02)
        assert seen["thread"] is not threading.main_thread()
        assert screen.plan_table.rowCount() == 2 and screen.plan_table.item(1, 8).text() == "forward"
        assert screen.reference_status.text().startswith("SIM, not a hardware result") and screen.reference_button.isEnabled()
    finally:
        screen.shutdown()


def test_a_failing_reference_is_reported_not_hidden(qapp, hub, tmp_path):
    screen = make_screen(hub, tmp_path)
    try:
        screen.compute_reference()
        deadline = time.monotonic() + 5
        while not screen.reference_status.text().startswith("Could not") and time.monotonic() < deadline:
            pump(0.02)
        assert "not computed in this test" in screen.reference_status.text() and screen.plan_table.rowCount() == 0
    finally:
        screen.shutdown()


# ================================================================ static guarantees


def _code(path: Path) -> str:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):                           # drop docstrings so prose cannot trip the checks
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef)) and node.body and isinstance(node.body[0], ast.Expr) \
                and isinstance(getattr(node.body[0], "value", None), ast.Constant) and isinstance(node.body[0].value.value, str):
            node.body[0].value.value = ""
    return ast.unparse(tree)


@pytest.mark.parametrize("name", ["backend/control_status.py", "screens/control.py"])
def test_the_new_modules_never_import_serial_or_write_to_the_controller_link(name):
    path = ROOT / "control_center" / name
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert "serial" not in imported and "socket" not in imported and "handoff_bench_client" not in imported
    code = _code(path)
    for forbidden in (".write(", "sendall", "sendto", "send_command", "format_command", "open_port", "UdpPort", "SerialLineSource",
                      "connect_controller", "disconnect_controller", "os.system", "shell=True"):
        assert forbidden not in code, forbidden
    if name == "screens/control.py":
        assert code.count("subprocess.Popen(") == 1


def test_the_only_process_this_screen_can_start_is_motion_check():
    code = (ROOT / "control_center" / "backend" / "control_status.py").read_text(encoding="utf-8")
    assert code.count('"checks"') >= 1 and code.count('"run"') == 1 and '"--surface", "floor"' in code
    assert "subprocess" not in code                        # only the screen launches, and only via launch()
    screen = (ROOT / "control_center" / "screens" / "control.py").read_text(encoding="utf-8")
    assert screen.count("self.launch(") == 2 and screen.count('lambda: self.launch("') == 2
    assert "def showEvent" in screen and "launch" not in screen.split("def showEvent")[1].split("def compute_reference")[0]


def test_the_link_module_is_still_listen_only():
    link = (ROOT / "control_center" / "backend" / "controller_link.py").read_text(encoding="utf-8")
    code = "\n".join(line.split("#")[0] for line in link.splitlines())
    assert ".write(" not in code and "sendall" not in code and "send_command" not in code
