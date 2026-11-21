"""CONNECTIONS screen, authority strip and the Wi-Fi readiness rules (control_center/screens/connections.py).

Offscreen Qt, fake transports, fake clocks and scripted snapshots only. No socket, serial port or camera is opened (a fixture
makes any attempt fail the test), nothing is sent to a robot, nothing under config/ is written (a fixture proves it) and
none of this is hardware verification: every state below is SIMULATED, or a hand-built snapshot.
"""

from __future__ import annotations

import hashlib
import os
import sys
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("QT_QPA_FONTDIR", r"C:\Windows\Fonts")

import pytest
from PySide6.QtWidgets import QApplication

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tests"))

import test_acceptance as ta  # noqa: E402
import test_link_service as tls  # noqa: E402  (the fake-transport rig)
import test_wifi_link as tw  # noqa: E402

from control_center.backend import wifi_link as wl  # noqa: E402
from control_center.backend.control_modes import Authority, AuthorityState, Mode  # noqa: E402
from control_center.backend.link_service import LinkService  # noqa: E402
from control_center.backend.readiness import BLOCKED, FAILED, LEVELS, NOT_CONFIGURED, READY, WARNING  # noqa: E402
from control_center.backend.wifi_link import LinkState  # noqa: E402
from control_center.screens import connections as cn  # noqa: E402
from control_center.screens.connections import ACCEPTANCE_PHRASE, BLOCKED_HEADLINE, ConnectionsScreen, wifi_readiness  # noqa: E402
from control_center.widgets.authority_strip import DISARM_NOTE, SOFTWARE_STOP_NOTE, AuthorityStrip  # noqa: E402

HUB = SimpleNamespace(options=SimpleNamespace(mock=False))


def tree_digest() -> str:
    files = sorted(p for p in (ROOT / "config").rglob("*") if p.is_file())
    return hashlib.sha256(b"".join(p.name.encode() + p.read_bytes() for p in files)).hexdigest()


@pytest.fixture(autouse=True)
def safe_environment(monkeypatch):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("a real transport was requested in a test")
    monkeypatch.setattr(wl.client, "UdpPort", forbidden)
    monkeypatch.setattr(wl, "default_transport_factory", forbidden)
    before = tree_digest()
    yield
    assert tree_digest() == before


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def pump(seconds: float = 0.05) -> None:
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        QApplication.processEvents()
        time.sleep(0.005)


def wait_for(predicate, timeout: float = 4.0) -> bool:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        QApplication.processEvents()
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


@pytest.fixture
def rig(tmp_path, qapp):
    made = tls.Rig(tmp_path)
    yield made
    made.service.shutdown()


@pytest.fixture
def real_rig(tmp_path, qapp):
    """A fake transport that the service counts as REAL (test only: it lets the readiness rules be reached)."""
    made = tls.Rig(tmp_path)
    made.service.link.real = True
    made.step(0.2, status=False)                    # publish a snapshot that knows the link is counted as real
    yield made
    made.service.shutdown()


def make_screen(service=None) -> ConnectionsScreen:
    return ConnectionsScreen(HUB, service)


def texts(screen: ConnectionsScreen) -> str:
    return "\n".join(
        [screen.status_chip.text(), screen.headline_label.text(), screen.detail_label.text(), screen.next_label.text()])


def healthy(rig, **fields):
    rig.connect(**fields)
    rig.step(0.02)
    snapshot = rig.service.latest()
    assert snapshot.health.state is LinkState.CONNECTED and snapshot.health.heartbeat_age_s < 0.5
    return snapshot


# ---------------------------------------------------------------- readiness: every level is reachable


def test_the_five_levels_are_each_reachable_from_snapshots(real_rig):
    snap_down = real_rig.service.latest()
    assert snap_down.simulated is False
    assert wifi_readiness(snap_down).level == NOT_CONFIGURED                   # never connected

    connected = healthy(real_rig)
    blocked = wifi_readiness(connected)
    assert blocked.level == BLOCKED and blocked.headline == BLOCKED_HEADLINE
    assert blocked.headline == "Wi-Fi connected, but motor command transmission has not been physically verified"
    assert "Missing" in blocked.detail and "ACCEPTANCE WORKFLOW" in blocked.fix

    failed_health = replace(connected.health, state=LinkState.FAILED, error=wl.FAILURES["NO_REPLY"][0], error_kind="NO_REPLY",
                            hint=wl.FAILURES["NO_REPLY"][1])
    failed = wifi_readiness(replace(connected, health=failed_health))
    assert failed.level == FAILED and failed.headline == wl.FAILURES["NO_REPLY"][0] and failed.fix == wl.FAILURES["NO_REPLY"][1]

    for change in (dict(state=LinkState.DEGRADED), dict(protocol_ok=None, protocol_note="no HELLO line received yet"),
                   dict(protocol_ok=False, protocol_note="firmware id 'x' is not in the known set"),
                   dict(heartbeat_age_s=0.9), dict(heartbeat_age_s=None)):
        warning = wifi_readiness(replace(connected, health=replace(connected.health, **change)))
        assert warning.level == WARNING, change
    connecting = wifi_readiness(replace(connected, health=replace(connected.health, state=LinkState.CONNECTING)))
    assert connecting.level == WARNING

    complete = replace(connected, acceptance=replace(connected.acceptance, complete=True, missing=()),
                       gate=tuple(replace(item, ok=True) for item in connected.gate))
    ready = wifi_readiness(complete)
    assert ready.level == READY and ready.ok
    assert {NOT_CONFIGURED, READY, WARNING, FAILED, BLOCKED} == set(LEVELS)


def test_ready_needs_the_acceptance_record_the_build_lock_a_real_link_and_a_live_service(real_rig):
    connected = healthy(real_rig)
    complete = replace(connected, acceptance=replace(connected.acceptance, complete=True, missing=()),
                       gate=tuple(replace(item, ok=True) for item in connected.gate))
    assert wifi_readiness(complete).level == READY
    build_closed = replace(complete, gate=tuple(replace(item, ok=False) if item.name == "build" else item for item in complete.gate))
    assert wifi_readiness(build_closed).level == BLOCKED                       # the record is complete but the build lock holds
    assert wifi_readiness(replace(complete, simulated=True)).level != READY    # simulated is never Ready
    assert wifi_readiness(replace(complete, running=True), now=complete.t + 10.0).level == WARNING   # stale: not trusted
    assert wifi_readiness(replace(complete, worker_error="boom")).level == WARNING
    assert wifi_readiness(None).level == NOT_CONFIGURED


def test_simulated_is_never_ready_and_is_labelled(tmp_path, qapp):
    clock = tw.FakeClock()
    service = LinkService(mock_session=True, clock=clock, unix_clock=lambda: ta.NOW, hook_app_quit=False)
    screen = make_screen(service)
    try:
        assert screen.readiness().level == NOT_CONFIGURED and not screen.sim_badge.isHidden()
        assert "SIMULATED" in screen.headline_label.text()
        service.connect()
        for _ in range(8):
            service.pump(clock.advance(0.1))
        screen.show_snapshot(service.latest())
        assert service.latest().health.state is LinkState.CONNECTED
        assert screen.readiness().level == WARNING and screen.status_chip.text() == "WARNING"
        assert "SIMULATED" in texts(screen) and not screen.sim_badge.isHidden()
        scripted = replace(service.latest(), acceptance=replace(service.latest().acceptance, complete=True, missing=()),
                           gate=tuple(replace(item, ok=True) for item in service.latest().gate))
        screen.show_snapshot(scripted)
        assert screen.readiness().level != READY and screen.status_chip.text() != "READY"
        assert any("SIMULATED" in row[3] for row in screen.gate_table.cell_texts())          # live items say so
    finally:
        screen.shutdown()
        service.shutdown()


def test_readiness_is_never_ready_in_the_repositorys_current_state(tmp_path, qapp):
    assert make_screen(None).readiness().level == NOT_CONFIGURED
    port = tw.FakePort()
    service = LinkService(transport_factory=lambda endpoint: port, clock=tw.FakeClock(), unix_clock=lambda: ta.NOW,
                          hook_app_quit=False)                                 # default acceptance file: read only here
    screen = make_screen(service)
    try:
        assert screen.readiness().level != READY
        service.link.real = True                                              # test only: even a link counted as real...
        service.connect()
        port.feed(tw.HELLO, tw.st(t=5000))
        service.pump(service._clock.advance(0.05))
        assert service.latest().health.state is LinkState.CONNECTED and not service.latest().acceptance.complete
        screen.show_snapshot(service.latest())
        assert screen.readiness().level == BLOCKED and screen.status_chip.text() == "BLOCKED"
        assert screen.headline_label.text() == BLOCKED_HEADLINE
    finally:
        screen.shutdown()
        service.shutdown()


# ---------------------------------------------------------------- the screen with scripted snapshots


def test_the_chip_headline_and_next_action_follow_the_snapshot(real_rig, qapp):
    screen = make_screen(real_rig.service)
    try:
        assert screen.status_chip.text() == "NOT CONFIGURED" and "Join the robot's own network" in screen.next_label.text()
        connected = healthy(real_rig)
        screen.show_snapshot(connected)
        assert screen.status_chip.text() == "BLOCKED" and screen.headline_label.text() == BLOCKED_HEADLINE
        assert "Next:" in screen.next_label.text()
        failed_health = replace(connected.health, state=LinkState.FAILED, error="The robot did not answer.", error_kind="NO_REPLY",
                                hint="Check in this order: (1) the laptop is joined to the robot's Wi-Fi")
        screen.show_snapshot(replace(connected, health=failed_health))
        assert screen.status_chip.text() == "FAILED" and screen.headline_label.text() == "The robot did not answer."
        assert "Check in this order" in screen.next_label.text()
        screen.show_snapshot(replace(connected, health=replace(connected.health, state=LinkState.DEGRADED)))
        assert screen.status_chip.text() == "WARNING"
    finally:
        screen.shutdown()


def test_the_health_grid_says_not_measurable_and_never_invents_an_acknowledgement(real_rig):
    screen = make_screen(real_rig.service)
    try:
        snapshot = healthy(real_rig)
        screen.show_snapshot(snapshot)
        values = {key: pair[1].text() for key, pair in screen.health_labels.items()}
        assert "not supported by this firmware" in values["ack"]
        assert values["firmware"] == wl.WIFI_FIRMWARE_ID and values["state"] == "CONNECTED"
        assert values["protocol"].startswith("OK") and "ms" in values["heartbeat"]
        assert "ms" in values["latency"] and "median of 1" in values["latency"]
        unmeasured = replace(snapshot, health=replace(snapshot.health, latency_ms=None, latency_samples=0,
                                                      unavailable={"latency_ms": "no status request has been answered yet"}))
        screen.show_snapshot(unmeasured)
        assert screen.health_labels["latency"][1].text().startswith("not measurable: no status request")
        assert "not measurable" in values["loss"]                                  # the fake robot sent no NET line
        assert values["last_command"] == "none sent this session"
        assert "armed 1" in values["controller"] and "MODE switch asks PC" in values["controller"]
        assert values["rearm"].startswith("REQUIRED")
        assert "not measurable" in values["local_ip"]
        assert not any("acknowledged: yes" in v.lower() for v in values.values())
    finally:
        screen.shutdown()


def test_the_gate_checklist_shows_every_item_with_its_reason_and_evidence_kind(rig):
    screen = make_screen(rig.service)
    try:
        healthy(rig)
        rig.service.request_mode(Mode.AI_SHADOW)
        rig.step()
        snapshot = rig.service.latest()
        rows = screen.gate_table.cell_texts()
        assert len(rows) == len(snapshot.gate) >= 10
        by_name = {row[0]: row for row in rows}
        assert by_name["acceptance"][1] == "FAIL" and by_name["acceptance"][3] == "HARDWARE record" and by_name["acceptance"][2]
        assert by_name["real_session"][1] == "FAIL" and "mock" in by_name["real_session"][2]
        assert by_name["build"][3] == "SOFTWARE" and "AUTONOMOUS_ENABLEABLE_IN_THIS_BUILD" in by_name["build"][2]
        assert by_name["link"][3].startswith("SIMULATED")                         # a fake link's live items are SIMULATED
        assert "GATE CLOSED" in screen.gate_summary.text()
    finally:
        screen.shutdown()


def test_the_acceptance_table_shows_seven_steps_with_4_and_6_blocked_for_a_stated_reason(rig):
    screen = make_screen(rig.service)
    try:
        healthy(rig)
        screen.show_snapshot(rig.service.latest())
        rows = screen.acceptance_table.cell_texts()
        steps = [row for row in rows if row[0][:2] in {f"{n}." for n in range(1, 8)}]
        assert [row[0][0] for row in steps] == list("1234567")
        by_number = {row[0][0]: row for row in steps}
        assert by_number["4"][1] == "BLOCKED" and "motors-disabled" in by_number["4"][4]
        assert by_number["6"][1] == "BLOCKED" and "steps 1 to 5" in by_number["6"][4]
        assert by_number["7"][1] == "BLOCKED" and by_number["1"][1] == "NOT TESTED"
        assert len(rows) == 7 + 5                                                # the five transition sub-checks too
        assert all(row[1] != "PASS" for row in rows)
    finally:
        screen.shutdown()


# ---------------------------------------------------------------- the controls


def enabled(screen) -> dict[str, bool]:
    return {name: getattr(screen, f"{name}_button").isEnabled()
            for name in ("apply", "connect", "disconnect", "reconnect", "discover", "manual", "wifi_record", "stability", "report")}


def test_without_a_service_every_control_is_disabled_and_it_says_so(qapp):
    screen = make_screen(None)
    try:
        assert not any(enabled(screen).values())
        assert screen.status_chip.text() == "NOT CONFIGURED" and "link service" in screen.headline_label.text().lower()
        assert screen.strip.authority_label.text() == "Authority: unknown"
        assert "link service" in screen.connect_button.toolTip()
        screen.press_connect()                                                   # nothing to send to: no crash
        screen.press_return_to_manual()
        text = screen.copy_diagnostics()
        assert "No link service" in text
        screen.shutdown()
        screen.shutdown()
    finally:
        screen.shutdown()


def test_buttons_follow_the_link_state(real_rig):
    screen = make_screen(real_rig.service)
    try:
        down = real_rig.service.latest()
        screen.show_snapshot(down)
        flags = enabled(screen)
        assert flags["apply"] and flags["connect"] and flags["discover"] and flags["manual"]
        assert not flags["disconnect"] and not flags["reconnect"] and not flags["wifi_record"] and not flags["stability"]
        assert screen.host_edit.isEnabled() and screen.port_edit.isEnabled()
        assert "Disabled" in screen.disconnect_button.toolTip()

        connected = healthy(real_rig)
        screen.show_snapshot(connected)
        flags = enabled(screen)
        assert flags["disconnect"] and flags["reconnect"] and flags["manual"]
        assert not flags["connect"] and not flags["apply"] and not flags["discover"]
        assert not screen.host_edit.isEnabled()
        assert not flags["wifi_record"] and not flags["stability"] and not flags["report"]   # until the sentence is typed
        screen.confirm_edit.setText("i am at the real robot")
        assert not enabled(screen)["stability"]                                   # exact, case-sensitive
        screen.confirm_edit.setText(ACCEPTANCE_PHRASE)
        flags = enabled(screen)
        assert flags["wifi_record"] and flags["stability"] and flags["report"]

        screen.show_snapshot(replace(connected, health=replace(connected.health, state=LinkState.FAILED, error="x", hint="y")))
        flags = enabled(screen)
        assert flags["connect"] and flags["reconnect"] and not flags["disconnect"] and not flags["wifi_record"]
        screen.show_snapshot(replace(connected, health=replace(connected.health, state=LinkState.CONNECTING)))
        flags = enabled(screen)
        assert flags["disconnect"] and not flags["connect"] and not flags["reconnect"]
        screen.show_snapshot(replace(down, discovering=True))
        assert not enabled(screen)["discover"]
    finally:
        screen.shutdown()


def test_endpoint_validation_messages_and_buttons(real_rig):
    screen = make_screen(real_rig.service)
    try:
        assert screen.host_edit.text() == "192.168.4.1" and screen.port_edit.text() == "4210"
        assert "usable" in screen.endpoint_message.text() and screen.connect_button.isEnabled()
        for host, port, word in (("192.168.4.255", "4210", "broadcast"), ("0.0.0.0", "4210", "not a robot address"),
                                 ("224.0.0.1", "4210", "multicast"), ("http://robot", "4210", "not a plain address"),
                                 ("192.168.4.1:4210", "4210", "not a plain address"), ("", "4210", "enter the robot's IP"),
                                 ("192.168.4.1", "99999", "65535"), ("192.168.4.1", "abc", "not a port number"),
                                 ("192.168.4.300", "4210", "not a valid IPv4")):
            screen.host_edit.setText(host)
            screen.port_edit.setText(port)
            screen._on_edited()
            assert word in screen.endpoint_message.text(), (host, port, screen.endpoint_message.text())
            assert not screen.connect_button.isEnabled() and not screen.apply_button.isEnabled() and not screen.discover_button.isEnabled()
        screen.host_edit.setText("10.1.2.3")
        screen.port_edit.setText("5000")
        screen._on_edited()
        assert screen.connect_button.isEnabled() and screen.apply_button.isEnabled()
        screen.apply_button.click()
        real_rig.step()
        assert real_rig.service.link.endpoint == wl.Endpoint("10.1.2.3", 5000) and "applied" in screen.endpoint_message.text()
        assert real_rig.endpoints == []                                           # Apply never connects
    finally:
        screen.shutdown()


def test_typing_is_not_overwritten_by_snapshots(real_rig):
    screen = make_screen(real_rig.service)
    try:
        screen.host_edit.setText("10.9.9.9")
        screen._on_edited()
        real_rig.service.configure_endpoint("10.1.1.1", 4210)
        real_rig.step()
        screen.show_snapshot(real_rig.service.latest())
        assert screen.host_edit.text() == "10.9.9.9"                              # still the operator's text
    finally:
        screen.shutdown()


def test_connect_sends_a_request_that_only_asks_for_status(real_rig):
    screen = make_screen(real_rig.service)
    try:
        screen.host_edit.setText("10.4.4.4")
        screen.port_edit.setText("4300")
        screen._on_edited()
        screen.connect_button.click()
        real_rig.port.feed(tw.HELLO, tw.st(t=5000))
        real_rig.service.pump(real_rig.clock.advance(0.05))
        assert real_rig.endpoints == [wl.Endpoint("10.4.4.4", 4300)]
        assert real_rig.port.drives() == [] and real_rig.port.probes()
        assert screen.status_chip.text() == "BLOCKED"
        screen.disconnect_button.click()
        real_rig.step()
        assert screen.status_chip.text() == "NOT CONFIGURED" and "disconnected" in screen.action_label.text()
        screen.connect_button.click()
        real_rig.port.feed(tw.HELLO, tw.st(t=5000))
        real_rig.service.pump(real_rig.clock.advance(0.05))
        screen.reconnect_button.click()
        real_rig.step()
        assert real_rig.service.latest().health.reconnects >= 1 and real_rig.service.latest().mode.mode is Mode.MANUAL
        assert real_rig.service.link.needs_rearm and real_rig.port.drives() == []
    finally:
        screen.shutdown()


def test_discover_shows_the_probe_result_and_a_plain_error(real_rig):
    screen = make_screen(real_rig.service)
    results = []
    real_rig.service.discovery_finished.connect(results.append)
    try:
        real_rig.port.feed(tw.HELLO, tw.st(t=5000))
        screen.discover_button.click()
        real_rig.step(status=False)
        assert wait_for(lambda: bool(results))
        real_rig.step(status=False)
        assert "answered" in screen.discover_label.text() and wl.WIFI_FIRMWARE_ID in screen.discover_label.text()
        results.clear()
        screen.discover_button.click()
        real_rig.step(status=False)
        assert wait_for(lambda: bool(results))
        real_rig.step(status=False)
        assert "no usable answer" in screen.discover_label.text() and "What to try" in screen.discover_label.text()
        assert set(real_rig.port.written) == {b"?\n"}
    finally:
        screen.shutdown()


# ---------------------------------------------------------------- the acceptance panel


def test_the_backends_refusal_text_is_shown_for_operator_reported_and_simulated_records(rig):
    screen = make_screen(rig.service)
    try:
        healthy(rig)
        screen.confirm_edit.setText(ACCEPTANCE_PHRASE)
        assert screen.report_button.isEnabled()
        screen.report_step.setCurrentIndex(0)
        screen.report_status.setCurrentText("PASS")
        for edit, value in zip(screen.reading_edits.values(), ("5.4", "5.0", "4.9", "6")):
            edit.setText(value)
        screen.report_button.click()
        rig.step()
        shown = screen.acceptance_message.text()
        assert "REFUSED" in shown and "operator-reported only" in shown and "not physical verification" in shown
        steps = {row[0][0]: row for row in screen.acceptance_table.cell_texts() if row[0][:2] == "1."}
        assert steps["1"][1] == "NOT TESTED" and steps["1"][2] == "OPERATOR-REPORTED"

        screen.report_step.setCurrentIndex(1)                                      # step 5, takeover
        screen.report_button.click()
        rig.step()
        assert "operator-reported only" in screen.acceptance_message.text()
        assert all(row[1] != "PASS" for row in screen.acceptance_table.cell_texts())

        screen.wifi_record_button.click()
        rig.step()
        assert "simulated" in screen.acceptance_message.text().lower() and "REFUSED" in screen.acceptance_message.text()

        screen.reading_edits["pack_rest_v"].setText("lots")
        screen.report_step.setCurrentIndex(0)
        screen.report_button.click()
        assert "not a number" in screen.acceptance_message.text()
    finally:
        screen.shutdown()


def test_the_stability_window_runs_labelled_simulated_and_records_nothing_that_counts(rig):
    screen = make_screen(rig.service)
    try:
        healthy(rig)
        screen.confirm_edit.setText(ACCEPTANCE_PHRASE)
        assert screen.stability_button.text() == "Start stability window (60 s)"
        screen.stability_button.click()
        rig.step(0.1)
        text = screen.stability_label.text()
        assert "RUNNING" in text and "SIMULATED" in text and not screen.stability_button.isEnabled()
        for _ in range(620):
            rig.step(0.1)
        text = screen.stability_label.text()
        assert "PASSED" in text and "SIMULATED (mock)" in text and "NOT TESTED" in text
        assert screen.stability_bar.value() == 60
        stability_rows = [row for row in screen.acceptance_table.cell_texts() if row[0].startswith("3.")]
        assert stability_rows[0][1] == "NOT TESTED" and stability_rows[0][2] == "SIMULATED (mock)"
        assert screen.readiness().level != READY
    finally:
        screen.shutdown()


# ---------------------------------------------------------------- Return to MANUAL and the authority strip


def test_return_to_manual_always_works(rig):
    screen = make_screen(rig.service)
    try:
        screen.manual_button.click()                                              # no link at all
        rig.step()
        assert rig.service.latest().mode.mode is Mode.MANUAL and rig.port.written == []
        assert "Return to MANUAL" in screen.action_label.text() and "no open link" in screen.action_label.text()
        healthy(rig)
        rig.service.request_mode(Mode.AI_SHADOW)
        rig.step()
        assert "AI SHADOW  (selected)" in screen.strip.mode_chips["AI_SHADOW"].text()
        screen.manual_button.click()
        rig.step()
        assert "MANUAL  (selected)" in screen.strip.mode_chips["MANUAL"].text()
        assert rig.service.latest().mode.mode is Mode.MANUAL
        assert rig.port.drives() and tls.drive_values(rig.port.drives()[-1]) == (0, 0)
        assert screen.manual_button.isEnabled()
    finally:
        screen.shutdown()


def test_the_authority_strip_wording_and_states(real_rig, qapp):
    strip = AuthorityStrip()
    strip.update_from(None)
    assert "No mode information" in strip.mode_label.text()
    connected = healthy(real_rig)
    strip.update_from(connected)
    assert strip.mode_label.text().startswith("MANUAL")
    assert "(selected)" in strip.mode_chips["MANUAL"].text() and "blocked" in strip.mode_chips["AUTONOMOUS"].text()
    assert strip.authority_label.text().startswith("Authority: NONE") or strip.authority_label.text().startswith("Authority: RADIO")
    assert strip.neutral_label.text().startswith("Neutral: nothing non-zero is being sent")
    assert strip.disarm_label.text() == DISARM_NOTE and "Physical disarm always wins" in strip.disarm_label.text()
    assert strip.stop_label.text() == SOFTWARE_STOP_NOTE
    assert "cannot work while Wi-Fi is down" in strip.stop_label.text() and "250 ms" in strip.stop_label.text()
    assert "RadioMaster" in strip.stop_label.text()
    assert strip.sim_label.isHidden()                                              # counted as real in this rig
    strip.update_from(replace(connected, simulated=True))
    assert not strip.sim_label.isHidden()

    driving = replace(connected, transmitting=True,
                      mode=replace(connected.mode, mode=Mode.AUTONOMOUS, authority=AuthorityState(Authority.AI, "AUTONOMOUS: open")))
    strip.update_from(driving)
    assert "(selected)" in strip.mode_chips["AUTONOMOUS"].text() and strip.authority_label.text().startswith("Authority: AI")
    assert strip.neutral_label.text().startswith("NOT neutral")
    held = replace(connected, mode=replace(connected.mode, mode=Mode.AI_SHADOW, neutral_hold=True))
    strip.update_from(held)
    assert "Neutral hold is ON" in strip.neutral_label.text()


# ---------------------------------------------------------------- diagnostics


def test_the_log_is_capped_and_copy_diagnostics_has_what_support_needs(rig, qapp):
    screen = make_screen(rig.service)
    try:
        healthy(rig)
        for index in range(300):
            rig.service._log(f"event {index}")
        pump(0.05)
        assert screen.log_view.document().blockCount() <= 200
        assert "event 299" in screen.log_view.toPlainText() and "event 0\n" not in screen.log_view.toPlainText()
        screen.show_snapshot(rig.service.latest())
        text = screen.copy_diagnostics()
        assert QApplication.clipboard().text() == text
        for word in ("CONNECTIONS diagnostics", "SIMULATED SESSION", "Readiness:", "LINK", "AUTONOMOUS GATE (CLOSED)", "ACCEPTANCE",
                     "EVENTS", "event 299", "not supported by this firmware", "[FAIL] acceptance"):
            assert word in text, word
        assert "Copied" in screen.copy_label.text()
    finally:
        screen.shutdown()


# ---------------------------------------------------------------- lifecycle


def test_shutdown_is_clean_and_stops_listening(rig):
    screen = make_screen(rig.service)
    screen.shutdown()
    screen.shutdown()
    assert not screen._timer.isActive()
    chip = screen.status_chip.text()
    healthy(rig)                                                                   # the service keeps publishing...
    assert screen.status_chip.text() == chip                                       # ...and the closed screen no longer reacts
    screen.close()                                                                 # closeEvent after shutdown: fine
    rig.service.shutdown()


def test_the_screen_never_creates_a_transport_by_itself(tmp_path, qapp):
    calls = []
    service = LinkService(transport_factory=lambda endpoint: calls.append(endpoint), clock=tw.FakeClock(), unix_clock=lambda: ta.NOW,
                          acceptance_path=tmp_path / "a.json", motion_check_path=tmp_path / "n.json", hook_app_quit=False)
    screen = make_screen(service)
    try:
        service.start()
        pump(0.3)
        screen.readiness()
        assert calls == [] and screen.status_chip.text() == "NOT CONFIGURED"
        assert service.latest().mode.mode is Mode.MANUAL and service.latest().health.state is LinkState.DISCONNECTED
    finally:
        screen.shutdown()
        service.shutdown()


def test_with_the_real_worker_thread_the_screen_updates_by_signal(tmp_path, qapp):
    service = LinkService(mock_session=True, hook_app_quit=False)
    screen = make_screen(service)
    try:
        service.start()
        assert wait_for(lambda: service.latest().running)
        screen.connect_button.click()
        assert wait_for(lambda: screen.status_chip.text() == "WARNING")             # a SIMULATED link, connected, never READY
        assert "SIMULATED" in screen.headline_label.text() and not screen.sim_badge.isHidden()
        assert wait_for(lambda: "CONNECTED" in screen.log_view.toPlainText())
        screen.manual_button.click()
        screen.disconnect_button.click()
        assert wait_for(lambda: screen.status_chip.text() == "NOT CONFIGURED")
        assert service.latest().mode.mode is Mode.MANUAL
    finally:
        screen.shutdown()
        service.shutdown()
    assert not service.is_running()
