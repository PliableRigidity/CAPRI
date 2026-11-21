"""The link service (control_center/backend/link_service.py): one link, one mode manager, one acceptance record, one worker.

SOFTWARE TESTS ONLY. Fake transports, fake clocks, an in-process MockTransport and temporary files. No socket, serial port or
camera is opened (a fixture makes any attempt fail the test), nothing is sent to a robot, nothing under config/ is written
(a fixture proves it) and none of this is physical verification. Where a test sets ``link.real = True`` it is exercising the
gates with a fake transport, exactly as tests/test_control_modes.py does; it says nothing about hardware.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtCore import QObject
from PySide6.QtWidgets import QApplication

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tests"))

import test_acceptance as ta  # noqa: E402  (synthetic records)
import test_control_modes as tcm  # noqa: E402  (decisions)
import test_wifi_link as tw  # noqa: E402  (fake port, clock, ST / HELLO lines)

from control_center.backend import link_service as ls  # noqa: E402
from control_center.backend import wifi_link as wl  # noqa: E402
from control_center.backend.acceptance import Evidence  # noqa: E402
from control_center.backend.control_modes import CONFIRM_PHRASE, Confirmation, Mode  # noqa: E402
from control_center.backend.link_service import LinkService, ServiceSnapshot  # noqa: E402
from control_center.backend.profile_settings import ProfileSettingsStore, find_forbidden_keys  # noqa: E402
from control_center.backend.wifi_link import Endpoint, LinkState  # noqa: E402


def tree_digest() -> str:
    files = sorted(p for p in (ROOT / "config").rglob("*") if p.is_file())
    return hashlib.sha256(b"".join(p.name.encode() + p.read_bytes() for p in files)).hexdigest()


@pytest.fixture(autouse=True)
def safe_environment(monkeypatch):
    """Any real socket request fails the test; nothing under config/ may change."""
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


def wait_for(predicate, timeout: float = 3.0) -> bool:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        QApplication.processEvents()
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


class Rig:
    """A service on a fake transport and a fake clock, stepped by hand with ``pump`` (no worker thread)."""

    def __init__(self, tmp_path, store=None, **kwargs) -> None:
        self.clock = tw.FakeClock()
        self.port = tw.FakePort()
        self.endpoints: list[Endpoint] = []
        self.uptime = 5000
        self.decision = tcm.make_decision()

        def factory(endpoint: Endpoint):
            self.endpoints.append(endpoint)
            return self.port

        kwargs.setdefault("sketch_hash", ta.SKETCH)
        self.service = LinkService(store=store, profile="live", transport_factory=factory, clock=self.clock,
                                   acceptance_path=tmp_path / "acceptance.json", unix_clock=lambda: ta.NOW,
                                   motion_check_path=tmp_path / "none.json", hook_app_quit=False,
                                   sleep=lambda seconds: self.clock.advance(seconds), **kwargs)
        self.service.setObjectName("service-under-test")

    def step(self, seconds: float = 0.02, *, status: bool = True, ai: bool = False, **fields) -> None:
        self.uptime += int(seconds * 1000)
        now = self.clock.advance(seconds)
        if status:
            values = dict(t=self.uptime)
            values.update(fields)
            self.port.feed(tw.st(**values))
        if ai:
            self.service.submit_decision(self.decision, policy_loaded=True)
        self.service.pump(now)

    def run(self, seconds: float, **kwargs) -> None:
        for _ in range(int(round(seconds / 0.02))):
            self.step(0.02, **kwargs)

    def connect(self, **fields) -> "Rig":
        self.service.connect()
        values = dict(armed=1, req="PC", act="NONE", pc_ok=0, block="WAIT_PC_NEUTRAL")
        values.update(fields)
        self.port.feed(tw.HELLO, tw.st(t=self.uptime, **values))
        self.service.pump(self.clock.advance(0.05))
        assert self.service.latest().health.state is LinkState.CONNECTED
        return self

    def last_request(self):
        return self.service.latest().last_request


@pytest.fixture
def rig(tmp_path, qapp):
    made = Rig(tmp_path)
    yield made
    made.service.shutdown()


def drive_values(datagram: bytes) -> tuple[int, int]:
    parts = datagram.decode().strip().split(",")
    return int(parts[2]), int(parts[3])


# ---------------------------------------------------------------- the safe starting state


def test_it_starts_safe_and_opens_nothing(rig):
    snapshot = rig.service.latest()
    assert isinstance(snapshot, ServiceSnapshot)
    assert snapshot.mode.mode is Mode.MANUAL and snapshot.health.state is LinkState.DISCONNECTED
    assert snapshot.authority.authority.value == "NONE" and snapshot.simulated and not snapshot.running
    assert not snapshot.ever_sent and not snapshot.transmitting and snapshot.last_command_age_s is None
    assert not rig.service.is_running() and rig.endpoints == [] and rig.port.written == []
    assert snapshot.acceptance.complete is False and snapshot.mode.gate_open is False


def test_the_real_transport_is_created_only_when_the_operator_connects(tmp_path, qapp, monkeypatch):
    calls = []
    port = tw.FakePort()

    def counting(endpoint):
        calls.append(endpoint)
        return port

    monkeypatch.setattr(wl, "default_transport_factory", counting)        # the built-in factory, replaced by a fake
    service = LinkService(clock=tw.FakeClock(), acceptance_path=tmp_path / "a.json", unix_clock=lambda: ta.NOW,
                          motion_check_path=tmp_path / "n.json", hook_app_quit=False)
    try:
        assert service.simulated is False and calls == []                   # constructed: nothing opened
        service.restore_preferences()
        service.pump()
        assert calls == []                                                  # a plain tick, still nothing
        service.connect()
        service.pump()
        assert len(calls) == 1 and port.written == [b"?\n"]                 # only the status request
    finally:
        service.shutdown()


def test_a_mock_session_is_simulated_and_never_touches_the_network(tmp_path, qapp):
    clock = tw.FakeClock()
    service = LinkService(mock_session=True, clock=clock, unix_clock=lambda: ta.NOW, hook_app_quit=False)
    try:
        assert service.simulated and service.latest().mock_session
        service.connect()
        for _ in range(6):
            service.pump(clock.advance(0.1))
        health = service.latest().health
        assert health.state is LinkState.CONNECTED and health.real is False and health.firmware_id == wl.WIFI_FIRMWARE_ID
        assert service.latest().simulated and service.latest().mode.gate_open is False
        temp = service._temp_dir
        assert temp is not None and temp.exists()
    finally:
        service.shutdown()
    assert not temp.exists()                                                # the private record is gone


# ---------------------------------------------------------------- the worker thread


def test_one_worker_no_duplicates_and_shutdown_is_idempotent(tmp_path, qapp):
    service = LinkService(mock_session=True, hook_app_quit=False)
    try:
        assert service.start() is True
        assert service.start() is False and service.start() is False       # no duplicate worker
        first = service._thread
        assert wait_for(lambda: service.iterations >= 5)
        assert service.is_running() and service._thread is first and service._workers_peak == 1
    finally:
        service.shutdown()
    assert not service.is_running() and service._workers_active == 0 and service.closed
    service.shutdown()                                                      # twice is fine
    assert service.start() is False                                         # and it cannot be restarted
    service.connect()                                                       # requests after shutdown are ignored
    assert service._requests.empty()


def test_snapshots_arrive_by_signal_at_about_10_hz(qapp):
    service = LinkService(mock_session=True, hook_app_quit=False)

    class Receiver(QObject):
        def __init__(self) -> None:
            super().__init__()
            self.snapshots: list = []
            self.events: list[str] = []

        def on_snapshot(self, snapshot) -> None:
            self.snapshots.append(snapshot)

        def on_event(self, line: str) -> None:
            self.events.append(line)

    receiver = Receiver()
    service.snapshot_changed.connect(receiver.on_snapshot)
    service.event_added.connect(receiver.on_event)
    try:
        service.start()
        assert wait_for(lambda: len(receiver.snapshots) >= 4, 4.0)
        service.connect()
        assert wait_for(lambda: any("CONNECTED" in line for line in receiver.events))
        assert all(isinstance(item, ServiceSnapshot) for item in receiver.snapshots)
        rate = len(receiver.snapshots)
    finally:
        service.shutdown()
    assert rate < 60                                                        # 10 Hz, not 50 Hz
    with pytest.raises(AttributeError):
        receiver.snapshots[0].mode = None                                   # snapshots are frozen


def test_an_exception_in_the_worker_forces_neutral_and_manual_and_the_loop_survives(rig):
    rig.connect()
    rig.service.request_mode(Mode.AI_SHADOW)
    rig.step()
    assert rig.service.latest().mode.mode is Mode.AI_SHADOW
    sent_before = len(rig.port.drives())
    real_tick = rig.service.manager.tick

    def boom(_now):
        raise RuntimeError("boom")

    rig.service.manager.tick = boom
    rig.step()
    snapshot = rig.service.latest()
    assert snapshot.mode.mode is Mode.MANUAL and "boom" in (snapshot.worker_error or "")
    assert any("INTERNAL ERROR" in line and "boom" in line for line in rig.service.event_log())
    assert len(rig.port.drives()) > sent_before and drive_values(rig.port.drives()[-1]) == (0, 0)   # one neutral went out
    rig.service.manager.tick = real_tick
    count = rig.service.iterations
    rig.step()
    assert rig.service.iterations == count + 1 and rig.service.latest().health.state is LinkState.CONNECTED


def test_an_exception_in_a_request_is_reported_not_swallowed(rig):
    def boom(_now):
        raise OSError("socket exploded")

    rig.service.link.connect = boom
    rig.service.connect()
    rig.step()
    result = rig.last_request()
    assert result is not None and not result.ok and "socket exploded" in result.text
    assert rig.service.latest().mode.mode is Mode.MANUAL


def test_the_real_thread_survives_a_persistent_error(qapp):
    service = LinkService(mock_session=True, hook_app_quit=False)

    def boom(_now):
        raise RuntimeError("always failing")

    service.manager.tick = boom
    try:
        service.start()
        assert wait_for(lambda: service.iterations >= 15)                  # still looping
        assert service.is_running() and service.latest().mode.mode is Mode.MANUAL
        errors = [line for line in service.event_log() if "INTERNAL ERROR" in line]
        assert 1 <= len(errors) <= 3                                        # reported, not a flood
    finally:
        service.shutdown()


def test_shutdown_sends_neutral_only_if_something_was_sent(tmp_path, qapp):
    quiet = Rig(tmp_path)
    quiet.connect()
    quiet.service.shutdown()
    assert quiet.port.drives() == [] and quiet.port.closed                  # connected, nothing sent: nothing sent now
    quiet.service.shutdown()                                                # idempotent
    assert quiet.port.drives() == []

    sent = Rig(tmp_path / "second")
    sent.connect()
    assert sent.service.link.send_neutral(sent.clock()).sent                # something went out this session
    before = len(sent.port.drives())
    sent.service.shutdown()
    drives = sent.port.drives()
    assert len(drives) > before and all(drive_values(item) == (0, 0) for item in drives) and sent.port.closed
    count = len(drives)
    sent.service.shutdown()
    assert len(sent.port.drives()) == count


# ---------------------------------------------------------------- requests and refusals


def test_connect_sends_status_requests_only_and_leaves_the_link_needing_rearm(rig):
    rig.connect()
    assert rig.port.drives() == [] and rig.port.probes()
    health = rig.service.latest().health
    assert health.needs_rearm and rig.service.latest().mode.mode is Mode.MANUAL
    assert rig.service.latest().authority.authority.value in ("NONE", "RADIO")


def test_a_refusal_comes_back_in_the_backends_words(rig):
    seen = []
    rig.service.request_finished.connect(seen.append)
    rig.connect()
    rig.service.request_mode("autonomous")
    rig.step()
    result = rig.last_request()
    assert result is not None and not result.ok and "cannot be entered directly from MANUAL" in result.text
    assert seen and seen[-1] == result and rig.service.latest().mode.mode is Mode.MANUAL
    rig.service.request_mode(Mode.AI_SHADOW)
    rig.step()
    rig.service.request_mode(Mode.AUTONOMOUS, Confirmation(CONFIRM_PHRASE, True, True, True, rig.service.now()))
    rig.step()
    result = rig.last_request()
    assert not result.ok and "gate closed" in result.text and "acceptance" in result.text
    assert rig.service.latest().mode.mode is Mode.AI_SHADOW and rig.port.drives() == []


def test_return_to_manual_always_works(rig):
    rig.service.return_to_manual()                                           # no link at all
    rig.step()
    result = rig.last_request()
    assert result.ok and "no open link" in result.text and rig.port.written == []
    rig.connect()
    rig.service.request_mode(Mode.AI_SHADOW)
    rig.step()
    rig.service.return_to_manual()
    rig.step()
    assert rig.service.latest().mode.mode is Mode.MANUAL
    assert rig.port.drives() and drive_values(rig.port.drives()[-1]) == (0, 0)
    rig.service.return_to_manual()                                           # also when already MANUAL: a neutral again
    count = len(rig.port.drives())
    rig.step()
    assert len(rig.port.drives()) > count


def test_disconnect_returns_to_manual_and_cancels_a_stability_window(rig):
    rig.connect()
    rig.service.start_stability_window(operator_confirmed=True)
    rig.step()
    assert rig.service.latest().stability.state == "RUNNING"
    rig.service.disconnect()
    rig.step()
    snapshot = rig.service.latest()
    assert snapshot.health.state is LinkState.DISCONNECTED and snapshot.mode.mode is Mode.MANUAL
    assert snapshot.stability.state == "IDLE" and "cancelled" in snapshot.stability.recorded
    summary = {step.key: step for step in snapshot.acceptance.steps}
    assert summary["stability"].status == "NOT TESTED"                       # a deliberate disconnect is not a failure


def test_decisions_are_latest_wins_and_forwarded_with_the_time_of_the_call(rig):
    first, second = tcm.make_decision("advance"), tcm.make_decision("stop")
    rig.clock.advance(1.0)
    rig.service.submit_decision(first)
    rig.clock.advance(0.5)
    rig.service.submit_decision(second, policy_loaded=True)
    rig.service.pump()
    assert rig.service.manager.feed.decision is second and rig.service.manager.feed.policy_loaded
    assert rig.service.manager.feed.decided_at == pytest.approx(rig.clock())


def test_events_are_timestamped_and_capped(rig):
    rig.connect()
    lines = rig.service.event_log()
    assert any("connect: ok: connecting to 192.168.4.1:4210" in line for line in lines) and any("connected to 192.168.4.1:4210" in line for line in lines)
    assert all(line[2] == ":" and line[5] == ":" for line in lines)
    for index in range(300):
        rig.service._log(f"line {index}")
    assert len(rig.service.event_log()) == ls.EVENT_LOG_SIZE == 200
    assert len(rig.service.latest().events) <= ls.SNAPSHOT_EVENTS


# ---------------------------------------------------------------- reconnect never resumes


def test_a_reconnect_never_resumes_a_command(tmp_path, qapp):
    """A fully open gate (SYNTHETIC records, fake link marked real, the build lock lifted) driving a non-zero command; then
    the operator reconnects. Afterwards only neutral commands go out, whatever the AI keeps proposing."""
    rig = Rig(tmp_path, capabilities=ta.CAPS)
    try:
        rig.service.link.real = True                                        # test only: exercise the gates with a fake transport
        rig.service.manager._build_override = True                          # test only: the build lock is tested on its own
        ta.pass_everything(rig.service.acceptance)
        rig.connect()
        rig.service.request_mode(Mode.AI_SHADOW)
        rig.step(ai=True)
        rig.run(1.2, ai=True)
        rig.service.set_neutral_hold(True)
        rig.step(ai=True)
        rig.run(0.4, ai=True, act="PC", pc_ok=1, block="NONE")
        assert rig.service.latest().mode.gate_open, [(i.name, i.reason) for i in rig.service.latest().gate if not i.ok]
        rig.service.request_mode(Mode.AUTONOMOUS, Confirmation(CONFIRM_PHRASE, True, True, True, rig.service.now()))
        rig.step(ai=True, act="PC", pc_ok=1, block="NONE")
        assert rig.last_request().ok, rig.last_request().text
        rig.run(0.4, ai=True, act="PC", pc_ok=1, block="NONE")
        moving = [drive_values(item) for item in rig.port.drives()]
        assert any(values != (0, 0) for values in moving)                   # it really was driving
        assert rig.service.latest().transmitting

        mark = len(rig.port.drives())
        rig.service.reconnect()
        rig.step(ai=True)
        assert rig.service.latest().mode.mode is Mode.MANUAL
        rig.port.feed(tw.HELLO)
        for _ in range(100):                                                # two seconds of an eager AI and a healthy robot
            rig.step(ai=True, act="PC", pc_ok=1, block="NONE")
        after = [drive_values(item) for item in rig.port.drives()[mark:]]
        assert all(values == (0, 0) for values in after), after
        assert rig.service.link.needs_rearm and rig.service.latest().mode.mode is Mode.MANUAL
        assert rig.service.latest().authority.authority.value != "AI" and rig.service.latest().health.reconnects == 1
    finally:
        rig.service.shutdown()


# ---------------------------------------------------------------- the profile store


def make_store(tmp_path) -> ProfileSettingsStore:
    return ProfileSettingsStore(tmp_path / "profile_settings")


def test_the_endpoint_is_remembered_across_a_restart_and_nothing_connects_by_itself(tmp_path, qapp):
    store = make_store(tmp_path)
    first = Rig(tmp_path / "one", store=store)
    ok, why = first.service.configure_endpoint("10.1.2.3", 5000)
    assert (ok, why) == (True, "")
    first.step()
    assert first.service.link.endpoint == Endpoint("10.1.2.3", 5000)
    first.service.shutdown()
    section = store.get_section("live", "wifi")
    assert section["host"] == "10.1.2.3" and section["port"] == 5000 and section["auto_connect"] is False

    second = Rig(tmp_path / "two", store=store)                              # "restart": a new service on the same store
    try:
        prefs = second.service.restore_preferences()
        second.step()
        assert (prefs.host, prefs.port, prefs.auto_connect, prefs.problem) == ("10.1.2.3", 5000, False, "")
        snapshot = second.service.latest()
        assert snapshot.endpoint == Endpoint("10.1.2.3", 5000)
        assert snapshot.health.state is LinkState.DISCONNECTED and snapshot.mode.mode is Mode.MANUAL
        assert second.endpoints == [] and second.port.written == []         # no auto-connect by default
    finally:
        second.service.shutdown()


def test_start_restores_the_endpoint_but_still_does_not_connect(tmp_path, qapp):
    store = make_store(tmp_path)
    store.update_section("live", "wifi", {"host": "10.9.8.7", "port": 4999})
    service = LinkService(store=store, mock_session=False, transport_factory=lambda e: pytest.fail("connected by itself"),
                          acceptance_path=tmp_path / "a.json", motion_check_path=tmp_path / "n.json", hook_app_quit=False)
    try:
        service.start()
        assert wait_for(lambda: service.latest().endpoint == Endpoint("10.9.8.7", 4999))
        assert service.latest().health.state is LinkState.DISCONNECTED
    finally:
        service.shutdown()


def test_bad_endpoints_are_refused_and_not_remembered(tmp_path, qapp):
    store = make_store(tmp_path)
    rig = Rig(tmp_path / "r", store=store)
    try:
        for host, port, word in (("192.168.4.255", 4210, "broadcast"), ("", 4210, "IP address"), ("192.168.4.1", 70000, "65535"),
                                 ("http://robot", 4210, "plain address")):
            ok, why = rig.service.configure_endpoint(host, port)
            assert not ok and word in why
        assert not store.has_section("live", "wifi")
        rig.connect()
        ok, why = rig.service.configure_endpoint("10.1.1.1", 4210)
        assert not ok and "disconnect first" in why
        assert not store.has_section("live", "wifi")
    finally:
        rig.service.shutdown()


def test_connecting_with_an_endpoint_remembers_it_and_a_simulated_link_never_claims_the_robot_answered(tmp_path, qapp):
    store = make_store(tmp_path)
    rig = Rig(tmp_path / "r", store=store)
    try:
        rig.service.connect("10.4.4.4", 4300)
        rig.port.feed(tw.HELLO, tw.st(t=5000))
        rig.service.pump(rig.clock.advance(0.05))
        assert rig.service.latest().health.state is LinkState.CONNECTED
        assert rig.endpoints[-1] == Endpoint("10.4.4.4", 4300)
        section = store.get_section("live", "wifi")
        assert (section["host"], section["port"]) == ("10.4.4.4", 4300)
        assert section["last_endpoint_ok_at"] is None                        # a fake link did not answer for the robot
    finally:
        rig.service.shutdown()


def test_a_real_link_remembers_when_the_endpoint_last_answered(tmp_path, qapp):
    store = make_store(tmp_path)
    rig = Rig(tmp_path / "r", store=store)
    try:
        rig.service.link.real = True                                         # test only: stands in for a real link
        rig.connect()
        assert isinstance(store.get_section("live", "wifi")["last_endpoint_ok_at"], str)
    finally:
        rig.service.shutdown()


def test_a_damaged_or_odd_store_is_tolerated(tmp_path, qapp):
    store = make_store(tmp_path)
    store.directory.mkdir(parents=True)
    store.path_for("live").write_text("{ this is not json", encoding="utf-8")
    rig = Rig(tmp_path / "r", store=store)
    try:
        prefs = rig.service.restore_preferences()                            # never raises
        rig.step()
        assert prefs.problem and rig.service.latest().endpoint == Endpoint()  # defaults
        assert rig.service.latest().store_note and any("settings:" in line for line in rig.service.event_log())
        assert rig.endpoints == []
    finally:
        rig.service.shutdown()

    odd = make_store(tmp_path / "odd")
    odd.set_section("live", "wifi", {"host": "192.168.4.255", "port": 4210})   # stored, but not a legal robot address
    rig = Rig(tmp_path / "r2", store=odd)
    try:
        prefs = rig.service.restore_preferences()
        rig.step()
        assert prefs.host is None and "ignored" in prefs.problem and rig.service.latest().endpoint == Endpoint()
    finally:
        rig.service.shutdown()


def test_auto_connect_is_off_unless_stored_true_and_even_then_it_never_arms(tmp_path, qapp):
    store = make_store(tmp_path)
    store.update_section("live", "wifi", {"host": "192.168.4.1", "port": 4210, "auto_connect": True})
    rig = Rig(tmp_path / "r", store=store)
    try:
        assert rig.service.restore_preferences().auto_connect is True
        rig.step()
        assert rig.endpoints == [Endpoint()]                                 # it connected (status requests)
        assert rig.port.drives() == [] and rig.port.written == [b"?\n"]
        snapshot = rig.service.latest()
        assert snapshot.mode.mode is Mode.MANUAL and snapshot.health.needs_rearm
        assert snapshot.authority.authority.value in ("NONE", "RADIO")
    finally:
        rig.service.shutdown()


def test_no_connection_state_or_authority_is_ever_persisted(tmp_path, qapp):
    store = make_store(tmp_path)
    rig = Rig(tmp_path / "r", store=store)
    try:
        rig.service.link.real = True
        rig.service.configure_endpoint("10.1.2.3", 5000)
        rig.step()
        rig.connect()
        rig.service.request_mode(Mode.AI_SHADOW)
        rig.step()
        rig.service.return_to_manual()
        rig.step()
        text = store.path_for("live").read_text(encoding="utf-8")
        document = json.loads(text)
        assert find_forbidden_keys(document) == [] and set(document["sections"]) == {"wifi"}
        assert set(document["sections"]["wifi"]["data"]) <= {"host", "port", "preferred_ssid", "auto_connect", "last_endpoint_ok_at"}
        for word in ("CONNECTED", "AI_SHADOW", "MANUAL", "authority", "RADIO"):
            assert word not in text
    finally:
        rig.service.shutdown()


# ---------------------------------------------------------------- discovery


def test_discover_probes_only_the_configured_endpoint_from_a_short_lived_thread(rig):
    ok, _ = rig.service.configure_endpoint("10.5.5.5", 4210)
    assert ok
    rig.port.feed(tw.HELLO, tw.st(t=5000))
    results = []
    rig.service.discovery_finished.connect(results.append)
    rig.service.discover()
    rig.step(status=False)
    assert wait_for(lambda: bool(results))
    result = results[0]
    assert result.found and result.firmware_id == wl.WIFI_FIRMWARE_ID and result.endpoint == Endpoint("10.5.5.5", 4210)
    assert rig.endpoints == [Endpoint("10.5.5.5", 4210)]                      # the one endpoint, nothing scanned
    assert set(rig.port.written) == {b"?\n"}                                  # status requests only
    rig.step(status=False)
    assert rig.service.latest().discovery == result and not rig.service.latest().discovering
    assert rig.service._discover_thread is not None
    rig.service._discover_thread.join(2.0)
    assert not rig.service._discover_thread.is_alive()                        # short-lived: it ended


def test_discover_is_refused_while_the_link_is_open_and_reports_silence(tmp_path, qapp):
    rig = Rig(tmp_path)
    try:
        rig.connect()
        rig.service.discover()
        rig.step()
        assert not rig.last_request().ok and "disconnect" in rig.last_request().text
        rig.service.disconnect()
        rig.step()
        rig.port.inbox.clear()
        results = []
        rig.service.discovery_finished.connect(results.append)
        rig.service.discover()
        rig.step(status=False)
        assert wait_for(lambda: bool(results))
        assert not results[0].found and results[0].error and results[0].hint      # plain-language error and hint
    finally:
        rig.service.shutdown()


# ---------------------------------------------------------------- acceptance through the service


READINGS = dict(ta.READINGS)


def test_a_simulated_service_never_records_hardware_evidence(rig):
    rig.service.record_acceptance("baseline", "PASS", Evidence.HARDWARE, data=READINGS, operator_confirmed=True)
    rig.step()
    result = rig.last_request()
    assert not result.ok and "simulated" in result.text and "MOCK" in result.what
    steps = {step.key: step for step in rig.service.latest().acceptance.steps}
    assert steps["baseline"].status == "NOT TESTED" and steps["baseline"].evidence == Evidence.MOCK
    assert not rig.service.latest().acceptance.complete

    rig.service.record_acceptance("transitions", "PASS", Evidence.OPERATOR_REPORTED, check="takeover", operator_confirmed=True)
    rig.step()
    assert not rig.last_request().ok and "operator-reported only" in rig.last_request().text
    child = {c.key: c for c in rig.service.latest().acceptance.steps[4].subchecks}["transitions.takeover"]
    assert child.status in ("NOT TESTED", "BLOCKED")

    rig.service.record_acceptance("autonomous", "PASS", Evidence.HARDWARE)   # derived: not recordable
    rig.step()
    assert not rig.last_request().ok and "not recordable" in rig.last_request().text


def test_step_2_needs_a_connected_identified_link_and_the_operators_confirmation(rig):
    rig.service.record_wifi_established(operator_confirmed=True)
    rig.step()
    assert not rig.last_request().ok and "DISCONNECTED" in rig.last_request().text
    rig.connect()
    rig.service.record_wifi_established(operator_confirmed=True)
    rig.step()
    result = rig.last_request()
    assert not result.ok and "simulated" in result.text                       # fake link: stored as MOCK, NOT TESTED
    steps = {step.key: step for step in rig.service.latest().acceptance.steps}
    assert steps["wifi"].status == "NOT TESTED"


def test_step_2_on_a_link_counted_as_real_needs_the_typed_confirmation(tmp_path, qapp):
    rig = Rig(tmp_path)
    try:
        rig.service.link.real = True                                         # test only
        rig.connect()
        rig.service.record_wifi_established(operator_confirmed=False)
        rig.step()
        assert not rig.last_request().ok and "operator has not confirmed" in rig.last_request().text
        rig.service.record_wifi_established(operator_confirmed=True)
        rig.step()
        assert rig.last_request().ok
        steps = {step.key: step for step in rig.service.latest().acceptance.steps}
        assert steps["wifi"].status == "PASS" and steps["wifi"].evidence == Evidence.HARDWARE
        rig.service.link.send_neutral(rig.clock())                          # now a drive datagram has gone out
        rig.service.record_wifi_established(operator_confirmed=True)
        rig.step()
        assert not rig.last_request().ok and "WITHOUT motor commands" in rig.last_request().text
        assert not rig.service.latest().acceptance.complete and rig.service.latest().mode.gate_open is False
    finally:
        rig.service.shutdown()


def run_window(rig, seconds: float, step: float = 0.1, **fields) -> None:
    for _ in range(int(round(seconds / step))):
        rig.step(step, **fields)


def test_the_stability_window_on_a_simulated_link_is_recorded_as_mock(rig):
    rig.connect()
    rig.service.start_stability_window(operator_confirmed=True)
    rig.step(0.1)
    assert rig.service.latest().stability.state == "RUNNING" and "SIMULATED" in rig.last_request().text
    run_window(rig, 61.0)
    stability = rig.service.latest().stability
    assert stability.state == "PASSED" and stability.evidence == Evidence.MOCK and "NOT TESTED" in stability.recorded
    steps = {step.key: step for step in rig.service.latest().acceptance.steps}
    assert steps["stability"].status == "NOT TESTED" and steps["stability"].evidence == Evidence.MOCK
    assert not rig.service.latest().acceptance.complete


def test_the_stability_window_on_a_link_counted_as_real_records_hardware_and_failure_records_fail(tmp_path, qapp):
    rig = Rig(tmp_path)
    try:
        rig.service.link.real = True                                         # test only
        rig.connect()
        rig.service.start_stability_window(operator_confirmed=True)
        run_window(rig, 61.0)
        stability = rig.service.latest().stability
        assert stability.state == "PASSED" and stability.evidence == Evidence.HARDWARE
        steps = {step.key: step for step in rig.service.latest().acceptance.steps}
        assert steps["stability"].status == "PASS" and steps["stability"].evidence == Evidence.HARDWARE

        rig.service.start_stability_window(operator_confirmed=True)          # a second window, which then loses its heartbeat
        rig.step(0.1)
        for _ in range(8):
            rig.step(0.1, status=False)                                      # no ST for 0.8 s
        assert rig.service.latest().stability.state == "FAILED"
        steps = {step.key: step for step in rig.service.latest().acceptance.steps}
        assert steps["stability"].status == "FAIL"
    finally:
        rig.service.shutdown()


def test_a_window_cannot_start_on_a_link_that_is_not_connected_and_needs_the_confirmation_to_pass(rig):
    rig.service.start_stability_window(operator_confirmed=True)
    rig.step()
    assert not rig.last_request().ok and "connect first" in rig.last_request().text
    assert rig.service.latest().stability.state == "IDLE"
