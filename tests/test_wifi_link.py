"""The Wi-Fi link manager (control_center/backend/wifi_link.py).

SOFTWARE TESTS ONLY: fake transports and fake clocks. No socket, serial port or camera is opened, nothing is sent to a
robot, and none of this is physical verification (evidence kind MOCK/SIM, never HARDWARE).
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import handoff_bench_client as client  # noqa: E402

from control_center.backend import wifi_link as wl  # noqa: E402
from control_center.backend.wifi_link import (Endpoint, LinkState, WifiLink, validate_endpoint)  # noqa: E402

FW = "autobot-handoff-wifi-1"
HELLO = (f"HELLO fw={FW} arm=CH5:HIGH pc=CH6:HIGH pc_limit=128 pc_timeout_ms=250 rc_timeout_ms=250 "
         "wifi_ap=AutoBot-Handoff wifi_ok=1 wifi_channel=6 udp_port=4210")
BACKEND = ROOT / "control_center" / "backend"


def st(**overrides) -> str:
    values = dict(t=5000, tok=7, link=1, armed=1, req="PC", act="PC", pc_ok=1, block="NONE", fault="NONE", stop="NONE",
                  L=0, R=0, pc_age=10)
    values.update(overrides)
    return "ST " + " ".join(f"{key}={values[key]}" for key in client.STATUS_KEYS)


def net(rx=100, bad=0, old=0, other=0, peer="192.168.4.2:50123") -> str:
    return f"NET ap=1 sta=1 peer={peer} pc_via=WIFI rx={rx} bad={bad} old={old} other={other}"


class FakeClock:
    def __init__(self, start: float = 1000.0) -> None:
        self.t = start

    def __call__(self) -> float:
        return self.t

    def advance(self, seconds: float) -> float:
        self.t += seconds
        return self.t


class FakePort:
    """Looks like UdpPort (read / write / close / error / probe) and records everything written."""

    probe = True
    label = "FAKE Wi-Fi 192.168.4.1:4210 (UDP)"

    def __init__(self) -> None:
        self.inbox: list[bytes] = []
        self.written: list[bytes] = []
        self.closed = False
        self.error: str | None = None
        self.fail_writes = False

    def feed(self, *lines: str) -> None:
        self.inbox.extend((line + "\n").encode("ascii") for line in lines)

    def read(self, _size: int = 4096) -> bytes:
        data, self.inbox = b"".join(self.inbox), []
        return data

    def write(self, data: bytes) -> int:
        if self.fail_writes:
            self.error = "[WinError 10051] network unreachable"
            return 0
        self.written.append(data)
        return len(data)

    def close(self) -> None:
        self.closed = True

    def drives(self) -> list[bytes]:
        return [item for item in self.written if item.startswith(b"D,")]

    def probes(self) -> list[bytes]:
        return [item for item in self.written if item == b"?\n"]


class Rig:
    def __init__(self, **link_kwargs) -> None:
        self.clock = FakeClock()
        self.port = FakePort()
        self.endpoints: list[Endpoint] = []

        def factory(endpoint: Endpoint):
            self.endpoints.append(endpoint)
            return self.port

        self.link = WifiLink(transport_factory=factory, clock=self.clock, unix_clock=lambda: 1_700_000_000.0, **link_kwargs)
        self.uptime = 5000

    def tick(self, seconds: float = 0.0, *lines: str) -> None:
        now = self.clock.advance(seconds)
        if lines:
            self.port.feed(*lines)
        self.link.tick(now)

    def connected(self, **status) -> "Rig":
        assert self.link.connect(self.clock())
        self.tick(0.05, HELLO, st(**status))
        assert self.link.state is LinkState.CONNECTED
        return self

    def beat(self, seconds: float = 0.02, **status) -> None:
        """Advance, with a fresh ST line, and tick."""
        self.uptime += int(seconds * 1000)
        values = dict(t=self.uptime)
        values.update(status)
        self.tick(seconds, st(**values))

    def ready(self) -> "Rig":
        """Connected, re-armed on the controller's report, a neutral sent: drive commands can now be considered."""
        self.connected()
        self.link.real = True                         # TEST ONLY: lets the gates be exercised with a fake transport
        self.link.drive_allowed = lambda: True
        ok, why = self.link.rearm(self.clock(), operator_confirmed=True)
        assert ok, why
        assert self.link.send_neutral(self.clock()).sent
        self.beat(0.03)
        return self

    def drive(self, throttle=40, steer=0, age=0.05):
        now = self.clock()
        return self.link.send_drive(throttle, steer, now, now - age)


# ---------------------------------------------------------------- endpoint

@pytest.mark.parametrize("host", ["192.168.4.1", "10.0.0.7", "robot", "robot-1.local", "my.host.example"])
def test_endpoint_accepts_unicast_ipv4_and_names(host):
    endpoint, why = validate_endpoint(host, 4210)
    assert endpoint == Endpoint(host, 4210), why


@pytest.mark.parametrize("host", ["", "   ", None, "garbage!", "192.168.4.256", "192.168.04.1", "1.2.3", "255.255.255.255",
                                  "192.168.4.255", "224.0.0.1", "0.0.0.0", "192.168.4.0", "192.168.4.1:4210", "http://robot",
                                  "a b", "240.0.0.1", "::1", "broadcast"])
def test_endpoint_rejects_garbage_and_broadcast(host):
    endpoint, why = validate_endpoint(host, 4210)
    assert endpoint is None and why


@pytest.mark.parametrize("port", [0, 65536, -1, "x", "", True, None, 12.5])
def test_endpoint_rejects_bad_ports(port):
    endpoint, why = validate_endpoint("192.168.4.1", port)
    assert endpoint is None and why


def test_default_endpoint_is_the_sketchs_soft_ap():
    assert Endpoint() == Endpoint("192.168.4.1", 4210)
    assert WifiLink().endpoint == Endpoint("192.168.4.1", 4210)


def test_configure_validates_and_refuses_while_open():
    rig = Rig()
    assert rig.link.configure(Endpoint("10.1.2.3", 5000)) == (True, "")
    assert rig.link.endpoint == Endpoint("10.1.2.3", 5000)
    ok, why = rig.link.configure("192.168.4.255")
    assert not ok and "broadcast" in why and rig.link.endpoint == Endpoint("10.1.2.3", 5000)
    with pytest.raises(ValueError):
        WifiLink(Endpoint("255.255.255.255", 4210))
    rig.link.connect(rig.clock())
    ok, why = rig.link.configure("10.9.9.9")
    assert not ok and "disconnect" in why


# ---------------------------------------------------------------- lifecycle

def test_connect_sends_status_requests_only_and_needs_hello_and_status():
    rig = Rig()
    assert rig.link.state is LinkState.DISCONNECTED
    assert rig.link.connect(rig.clock())
    assert rig.link.state is LinkState.CONNECTING
    assert rig.port.written == [b"?\n"]                       # the only thing sent: a status request
    rig.tick(0.05, st())                                      # status but no HELLO yet: not connected
    assert rig.link.state is LinkState.CONNECTING
    rig.tick(0.05, HELLO)
    assert rig.link.state is LinkState.CONNECTED
    assert rig.port.drives() == []
    health = rig.link.health(rig.clock())
    assert health.firmware_id == FW and health.protocol_ok is True and health.real is False
    assert rig.endpoints == [Endpoint()]


def test_hello_must_identify_a_known_firmware():
    rig = Rig()
    rig.link.connect(rig.clock())
    rig.tick(0.05, HELLO.replace(FW, "some-other-firmware"), st())
    assert rig.link.state is LinkState.FAILED
    assert rig.link.error_kind == "UNKNOWN_FIRMWARE"
    assert "some-other-firmware" in rig.link.error and rig.link.hint
    assert rig.port.closed and rig.port.drives() == []


def test_no_reply_times_out_with_a_plain_hint():
    rig = Rig()
    rig.link.connect(rig.clock())
    for _ in range(12):
        rig.tick(0.5)
    assert rig.link.state is LinkState.FAILED and rig.link.error_kind == "NO_REPLY"
    assert "did not answer" in rig.link.error
    for expected in ("joined", "Wi-Fi sketch", "192.168.4.1", "Firewall"):
        assert expected in rig.link.hint
    assert len(rig.port.probes()) >= 4                        # it kept asking, only with '?'
    assert rig.port.drives() == [] and rig.port.closed


def test_failure_kinds_each_have_an_error_and_a_hint():
    for kind, (text, hint) in wl.FAILURES.items():
        assert text and hint, kind
    # not joined: the transport reports a send error and nothing ever comes back
    rig = Rig()
    rig.port.fail_writes = True
    rig.link.connect(rig.clock())
    for _ in range(12):
        rig.tick(0.5)
    assert rig.link.error_kind == "NOT_JOINED" and "network" in rig.link.hint.lower()
    # wrong ip: something answers, but not in the protocol
    rig = Rig()
    rig.link.connect(rig.clock())
    for _ in range(12):
        rig.tick(0.5, "HTTP/1.1 400 not the robot")
    assert rig.link.error_kind == "WRONG_ENDPOINT" and "IP" in rig.link.hint
    # firmware that sends ST but never HELLO
    rig = Rig()
    rig.link.connect(rig.clock())
    for _ in range(12):
        rig.tick(0.5, st())
    assert rig.link.error_kind == "NO_HELLO"


@pytest.mark.parametrize("error, kind", [(OSError(98, "Address already in use"), "IN_USE"),
                                         (PermissionError(13, "forbidden by an access permission"), "BLOCKED"),
                                         (OSError(51, "Network is unreachable"), "NOT_JOINED")])
def test_open_errors_are_classified(error, kind):
    def factory(_endpoint):
        raise error

    link = WifiLink(transport_factory=factory, clock=FakeClock())
    assert link.connect(0.0) is False
    assert link.state is LinkState.FAILED and link.error_kind == kind and link.hint


def test_disconnect_sends_neutral_only_if_a_drive_command_was_sent():
    rig = Rig().connected()
    rig.link.disconnect(rig.clock())
    assert rig.port.drives() == [] and rig.port.closed and rig.link.state is LinkState.DISCONNECTED
    rig = Rig().ready()
    before = len(rig.port.drives())
    assert before >= 1
    rig.link.disconnect(rig.clock())
    sent = rig.port.drives()
    assert len(sent) == before + 1 and sent[-1].startswith(b"D,7,0,0,")
    assert rig.port.closed and rig.link.state is LinkState.DISCONNECTED


def test_shutdown_is_idempotent_and_ends_with_one_neutral():
    rig = Rig().ready()
    rig.link.shutdown(rig.clock())
    count = len(rig.port.written)
    assert rig.port.drives()[-1].startswith(b"D,7,0,0,") and rig.port.closed
    rig.link.shutdown(rig.clock())
    rig.link.shutdown(rig.clock())
    assert len(rig.port.written) == count
    assert rig.link.connect(rig.clock()) is False and rig.link.state is LinkState.DISCONNECTED


def test_discover_probes_only_the_one_endpoint():
    rig = Rig()
    rig.link.configure("192.168.4.1", 4210)
    clock = rig.clock

    def sleep(seconds: float) -> None:
        clock.advance(seconds)
        if len(rig.port.probes()) >= 1 and not rig.port.inbox and not getattr(sleep, "answered", False):
            sleep.answered = True
            rig.port.feed(HELLO, st())

    result = rig.link.discover(timeout=2.0, sleep=sleep)
    assert result.found and result.firmware_id == FW and result.known_firmware and result.protocol_ok
    assert result.hello["pc_limit"] == "128" and result.status_seen
    assert rig.endpoints == [Endpoint("192.168.4.1", 4210)]          # one transport, to the one endpoint
    assert set(rig.port.written) == {b"?\n"}                         # only status requests: no scan, no drive
    assert rig.port.closed and rig.link.state is LinkState.DISCONNECTED


def test_discover_reports_silence_and_never_runs_beside_an_open_link():
    rig = Rig()
    result = rig.link.discover(timeout=1.0, sleep=rig.clock.advance)
    assert not result.found and result.error and result.hint and "joined" in result.hint
    rig = Rig().connected()
    result = rig.link.discover(timeout=1.0, sleep=rig.clock.advance)
    assert not result.found and "already open" in result.error
    assert len(rig.endpoints) == 1                                   # no second transport was created


# ---------------------------------------------------------------- health

def test_latency_is_the_median_of_question_to_hello_and_none_when_unmeasurable():
    rig = Rig()
    assert rig.link.health(rig.clock()).latency_ms is None
    assert "latency_ms" in rig.link.health(rig.clock()).unavailable
    rig.link.connect(rig.clock())                       # '?' sent at t0
    rig.tick(0.040, HELLO, st())                        # answered 40 ms later
    rig.tick(0.100, st())
    health = rig.link.health(rig.clock())
    assert health.latency_ms == pytest.approx(40.0) and health.latency_samples == 1
    for _ in range(5):                                   # next probe is due after 1 s of ticks
        rig.tick(0.2, st())
    probe_at = rig.clock()
    assert len(rig.port.probes()) == 2
    rig.tick(0.100, HELLO)                               # 100 ms
    health = rig.link.health(rig.clock())
    assert health.latency_samples == 2 and health.latency_ms == pytest.approx(70.0)
    assert probe_at <= rig.clock()


def test_heartbeat_and_last_rx_age_come_from_the_clock():
    rig = Rig().connected()
    rig.tick(0.3)
    health = rig.link.health(rig.clock())
    assert health.heartbeat_age_s == pytest.approx(0.3) and health.last_rx_age_s == pytest.approx(0.3)
    assert health.last_success_unix == 1_700_000_000.0
    rig = Rig()
    health = rig.link.health(rig.clock())
    assert health.heartbeat_age_s is None and health.last_rx_age_s is None and health.last_success_unix is None
    assert {"heartbeat_age_s", "last_rx_age_s", "last_success_unix", "loss"} <= set(health.unavailable)
    assert health.loss is None and health.protocol_ok is None


def test_loss_info_is_none_or_measured_never_invented():
    rig = Rig().connected()
    loss = rig.link.health(rig.clock()).loss
    assert loss is not None and loss.firmware_rx is None and loss.firmware_loss_ratio is None
    assert loss.pc_status_gaps == 0 and loss.pc_send_errors == 0
    assert "loss.firmware" in rig.link.health(rig.clock()).unavailable
    rig.beat(0.1, )
    rig.tick(0.1, net(rx=100, bad=0, old=0))
    rig.tick(0.5, st(t=6000), net(rx=150, bad=2, old=8))            # 10 of 50 datagrams bad or old
    health = rig.link.health(rig.clock())
    assert health.loss.firmware_rx == 150 and health.loss.firmware_old == 8
    assert health.loss.firmware_loss_ratio == pytest.approx(10 / 50)
    assert health.loss.high is True and rig.link.state is LinkState.DEGRADED
    assert health.local_ip == "192.168.4.2" and health.peer_ip == "192.168.4.1"
    # too few datagrams: no ratio rather than a noisy one
    rig = Rig().connected()
    rig.tick(0.1, net(rx=100), st(t=5100))
    rig.tick(0.1, net(rx=105, old=1), st(t=5200))
    assert rig.link.health(rig.clock()).loss.firmware_loss_ratio is None


def test_late_status_degrades_then_recovers_but_stays_latched_and_long_silence_fails():
    rig = Rig().ready()
    assert rig.link.needs_rearm is False
    rig.tick(0.7)
    assert rig.link.state is LinkState.DEGRADED
    assert rig.link.needs_rearm and "late" in rig.link.latched_reason
    assert rig.link.health(rig.clock()).loss.pc_status_gaps == 1
    rig.beat(0.05)
    assert rig.link.state is LinkState.CONNECTED and rig.link.needs_rearm          # recovered, still latched
    assert rig.drive().sent is False
    rig.tick(6.0)
    assert rig.link.state is LinkState.FAILED and rig.link.error_kind == "LINK_LOST" and rig.link.hint


# ---------------------------------------------------------------- command gating

def test_every_single_gate_closed_means_nothing_is_sent():
    def drive_allowed_false(rig):
        rig.link.drive_allowed = lambda: False

    def drive_allowed_missing(rig):
        rig.link.drive_allowed = None

    def degraded(rig):
        rig.link.state = LinkState.DEGRADED

    def rearm_needed(rig):
        rig.link._latch("test latch")

    def status_stale(rig):
        rig.clock.advance(0.6)

    def disarmed(rig):
        rig.beat(0.02, armed=0)

    def mode_radio(rig):
        rig.beat(0.02, req="RADIO", act="RADIO", pc_ok=0)

    def not_accepted(rig):
        rig.beat(0.02, pc_ok=0, act="NONE", block="WAIT_PC_NEUTRAL")

    def fault(rig):
        rig.beat(0.02, fault="PC_STALE", pc_ok=0, act="NONE", block="REARM_REQUIRED")

    def radio_down(rig):
        rig.beat(0.02, link=0)

    def token_changed(rig):
        rig.beat(0.02, tok=8)

    def no_limit(rig):
        rig.link._core.pc_limit = None

    def previous_stale(rig):
        rig.clock.advance(0.3)
        rig.port.feed(st(t=rig.uptime + 300))
        rig.link.poll(rig.clock())

    # (how to close one gate, the refusal when the latch also fires, the refusal when ONLY that gate is closed)
    closers = [(drive_allowed_false, "NOT_ALLOWED", "NOT_ALLOWED"), (drive_allowed_missing, "NOT_ALLOWED", "NOT_ALLOWED"),
               (degraded, "NOT_CONNECTED", "NOT_CONNECTED"), (rearm_needed, "NEEDS_REARM", None),
               (status_stale, "STATUS_STALE", "STATUS_STALE"), (disarmed, "NEEDS_REARM", "CONTROLLER"),
               (mode_radio, "NEEDS_REARM", "CONTROLLER"), (not_accepted, "NEEDS_REARM", "CONTROLLER"),
               (fault, "NEEDS_REARM", "CONTROLLER"), (radio_down, "NEEDS_REARM", "CONTROLLER"),
               (token_changed, "NEEDS_REARM", "TOKEN"), (no_limit, "NO_LIMIT", "NO_LIMIT"),
               (previous_stale, "PREVIOUS_STALE", "PREVIOUS_STALE")]
    for close, natural, isolated in closers:
        for only_this_gate in (False, True):
            if only_this_gate and isolated is None:
                continue                                # this gate IS the latch: nothing is left to isolate
            rig = Rig().ready()
            assert rig.drive().sent is True, "the open rig must be able to send"
            rig.clock.advance(0.03)
            before = len(rig.port.written)
            close(rig)
            if only_this_gate:
                rig.link._needs_rearm = False       # take the latch away so the gate itself is what refuses
            rig.clock.advance(0.03)
            result = rig.drive()
            assert result.sent is False and result.reason, close.__name__
            assert result.code == (isolated if only_this_gate else natural), (close.__name__, only_this_gate, result)
            assert [w for w in rig.port.written[before:] if w.startswith(b"D,")] == [], close.__name__


def test_open_gates_send_with_rising_sequence_numbers_and_the_current_token():
    rig = Rig().ready()
    sequences = []
    for _ in range(4):
        rig.beat(0.02)
        result = rig.drive(40, -20)
        assert result.sent and result.code == "SENT"
        sequences.append(result.sequence)
    assert sequences == sorted(set(sequences)) and sequences[0] > 1
    last = rig.port.drives()[-1]
    assert last == f"D,7,40,-20,{sequences[-1]}\n".encode()
    numbers = [int(item.split(b",")[4]) for item in rig.port.drives()]
    assert numbers == sorted(set(numbers))                         # strictly rising, neutrals included
    rig.beat(0.02)
    assert rig.link.send_drive(0, 0, rig.clock(), None).sent         # a zero command needs no observation time


def test_value_caps_follow_max_command_and_the_firmware_limit():
    rig = Rig().ready()
    rig.beat(0.02)
    assert rig.drive(60, 60).sent
    rig.beat(0.02)
    refused = rig.drive(61, 0)
    assert not refused.sent and refused.code == "LIMIT" and "60" in refused.reason
    rig.beat(0.02)
    assert not rig.drive(0, -61).sent
    # a higher max_command is still bounded by the firmware's own pc_limit (128 in HELLO)
    rig = Rig(max_command=255).ready()
    rig.beat(0.02)
    assert rig.drive(128, 0).sent
    rig.beat(0.02)
    assert rig.drive(129, 0).code == "LIMIT"
    rig.beat(0.02)
    for bad in (float("nan"), float("inf"), "40", True, None):
        assert rig.link.send_drive(bad, 0, rig.clock(), rig.clock()).sent is False
    with pytest.raises(ValueError):
        WifiLink(max_command=256)


def test_rate_limit_is_50_hz():
    rig = Rig().ready()
    rig.beat(0.02)
    assert rig.drive().sent
    rig.clock.advance(0.01)
    blocked = rig.drive()
    assert not blocked.sent and blocked.code == "RATE_LIMIT"
    rig.clock.advance(0.011)
    rig.port.feed(st(t=rig.uptime + 100))
    rig.link.poll(rig.clock())
    assert rig.drive().sent
    assert rig.link.send_neutral(rig.clock()).code == "RATE_LIMIT"
    assert rig.link.send_neutral(rig.clock(), force=True).sent          # a final stop is never rate limited


def test_stale_observation_is_never_sent():
    rig = Rig().ready()
    rig.beat(0.02)
    now = rig.clock()
    assert rig.link.send_drive(40, 0, now, now - 0.30).code == "OBSERVATION_STALE"
    rig.beat(0.02)
    now = rig.clock()
    assert rig.link.send_drive(40, 0, now, None).code == "NO_OBSERVATION_TIME"
    rig.beat(0.02)
    now = rig.clock()
    assert rig.link.send_drive(40, 0, now, now - 0.20).sent
    rig.beat(0.02)
    assert rig.link.send_drive(0, 0, rig.clock(), None).sent          # zero never needs an observation


def test_previous_command_stale_latches_and_needs_rearm():
    rig = Rig().ready()
    rig.beat(0.02)
    assert rig.drive().sent
    for _ in range(3):
        rig.beat(0.1)                                                 # status continues; no command for 300 ms
    result = rig.drive()
    assert result.code == "PREVIOUS_STALE" and rig.link.needs_rearm
    assert "stopped" in rig.link.latched_reason


def test_first_command_needs_a_neutral_first():
    rig = Rig().connected()
    rig.link.real = True
    rig.link.drive_allowed = lambda: True
    assert rig.link.rearm(rig.clock(), operator_confirmed=True)[0]
    rig.beat(0.02)
    assert rig.drive().code == "NEUTRAL_FIRST"


# ---------------------------------------------------------------- re-arm and latching

def test_link_starts_needing_a_deliberate_rearm_and_rearm_needs_confirmation_and_a_good_controller():
    rig = Rig().connected()
    assert rig.link.needs_rearm and "confirm" in rig.link.latched_reason
    assert rig.link.rearm(rig.clock())[0] is False                    # no operator confirmation
    rig.beat(0.02, armed=0)
    ok, why = rig.link.rearm(rig.clock(), operator_confirmed=True)
    assert not ok and "DISARMED" in why
    rig.beat(0.02, fault="PC_STALE", pc_ok=0, act="NONE")
    assert not rig.link.rearm(rig.clock(), operator_confirmed=True)[0]
    rig.beat(0.02, pc_ok=0, act="NONE", block="WAIT_PC_NEUTRAL")
    assert not rig.link.rearm(rig.clock(), operator_confirmed=True)[0]
    rig.beat(0.02)
    assert rig.link.rearm(rig.clock(), operator_confirmed=True)[0] and not rig.link.needs_rearm


@pytest.mark.parametrize("change, expected", [
    (dict(tok=9), "token"),
    (dict(armed=0, act="NONE", pc_ok=0, block="DISARMED"), "ARM"),
    (dict(req="RADIO", act="RADIO", pc_ok=0), "MODE"),
    (dict(fault="PC_STALE", act="NONE", pc_ok=0, block="REARM_REQUIRED"), "PC_STALE"),
    (dict(fault="RADIO_LOST", link=0, act="NONE", pc_ok=0), "RADIO_LOST"),
    (dict(link=0), "radio link"),
    (dict(pc_ok=0, act="NONE", block="WAIT_PC_NEUTRAL"), "no longer accepts"),
])
def test_token_arm_mode_and_faults_latch_a_rearm(change, expected):
    rig = Rig().ready()
    assert not rig.link.needs_rearm
    rig.beat(0.02, **change)
    assert rig.link.needs_rearm and expected in rig.link.latched_reason
    assert rig.drive().sent is False
    assert rig.link.send_neutral(rig.clock() + 0.05).sent                   # neutral is still allowed


def test_a_controller_restart_is_seen_as_uptime_going_backwards():
    rig = Rig().ready()
    assert rig.link.health(rig.clock()).restarts == 0
    rig.tick(0.02, st(t=300))                                                # was about 5000 ms
    health = rig.link.health(rig.clock())
    assert health.restarts == 1 and rig.link.needs_rearm and "restarted" in rig.link.latched_reason


def test_disconnect_and_reconnect_never_resume_motion():
    rig = Rig().ready()
    assert rig.drive().sent
    rig.link.disconnect(rig.clock())
    assert rig.link.needs_rearm and rig.link.state is LinkState.DISCONNECTED
    rig.port.closed = False
    assert rig.link.connect(rig.clock())
    assert rig.link.health(rig.clock()).reconnects == 1
    rig.tick(0.05, HELLO, st())                                              # the controller still looks perfect
    assert rig.link.state is LinkState.CONNECTED and rig.link.needs_rearm
    rig.link.drive_allowed = lambda: True
    rig.link.real = True
    sent_before = len(rig.port.drives())
    rig.beat(0.05)
    result = rig.drive()
    assert result.code == "NEEDS_REARM" and len(rig.port.drives()) == sent_before
    assert rig.link.rearm(rig.clock(), operator_confirmed=True)[0]           # only a deliberate re-arm clears it


def test_failure_latches_and_a_reconnect_after_failure_needs_rearm():
    rig = Rig().ready()
    rig.tick(6.0)
    assert rig.link.state is LinkState.FAILED and rig.link.needs_rearm
    assert rig.port.drives()[-1].startswith(b"D,7,0,0,")                      # best-effort final neutral
    assert rig.link.connect(rig.clock()) and rig.link.health(rig.clock()).reconnects == 1
    assert rig.link.needs_rearm


# ---------------------------------------------------------------- the controller report

def test_controller_report_reflects_the_st_line_and_goes_away_with_the_link():
    rig = Rig().connected()
    report = rig.link.controller(rig.clock())
    assert report.armed and report.link_up and report.requested == "PC" and report.active == "PC" and report.pc_ok
    assert report.fresh and report.fault == "NONE" and report.token == 7
    rig.tick(0.7)
    assert rig.link.controller(rig.clock()).fresh is False
    rig.link.disconnect(rig.clock())
    assert rig.link.controller(rig.clock()) is None


# ---------------------------------------------------------------- source rules

def _tree(name: str) -> ast.Module:
    return ast.parse((BACKEND / name).read_text(encoding="utf-8"))


def test_no_socket_or_serial_import_and_only_the_existing_udp_port_is_used():
    for name in ("wifi_link.py", "control_modes.py", "acceptance.py", "ai_observation.py"):
        tree = _tree(name)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                modules = [node.module or ""]
            else:
                continue
            for module in modules:
                assert module.split(".")[0] not in {"socket", "serial", "cv2", "PySide6", "threading", "asyncio"}, (name, module)
    source = (BACKEND / "wifi_link.py").read_text(encoding="utf-8")
    assert source.count("client.UdpPort(") == 1 and "handoff_bench_client" in source
    names = {node.id for node in ast.walk(_tree("wifi_link.py")) if isinstance(node, ast.Name)}
    attrs = {node.attr for node in ast.walk(_tree("wifi_link.py")) if isinstance(node, ast.Attribute)}
    assert not names & {"socket", "serial"} and "open_port" not in attrs and attrs & {"UdpPort"} == {"UdpPort"}


def test_send_drive_is_the_only_non_neutral_path():
    tree = _tree("wifi_link.py")
    calls = []
    for function in [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]:
        for node in ast.walk(function):
            if isinstance(node, ast.Call):
                name = getattr(node.func, "attr", getattr(node.func, "id", ""))
                if name in {"_transmit", "format_command"}:
                    calls.append((function.name, name, node))
    transmit_callers = {(fn, ) for fn, name, _ in calls if name == "_transmit"}
    assert {fn for (fn,) in transmit_callers} == {"send_drive", "send_neutral", "_final_neutral"}
    for fn, name, node in calls:
        if name == "_transmit" and fn != "send_drive":
            assert [isinstance(a, ast.Constant) and a.value == 0 for a in node.args[:2]] == [True, True], fn
    assert [fn for fn, name, _ in calls if name == "format_command"] == ["_transmit"]
    # in the manager: non-zero commands go only through link.send_drive, from compute_command
    modes = _tree("control_modes.py")
    users = {fn.name for fn in ast.walk(modes) if isinstance(fn, ast.FunctionDef)
             for node in ast.walk(fn) if isinstance(node, ast.Call) and getattr(node.func, "attr", "") == "send_drive"}
    assert users == {"compute_command"}
    for fn in ast.walk(modes):
        if isinstance(fn, ast.FunctionDef):
            for node in ast.walk(fn):
                if isinstance(node, ast.Call) and getattr(node.func, "attr", "") in {"_transmit", "format_command"}:
                    raise AssertionError(f"control_modes.{fn.name} must not format commands itself")
