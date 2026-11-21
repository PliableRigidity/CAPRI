"""The ONE Wi-Fi link manager for AI control. Pure Python: no Qt, no threads, no clock of its own, no hard-wired socket.

It sits on the existing, tested client code (``scripts/handoff_bench_client.py``): ``ClientCore`` parses the firmware's
ST / CH / HELLO / NET lines and ``UdpPort`` is the UDP transport. The protocol is not re-implemented here.

    PC -> robot    ?                          status request (the robot only reports to an address it has heard from)
                   D,<tok>,<thr>,<steer>,<n>  one drive datagram; <tok> is the token of the newest ST line, <n> only rises
    robot -> PC    ST ...  (at least every 0.2 s)   HELLO ...  (when asked)   NET ...  (twice a second)

What this module guarantees
---------------------------
* ``connect()`` sends status requests (``?``) only. It never sends a drive command.
* A non-zero command can leave this module through exactly one door, ``send_drive``. It is refused unless EVERY gate is
  open: the mode manager's ``drive_allowed`` callback, link CONNECTED, a fresh ST line (< 0.5 s), the controller itself
  reporting armed + PC selected + PC accepted + radio link up + no fault, the token the operator re-armed on, values
  inside the firmware's ``pc_limit`` and the ``max_command`` set here (default 60 of 255), an observation younger than
  ``max_observation_age_s`` (default 0.25 s), a previous command that is not stale, and the 50 Hz rate limit.
* ``send_neutral`` (always 0,0) is allowed whenever the link is connected. It is the only other way to send a command.
* After ANY disconnect, failure, reconnect, token change, controller restart (uptime going backwards), latched fault
  (PC_STALE, RADIO_LOST, any), or ARM / MODE change, the link is in ``needs_rearm``: it sends only neutral commands and
  status requests until the operator re-confirmed through the mode manager (``rearm``). A reconnect never resumes motion.
* The firmware, not this module, enforces the stops. If Wi-Fi fails, the firmware stops the motors 250 ms after the last
  command it received. A software stop button that depends on this link cannot work once the link has failed.

How to drive it (one thread, about 50 Hz, e.g. a ``QThread`` loop or a ``QTimer`` of 20 ms)::

    link.connect(now)                      # once, when the operator asks
    while running:
        manager.tick(now)                  # calls link.tick(now): reads replies, sends '?' probes, updates state
        sleep(0.02)
    link.shutdown()                        # on application exit

Not thread-safe by design: every method is called from the one thread that ticks it (the mode manager's).

Every time argument is on the same monotonic clock (``time.monotonic()``). Tests pass a fake clock and a fake transport,
so nothing here ever opens a socket by itself: only the DEFAULT transport factory does, and only inside ``connect()`` /
``discover()``. A link built with an injected transport factory reports ``real = False``: it is a MOCK link, and the
acceptance workflow and the autonomous gate refuse to count it as hardware.

Nothing here is evidence about the robot. Software tests of this module are NOT physical verification.
"""

from __future__ import annotations

import ipaddress
import re
import statistics
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import handoff_bench_client as client  # noqa: E402  (the tested PC client: parsing, ClientCore, UdpPort)

DEFAULT_HOST = client.WIFI_HOST                 # 192.168.4.1: the sketch's soft access point
DEFAULT_PORT = client.WIFI_UDP_PORT             # 4210
WIFI_FIRMWARE_ID = "autobot-handoff-wifi-1"
KNOWN_FIRMWARE_IDS = frozenset(client.FIRMWARE_IDS)

STATUS_FRESH_S = client.STATUS_FRESH_S          # 0.5: the firmware sends ST at least every 0.2 s
MIN_SEND_PERIOD_S = client.SEND_PERIOD_S        # 0.02: 50 Hz at most
DEFAULT_MAX_COMMAND = 60                        # of 255, about 23 %; the firmware's own limit is 128
DEFAULT_MAX_OBSERVATION_AGE_S = 0.25
DEFAULT_PC_TIMEOUT_S = 0.25                     # PC_TIMEOUT in the sketch
CONNECT_TIMEOUT_S = 4.0
LOST_AFTER_S = 5.0                              # no ST for this long: the link has FAILED, not merely lagged
PROBE_PERIOD_S = client.PROBE_PERIOD_S          # 0.5 while no status arrives
LATENCY_PROBE_PERIOD_S = 1.0
LATENCY_WINDOW = 9
PROBE_ANSWER_TIMEOUT_S = 2.0
RESTART_DROP_MS = 100                           # uptime must fall by this much to count as a restart (as control_status)
LOSS_WINDOW_S = 30.0
LOSS_RATIO_HIGH = 0.10                          # firmware-counted old+bad datagrams over datagrams received
LOSS_MIN_DATAGRAMS = 20
GAPS_HIGH = 3                                   # status gaps inside LOSS_WINDOW_S

# ---------------------------------------------------------------- endpoint


@dataclass(frozen=True)
class Endpoint:
    host: str = DEFAULT_HOST
    port: int = DEFAULT_PORT

    @property
    def label(self) -> str:
        return f"{self.host}:{self.port}"


_HOSTNAME = re.compile(r"^(?=.{1,253}$)[A-Za-z0-9]([A-Za-z0-9-]{0,61}[A-Za-z0-9])?(\.[A-Za-z0-9]([A-Za-z0-9-]{0,61}[A-Za-z0-9])?)*$")


def validate_endpoint(host: object, port: object = DEFAULT_PORT) -> tuple[Endpoint | None, str]:
    """(endpoint, "") when acceptable, else (None, plain reason). Only a single unicast IPv4 address or a host name."""
    text = host.strip() if isinstance(host, str) else ""
    if not text:
        return None, f"enter the robot's IP address or name (the robot's own network uses {DEFAULT_HOST})"
    if any(ch in text for ch in ":/\\@ \t,;<>*?") or "://" in text:
        return None, (f"{text!r} is not a plain address: type only the IP address or name here "
                      "(the port goes in its own box)")
    if re.fullmatch(r"[0-9.]+", text):
        try:
            address = ipaddress.IPv4Address(text)
        except ValueError:
            return None, f"{text!r} is not a valid IPv4 address (four numbers 0-255 separated by dots)"
        octets = address.packed
        if address.is_unspecified:
            return None, "0.0.0.0 is not a robot address"
        if address.is_multicast:
            return None, f"{text} is a multicast address: commands may only go to one robot (a unicast address)"
        if address == ipaddress.IPv4Address("255.255.255.255") or octets[3] == 255:
            return None, f"{text} looks like a broadcast address: commands may only go to one robot (a unicast address)"
        if octets[3] == 0:
            return None, f"{text} looks like a network address, not a robot: use the robot's own address"
        if address.is_reserved:
            return None, f"{text} is in a reserved range and is not a robot address"
    elif not _HOSTNAME.match(text):
        return None, f"{text!r} is not a valid host name or IPv4 address"
    elif text.lower() in {"broadcasthost", "broadcast", "all"}:
        return None, f"{text!r} stands for broadcast: commands may only go to one robot"
    if isinstance(port, bool):
        return None, "the port must be a number from 1 to 65535"
    try:
        number = int(str(port).strip()) if not isinstance(port, int) else port
    except ValueError:
        return None, f"{port!r} is not a port number (1 to 65535)"
    if not 1 <= number <= 65535:
        return None, f"port {number} is outside 1 to 65535"
    return Endpoint(text, number), ""


# ---------------------------------------------------------------- states, failures, results


class LinkState(Enum):
    DISCONNECTED = "DISCONNECTED"
    CONNECTING = "CONNECTING"
    CONNECTED = "CONNECTED"
    DEGRADED = "DEGRADED"
    FAILED = "FAILED"


# kind -> (what went wrong, what to try). Plain language: shown to the operator as it is.
FAILURES: dict[str, tuple[str, str]] = {
    "NOT_JOINED": (
        "The laptop could not send to the robot: it is probably not joined to the robot's Wi-Fi network.",
        "Join the robot's own network (default name AutoBot-Handoff) in the Windows Wi-Fi list. Windows will say 'No internet, "
        "secured': that is expected, stay connected. Untick 'Connect automatically' on your other networks so Windows does "
        "not switch away."),
    "BLOCKED": (
        "Windows or security software refused to open the Wi-Fi link.",
        "Allow Python through Windows Firewall (Windows Security > Firewall > Allow an app), including on the Public network "
        "that the robot's network is classed as, then connect again."),
    "IN_USE": (
        "Another program is holding the link.",
        "Close other robot tools that use the same network port (handoff_bench_client.py, motion_check.py, a second Command "
        "Centre window), then connect again."),
    "NO_REPLY": (
        "The robot did not answer.",
        "Check in this order: (1) the laptop is joined to the robot's Wi-Fi; (2) the robot is powered and running the "
        "Wi-Fi sketch (ELRS-Handoff-WiFi), not the USB build; (3) the address is right (default 192.168.4.1, port 4210); "
        "(4) Windows Firewall allows Python on the Public network, because the robot's replies come in from outside."),
    "WRONG_ENDPOINT": (
        "Something answered at that address, but it is not speaking the robot's protocol.",
        "The IP address or port is probably wrong. The robot's own network uses 192.168.4.1, port 4210."),
    "UNKNOWN_FIRMWARE": (
        "The device identified itself as firmware this app does not know.",
        "This app only drives the builds it knows (autobot-handoff-wifi-1 for Wi-Fi). Check what the robot is running; "
        "nothing was sent to it except status requests."),
    "NO_HELLO": (
        "The robot sends status but never said which firmware it is running.",
        "It may be an older sketch. Wait a few seconds and connect again; if it repeats, the robot is not running the Wi-Fi "
        "build this app expects."),
    "LINK_LOST": (
        "The robot stopped answering.",
        "The robot stops its motors by itself 250 ms after the last command. Check the laptop is still joined to the "
        "robot's Wi-Fi and that the robot has power (on 20 Nov the supply collapsed and the controller restarted). "
        "Connect again, then re-arm deliberately."),
    "TRANSPORT_ERROR": (
        "The Wi-Fi link failed while reading.",
        "Check the laptop's Wi-Fi connection, then connect again."),
}


def classify_open_error(error: BaseException) -> str:
    code = getattr(error, "winerror", None) or getattr(error, "errno", None)
    text = str(error).lower()
    if code in (10048, 98, 48) or "address already in use" in text or "only one usage" in text:
        return "IN_USE"
    if code in (10013, 13) or isinstance(error, PermissionError) or "forbidden" in text or "permission" in text:
        return "BLOCKED"
    return "NOT_JOINED"


@dataclass(frozen=True)
class SendResult:
    """Outcome of one send. Truthy only when a datagram was written. Unpacks as (sent, reason)."""

    sent: bool
    reason: str
    code: str = ""
    sequence: int | None = None

    def __bool__(self) -> bool:
        return self.sent

    def __iter__(self):
        yield self.sent
        yield self.reason


@dataclass(frozen=True)
class ControllerReport:
    """What the controller (the firmware's ST line) says now. Never an inference."""

    link_up: bool
    armed: bool
    requested: str
    active: str
    pc_ok: bool
    block: str
    fault: str
    stop: str
    token: int
    uptime_ms: int
    pc_age_ms: int
    left: int
    right: int
    age_s: float
    fresh: bool
    token_stable_s: float | None


@dataclass(frozen=True)
class LossInfo:
    """Packet loss as far as it is measurable. A field is None when it cannot be measured; nothing is estimated."""

    firmware_rx: int | None
    firmware_bad: int | None
    firmware_old: int | None
    firmware_other: int | None
    firmware_loss_ratio: float | None          # (old + bad) / rx over the recent window, None until enough datagrams
    pc_status_gaps: int                         # times the ST heartbeat went late (> 0.5 s) since connecting
    pc_status_gaps_recent: int
    pc_send_errors: int
    pc_probe_timeouts: int
    high: bool
    note: str


@dataclass(frozen=True)
class LinkHealth:
    state: LinkState
    endpoint: Endpoint
    real: bool                                  # False: injected / mock transport, never counts as hardware
    local_ip: str | None
    peer_ip: str | None
    last_rx_age_s: float | None
    last_success_unix: float | None
    latency_ms: float | None
    latency_samples: int
    heartbeat_age_s: float | None
    loss: LossInfo | None
    firmware_id: str | None
    protocol_ok: bool | None
    protocol_note: str
    reconnects: int
    restarts: int
    error: str | None
    error_kind: str | None
    hint: str | None
    needs_rearm: bool
    latched_reason: str | None
    unavailable: dict[str, str] = field(default_factory=dict)      # field name -> why it is None


@dataclass(frozen=True)
class DiscoveryResult:
    endpoint: Endpoint
    found: bool
    firmware_id: str | None
    known_firmware: bool | None
    protocol_ok: bool | None
    protocol_note: str
    hello: dict[str, str]
    status_seen: bool
    probes_sent: int
    error: str | None
    hint: str | None


def default_transport_factory(endpoint: Endpoint):
    """The one place a socket is created: ``UdpPort`` from the existing client. Called only by connect() / discover()."""
    return client.UdpPort(endpoint.host, endpoint.port)


def _int(value: Any) -> int | None:
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return None


def protocol_check(hello: dict[str, str], pc_limit: int | None) -> tuple[bool | None, str]:
    if not hello:
        return None, "no HELLO line received yet"
    notes = []
    firmware = hello.get("fw")
    if firmware not in KNOWN_FIRMWARE_IDS:
        notes.append(f"firmware id {firmware!r} is not in the known set {sorted(KNOWN_FIRMWARE_IDS)}")
    if pc_limit is None:
        notes.append("HELLO has no readable pc_limit")
    timeout = _int(hello.get("pc_timeout_ms"))
    if timeout is None or timeout <= 0:
        notes.append("HELLO has no readable pc_timeout_ms")
    return (not notes), "; ".join(notes) if notes else "firmware id known; pc_limit and pc_timeout_ms present"


# ---------------------------------------------------------------- the link


class WifiLink:
    def __init__(self, endpoint: Endpoint | None = None, *, transport_factory: Callable[[Endpoint], Any] | None = None,
                 clock: Callable[[], float] = time.monotonic, unix_clock: Callable[[], float] = time.time,
                 max_command: int = DEFAULT_MAX_COMMAND, max_observation_age_s: float = DEFAULT_MAX_OBSERVATION_AGE_S,
                 connect_timeout_s: float = CONNECT_TIMEOUT_S, lost_after_s: float = LOST_AFTER_S) -> None:
        checked, why = validate_endpoint((endpoint or Endpoint()).host, (endpoint or Endpoint()).port)
        if checked is None:
            raise ValueError(why)
        if not (isinstance(max_command, int) and 0 <= max_command <= client.MAX_COMMAND):
            raise ValueError("max_command must be an integer 0..255")
        self.endpoint = checked
        self._factory = transport_factory or default_transport_factory
        # Only the built-in transport is a REAL link. Tests may set this attribute to exercise the gates with a fake
        # transport; nothing else should.
        self.real = transport_factory is None
        self._clock, self._unix = clock, unix_clock
        self.max_command = max_command
        self.max_observation_age_s = max_observation_age_s
        self.connect_timeout_s, self.lost_after_s = connect_timeout_s, lost_after_s
        self.drive_allowed: Callable[[], bool] | None = None       # installed by the mode manager
        self.state = LinkState.DISCONNECTED
        self.error: str | None = None
        self.error_kind: str | None = None
        self.hint: str | None = None
        self.reconnects = 0
        self.restarts = 0
        self.latch_log: deque[str] = deque(maxlen=50)
        self._closed = False
        self._port: Any = None
        self._core: Any = None
        self._sequence = 0                      # rises for the life of this object, across reconnects
        self._needs_rearm = True
        self._latched_reason: str | None = "no operator re-arm yet: the link starts needing a deliberate re-arm"
        self._baseline: tuple[int, bool, str] | None = None
        self._had_connected = False
        self._reset_session()

    # ------------------------------------------------------------ session bookkeeping

    def _reset_session(self) -> None:
        self._connect_started = 0.0
        self._command_ever_sent = False
        self._last_tx_at: float | None = None
        self._last_status_obj: Any = None
        self._prev_uptime: int | None = None
        self._last_token: int | None = None
        self._token_since: float | None = None
        self._last_rx_at: float | None = None
        self._last_success_unix: float | None = None
        self._probe_sent_at: float | None = None
        self._last_probe_at = float("-inf")
        self._latencies: deque[float] = deque(maxlen=LATENCY_WINDOW)
        self._net_samples: deque[tuple[float, int, int, int]] = deque()
        self._net: dict[str, str] = {}
        self._gaps: deque[float] = deque()
        self._gap_total = 0
        self._send_errors = 0
        self._probe_timeouts = 0
        self._loss_high = False

    # ------------------------------------------------------------ configuration

    def configure(self, endpoint: Endpoint | str, port: object | None = None) -> tuple[bool, str]:
        """Set the robot's address. Refused while a link is open (disconnect first) or when it is not a unicast address."""
        if self.state in (LinkState.CONNECTING, LinkState.CONNECTED, LinkState.DEGRADED):
            return False, "disconnect first: the address cannot change while the link is open"
        if isinstance(endpoint, Endpoint):
            host, number = endpoint.host, endpoint.port
        else:
            host, number = endpoint, DEFAULT_PORT if port is None else port
        checked, why = validate_endpoint(host, number)
        if checked is None:
            return False, why
        self.endpoint = checked
        return True, ""

    # ------------------------------------------------------------ properties for the manager and the UI

    @property
    def needs_rearm(self) -> bool:
        return self._needs_rearm

    @property
    def latched_reason(self) -> str | None:
        return self._latched_reason if self._needs_rearm else None

    @property
    def firmware_id(self) -> str | None:
        return (self._core.hello.get("fw") if self._core is not None and self._core.hello else None)

    @property
    def pc_limit(self) -> int | None:
        return None if self._core is None else self._core.pc_limit

    @property
    def pc_timeout_s(self) -> float:
        ms = _int(self._core.hello.get("pc_timeout_ms")) if self._core is not None and self._core.hello else None
        return ms / 1000.0 if ms and ms > 0 else DEFAULT_PC_TIMEOUT_S

    @property
    def connected(self) -> bool:
        return self.state in (LinkState.CONNECTED, LinkState.DEGRADED)

    @property
    def drive_command_ever_sent(self) -> bool:
        return self._command_ever_sent

    # ------------------------------------------------------------ lifecycle

    def connect(self, now: float | None = None) -> bool:
        """Open the transport and ask for status. Sends ``?`` only: never a drive command. Returns False on a failure."""
        now = self._clock() if now is None else now
        if self._closed:
            self.error = "the link manager was shut down (the application is closing)"
            return False
        if self.state in (LinkState.CONNECTING, LinkState.CONNECTED, LinkState.DEGRADED):
            return True
        reconnect = self._had_connected
        self._reset_session()
        self.error = self.error_kind = self.hint = None
        try:
            port = self._factory(self.endpoint)
        except Exception as error:                      # opening can fail: no route, port in use, firewall
            self._fail(classify_open_error(error), now, detail=str(error))
            return False
        self._port = port
        self._core = client.ClientCore(speed=0, turn=0, sequenced=True,
                                       link_label=getattr(port, "label", f"Wi-Fi {self.endpoint.label} (UDP)"))
        self._core.sequence = self._sequence
        if reconnect:
            self.reconnects += 1
        self.state = LinkState.CONNECTING
        self._connect_started = now
        self._needs_rearm = True
        self._latched_reason = ("reconnected: the operator must re-confirm before anything can move" if reconnect
                                else "new connection: the operator must confirm before anything can move")
        self.latch_log.append(self._latched_reason)
        self._baseline = None
        self._core.probe_due(now)                       # starts the probe period so tick() does not double up
        self._send_probe(now)
        return True

    def disconnect(self, now: float | None = None) -> None:
        """One neutral command ONLY if a drive command was sent this session, then close. Safe to call at any time."""
        now = self._clock() if now is None else now
        if self._port is not None:
            self._final_neutral(now)
            self._close_port()
        if self.state is not LinkState.FAILED:
            self.state = LinkState.DISCONNECTED
        self._latch("disconnected")
        self._baseline = None

    def shutdown(self, now: float | None = None) -> None:
        """Application-exit hook. Neutral, then close. Idempotent."""
        if self._closed:
            return
        self.disconnect(now)
        self._closed = True
        self.state = LinkState.DISCONNECTED

    def _close_port(self) -> None:
        port, self._port = self._port, None
        if port is not None:
            try:
                port.close()
            except Exception:
                pass

    def _final_neutral(self, now: float) -> None:
        """The single neutral at the end of a session in which a drive command was sent. Never rate limited."""
        if self._command_ever_sent and self._core is not None and self._core.status is not None:
            self._transmit(0, 0, now)

    def _fail(self, kind: str, now: float, detail: str = "") -> None:
        text, hint = FAILURES[kind]
        self.error = text + (f" ({detail})" if detail else "")
        self.error_kind, self.hint = kind, hint
        if self._port is not None:
            self._final_neutral(now)                    # best effort: the network may already be gone
            self._close_port()
        self.state = LinkState.FAILED
        self._latch(f"link failed: {text}")

    # ------------------------------------------------------------ latching

    def _latch(self, reason: str) -> None:
        self.latch_log.append(reason)
        if not self._needs_rearm:
            self._needs_rearm = True
            self._latched_reason = reason

    def rearm(self, now: float | None = None, *, operator_confirmed: bool = False) -> tuple[bool, str]:
        """Clear ``needs_rearm``. Called ONLY by the mode manager, after the operator's confirmation was validated.

        Refused unless the controller itself reports, right now: armed, MODE=PC, PC accepted, radio link up, no fault.
        """
        now = self._clock() if now is None else now
        if not operator_confirmed:
            return False, "re-arm needs the operator's confirmation through the mode manager"
        if self.state is not LinkState.CONNECTED:
            return False, f"the link is {self.state.value}, not CONNECTED"
        core = self._core
        if core is None or not core.fresh(now):
            return False, "no fresh status from the controller"
        problem = self._controller_problem(core.status)
        if problem:
            return False, problem
        self._baseline = (core.status.token, core.status.armed, core.status.requested)
        self._needs_rearm = False
        self._latched_reason = None
        return True, "re-armed: the controller is armed, PC selected and accepting PC commands"

    @staticmethod
    def _controller_problem(status: Any) -> str:
        if not status.link:
            return "the controller reports the radio link down"
        if status.fault != "NONE":
            return f"the controller has fault {status.fault} latched: turn ARM off, then on, on the RadioMaster"
        if not status.armed:
            return "the controller reports DISARMED (ARM switch is off)"
        if status.requested != "PC":
            return f"the MODE switch asks for {status.requested}, not PC"
        if status.active != "PC" or not status.pc_ok:
            return f"the controller is not accepting PC commands (active={status.active}, block={status.block})"
        return ""

    # ------------------------------------------------------------ receiving

    def poll(self, now: float) -> list[str]:
        """Read whatever has arrived, one line at a time, and update the state. Returns the lines (for logging)."""
        if self._port is None or self._core is None:
            return []
        try:
            data = self._port.read(4096)
        except Exception as error:
            self._fail("TRANSPORT_ERROR", now, detail=str(error))
            return []
        lines: list[str] = []
        for piece in (data or b"").split(b"\n"):
            if not piece.strip():
                continue
            before = self._core.status
            got = self._core.receive(piece + b"\n", now)
            lines.extend(got)
            for text in got:
                self._on_line(text, now, before)
                before = self._core.status
        self._update_state(now)
        return lines

    def _on_line(self, text: str, now: float, status_before: Any) -> None:
        split = client.split_line(text)
        kind = split[0] if split else ""
        if kind in ("ST", "HELLO", "NET", "CH"):
            self._last_rx_at = now
        if kind == "ST" and self._core.status is not status_before:
            self._on_status(self._core.status, now)
        elif kind == "HELLO":
            if self._probe_sent_at is not None:
                self._latencies.append(max(0.0, now - self._probe_sent_at))
                self._probe_sent_at = None
        elif kind == "NET":
            self._on_net(split[1], now)

    def _on_status(self, status: Any, now: float) -> None:
        self._last_success_unix = self._unix()
        if self._prev_uptime is not None and status.firmware_ms + RESTART_DROP_MS <= self._prev_uptime:
            self.restarts += 1
            self._net_samples.clear()
            self._latch(f"controller restarted (uptime fell from {self._prev_uptime} ms to {status.firmware_ms} ms)")
        self._prev_uptime = status.firmware_ms
        if self._last_token is None or status.token != self._last_token:
            self._token_since = now
        self._last_token = status.token
        if self._baseline is not None and not self._needs_rearm:
            token, armed, requested = self._baseline
            if status.fault != "NONE":
                self._latch(f"controller latched fault {status.fault}")
            elif not status.link:
                self._latch("radio link lost")
            elif status.token != token:
                self._latch(f"controller token changed ({token} -> {status.token}): control changed hands")
            elif status.armed != armed:
                self._latch(f"ARM changed: the controller now reports armed={int(status.armed)}")
            elif status.requested != requested:
                self._latch(f"MODE changed: the switch now asks for {status.requested}")
            elif not status.pc_ok:
                self._latch(f"the controller no longer accepts PC commands (block={status.block})")

    def _on_net(self, fields: dict[str, str], now: float) -> None:
        self._net = dict(fields)
        rx, bad, old = (_int(fields.get(key)) for key in ("rx", "bad", "old"))
        if None in (rx, bad, old):
            return
        if self._net_samples and rx < self._net_samples[-1][1]:
            self._net_samples.clear()                   # the counters went backwards: the controller restarted
        self._net_samples.append((now, rx, bad, old))
        while self._net_samples and now - self._net_samples[0][0] > LOSS_WINDOW_S:
            self._net_samples.popleft()

    # ------------------------------------------------------------ state machine

    def heartbeat_age(self, now: float) -> float | None:
        core = self._core
        return None if core is None or core.status is None else max(0.0, now - core.status_at)

    def _firmware_loss_ratio(self) -> float | None:
        if len(self._net_samples) < 2:
            return None
        first, last = self._net_samples[0], self._net_samples[-1]
        received = last[1] - first[1]
        if received < LOSS_MIN_DATAGRAMS:
            return None
        return max(0.0, ((last[2] - first[2]) + (last[3] - first[3])) / received)

    def _update_state(self, now: float) -> None:
        core = self._core
        if core is None or self._port is None or self.state in (LinkState.DISCONNECTED, LinkState.FAILED):
            return
        while self._gaps and now - self._gaps[0] > LOSS_WINDOW_S:
            self._gaps.popleft()
        hello = core.hello
        if hello and not core.firmware_known:
            self._fail("UNKNOWN_FIRMWARE", now, detail=f"fw={hello.get('fw')!r}")
            return
        age = self.heartbeat_age(now)
        if self.state is LinkState.CONNECTING:
            if core.fresh(now) and hello:
                self.state = LinkState.CONNECTED
                self._had_connected = True
            elif now - self._connect_started > self.connect_timeout_s:
                self._fail(self._timeout_kind(core), now, detail=f"no usable reply within {self.connect_timeout_s:.0f} s")
            return
        # CONNECTED or DEGRADED
        if age is None or age > self.lost_after_s:
            self._fail("LINK_LOST", now, detail="no status for " + ("ever" if age is None else f"{age:.1f} s"))
            return
        ratio = self._firmware_loss_ratio()
        self._loss_high = (ratio is not None and ratio >= LOSS_RATIO_HIGH) or len(self._gaps) >= GAPS_HIGH
        late = age > STATUS_FRESH_S
        if late and self.state is LinkState.CONNECTED:
            self._gaps.append(now)
            self._gap_total += 1
            self._latch(f"status was late ({age:.1f} s): the firmware may have stopped on its own 250 ms rule")
        if late or self._loss_high:
            self.state = LinkState.DEGRADED
        else:
            self.state = LinkState.CONNECTED

    def _timeout_kind(self, core: Any) -> str:
        if core.status is not None and not core.hello:
            return "NO_HELLO"
        if core.status is None and core.unknown_lines:
            return "WRONG_ENDPOINT"
        if self._send_errors and core.status is None:
            return "NOT_JOINED"
        return "NO_REPLY"

    # ------------------------------------------------------------ periodic work

    def tick(self, now: float) -> None:
        """Read replies, update the state, send status requests when they are due. Never sends a drive command."""
        self.poll(now)
        core = self._core
        if core is None or self._port is None:
            return
        if self._probe_sent_at is not None and now - self._probe_sent_at > PROBE_ANSWER_TIMEOUT_S:
            self._probe_sent_at = None
            self._probe_timeouts += 1                   # the PC's own timeout: a '?' that got no HELLO back
        if self.state is LinkState.CONNECTING:
            if now - self._last_probe_at >= PROBE_PERIOD_S:
                self._send_probe(now)
        elif self.connected:
            if not core.fresh(now):
                if core.probe_due(now):
                    self._send_probe(now)
            elif now - self._last_probe_at >= LATENCY_PROBE_PERIOD_S:
                self._send_probe(now)

    def _send_probe(self, now: float) -> None:
        self._last_probe_at = now
        if self._write(b"?\n") and self._probe_sent_at is None:
            self._probe_sent_at = now

    def _write(self, data: bytes) -> bool:
        port = self._port
        if port is None:
            return False
        try:
            written = port.write(data)
        except Exception as error:
            self._send_errors += 1
            self.error = f"send failed: {error}"
            return False
        problem = getattr(port, "error", None)
        if problem:
            self._send_errors += 1
            return False
        return written != 0

    # ------------------------------------------------------------ sending commands

    def controller(self, now: float | None = None) -> ControllerReport | None:
        """The newest ST line as a report, or None when there is no open link or none has arrived."""
        now = self._clock() if now is None else now
        core = self._core
        if core is None or core.status is None or self._port is None or self.state in (LinkState.DISCONNECTED, LinkState.FAILED):
            return None
        s = core.status
        age = max(0.0, now - core.status_at)
        stable = None if self._token_since is None else max(0.0, now - self._token_since)
        return ControllerReport(link_up=s.link, armed=s.armed, requested=s.requested, active=s.active, pc_ok=s.pc_ok,
                                block=s.block, fault=s.fault, stop=s.stop, token=s.token, uptime_ms=s.firmware_ms,
                                pc_age_ms=s.pc_age_ms, left=s.left, right=s.right, age_s=age, fresh=age <= STATUS_FRESH_S,
                                token_stable_s=stable)

    def send_neutral(self, now: float | None = None, *, force: bool = False) -> SendResult:
        """A 0,0 command. Allowed whenever the link is connected; at most 50 Hz unless ``force`` (final stops)."""
        now = self._clock() if now is None else now
        if not self.connected:
            return SendResult(False, f"the link is {self.state.value}: no neutral can be sent", "NOT_CONNECTED")
        if not force and self._last_tx_at is not None and now - self._last_tx_at < MIN_SEND_PERIOD_S - 1e-9:
            return SendResult(False, "rate limit: at most one command every 20 ms", "RATE_LIMIT")
        return self._transmit(0, 0, now)

    def send_drive(self, throttle: float, steer: float, now: float, observation_t: float | None = None) -> SendResult:
        """The ONLY way a non-zero command leaves this module. Refused unless every gate is open (see the module docstring)."""
        core = self._core

        def refuse(code: str, why: str) -> SendResult:
            return SendResult(False, why, code)

        if self.drive_allowed is None or not self.drive_allowed():
            return refuse("NOT_ALLOWED", "the mode manager has not allowed driving (autonomous authority is not held)")
        if self.state is not LinkState.CONNECTED or core is None or self._port is None:
            return refuse("NOT_CONNECTED", f"the link is {self.state.value}, not CONNECTED")
        if self._needs_rearm:
            return refuse("NEEDS_REARM", f"needs re-arm: {self._latched_reason}")
        if not core.fresh(now):
            return refuse("STATUS_STALE", "the controller's status is older than 0.5 s")
        s = core.status
        problem = self._controller_problem(s)
        if problem:
            return refuse("CONTROLLER", problem)
        if self._baseline is None or s.token != self._baseline[0]:
            return refuse("TOKEN", "the controller's token is not the one the operator armed on")
        limit = self.max_command if core.pc_limit is None else min(self.max_command, core.pc_limit)
        if core.pc_limit is None:
            return refuse("NO_LIMIT", "the firmware's pc_limit is unknown (no usable HELLO): the value cannot be bounded")
        values = []
        for name, value in (("throttle", throttle), ("steer", steer)):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or value != value or abs(value) == float("inf"):
                return refuse("VALUE", f"{name} is not a finite number")
            values.append(int(round(value)))
        if abs(values[0]) > limit or abs(values[1]) > limit:
            return refuse("LIMIT", f"command {values[0]},{values[1]} is above the limit {limit} "
                                   f"(max_command {self.max_command}, firmware pc_limit {core.pc_limit})")
        if values[0] or values[1]:
            if observation_t is None:
                return refuse("NO_OBSERVATION_TIME", "a non-zero command needs the time of the observation it came from")
            age = now - observation_t
            if age > self.max_observation_age_s:
                return refuse("OBSERVATION_STALE", f"the observation is {age * 1000:.0f} ms old "
                                                   f"(limit {self.max_observation_age_s * 1000:.0f} ms): not sent")
        if self._last_tx_at is None:
            return refuse("NEUTRAL_FIRST", "no neutral command has been sent yet: the firmware needs a neutral first")
        gap = now - self._last_tx_at
        if gap > self.pc_timeout_s:
            self._latch(f"the command stream stopped for {gap * 1000:.0f} ms (the firmware's stop is {self.pc_timeout_s * 1000:.0f} ms)")
            return refuse("PREVIOUS_STALE", f"the previous command is {gap * 1000:.0f} ms old: the firmware has probably stopped; "
                                            "needs re-arm")
        if gap < MIN_SEND_PERIOD_S - 1e-9:
            return refuse("RATE_LIMIT", "rate limit: at most one command every 20 ms")
        return self._transmit(values[0], values[1], now)

    def _transmit(self, throttle: int, steer: int, now: float) -> SendResult:
        """Format and write one D datagram with the newest token and the next sequence number."""
        core = self._core
        if core is None or core.status is None:
            return SendResult(False, "no status line yet: the token is unknown", "NO_TOKEN")
        self._sequence += 1
        sequence = self._sequence
        core.sequence = sequence
        data = client.format_command(core.status.token, throttle, steer, sequence)
        if not self._write(data):
            return SendResult(False, "the datagram could not be written: " + str(getattr(self._port, "error", None) or self.error),
                              "SEND_FAILED")
        self._command_ever_sent = True
        self._last_tx_at = now
        return SendResult(True, "sent", "SENT", sequence)

    # ------------------------------------------------------------ discovery

    def discover(self, timeout: float = 2.0, *, sleep: Callable[[float], None] = time.sleep) -> DiscoveryResult:
        """Probe ONLY the configured endpoint with ``?`` and report what answers. No scanning, no broadcast.

        Blocking (up to ``timeout``): call it from a worker thread. Refused while the link is open, because a second
        socket asking for status would take the robot's status stream away from the open link.
        """
        endpoint = self.endpoint
        if self.state in (LinkState.CONNECTING, LinkState.CONNECTED, LinkState.DEGRADED):
            return DiscoveryResult(endpoint, False, self.firmware_id, None, None, "", {}, False, 0,
                                   "the link is already open: disconnect before probing", None)
        try:
            port = self._factory(endpoint)
        except Exception as error:
            kind = classify_open_error(error)
            return DiscoveryResult(endpoint, False, None, None, None, "", {}, False, 0, f"{FAILURES[kind][0]} ({error})",
                                   FAILURES[kind][1])
        core = client.ClientCore(speed=0, turn=0, sequenced=True)
        started = self._clock()
        probes, last_probe, errors = 0, float("-inf"), 0
        try:
            while True:
                now = self._clock()
                if now - last_probe >= 0.4:
                    last_probe = now
                    probes += 1
                    try:
                        port.write(b"?\n")
                    except Exception:
                        errors += 1
                    if getattr(port, "error", None):
                        errors += 1
                try:
                    data = port.read(4096)
                except Exception:
                    data = b""
                if data:
                    core.receive(data, now)
                if core.hello and core.status is not None:
                    break
                if now - started >= timeout:
                    break
                sleep(0.02)
        finally:
            try:
                port.close()
            except Exception:
                pass
        hello = dict(core.hello)
        if hello:
            ok, note = protocol_check(hello, core.pc_limit)
            return DiscoveryResult(endpoint, True, hello.get("fw"), core.firmware_known, ok, note, hello,
                                   core.status is not None, probes, None, None)
        kind = ("NO_HELLO" if core.status is not None else "WRONG_ENDPOINT" if core.unknown_lines
                else "NOT_JOINED" if errors else "NO_REPLY")
        return DiscoveryResult(endpoint, False, None, None, None, "", {}, core.status is not None, probes,
                               FAILURES[kind][0], FAILURES[kind][1])

    # ------------------------------------------------------------ health

    def health(self, now: float | None = None) -> LinkHealth:
        now = self._clock() if now is None else now
        core = self._core
        missing: dict[str, str] = {}
        heartbeat = self.heartbeat_age(now)
        if heartbeat is None:
            missing["heartbeat_age_s"] = "no ST line has been received"
        last_rx = None if self._last_rx_at is None else max(0.0, now - self._last_rx_at)
        if last_rx is None:
            missing["last_rx_age_s"] = "nothing has been received from the robot yet"
        if self._last_success_unix is None:
            missing["last_success_unix"] = "no ST line has been received"
        latency = None
        if self._latencies:
            latency = 1000.0 * statistics.median(self._latencies)
        else:
            missing["latency_ms"] = ("not connected" if core is None else
                                     "no status request has been answered with a HELLO line yet")
        peer = (self._net.get("peer") or "")
        local_ip = peer.rsplit(":", 1)[0] if peer and peer != "NONE" else None
        if local_ip is None:
            missing["local_ip"] = "the robot's NET line has not reported this laptop's address yet"
        hello = core.hello if core is not None else {}
        ok, note = protocol_check(hello, None if core is None else core.pc_limit)
        if ok is None:
            missing["protocol_ok"] = note
        loss: LossInfo | None = None
        if core is None:
            missing["loss"] = "not connected: nothing to measure"
        else:
            rx, bad, old, other = (_int(self._net.get(key)) for key in ("rx", "bad", "old", "other"))
            ratio = self._firmware_loss_ratio()
            if rx is None:
                missing["loss.firmware"] = "the robot has not sent a NET line yet"
            elif ratio is None:
                missing["loss.firmware_loss_ratio"] = f"fewer than {LOSS_MIN_DATAGRAMS} datagrams in the window: no ratio"
            loss = LossInfo(rx, bad, old, other, ratio, self._gap_total, len(self._gaps), self._send_errors,
                            self._probe_timeouts, self._loss_high,
                            "firmware counters are cumulative since its boot; the ratio is over the last "
                            f"{LOSS_WINDOW_S:.0f} s; PC gaps are ST heartbeats later than 0.5 s")
        return LinkHealth(
            state=self.state, endpoint=self.endpoint, real=self.real, local_ip=local_ip, peer_ip=self.endpoint.host,
            last_rx_age_s=last_rx, last_success_unix=self._last_success_unix, latency_ms=latency,
            latency_samples=len(self._latencies), heartbeat_age_s=heartbeat, loss=loss,
            firmware_id=hello.get("fw") if hello else None, protocol_ok=ok, protocol_note=note, reconnects=self.reconnects,
            restarts=self.restarts, error=self.error, error_kind=self.error_kind, hint=self.hint,
            needs_rearm=self._needs_rearm, latched_reason=self.latched_reason, unavailable=missing)
