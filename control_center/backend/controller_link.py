"""Listen-only view of the on-robot controller's status lines.

The controller (ESP32-C3) decides who has authority and enforces every stop. This module only
reads what it reports. Nothing here writes to the robot: there is no write call in this file, and
the serial source opens the port without ever sending a byte.

Lines understood (the protocol of ``ELRS-Handoff-Bench/README.md``):

    ST t=.. tok=.. link=.. armed=.. req=.. act=.. pc_ok=.. block=.. fault=.. stop=.. L=.. R=.. pc_age=..
    CH ...raw radio channels...
    HELLO fw=.. arm=.. pc=.. pc_limit=..

plus one line that the bench sketch does NOT send (it has no IMU); it is the raw IMU line of the
earlier merged prototype and is parsed so that the stream is recorded if a firmware ever sends it:

    IMU,<device_ms>,<ax>,<ay>,<az>,<gx>,<gy>,<gz>      accel in m/s^2, gyro in deg/s, sensor axes
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

STATUS_KEYS = ("t", "tok", "link", "armed", "req", "act", "pc_ok", "block", "fault", "stop", "L", "R", "pc_age")
SOURCES = ("NONE", "RADIO", "PC")
STALE_AFTER_S = 0.6          # the controller sends ST at least every 0.2 s
IMU_STALE_AFTER_S = 0.5
MAX_LINE = 300


@dataclass(frozen=True)
class ControllerStatus:
    """One ST line."""

    token: int
    link: bool
    armed: bool
    requested: str      # what the RadioMaster MODE switch asks for
    active: str         # the source whose commands reach the motors now
    pc_ok: bool
    block: str
    fault: str
    stop: str
    left: int
    right: int
    pc_age_ms: int
    device_ms: int


@dataclass(frozen=True)
class ImuSample:
    """Raw six-axis sample as sent by the controller. Sensor axes, no orientation applied."""

    device_ms: int
    accel_m_s2: tuple[float, float, float]
    gyro_deg_s: tuple[float, float, float]
    sequence: int | None = None         # telemetry sketch only: the controller's sample counter
    device_us: int | None = None        # telemetry sketch only: controller microseconds since boot


@dataclass(frozen=True)
class PcCommand:
    """A drive command line a PC client SENT. Sent is not the same as applied."""

    token: int
    throttle: int
    steer: int


def _fields(line: str) -> tuple[str, dict[str, str]] | None:
    parts = line.split(" ")
    if not parts or not parts[0].isalpha():
        return None
    out: dict[str, str] = {}
    for item in parts[1:]:
        key, separator, value = item.partition("=")
        if not separator or not key or key in out:
            return None
        out[key] = value
    return parts[0], out


def parse_status_line(line: str) -> ControllerStatus | None:
    """Strict: a missing, extra, reordered or malformed field means no status at all."""
    split = _fields(line)
    if split is None or split[0] != "ST" or tuple(split[1]) != STATUS_KEYS:
        return None
    f = split[1]
    try:
        token, left, right = int(f["tok"]), int(f["L"]), int(f["R"])
        pc_age, device_ms = int(f["pc_age"]), int(f["t"])
    except ValueError:
        return None
    if any(f[key] not in ("0", "1") for key in ("link", "armed", "pc_ok")):
        return None
    if f["req"] not in SOURCES or f["act"] not in SOURCES:
        return None
    if not 0 <= token <= 0xFFFFFFFF or abs(left) > 255 or abs(right) > 255:
        return None
    return ControllerStatus(token, f["link"] == "1", f["armed"] == "1", f["req"], f["act"], f["pc_ok"] == "1",
                            f["block"], f["fault"], f["stop"], left, right, pc_age, device_ms)


def parse_imu_line(line: str) -> ImuSample | None:
    """A VALID IMU sample in either format, else None. A failure line (ok=0) is not a sample: see parse_imu_failure."""
    if line.startswith("IMU "):         # ELRS-Handoff-Telemetry: IMU seq= t_us= ok=1 ax= ay= az= gx= gy= gz= unit=m/s2,deg/s
        from control_center.tools.live_telemetry import parse_imu

        reading = parse_imu(line)
        if reading is None or not reading.valid:
            return None
        return ImuSample(reading.device_us // 1000, reading.accel_m_s2, reading.gyro_deg_s, reading.sequence, reading.device_us)
    parts = line.split(",")
    if len(parts) != 8 or parts[0] != "IMU":
        return None
    try:
        device_ms = int(parts[1])
        values = [float(v) for v in parts[2:]]
    except ValueError:
        return None
    if device_ms < 0 or not all(math.isfinite(v) for v in values):
        return None
    return ImuSample(device_ms, (values[0], values[1], values[2]), (values[3], values[4], values[5]))


def parse_imu_failure(line: str) -> str | None:
    """The reason the controller gives when it reports the IMU as not valid (no values are sent with it)."""
    if not line.startswith("IMU "):
        return None
    from control_center.tools.live_telemetry import parse_imu

    reading = parse_imu(line)
    return reading.why if reading is not None and not reading.valid else None


def parse_pc_command(line: str) -> PcCommand | None:
    parts = line.split(",")
    if len(parts) != 4 or parts[0] != "D":
        return None
    try:
        token, throttle, steer = int(parts[1]), int(parts[2]), int(parts[3])
    except ValueError:
        return None
    if not 0 <= token <= 0xFFFFFFFF or abs(throttle) > 255 or abs(steer) > 255:
        return None
    return PcCommand(token, throttle, steer)


def parse_keyed_line(line: str, kind: str) -> dict[str, str] | None:
    split = _fields(line)
    return split[1] if split is not None and split[0] == kind else None


# ---------------------------------------------------------------- what the dashboard may claim


@dataclass(frozen=True)
class Authority:
    """Who has control, as far as fresh controller telemetry says. Never derived from a button."""

    freshness: str                  # "FRESH", "STALE" or "UNKNOWN" (never heard from the controller)
    source: str                     # "RADIO", "PC", "NONE", or the freshness word when not fresh
    requested: str                  # same vocabulary
    armed: bool | None              # None when not fresh
    age_s: float | None
    status: ControllerStatus | None = None   # the last status seen, even when stale (for reference only)

    @property
    def fresh(self) -> bool:
        return self.freshness == "FRESH"


class ControllerMonitor:
    """Turns received lines into the current authority and a short list of meaningful changes.

    Pure Python: time comes in as an argument, so it can be tested without a clock or a port.
    """

    def __init__(self, stale_after_s: float = STALE_AFTER_S) -> None:
        self.stale_after_s = stale_after_s
        self.status: ControllerStatus | None = None
        self.status_at: float | None = None
        self.hello: dict[str, str] = {}
        self.channels: dict[str, str] = {}
        self.last_imu: ImuSample | None = None
        self.last_imu_at: float | None = None
        self.imu_problem: str | None = None         # the reason given while the controller reports the IMU as not valid
        self.imu_problem_at: float | None = None
        self.imu_count = 0
        self.last_pc_command: PcCommand | None = None
        self.last_pc_command_at: float | None = None
        self.unknown_lines = 0
        self.lines = 0
        self._was_fresh = False
        self._first_imu_at: float | None = None

    # ---- input ----

    def feed(self, line: str, now: float, direction: str = "RX") -> list[tuple[str, str, dict[str, Any]]]:
        """Take one line. Returns (kind, text, data) events for changes worth an operator's attention."""
        self.lines += 1
        events: list[tuple[str, str, dict[str, Any]]] = []
        if direction == "TX":
            command = parse_pc_command(line)
            if command is not None:
                self.last_pc_command, self.last_pc_command_at = command, now
            return events
        status = parse_status_line(line)
        if status is not None:
            previous = self.status if self._was_fresh else None
            self.status, self.status_at = status, now
            if not self._was_fresh:
                events.append(("controller_link", "Controller telemetry is live", {}))
                self._was_fresh = True
            events += self._changes(previous, status)
            return events
        problem = parse_imu_failure(line)
        if problem is not None:
            if problem != self.imu_problem or self.imu_problem_at is None or now - self.imu_problem_at > 2.0:
                events.append(("imu", f"Controller reports the IMU as {problem}: no values are being sent", {"why": problem}))
            self.imu_problem, self.imu_problem_at = problem, now
            return events
        imu = parse_imu_line(line)
        if imu is not None:
            self.imu_problem = None
            if self.last_imu_at is None or now - self.last_imu_at > IMU_STALE_AFTER_S:
                events.append(("imu", "IMU stream started", {}))
                self._first_imu_at = now
                self.imu_count = 0
            self.last_imu, self.last_imu_at = imu, now
            self.imu_count += 1
            return events
        hello = parse_keyed_line(line, "HELLO")
        if hello is not None:
            self.hello = hello
            return events
        channels = parse_keyed_line(line, "CH")
        if channels is not None:
            self.channels = channels
            return events
        self.unknown_lines += 1
        return events

    def poll(self, now: float) -> list[tuple[str, str, dict[str, Any]]]:
        """Call regularly: reports the moment telemetry goes stale."""
        if self._was_fresh and self.status_at is not None and now - self.status_at > self.stale_after_s:
            self._was_fresh = False
            return [("controller_link", "Controller telemetry STALE: control source is no longer confirmed",
                     {"last_source": self.status.active if self.status else None})]
        return []

    @staticmethod
    def _changes(old: ControllerStatus | None, new: ControllerStatus) -> list[tuple[str, str, dict[str, Any]]]:
        events: list[tuple[str, str, dict[str, Any]]] = []
        if old is None:
            events.append(("source_confirmed", f"Controller confirms source {new.active}"
                                               f" ({'armed' if new.armed else 'disarmed'}, requested {new.requested})",
                           {"source": new.active, "armed": new.armed, "requested": new.requested}))
            return events
        if new.requested != old.requested:
            events.append(("mode_requested", f"RadioMaster requests {new.requested}", {"requested": new.requested}))
        if new.armed != old.armed:
            if new.armed:
                events.append(("arm", "Controller reports ARMED", {}))
            else:
                events.append(("disarm", f"Controller reports DISARMED ({new.stop})", {"cause": new.stop}))
        if new.active != old.active:
            reason = "" if new.active != "NONE" else f" ({new.block})"
            events.append(("source_confirmed", f"Controller confirms source {new.active}{reason}",
                           {"source": new.active, "block": new.block, "stop": new.stop}))
        if new.fault != old.fault and new.fault != "NONE":
            events.append(("fault", f"Controller fault latched: {new.fault}", {"fault": new.fault}))
        return events

    # ---- output ----

    def authority(self, now: float) -> Authority:
        if self.status is None or self.status_at is None:
            return Authority("UNKNOWN", "UNKNOWN", "UNKNOWN", None, None)
        age = max(0.0, now - self.status_at)
        if age > self.stale_after_s:
            return Authority("STALE", "STALE", "STALE", None, age, self.status)
        return Authority("FRESH", self.status.active, self.status.requested, self.status.armed, age, self.status)

    def imu_present(self, now: float) -> bool:
        return self.last_imu_at is not None and now - self.last_imu_at <= IMU_STALE_AFTER_S

    def imu_rate_hz(self, now: float) -> float | None:
        if not self.imu_present(now) or self._first_imu_at is None or now - self._first_imu_at < 0.5:
            return None
        return self.imu_count / (now - self._first_imu_at)

    def firmware(self) -> str | None:
        return self.hello.get("fw")


# ---------------------------------------------------------------- where lines come from


class LineAssembler:
    """Bytes -> complete text lines, dropping over-long ones."""

    def __init__(self) -> None:
        self._buffer = bytearray()
        self._discard = False

    def feed(self, data: bytes) -> list[str]:
        lines = []
        for byte in data:
            if byte == 10:
                if not self._discard:
                    lines.append(self._buffer.decode("ascii", errors="replace").rstrip("\r"))
                self._buffer.clear()
                self._discard = False
            elif not self._discard:
                if len(self._buffer) >= MAX_LINE:
                    self._buffer.clear()
                    self._discard = True
                else:
                    self._buffer.append(byte)
        return lines


class SerialLineSource:
    """The controller's USB serial port, opened to LISTEN. This class has no way to send.

    Only one program can hold a COM port. While the PC drive client has it, use
    ``ClientLogSource`` on that client's ``--log`` file instead.
    """

    kind = "serial"

    def __init__(self, port_name: str, baud: int = 115200) -> None:
        import serial  # imported here so the dashboard runs without pyserial when no port is used

        self.label = f"{port_name} (listen-only)"
        self._lines = LineAssembler()
        port = serial.Serial()
        port.port = port_name
        port.baudrate = baud
        port.timeout = 0
        port.dtr = False     # do not pulse the ESP32-C3 reset lines when opening
        port.rts = False
        port.open()
        self._port = port

    def read(self) -> list[tuple[str, str]]:
        return [("RX", text) for text in self._lines.feed(self._port.read(4096))]

    def close(self) -> None:
        self._port.close()


class ClientLogSource:
    """Follows the file written by ``scripts/handoff_bench_client.py --log``.

    Each line there is ``<unix time> RX <line from the controller>`` or ``<unix time> TX <command
    the client sent>``. Reading starts at the end of the file: old lines are never replayed as live.
    """

    kind = "client-log"

    def __init__(self, path: Path | str, from_start: bool = False) -> None:
        self.path = Path(path)
        self.label = f"client log {self.path.name}"
        self._handle = self.path.open("r", encoding="utf-8", errors="replace")
        if not from_start:
            self._handle.seek(0, 2)
        self._partial = ""

    def read(self) -> list[tuple[str, str]]:
        chunk = self._handle.read()
        if not chunk:
            return []
        text = self._partial + chunk
        lines = text.split("\n")
        self._partial = lines.pop()
        out = []
        for line in lines:
            parts = line.rstrip("\r").split(" ", 2)
            if len(parts) == 3 and parts[1] in ("RX", "TX"):
                out.append((parts[1], parts[2]))
        return out

    def close(self) -> None:
        self._handle.close()


class RelaySource:
    """Listens to the local telemetry relay of ``scripts/telemetry_link.py`` (127.0.0.1 only).

    That script owns the link to the robot; this source only subscribes on this PC, so the dashboard
    can record status and IMU while the shadow runner listens to the same relay.
    """

    kind = "relay"

    def __init__(self, port: int | None = None) -> None:
        from control_center.tools.live_telemetry import RELAY_PORT, TapSubscriber

        self._tap = TapSubscriber(port or RELAY_PORT)
        self.label = f"telemetry relay 127.0.0.1:{port or RELAY_PORT} (listen-only)"

    def read(self) -> list[tuple[str, str]]:
        return [(direction, line) for direction, line, _ in self._tap.poll(time.monotonic())]

    def close(self) -> None:
        self._tap.close()


class ScriptedSource:
    """MOCK controller: replays a fixed script of (seconds from start, direction, line). Not hardware."""

    kind = "mock"

    def __init__(self, script: list[tuple[float, str, str]], label: str = "MOCK controller replay",
                 loop: bool = False, clock=time.monotonic) -> None:
        self.label = label
        self._script = sorted(script, key=lambda item: item[0])
        self._clock = clock
        self._start = clock()
        self._next = 0
        self._loop = loop

    def read(self) -> list[tuple[str, str]]:
        elapsed = self._clock() - self._start
        out = []
        while self._next < len(self._script) and self._script[self._next][0] <= elapsed:
            _, direction, line = self._script[self._next]
            out.append((direction, line))
            self._next += 1
        if self._loop and self._next >= len(self._script) and self._script:
            self._start, self._next = self._clock(), 0
        return out

    def close(self) -> None:
        pass


def status_line(device_ms: int, token: int = 1, link: int = 1, armed: int = 0, req: str = "RADIO", act: str = "NONE",
                pc_ok: int = 0, block: str = "DISARMED", fault: str = "NONE", stop: str = "NONE",
                left: int = 0, right: int = 0, pc_age: int = -1) -> str:
    """Build an ST line in the controller's exact format (for mocks and tests)."""
    return (f"ST t={device_ms} tok={token} link={link} armed={armed} req={req} act={act} pc_ok={pc_ok} "
            f"block={block} fault={fault} stop={stop} L={left} R={right} pc_age={pc_age}")


def demo_script(with_imu: bool = False) -> list[tuple[float, str, str]]:
    """A labelled mock sequence: disarmed, armed in RADIO, handoff to PC, telemetry gap, disarm."""
    script: list[tuple[float, str, str]] = [(0.0, "RX", "HELLO fw=MOCK-not-hardware arm=CH5:HIGH pc=CH6:HIGH pc_limit=128")]
    phases = [
        (0.0, 4.0, dict(armed=0, req="RADIO", act="NONE", block="DISARMED")),
        (4.0, 5.0, dict(armed=1, req="RADIO", act="NONE", block="WAIT_RADIO_NEUTRAL", token=2)),
        (5.0, 10.0, dict(armed=1, req="RADIO", act="RADIO", block="NONE", token=2, left=90, right=90)),
        (10.0, 11.0, dict(armed=1, req="PC", act="NONE", block="WAIT_PC_NEUTRAL", stop="HANDOFF", token=3)),
        (11.0, 16.0, dict(armed=1, req="PC", act="PC", pc_ok=1, block="NONE", stop="HANDOFF", token=3, left=60, right=60, pc_age=10)),
        # 16.0-18.0: no lines at all (telemetry gap -> STALE)
        (18.0, 24.0, dict(armed=0, req="PC", act="NONE", block="REARM_REQUIRED", fault="PC_STALE", stop="PC_STALE", token=4)),
    ]
    for start, end, fields in phases:
        t = start
        while t < end:
            script.append((t, "RX", status_line(int(t * 1000), **fields)))
            if fields.get("act") == "PC":
                script.append((t, "TX", f"D,{fields['token']},60,0"))
            t += 0.2
    if with_imu:
        t = 0.0
        while t < 24.0:
            script.append((t, "RX", f"IMU,{int(t * 1000)},0.1200,-0.0500,9.8070,0.100,-0.200,{12.0 * math.sin(t):.3f}"))
            t += 0.02
    return script
