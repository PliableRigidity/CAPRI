"""Robot telemetry on the PC: one owner of the robot link, a local relay, and the IMU as the policy needs it.

Nothing in this file can command the robot. It does three things:

1. ``TelemetryRelay``: the ONE program that owns the link to the robot (``scripts/telemetry_link.py``)
   republishes every line it receives on 127.0.0.1 so other local programs can listen.
2. ``TapSubscriber``: how those programs listen. The only thing a subscriber ever sends is the word
   "SUB" to the relay on this PC.
3. ``LiveTelemetry``: turns the controller's IMU lines into what the observation contract expects
   (body frame, rad/s and m/s^2), or says exactly why it cannot. It never invents a sample.

Relay protocol (UDP, 127.0.0.1:4211 by default):
    subscriber -> relay   "SUB"                                 repeat at least every few seconds
    relay -> subscriber   "TAP <pc unix time> <RX|TX> <line>"   RX = line from the robot,
                                                                 TX = command the owner sent to it

IMU lines of ELRS-Handoff-Telemetry.ino:
    IMU seq=<n> t_us=<device microseconds> ok=1 ax= ay= az= gx= gy= gz= unit=m/s2,deg/s    (sensor axes)
    IMU seq=<n> t_us=<..> ok=0 why=NOT_FOUND|CONFIG_FAILED|READ_FAILED                      (no numbers)

Clocks: ``t_us`` is the controller's own clock; freshness is judged on PC receipt time. The two are
never treated as synchronised. The ratio between them is reported for information only.
"""

from __future__ import annotations

import math
import socket
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from control_center.backend.controller_link import ControllerStatus, parse_status_line
from control_center.tools.stream_adapter import ImuSample

REPO_ROOT = Path(__file__).resolve().parents[2]
RELAY_HOST = "127.0.0.1"                # the relay never listens on, or sends to, any other address
RELAY_PORT = 4211
SUBSCRIBER_TIMEOUT_S = 5.0
DEFAULT_MOUNTING = REPO_ROOT / "config" / "imu_mounting.yaml"
EXPECTED_UNIT = "m/s2,deg/s"
GRAVITY = 9.80665
STATUS_STALE_S = 0.6
LINK_STALE_S = 1.0
IMU_KEYS = ("ax", "ay", "az", "gx", "gy", "gz")


# ---------------------------------------------------------------- relay (owner side) and subscriber


class TelemetryRelay:
    """Owner side. Subscribers can only listen: whatever else they send is ignored."""

    def __init__(self, port: int = RELAY_PORT) -> None:
        self.port = port
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._socket.setblocking(False)
        self._socket.bind((RELAY_HOST, port))
        self.subscribers: dict[tuple[str, int], float] = {}
        self.published = 0

    def service(self, now: float) -> None:
        for _ in range(32):
            try:
                data, sender = self._socket.recvfrom(64)
            except (BlockingIOError, InterruptedError):
                break
            except OSError:                         # Windows reports a subscriber that has gone away
                continue
            if data.strip() == b"SUB" and sender[0] == RELAY_HOST:
                self.subscribers[sender] = now
        for sender in [key for key, seen in self.subscribers.items() if now - seen > SUBSCRIBER_TIMEOUT_S]:
            del self.subscribers[sender]

    def publish(self, direction: str, lines: list[str]) -> None:
        if not self.subscribers or not lines:
            return
        stamp = f"{time.time():.4f}"
        for text in lines:
            packet = f"TAP {stamp} {direction} {text}\n".encode("ascii", errors="replace")
            for sender in list(self.subscribers):
                try:
                    self._socket.sendto(packet, sender)
                except OSError:
                    pass
            self.published += 1

    def close(self) -> None:
        self._socket.close()


class TapSubscriber:
    """Listener side. Sends "SUB" to the relay on this PC once a second and reads what it republishes."""

    def __init__(self, port: int = RELAY_PORT) -> None:
        self.relay = (RELAY_HOST, port)
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._socket.setblocking(False)
        self._socket.bind((RELAY_HOST, 0))
        self._asked = float("-inf")
        self.received = 0

    def poll(self, now: float) -> list[tuple[str, str, float]]:
        """[(direction, line, owner's unix receipt time), ...]. Never blocks."""
        if now - self._asked >= 1.0:
            self._asked = now
            try:
                self._socket.sendto(b"SUB\n", self.relay)
            except OSError:
                pass
        out: list[tuple[str, str, float]] = []
        for _ in range(512):
            try:
                data, sender = self._socket.recvfrom(1024)
            except (BlockingIOError, InterruptedError):
                break
            except OSError:                         # no relay listening yet
                break
            if sender != self.relay:
                continue
            parts = data.decode("ascii", errors="replace").rstrip("\r\n").split(" ", 3)
            if len(parts) == 4 and parts[0] == "TAP" and parts[2] in ("RX", "TX"):
                try:
                    out.append((parts[2], parts[3], float(parts[1])))
                except ValueError:
                    continue
        self.received += len(out)
        return out

    def close(self) -> None:
        self._socket.close()


# ---------------------------------------------------------------- IMU lines


@dataclass(frozen=True)
class ImuReading:
    """One IMU line exactly as the controller sent it. ``accel`` and ``gyro`` are None when not valid."""

    sequence: int
    device_us: int
    valid: bool
    why: str = ""
    accel_m_s2: tuple[float, float, float] | None = None      # SENSOR axes
    gyro_deg_s: tuple[float, float, float] | None = None      # SENSOR axes


def parse_imu(line: str) -> ImuReading | None:
    """Strict. A valid sample has all six finite numbers and the expected units; a failure has none."""
    parts = line.split(" ")
    if not parts or parts[0] != "IMU":
        return None
    fields: dict[str, str] = {}
    for item in parts[1:]:
        key, separator, value = item.partition("=")
        if not separator or not key or key in fields:
            return None
        fields[key] = value
    try:
        sequence, device_us = int(fields["seq"]), int(fields["t_us"])
    except (KeyError, ValueError):
        return None
    if sequence < 0 or device_us < 0 or fields.get("ok") not in ("0", "1"):
        return None
    numbers = [key for key in IMU_KEYS if key in fields]
    if fields["ok"] == "0":
        if numbers or not fields.get("why") or set(fields) != {"seq", "t_us", "ok", "why"}:
            return None                                       # a failure that carries numbers is not trusted
        return ImuReading(sequence, device_us, False, fields["why"])
    if len(numbers) != 6 or fields.get("unit") != EXPECTED_UNIT or set(fields) != {"seq", "t_us", "ok", "unit", *IMU_KEYS}:
        return None
    try:
        values = [float(fields[key]) for key in IMU_KEYS]
    except ValueError:
        return None
    if not all(math.isfinite(value) for value in values):
        return None
    return ImuReading(sequence, device_us, True, "", (values[0], values[1], values[2]), (values[3], values[4], values[5]))


# ---------------------------------------------------------------- mounting: sensor axes -> body axes


@dataclass
class ImuMounting:
    """body = matrix @ sensor. Body frame of the checkpoint: x toward the nose, y to the robot's left, z up."""

    matrix: np.ndarray = field(default_factory=lambda: np.eye(3))
    verified: bool = False
    path: Path | None = None
    note: str = ""

    def to_body(self, vector: Any) -> np.ndarray:
        return self.matrix @ np.asarray(vector, float)


def load_mounting(path: Path | str | None = None) -> ImuMounting:
    """Read config/imu_mounting.yaml. Only right-handed axis remaps are accepted (no skew, no mirror)."""
    file = Path(path) if path else DEFAULT_MOUNTING
    data = (yaml.safe_load(file.read_text(encoding="utf-8")) or {}).get("imu") or {}
    matrix = np.asarray(data.get("sensor_to_body", np.eye(3)), float)
    if matrix.shape != (3, 3):
        raise ValueError(f"{file}: imu.sensor_to_body must be 3 rows of 3 numbers")
    if not np.allclose(matrix @ matrix.T, np.eye(3), atol=1e-6) or not math.isclose(float(np.linalg.det(matrix)), 1.0, abs_tol=1e-6):
        raise ValueError(f"{file}: imu.sensor_to_body must be a rotation (orthonormal, determinant +1); "
                         "a mirrored or skewed matrix would silently flip a turn direction")
    return ImuMounting(matrix, bool(data.get("verified", False)), file, str(data.get("note", "")))


def describe_motion(gyro_rad_s: Any, accel_m_s2: Any) -> list[str]:
    """Plain words for what body-frame readings mean, for the stationary and hand-rotation checks."""
    gyro, accel = np.asarray(gyro_rad_s, float), np.asarray(accel_m_s2, float)
    lines = []
    rate = float(np.max(np.abs(gyro)))
    magnitude = float(np.linalg.norm(accel))
    if rate < math.radians(5.0):
        if abs(magnitude - GRAVITY) > 1.5:
            lines.append(f"STILL, but |accel| = {magnitude:.2f} m/s^2 (expected about 9.81): scale or sensor problem")
        elif np.linalg.norm(accel - np.array([0.0, 0.0, GRAVITY])) <= 1.5:
            lines.append("STILL AND FLAT: gravity is on body +z, as the checkpoint expects for an upright robot")
        else:
            axis = int(np.argmax(np.abs(accel)))
            sign = "+" if accel[axis] > 0 else "-"
            meaning = {("x", "+"): "nose pointing UP", ("x", "-"): "nose pointing DOWN", ("y", "+"): "LEFT side up",
                       ("y", "-"): "RIGHT side up", ("z", "-"): "UPSIDE DOWN"}.get(("xyz"[axis], sign), "tilted")
            lines.append(f"STILL: gravity is mostly on body {sign}{'xyz'[axis]} -> reads as {meaning}. "
                         "If the robot is flat and upright, the mounting transform is wrong")
    else:
        axis = int(np.argmax(np.abs(gyro)))
        sign = "+" if gyro[axis] > 0 else "-"
        meaning = {("z", "+"): "turning LEFT (counter-clockwise seen from above)", ("z", "-"): "turning RIGHT (clockwise seen from above)",
                   ("y", "+"): "nose pitching DOWN", ("y", "-"): "nose pitching UP",
                   ("x", "+"): "rolling to the RIGHT (left side rising)", ("x", "-"): "rolling to the LEFT (right side rising)"}[("xyz"[axis], sign)]
        lines.append(f"ROTATING {math.degrees(rate):.0f} deg/s about body {sign}{'xyz'[axis]} -> reads as {meaning}")
    return lines


# ---------------------------------------------------------------- the IMU as an input to the observation


class LiveTelemetry:
    """A ``TelemetrySource`` for ``ShadowAssembler`` fed from the relay. Pure: time comes in as an argument.

    ``latest(t)`` returns a body-frame sample only when one really exists; otherwise None, with the
    reason in ``description``. The assembler then judges its age (PC receipt time).
    """

    def __init__(self, mounting: ImuMounting, source_label: str = "relay 127.0.0.1:4211") -> None:
        self.mounting = mounting
        self.source_label = source_label
        self.description = f"nothing received from the {source_label} yet"
        self.last_line_at: float | None = None
        self.reading: ImuReading | None = None              # newest valid reading (sensor axes)
        self.reading_at: float | None = None                # PC receipt time of it
        self.failure: ImuReading | None = None              # newest failure line
        self.failure_at: float | None = None
        self.status: ControllerStatus | None = None
        self.status_at: float | None = None
        self.firmware: str | None = None
        self.imu_info: dict[str, str] = {}
        self.valid_lines = self.failure_lines = self.malformed = self.out_of_order = self.gaps = self.restarts = 0
        self._last_sequence: int | None = None
        self._last_device_us: int | None = None
        self._behind_run, self._behind_sequence = 0, -2
        self._first: tuple[int, float] | None = None        # (device_us, PC time) of the first sample of this run

    # ---- input ----

    def feed(self, direction: str, line: str, now: float) -> None:
        if direction != "RX":
            return                                          # a command the owner SENT says nothing about the robot
        self.last_line_at = now
        if line.startswith("IMU "):
            reading = parse_imu(line)
            if reading is None:
                self.malformed += 1
                return
            if self._last_sequence is not None and reading.sequence <= self._last_sequence:
                # Late or repeated: never used. Five in a row that count up by one are not stragglers,
                # they are a new stream: the controller has rebooted and started counting again.
                self._behind_run = self._behind_run + 1 if reading.sequence == self._behind_sequence + 1 else 1
                self._behind_sequence = reading.sequence
                self.out_of_order += 1
                if self._behind_run < 5:
                    return
                self.restarts += 1
                self._first = None
            elif self._last_sequence is not None and reading.sequence > self._last_sequence + 1:
                self.gaps += 1
            self._behind_run, self._behind_sequence = 0, -2
            self._last_sequence, self._last_device_us = reading.sequence, reading.device_us
            if reading.valid:
                self.valid_lines += 1
                self.reading, self.reading_at = reading, now
                if self._first is None:
                    self._first = (reading.device_us, now)
            else:
                self.failure_lines += 1
                self.failure, self.failure_at = reading, now
            return
        status = parse_status_line(line)
        if status is not None:
            self.status, self.status_at = status, now
        elif line.startswith("HELLO "):
            for item in line.split(" ")[1:]:
                if item.startswith("fw="):
                    self.firmware = item[3:]
        elif line.startswith("IMUINFO "):
            self.imu_info = dict(item.split("=", 1) for item in line.split(" ")[1:] if "=" in item)

    # ---- TelemetrySource ----

    def latest(self, t: float) -> ImuSample | None:
        if self.last_line_at is None:
            self.description = f"nothing received from the {self.source_label} (is scripts\\telemetry_link.py running?)"
            return None
        if t - self.last_line_at > LINK_STALE_S:
            self.description = f"telemetry link silent for {t - self.last_line_at:.1f} s"
            return None
        if self.failure_at is not None and (self.reading_at is None or self.failure_at >= self.reading_at):
            self.description = f"the controller reports the IMU as {self.failure.why}; it sent no values"
            return None
        if self.reading is None:
            self.description = "status is arriving but no IMU line: this firmware does not send IMU telemetry"
            return None
        if not self.mounting.verified:
            self.description = ("IMU mounting transform is not verified (imu.verified is false in "
                                f"{self.mounting.path.name if self.mounting.path else 'the mounting config'}): do the axis check first")
            return None
        self.description = f"LSM6DS3 over the {self.source_label}, seq {self.reading.sequence}"
        gyro = self.mounting.to_body(np.radians(self.reading.gyro_deg_s))
        accel = self.mounting.to_body(self.reading.accel_m_s2)
        return ImuSample(self.reading_at, tuple(float(v) for v in gyro), tuple(float(v) for v in accel))

    # ---- for the screen and the axis check ----

    def body_values(self) -> tuple[np.ndarray, np.ndarray] | None:
        """(gyro rad/s, accel m/s^2) of the newest valid reading in body axes, verified or not."""
        if self.reading is None:
            return None
        return self.mounting.to_body(np.radians(self.reading.gyro_deg_s)), self.mounting.to_body(self.reading.accel_m_s2)

    def clock_ratio(self) -> float | None:
        """Controller time elapsed / PC time elapsed since the first sample. Information only."""
        if self._first is None or self.reading is None or self.reading_at is None or self.reading_at - self._first[1] < 5.0:
            return None
        return (self.reading.device_us - self._first[0]) / 1e6 / (self.reading_at - self._first[1])

    def controller_line(self, now: float) -> tuple[str, str]:
        """(text, level): requested mode, confirmed source, arm state, stop reason, applied outputs."""
        if self.status is None or self.status_at is None:
            return "CONTROLLER no status line received", "note"
        age = now - self.status_at
        if age > STATUS_STALE_S:
            return f"CONTROLLER status STALE ({age:.1f} s old): source not confirmed", "bad"
        s = self.status
        return (f"CONTROLLER requested {s.requested}, confirmed source {s.active}, {'ARMED' if s.armed else 'disarmed'}, "
                f"last stop {s.stop}, applying L={s.left} R={s.right}", "note")

    def lines(self, now: float) -> list[tuple[str, str]]:
        out = [self.controller_line(now)]
        if self.reading is not None and self.reading_at is not None:
            a, g = self.reading.accel_m_s2, self.reading.gyro_deg_s
            out.append((f"IMU sensor axes  accel {a[0]:+.2f} {a[1]:+.2f} {a[2]:+.2f} m/s^2   gyro {g[0]:+.1f} {g[1]:+.1f} {g[2]:+.1f} deg/s   "
                        f"seq {self.reading.sequence}, {1000 * (now - self.reading_at):.0f} ms ago (PC receipt time)", "note"))
            gyro, accel = self.body_values()
            out.append((f"IMU body axes    accel {accel[0]:+.2f} {accel[1]:+.2f} {accel[2]:+.2f} m/s^2   gyro {gyro[0]:+.2f} {gyro[1]:+.2f} {gyro[2]:+.2f} rad/s   "
                        f"mounting {'VERIFIED' if self.mounting.verified else 'NOT VERIFIED'}", "ok" if self.mounting.verified else "bad"))
        ratio = self.clock_ratio()
        counts = f"IMU lines: {self.valid_lines} valid, {self.failure_lines} failed, {self.out_of_order} late/repeated dropped, {self.gaps} gaps"
        if ratio is not None:
            counts += f"; controller clock runs at {ratio:.4f} x PC clock (not synchronised)"
        out.append((counts, "note"))
        return out
