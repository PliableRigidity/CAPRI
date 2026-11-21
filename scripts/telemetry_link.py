"""The one PC program that owns the link to the robot when telemetry is wanted.

The robot reports to a single address, so only one program can talk to it. This one does, and
republishes every line it receives (status, IMU) on 127.0.0.1:4211 for local listeners: the live
shadow runner and the Command Centre recorder. Listeners cannot send anything to the robot.

    python scripts/telemetry_link.py --wifi               receive only: sends "?" once a second, never a drive command
    python scripts/telemetry_link.py --wifi --drive       the same link, plus the keyboard drive window
    python scripts/telemetry_link.py --port COM5          the same over USB

It reuses scripts/handoff_bench_client.py unchanged: its Wi-Fi/USB link code and, with --drive, its
drive window. Needs the ELRS-Handoff-Telemetry sketch for IMU lines; with the handoff-only sketches
it relays status only and says that no IMU line is arriving.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import handoff_bench_client as client  # noqa: E402
from control_center.tools.live_telemetry import RELAY_PORT, LiveTelemetry, ImuMounting, TelemetryRelay  # noqa: E402


class RelayingPort:
    """Wraps the client's port: everything read is also republished; drive commands sent are republished as TX."""

    def __init__(self, port, relay: TelemetryRelay, on_lines=None) -> None:
        self._port, self._relay, self._on_lines = port, relay, on_lines
        self._buffer = bytearray()
        self._last_command: bytes | None = None
        self.probe = getattr(port, "probe", False)
        self.label = getattr(port, "label", "USB")

    @property
    def error(self):
        return getattr(self._port, "error", None)

    def read(self, size: int = 4096) -> bytes:
        data = self._port.read(size)
        now = time.monotonic()
        self._relay.service(now)
        if data:
            self._buffer += data
            *complete, rest = bytes(self._buffer).split(b"\n")
            self._buffer = bytearray(rest[-400:])
            lines = [line.decode("ascii", errors="replace").rstrip("\r") for line in complete if line.strip()]
            self._relay.publish("RX", lines)
            if self._on_lines:
                self._on_lines(lines, now)
        return data

    def write(self, data: bytes):
        result = self._port.write(data)
        if data.startswith(b"D,") and data != self._last_command:      # every CHANGE of command, marked as sent
            self._last_command = data
            self._relay.publish("TX", [data.decode("ascii").strip()])
        return result

    def close(self) -> None:
        self._port.close()


def receive_only(port: RelayingPort, view: LiveTelemetry, relay: TelemetryRelay) -> int:
    print('Receiving only (Ctrl+C to stop). The only thing sent to the robot is "?" once a second.')
    asked = reported = float("-inf")
    try:
        while True:
            now = time.monotonic()
            if port.probe and now - asked >= 1.0:
                asked = now
                port.write(b"?\n")
            port.read()
            if now - reported >= 1.0:
                reported = now
                view.latest(now)
                text, _ = view.controller_line(now)
                print(f"{time.strftime('%H:%M:%S')}  fw={view.firmware or '?'}  {text}")
                if view.reading is not None and view.reading_at is not None and now - view.reading_at < 0.5 and \
                        (view.failure_at is None or view.failure_at < view.reading_at):
                    a, g = view.reading.accel_m_s2, view.reading.gyro_deg_s
                    print(f"          IMU ok  seq {view.reading.sequence}  t_us {view.reading.device_us}  "
                          f"accel {a[0]:+.3f} {a[1]:+.3f} {a[2]:+.3f} m/s^2  gyro {g[0]:+.2f} {g[1]:+.2f} {g[2]:+.2f} deg/s  (sensor axes)")
                elif view.failure is not None and view.failure_at is not None and now - view.failure_at < 1.5:
                    print(f"          IMU NOT VALID: the controller reports {view.failure.why} (no values were sent)")
                elif view.last_line_at is None or now - view.last_line_at > 1.0:
                    print(f"          no line from the robot ({port.error or 'is the laptop joined to its Wi-Fi?'})")
                else:
                    print("          no IMU line: this firmware does not send IMU telemetry")
                print(f"          {view.valid_lines} valid, {view.failure_lines} failed, {view.out_of_order} late/repeated, {view.gaps} gaps"
                      f"  |  relay 127.0.0.1:{relay.port}: {len(relay.subscribers)} listener(s)")
            time.sleep(0.005)
    except KeyboardInterrupt:
        return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Own the robot link and relay its telemetry to local listeners.")
    parser.add_argument("--wifi", nargs="?", const=client.WIFI_HOST, metavar="ROBOT_IP", help=f"Wi-Fi link to the robot (default {client.WIFI_HOST})")
    parser.add_argument("--udp-port", type=int, default=client.WIFI_UDP_PORT)
    parser.add_argument("--port", help="USB serial port instead, e.g. COM5")
    parser.add_argument("--relay-port", type=int, default=RELAY_PORT, help=f"local relay port on 127.0.0.1 (default {RELAY_PORT})")
    parser.add_argument("--drive", action="store_true", help="also open the keyboard drive window on this same link")
    parser.add_argument("--speed", type=int, default=client.DEFAULT_SPEED)
    parser.add_argument("--turn", type=int, default=client.DEFAULT_TURN)
    parser.add_argument("--log", type=Path, help="append every received line and every change of command to this file")
    args = parser.parse_args(argv)
    if bool(args.port) == bool(args.wifi):
        parser.error("give exactly one of --wifi or --port COMx")
    try:
        relay = TelemetryRelay(args.relay_port)
    except OSError as error:
        print(f"Could not open the relay on 127.0.0.1:{args.relay_port}: {error} (is another telemetry_link already running?)", file=sys.stderr)
        return 1
    try:
        raw = client.UdpPort(args.wifi, args.udp_port) if args.wifi else client.open_port(args.port)
    except Exception as error:
        relay.close()
        print(f"Could not open {args.wifi or args.port}: {error}", file=sys.stderr)
        return 1
    view = LiveTelemetry(ImuMounting(), "robot link")           # display only here; the mounting is applied by the listeners
    log = client.Log(args.log)

    def on_lines(lines: list[str], now: float) -> None:
        log.received(lines)
        for line in lines:
            view.feed("RX", line, now)

    port = RelayingPort(raw, relay, on_lines)
    try:
        if not args.drive:
            return receive_only(port, view, relay)
        core = client.ClientCore(args.speed, args.turn, sequenced=bool(args.wifi),
                                 link_label=(f"Wi-Fi {args.wifi}:{args.udp_port} (UDP)" if args.wifi else f"USB {args.port}") + ", relayed")
        return client.drive_window(port, port.label, core, client.Log(None))
    finally:
        log.close()
        port.close()
        relay.close()


if __name__ == "__main__":
    raise SystemExit(main())
