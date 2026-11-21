"""Manual keyboard drive client for the ELRS-Handoff-Bench sketch (Windows, USB serial).

Bench tool only. The RadioMaster arms the robot and selects RADIO or PC. This
program can only send drive commands, and the firmware decides whether to use them.

    python scripts/handoff_bench_client.py --list
    python scripts/handoff_bench_client.py --port COM5 --monitor     (listen only, sends nothing)
    python scripts/handoff_bench_client.py --port COM5               (keyboard drive window, USB)
    python scripts/handoff_bench_client.py --wifi                    (keyboard drive window, Wi-Fi)

--wifi needs the ELRS-Handoff-WiFi sketch: join the robot's own Wi-Fi network from this laptop
first, then the robot is 192.168.4.1 and commands go as UDP datagrams to port 4210. The window,
the keys and the rules are the same on both; only one of USB / Wi-Fi is used per run.

Keys in the drive window: hold SPACE (deadman) and W/A/S/D or the arrow keys.
Releasing SPACE, releasing the direction keys, or clicking another window sends zero.
P stops sending altogether (to test the firmware's stale-command stop); P again resumes.

Serial protocol (text lines, "\\n" terminated; full description in ELRS-Handoff-Bench/README.md):

    firmware -> PC   ST t=.. tok=.. link=.. armed=.. req=.. act=.. pc_ok=.. block=.. fault=.. stop=.. L=.. R=.. pc_age=..
                     CH ...raw radio channels...
                     HELLO fw=autobot-handoff-bench-1 arm=CH5:HIGH pc=CH6:HIGH pc_limit=128 ...
    PC -> firmware   D,<tok>,<throttle>,<steer>      throttle and steer are -255..255
                     ?                               ask for the HELLO line
    over Wi-Fi       D,<tok>,<throttle>,<steer>,<n>  one datagram each; n counts up, so a late or
                                                     repeated datagram is never applied

A command is sent every 20 ms whether or not a key is held, always with the token from
the latest ST line, so the firmware can time out on its own if this program or USB dies.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path

FIRMWARE_ID = "autobot-handoff-bench-1"
FIRMWARE_IDS = {FIRMWARE_ID, "autobot-handoff-wifi-1"}
WIFI_HOST = "192.168.4.1"      # the robot, when the laptop has joined the robot's own network
WIFI_UDP_PORT = 4210
PROBE_PERIOD_S = 0.5           # Wi-Fi only: how often to ask for status while none is arriving
BAUD = 115200
SEND_PERIOD_S = 0.02     # 50 Hz
STATUS_FRESH_S = 0.5     # the firmware sends ST at least every 0.2 s
RELEASE_QUIET_S = 0.7    # longer than the slowest Windows key repeat (0.4 s)
DEFAULT_SPEED = 80       # of 255: about 31 %
DEFAULT_TURN = 70        # of 255
MAX_COMMAND = 255
MAX_LINE = 300

DEADMAN = "space"
FORWARD = {"w", "up"}
BACK = {"s", "down"}
LEFT = {"a", "left"}
RIGHT = {"d", "right"}

SOURCES = {"NONE", "RADIO", "PC"}
STATUS_KEYS = ("t", "tok", "link", "armed", "req", "act", "pc_ok", "block", "fault", "stop", "L", "R", "pc_age")

BLOCK_HELP = {
    "NONE": "",
    "NO_RADIO": "no radio link yet: switch the RadioMaster on",
    "RADIO_LOST": "radio link lost",
    "ARM_SWITCH_INVALID": "ARM channel is not clearly low or high",
    "DISARMED": "ARM switch is off",
    "ARM_ON_AT_BOOT": "ARM switch was already on: turn it off, then on",
    "REARM_REQUIRED": "stopped by a fault: turn ARM off, then on",
    "MODE_SWITCH_INVALID": "MODE channel is not clearly low or high",
    "WAIT_RADIO_NEUTRAL": "RADIO selected: centre both sticks",
    "WAIT_PC_NEUTRAL": "PC selected: release all keys (the firmware wants a zero command first)",
}


@dataclass(frozen=True)
class Status:
    """One ST line from the firmware."""

    token: int
    link: bool
    armed: bool
    requested: str
    active: str
    pc_ok: bool
    block: str
    fault: str
    stop: str
    left: int
    right: int
    pc_age_ms: int
    firmware_ms: int


def split_line(line: str) -> tuple[str, dict[str, str]] | None:
    """'KIND key=value key=value' -> (KIND, {key: value}); None if it is not that shape."""
    parts = line.split(" ")
    if not parts or not parts[0].isalpha():
        return None
    fields: dict[str, str] = {}
    for item in parts[1:]:
        key, separator, value = item.partition("=")
        if not separator or not key or key in fields:
            return None
        fields[key] = value
    return parts[0], fields


def parse_status(line: str) -> Status | None:
    """Strict: any missing, extra or malformed field means no status at all."""
    split = split_line(line)
    if split is None or split[0] != "ST":
        return None
    fields = split[1]
    if tuple(fields) != STATUS_KEYS:
        return None
    try:
        token = int(fields["tok"])
        left, right = int(fields["L"]), int(fields["R"])
        pc_age, firmware_ms = int(fields["pc_age"]), int(fields["t"])
    except ValueError:
        return None
    flags = (fields["link"], fields["armed"], fields["pc_ok"])
    if any(flag not in {"0", "1"} for flag in flags):
        return None
    if fields["req"] not in SOURCES or fields["act"] not in SOURCES:
        return None
    if not 0 <= token <= 0xFFFFFFFF or abs(left) > 255 or abs(right) > 255:
        return None
    return Status(
        token=token, link=flags[0] == "1", armed=flags[1] == "1", requested=fields["req"], active=fields["act"],
        pc_ok=flags[2] == "1", block=fields["block"], fault=fields["fault"], stop=fields["stop"],
        left=left, right=right, pc_age_ms=pc_age, firmware_ms=firmware_ms,
    )


def drive_command(keys: set[str], speed: int, turn: int) -> tuple[int, int]:
    """Keys held -> (throttle, steer). Zero unless the deadman key is held."""
    if DEADMAN not in keys:
        return 0, 0
    throttle = speed * (bool(keys & FORWARD) - bool(keys & BACK))
    steer = turn * (bool(keys & RIGHT) - bool(keys & LEFT))
    return throttle, steer


def format_command(token: int, throttle: int, steer: int, sequence: int | None = None) -> bytes:
    """USB: D,tok,throttle,steer. Wi-Fi adds a sequence number that counts up."""
    if not (0 <= token <= 0xFFFFFFFF and abs(throttle) <= MAX_COMMAND and abs(steer) <= MAX_COMMAND):
        raise ValueError("command out of range")
    if sequence is None:
        return f"D,{token},{throttle},{steer}\n".encode("ascii")
    if not 0 < sequence <= 0xFFFFFFFF:
        raise ValueError("sequence out of range")
    return f"D,{token},{throttle},{steer},{sequence}\n".encode("ascii")


class ClientCore:
    """Everything except the window and the serial port, so it can be tested without either."""

    def __init__(self, speed: int = DEFAULT_SPEED, turn: int = DEFAULT_TURN, sequenced: bool = False,
                 link_label: str = "USB") -> None:
        if not (0 <= speed <= MAX_COMMAND and 0 <= turn <= MAX_COMMAND):
            raise ValueError("speed and turn must be 0..255")
        self.speed = speed
        self.turn = turn
        self.sequenced = sequenced      # Wi-Fi: every command carries a number that counts up
        self.sequence = 0
        self.link_label = link_label
        self.link_error: str | None = None      # Wi-Fi: why the last datagram could not be sent
        self.net: dict[str, str] = {}           # the firmware's NET line (Wi-Fi sketch only)
        self._probed_at = float("-inf")
        self.keys: set[str] = set()
        self.focused = False
        # After the window regains focus (or status returns) the keyboard is not trusted
        # until the deadman key is seen released: a real key-up, or no key-down from it
        # for RELEASE_QUIET_S (a held key keeps repeating, a released one goes quiet).
        self.release_required = True
        self._latched_at = float("-inf")
        self.status: Status | None = None
        self.status_at = float("-inf")
        self.hello: dict[str, str] = {}
        self.channels: dict[str, str] = {}
        self.pc_limit: int | None = None
        self.sending = (0, 0)
        self.paused = False  # bench aid: stop sending, to imitate a client that has died
        self.unknown_lines = 0
        self._buffer = bytearray()
        self._discard = False

    # ---- from the firmware ----

    def receive(self, data: bytes, now: float) -> list[str]:
        """Feed raw serial bytes. Returns the complete lines, for logging."""
        lines = []
        for byte in data:
            if byte == 10:
                if not self._discard:
                    text = self._buffer.decode("ascii", errors="replace").rstrip("\r")
                    self._handle(text, now)
                    lines.append(text)
                self._buffer.clear()
                self._discard = False
            elif not self._discard:
                if len(self._buffer) >= MAX_LINE:
                    self._buffer.clear()
                    self._discard = True
                else:
                    self._buffer.append(byte)
        return lines

    def _handle(self, text: str, now: float) -> None:
        status = parse_status(text)
        if status is not None:
            if not self.fresh(now):
                self._latch(now)
            self.status = status
            self.status_at = now
            return
        split = split_line(text)
        if split is not None and split[0] == "HELLO":
            self.hello = split[1]
            try:
                limit = int(split[1].get("pc_limit", ""))
            except ValueError:
                limit = None
            self.pc_limit = limit if limit is not None and 0 <= limit <= MAX_COMMAND else None
        elif split is not None and split[0] == "CH":
            self.channels = split[1]
        elif split is not None and split[0] == "NET":
            self.net = split[1]
        else:
            self.unknown_lines += 1

    def fresh(self, now: float) -> bool:
        return self.status is not None and now - self.status_at <= STATUS_FRESH_S

    @property
    def firmware_known(self) -> bool:
        return self.hello.get("fw") in FIRMWARE_IDS

    def probe_due(self, now: float) -> bool:
        """Wi-Fi: True when it is time to send "?" because no status is arriving.

        The robot only sends status to an address it has heard from, so a client with no fresh
        status has to ask. A probe is not a drive command.
        """
        if self.fresh(now) or now - self._probed_at < PROBE_PERIOD_S:
            return False
        self._probed_at = now
        return True

    # ---- from the keyboard ----

    def _latch(self, now: float) -> None:
        self.release_required = True
        self._latched_at = now

    def set_focus(self, focused: bool, now: float) -> None:
        self.focused = focused
        self.keys.clear()
        self._latch(now)

    def key_down(self, key: str, now: float) -> None:
        if not self.focused:
            return
        self.keys.add(key)
        if key == DEADMAN and self.release_required:
            self._latched_at = now  # still held (or repeating): keep waiting

    def key_up(self, key: str) -> None:
        self.keys.discard(key)
        if key == DEADMAN:
            self.release_required = False

    def toggle_pause(self, now: float) -> None:
        self.paused = not self.paused
        self._latch(now)

    # ---- to the firmware ----

    def limits(self) -> tuple[int, int]:
        cap = self.pc_limit if self.pc_limit is not None else MAX_COMMAND
        return min(self.speed, cap), min(self.turn, cap)

    def next_command(self, now: float) -> bytes | None:
        """The line to send this period, or None when there is no fresh status to answer.

        Truthful about the keyboard: a key held through a handoff keeps being sent as
        non-zero, and it is the firmware that refuses it until the keys are released.
        """
        if not self.fresh(now):
            self._latch(now)
            self.sending = (0, 0)
            return None
        if self.paused:
            self._latch(now)
            self.sending = (0, 0)
            return None
        if self.release_required and DEADMAN not in self.keys and now - self._latched_at >= RELEASE_QUIET_S:
            self.release_required = False
        if self.focused and not self.release_required:
            speed, turn = self.limits()
            self.sending = drive_command(self.keys, speed, turn)
        else:
            self.sending = (0, 0)
        if not self.sequenced:
            return format_command(self.status.token, *self.sending)
        self.sequence += 1
        return format_command(self.status.token, *self.sending, self.sequence)

    # ---- for the window ----

    def summary(self, now: float) -> tuple[str, list[str]]:
        """(headline, detail lines)."""
        wifi = self.sequenced
        if self.status is None:
            if wifi:
                return "NO STATUS FROM THE ROBOT", [
                    f"PC link           : {self.link_label}",
                    "No drive commands are being sent (only a status request twice a second).",
                    "Is this laptop joined to the robot's Wi-Fi network, and is the ELRS-Handoff-WiFi sketch running?",
                ] + ([f"Last send error   : {self.link_error}"] if self.link_error else [])
            return "NO STATUS FROM THE BOARD", [
                "Nothing is being sent.",
                "Is the ELRS-Handoff-Bench sketch running, and is this the right COM port?",
            ]
        if not self.fresh(now):
            if wifi:
                return "WI-FI LINK LOST", [
                    f"PC link           : {self.link_label}",
                    "No status for more than 0.5 s. No drive commands are being sent.",
                    "The robot stops by itself 250 ms after the last command it received.",
                    "When the link returns the robot stays stopped: turn ARM off then on, and release the keys.",
                ] + ([f"Last send error   : {self.link_error}"] if self.link_error else [])
            return "STATUS STOPPED", ["No ST line for more than 0.5 s. Nothing is being sent."]
        status = self.status
        if status.pc_ok:
            headline = "PC HAS CONTROL"
        elif status.active == "RADIO":
            headline = "RADIO HAS CONTROL"
        else:
            headline = "NOBODY HAS CONTROL - motors braked"
        speed, turn = self.limits()
        held = "yes" if DEADMAN in self.keys else "no"
        wait = "   (release SPACE once first)" if self.release_required and DEADMAN in self.keys else ""
        if self.release_required and DEADMAN not in self.keys and self.focused:
            wait = "   (checking the keys are released...)"
        via = self.net.get("pc_via")
        link = self.link_label + (f"   (robot says the PC input is held by: {via})" if via else "")
        lines = [
            f"PC link           : {link}",
            f"Radio link        : {'OK' if status.link else 'DOWN'}",
            f"Armed             : {'YES' if status.armed else 'no'}",
            f"MODE switch asks  : {status.requested}",
            f"Active source     : {status.active}",
            f"PC commands       : {'ACCEPTED' if status.pc_ok else 'not accepted'}",
            f"Blocked by        : {status.block}" + (f"  - {BLOCK_HELP.get(status.block, '')}" if status.block != "NONE" else ""),
            f"Fault (latched)   : {status.fault}",
            f"Last stop         : {status.stop}",
            f"Motor outputs     : L={status.left}  R={status.right}",
            "",
            ("Sending           : NOTHING - paused with P to imitate a dead client (P again resumes)" if self.paused else
             f"Sending           : throttle={self.sending[0]}  steer={self.sending[1]}"),
            f"SPACE held        : {held}{wait}",
            f"Window focused    : {'yes' if self.focused else 'NO - click this window to drive'}",
            f"Speed cap         : forward/back {speed}, turn {turn} (of 255)"
            + (f", firmware limit {self.pc_limit}" if self.pc_limit is not None else ""),
        ]
        if self.channels:
            lines += ["", "Radio             : " + " ".join(
                f"{key}={self.channels[key]}" for key in ("ch1", "ch2", "arm", "mode", "lq", "rc_age") if key in self.channels)]
        if not self.firmware_known:
            lines += ["", "Firmware not identified yet (no HELLO line seen)."]
        return headline, lines


# ---------------------------------------------------------------- serial port


def open_port(name: str):
    import serial

    port = serial.Serial()
    port.port = name
    port.baudrate = BAUD
    port.timeout = 0
    port.write_timeout = 0.05
    port.dtr = False  # do not pulse the ESP32-C3 reset lines when opening
    port.rts = False
    port.open()
    return port


def list_ports() -> int:
    from serial.tools import list_ports as tool

    ports = sorted(tool.comports(), key=lambda item: item.device)
    if not ports:
        print("No serial ports found.")
        return 1
    for item in ports:
        print(f"{item.device}  {item.description}")
    return 0


class UdpPort:
    """The Wi-Fi link: UDP datagrams to and from the robot. Same read/write shape as a serial port.

    It never blocks and never raises on a lost network: if the laptop drops off the robot's Wi-Fi,
    sends fail quietly (the reason is kept in ``error``) and the robot times out on its own.
    """

    probe = True        # the robot must hear from this address before it sends status here

    def __init__(self, host: str = WIFI_HOST, port: int = WIFI_UDP_PORT) -> None:
        import socket

        self.address = (host, port)
        self.label = f"Wi-Fi {host}:{port} (UDP)"
        self.error: str | None = None
        self._blocking = (BlockingIOError, InterruptedError)
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._socket.setblocking(False)
        self._socket.bind(("", 0))

    def write(self, data: bytes) -> int:
        try:
            sent = self._socket.sendto(data, self.address)
            self.error = None
            return sent
        except self._blocking:
            return 0
        except OSError as error:        # no route: the laptop is not on the robot's network
            self.error = str(error)
            return 0

    def read(self, _size: int = 4096) -> bytes:
        chunks = []
        for _ in range(64):             # bounded: never spin on a flood
            try:
                data, sender = self._socket.recvfrom(2048)
            except self._blocking:
                break
            except OSError:             # Windows reports an unreachable peer here; nothing was received
                break
            if sender != self.address:
                continue                # only the robot is listened to
            chunks.append(data if data.endswith(b"\n") else data + b"\n")
        return b"".join(chunks)

    def close(self) -> None:
        self._socket.close()


class Log:
    def __init__(self, path: Path | None) -> None:
        self.file = path.open("a", encoding="utf-8") if path else None
        self.last_sent: bytes | None = None

    def received(self, lines: list[str]) -> None:
        if self.file:
            for text in lines:
                self.file.write(f"{time.time():.3f} RX {text}\n")

    def sent(self, command: bytes | None) -> None:
        if self.file and command is not None and command != self.last_sent:
            self.file.write(f"{time.time():.3f} TX {command.decode('ascii').strip()}\n")
        self.last_sent = command

    def close(self) -> None:
        if self.file:
            self.file.close()


def monitor(port, log: Log) -> int:
    """Listen only. Never writes to a serial port; over Wi-Fi it sends only "?" so the robot knows where to report."""
    core = ClientCore()
    asks = getattr(port, "probe", False)
    print('Listening only (Ctrl+C to stop). Over Wi-Fi only "?" is sent, once a second.' if asks
          else "Listening only (Ctrl+C to stop). Nothing is sent to the board.")
    asked = float("-inf")
    try:
        while True:
            if asks and time.monotonic() - asked >= 1.0:
                asked = time.monotonic()
                port.write(b"?\n")
            lines = core.receive(port.read(4096), time.monotonic())
            log.received(lines)
            for text in lines:
                print(text)
            time.sleep(0.01)
    except KeyboardInterrupt:
        return 0


def drive_window(port, port_name: str, core: ClientCore, log: Log) -> int:
    import tkinter as tk

    try:
        import serial
        port_errors: tuple = (OSError, serial.SerialException)
    except ImportError:                 # Wi-Fi does not need pyserial
        port_errors = (OSError,)
    wifi = getattr(port, "probe", False)

    root = tk.Tk()
    root.title(f"AutoBot handoff bench - {port_name}")
    headline = tk.StringVar(value="Connecting")
    detail = tk.StringVar(value="")
    tk.Label(root, text="Hold SPACE and W/A/S/D or the arrow keys. Release to stop.  P: stop sending (dead-client test).  Esc: quit.",
             font=("Segoe UI", 10)).pack(padx=16, pady=(12, 4), anchor="w")
    banner = tk.Label(root, textvariable=headline, font=("Segoe UI", 16, "bold"), fg="white", bg="#555555", padx=12, pady=8)
    banner.pack(padx=16, pady=4, fill="x")
    tk.Label(root, textvariable=detail, font=("Consolas", 10), justify="left", anchor="w").pack(padx=16, pady=(4, 14), fill="x")

    state = {"dead": None, "next": time.monotonic()}

    def write(data: bytes) -> None:
        try:
            port.write(data)
        except port_errors as error:
            state["dead"] = str(error)

    def poll() -> None:
        now = time.monotonic()
        if state["dead"] is None:
            try:
                log.received(core.receive(port.read(4096), now))
            except port_errors as error:
                state["dead"] = str(error)
        if state["dead"] is None:
            command = core.next_command(now)
            if command is not None:
                write(command)
            elif wifi and core.probe_due(now):
                write(b"?\n")          # a status request, never a drive command
            if wifi:
                core.link_error = port.error
            log.sent(command)
        if state["dead"] is not None:
            headline.set("SERIAL PORT LOST")
            detail.set(f"{state['dead']}\n\nNothing is being sent. The firmware stops by itself after 250 ms.\n"
                       "Close this window and start the client again.")
            banner.configure(bg="#aa2222")
            return
        title, lines = core.summary(now)
        headline.set(title)
        detail.set("\n".join(lines))
        banner.configure(bg="#1f7a33" if title == "PC HAS CONTROL" else "#aa2222" if title == "WI-FI LINK LOST" else "#555555")
        state["next"] = max(state["next"] + SEND_PERIOD_S, now)
        state["timer"] = root.after(max(1, int((state["next"] - time.monotonic()) * 1000)), poll)

    def close(_event=None) -> None:
        if state.get("timer") is not None:
            root.after_cancel(state["timer"])
        core.set_focus(False, time.monotonic())
        if state["dead"] is None and core.status is not None:
            core.keys.clear()
            final = core.next_command(time.monotonic())     # a zero command, with the next sequence number on Wi-Fi
            if final is not None:
                write(final)
        root.destroy()

    root.bind("<FocusIn>", lambda _event: core.set_focus(True, time.monotonic()))
    root.bind("<FocusOut>", lambda _event: core.set_focus(False, time.monotonic()))
    root.bind("<KeyPress>", lambda event: core.key_down(event.keysym.lower(), time.monotonic()))
    for pause_key in ("<KeyPress-p>", "<KeyPress-P>"):
        root.bind(pause_key, lambda _event: core.toggle_pause(time.monotonic()))
    root.bind("<KeyRelease>", lambda event: core.key_up(event.keysym.lower()))
    root.bind("<Escape>", close)
    root.protocol("WM_DELETE_WINDOW", close)
    write(b"?\n")
    state["timer"] = root.after(20, poll)
    root.mainloop()
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Keyboard drive client for the ELRS-Handoff-Bench (USB) and ELRS-Handoff-WiFi sketches.")
    parser.add_argument("--port", help="USB: serial port of the ESP32-C3, e.g. COM5")
    parser.add_argument("--wifi", nargs="?", const=WIFI_HOST, metavar="ROBOT_IP",
                        help=f"Wi-Fi: send over UDP to the robot (default {WIFI_HOST}); join the robot's network first")
    parser.add_argument("--udp-port", type=int, default=WIFI_UDP_PORT, help=f"Wi-Fi: the robot's UDP port (default {WIFI_UDP_PORT})")
    parser.add_argument("--list", action="store_true", help="list serial ports and exit")
    parser.add_argument("--monitor", action="store_true", help="listen only: print the firmware's lines, send nothing")
    parser.add_argument("--speed", type=int, default=DEFAULT_SPEED, help=f"forward/back command, 0..255 (default {DEFAULT_SPEED})")
    parser.add_argument("--turn", type=int, default=DEFAULT_TURN, help=f"turn command, 0..255 (default {DEFAULT_TURN})")
    parser.add_argument("--log", type=Path, help="append every received line and every change of command to this file")
    args = parser.parse_args(argv)

    if args.list:
        return list_ports()
    if args.port and args.wifi:
        parser.error("use --port (USB) or --wifi, not both: only one link drives at a time")
    if not args.port and not args.wifi:
        parser.error("--port COMx (USB) or --wifi is required (use --list to see the serial ports)")
    try:
        core = ClientCore(args.speed, args.turn, sequenced=bool(args.wifi),
                          link_label=f"Wi-Fi {args.wifi}:{args.udp_port} (UDP)" if args.wifi else f"USB {args.port}")
    except ValueError as error:
        parser.error(str(error))
    name = args.wifi or args.port
    try:
        port = UdpPort(args.wifi, args.udp_port) if args.wifi else open_port(args.port)
    except Exception as error:  # pyserial raises SerialException; a missing module raises ImportError
        print(f"Could not open {name}: {error}", file=sys.stderr)
        return 1
    log = Log(args.log)
    try:
        if args.monitor:
            return monitor(port, log)
        return drive_window(port, port.label if args.wifi else args.port, core, log)
    finally:
        log.close()
        port.close()


if __name__ == "__main__":
    raise SystemExit(main())
