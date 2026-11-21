"""ELRS-Handoff-WiFi: the USB handoff sketch plus a Wi-Fi (UDP) PC link, and the client's --wifi option.

No hardware, no real Wi-Fi network, no serial port. The firmware tests compile the real .ino for the PC
with MSVC against a mocked board, access point and UDP socket (skipped when the build tools are missing).
The client's UDP code is exercised over the loopback interface only.
"""
import hashlib
import re
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tests"))
import handoff_bench_client as client  # noqa: E402
import test_handoff_bench as usb  # noqa: E402  (its scenario list, mutants and MSVC lookup)

WIFI_DIR = ROOT / "ELRS-Handoff-WiFi"
SKETCH = WIFI_DIR / "ELRS-Handoff-WiFi.ino"
USB_SKETCH = ROOT / "ELRS-Handoff-Bench" / "ELRS-Handoff-Bench.ino"
USB_SKETCH_SHA256 = "1a400d3fdf66d0a718d6c96166a6a7a891641ec47a93f31c5fefea84d8cb5fbc"     # the version tested on the robot
TESTED_CLIENT = ROOT / "artifacts" / "code-snapshots" / "handoff-usb-tested-2026-11-20" / "handoff_bench_client.py"
TESTED_CLIENT_SHA256 = "35130dcdd3224b6635a18963cd6e1ecf6ca6a877133de4427d5396cb8bbd91b2"

WIFI_SCENARIOS = [
    "wifi_drive_without_usb", "wifi_link_drop_latches", "wifi_radio_takeover_while_sending", "wifi_disarm_and_radio_loss",
    "wifi_and_usb_never_command_together", "wifi_old_duplicate_and_foreign_datagrams", "wifi_flood_cannot_delay_the_radio",
    "wifi_access_point_failure_changes_nothing_else", "wifi_status_follows_the_laptop",
]
SCENARIOS = usb.SCENARIOS + WIFI_SCENARIOS

# Broken copies of the Wi-Fi additions. Each must make at least one scenario fail.
WIFI_MUTANTS = {
    "usb_and_wifi_may_command_together": ("    if (via != pcVia) {\n\n      pcOtherTransport++;\n      return;\n    }\n", ""),
    "any_address_may_command": ("      if (peerIp != pcPeerIp || peerPort != pcPeerPort) {\n\n        pcOtherTransport++;\n        return;\n      }\n", ""),
    "late_datagrams_applied": ("      if (sequence <= pcLastSequence) {", "      if (false) {"),
    "owner_kept_after_a_handoff": ("  pcSteer = 0;\n  pcVia = PC_VIA_NONE;\n", "  pcSteer = 0;\n"),
    "unread_datagram_not_dropped": ("    wifiUdp.clear();\n", ""),
    "flood_not_bounded": ("  for (int i = 0; i < WIFI_MAX_DATAGRAMS; i++) {", "  for (int i = 0; i < 100000; i++) {"),
    "wifi_accepts_the_usb_format": ("    if (!parsePcCommandText(text, true, lineToken, throttle, steer, sequence)) {",
                                    "    sequence = pcLastSequence + 1;\n    if (!parsePcCommandText(text, false, lineToken, throttle, steer, sequence)) {"),
    "status_sent_to_any_sender": ("  if (\n    granted && pcVia == PC_VIA_WIFI &&\n    (peerIp != pcPeerIp || peerPort != pcPeerPort)\n  ) {\n    return;\n  }\n", ""),
    "status_peer_never_forgotten": ("  if (statusPeerKnown && now - statusPeerAt > WIFI_PEER_TIMEOUT) {", "  if (false) {"),
    "more_than_one_laptop": ("    0,      // name is visible\n    1       // one laptop at a time", "    0,      // name is visible\n    4       // one laptop at a time"),
    "open_network": ("    WIFI_AP_SSID,\n    WIFI_AP_PASSWORD,\n    WIFI_AP_CHANNEL,", "    WIFI_AP_SSID,\n    \"\",\n    WIFI_AP_CHANNEL,"),
}


# ---------------------------------------------------------------- building


def build(sketch_dir: Path, out: Path):
    vcvars = next((path for path in usb.VCVARS if path.exists()), None)
    if vcvars is None:
        pytest.skip("MSVC build tools not found: firmware host tests not run")
    out.mkdir(parents=True, exist_ok=True)
    script = out / "build.cmd"
    script.write_text(
        "@echo off\r\n"
        f'call "{vcvars}" >nul 2>&1 || exit /b 9\r\n'
        f'cl /nologo /std:c++17 /EHsc /W3 /we4005 /D_CRT_SECURE_NO_WARNINGS /I"{WIFI_DIR / "host_test" / "mock"}" '
        f'/I"{sketch_dir}" /I"{usb.ORIGINAL_DIR}" /Fo"{out}\\\\" "{WIFI_DIR / "host_test" / "test_wifi_handoff.cpp"}" /Fe"{out / "test_handoff.exe"}"\r\n',
        encoding="ascii")
    done = subprocess.run(["cmd", "/c", str(script)], capture_output=True, text=True, timeout=300)
    return done.returncode, done.stdout + done.stderr, out / "test_handoff.exe"


@pytest.fixture(scope="session")
def firmware(tmp_path_factory):
    code, output, exe = build(WIFI_DIR, tmp_path_factory.mktemp("handoff_wifi_host"))
    assert code == 0, output
    return exe


def first_failing(exe: Path, names) -> str | None:
    for name in names:
        if usb.run_scenario(exe, name)[0] != 0:
            return name
    return None


# ---------------------------------------------------------------- files


def test_fallbacks_are_untouched():
    assert hashlib.sha256((usb.ORIGINAL_DIR / "ELRS-Control.ino").read_bytes()).hexdigest() == usb.ORIGINAL_SHA256
    assert hashlib.sha256(USB_SKETCH.read_bytes()).hexdigest() == USB_SKETCH_SHA256
    assert hashlib.sha256(TESTED_CLIENT.read_bytes()).hexdigest() == TESTED_CLIENT_SHA256


def function_text(source: str, name: str) -> str:
    match = re.search(rf"^[A-Za-z_][\w \*]*\b{name}\(.*?^\}}$", source, re.M | re.S)
    assert match, name
    return match.group(0)


def code_only(source: str) -> str:
    return "\n".join(line.split("//")[0] for line in source.splitlines())


def test_wifi_sketch_keeps_the_tested_radio_arming_and_motor_code():
    """The Wi-Fi sketch is the tested USB sketch plus a transport: these parts are character-for-character the same."""
    assert SKETCH.name == SKETCH.parent.name + ".ino"
    assert [p.name for p in WIFI_DIR.iterdir() if p.suffix in {".ino", ".h", ".cpp", ".c"}] == [SKETCH.name]
    old, new = USB_SKETCH.read_text(encoding="utf-8"), SKETCH.read_text(encoding="utf-8")
    for name in ("brakeMotors", "setLeftMotor", "setRightMotor", "crc8DvbS2", "calculateCRC", "decodeChannels", "processCRSFFrame",
                 "readCRSF", "normalizeChannel", "driveMixed", "switchPosition", "disarm", "changeTarget", "parseUnsigned",
                 "parseCommandValue", "readPc", "sendStatus", "sendChannels"):
        assert function_text(old, name) == function_text(new, name), name
    assert function_text(new, "clearCommands") == function_text(old, "clearCommands").replace(
        "  pcSteer = 0;\n", "  pcSteer = 0;\n  pcVia = PC_VIA_NONE;\n")

    def decisions(source):                                  # loop(): from the radio link check to the motor outputs
        return source[source.index("  // RADIO LINK\n"):source.index("  // DIAGNOSTICS (after the motors are set)\n")]

    assert decisions(old) == decisions(new)
    old_code, code = code_only(old), code_only(new)
    for same in ("#define AIN1 3", "#define AIN2 2", "#define BIN1 4", "#define BIN2 5", "#define ELRS_RX_PIN 20",
                 "#define ARM_CHANNEL       5", "#define MODE_CHANNEL      6", "#define ARM_WHEN_HIGH  1", "#define PC_WHEN_HIGH   1",
                 "#define PC_TIMEOUT 250", "#define FAILSAFE_TIMEOUT 250", "#define PC_COMMAND_LIMIT 128"):
        assert same in code and same in old_code, same
    for motor_call in ("driveMixed(", "setLeftMotor(", "setRightMotor(", "brakeMotors()", "analogWrite("):
        assert code.count(motor_call) == old_code.count(motor_call), motor_call     # no second motor path
    assert "digitalWrite" not in code and "Serial.print" not in code
    assert code.count("delay(") == 1                        # the tested sketch's 500 ms at boot; nothing else waits
    assert "while (!Serial" not in code and "waitForConnectResult" not in code and "esp_now" not in code
    assert "Wire" not in code and "IMU" not in code.upper().replace("MAXIMUM", "")


# ---------------------------------------------------------------- firmware on the mocked board


def test_scenario_list_is_complete(firmware):
    done = subprocess.run([str(firmware), "--list"], capture_output=True, text=True, timeout=30)
    assert done.stdout.split() == SCENARIOS


@pytest.mark.parametrize("name", SCENARIOS)
def test_firmware_scenario(firmware, name):
    code, output = usb.run_scenario(firmware, name)
    assert code == 0 and "[FAIL]" not in output and "[PASS]" in output, output


@pytest.mark.parametrize("name", sorted(WIFI_MUTANTS))
def test_broken_wifi_addition_is_caught(firmware, tmp_path, name):
    old, new = WIFI_MUTANTS[name]
    text = SKETCH.read_text(encoding="utf-8")
    assert text.count(old) == 1, f"mutation target for {name} not found exactly once"
    (tmp_path / SKETCH.name).write_text(text.replace(old, new), encoding="utf-8", newline="\n")
    code, output, exe = build(tmp_path, tmp_path / "out")
    assert code == 0, output
    assert first_failing(exe, WIFI_SCENARIOS + usb.SCENARIOS) is not None, f"no scenario noticed the mutation {name}"


@pytest.mark.parametrize("name", sorted(usb.MUTANTS))
def test_broken_copy_of_the_inherited_logic_is_still_caught(firmware, tmp_path, name):
    """The 16 deliberate breaks of the USB version, applied to the Wi-Fi sketch."""
    old, new = usb.MUTANTS[name]
    if name == "handoff_keeps_old_pc_command":              # this function has one more line in the Wi-Fi sketch
        old = "  pcThrottle = 0;\n  pcSteer = 0;\n  pcVia = PC_VIA_NONE;\n\n  granted = false;"
    text = SKETCH.read_text(encoding="utf-8")
    assert text.count(old) == 1, f"mutation target for {name} not found exactly once in the Wi-Fi sketch"
    (tmp_path / SKETCH.name).write_text(text.replace(old, new), encoding="utf-8", newline="\n")
    code, output, exe = build(tmp_path, tmp_path / "out")
    assert code == 0, output
    assert first_failing(exe, SCENARIOS) is not None, f"no scenario noticed the mutation {name}"


def test_a_constant_that_differs_from_the_original_sketch_fails_the_build(firmware, tmp_path):
    text = SKETCH.read_text(encoding="utf-8")
    old = "#define AIN2 2\n"
    assert text.count(old) == 1
    (tmp_path / SKETCH.name).write_text(text.replace(old, "#define AIN2 6\n"), encoding="utf-8", newline="\n")
    code, output, _ = build(tmp_path, tmp_path / "out")
    assert code != 0 and "C4005" in output, output


# ---------------------------------------------------------------- client: Wi-Fi option

ST = ("ST t=11340 tok=1511506142 link=1 armed=1 req=PC act=PC pc_ok=1 block=NONE fault=NONE stop=HANDOFF "
      "L=80 R=80 pc_age=12")


def test_wifi_commands_carry_a_rising_sequence_and_usb_commands_are_unchanged():
    assert client.format_command(7, 80, -70) == b"D,7,80,-70\n"                       # the tested USB format
    assert client.format_command(7, 80, -70, 12) == b"D,7,80,-70,12\n"
    with pytest.raises(ValueError):
        client.format_command(7, 80, 0, 0)

    core = client.ClientCore(sequenced=True, link_label="Wi-Fi test")
    core.set_focus(True, 0.0)
    for t in (0.0, 0.4, 0.8):
        core.receive((ST + "\n").encode(), t)
    assert core.next_command(1.0) == b"D,1511506142,0,0,1\n"
    core.key_down("space", 1.0); core.key_down("w", 1.0)
    assert core.next_command(1.02) == b"D,1511506142,80,0,2\n"
    assert core.next_command(1.04) == b"D,1511506142,80,0,3\n"
    core.key_up("space")
    assert core.next_command(1.06) == b"D,1511506142,0,0,4\n"

    plain = client.ClientCore()
    plain.set_focus(True, 0.0)
    for t in (0.0, 0.4, 0.8):
        plain.receive((ST + "\n").encode(), t)
    assert plain.next_command(1.0) == b"D,1511506142,0,0\n" and plain.sequence == 0


def test_wifi_client_asks_for_status_but_sends_no_drive_command_without_it():
    core = client.ClientCore(sequenced=True, link_label="Wi-Fi 192.168.4.1:4210 (UDP)")
    core.set_focus(True, 0.0)
    core.key_down("space", 0.0); core.key_down("w", 0.0)
    assert core.next_command(0.0) is None and core.probe_due(0.0)                 # no status yet: only "?" goes out
    assert not core.probe_due(0.2) and core.probe_due(0.5)                        # twice a second, not a flood
    headline, lines = core.summary(0.6)
    assert headline == "NO STATUS FROM THE ROBOT" and any("joined to the robot's Wi-Fi" in line for line in lines)
    core.receive((ST + "\n").encode(), 1.0)
    assert not core.probe_due(1.2) and core.next_command(1.2) is not None
    # The link drops: after 0.5 s without status nothing but probes is sent, even with the keys held.
    assert core.next_command(1.6) is None and core.probe_due(1.6)
    headline, lines = core.summary(1.7)
    assert headline == "WI-FI LINK LOST" and any("stays stopped" in line for line in lines)
    # Status comes back with the keys still held: zero is sent until SPACE has really been released.
    core.receive((ST + "\n").encode(), 3.0)
    core.key_down("space", 3.0)
    assert core.next_command(3.02).startswith(b"D,1511506142,0,0,")
    core.key_up("space"); core.key_down("space", 3.1); core.receive((ST + "\n").encode(), 3.1)
    assert core.next_command(3.12).startswith(b"D,1511506142,80,0,")


def test_net_line_and_both_firmware_names_are_understood():
    core = client.ClientCore(sequenced=True, link_label="Wi-Fi test")
    core.set_focus(True, 0.0)
    core.receive(b"HELLO fw=autobot-handoff-wifi-1 arm=CH5:HIGH pc=CH6:HIGH pc_limit=128 pc_timeout_ms=250 rc_timeout_ms=250 "
                 b"wifi_ap=AutoBot-Handoff wifi_ok=1 wifi_channel=6 udp_port=4210\n"
                 b"NET ap=1 sta=1 peer=192.168.4.2:50123 pc_via=WIFI rx=10 bad=0 old=0 other=0\n" + ST.encode() + b"\n", 0.0)
    assert core.firmware_known and core.pc_limit == 128 and core.net["pc_via"] == "WIFI" and core.unknown_lines == 0
    assert any("held by: WIFI" in line for line in core.summary(0.1)[1])
    core.receive(b"HELLO fw=autobot-handoff-bench-1 arm=CH5:HIGH pc=CH6:HIGH pc_limit=128 pc_timeout_ms=250 rc_timeout_ms=250\n", 0.2)
    assert core.firmware_known


@pytest.fixture
def robot_socket():
    """A stand-in robot on the loopback interface (not a real network)."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("127.0.0.1", 0))
    sock.settimeout(1.0)
    yield sock
    sock.close()


def test_udp_port_sends_one_datagram_per_command_and_reads_only_the_robot(robot_socket):
    host, port = robot_socket.getsockname()
    link = client.UdpPort(host, port)
    try:
        assert link.read() == b""                                              # nothing waiting: returns at once
        assert link.write(b"?\n") == 2
        data, laptop = robot_socket.recvfrom(100)
        assert data == b"?\n"
        link.write(client.format_command(5, 80, 0, 1)); link.write(client.format_command(5, 0, 0, 2))
        assert [robot_socket.recvfrom(100)[0] for _ in range(2)] == [b"D,5,80,0,1\n", b"D,5,0,0,2\n"]
        robot_socket.sendto((ST + "\n").encode(), laptop)
        robot_socket.sendto(b"NET ap=1 sta=1 peer=x pc_via=NONE rx=1 bad=0 old=0 other=0", laptop)     # no newline: one is added
        stranger = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        stranger.sendto(ST.replace("tok=1511506142", "tok=666").encode() + b"\n", laptop)                # not the robot: ignored
        stranger.close()
        time.sleep(0.1)
        core = client.ClientCore(sequenced=True)
        core.receive(link.read(), 0.0)
        assert core.status.token == 1511506142 and core.net["sta"] == "1" and core.unknown_lines == 0
    finally:
        link.close()


def test_udp_port_survives_an_unreachable_robot_without_raising():
    closed = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    closed.bind(("127.0.0.1", 0))
    host, port = closed.getsockname()
    closed.close()                                                             # nobody is listening there any more
    link = client.UdpPort(host, port)
    try:
        started = time.perf_counter()
        for _ in range(50):
            link.write(b"D,1,80,0,1\n")
            assert link.read() == b""
        assert time.perf_counter() - started < 0.5                              # never waits
    finally:
        link.close()
    nowhere = client.UdpPort("203.0.113.1", 4210)                              # documentation address: no such route or no answer
    try:
        nowhere.write(b"?\n")
        assert nowhere.read() == b""
    finally:
        nowhere.close()


def test_usb_and_wifi_are_exclusive_on_the_command_line(monkeypatch):
    with pytest.raises(SystemExit):
        client.main(["--port", "COM5", "--wifi"])
    with pytest.raises(SystemExit):
        client.main([])
    opened = {}

    class FakeUdp:
        probe = True
        label = "Wi-Fi fake"

        def __init__(self, host, port):
            opened["address"] = (host, port)

        def close(self):
            opened["closed"] = True

    monkeypatch.setattr(client, "UdpPort", FakeUdp)
    monkeypatch.setattr(client, "drive_window", lambda port, name, core, log: opened.update(name=name, sequenced=core.sequenced) or 0)
    monkeypatch.setattr(client, "open_port", lambda name: pytest.fail("the serial port must not be opened for --wifi"))
    assert client.main(["--wifi"]) == 0
    assert opened == {"address": ("192.168.4.1", 4210), "name": "Wi-Fi fake", "sequenced": True, "closed": True}
    assert client.main(["--wifi", "10.0.0.7", "--udp-port", "5000", "--speed", "60"]) == 0 and opened["address"] == ("10.0.0.7", 5000)


def test_wifi_monitor_sends_only_status_requests():
    class Port:
        probe = True
        written = []
        reads = [b"", (ST + "\n").encode(), b""]

        def read(self, _size):
            if not self.reads:
                raise KeyboardInterrupt
            return self.reads.pop(0)

        def write(self, data):
            self.written.append(data)

    port = Port()
    assert client.monitor(port, client.Log(None)) == 0
    assert port.written and set(port.written) == {b"?\n"}


# ---------------------------------------------------------------- the real client against the real sketch, over the simulated Wi-Fi


class WifiBench:
    """The compiled sketch on the mocked board, stepped 20 ms at a time. The laptop is a real ClientCore.

    ``usb`` and ``wifi`` are separate clients. ``wifi_up`` False means the laptop is off the robot's network:
    nothing it sends arrives and nothing the robot sends reaches it.
    """

    LOW, HIGH = 191, 1792

    def __init__(self, exe):
        self.process = subprocess.Popen([str(exe), "cosim"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
        self.now = 0.0
        self.radio, self.steer, self.throttle, self.arm, self.mode = True, 992, 992, self.LOW, self.LOW
        self.usb = None
        self.wifi = None
        self.wifi_up = True
        self.left = self.right = 0
        self.stopped = True
        self.violations = 0
        self.sent_over_wifi: list[bytes] = []

    def start_wifi(self, **kwargs):
        self.wifi = client.ClientCore(sequenced=True, link_label="Wi-Fi (simulated)", **kwargs)
        self.wifi.set_focus(True, self.now)
        return self.wifi

    def start_usb(self):
        self.usb = client.ClientCore()
        self.usb.set_focus(True, self.now)
        return self.usb

    def step(self, count=1):
        for _ in range(count):
            usb_out = self.usb.next_command(self.now) if self.usb else None
            wifi_out = None
            if self.wifi:
                wifi_out = self.wifi.next_command(self.now)
                if wifi_out is None and self.wifi.probe_due(self.now):
                    wifi_out = b"?\n"                                  # exactly what drive_window() sends
                if wifi_out is not None and wifi_out != b"?\n":
                    self.sent_over_wifi.append(wifi_out)
            delivered = wifi_out if self.wifi_up else None
            self.process.stdin.write(f"20 {int(self.radio)} {self.steer} {self.throttle} {self.arm} {self.mode} "
                                     f"{usb_out.hex() if usb_out else '-'} {delivered.hex() if delivered else '-'}\n")
            self.process.stdin.flush()
            reply = self.process.stdout.readline().split()
            while reply and reply[0].startswith("["):
                assert reply[0] == "[PASS]", reply
                reply = self.process.stdout.readline().split()
            self.left, self.right, stopped, self.violations = int(reply[0]), int(reply[1]), int(reply[2]), int(reply[3])
            self.stopped = stopped == 1
            self.now += 0.02
            if self.usb and reply[4] != "-":
                self.usb.receive(bytes.fromhex(reply[4]), self.now)
            if self.wifi and self.wifi_up and reply[5] != "-":
                self.wifi.receive(bytes.fromhex(reply[5]), self.now)

    def raw_wifi(self, datagram: bytes):
        """Deliver one datagram by hand (for replaying old ones)."""
        self.process.stdin.write(f"20 {int(self.radio)} {self.steer} {self.throttle} {self.arm} {self.mode} - {datagram.hex()}\n")
        self.process.stdin.flush()
        reply = self.process.stdout.readline().split()
        self.left, self.right, self.stopped, self.violations = int(reply[0]), int(reply[1]), reply[2] == "1", int(reply[3])
        self.now += 0.02

    def stays_stopped(self, count):
        ok = True
        for _ in range(count):
            self.step()
            ok = ok and self.stopped
        return ok

    def hold(self, core, *keys):
        """Keys physically held: Windows keeps repeating key-down events."""
        for key in keys:
            core.key_down(key, self.now)

    def close(self):
        self.process.stdin.close()
        assert self.process.wait(timeout=30) == 0 and self.violations == 0


@pytest.fixture
def bench(firmware):
    rig = WifiBench(firmware)
    yield rig
    rig.close()


def arm(bench, mode):
    bench.mode = mode
    bench.arm = WifiBench.LOW
    bench.step(10)
    bench.arm = WifiBench.HIGH
    bench.step(10)


def test_cosim_acceptance_sequence_over_wifi_with_no_usb(bench):
    """The physical acceptance test, on the model: drive, drop the Wi-Fi, reconnect, RadioMaster takeover."""
    laptop = bench.start_wifi()
    bench.step(60)                                           # joined; startup key latch clears
    assert laptop.status is not None and laptop.firmware_known and laptop.net["ap"] == "1"
    arm(bench, WifiBench.HIGH)                               # RadioMaster: ARM on, MODE = PC
    bench.step(5)
    assert laptop.status.pc_ok and laptop.net["pc_via"] == "WIFI" and laptop.summary(bench.now)[0] == "PC HAS CONTROL"

    for keys, expected in ((("space", "w"), (80, 80)), (("space", "s"), (-80, -80)), (("space", "d"), (70, -70)), (("space", "a"), (-70, 70))):
        bench.hold(laptop, *keys)
        bench.step(5)
        assert (bench.left, bench.right) == expected, keys   # forward, reverse, turn right, turn left
        for key in keys:
            laptop.key_up(key)
        bench.step(3)
        assert bench.stopped and laptop.status.pc_ok         # stop: zero command, the laptop keeps control

    bench.hold(laptop, "space", "w")
    bench.step(5)
    assert (bench.left, bench.right) == (80, 80)
    bench.wifi_up = False                                    # laptop leaves the robot's Wi-Fi while commanding motion
    bench.step(11)
    assert not bench.stopped                                 # about 240 ms on: inside the robot's 250 ms window
    bench.step(2)
    assert bench.stopped                                     # wheels stop although the last command was forward
    for _ in range(100):                                     # 2 s off the network, keys still held
        bench.hold(laptop, "space", "w")
        bench.step()
        assert bench.stopped
    assert laptop.summary(bench.now)[0] == "WI-FI LINK LOST" and laptop.next_command(bench.now) is None

    bench.wifi_up = True                                     # laptop rejoins, keys STILL held
    for _ in range(150):                                     # 3 s: no automatic motion
        bench.hold(laptop, "space", "w")
        bench.step()
        assert bench.stopped
    assert laptop.status.fault == "PC_STALE" and laptop.status.block == "REARM_REQUIRED" and not laptop.status.armed
    assert laptop.status.stop == "PC_STALE" and laptop.net["pc_via"] == "NONE"

    for datagram in bench.sent_over_wifi[:40]:               # the commands from before the drop, replayed byte for byte
        bench.raw_wifi(datagram)
        assert bench.stopped

    bench.arm = WifiBench.LOW; bench.step(10)                # deliberate re-enable on the RadioMaster ...
    bench.arm = WifiBench.HIGH
    for _ in range(100):                                     # ... with the laptop keys still held: still no motion
        bench.hold(laptop, "space", "w")
        bench.step()
        assert bench.stopped
    laptop.key_up("space"); laptop.key_up("w")               # ... and a fresh neutral from the laptop
    bench.step(5)
    assert laptop.status.pc_ok
    bench.hold(laptop, "space", "w")
    bench.step(5)
    assert (bench.left, bench.right) == (80, 80)

    bench.mode = WifiBench.LOW                               # RadioMaster takeover while the laptop keeps sending forward
    bench.step(1)
    assert bench.stopped
    for _ in range(25):
        bench.hold(laptop, "space", "w")
        bench.step()
        assert bench.stopped                                 # sticks centred: RADIO has control, nothing moves
    assert laptop.status.active == "RADIO" and not laptop.status.pc_ok and laptop.summary(bench.now)[0] == "RADIO HAS CONTROL"
    bench.throttle = 1811
    for _ in range(100):                                     # 2 s of stick with the laptop still sending forward 80
        bench.hold(laptop, "space", "w")
        bench.step()
        assert (bench.left, bench.right) == (255, 255)
    assert laptop.status.pc_age_ms <= 25                     # the robot was still hearing the laptop all along


def test_cosim_usb_client_and_wifi_client_never_drive_together(bench):
    cable = bench.start_usb()
    laptop = bench.start_wifi()
    bench.step(60)
    bench.hold(laptop, "space", "s")                         # Wi-Fi key held (reverse) from before arming
    arm(bench, WifiBench.HIGH)
    bench.step(5)
    assert cable.status.pc_ok and laptop.net["pc_via"] == "USB"       # only the USB client was neutral
    bench.hold(cable, "space", "w")
    for _ in range(50):
        bench.hold(laptop, "space", "s")
        bench.step()
    assert (bench.left, bench.right) == (80, 80)             # USB forward; Wi-Fi reverse ignored throughout
    bench.usb = None                                         # USB cable pulled
    bench.step(11)
    assert (bench.left, bench.right) == (80, 80)
    bench.step(3)
    assert bench.stopped
    for _ in range(50):                                      # the Wi-Fi laptop cannot pick the drive up by itself
        bench.hold(laptop, "space", "s")
        bench.step()
        assert bench.stopped
    assert laptop.status.fault == "PC_STALE"
