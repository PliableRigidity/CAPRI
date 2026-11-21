"""ELRS-Handoff-Bench: the PC client, the sketch on a mocked board, and the two wired together.

No hardware, no serial port. The firmware tests compile the real .ino for the PC with MSVC
(skipped when the build tools are missing). They show the logic, not real pin levels.
"""
import hashlib
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import handoff_bench_client as client  # noqa: E402

SKETCH_DIR = ROOT / "ELRS-Handoff-Bench"
SKETCH = SKETCH_DIR / "ELRS-Handoff-Bench.ino"
ORIGINAL_DIR = ROOT / "ELRS-Control"
ORIGINAL_SHA256 = "6040bed377e40affa3dc4105d23e96eacb788e6e5afcf4fcaa60cdbfcfdea6dd"
VCVARS = [Path(base) / "Microsoft Visual Studio" / year / edition / "VC/Auxiliary/Build/vcvars64.bat"
          for base in (r"C:\Program Files (x86)", r"C:\Program Files")
          for year in ("2019", "2022") for edition in ("BuildTools", "Community", "Professional")]

SCENARIOS = [
    "boot_no_radio", "boot_arm_already_on", "boot_arm_already_on_pc", "drive_matches_original", "stop_path",
    "original_stop_path_model", "disarm_radio", "disarm_pc", "handoff_radio_to_pc", "handoff_pc_to_radio",
    "handoff_flapping", "pc_stale", "pc_garbage_is_not_alive", "pc_reconnect_old_command", "radio_loss_in_radio",
    "radio_loss_in_pc", "radio_reports_zero_link_quality", "radio_noise_is_not_a_link", "short_rc_frame",
    "switch_invalid", "pc_cannot_arm_or_select", "pc_limit", "usb_not_reading", "diagnostics_rate",
]

# Deliberately broken copies of the sketch. Each must make at least one scenario fail.
MUTANTS = {
    "stop_with_digitalWrite_again": (
        "void brakeMotors() {\n\n  // DRV8833 active brake\n\n  analogWrite(AIN1, 255);\n  analogWrite(AIN2, 255);\n\n  analogWrite(BIN1, 255);\n  analogWrite(BIN2, 255);\n}",
        "void brakeMotors() {\n  digitalWrite(AIN1, HIGH);\n  digitalWrite(AIN2, HIGH);\n  digitalWrite(BIN1, HIGH);\n  digitalWrite(BIN2, HIGH);\n}"),
    "left_stop_with_digitalWrite_again": (
        "    // Active brake\n    analogWrite(AIN1, 255);\n    analogWrite(AIN2, 255);",
        "    digitalWrite(AIN1, HIGH);\n    digitalWrite(AIN2, HIGH);"),
    "radio_takes_over_without_neutral": ("          throttle == 0 &&\n          steering == 0\n", "          true\n"),
    "pc_takes_over_without_neutral": ("    if (throttle != 0 || steer != 0) {\n      return;\n    }\n", ""),
    "pc_token_not_checked": ("if (!armed || target != SOURCE_PC || lineToken != token) {", "if (!armed || target != SOURCE_PC) {"),
    "pc_stale_falls_back_to_radio": (
        "          fault = CAUSE_PC_STALE;\n\n          disarm(CAUSE_PC_STALE);\n          armOffSeen = false;\n\n          block = BLOCK_REARM_REQUIRED;",
        "          changeTarget(SOURCE_RADIO, CAUSE_PC_STALE);"),
    "pc_stale_not_latched": ("          disarm(CAUSE_PC_STALE);\n          armOffSeen = false;", "          disarm(CAUSE_PC_STALE);"),
    "radio_loss_not_latched": ("    disarm(CAUSE_RADIO_LOST);\n    armOffSeen = false;", "    disarm(CAUSE_RADIO_LOST);"),
    "radio_timeout_2500": ("    now - lastValidRCFrame <= FAILSAFE_TIMEOUT;", "    now - lastValidRCFrame <= 2500;"),
    "pc_timeout_2500": ("#define PC_TIMEOUT 250\n", "#define PC_TIMEOUT 2500\n"),
    "arm_without_seeing_the_switch_off": ("      if (armOffSeen) {", "      if (true) {"),
    "pc_can_arm": ("  if (!armed || target != SOURCE_PC || lineToken != token) {\n    return;\n  }",
                   "  if (lineToken == token) armed = true;\n  if (!armed || target != SOURCE_PC || lineToken != token) {\n    return;\n  }"),
    "rc_length_not_checked": ("    if (frameLength != CRSF_RC_FRAME_LENGTH) {", "    if (false) {"),
    "mid_switch_counts_as_low": ("  if (value <= SWITCH_LOW_MAX) {", "  if (value <= SWITCH_HIGH_MIN - 1) {"),
    "pc_limit_not_applied": ("  throttle = constrain(throttle, -PC_COMMAND_LIMIT, PC_COMMAND_LIMIT);", "  throttle = constrain(throttle, -255, 255);"),
    "handoff_keeps_old_pc_command": ("  pcThrottle = 0;\n  pcSteer = 0;\n\n  granted = false;", "  granted = granted && target == SOURCE_PC;"),
}


# ---------------------------------------------------------------- building the firmware for the PC


def build(sketch_dir: Path, out: Path):
    vcvars = next((path for path in VCVARS if path.exists()), None)
    if vcvars is None:
        pytest.skip("MSVC build tools not found: firmware host tests not run")
    out.mkdir(parents=True, exist_ok=True)
    script = out / "build.cmd"
    script.write_text(
        "@echo off\r\n"
        f'call "{vcvars}" >nul 2>&1 || exit /b 9\r\n'
        f'cl /nologo /std:c++17 /EHsc /W3 /we4005 /D_CRT_SECURE_NO_WARNINGS /I"{SKETCH_DIR / "host_test" / "mock"}" '
        f'/I"{sketch_dir}" /I"{ORIGINAL_DIR}" /Fo"{out}\\\\" "{SKETCH_DIR / "host_test" / "test_handoff.cpp"}" /Fe"{out / "test_handoff.exe"}"\r\n',
        encoding="ascii")
    done = subprocess.run(["cmd", "/c", str(script)], capture_output=True, text=True, timeout=300)
    return done.returncode, done.stdout + done.stderr, out / "test_handoff.exe"


def run_scenario(exe: Path, name: str):
    done = subprocess.run([str(exe), name], capture_output=True, text=True, timeout=120)
    return done.returncode, done.stdout


@pytest.fixture(scope="session")
def firmware(tmp_path_factory):
    code, output, exe = build(SKETCH_DIR, tmp_path_factory.mktemp("handoff_host"))
    assert code == 0, output
    return exe


# ---------------------------------------------------------------- files


def test_original_sketch_is_byte_for_byte_untouched():
    data = (ORIGINAL_DIR / "ELRS-Control.ino").read_bytes()
    assert hashlib.sha256(data).hexdigest() == ORIGINAL_SHA256


def test_new_sketch_is_standalone_and_in_scope():
    assert SKETCH.name == SKETCH.parent.name + ".ino"
    assert [path.name for path in SKETCH_DIR.iterdir() if path.suffix in {".ino", ".h", ".cpp", ".c"}] == [SKETCH.name]
    code = "\n".join(line.split("//")[0] for line in SKETCH.read_text(encoding="utf-8").splitlines())
    assert "digitalWrite" not in code                       # the stop-path fix: PWM writes only
    assert "Serial.print" not in code and "Wire" not in code and "IMU" not in code.upper().replace("MAXIMUM", "")
    for define in ("#define AIN1 3", "#define AIN2 2", "#define BIN1 4", "#define BIN2 5", "#define ELRS_RX_PIN 20",
                   "#define CRSF_BAUD 420000", "#define FAILSAFE_TIMEOUT 250", "#define ARM_FRAME_COUNT 10",
                   "#define ARM_CHANNEL       5", "#define MODE_CHANNEL      6"):
        assert define in code


# ---------------------------------------------------------------- firmware on the mocked board


def test_scenario_list_is_complete(firmware):
    done = subprocess.run([str(firmware), "--list"], capture_output=True, text=True, timeout=30)
    assert done.stdout.split() == SCENARIOS


@pytest.mark.parametrize("name", SCENARIOS)
def test_firmware_scenario(firmware, name):
    code, output = run_scenario(firmware, name)
    assert code == 0 and "[FAIL]" not in output and "[PASS]" in output, output


@pytest.mark.parametrize("name", sorted(MUTANTS))
def test_broken_copy_of_the_sketch_is_caught(firmware, tmp_path, name):
    old, new = MUTANTS[name]
    text = SKETCH.read_text(encoding="utf-8")
    assert text.count(old) == 1, f"mutation target for {name} not found exactly once"
    (tmp_path / SKETCH.name).write_text(text.replace(old, new), encoding="utf-8", newline="\n")
    code, output, exe = build(tmp_path, tmp_path / "out")
    assert code == 0, output                                # it still compiles; only the tests can catch it
    failed = [scenario for scenario in SCENARIOS if run_scenario(exe, scenario)[0] != 0]
    assert failed, f"no scenario noticed the mutation {name}"


@pytest.mark.parametrize("old, new", [
    ("#define FAILSAFE_TIMEOUT 250\n", "#define FAILSAFE_TIMEOUT 2500\n"),
    ("#define AIN2 2\n", "#define AIN2 6\n"),
    ("#define DEADZONE   45\n", "#define DEADZONE   20\n"),
    ("#define ARM_FRAME_COUNT 10\n", "#define ARM_FRAME_COUNT 1\n"),
])
def test_a_constant_that_differs_from_the_original_sketch_fails_the_build(firmware, tmp_path, old, new):
    text = SKETCH.read_text(encoding="utf-8")
    assert text.count(old) == 1
    (tmp_path / SKETCH.name).write_text(text.replace(old, new), encoding="utf-8", newline="\n")
    code, output, _ = build(tmp_path, tmp_path / "out")
    assert code != 0 and "C4005" in output, output


# ---------------------------------------------------------------- PC client on its own

ST = ("ST t=11340 tok=1511506142 link=1 armed=1 req=PC act=PC pc_ok=1 block=NONE fault=NONE stop=HANDOFF "
      "L=80 R=80 pc_age=12")


def ready_client(now=0.0, **kwargs):
    """A focused client that has been hearing the board for 0.8 s (status stays fresh until now + 1.3 s)."""
    core = client.ClientCore(**kwargs)
    core.set_focus(True, now)
    for later in (0.0, 0.4, 0.8):
        core.receive((ST + "\n").encode(), now + later)
    return core


def test_status_line_parses_and_anything_malformed_is_no_status():
    status = client.parse_status(ST)
    assert (status.token, status.link, status.armed, status.requested, status.active, status.pc_ok) == (1511506142, True, True, "PC", "PC", True)
    assert (status.block, status.fault, status.stop, status.left, status.right, status.pc_age_ms) == ("NONE", "NONE", "HANDOFF", 80, 80, 12)
    for bad in (ST.replace(" pc_ok=1", ""), ST.replace("tok=1511506142", "tok=x"), ST.replace("armed=1", "armed=2"),
                ST.replace("req=PC", "req=AUTO"), ST + " extra=1", ST.replace("ST ", "CH "), ST.replace("link=1 armed=1", "armed=1 link=1"),
                ST.replace("L=80", "L=800"), ST.replace("tok=1511506142", "tok=4294967296"), "", "ST"):
        assert client.parse_status(bad) is None, bad


def test_nothing_is_sent_without_a_fresh_status():
    core = client.ClientCore()
    core.set_focus(True, 0.0)
    assert core.next_command(1.0) is None                   # never heard from the board
    core.receive((ST + "\n").encode(), 1.0)
    assert core.next_command(1.1) == b"D,1511506142,0,0\n"
    assert core.next_command(1.5) is not None and core.next_command(1.51) is None   # 0.5 s without ST: stop sending
    assert core.summary(1.6)[0] == "STATUS STOPPED"


def test_deadman_key_speed_caps_and_zero_on_release():
    core = ready_client()
    assert core.next_command(1.0) == b"D,1511506142,0,0\n"  # startup latch has cleared by now (keys quiet)
    core.key_down("w", 1.0)
    assert core.next_command(1.02) == b"D,1511506142,0,0\n"  # W without SPACE: zero
    core.key_down("space", 1.02)
    assert core.next_command(1.04) == b"D,1511506142,80,0\n"  # conservative default: 80 of 255
    core.key_down("d", 1.04)
    assert core.next_command(1.06) == b"D,1511506142,80,70\n"
    core.key_up("w"); core.key_down("s", 1.06); core.key_up("d"); core.key_down("left", 1.06)
    assert core.next_command(1.08) == b"D,1511506142,-80,-70\n"
    core.key_down("up", 1.08)
    assert core.next_command(1.10) == b"D,1511506142,0,-70\n"  # forward and back together cancel
    core.key_up("space")
    assert core.next_command(1.12) == b"D,1511506142,0,0\n"  # SPACE released: zero, other keys still down
    core.receive(b"HELLO fw=autobot-handoff-bench-1 arm=CH5:HIGH pc=CH6:HIGH pc_limit=60 pc_timeout_ms=250 rc_timeout_ms=250\n", 1.12)
    core.key_down("space", 1.12)
    assert core.firmware_known and core.next_command(1.14) == b"D,1511506142,0,-60\n"   # capped by the firmware's limit
    with pytest.raises(ValueError):
        client.ClientCore(speed=300)
    with pytest.raises(ValueError):
        client.format_command(1, 256, 0)


def test_the_token_follows_the_latest_status_and_a_held_key_is_sent_truthfully():
    core = ready_client()
    core.next_command(1.0); core.key_down("space", 1.0); core.key_down("w", 1.0)
    assert core.next_command(1.02) == b"D,1511506142,80,0\n"
    # Handoff: new token, firmware waiting for neutral. The client does not fake a release.
    core.receive(ST.replace("tok=1511506142", "tok=77").replace("act=PC pc_ok=1 block=NONE", "act=NONE pc_ok=0 block=WAIT_PC_NEUTRAL").encode() + b"\n", 1.03)
    assert core.next_command(1.04) == b"D,77,80,0\n"
    headline, lines = core.summary(1.04)
    assert headline.startswith("NOBODY HAS CONTROL") and any("release all keys" in line for line in lines)
    assert any(line.startswith("PC commands       : not accepted") for line in lines)


def test_keyboard_is_not_trusted_after_focus_returns_until_space_is_released():
    core = ready_client()
    core.next_command(1.0); core.key_down("space", 1.0); core.key_down("w", 1.0)
    assert core.next_command(1.02) == b"D,1511506142,80,0\n"
    core.set_focus(False, 1.03)                              # another window clicked: no key-up events will arrive
    assert core.next_command(1.04) == b"D,1511506142,0,0\n"
    core.key_down("w", 1.05)
    assert core.keys == set()                                # key events while unfocused are ignored
    core.set_focus(True, 2.0)
    now = 2.0
    for _ in range(100):                                     # SPACE and W still physically held: Windows keeps repeating them
        now += 0.03
        core.key_down("space", now); core.key_down("w", now)
        core.receive((ST + "\n").encode(), now)
        assert core.next_command(now) == b"D,1511506142,0,0\n"
    core.key_up("space")                                     # a real release
    core.key_down("space", now + 0.1); core.receive((ST + "\n").encode(), now + 0.1)
    assert core.next_command(now + 0.12) == b"D,1511506142,80,0\n"
    # Same when the board's status disappears and comes back with the key held.
    assert core.next_command(now + 0.7) is None
    core.receive((ST + "\n").encode(), now + 0.8)
    core.key_down("space", now + 0.8)
    assert core.next_command(now + 0.82) == b"D,1511506142,0,0\n"


def test_long_or_unknown_lines_are_dropped_not_parsed():
    core = client.ClientCore()
    core.receive(b"ST " + b"x" * 400 + b"\n" + ST.encode() + b"\ngarbage here\nCH ch5=191 arm=CH5:191:DISARM mode=CH6:191:RADIO\n", 0.0)
    assert core.status is not None and core.unknown_lines == 1 and core.channels["arm"] == "CH5:191:DISARM"


def test_monitor_mode_never_writes_to_the_port(capsys):
    class Port:
        reads = [b"ST junk\n", (ST + "\n").encode(), b""]

        def read(self, _size):
            if not self.reads:
                raise KeyboardInterrupt
            return self.reads.pop(0)

        def write(self, _data):
            raise AssertionError("monitor mode wrote to the port")

    assert client.monitor(Port(), client.Log(None)) == 0
    assert ST in capsys.readouterr().out


def test_port_is_required_and_listed_ports_need_no_port(monkeypatch, capsys):
    with pytest.raises(SystemExit):
        client.main([])
    monkeypatch.setattr(client, "list_ports", lambda: 0)
    assert client.main(["--list"]) == 0


# ---------------------------------------------------------------- the real client against the real sketch


class Bench:
    """The compiled sketch on the mocked board, driven in 20 ms steps, talking to a real ClientCore."""

    LOW, HIGH = 191, 1792

    def __init__(self, exe):
        self.process = subprocess.Popen([str(exe), "cosim"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
        self.now = 0.0
        self.radio, self.steer, self.throttle, self.arm, self.mode = True, 992, 992, self.LOW, self.LOW
        self.core = None
        self.left = self.right = 0
        self.stopped = True
        self.violations = 0
        self.transcript = ""

    def start_client(self, **kwargs):
        self.core = client.ClientCore(**kwargs)
        self.core.set_focus(True, self.now)
        return self.core

    def step(self, count=1):
        for _ in range(count):
            command = self.core.next_command(self.now) if self.core else None
            payload = command.hex() if command else "-"
            self.process.stdin.write(f"20 {int(self.radio)} {self.steer} {self.throttle} {self.arm} {self.mode} {payload}\n")
            self.process.stdin.flush()
            reply = self.process.stdout.readline().split()
            while reply and reply[0].startswith("["):       # the rig's boot checks come out with the first reply
                assert reply[0] == "[PASS]", reply
                reply = self.process.stdout.readline().split()
            self.left, self.right, stopped, self.violations = int(reply[0]), int(reply[1]), int(reply[2]), int(reply[3])
            self.stopped = stopped == 1
            self.now += 0.02
            if reply[4] != "-":
                data = bytes.fromhex(reply[4])
                self.transcript += data.decode("ascii")
                if self.core:
                    self.core.receive(data, self.now)

    def stays_stopped(self, count):
        ok = True
        for _ in range(count):
            self.step()
            ok = ok and self.stopped
        return ok

    def close(self):
        self.process.stdin.close()
        code = self.process.wait(timeout=30)
        assert code == 0 and self.violations == 0
        for line in self.transcript.splitlines():           # every status line the firmware wrote parses in the client
            if line.startswith("ST "):
                assert client.parse_status(line) is not None, line


@pytest.fixture
def bench(firmware):
    rig = Bench(firmware)
    yield rig
    rig.close()


def arm(bench, mode):
    bench.mode = mode
    bench.arm = Bench.LOW
    bench.step(10)
    bench.arm = Bench.HIGH
    bench.step(10)


def test_cosim_pc_drives_only_with_the_radio_armed_pc_selected_and_keys_fresh(bench):
    core = bench.start_client()
    bench.step(50)                                           # 1 s: startup key latch clears
    core.key_down("space", bench.now); core.key_down("w", bench.now)
    assert bench.stays_stopped(50) and core.status.block == "DISARMED" and not core.status.pc_ok
    arm(bench, Bench.HIGH)                                   # armed in PC with the key already held
    assert bench.stays_stopped(50) and core.status.block == "WAIT_PC_NEUTRAL" and core.sending == (80, 0)
    core.key_up("space"); core.key_up("w")
    bench.step(5)
    assert core.status.pc_ok and core.status.active == "PC" and bench.stopped
    assert core.summary(bench.now)[0] == "PC HAS CONTROL"
    core.key_down("space", bench.now); core.key_down("w", bench.now)
    bench.step(5)
    assert (bench.left, bench.right) == (80, 80)
    bench.step(11)                                           # L/R reach the client with the next heartbeat (every 0.2 s)
    assert (core.status.left, core.status.right) == (80, 80)
    core.key_up("w"); core.key_down("d", bench.now)
    bench.step(5)
    assert (bench.left, bench.right) == (70, -70)
    core.key_up("space")
    bench.step(3)
    assert bench.stopped and core.status.pc_ok              # released: zero command, PC keeps control


def test_cosim_key_and_stick_held_through_both_handoffs_never_move_the_robot(bench):
    core = bench.start_client()
    bench.step(50)
    arm(bench, Bench.LOW)                                    # RADIO
    assert core.status.active == "RADIO" and core.summary(bench.now)[0] == "RADIO HAS CONTROL"
    core.key_down("space", bench.now); core.key_down("w", bench.now)
    bench.throttle = 1811
    bench.step(5)
    assert (bench.left, bench.right) == (255, 255)           # sticks drive; the PC's forward is ignored
    bench.mode = Bench.HIGH                                  # to PC, stick and key both held
    assert bench.stays_stopped(100)
    assert core.status.block == "WAIT_PC_NEUTRAL" and core.status.stop == "HANDOFF" and not core.status.pc_ok
    core.key_up("space"); core.key_up("w")
    bench.step(5)
    core.key_down("space", bench.now); core.key_down("w", bench.now)
    bench.step(5)
    assert (bench.left, bench.right) == (80, 80)             # PC drives; the stick at full is ignored
    bench.mode = Bench.LOW                                   # back to RADIO, both still held
    assert bench.stays_stopped(100) and core.status.block == "WAIT_RADIO_NEUTRAL"
    bench.throttle = 992
    bench.step(3)
    assert bench.stopped and core.status.active == "RADIO"
    bench.throttle = 1811
    bench.step(3)
    assert (bench.left, bench.right) == (255, 255)


def test_cosim_client_killed_then_restarted_with_the_key_held(bench):
    core = bench.start_client()
    bench.step(50)
    arm(bench, Bench.HIGH)
    core.key_down("space", bench.now); core.key_down("w", bench.now)
    bench.step(5)
    assert (bench.left, bench.right) == (80, 80)
    bench.core = None                                        # process killed / USB pulled: commands just stop
    bench.step(11)
    assert not bench.stopped                                 # about 240 ms on: still inside the firmware's 250 ms window
    bench.step(2)
    assert bench.stopped
    core = bench.start_client()                              # restarted, the user still leaning on SPACE + W
    for _ in range(100):
        core.key_down("space", bench.now); core.key_down("w", bench.now)
        bench.step()
        assert bench.stopped
    assert core.status.fault == "PC_STALE" and core.status.block == "REARM_REQUIRED" and not core.status.armed
    bench.arm = Bench.LOW; bench.step(10); bench.arm = Bench.HIGH   # the radio re-arms; the key is still held
    for _ in range(100):
        core.key_down("space", bench.now); core.key_down("w", bench.now)
        bench.step()
        assert bench.stopped
    core.key_up("space"); core.key_up("w")
    bench.step(3)
    core.key_down("space", bench.now); core.key_down("w", bench.now)
    bench.step(5)
    assert (bench.left, bench.right) == (80, 80)


def test_cosim_pause_key_makes_the_firmware_stop_latch_and_ignore_the_held_key(bench):
    core = bench.start_client()
    bench.step(50)
    arm(bench, Bench.HIGH)
    core.key_down("space", bench.now); core.key_down("w", bench.now)
    bench.step(5)
    assert (bench.left, bench.right) == (80, 80)
    core.toggle_pause(bench.now)                             # P: the client stops sending but keeps listening
    assert core.next_command(bench.now) is None
    bench.step(14)
    assert bench.stopped and core.status.fault == "PC_STALE" and core.status.block == "REARM_REQUIRED" and core.status.stop == "PC_STALE"
    assert "paused" in "\n".join(core.summary(bench.now)[1])
    core.toggle_pause(bench.now)                             # P again, SPACE + W still held
    for _ in range(50):
        core.key_down("space", bench.now); core.key_down("w", bench.now)
        bench.step()
        assert bench.stopped and core.sending == (0, 0)
    assert core.status.block == "REARM_REQUIRED" and not core.status.armed


def test_cosim_disarm_radio_loss_and_focus_loss_stop_pc_drive(bench):
    core = bench.start_client()
    bench.step(50)
    arm(bench, Bench.HIGH)
    core.key_down("space", bench.now); core.key_down("w", bench.now)
    bench.step(5)
    assert (bench.left, bench.right) == (80, 80)
    core.set_focus(False, bench.now)                         # another window clicked
    bench.step(2)
    assert bench.stopped and core.status.pc_ok              # zero commands; PC still holds control
    core.set_focus(True, bench.now); core.key_up("space"); core.key_down("space", bench.now); core.key_down("w", bench.now)
    bench.step(5)
    assert (bench.left, bench.right) == (80, 80)
    bench.arm = Bench.LOW                                    # disarm with the key held
    bench.step(1)
    assert bench.stopped
    assert bench.stays_stopped(50) and core.status.block == "DISARMED" and core.status.stop == "DISARM"
    bench.arm = Bench.HIGH
    assert bench.stays_stopped(50) and core.status.block == "WAIT_PC_NEUTRAL"
    core.key_up("space"); bench.step(3); core.key_down("space", bench.now); bench.step(5)
    assert (bench.left, bench.right) == (80, 80)
    bench.radio = False                                      # transmitter off, key held, PC still sending
    bench.step(14)
    assert bench.stopped
    assert bench.stays_stopped(50) and core.status.block == "RADIO_LOST" and not core.status.link
    bench.radio = True
    assert bench.stays_stopped(100) and core.status.block == "REARM_REQUIRED" and core.status.fault == "RADIO_LOST"
