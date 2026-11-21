"""ELRS-Handoff-Telemetry: the Wi-Fi handoff sketch plus read-only IMU telemetry.

MOCK evidence only: the real .ino compiled for the PC against a mocked board, access point, UDP
socket and I2C sensor (skipped without the MSVC build tools). Nothing here touches hardware.
"""
import hashlib
import importlib.util
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
import test_handoff_bench as usb  # noqa: E402
import test_handoff_wifi as wifi  # noqa: E402

TEL_DIR = ROOT / "ELRS-Handoff-Telemetry"
SKETCH = TEL_DIR / "ELRS-Handoff-Telemetry.ino"
HARNESS = TEL_DIR / "host_test" / "test_telemetry_handoff.cpp"
TELEMETRY_SCENARIOS = ["telemetry_imu_over_wifi_without_usb", "telemetry_imu_absent_is_reported_not_zeroed",
                       "telemetry_imu_failure_and_recovery", "telemetry_hung_bus_is_bounded", "telemetry_i2c_cannot_start"]
SCENARIOS = wifi.SCENARIOS + ["pin_trace"] + TELEMETRY_SCENARIOS
SENSOR_STATES = ("present", "absent", "hung")

# Broken copies of the telemetry additions. Each must make at least one scenario fail.
MUTANTS = {
    "failed_read_sent_as_zeros": ("      sendImuInvalid(\"READ_FAILED\");\n\n      if (imuConsecutiveFails",
                                  "      sendImuLine(\"IMU seq=1 t_us=1 ok=1 ax=0.0000 ay=0.0000 az=0.0000 gx=0.000 gy=0.000 gz=0.000 unit=m/s2,deg/s\\n\", 93);\n\n      if (imuConsecutiveFails"),
    "absent_sensor_not_reported": ("  imuSequence++;\n  sendImuInvalid(imuState == IMU_READ_FAILED ? \"READ_FAILED\" : \"NOT_FOUND\");\n", "  imuSequence++;\n"),
    "sequence_not_counted": ("    lastImuSample = now;\n    imuSequence++;\n", "    lastImuSample = now;\n"),
    "two_transactions_per_pass": ("    imuConfigStep++;\n\n    if (imuConfigStep >= 3) {",
                                  "    imuConfigStep++;\n    imuWriteRegister(imuAddress, 0x12, 0x44);\n\n    if (imuConfigStep >= 3) {"),
    "gyro_scale_wrong": ("#define IMU_GYRO_DPS_PER_COUNT   0.070", "#define IMU_GYRO_DPS_PER_COUNT   0.0175"),
    "accel_sent_in_g": ("#define IMU_ACCEL_MS2_PER_COUNT  (0.000122 * 9.80665)", "#define IMU_ACCEL_MS2_PER_COUNT  (0.000122)"),
    "imu_sent_without_a_peer": ("  if (wifiUdpOk && statusPeerKnown) {\n\n    if (wifiUdp.beginPacket(statusPeerIp, statusPeerPort)) {\n\n      wifiUdp.write((const uint8_t *)line, (size_t)size);\n      wifiUdp.endPacket();\n    }\n  }\n}\n\n\nvoid sendImuInvalid",
                                "  if (wifiUdpOk) {\n\n    if (wifiUdp.beginPacket(statusPeerIp, statusPeerPort)) {\n\n      wifiUdp.write((const uint8_t *)line, (size_t)size);\n      wifiUdp.endPacket();\n    }\n  }\n}\n\n\nvoid sendImuInvalid"),
    "imu_decides_driving": ("  imuService(now);\n", "  imuService(now);\n  if (imuState != IMU_RUNNING) { brakeMotors(); }\n"),
}


def build(out: Path, sketch_dir: Path = TEL_DIR, baseline: bool = False):
    vcvars = next((path for path in usb.VCVARS if path.exists()), None)
    if vcvars is None:
        pytest.skip("MSVC build tools not found: firmware host tests not run")
    out.mkdir(parents=True, exist_ok=True)
    includes = [wifi.WIFI_DIR / "host_test" / "mock", wifi.WIFI_DIR if baseline else sketch_dir, usb.ORIGINAL_DIR]
    if not baseline:
        includes.insert(0, TEL_DIR / "host_test" / "mock")
    script = out / "build.cmd"
    script.write_text(
        "@echo off\r\n"
        f'call "{vcvars}" >nul 2>&1 || exit /b 9\r\n'
        "cl /nologo /std:c++17 /EHsc /W3 /we4005 /D_CRT_SECURE_NO_WARNINGS " + ("/DBASELINE_WIFI_SKETCH " if baseline else "")
        + " ".join(f'/I"{path}"' for path in includes) + f' /Fo"{out}\\\\" "{HARNESS}" /Fe"{out / "test_handoff.exe"}"\r\n',
        encoding="ascii")
    done = subprocess.run(["cmd", "/c", str(script)], capture_output=True, text=True, timeout=300)
    return done.returncode, done.stdout + done.stderr, out / "test_handoff.exe"


def run(exe: Path, scenario: str, sensor: str | None = None):
    done = subprocess.run([str(exe), scenario] + ([sensor] if sensor else []), capture_output=True, text=True, timeout=180)
    return done.returncode, done.stdout


@pytest.fixture(scope="session")
def firmware(tmp_path_factory):
    code, output, exe = build(tmp_path_factory.mktemp("handoff_telemetry_host"))
    assert code == 0, output
    return exe


@pytest.fixture(scope="session")
def handoff_only(tmp_path_factory):
    """The same harness built with the handoff-only Wi-Fi sketch."""
    code, output, exe = build(tmp_path_factory.mktemp("handoff_only_host"), baseline=True)
    assert code == 0, output
    return exe


# ---------------------------------------------------------------- files


def test_fallback_builds_are_untouched_and_the_telemetry_build_is_separate():
    assert hashlib.sha256((usb.ORIGINAL_DIR / "ELRS-Control.ino").read_bytes()).hexdigest() == usb.ORIGINAL_SHA256
    assert hashlib.sha256(wifi.USB_SKETCH.read_bytes()).hexdigest() == wifi.USB_SKETCH_SHA256
    assert SKETCH.name == SKETCH.parent.name + ".ino"
    assert [p.name for p in TEL_DIR.iterdir() if p.suffix in {".ino", ".h", ".cpp", ".c"}] == [SKETCH.name]
    handoff = wifi.SKETCH.read_text(encoding="utf-8")
    assert "Wire" not in wifi.code_only(handoff) and "imuService" not in handoff      # no IMU code in the handoff-only build


def test_telemetry_sketch_is_the_wifi_sketch_plus_insertions():
    spec = importlib.util.spec_from_file_location("make_telemetry_sketch", ROOT / "scripts" / "make_telemetry_sketch.py")
    generator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(generator)
    handoff = wifi.SKETCH.read_text(encoding="utf-8")
    telemetry = SKETCH.read_text(encoding="utf-8")
    assert generator.build(handoff) == telemetry, "regenerate with: python scripts/make_telemetry_sketch.py"

    # Every line of the handoff-only sketch is still there, in order; only two lines were reworded.
    changed = {"// RADIO / PC HANDOFF BENCH TEST  +  WI-FI PC LINK", '    "HELLO fw=autobot-handoff-wifi-1 arm=CH%d:%s pc=CH%d:%s "', "}"}
    remaining = iter(telemetry.splitlines())
    missing = [line for line in handoff.splitlines() if line not in changed and not any(line == other for other in remaining)]
    assert missing == [], missing[:5]
    for name in ("brakeMotors", "setLeftMotor", "setRightMotor", "processCRSFFrame", "readCRSF", "normalizeChannel", "driveMixed",
                 "switchPosition", "clearCommands", "disarm", "changeTarget", "acceptPcCommand", "handlePcLine", "readWifi", "readPc",
                 "sendStatus", "sendChannels", "sendNet", "sendLine"):
        assert wifi.function_text(handoff, name) == wifi.function_text(telemetry, name), name

    def decisions(source):                                  # loop(): from the radio link check to the motor outputs
        return source[source.index("  // RADIO LINK\n"):source.index("  // DIAGNOSTICS (after the motors are set)\n")]

    assert decisions(handoff) == decisions(telemetry)
    code, old_code = wifi.code_only(telemetry), wifi.code_only(handoff)
    for motor_call in ("driveMixed(", "setLeftMotor(", "setRightMotor(", "brakeMotors()", "analogWrite("):
        assert code.count(motor_call) == old_code.count(motor_call), motor_call
    loop = telemetry[telemetry.index("void loop() {"):]
    assert loop.index("imuService(now);") > loop.index("// DIAGNOSTICS (after the motors are set)")   # sampled after the motors are set
    imu_code = wifi.code_only(telemetry[telemetry.index("// IMU TELEMETRY\n// ====="):telemetry.index("// SETUP\n")])
    for forbidden in ("armed", "granted", "target", "fault", "block", "pcThrottle", "analogWrite", "brakeMotors", "driveMixed", "delay("):
        assert not re.search(rf"\b{re.escape(forbidden.rstrip('('))}\b", imu_code), forbidden     # the IMU code reads and reports, nothing else
    assert "#define IMU_SDA_PIN  6" in telemetry and "#define IMU_SCL_PIN  7" in telemetry and "#define IMU_I2C_HZ   100000" in telemetry


# ---------------------------------------------------------------- firmware on the mocked board


def test_scenario_list_is_complete(firmware):
    done = subprocess.run([str(firmware), "--list"], capture_output=True, text=True, timeout=30)
    assert done.stdout.split() == SCENARIOS


@pytest.mark.parametrize("sensor", SENSOR_STATES)
@pytest.mark.parametrize("name", SCENARIOS)
def test_firmware_scenario(firmware, name, sensor):
    """Every handoff scenario of the USB and Wi-Fi builds, with the IMU present, absent, and its bus hung."""
    code, output = run(firmware, name, sensor)
    assert code == 0 and "[FAIL]" not in output and "[PASS]" in output, output


def test_motor_pins_match_the_handoff_only_build_tick_for_tick(firmware, handoff_only):
    """One scripted session; the four driver inputs are hashed every millisecond."""
    traces = {"handoff-only": re.search(r"TRACE (\w+) (\d+)", run(handoff_only, "pin_trace")[1]).groups()}
    for sensor in SENSOR_STATES:
        traces[f"telemetry, sensor {sensor}"] = re.search(r"TRACE (\w+) (\d+)", run(firmware, "pin_trace", sensor)[1]).groups()
    assert len(set(traces.values())) == 1, traces
    assert int(traces["handoff-only"][1]) > 7000                # the session really ran (milliseconds)


def test_handoff_only_build_passes_the_same_harness(handoff_only):
    done = subprocess.run([str(handoff_only), "--list"], capture_output=True, text=True, timeout=30)
    assert done.stdout.split() == wifi.SCENARIOS + ["pin_trace"]
    failing = [name for name in wifi.WIFI_SCENARIOS if run(handoff_only, name)[0] != 0]
    assert failing == []


@pytest.mark.parametrize("name", sorted(MUTANTS))
def test_broken_telemetry_addition_is_caught(firmware, handoff_only, tmp_path, name):
    old, new = MUTANTS[name]
    text = SKETCH.read_text(encoding="utf-8")
    assert text.count(old) == 1, f"mutation target for {name} not found exactly once"
    (tmp_path / SKETCH.name).write_text(text.replace(old, new), encoding="utf-8", newline="\n")
    code, output, exe = build(tmp_path / "out", sketch_dir=tmp_path)
    assert code == 0, output
    caught = any(run(exe, scenario, sensor)[0] != 0 for scenario in TELEMETRY_SCENARIOS + ["wifi_status_follows_the_laptop"] for sensor in ("present",))
    if not caught:                                              # a change to driving shows up in the pin trace
        baseline = re.search(r"TRACE (\w+)", run(handoff_only, "pin_trace")[1]).group(1)
        caught = any(re.search(r"TRACE (\w+)", run(exe, "pin_trace", sensor)[1]).group(1) != baseline for sensor in SENSOR_STATES)
    assert caught, f"nothing noticed the mutation {name}"
