"""Telemetry on the PC and the read-only live shadow path.

Evidence classes used here, none of them hardware:
  MOCK     the real telemetry sketch on the mocked board, its lines fed to the real PC parser
  FIXTURE  synthetic rendered frames + scripted telemetry lines in the firmware's exact format
No camera, serial port or robot is opened. The relay tests use the loopback interface only.
"""
import ast
import math
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tests"))

from control_center.tools import live_shadow, live_telemetry  # noqa: E402
from control_center.tools.live_telemetry import (ImuMounting, LiveTelemetry, TapSubscriber, TelemetryRelay, describe_motion,  # noqa: E402
                                                 load_mounting, parse_imu)
from control_center.tools.observation_contract import RECOMMENDED_CHECKPOINT, load_contract  # noqa: E402
from control_center.tools.overlay_core import BotPose, load_config  # noqa: E402

VALID = "IMU seq=10 t_us=200000 ok=1 ax=0.1000 ay=-0.2000 az=9.8067 gx=1.000 gy=-2.000 gz=57.296 unit=m/s2,deg/s"
FAILED = "IMU seq=11 t_us=220000 ok=0 why=READ_FAILED"
STATUS = "ST t=1000 tok=7 link=1 armed=1 req=PC act=NONE pc_ok=0 block=WAIT_PC_NEUTRAL fault=NONE stop=HANDOFF L=0 R=0 pc_age=12"
FIXTURE_LABEL = "FIXTURE SHADOW (synthetic frames + scripted telemetry lines; not live)"
needs_checkpoint = pytest.mark.skipif(not Path(RECOMMENDED_CHECKPOINT).exists(), reason="checkpoint not in runs/")


def verified(matrix=None) -> ImuMounting:
    return ImuMounting(np.eye(3) if matrix is None else np.asarray(matrix, float), True, Path("fixture-mounting.yaml"))


# ---------------------------------------------------------------- IMU lines


def test_imu_line_is_a_complete_sample_or_an_explicit_failure():
    reading = parse_imu(VALID)
    assert (reading.sequence, reading.device_us, reading.valid) == (10, 200000, True)
    assert reading.accel_m_s2 == (0.1, -0.2, 9.8067) and reading.gyro_deg_s == (1.0, -2.0, 57.296)
    failure = parse_imu(FAILED)
    assert (failure.sequence, failure.valid, failure.why, failure.accel_m_s2, failure.gyro_deg_s) == (11, False, "READ_FAILED", None, None)
    for bad in (VALID.replace(" gz=57.296", ""),                       # a value missing
                VALID.replace("unit=m/s2,deg/s", "unit=g,deg/s"),       # other units are not silently accepted
                VALID.replace(" unit=m/s2,deg/s", ""),
                VALID.replace("az=9.8067", "az=nan"), VALID.replace("az=9.8067", "az=inf"),
                VALID.replace("ok=1", "ok=0"),                          # a "failure" that carries numbers
                FAILED + " ax=0.0000 ay=0.0000 az=0.0000 gx=0.000 gy=0.000 gz=0.000",
                FAILED.replace(" why=READ_FAILED", ""), VALID.replace("seq=10", "seq=x"), VALID + " extra=1",
                "IMU,1500,0.12,-0.05,9.807,0.1,-0.2,12.5", "IMU", ""):
        assert parse_imu(bad) is None, bad


def test_mounting_file_is_unverified_by_default_and_refuses_a_mirror(tmp_path):
    shipped = load_mounting()
    assert shipped.verified is False and np.array_equal(shipped.matrix, np.eye(3))     # a placeholder, not a claim
    good = tmp_path / "ok.yaml"
    good.write_text("imu:\n  verified: true\n  sensor_to_body: [[0, -1, 0], [1, 0, 0], [0, 0, 1]]\n", encoding="utf-8")
    mounting = load_mounting(good)
    assert mounting.verified and np.allclose(mounting.to_body([1.0, 0.0, 0.0]), [0.0, 1.0, 0.0])
    for text in ("[[1, 0, 0], [0, -1, 0], [0, 0, 1]]",                # mirrored: would flip a turn direction
                 "[[1, 0, 0], [0, 1, 0], [0, 0, 2]]", "[[1, 0], [0, 1]]"):
        bad = tmp_path / "bad.yaml"
        bad.write_text(f"imu:\n  verified: true\n  sensor_to_body: {text}\n", encoding="utf-8")
        with pytest.raises(ValueError):
            load_mounting(bad)


def test_axis_check_words_match_the_body_frame():
    g = 9.80665
    assert "STILL AND FLAT" in describe_motion([0, 0, 0], [0.1, -0.2, g])[0]
    assert "nose pointing UP" in describe_motion([0, 0, 0], [g, 0, 0.5])[0]
    assert "LEFT side up" in describe_motion([0, 0, 0], [0, g, 0.5])[0]
    assert "UPSIDE DOWN" in describe_motion([0, 0, 0], [0, 0, -g])[0]
    assert "scale or sensor problem" in describe_motion([0, 0, 0], [0, 0, 1.0])[0]      # values in g, not m/s^2
    assert "turning LEFT (counter-clockwise" in describe_motion([0, 0, 1.0], [0, 0, g])[0]
    assert "turning RIGHT" in describe_motion([0, 0, -1.0], [0, 0, g])[0]
    assert "nose pitching UP" in describe_motion([0, -1.0, 0], [0, 0, g])[0]
    assert "left side rising" in describe_motion([1.0, 0, 0], [0, 0, g])[0]


def test_live_telemetry_converts_units_and_axes_and_never_invents_a_sample():
    telemetry = LiveTelemetry(verified([[0, -1, 0], [1, 0, 0], [0, 0, 1]]))            # sensor x points to the robot's left
    assert telemetry.latest(1.0) is None and "nothing received" in telemetry.description
    telemetry.feed("RX", STATUS, 1.0)
    assert telemetry.latest(1.0) is None and "does not send IMU telemetry" in telemetry.description
    telemetry.feed("RX", VALID, 1.00)
    sample = telemetry.latest(1.02)
    assert sample.t == 1.00                                                             # PC receipt time, not the device clock
    assert np.allclose(sample.gyro_rad_s, [math.radians(2.0), math.radians(1.0), 1.0], atol=1e-5)     # deg/s -> rad/s, then rotated
    assert np.allclose(sample.accel_m_s2, [0.2, 0.1, 9.8067])
    assert telemetry.controller_line(1.02)[0] == ("CONTROLLER requested PC, confirmed source NONE, ARMED, last stop HANDOFF, applying L=0 R=0")
    assert "STALE" in telemetry.controller_line(2.0)[0]

    telemetry.feed("RX", FAILED, 1.04)                                                  # the controller says the read failed
    assert telemetry.latest(1.05) is None and "READ_FAILED" in telemetry.description and "no values" in telemetry.description
    telemetry.feed("RX", VALID.replace("seq=10", "seq=12").replace("t_us=200000", "t_us=240000"), 1.06)
    assert telemetry.latest(1.07) is not None                                           # valid again

    telemetry.feed("RX", VALID.replace("gz=57.296", "gz=999.000"), 1.08)                # seq 10 again: late or repeated
    assert telemetry.out_of_order == 1 and abs(telemetry.latest(1.09).gyro_rad_s[2] - 1.0) < 1e-4
    telemetry.feed("RX", VALID.replace("seq=10", "seq=20").replace("t_us=200000", "t_us=400000"), 1.10)
    assert telemetry.gaps == 1
    telemetry.feed("TX", "D,7,80,0,5", 1.11)                                            # a command the owner SENT is not robot state
    telemetry.feed("RX", "IMU seq=21 t_us=1 ok=1 garbage", 1.12)
    assert telemetry.malformed == 1 and telemetry.reading.sequence == 20
    assert telemetry.latest(2.5) is None and "silent" in telemetry.description          # the link went quiet

    for n in range(1, 6):                                                               # the controller rebooted: it counts from 1 again
        telemetry.feed("RX", VALID.replace("seq=10", f"seq={n}").replace("t_us=200000", f"t_us={20000 * n}"), 3.0 + 0.02 * n)
        assert (telemetry.restarts, telemetry.reading.sequence) == ((0, 20) if n < 5 else (1, 5))
    assert telemetry.latest(3.11) is not None                                           # one stray old line is dropped; a new stream is adopted

    unverified = LiveTelemetry(ImuMounting(path=Path("imu_mounting.yaml")))
    unverified.feed("RX", VALID, 1.0)
    assert unverified.latest(1.01) is None and "not verified" in unverified.description
    assert unverified.body_values() is not None                                         # still shown, for the axis check


# ---------------------------------------------------------------- one owner, local listeners


def free_port() -> int:
    probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    probe.bind(("127.0.0.1", 0))
    port = probe.getsockname()[1]
    probe.close()
    return port


def test_relay_republishes_to_local_listeners_and_takes_nothing_from_them():
    port = free_port()
    relay = TelemetryRelay(port)
    first, second = TapSubscriber(port), TapSubscriber(port)
    try:
        assert first.poll(0.0) == [] and second.poll(0.0) == []                         # each has now sent "SUB"
        time.sleep(0.05)
        relay.service(100.0)
        assert len(relay.subscribers) == 2
        relay.publish("RX", [VALID, STATUS])
        relay.publish("TX", ["D,7,80,0,5"])
        time.sleep(0.05)
        for listener in (first, second):
            got = listener.poll(0.1)
            assert [(direction, line) for direction, line, _ in got] == [("RX", VALID), ("RX", STATUS), ("TX", "D,7,80,0,5")]
            assert all(abs(stamp - time.time()) < 5 for _, _, stamp in got)             # the owner's PC receipt time travels with each line
        # A listener cannot reach the robot through the relay: anything but "SUB" is dropped.
        intruder = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        intruder.sendto(b"D,7,255,0,99\n", ("127.0.0.1", port))
        intruder.close()
        time.sleep(0.05)
        relay.service(100.1)
        assert len(relay.subscribers) == 2 and not hasattr(relay, "write") and not hasattr(relay, "send")
        relay.service(100.0 + live_telemetry.SUBSCRIBER_TIMEOUT_S + 1)                  # listeners that go quiet are forgotten
        assert relay.subscribers == {}
        with pytest.raises(OSError):
            TelemetryRelay(port)                                                        # a second owner is refused
    finally:
        relay.close(), first.close(), second.close()


def test_link_owner_relays_what_it_reads_and_receive_only_sends_only_status_requests(capsys):
    import telemetry_link

    class Port:
        probe = True
        label = "fake link"
        error = None

        def __init__(self):
            self.written, self.reads = [], [b"HELLO fw=autobot-handoff-telemetry-1 x=1\n" + VALID.encode() + b"\nST par", b"tial\n", (STATUS + "\n").encode()]

        def read(self, _size=4096):
            if not self.reads:
                raise KeyboardInterrupt
            time.sleep(0.4)
            return self.reads.pop(0)

        def write(self, data):
            self.written.append(data)
            return len(data)

        def close(self):
            pass

    port_number = free_port()
    relay, listener = TelemetryRelay(port_number), TapSubscriber(port_number)
    view = LiveTelemetry(ImuMounting(), "robot link")
    raw = Port()
    try:
        listener.poll(0.0)
        time.sleep(0.05)
        port = telemetry_link.RelayingPort(raw, relay, lambda lines, now: [view.feed("RX", line, now) for line in lines])
        assert telemetry_link.receive_only(port, view, relay) == 0
        assert raw.written and set(raw.written) == {b"?\n"}                             # never a drive command
        time.sleep(0.05)
        lines = [line for _, line, _ in listener.poll(1.0)]
        assert lines == ["HELLO fw=autobot-handoff-telemetry-1 x=1", VALID, "ST partial", STATUS]      # split reads are reassembled
        assert view.firmware == "autobot-handoff-telemetry-1" and view.reading.sequence == 10 and view.status.requested == "PC"
        port.write(b"D,7,80,0,1\n"); port.write(b"D,7,80,0,2\n")                         # (drive mode) every change is relayed as SENT
        time.sleep(0.05)
        assert [(d, line) for d, line, _ in listener.poll(1.1)] == [("TX", "D,7,80,0,1"), ("TX", "D,7,80,0,2")]
    finally:
        relay.close(), listener.close()
    assert "IMU ok  seq 10" in capsys.readouterr().out


def test_shadow_and_telemetry_modules_have_no_path_to_the_robot():
    for name in ("live_shadow.py", "live_telemetry.py"):
        path = ROOT / "control_center" / "tools" / name
        source = path.read_text(encoding="utf-8")
        imported = set()
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.Import):
                imported |= {alias.name.split(".")[0] for alias in node.names}
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".")[0])
                imported |= {alias.name for alias in node.names}
        assert not imported & {"serial", "handoff_bench_client", "telemetry_link", "UdpPort", "format_command", "SerialLineSource"}, name
        code = "\n".join(line.split("#")[0] for line in source.splitlines())
        assert 'f"D,' not in code and "format_command" not in code and "192.168" not in code, name
    telemetry_source = (ROOT / "control_center" / "tools" / "live_telemetry.py").read_text(encoding="utf-8")
    assert telemetry_source.count("sendto(") == 2 and 'RELAY_HOST = "127.0.0.1"' in telemetry_source      # relay -> listener, listener -> relay
    assert "import socket" not in (ROOT / "control_center" / "tools" / "live_shadow.py").read_text(encoding="utf-8")


def test_tag_switch_with_a_position_step_is_refused_not_reported_as_movement():
    guard = live_shadow.TagSwitchGuard(tolerance_m=0.020)

    def pose(x, ids):
        return BotPose(state="ok", reason="ok", x_m=x, y_m=0.2, heading_rad=0.0, used_ids=ids, visible_ids=ids)

    assert guard.check(pose(0.200, [0, 1, 2, 3]), 0.00) == ""
    assert guard.check(pose(0.205, [0, 1, 2, 3]), 0.03) == ""                           # same tags: ordinary movement
    assert guard.check(pose(0.212, [2]), 0.06) == ""                                    # tags changed, centre agrees: accepted
    refused = guard.check(pose(0.300, [0]), 0.09)                                       # tags changed and the centre jumped 88 mm
    assert "tag switch" in refused and "not movement" in refused and guard.trips == 1
    assert guard.check(pose(0.301, [0]), 0.12) == ""                                    # same tag again: usable
    assert guard.check(BotPose(reason="no configured bot markers visible"), 0.15) == ""


# ---------------------------------------------------------------- MOCK: the real sketch's lines through the real PC parser


def test_firmware_imu_lines_are_understood_by_the_pc_side(tmp_path):
    import test_handoff_telemetry as tel

    code, output, exe = tel.build(tmp_path / "tel")
    assert code == 0, output
    process = subprocess.Popen([str(exe), "cosim"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    telemetry = LiveTelemetry(verified([[0, 1, 0], [-1, 0, 0], [0, 0, 1]]))
    now, received = 0.0, []
    try:
        for step in range(100):                                                         # 2 s; the laptop sends only "?"
            process.stdin.write(f"20 1 992 992 191 191 - {b'?'.hex() + '0a' if step % 25 == 0 else '-'}\n")
            process.stdin.flush()
            reply = process.stdout.readline().split()
            while reply and reply[0].startswith("["):
                reply = process.stdout.readline().split()
            now += 0.02
            if reply[5] != "-":
                for line in bytes.fromhex(reply[5]).decode("ascii").splitlines():
                    received.append(line)
                    telemetry.feed("RX", line, now)
    finally:
        process.stdin.close()
        assert process.wait(timeout=30) == 0
    imu = [line for line in received if line.startswith("IMU ")]
    assert len(imu) >= 60 and all(parse_imu(line) is not None for line in imu)          # every line the sketch sent parses strictly
    assert telemetry.malformed == 0 and telemetry.out_of_order == 0 and telemetry.firmware == "autobot-handoff-telemetry-1"
    sample = telemetry.latest(now)
    assert sample is not None and np.allclose(sample.accel_m_s2, [0.0, 0.0, 9.807], atol=1e-3)       # the mock sensor reads +1 g on z
    assert telemetry.status is not None and telemetry.status.requested == "RADIO" and telemetry.status.armed is False
    assert telemetry.imu_info.get("state") == "OK" and telemetry.imu_info.get("unit") == "m/s2,deg/s"


# ---------------------------------------------------------------- FIXTURE: frames + telemetry -> blocked or proposed


def scripted_line(sequence: int, t: float, yaw_rate: float) -> str:
    return (f"IMU seq={sequence} t_us={int(t * 1e6)} ok=1 ax=0.0000 ay=0.0000 az=9.8067 "
            f"gx=0.000 gy=0.000 gz={math.degrees(yaw_rate):.3f} unit=m/s2,deg/s")


def fixture_shadow(telemetry: LiveTelemetry):
    from control_center.tools.overlay_fixture import default_camera, fixture_clicks
    from control_center.tools.shadow_stream_demo import fixture_timeline, render_fixture_frame

    config, camera = load_config(), default_camera()
    shadow = live_shadow.LiveShadow(load_contract(), config, fixture_clicks(config, camera), telemetry,
                                    "SYNTHETIC fixture frames + scripted telemetry lines", shadow_label=FIXTURE_LABEL)
    shadow.start(camera.image_size)
    shadow.take_reference([render_fixture_frame(config, camera, None, seed=9000 + i) for i in range(15)], "15 SYNTHETIC empty-floor frames")
    return shadow, config, camera, fixture_timeline(30.0), render_fixture_frame


@needs_checkpoint
def test_fixture_without_usable_imu_is_blocked_with_the_reason():
    for telemetry, expected in ((LiveTelemetry(verified()), "nothing received"),
                                (LiveTelemetry(ImuMounting(path=Path("imu_mounting.yaml"))), "not verified")):
        shadow, config, camera, states, render = fixture_shadow(telemetry)
        sequence = 0

        def feed(t, telemetry=telemetry, expected=expected):
            nonlocal sequence
            if expected == "not verified":                                              # valid lines arrive; the mounting is the problem
                sequence += 1
                telemetry.feed("RX", scripted_line(sequence, t, 0.0), t)

        frames = ((state.t, render(config, camera, state, seed=100 + i)) for i, state in enumerate(states[:24]))
        decisions = live_shadow.run_frames(shadow, frames, feed)
        assert all(d.proposed is None and d.observation is None for d in decisions)
        assert all(any("imu" in entry and expected in entry for entry in d.blocked_by) for d in decisions), decisions[-1].blocked_by
        lines = [text for text, _ in shadow.lines(decisions[-1], decisions[-1].t)]
        assert any(text.startswith("CHECKPOINT recovery-challenger-v2-seed1") for text in lines)
        assert any("SUPERVISED" in text for text in lines) and "BLOCKED BY:" in lines
        assert any("no connection to the robot" in text for text in lines)


@needs_checkpoint
@pytest.mark.skipif(os.environ.get("AUTOBOT_SKIP_SLOW") == "1", reason="AUTOBOT_SKIP_SLOW=1")
def test_slow_fixture_shadow_proposes_only_while_every_input_is_valid():
    """FIXTURE, not live: rendered frames and scripted telemetry through the real checkpoint (about 2 minutes)."""
    from control_center.tools.stream_adapter import CheckpointPolicy

    telemetry = LiveTelemetry(verified())
    shadow, config, camera, states, render = fixture_shadow(telemetry)
    policy = CheckpointPolicy(shadow.contract)
    policy.load()
    shadow.set_policy(policy, "checkpoint loaded")
    assert shadow.contract.training.startswith("SUPERVISED") and shadow.contract.identifier.startswith("recovery-challenger-v2-seed1")

    silent, failing = (0.40, 0.70), (1.00, 1.20)
    sequence, next_line = 0, 0.0
    by_time = {state.t: state for state in states}
    late_sent = False

    def feed(t):                                                                        # the telemetry that has "arrived" by frame time t
        nonlocal sequence, next_line, late_sent
        while next_line <= t + 1e-9:
            sequence += 1
            state = by_time[min(by_time, key=lambda key: abs(key - next_line))]
            if silent[0] <= next_line < silent[1]:
                pass                                                                    # interrupted: nothing arrives
            elif failing[0] <= next_line < failing[1]:
                telemetry.feed("RX", f"IMU seq={sequence} t_us={int(next_line * 1e6)} ok=0 why=READ_FAILED", next_line)
            else:
                telemetry.feed("RX", scripted_line(sequence, next_line, state.own_yaw_rate), next_line)
            next_line += 0.02
        if t >= 1.35 and not late_sent:
            late_sent = True
            telemetry.feed("RX", scripted_line(5, 0.10, 17.0), t)                       # an old datagram arriving late, with a wild value

    frames = ((state.t, render(config, camera, state, seed=100 + i)) for i, state in enumerate(states) if state.delivered)
    decisions = live_shadow.run_frames(shadow, frames, feed)
    phase_of = {round(state.t, 6): state.phase for state in states}
    rows = [(d, phase_of[round(d.frame_t, 6)]) for d in decisions]

    proposals = [d for d, _ in rows if d.proposed is not None]
    assert len(proposals) > 100
    assert all(d.mode == FIXTURE_LABEL and not d.blocked_by and d.observation.shape == (19,) and np.isfinite(d.observation).all() for d in proposals)
    assert all(d.proposed["tactic"] in shadow.contract.tactics for d in proposals)
    assert all(d.proposed is None and d.observation is None for d, _ in rows if d.blocked_by)      # blocked = no inference at all

    def blocked_in(t0, t1, group, text):
        chosen = [d for d, _ in rows if t0 <= d.t <= t1]
        return chosen and all(d.proposed is None and any(group in e and text in e for e in d.blocked_by) for d in chosen)

    assert blocked_in(0.52, 0.68, "imu", "stale"), "interrupted telemetry must block as stale"
    assert blocked_in(1.03, 1.18, "imu", "READ_FAILED"), "an invalid IMU must block with the controller's reason"
    assert telemetry.out_of_order == 1                                                   # the late datagram was dropped ...
    assert all(abs(d.inputs["imu"].values["gyro_z_rad_s"]) < 3.0 for d in proposals)       # ... and its 17 rad/s never reached an observation

    turning = [d for d, phase in rows if phase.startswith("turning, ONE tag covered") and d.proposed is not None]
    assert turning and all(abs(d.observation[2] - 0.8 / shadow.contract.gyro_range_rad_s) < 1e-3 for d in turning[5:])   # gyro z, normalised
    assert all(abs(d.observation[5] - 9.8067 / shadow.contract.accel_range_m_s2) < 1e-3 for d in proposals)              # accel z

    covered = [d for d, phase in rows if phase == "ALL tags covered"]
    assert covered and all(d.proposed is None and any("own_pose" in e for e in d.blocked_by) for d in covered)
    one_tag = [d for d, phase in rows if phase.startswith("THREE tags covered")]
    assert sum(d.inputs["own_pose"].valid and "[2]" in d.inputs["own_pose"].source for d in one_tag) >= len(one_tag) - 2   # one tag is enough
    assert shadow.guard.trips == 0                                                       # correct offsets: switching tags is not a jump

    touching = [d for d, phase in rows if phase == "TOUCHING the other object"]
    late = touching[len(touching) // 3:]
    assert late and all(d.proposed is None and any("opponent" in e for e in d.blocked_by) for d in late)   # merged: position unknown
    assert all(not d.inputs["opponent_position"].valid and not d.inputs["opponent_position"].values for d in late)   # no stale or guessed value
    backing = [d for d, phase in rows if phase == "backing away"]
    assert any(d.proposed is not None for d in backing[len(backing) // 2:]), "proposals must resume after the objects separate"

    record = live_shadow.trace_record(proposals[-1], telemetry, proposals[-1].t)
    assert record["output"].startswith("none") and record["imu_line"]["sequence"] > 0 and "not synchronised" in record["imu_line"]["clock_note"]
