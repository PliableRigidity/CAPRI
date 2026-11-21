"""scripts/motion_check.py: the plan, the simulator reference, the runner and its interlocks.

No hardware and no serial port. The co-simulation tests run the real ELRS-Handoff-Bench sketch on the
mocked board (built with MSVC, skipped without it) against the real runner. They show the logic and the
protocol, not what the robot's wheels do.
"""
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import handoff_bench_client as client  # noqa: E402
import motion_check as mc  # noqa: E402
import test_handoff_bench as bench_tests  # noqa: E402  (reuses its firmware build, nothing else)


def status_line(t=1000, tok=77, link=1, armed=1, req="PC", act="PC", pc_ok=1, block="NONE", fault="NONE",
                stop="HANDOFF", left=0, right=0, pc_age=5) -> bytes:
    return (f"ST t={t} tok={tok} link={link} armed={armed} req={req} act={act} pc_ok={pc_ok} block={block} "
            f"fault={fault} stop={stop} L={left} R={right} pc_age={pc_age}\n").encode("ascii")


HELLO = b"HELLO fw=autobot-handoff-bench-1 arm=CH5:HIGH pc=CH6:HIGH pc_limit=128 pc_timeout_ms=250 rc_timeout_ms=250\n"


def sent(datagrams):
    """(throttle, steer) of every drive command in a tick's output."""
    out = []
    for datagram in datagrams:
        for line in datagram.decode("ascii").splitlines():
            if line.startswith("D,"):
                parts = line.split(",")
                out.append((int(parts[2]), int(parts[3])))
    return out


# ---------------------------------------------------------------- plan


def test_default_plan_uses_the_simulator_tactic_table_and_stays_inside_the_firmware_limit():
    from autobot.sim.env import tactic_to_drive

    plan = mc.build_plan()
    assert [step.tactic for step in plan.steps] == ["stop", "advance", "stop", "turn_left", "stop", "turn_right", "stop", "reverse", "stop"]
    assert plan.max_command <= mc.FIRMWARE_PC_LIMIT and plan.total_s == pytest.approx(7.0)
    by_name = {step.tactic: step for step in plan.steps}
    # advance: both wheels forward, equal. Simulator wheel commands are negative-forward.
    assert by_name["advance"].left == by_name["advance"].right == 102 and by_name["advance"].steer == 0
    assert by_name["reverse"].left == by_name["reverse"].right == -66
    # turn_left: left wheel back, right wheel forward; the client's convention is positive steer = right.
    assert (by_name["turn_left"].left, by_name["turn_left"].right, by_name["turn_left"].steer) == (-71, 71, -71)
    assert (by_name["turn_right"].left, by_name["turn_right"].right, by_name["turn_right"].steer) == (71, -71, 71)
    for step in plan.steps:
        table = tactic_to_drive(["stop", "advance", "reverse", "turn_left", "turn_right"].index(step.tactic))
        for sim_value, table_value in zip(step.sim_wheels, table):
            assert sim_value == pytest.approx(table_value * plan.scale, abs=0.5 / 255)     # only integer rounding apart
        assert (step.left, step.right) == (step.throttle + step.steer, step.throttle - step.steer)


@pytest.mark.parametrize("sequence, scale, message", [
    ("stop:1.0,advance:0.6,stop:1.0", 0.6, "scale"),
    ("stop:1.0,advance:0.6,stop:1.0", 0.0, "scale"),
    ("advance:0.6,stop:1.0", 0.4, "begin and end with stop"),
    ("stop:1.0,advance:0.6", 0.4, "begin and end with stop"),
    ("stop:1.0,advance:1.5,stop:1.0", 0.4, "longer than"),
    ("stop:1.0,advance:0.5,turn_left:0.4,stop:1.0", 0.4, "followed by a stop"),
    ("stop:1.0,advance:0.5,stop:0.2,advance:0.5,stop:1.0", 0.4, "followed by a stop"),
    ("stop:1.0,spin:0.5,stop:1.0", 0.4, "unknown tactic"),
    ("stop:1.0,advance:0.505,stop:1.0", 0.4, "multiple of 20 ms"),
    ("stop:1.0,advance,stop:1.0", 0.4, "tactic:seconds"),
    (",".join(["stop:1.0,advance:1.0"] * 16) + ",stop:1.0", 0.4, "limit"),
])
def test_plans_outside_the_limits_are_refused(sequence, scale, message):
    with pytest.raises(mc.PlanError, match=message):
        mc.build_plan(sequence, scale)


def test_arc_tactics_map_to_curves_in_the_right_direction():
    plan = mc.build_plan("stop:1.0,forward_left:0.4,stop:1.0,forward_right:0.4,stop:1.0", 0.4)
    left, right = plan.steps[1], plan.steps[3]
    assert left.command_class == "forward_left" and left.right > left.left > 0
    assert right.command_class == "forward_right" and right.left > right.right > 0


# ---------------------------------------------------------------- simulator reference


@pytest.fixture(scope="module")
def reference():
    return mc.sim_reference(mc.build_plan())


def test_simulator_reference_moves_in_the_commanded_direction_and_is_repeatable(reference):
    steps = {step["tactic"] + str(step["index"]): step for step in reference["steps"]}
    assert steps["advance1"]["motion_class"] == "forward" and 0.05 < steps["advance1"]["forward_m"] < 0.40
    assert steps["reverse7"]["motion_class"] == "reverse" and -0.40 < steps["reverse7"]["forward_m"] < -0.03
    assert steps["turn_left3"]["motion_class"] == "turn_left" and steps["turn_left3"]["heading_change_deg"] > 10
    assert steps["turn_right5"]["motion_class"] == "turn_right" and steps["turn_right5"]["heading_change_deg"] < -10
    for index in (2, 4, 6, 8):                                    # after each move the simulated robot comes to rest
        assert reference["steps"][index]["motion_class"] == "stop"
    assert reference["evidence"] == "SIM" and len(reference["samples"]) == round(7.0 * 50)
    assert mc.sim_reference(mc.build_plan())["steps"] == reference["steps"]


def test_the_frozen_experiment_config_gives_the_same_reference_when_present(reference):
    frozen = ROOT / "runs" / "ppo-curriculum-overnight" / "effective_sim.yaml"
    if not frozen.is_file():
        pytest.skip("frozen experiment config not present")
    assert mc.sim_reference(mc.build_plan(), frozen)["steps"] == reference["steps"]


def test_motion_class_words():
    assert mc.motion_class(0.12, 1.0) == "forward" and mc.motion_class(-0.08, -2.0) == "reverse"
    assert mc.motion_class(-0.005, 38.0) == "turn_left" and mc.motion_class(0.004, -37.0) == "turn_right"
    assert mc.motion_class(0.003, 0.8) == "stop" and mc.motion_class(0.15, 25.0) == "forward_left"


# ---------------------------------------------------------------- runner on scripted status lines


def ready_runner(**kwargs):
    runner = mc.Runner(mc.build_plan(), unix=lambda: 0.0, **kwargs)
    now = 0.0
    runner.receive(HELLO, now)
    outputs = []
    for _ in range(30):
        runner.receive(status_line(), now)
        outputs += runner.tick(now)
        now += 0.02
    return runner, now, outputs


def run_ticks(runner, now, count, status=None, feed=True):
    """Advance `count` periods; the scripted controller echoes the expected outputs unless `status` is given."""
    outputs = []
    for _ in range(count):
        if feed:
            if status is not None:
                runner.receive(status, now)
            else:
                step = runner.plan.steps[min(max(runner._step, 0), len(runner.plan.steps) - 1)]
                moving = runner.state == mc.RUNNING
                runner.receive(status_line(left=step.left if moving else 0, right=step.right if moving else 0), now)
        outputs += sent(runner.tick(now))
        now += 0.02
    return now, outputs


def test_nothing_but_zero_is_sent_until_the_operator_confirms():
    runner, now, outputs = ready_runner()
    assert runner.state == mc.READY and set(sent(outputs)) == {(0, 0)}
    now, more = run_ticks(runner, now, 250)                       # five seconds of waiting: still only zeros
    assert runner.state == mc.READY and set(more) == {(0, 0)}
    assert runner.headline(now).startswith("READY")


def test_no_status_means_nothing_is_sent_except_a_status_request():
    runner = mc.Runner(mc.build_plan())
    outputs = [datagram for tick in range(100) for datagram in runner.tick(tick * 0.02)]
    assert set(outputs) == {b"?\n"} and len(outputs) == 4            # one request every half second, no drive command
    assert runner.state == mc.WAITING and "no status" in runner.not_ready_reason(2.0)
    assert runner.go(2.0).startswith("not ready")


@pytest.mark.parametrize("fields, reason", [
    (dict(armed=0, act="NONE", pc_ok=0, block="DISARMED"), "DISARMED"),
    (dict(req="RADIO", act="RADIO", pc_ok=0), "MODE to PC"),
    (dict(act="NONE", pc_ok=0, block="WAIT_PC_NEUTRAL"), "WAIT_PC_NEUTRAL"),
    (dict(armed=0, act="NONE", pc_ok=0, block="REARM_REQUIRED", fault="PC_STALE"), "fault PC_STALE"),
    (dict(link=0, armed=0, act="NONE", pc_ok=0, block="NO_RADIO"), "no radio link"),
    (dict(left=40, right=40), "not zero"),
])
def test_go_is_refused_unless_the_controller_itself_reports_pc_control_at_rest(fields, reason):
    runner = mc.Runner(mc.build_plan())
    runner.receive(HELLO, 0.0)
    now = 0.0
    for _ in range(40):
        runner.receive(status_line(**fields), now)
        assert set(sent(runner.tick(now))) <= {(0, 0)}
        now += 0.02
    assert runner.state == mc.WAITING and reason in runner.not_ready_reason(now)
    assert runner.go(now).startswith("not ready") and runner.state == mc.WAITING


def test_unknown_firmware_or_a_lower_pc_limit_is_not_ready():
    for hello, reason in ((b"HELLO fw=something-else pc_limit=128\n", "not a handoff build"),
                          (b"HELLO fw=autobot-handoff-bench-1 pc_limit=60\n", "below the plan")):
        runner = mc.Runner(mc.build_plan())
        runner.receive(hello + status_line(), 0.0)
        runner.tick(0.0)
        assert runner.state == mc.WAITING and reason in runner.not_ready_reason(0.0)


def test_confirmed_sequence_sends_exactly_the_plan_then_zero_until_disarmed():
    runner, now, _ = ready_runner()
    assert runner.go(now) == ""
    now, outputs = run_ticks(runner, now, 351)
    plan = runner.plan
    expected = [(step.throttle, step.steer) for step in plan.steps for _ in range(round(step.duration_s / 0.02))]
    assert outputs[:350] == expected and outputs[350] == (0, 0)
    assert runner.state == mc.HOLDING and "ARM off" in runner.headline(now)
    now, tail = run_ticks(runner, now, 5, status=status_line(armed=0, act="NONE", pc_ok=0, block="DISARMED", stop="DISARM"))
    assert runner.state == mc.DONE and set(tail) <= {(0, 0)} and runner.tick(now) == []
    result = runner.result()
    assert result["completed"] and result["abort"] is None and result["controller_reported_every_moving_step"]
    assert [record["tactic"] for record in result["steps"]] == [step.tactic for step in plan.steps]
    assert all(record["source"].startswith("PC scripts/motion_check.py over USB") for record in result["steps"])
    assert [round(record["actual_s"], 2) for record in result["steps"]] == [step.duration_s for step in plan.steps]
    assert {item["tactic"] for item in result["stop_latencies"]} == {"advance", "turn_left", "turn_right", "reverse"}


@pytest.mark.parametrize("fields, code", [
    (dict(armed=0, act="NONE", pc_ok=0, block="DISARMED", stop="DISARM"), "DISARMED"),
    (dict(req="RADIO", act="NONE", pc_ok=0, block="WAIT_RADIO_NEUTRAL"), "NOT_PC"),
    (dict(tok=78, left=102, right=102), "TOKEN_CHANGED"),
    (dict(link=0, armed=0, act="NONE", pc_ok=0, block="RADIO_LOST", fault="RADIO_LOST"), "RADIO_LOST"),
    (dict(armed=0, act="NONE", pc_ok=0, block="REARM_REQUIRED", fault="PC_STALE"), "PC_STALE"),
])
def test_any_loss_of_authority_during_a_move_stops_the_sequence_for_good(fields, code):
    runner, now, _ = ready_runner()
    runner.go(now)
    now, _ = run_ticks(runner, now, 60)                           # into the advance step
    assert runner.state == mc.RUNNING and runner._step == 1
    now, outputs = run_ticks(runner, now, 3, status=status_line(**fields))
    assert runner.state == mc.ABORTED and runner.abort_code == code and set(outputs) == {(0, 0)}
    # The controller comes back looking perfect: the sequence still does not resume.
    now, later = run_ticks(runner, now, 100, status=status_line())
    assert set(later) <= {(0, 0)} and runner.state == mc.ABORTED and runner.go(now).startswith("not ready")
    assert runner.finished(now) and not runner.result()["completed"]
    assert runner.tick(now) == []                                 # zeros for half a second, then silence


def test_status_silence_during_a_move_stops_sending_altogether():
    runner, now, _ = ready_runner()
    runner.go(now)
    now, _ = run_ticks(runner, now, 60)
    now, outputs = run_ticks(runner, now, 40, feed=False)         # the controller goes quiet
    assert runner.abort_code == "STATUS_LOST"
    assert outputs.count((102, 0)) <= 26 and outputs[-1:] != [(102, 0)]     # at most the 0.5 s freshness window
    assert runner.tick(now) == []


def test_controller_reporting_outputs_the_step_did_not_command_stops_the_sequence():
    runner, now, _ = ready_runner()
    runner.go(now)
    now, _ = run_ticks(runner, now, 20)                           # still inside the first stop step
    now, outputs = run_ticks(runner, now, 15, status=status_line(left=60, right=60))
    assert runner.abort_code == "OUTPUT_MISMATCH" and "L=60 R=60" in runner.abort_text
    assert set(outputs) == {(0, 0)}


def test_operator_stop_and_the_assertion_that_guards_non_zero_commands():
    runner, now, _ = ready_runner()
    with pytest.raises(AssertionError):
        runner._command(50, 0, now)                               # not running: a non-zero command is a bug
    runner.go(now)
    now, _ = run_ticks(runner, now, 60)
    runner.abort("OPERATOR", "stopped from the keyboard", now)
    now, outputs = run_ticks(runner, now, 50)
    assert set(outputs) == {(0, 0)} and len(outputs) <= 26 and runner.result()["abort"]["code"] == "OPERATOR"
    runner.abort("LATER", "a second reason must not replace the first", now)
    assert runner.result()["abort"]["code"] == "OPERATOR"


def test_wifi_commands_carry_a_rising_sequence_number():
    runner, now, outputs = ready_runner(sequenced=True, link_label="Wi-Fi 192.168.4.1:4210 (UDP)")
    numbers = [int(line.decode().strip().split(",")[4]) for line in outputs if line.startswith(b"D,")]
    assert numbers == list(range(1, len(numbers) + 1))


# ---------------------------------------------------------------- the runner against the real sketch (mocked board)


@pytest.fixture(scope="module")
def firmware(tmp_path_factory):
    code, output, exe = bench_tests.build(bench_tests.SKETCH_DIR, tmp_path_factory.mktemp("motion_host"))
    assert code == 0, output
    return exe


class Rig:
    """The compiled ELRS-Handoff-Bench sketch on the mocked board, in 20 ms steps, talking to a Runner."""

    LOW, HIGH = 191, 1792

    def __init__(self, exe):
        self.process = subprocess.Popen([str(exe), "cosim"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
        self.now = 0.0
        self.radio, self.steer, self.throttle, self.arm, self.mode = True, 992, 992, self.LOW, self.LOW
        self.left = self.right = 0
        self.stopped, self.violations = True, 0
        self.outputs: list[tuple[int, int]] = []
        self.wire: list[bytes] = []

    def raw(self, data: bytes) -> bytes:
        """20 ms of firmware time with `data` arriving on its USB port; returns what it wrote back."""
        self.wire.append(data)
        self.process.stdin.write(f"20 {int(self.radio)} {self.steer} {self.throttle} {self.arm} {self.mode} {data.hex() or '-'}\n")
        self.process.stdin.flush()
        reply = self.process.stdout.readline().split()
        while reply and reply[0].startswith("["):
            assert reply[0] == "[PASS]", reply
            reply = self.process.stdout.readline().split()
        self.left, self.right, self.violations = int(reply[0]), int(reply[1]), int(reply[3])
        self.stopped = reply[2] == "1"
        self.outputs.append((self.left, self.right))
        self.now += 0.02
        return b"" if reply[4] == "-" else bytes.fromhex(reply[4])

    def step(self, runner, count=1):
        for _ in range(count):
            back = self.raw(b"".join(runner.tick(self.now)) if runner is not None else b"")
            if back and runner is not None:
                runner.receive(back, self.now)

    def arm_in(self, runner, mode):
        self.mode, self.arm = mode, self.LOW
        self.step(runner, 15)
        self.arm = self.HIGH
        self.step(runner, 25)

    def until(self, runner, condition, limit=600):
        for _ in range(limit):
            if condition():
                return True
            self.step(runner)
        return condition()

    def close(self):
        self.process.stdin.close()
        assert self.process.wait(timeout=30) == 0 and self.violations == 0


@pytest.fixture
def rig(firmware):
    bench = Rig(firmware)
    yield bench
    bench.close()


def segments(outputs):
    """Run-length encode a list of (L, R) motor outputs."""
    runs = []
    for pair in outputs:
        if runs and runs[-1][0] == pair:
            runs[-1][1] += 1
        else:
            runs.append([pair, 1])
    return [(pair, count) for pair, count in runs]


def test_cosim_the_real_sketch_drives_exactly_the_plan_and_only_after_go(rig):
    runner = mc.Runner(mc.build_plan())
    rig.step(runner, 60)                                          # boot, HELLO, disarmed
    assert runner.state == mc.WAITING and "DISARMED" in runner.not_ready_reason(rig.now)
    assert runner.go(rig.now).startswith("not ready")
    rig.arm_in(runner, Rig.HIGH)                                  # the operator arms with PC selected
    assert rig.until(runner, lambda: runner.state == mc.READY)
    rig.step(runner, 150)                                         # three seconds armed and ready: nothing moves
    assert set(rig.outputs) == {(0, 0)} and rig.stopped
    start = len(rig.outputs)
    assert runner.go(rig.now) == ""
    assert rig.until(runner, lambda: runner.state == mc.HOLDING)
    moves = [(pair, count) for pair, count in segments(rig.outputs[start:]) if pair != (0, 0)]
    plan_moves = [((step.left, step.right), round(step.duration_s / 0.02)) for step in runner.plan.steps if step.moving]
    assert [pair for pair, _ in moves] == [pair for pair, _ in plan_moves]
    for (_, count), (_, planned) in zip(moves, plan_moves):
        assert abs(count - planned) <= 2                          # within 40 ms of the planned duration
    assert rig.stopped
    rig.arm = Rig.LOW                                             # the operator disarms
    assert rig.until(runner, lambda: runner.state == mc.DONE)
    result = runner.result()
    assert result["completed"] and result["controller_reported_every_moving_step"]
    assert result["firmware_hello"]["fw"] == "autobot-handoff-bench-1"
    sent_values = {pair for datagrams in [rig.wire] for pair in sent(datagrams)}
    assert sent_values == {(0, 0)} | {(step.throttle, step.steer) for step in runner.plan.steps}


def run_check(rig, name, interrupt=None):
    plan, cut = mc.check_plans(mc.DEFAULT_SCALE)[name]
    runner = mc.Runner(plan, cut=cut)
    rig.step(runner, 60)
    rig.arm_in(runner, Rig.HIGH)
    assert rig.until(runner, lambda: runner.state == mc.READY)
    assert runner.go(rig.now) == ""
    if interrupt is not None:
        assert rig.until(runner, lambda: runner._step == 1 and not rig.stopped)
        rig.step(runner, 10)                                      # 0.2 s into the move
        interrupt()
    assert rig.until(runner, lambda: runner.finished(rig.now), limit=1500)
    return runner, runner.result()


def test_cosim_stop_check_passes_on_the_real_sketch(rig):
    runner, result = run_check(rig, "stop")
    ok, why = mc.judge_check("stop", result)
    assert ok, why
    assert rig.stopped


def test_cosim_disarm_during_a_move_stops_the_sketch_and_the_runner(rig):
    runner, result = run_check(rig, "disarm", lambda: setattr(rig, "arm", Rig.LOW))
    assert result["abort"]["code"] == "DISARMED" and rig.stopped
    ok, why = mc.judge_check("disarm", result)
    assert ok, why
    rig.arm = Rig.HIGH                                            # re-armed with the old runner still attached
    rig.step(runner, 100)
    assert set(rig.outputs[-100:]) == {(0, 0)} and runner.state == mc.ABORTED


def test_cosim_mode_to_radio_during_a_move_hands_over_and_the_runner_stays_out(rig):
    runner, result = run_check(rig, "takeover", lambda: setattr(rig, "mode", Rig.LOW))
    assert result["abort"]["code"] in ("NOT_PC", "TOKEN_CHANGED") and rig.stopped
    ok, why = mc.judge_check("takeover", result)
    assert ok, why
    rig.throttle = 1500                                           # the pilot's stick now drives
    rig.step(runner, 20)
    assert rig.left > 0 and rig.right > 0 and runner.tick(rig.now) == []


def test_cosim_cutting_the_pc_link_on_purpose_makes_the_sketch_stop_by_itself(rig):
    runner, result = run_check(rig, "pc_link_loss")
    ok, why = mc.judge_check("pc_link_loss", result)
    assert ok, why
    assert result["link_cut"]["fault"] == "PC_STALE" and result["link_cut"]["seconds_after_last_command"] <= 0.35
    assert rig.stopped and runner.tick(rig.now) == []
    rig.step(runner, 100)
    assert set(rig.outputs[-100:]) == {(0, 0)}


def test_cosim_transmitter_off_during_a_move_stops_the_sketch_and_the_runner(rig):
    runner, result = run_check(rig, "radio_loss", lambda: setattr(rig, "radio", False))
    assert result["abort"]["code"] == "RADIO_LOST" and rig.stopped
    ok, why = mc.judge_check("radio_loss", result)
    assert ok, why


def test_cosim_a_check_that_was_not_interrupted_does_not_pass(rig):
    runner, result = run_check(rig, "disarm")                     # the operator never touched ARM
    ok, why = mc.judge_check("disarm", result)
    assert not ok and "not operated" in why


# ---------------------------------------------------------------- the whole command line, real clock, mocked board


class Operator:
    """A scripted human with a RadioMaster: arms in PC mode, types GO, operates one switch per check, answers yes.

    It is also the serial port: every read is 20 ms of firmware time on the mocked board.
    """

    probe = False

    def __init__(self, rig, interrupts=(None,)):
        self.rig, self.interrupts = rig, list(interrupts)
        self.phase, self.since, self.moving, self.polls = "set_switches", 0, 0, 0
        self.pending = b""
        self.questions = []
        self.started = False                                      # set when a GO has been accepted

    # the serial port
    def write(self, data):
        self.pending += data

    def read(self, _size=4096):
        rig = self.rig
        self.since += 1
        if self.phase == "set_switches":
            rig.radio, rig.mode, rig.arm, self.phase, self.since = True, Rig.HIGH, Rig.LOW, "arm_off", 0
        elif self.phase == "arm_off" and self.since >= 25:
            rig.arm, self.phase, self.moving = Rig.HIGH, "armed", 0
        elif self.phase == "armed" and not rig.stopped:
            self.moving += 1
            if self.moving == 10 and self.interrupts and self.interrupts[0] is not None:
                self.interrupts[0](rig)
        elif self.phase == "armed" and self.moving and rig.stopped and self.interrupts[:1] == [None]:
            self.since = self.since if getattr(self, "resting", False) else 0
            self.resting = True
            if self.since >= 150:                                 # finished and at rest for 3 s: disarm
                rig.arm = Rig.LOW
        data, self.pending = self.pending, b""
        return rig.raw(data)

    def close(self):
        pass

    # the keyboard
    def poll(self):
        self.polls += 1
        return "GO" if self.phase == "armed" and not self.started and self.polls % 20 == 0 else None

    def yes(self, question, idle=None):
        self.questions.append(question)
        if "wheels off the ground" not in question and "clear, flat floor" not in question:
            self.interrupts = self.interrupts[1:] or [None]       # that check is over: set up for the next one
            self.phase, self.resting, self.started = "set_switches", False, False
        return True


@pytest.fixture
def workspace(tmp_path, monkeypatch, rig):
    monkeypatch.setattr(mc, "OUT_ROOT", tmp_path)
    monkeypatch.setattr(mc, "SAFETY_RECORD", tmp_path / "safety-checks.json")

    accepted_go = mc.Runner.go

    def attach(operator):
        def go(self, now):
            refusal = accepted_go(self, now)
            operator.started = operator.started or refusal == ""
            return refusal
        monkeypatch.setattr(mc.Runner, "go", go)
        monkeypatch.setattr(mc, "Lines", lambda: operator)
        monkeypatch.setattr(mc, "open_link", lambda args: (operator, "USB mocked board", False))
        return operator
    return tmp_path, rig, attach


SHORT = "stop:0.5,advance:0.4,stop:0.5,turn_left:0.3,stop:0.5"


def test_command_line_raised_run_writes_a_record_that_compare_reads(workspace, capsys):
    folder, rig, attach = workspace
    attach(Operator(rig))
    assert mc.main(["run", "--port", "COM99", "--surface", "raised", "--power", "usb-only", "--sequence", SHORT]) == 0
    moves = [pair for pair, _ in segments(rig.outputs) if pair != (0, 0)]
    assert moves == [(102, 102), (-71, 71)] and rig.stopped
    run_folder = next(folder.glob("run-*-raised"))
    record = json.loads((run_folder / "run.json").read_text(encoding="utf-8"))
    assert record["result"]["completed"] and record["declared"] == {"surface": "raised", "power": "usb-only", "wheels_raised_confirmed": True}
    assert record["result"]["firmware_hello"]["fw"] == "autobot-handoff-bench-1"
    assert [step["tactic"] for step in record["result"]["steps"]] == ["stop", "advance", "stop", "turn_left", "stop"]
    assert " TX D," in (run_folder / "link.txt").read_text(encoding="utf-8")
    capsys.readouterr()
    assert mc.main(["compare", str(run_folder)]) == 0             # nothing observed yet: incomplete, not a mismatch
    assert "INCOMPLETE: 2 of 2" in capsys.readouterr().out
    text = (run_folder / "observed.yaml").read_text(encoding="utf-8")
    text = text.replace("left_wheel: null", "left_wheel: forward", 1).replace("right_wheel: null", "right_wheel: forward", 1)
    text = text.replace("left_wheel: null", "left_wheel: reverse", 1).replace("right_wheel: null", "right_wheel: forward", 1)
    (run_folder / "observed.yaml").write_text(text, encoding="utf-8")
    assert mc.main(["compare", str(run_folder)]) == 0
    assert "SIGNS AGREE" in capsys.readouterr().out and (run_folder / "comparison.md").is_file()


def test_command_line_floor_run_is_refused_until_the_five_checks_pass_then_allowed(workspace, capsys):
    folder, rig, attach = workspace
    attach(Operator(rig))
    assert mc.main(["run", "--port", "COM99", "--surface", "floor", "--power", "usb-only", "--sequence", SHORT]) == 2
    assert "REFUSED: no wheels-raised safety checks" in capsys.readouterr().out
    assert set(rig.outputs) == {(0, 0)} and not any(pair != (0, 0) for data in rig.wire for pair in sent([data]))

    def radio_off(bench):
        bench.radio = False
    attach(Operator(rig, [None, lambda bench: setattr(bench, "arm", Rig.LOW), lambda bench: setattr(bench, "mode", Rig.LOW),
                          None, radio_off]))
    assert mc.main(["checks", "--port", "COM99", "--power", "usb-only"]) == 0
    out = capsys.readouterr().out
    assert "Floor runs are now permitted" in out and out.count("controller evidence: PASS") == 5
    saved = json.loads((folder / "safety-checks.json").read_text(encoding="utf-8"))
    assert saved[-1]["firmware"] == "autobot-handoff-bench-1"
    assert all(saved[-1]["checks"][name] == {"controller_pass": True, "controller_evidence": saved[-1]["checks"][name]["controller_evidence"],
                                             "operator_confirmed": True} for name in mc.REQUIRED_CHECKS)
    attach(Operator(rig))
    before = len(rig.outputs)
    assert mc.main(["run", "--port", "COM99", "--surface", "floor", "--power", "usb-only", "--sequence", SHORT]) == 0
    assert "allowed: all five wheels-raised checks passed" in capsys.readouterr().out
    assert [pair for pair, _ in segments(rig.outputs[before:]) if pair != (0, 0)] == [(102, 102), (-71, 71)]


def test_command_line_a_pulled_cable_during_a_move_ends_the_run_and_the_sketch_stops_by_itself(workspace, capsys):
    folder, rig, attach = workspace
    operator = attach(Operator(rig))
    healthy_read = operator.read

    def read(size=4096):
        if operator.moving >= 8:                                  # the cable comes out 0.16 s into the first move
            raise OSError("device disconnected")
        return healthy_read(size)
    operator.read = read
    assert mc.main(["run", "--port", "COM99", "--surface", "raised", "--power", "usb-only", "--sequence", SHORT]) == 1
    out = capsys.readouterr().out
    assert "PORT_LOST" in out and "stops by itself" in out
    record = json.loads((next(folder.glob("run-*-raised")) / "run.json").read_text(encoding="utf-8"))
    assert record["result"]["abort"]["code"] == "PORT_LOST" and not record["result"]["completed"]
    for _ in range(25):                                           # half a second with nothing arriving from the PC
        rig.raw(b"")
    assert rig.stopped and (102, 102) in rig.outputs


# ---------------------------------------------------------------- the floor interlock


def passed_record(firmware, finished, **overrides):
    checks = {name: {"controller_pass": True, "operator_confirmed": True} for name in mc.REQUIRED_CHECKS}
    checks.update(overrides)
    return {"firmware": firmware, "finished_unix": finished, "checks": checks}


def test_floor_runs_need_every_wheels_raised_check_on_this_firmware_recently(tmp_path):
    path = tmp_path / "safety-checks.json"
    now = 1_800_000_000.0
    assert mc.floor_permission("autobot-handoff-bench-1", now, path) == (
        False, "no wheels-raised safety checks are recorded for firmware autobot-handoff-bench-1: run `checks` first")
    path.write_text(json.dumps([passed_record("autobot-handoff-bench-1", now - 600)]), encoding="utf-8")
    assert mc.floor_permission("autobot-handoff-bench-1", now, path)[0]
    assert not mc.floor_permission("autobot-handoff-wifi-1", now, path)[0]          # another build: its own checks
    assert not mc.floor_permission(None, now, path)[0]
    assert "old" in mc.floor_permission("autobot-handoff-bench-1", now + 13 * 3600, path)[1]
    for name in mc.REQUIRED_CHECKS:
        for broken in ({"controller_pass": False, "operator_confirmed": True}, {"controller_pass": True, "operator_confirmed": False}):
            path.write_text(json.dumps([passed_record("autobot-handoff-bench-1", now - 600, **{name: broken})]), encoding="utf-8")
            allowed, why = mc.floor_permission("autobot-handoff-bench-1", now, path)
            assert not allowed and name in why
    # The newest record decides: an older pass does not outlive a newer failure.
    path.write_text(json.dumps([passed_record("autobot-handoff-bench-1", now - 3000),
                                passed_record("autobot-handoff-bench-1", now - 600, stop={"controller_pass": False})]), encoding="utf-8")
    assert not mc.floor_permission("autobot-handoff-bench-1", now, path)[0]
    path.write_text("not json", encoding="utf-8")
    assert not mc.floor_permission("autobot-handoff-bench-1", now, path)[0]


class Typed:
    """Scripted operator answers."""

    def __init__(self, *answers):
        self.answers = list(answers)
        self.asked = []

    def yes(self, question, idle=None):
        self.asked.append(question)
        return self.answers.pop(0)


def options(**values):
    defaults = dict(port="COM5", wifi=None, power="usb-only")
    defaults.update(values)
    return type("Options", (), defaults)()


def test_preflight_records_declarations_and_refuses_without_them(capsys):
    assert mc.preflight(options(port=None), Typed(), "raised") is None
    assert mc.preflight(options(power="battery-only"), Typed(), "raised") is None             # USB cable is plugged in
    assert mc.preflight(options(port=None, wifi="192.168.4.1", power="usb-only"), Typed(), "raised") is None
    assert mc.preflight(options(), Typed(False), "raised") is None                             # wheels not raised
    assert mc.preflight(options(), Typed(True), "raised") == {"surface": "raised", "power": "usb-only", "wheels_raised_confirmed": True}
    # USB and battery together: the measured reverse path is put to the operator, and "no" stops the run.
    typed = Typed(False)
    assert mc.preflight(options(power="battery-and-usb"), typed, "raised") is None
    assert "isolated" in typed.asked[0] and "4.8 V" in capsys.readouterr().out
    assert mc.preflight(options(power="battery-and-usb"), Typed(True, True), "floor") == {
        "surface": "floor", "power": "battery-and-usb", "usb_battery_isolation_confirmed": True, "floor_clear_confirmed": True}


def test_command_line_requires_the_surface_and_power_to_be_stated(capsys):
    for argv in (["run", "--port", "COM5", "--surface", "raised"], ["run", "--port", "COM5", "--power", "usb-only"],
                 ["checks", "--port", "COM5"]):
        with pytest.raises(SystemExit):
            mc.main(argv)
    assert mc.main(["plan", "--scale", "0.9"]) == 2 and "plan refused" in capsys.readouterr().err
    assert mc.main(["plan"]) == 0 and "Nothing was sent anywhere" in capsys.readouterr().out


# ---------------------------------------------------------------- comparison


def run_record(reference, surface, reported=True):
    steps = []
    for step in reference["plan"]["steps"]:
        steps.append({"index": step["index"], "tactic": step["tactic"],
                      "controller_reported_expected_outputs": reported, "reported": [[step["left"], step["right"]]]})
    return {"surface": surface, "result": {"steps": steps}}


def observed(surface_rows):
    return {"measured_by": "tape and phone video", "steps": surface_rows}


def test_comparison_without_a_run_is_labelled_simulator_only(reference):
    report = mc.compare(reference, None, None)
    assert report["verdict"].startswith("SIMULATOR ONLY") and not report["mismatches"]
    assert all(row["controller"] == "NOT RUN" and row["direction_match"] is None for row in report["rows"])
    assert "NOT RECORDED" in mc.format_comparison(report)


def test_wheels_raised_run_compares_wheel_directions_only(reference):
    rows = [{"index": 1, "left_wheel": "forward", "right_wheel": "forward"}, {"index": 3, "left_wheel": "reverse", "right_wheel": "forward"},
            {"index": 5, "left_wheel": "forward", "right_wheel": "reverse"}, {"index": 7, "left_wheel": "reverse", "right_wheel": "reverse"}]
    report = mc.compare(reference, run_record(reference, "raised"), observed(rows))
    assert report["verdict"].startswith("SIGNS AGREE") and "wheels were raised" in report["verdict"] and not report["calibration"]
    rows[1] = {"index": 3, "left_wheel": "forward", "right_wheel": "reverse"}                  # the robot turned the other way
    report = mc.compare(reference, run_record(reference, "raised"), observed(rows))
    assert report["verdict"].startswith("MISMATCH") and "step 3 turn_left" in report["mismatches"][0]


def test_floor_run_reports_sign_agreement_and_calibration_ratios_separately(reference):
    sim = {step["index"]: step for step in reference["steps"]}
    rows = [{"index": 1, "direction": "forward", "distance_mm": round(sim[1]["forward_m"] * 1000 * 1.5), "rotation_deg": 2},
            {"index": 3, "direction": "turn_left", "distance_mm": 5, "rotation_deg": sim[3]["heading_change_deg"] * 2.0},
            {"index": 5, "direction": "turn_right", "distance_mm": 5, "rotation_deg": sim[5]["heading_change_deg"] * 2.0},
            {"index": 7, "direction": "reverse", "distance_mm": round(-sim[7]["forward_m"] * 1000 * 1.5), "rotation_deg": -1}]
    report = mc.compare(reference, run_record(reference, "floor"), observed(rows))
    assert report["verdict"] == "SIGNS AGREE on every moving step" and not report["mismatches"]
    assert any("1.50 x" in text for text in report["calibration"]) and any("2.00 x" in text for text in report["calibration"])
    assert all("not a failure" in text for text in report["calibration"])
    rows[3] = {"index": 7, "direction": "forward", "distance_mm": 60, "rotation_deg": 0}       # reverse drove forward
    rows[1] = {"index": 3, "direction": "turn_left", "rotation_deg": -40}                      # sign of rotation contradicts
    report = mc.compare(reference, run_record(reference, "floor"), observed(rows))
    assert report["verdict"].startswith("MISMATCH") and len(report["mismatches"]) == 2


def test_missing_observations_are_incomplete_never_assumed(reference, tmp_path):
    report = mc.compare(reference, run_record(reference, "floor"), observed([{"index": 1, "direction": "forward", "distance_mm": 150}]))
    assert report["verdict"].startswith("INCOMPLETE: 3 of 4")
    report = mc.compare(reference, run_record(reference, "floor", reported=False), None)
    assert report["verdict"].startswith("MISMATCH") and "did NOT report" in report["mismatches"][0]
    # The template the operator fills in parses back, and an untouched template is INCOMPLETE.
    import yaml
    mc.write_observed_template(tmp_path / "observed.yaml", mc.build_plan(), "floor")
    template = yaml.safe_load((tmp_path / "observed.yaml").read_text(encoding="utf-8"))
    assert [item["index"] for item in template["steps"]] == [1, 3, 5, 7]
    assert mc.compare(reference, run_record(reference, "floor"), template)["verdict"].startswith("INCOMPLETE: 4 of 4")


def test_this_tool_never_opens_a_port_or_arms_by_itself():
    source = (ROOT / "scripts" / "motion_check.py").read_text(encoding="utf-8")
    assert "import serial" not in source                         # the port comes from the tested client's open_port
    assert source.count("client.open_port(") == 1 and source.count("client.UdpPort(") == 1
    for command in ("plan", "sim", "compare"):
        body = source.split(f"def command_{command}(")[1].split("\ndef ")[0]
        assert "open_link" not in body and "Runner(" not in body
