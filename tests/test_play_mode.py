"""Live play (`cli play`): real-time pacing, keys to red commands, neutral on release and focus loss, pause, reset, input log.

The real viewer loop runs headlessly with SYNTHETIC key input and a controlled wall clock (see
scripts/verify_play_mode.py). This checks the controls, the pace and the log, not how a checkpoint fares
against a person.
"""
import json
import sys
import time
import zipfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import verify_play_mode as play  # noqa: E402

from autobot.sim import cli
from autobot.sim.env import arcade_drive

REFERENCE = Path("runs/combined-clone-balanced-dagger-seed2/final_model.zip")
PPO_CHECKPOINT = Path("runs/ppo-curriculum-overnight/final_model.zip")
START = [(2, [], True), (2, ["space"], True)]


def needs(path):
    if not path.is_file(): pytest.skip(f"{path} not present (runs/ is not tracked)")


def steps_of(log):
    return [record for record in (json.loads(line) for line in Path(log).read_text(encoding="utf-8").splitlines()) if record["type"] == "step"]


def test_expected_command_is_the_arcade_mixer_and_neutral_without_focus():
    assert play.expected_command({"w"}, True) == arcade_drive(forward=True) == (-1.0, -1.0)
    assert play.expected_command({"w", "a"}, True) == arcade_drive(forward=True, left=True) == (-0.615, -1.0)      # forward left curve
    assert play.expected_command({"w", "d"}, True) == (-1.0, -0.615) and play.expected_command(set(), True) == (0.0, 0.0)
    assert play.expected_command({"w", "a"}, False) == (0.0, 0.0)                                                     # window not focused


def test_pacer_follows_the_wall_clock_at_any_frame_rate():
    for fps in (20, 30, 60, 144, 240):
        pacer = cli.RealTimePacer(0.02); per_frame = [pacer.due(5.0 + frame / fps) for frame in range(10 * fps + 1)]
        assert sum(per_frame) in (500, 501) and max(per_frame) <= max(1, -(-50 // fps) + 1)        # ten seconds of wall time, 20 ms steps
    slow = cli.RealTimePacer(0.02)
    assert [slow.due(now) for now in (0.0, 0.005, 0.010, 0.015, 0.020, 0.039, 0.040)] == [1, 0, 0, 0, 1, 0, 1]   # never faster than the clock


def test_pacer_does_not_owe_paused_time_and_drops_a_stall_instead_of_jumping():
    pacer = cli.RealTimePacer(0.02)
    assert pacer.due(1.00) == 1 and pacer.due(1.02) == 1
    pacer.hold()                                                   # paused, reset or round over
    assert pacer.due(61.0) == 1 and pacer.due(61.02) == 1          # a minute paused is not replayed
    assert pacer.due(64.0) == 5 and pacer.due(64.02) == 1          # a 3 s stall: five steps, the rest of the backlog is dropped
    assert pacer.due(63.0) == 0 and pacer.due(63.01) == 0          # a clock that steps backwards produces nothing


def checkpoint_stub(directory, timesteps, training=None):
    directory.mkdir(parents=True)
    with zipfile.ZipFile(directory / "final_model.zip", "w") as archive: archive.writestr("data", json.dumps({"num_timesteps": timesteps}))
    if training is not None: (directory / "effective_config.json").write_text(json.dumps({"training": training}), encoding="utf-8")
    return directory / "final_model.zip"


def test_hud_label_says_what_kind_of_checkpoint_blue_is(tmp_path):
    assert cli.blue_controller_label(checkpoint_stub(tmp_path / "ppo", 250_000)) == "PPO"
    assert cli.blue_controller_label(checkpoint_stub(tmp_path / "ppo_run", 250_000, training="")) == "PPO"
    assert cli.blue_controller_label(checkpoint_stub(tmp_path / "clone", 0, training="SUPERVISED imitation of the scripted controller")) == "Supervised clone"
    assert cli.blue_controller_label(checkpoint_stub(tmp_path / "bare_clone", 0)) == "Supervised clone"
    if REFERENCE.is_file(): assert cli.blue_controller_label(REFERENCE) == "Supervised clone"
    if PPO_CHECKPOINT.is_file(): assert cli.blue_controller_label(PPO_CHECKPOINT) == "PPO"


def test_every_session_gets_a_new_input_log_name(tmp_path):
    started = time.mktime((2026, 11, 19, 14, 5, 9, 0, 0, -1)); requested = tmp_path / "reference-normal-spawn.jsonl"
    first = cli.session_log_path(requested, started)
    assert first == tmp_path / "reference-normal-spawn-20261119-140509.jsonl" and not first.exists()
    first.touch(); second = cli.session_log_path(requested, started); second.touch()
    assert second == tmp_path / "reference-normal-spawn-20261119-140509-2.jsonl"
    assert cli.session_log_path(requested, started).name == "reference-normal-spawn-20261119-140509-3.jsonl"
    assert cli.session_log_path(tmp_path / "round", started).name == "round-20261119-140509.jsonl"
    assert cli.session_log_path(requested, started + 1).name == "reference-normal-spawn-20261119-140510.jsonl"


def test_play_loop_follows_keys_goes_neutral_and_logs_every_step(tmp_path):
    needs(REFERENCE); scenarios, index, seed = play.NORMAL; requested = tmp_path / "round.jsonl"
    schedule = START + [(10, ["w"], True), (10, ["w", "a"], True), (8, [], True), (8, ["w", "d"], False),
                        (2, ["space"], True), (6, ["w"], True), (2, ["space"], True), (5, ["s"], True), (3, ["r"], True), (4, ["w"], True)]
    harness = play.Harness(schedule).run(play.arguments("play", str(REFERENCE), scenarios, index, seed, "--input-log", str(requested)))
    assert harness.log != requested and harness.log.parent == tmp_path and harness.log.name.startswith("round-") and not requested.exists()
    records = [json.loads(line) for line in harness.log.read_text(encoding="utf-8").splitlines()]
    steps = [record for record in records if record["type"] == "step"]; commands = [tuple(record["red_command"]) for record in steps]
    # Stepped frames, in order: 2 SPACE frames (start), 10 W, 10 W+A, 8 released, 8 unfocused W+D, then SPACE pauses at once and
    # 6 frames of W do nothing, 2 SPACE frames (resume), 5 S. R then resets; W before SPACE does nothing, nor does the closing frame.
    assert commands == ([(0.0, 0.0)] * 2 + [(-1.0, -1.0)] * 10 + [(-0.615, -1.0)] * 10 + [(0.0, 0.0)] * 8 + [(0.0, 0.0)] * 8
                        + [(0.0, 0.0)] * 2 + [(0.65, 0.65)] * 5)
    times = [record["time_s"] for record in steps]
    assert times[0] == pytest.approx(0.02) and all(b - a == pytest.approx(0.02) for a, b in zip(times, times[1:]))
    assert all(record["blue_tactic"] in ("stop", "advance", "reverse", "turn_left", "turn_right") for record in steps)
    assert [record["type"] for record in records[len(steps):]] == ["reset"] and harness.session.sim.data.time == 0.0     # R held three frames: one reset
    assert harness.session.previous_action_encoding == "masked_zero_v1" and harness.session.end_reason is None
    assert any("blue=Supervised clone, red=human" in line for line in harness.hud) and not any("PPO" in line for line in harness.hud)


def test_two_presses_of_r_log_two_resets_and_a_second_session_writes_a_second_file(tmp_path):
    needs(REFERENCE); scenarios, index, seed = play.NORMAL; requested = tmp_path / "round.jsonl"
    schedule = START + [(5, ["w"], True), (1, ["r"], True), (3, [], True), (4, ["r"], True), (2, [], True)]
    first = play.Harness(schedule).run(play.arguments("play", str(REFERENCE), scenarios, index, seed, "--input-log", str(requested)))
    records = [json.loads(line) for line in first.log.read_text(encoding="utf-8").splitlines()]
    assert [record["type"] for record in records] == ["step"] * 7 + ["reset", "reset"]              # a one-frame tap and a four-frame hold
    second = play.Harness(START + [(3, ["w"], True)]).run(play.arguments("play", str(REFERENCE), scenarios, index, seed, "--input-log", str(requested)))
    assert second.log != first.log and sorted(path.name for path in tmp_path.iterdir()) == sorted([first.log.name, second.log.name])
    assert len(steps_of(first.log)) == 7 and len(steps_of(second.log)) == 6                          # nothing was appended to the first file


def test_simulated_pace_does_not_depend_on_the_render_rate(tmp_path):
    needs(REFERENCE); scenarios, index, seed = play.NORMAL; totals = {}
    for fps in (20, 60, 144):
        harness = play.Harness(START + [(2 * fps, [], True)], frame_seconds=1.0 / fps).run(
            play.arguments("play", str(REFERENCE), scenarios, index, seed, "--input-log", str(tmp_path / f"pace{fps}.jsonl")))
        steps = steps_of(harness.log); times = [record["time_s"] for record in steps]
        assert all(b - a == pytest.approx(0.02) for a, b in zip(times, times[1:]))                    # the 20 ms step never changes
        totals[fps] = times[-1]; wall = (2 * fps + 2) / fps
        assert -0.02 < times[-1] - wall <= 0.0201
    assert max(totals.values()) - min(totals.values()) <= 0.1                                          # about two simulated seconds at every rate
    blue = {fps: [record["blue_tactic"] for record in steps_of(tmp_path / name)[:100]]
            for fps, name in ((fps, next(p.name for p in tmp_path.iterdir() if p.name.startswith(f"pace{fps}-"))) for fps in totals)}
    assert blue[20] == blue[60] == blue[144]                                                           # same simulation whatever the frame rate


def test_ppo_checkpoints_are_still_labelled_ppo_in_the_hud(tmp_path):
    needs(PPO_CHECKPOINT)
    arguments = ["--config", "config/sim.yaml", "--seed", "10007", "play", "--model", str(PPO_CHECKPOINT), "--profile", "sim_debug",
                 "--eval-scenarios", "runs/ppo-curriculum-overnight/evaluation_scenarios.json", "--scenario-index", "2", "--input-log", str(tmp_path / "ppo.jsonl")]
    harness = play.Harness(START + [(3, [], True)]).run(arguments)
    assert any("blue=PPO, red=human" in line for line in harness.hud)
