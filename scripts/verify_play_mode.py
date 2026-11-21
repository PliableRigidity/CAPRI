"""Automated check of live play (`autobot.sim.cli play`) with SYNTHETIC key input. Simulation-only.

    python scripts/verify_play_mode.py --model runs/combined-clone-balanced-dagger-seed2/final_model.zip --output runs/overnight-2026-11-17/play-verification

The real Pygame loop of `cli.preview` is run headlessly (SDL dummy video driver). Only Pygame's input and the
clocks are replaced: a frame-indexed schedule says which keys are held and whether the window has focus, and the
wall clock advances a fixed amount per frame (one control step by default) so the run is reproducible. Everything
else is the shipped code path: argument parsing, checkpoint loading, the arcade mixer, real-time pacing,
pause/reset handling, round-end detection and the input log.

`--measure-real-time` instead keeps the real wall clock and times a full 20 s timeout at several render rates
(`--real-window` does it once in an actual window).

Synthetic input is not a human. This shows the controls and the log behave as specified; it says nothing about
how the checkpoint fares against a person.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

if "--real-window" not in sys.argv: os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
sys.path[:0] = [str(Path(__file__).resolve().parents[1])]

import numpy as np  # noqa: E402
import pygame  # noqa: E402

from autobot.sim import cli  # noqa: E402
from autobot.sim.env import arcade_drive  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CONFIG = "runs/ppo-curriculum-overnight/effective_sim.yaml"
NORMAL = ("runs/ppo-curriculum-overnight/evaluation_scenarios.json", 2, 10007)     # default_spawn of the frozen suite
DEMO = ("config/finetune_confirmation_ordinary_v1.json", 29, 80017)                 # the known 4.42 s demo start
KEY = {"w": pygame.K_w, "a": pygame.K_a, "s": pygame.K_s, "d": pygame.K_d, "space": pygame.K_SPACE, "r": pygame.K_r}


class Keys:
    def __init__(self, held): self.held = {KEY[name] for name in held}
    def __getitem__(self, key): return key in self.held


class Harness:
    """Feeds a schedule of (frames, keys held, window focused) to the play loop and records what it did."""

    def __init__(self, schedule, driver=None, max_frames=4000, screenshot=None, frame_seconds=0.02, real_fps=None):
        """`frame_seconds`: how far the substituted wall clock moves per frame. `real_fps`: use the real wall clock
        and render at that rate instead (0 keeps the viewer's own 60 frames per second cap)."""
        self.frames = [(set(keys), focused) for count, keys, focused in schedule for _ in range(count)]
        self.driver = driver; self.max_frames = max_frames; self.frame = -1; self.previous = set(); self.session = None
        self.screenshot = screenshot; self.trace = []; self.frame_seconds = frame_seconds; self.real_fps = real_fps
        self.args = None; self.log = None; self.keep_trace = True; self.hud = set()

    def state(self):
        if self.frame < len(self.frames): return self.frames[self.frame]
        if self.driver is not None and self.session is not None and self.session.end_reason is None: return self.driver(self.session), True
        return set(), True

    def events(self):
        self.frame += 1; held, _ = self.state(); events = []
        for name in ("space", "r"):
            if name in held and name not in self.previous: events.append(pygame.event.Event(pygame.KEYDOWN, key=KEY[name]))
        self.previous = set(held)
        done = self.frame >= len(self.frames) and (self.driver is None or self.session is None or self.session.end_reason is not None
                                                    or self.frame >= self.max_frames)
        if done and self.frame >= len(self.frames) + (0 if self.driver is None else 5): events.append(pygame.event.Event(pygame.QUIT))
        if self.session is not None and self.keep_trace:
            self.trace.append({"frame": self.frame, "held": sorted(held), "focused": self.state()[1], "sim_time_s": float(self.session.sim.data.time),
                               "end_reason": self.session.end_reason})
        if self.screenshot and self.frame == self.screenshot[0] and pygame.display.get_surface() is not None:
            pygame.image.save(pygame.display.get_surface(), str(self.screenshot[1]))
        return events

    def run(self, arguments):
        harness = self

        class Session(cli.ReplaySession):
            def __init__(self, args): super().__init__(args); harness.session = self; harness.args = args

        real_clock = pygame.time.Clock

        class Clock:
            def __init__(self): self.real = real_clock() if harness.real_fps is not None else None
            def tick(self, fps=0): return 0 if self.real is None else self.real.tick(harness.real_fps or fps)

        real_font = pygame.font.SysFont

        class Font:                                   # records the HUD text the viewer draws
            def __init__(self, *arguments): self.real = real_font(*arguments)
            def render(self, text, *arguments):
                harness.hud.add(text); return self.real.render(text, *arguments)

        originals = (pygame.event.get, pygame.key.get_pressed, pygame.key.get_focused, pygame.time.Clock, cli.ReplaySession, cli.wall_clock)
        pygame.font.SysFont = Font
        pygame.event.get = self.events; pygame.key.get_pressed = lambda: Keys(self.state()[0])
        pygame.key.get_focused = lambda: self.state()[1]; pygame.time.Clock = Clock; cli.ReplaySession = Session
        if self.real_fps is None: cli.wall_clock = lambda: self.frame * self.frame_seconds
        try: cli.main(arguments)
        finally:
            pygame.event.get, pygame.key.get_pressed, pygame.key.get_focused, pygame.time.Clock, cli.ReplaySession, cli.wall_clock = originals
            pygame.font.SysFont = real_font
        # Each play session writes its own, freshly named input log; this is the one this run produced.
        resolved = getattr(self.args, "resolved_input_log", None); self.log = Path(resolved) if resolved else None
        # The frame that carries the window-close event still runs once; record where the simulation ended up.
        self.trace.append({"frame": self.frame + 1, "held": [], "focused": True, "sim_time_s": float(self.session.sim.data.time),
                           "end_reason": self.session.end_reason})
        return self


def arguments(command, model, scenarios, index, seed, *extra):
    return ["--config", CONFIG, "--seed", str(seed), command, "--model", model, "--profile", "sim_debug", "--action-profile", "experimental5_v1",
            "--eval-scenarios", scenarios, "--scenario-index", str(index), *extra]


def read_log(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()]


def expected_command(held, focused):
    if not focused: return (0.0, 0.0)
    return arcade_drive("w" in held, "s" in held, "a" in held, "d" in held)


def check_controls(model, start, out):
    """Keys held -> red command, neutral on release and on focus loss, pause, reset, time stamps."""
    scenarios, index, seed = start
    schedule = [(2, [], True), (2, ["space"], True),
                (30, ["w"], True), (30, ["w", "a"], True), (30, ["w", "d"], True), (20, [], True), (20, ["s"], True), (15, ["a"], True),
                (15, ["d"], True), (20, ["s", "d"], True), (20, ["w", "a"], False), (10, [], True),
                (2, ["space"], True), (20, ["w"], True),                       # paused: W must do nothing
                (2, ["space"], True), (10, [], True),
                (2, ["r"], True), (5, ["w"], True),                             # reset: paused again, W must do nothing
                (2, ["space"], True), (25, ["w", "d"], True)]
    harness = Harness(schedule, screenshot=(60, out / "play-frame-synthetic.png")).run(
        arguments("play", model, scenarios, index, seed, "--input-log", str(out / "controls-synthetic.jsonl")))
    log = harness.log.resolve(); records = read_log(log); resets = [position for position, record in enumerate(records) if record["type"] == "reset"]
    first = [record for record in records[:resets[0]] if record["type"] == "step"]; second = [record for record in records[resets[-1] + 1:]]
    # Frames that advanced the simulation, in order, with the keys held on that frame (a reset moves time backwards).
    times = [row["sim_time_s"] for row in harness.trace]
    advanced = [harness.trace[position] for position in range(len(times) - 1) if times[position + 1] > times[position]]
    starts = np.cumsum([0] + [count for count, _, _ in schedule])
    frames_of = lambda segment: range(int(starts[segment]), int(starts[segment + 1]))
    expected = [list(expected_command(set(row["held"]), row["focused"])) for row in advanced]
    logged = [record["red_command"] for record in records if record["type"] == "step"]
    segments = {}
    for label, held, focused in (("forward", ["w"], True), ("forward_left_curve", ["a", "w"], True), ("forward_right_curve", ["d", "w"], True),
                                 ("released", [], True), ("reverse", ["s"], True), ("turn_left", ["a"], True), ("turn_right", ["d"], True),
                                 ("reverse_right_curve", ["d", "s"], True), ("focus_lost_with_keys_held", ["a", "w"], False)):
        rows = [position for position, row in enumerate(advanced) if row["held"] == held and row["focused"] == focused]
        segments[label] = {"steps": len(rows), "red_command": sorted({tuple(logged[position]) for position in rows})}
    step_times = [record["time_s"] for record in first]
    paused_rows = [row for row in harness.trace if row["frame"] in frames_of(13)]          # W held while paused
    blue = sorted({record["blue_tactic"] for record in records if record["type"] == "step"})
    return {"input": "synthetic key schedule", "log": str(log.relative_to(ROOT)).replace("\\", "/"), "start": {"scenarios": scenarios, "index": index, "seed": seed},
            "steps_logged": len(logged), "frames_that_advanced_the_simulation": len(advanced),
            "logged_red_command_matches_keys_on_every_step": logged == expected, "by_key_combination": segments,
            "curve_is_between_straight_and_spin": bool(-1.0 < segments["forward_left_curve"]["red_command"][0][0] < 0 and segments["forward_left_curve"]["red_command"][0][1] == -1.0),
            "neutral_on_release": segments["released"]["red_command"] == [(0.0, 0.0)],
            "neutral_on_focus_loss": segments["focus_lost_with_keys_held"]["red_command"] == [(0.0, 0.0)],
            "time_stamps": {"field": "time_s (simulation seconds; the log has no wall-clock time)", "first": step_times[0], "step": round(step_times[1] - step_times[0], 6),
                            "strictly_increasing_by_one_control_step": bool(np.allclose(np.diff(step_times), 0.02))},
            "pause": {"simulation_time_unchanged_while_paused_with_w_held": len({row["sim_time_s"] for row in paused_rows}) == 1 and len(paused_rows) == 20},
            "reset": {"reset_records_for_one_press_held_two_frames": len(resets), "one_reset_per_press": len(resets) == 1,
                      "time_restarts": second[0]["time_s"] == step_times[0],
                      # After R the round is paused: 5 frames of W do nothing. Then 2 SPACE frames, 25 W+D frames and the closing frame step.
                      "w_held_before_space_does_nothing": [tuple(r["red_command"]) for r in second] == [(0.0, 0.0)] * 2 + [(-1.0, -0.615)] * 25 + [(0.0, 0.0)]},
            "hud_names_blue_as": sorted({part.strip() for line in harness.hud for part in line.split("|") if part.strip().startswith("blue=")}),
            "blue_tactics_seen": blue, "screenshot": str((out / "play-frame-synthetic.png").relative_to(ROOT)).replace("\\", "/")}


def blue_is_the_checkpoint(model, start, log):
    """Feed the logged red commands to an independent ReplaySession: the checkpoint must choose the logged blue tactics."""
    scenarios, index, seed = start; records = read_log(log); rounds = []; current = []
    for record in records:
        if record["type"] == "reset":
            if current: rounds.append(current)
            current = []
        else: current.append(record)
    if current: rounds.append(current)
    captured = {}
    original = cli.preview; cli.preview = lambda args: captured.setdefault("args", args)
    try: cli.main(arguments("play", model, scenarios, index, seed))
    finally: cli.preview = original
    session = cli.ReplaySession(captured["args"]); same = 0; total = 0
    for steps in rounds:
        session.reset()
        for record in steps:
            session.step(tuple(record["red_command"])); total += 1
            same += session.current_tactic == record["blue_tactic"] and abs(float(session.sim.data.time) - record["time_s"]) < 1e-9
    return {"rounds": len(rounds), "steps": total, "steps_where_the_checkpoint_reproduces_the_logged_blue_tactic": same,
            "previous_action_encoding": session.previous_action_encoding}


def in_pit(sim, bot):
    position = sim.truth(bot)["position_m"]; arena = sim.cfg["arena"]
    return bool(position[2] < arena["fall_height_m"] or min(math.dist(position[:2], pit) for pit in arena["pit_centres_m"]) < arena["pit_radius_m"])


def check_round_end(model, start, out, name, driver, expect=None):
    """Play to the end of a round, then confirm the loop stops, ignores keys, and restarts on R."""
    scenarios, index, seed = start; requested = out / f"{name}-synthetic.jsonl"
    harness = Harness([(2, [], True), (2, ["space"], True)], driver=driver)
    after = {}
    harness.run(arguments("play", model, scenarios, index, seed, "--input-log", str(requested)))
    log = harness.log.resolve(); session = harness.session; records = read_log(log); steps = [record for record in records if record["type"] == "step"]
    after["end_reason"] = session.end_reason; after["logged_end_reason"] = steps[-1]["end_reason"]; after["steps"] = len(steps)
    after["simulated_seconds"] = steps[-1]["time_s"]
    after["red_in_pit"] = in_pit(session.sim, 1); after["blue_in_pit"] = in_pit(session.sim, 0)
    reason = session.end_reason or ""
    if reason.startswith("Timeout"): matches = not after["red_in_pit"] and not after["blue_in_pit"] and len(steps) == 1000
    elif reason.startswith("Simultaneous"): matches = after["red_in_pit"] and after["blue_in_pit"]
    else: matches = reason.startswith("Pit fall") and ("bot 1" in reason) == after["red_in_pit"] and ("bot 0" in reason) == after["blue_in_pit"]
    after["end_reason_matches_simulator_state"] = bool(matches)
    after["only_the_last_step_carries_an_end_reason"] = all(record["end_reason"] is None for record in steps[:-1]) and steps[-1]["end_reason"] is not None
    if expect: after["expected"] = expect; after["as_expected"] = expect in reason
    # Second and third sessions under the same --input-log name (each gets its own file): reach the end again,
    # then W + SPACE must do nothing, and R must restart.
    count = {}

    class AfterEnd(Harness):
        def state(self):
            if self.session is not None and self.session.end_reason is not None and "ended_at" not in count: count["ended_at"] = self.frame
            if "ended_at" not in count: return super().state()
            since = self.frame - count["ended_at"]
            if since < 20: return {"w"}, True
            if since < 22: return {"w", "space"}, True
            if since < 40: return {"w"}, True
            return set(), True

        def events(self):
            events = [event for event in super().events() if event.type != pygame.QUIT]
            if "ended_at" in count and self.frame - count["ended_at"] >= 45: events.append(pygame.event.Event(pygame.QUIT))
            if self.frame > self.max_frames: events.append(pygame.event.Event(pygame.QUIT))
            return events
    second = AfterEnd([(2, [], True), (2, ["space"], True)], driver=driver).run(arguments("play", model, scenarios, index, seed, "--input-log", str(requested)))
    again = [record for record in read_log(second.log) if record["type"] == "step"]
    after["no_steps_after_round_end_despite_w_and_space"] = len(again) == len(steps) and second.session.end_reason == session.end_reason
    count.clear()

    class Restart(AfterEnd):
        def state(self):
            if self.session is not None and self.session.end_reason is not None and "ended_at" not in count: count["ended_at"] = self.frame
            if "ended_at" not in count: return Harness.state(self)
            since = self.frame - count["ended_at"]
            if 5 <= since < 7: return {"r"}, True
            if 10 <= since < 12: return {"space"}, True
            if 12 <= since < 32: return {"w"}, True
            return set(), True
    third = Restart([(2, [], True), (2, ["space"], True)], driver=driver).run(arguments("play", model, scenarios, index, seed, "--input-log", str(requested)))
    records = read_log(third.log); last_reset = max(position for position, record in enumerate(records) if record["type"] == "reset")
    restarted = records[last_reset + 1:]
    after["one_reset_record_for_the_one_press"] = sum(record["type"] == "reset" for record in records) == 1
    after["each_session_wrote_its_own_log"] = len({log, second.log.resolve(), third.log.resolve()}) == 3 and not requested.exists()
    # After R: two SPACE frames step with no drive key, then 20 frames of W, then released keys until the window closes.
    commands = [tuple(record["red_command"]) for record in restarted]
    after["r_after_round_end_restarts_play"] = bool(restarted[0]["time_s"] == steps[0]["time_s"] and commands[:22] == [(0.0, 0.0)] * 2 + [(-1.0, -1.0)] * 20
                                                    and set(commands[22:]) <= {(0.0, 0.0)} and len(commands) > 22)
    after["log"] = str(log.relative_to(ROOT)).replace("\\", "/"); after["input"] = "synthetic"
    return after


def idle(session): return set()


def drive_into_pit(session):
    """Synthetic red: turn towards the nearest pit, then hold W. A way to end a round on purpose."""
    red = session.sim.truth(1); position = red["position_m"][:2]
    pit = min(session.sim.cfg["arena"]["pit_centres_m"], key=lambda centre: math.dist(position, centre))
    error = math.atan2(pit[1] - position[1], pit[0] - position[0]) - red["yaw_rad"]; error = math.atan2(math.sin(error), math.cos(error))
    if abs(error) < math.radians(10): return {"w"}
    return {"a"} if error > 0 else {"d"}


def check_replay(model, start, expect_seconds=None):
    """The ordinary `replay` command (scripted red), run through the real viewer loop to the end of the round."""
    scenarios, index, seed = start
    harness = Harness([(2, [], True), (2, ["space"], True)], driver=idle).run(arguments("replay", model, scenarios, index, seed, "--opponent", "scripted"))
    session = harness.session
    result = {"end_reason": session.end_reason, "simulated_seconds": round(float(session.sim.data.time), 2), "red": "scripted (not a human)"}
    if expect_seconds is not None: result["matches_the_known_demo"] = bool(session.end_reason == "Pit fall (bot 1)" and abs(session.sim.data.time - expect_seconds) < 0.011)
    return result


def check_frame_rate_independence(model, start, out):
    """Same simulated pace whatever the render rate: a controlled wall clock moves 1/fps per frame for three seconds."""
    scenarios, index, seed = start; rows = {}
    for fps in (20, 30, 60, 144):
        frames = round(3.0 * fps)
        # Red stays idle so the round cannot end inside the three seconds.
        harness = Harness([(2, [], True), (2, ["space"], True), (frames, [], True)], frame_seconds=1.0 / fps).run(
            arguments("play", model, scenarios, index, seed, "--input-log", str(out / f"pace-{fps}fps-synthetic.jsonl")))
        steps = [record for record in read_log(harness.log) if record["type"] == "step"]
        times = [row["sim_time_s"] for row in harness.trace]; per_frame = [round((b - a) / 0.02) for a, b in zip(times, times[1:])]
        wall = (frames + 2) / fps                      # from the frame SPACE went down to the closing frame
        rows[f"{fps}_fps"] = {"frames": frames + 2, "controlled_wall_seconds": round(wall, 4), "steps": len(steps), "simulated_seconds": round(steps[-1]["time_s"], 2),
                              "steps_per_frame": sorted(set(per_frame)), "step_is_20_ms": bool(np.allclose(np.diff([r["time_s"] for r in steps]), 0.02)),
                              "simulated_minus_wall_seconds": round(steps[-1]["time_s"] - wall, 4)}
    return {"input": "synthetic keys, controlled wall clock", "by_render_rate": rows,
            # Simulated time may lead the wall clock by the one step taken at the start, and trail it by less than one step.
            "pace_is_independent_of_render_rate": bool(all(-0.02 < row["simulated_minus_wall_seconds"] <= 0.0201 and row["step_is_20_ms"] for row in rows.values()))}


def check_recorded_replays(model, out):
    """Recorded-red replays must give the same blue tactics and the same ending as when they were recorded."""
    results = {}
    record = ROOT / "runs/overnight-2026-11-17/replays/curved-approach-pushes-blue-in.json"
    if record.is_file():
        saved = json.loads(record.read_text(encoding="utf-8")); red_log = ROOT / saved["red_log"]; expected = [row["blue_tactic"] for row in read_log(red_log)]
        command = saved["command"].split()[3:]; command[command.index("--input-log") + 1] = str(out / "recorded-replay-check.jsonl")
        for label, frame_seconds in (("one_step_per_frame", 0.02), ("30_fps_render", 1.0 / 30)):
            harness = Harness([(2, [], True), (2, ["space"], True)], driver=idle, frame_seconds=frame_seconds).run(list(command))
            steps = [row for row in read_log(harness.log) if row["type"] == "step"]
            results[f"curved_approach_recording_{label}"] = {
                "steps": len(steps), "blue_tactics_reproduced": sum(a["blue_tactic"] == b for a, b in zip(steps, expected)), "recorded_steps": len(expected),
                "end_reason": harness.session.end_reason, "simulated_seconds": round(float(harness.session.sim.data.time), 2),
                "reproduces": bool(len(steps) == len(expected) and all(a["blue_tactic"] == b for a, b in zip(steps, expected))
                                   and harness.session.end_reason == "Pit fall (bot 0)" and saved["replay_check"]["simulated_seconds"] == round(float(harness.session.sim.data.time), 2))}
    human = ROOT / "artifacts/human-vs-final-default.jsonl"; legacy = ROOT / "runs/ppo-curriculum-overnight/final_model.zip"
    if human.is_file() and legacy.is_file():
        recorded = read_log(human); first_round = []
        for row in recorded:
            if row["type"] != "step" or (first_round and row["time_s"] < first_round[-1]["time_s"]): break
            first_round.append(row)
        command = ["--config", "config/sim.yaml", "--seed", "10007", "play", "--model", str(legacy.relative_to(ROOT)), "--profile", "sim_debug",
                   "--eval-scenarios", "runs/ppo-curriculum-overnight/evaluation_scenarios.json", "--scenario-index", "2",
                   "--red-input-replay", str(human.relative_to(ROOT)), "--input-log", str(out / "recorded-human-round-check.jsonl")]

        class FirstRound(Harness):                     # the file holds two rounds; stop once the first has been replayed
            def events(self):
                events = [event for event in super().events() if event.type != pygame.QUIT]
                if self.session is not None and float(self.session.sim.data.time) >= first_round[-1]["time_s"] - 1e-9: events.append(pygame.event.Event(pygame.QUIT))
                if self.session is not None and self.session.end_reason is not None: events.append(pygame.event.Event(pygame.QUIT))
                return events
        harness = FirstRound([(2, [], True), (2, ["space"], True)], driver=idle, max_frames=5000).run(command)
        steps = [row for row in read_log(harness.log) if row["type"] == "step"][:len(first_round)]
        same = sum(a["blue_tactic"] == b["blue_tactic"] and a["red_command"] == b["red_command"] for a, b in zip(steps, first_round))
        results["recorded_human_round_against_the_overnight_checkpoint"] = {"recorded_steps_first_round": len(first_round), "steps_replayed": len(steps),
                                                                              "blue_tactic_and_red_command_reproduced": same,
                                                                              "reproduces": bool(same == len(first_round) == len(steps))}
    results["all_reproduce"] = bool(results and all(entry["reproduces"] for entry in results.values()))
    return results


def measure_real_time(model, start, out, fps, pause_s=0.0):
    """Wall-clock length of a full 20 s timeout (red idle at the normal spawn), with the real clock. A pause is excluded."""
    scenarios, index, seed = start; marks = {}

    class Timed(Harness):
        def state(self):
            now = time.perf_counter(); session = self.session
            if self.frame in (2, 3):
                marks.setdefault("start", now); return {"space"}, True
            if session is None or "start" not in marks: return set(), True
            if session.end_reason is not None:
                marks.setdefault("end", now); return set(), True
            simulated = float(session.sim.data.time)
            if pause_s and simulated >= 8.0 and "pause" not in marks: marks.update(pause=now, pause_frame=self.frame, simulated_at_pause=simulated)
            if "pause" in marks and "resume" not in marks:
                if self.frame - marks["pause_frame"] < 2: return {"space"}, True
                if now - marks["pause"] < pause_s:
                    marks["simulated_while_paused"] = simulated; return set(), True
                marks.update(resume=now, resume_frame=self.frame)
            if "resume" in marks and self.frame - marks["resume_frame"] < 2: return {"space"}, True
            return set(), True
    harness = Timed([], driver=idle, max_frames=10 ** 7, real_fps=fps); harness.keep_trace = False
    harness.run(arguments("play", model, scenarios, index, seed, "--input-log", str(out / "real-time-synthetic.jsonl")))
    steps = [record for record in read_log(harness.log) if record["type"] == "step"]
    paused = marks.get("resume", 0.0) - marks.get("pause", 0.0); total = marks["end"] - marks["start"]; running = total - paused
    result = {"render_rate": "viewer default (60 fps cap)" if not fps else f"{fps} fps", "video_driver": os.environ.get("SDL_VIDEODRIVER", "real window"),
              "end_reason": harness.session.end_reason, "control_steps": len(steps), "simulated_seconds": round(steps[-1]["time_s"], 2),
              "frames": harness.frame, "measured_frames_per_second": round(harness.frame / (marks["end"] - marks["start"] + 1e-9), 1),
              "wall_seconds_from_start_to_round_end": round(total, 3), "wall_seconds_paused": round(paused, 3),
              "wall_seconds_running": round(running, 3), "simulated_over_running_wall": round(steps[-1]["time_s"] / running, 4)}
    if pause_s: result["simulated_time_frozen_during_the_pause"] = marks.get("simulated_while_paused") == marks["simulated_at_pause"]
    result["about_20_real_seconds"] = bool(abs(running - 20.0) <= 0.25 and len(steps) == 1000 and result["end_reason"].startswith("Timeout"))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True); parser.add_argument("--output", required=True)
    parser.add_argument("--name", default="play-verification")
    parser.add_argument("--measure-real-time", action="store_true", help="time a full 20 s timeout with the real wall clock at several render rates")
    parser.add_argument("--real-window", action="store_true", help="with --measure-real-time: one measurement in an actual window instead of the headless driver")
    args = parser.parse_args()
    out = (ROOT / args.output) if not Path(args.output).is_absolute() else Path(args.output); out.mkdir(parents=True, exist_ok=True)
    model = args.model
    if args.measure_real_time:
        runs = [(0, 0.0)] if args.real_window else [(0, 0.0), (30, 0.0), (144, 0.0), (20, 0.0), (0, 3.0)]
        name = "real-time-measurement-window" if args.real_window else "real-time-measurement"
        report = {"purpose": "Wall-clock length of a full 20 s timeout in live play (red idle, synthetic SPACE presses). Excludes pauses.",
                  "model": model, "config": CONFIG, "start": {"scenarios": NORMAL[0], "index": NORMAL[1], "seed": NORMAL[2]}, "measurements": []}
        for fps, pause_s in runs:
            report["measurements"].append(measure_real_time(model, NORMAL, out, fps, pause_s)); print(json.dumps(report["measurements"][-1]), flush=True)
        report["passed"] = bool(all(row["about_20_real_seconds"] and row.get("simulated_time_frozen_during_the_pause", True) for row in report["measurements"]))
        (out / f"{name}.json").write_text(json.dumps(report, indent=1), encoding="utf-8"); print("PASSED" if report["passed"] else "FAILED")
        return
    report = {"purpose": "Automated check of live play with synthetic key input. Not evidence about play against a human.",
              "model": model, "config": CONFIG, "loaded": cli.checkpoint_observation_metadata(model),
              "hud_blue_label": cli.blue_controller_label(model),
              "controls_normal_spawn": check_controls(model, NORMAL, out)}
    report["blue_uses_the_checkpoint_normal_spawn"] = blue_is_the_checkpoint(model, NORMAL, ROOT / report["controls_normal_spawn"]["log"])
    report["round_end_red_idle_normal_spawn"] = check_round_end(model, NORMAL, out, "normal-spawn-red-idle", idle)
    report["round_end_red_drives_into_pit_normal_spawn"] = check_round_end(model, NORMAL, out, "normal-spawn-red-into-pit", drive_into_pit)
    report["round_end_red_idle_demo_start"] = check_round_end(model, DEMO, out, "demo-start-red-idle", idle)
    report["blue_uses_the_checkpoint_demo_start"] = blue_is_the_checkpoint(model, DEMO, ROOT / report["round_end_red_idle_demo_start"]["log"])
    report["known_demo_replay_scripted_red"] = check_replay(model, DEMO, expect_seconds=4.42)
    report["normal_spawn_replay_scripted_red"] = check_replay(model, NORMAL)
    report["frame_rate_independence"] = check_frame_rate_independence(model, NORMAL, out)
    report["recorded_replays"] = check_recorded_replays(model, out)
    flags = [report["controls_normal_spawn"][key] for key in ("logged_red_command_matches_keys_on_every_step", "curve_is_between_straight_and_spin",
                                                              "neutral_on_release", "neutral_on_focus_loss")]
    flags += [report["controls_normal_spawn"]["time_stamps"]["strictly_increasing_by_one_control_step"],
              report["controls_normal_spawn"]["pause"]["simulation_time_unchanged_while_paused_with_w_held"],
              report["controls_normal_spawn"]["reset"]["time_restarts"], report["controls_normal_spawn"]["reset"]["w_held_before_space_does_nothing"],
              report["controls_normal_spawn"]["reset"]["one_reset_per_press"], report["hud_blue_label"] == "Supervised clone",
              report["controls_normal_spawn"]["hud_names_blue_as"] == ["blue=Supervised clone, red=human"],
              report["frame_rate_independence"]["pace_is_independent_of_render_rate"], report["recorded_replays"]["all_reproduce"]]
    for key in ("blue_uses_the_checkpoint_normal_spawn", "blue_uses_the_checkpoint_demo_start"):
        flags.append(report[key]["steps"] == report[key]["steps_where_the_checkpoint_reproduces_the_logged_blue_tactic"] and report[key]["steps"] > 0)
    for key in ("round_end_red_idle_normal_spawn", "round_end_red_drives_into_pit_normal_spawn", "round_end_red_idle_demo_start"):
        flags += [report[key][name] for name in ("end_reason_matches_simulator_state", "only_the_last_step_carries_an_end_reason",
                                                 "no_steps_after_round_end_despite_w_and_space", "r_after_round_end_restarts_play",
                                                 "one_reset_record_for_the_one_press", "each_session_wrote_its_own_log")]
    report["passed"] = bool(all(flags))
    (out / f"{args.name}.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps(report, indent=1)); print("PASSED" if report["passed"] else "FAILED")


if __name__ == "__main__":
    main()
