"""LIVE SHADOW: the checkpoint proposes an action from the real camera and the real IMU. Nothing is sent.

Read-only. This program opens the camera and listens to the local telemetry relay. It has no
connection to the robot and no way to command it: proposals go to the screen and a trace file.

Start the link owner first, then this:

    python scripts\\telemetry_link.py --wifi
    python -m control_center.tools.live_shadow --camera 1 --config config\\home_test_overlay.yaml --profile home_test

Check the IMU axes without the camera (do this before the first shadow run):

    python -m control_center.tools.live_shadow --imu-check

It reuses the existing pieces unchanged: the observation contract, ``CameraPipeline`` (ArUco own pose +
blob opponent), ``ShadowAssembler`` and ``CheckpointPolicy`` from ``stream_adapter``, and the camera
opener of ``arena_overlay``. What is new here is only the live IMU source and the loop around them.

A proposal appears only when every required input is valid and fresh. Otherwise the screen names the
blocked input and why. No value is ever filled in from the simulator, a fixture or an earlier frame.
"""

from __future__ import annotations

import argparse
import json
import math
import threading
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable

import cv2
import numpy as np

from control_center.backend.observation_builder import SOURCE_LIVE, SOURCE_SIMULATED
from control_center.tools.live_telemetry import (DEFAULT_MOUNTING, RELAY_PORT, LiveTelemetry, TapSubscriber, describe_motion,
                                                 load_mounting)
from control_center.tools.observation_contract import RECOMMENDED_CHECKPOINT, ObservationContract, load_contract
from control_center.tools.arena_setup import with_arena_setup
from control_center.tools.overlay_core import BotPose, ClickSet, OverlayConfig, load_clicks, load_config, profile_points_path
from control_center.tools.stream_adapter import CameraPipeline, CameraSample, CheckpointPolicy, Decision, ShadowAssembler, fit_profile

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = REPO_ROOT / "artifacts" / "live-shadow"
LIVE_LABEL = "LIVE SHADOW"
WINDOW = "Capri LIVE SHADOW (read-only, nothing is sent)"
PLAUSIBLE_SPEED_M_S = 2.0


class TagSwitchGuard:
    """A step in own position that coincides with a change of the tags in use is not movement.

    With correct ``bot.markers`` offsets every tag gives the same bot centre. If the centre jumps at
    the moment the visible tags change, the configured layout is wrong: that frame's pose is refused
    and the fault is kept on screen.
    """

    def __init__(self, tolerance_m: float) -> None:
        self.tolerance_m = tolerance_m
        self.last: tuple[float, BotPose] | None = None
        self.trips = 0
        self.message = ""

    def check(self, pose: BotPose, t: float) -> str:
        """"" if the pose can be used, else the reason it cannot."""
        if not pose.ok:
            return ""
        reason = ""
        if self.last is not None:
            then, previous = self.last
            dt = t - then
            if 0 <= dt <= 0.5 and previous.used_ids != pose.used_ids:
                step = math.hypot(pose.x_m - previous.x_m, pose.y_m - previous.y_m)
                if step > self.tolerance_m + PLAUSIBLE_SPEED_M_S * dt:
                    self.trips += 1
                    self.message = (f"tag switch: position stepped {1000 * step:.0f} mm when the tags in use changed from "
                                    f"{previous.used_ids} to {pose.used_ids}; the tag offsets in bot.markers disagree (not movement)")
                    reason = self.message
        self.last = (t, pose)
        return reason


class LiveShadow:
    """Camera frame + telemetry -> decision. No window, no camera, no socket: those are the caller's."""

    def __init__(self, contract: ObservationContract, config: OverlayConfig, clicks: ClickSet, telemetry: LiveTelemetry,
                 input_source: str, shadow_label: str = LIVE_LABEL, multi_marker: bool = False,
                 click_opponent: bool = False, camera_origin: str = SOURCE_SIMULATED, imu_origin: str | None = None) -> None:
        # camera_origin / imu_origin: LIVE only for a real camera and a real controller (run_camera); everything else, including
        # every test with synthetic frames, stays SIMULATED and therefore never valid_for_control.
        self.camera_origin, self.imu_origin = camera_origin, imu_origin
        self.contract, self.config, self.clicks, self.telemetry = contract, config, clicks, telemetry
        self.input_source, self.shadow_label = input_source, shadow_label
        self.multi_marker, self.click_opponent = multi_marker, click_opponent      # opt-in trackers; off = as before
        self.pipeline: CameraPipeline | None = None
        self.assembler: ShadowAssembler | None = None
        self.guard = TagSwitchGuard(config.max_position_disagreement_m)
        self.policy_note = "checkpoint not loaded"
        self.map_note = ""

    def start(self, image_size: tuple[int, int]) -> None:
        mapping, camera, _quality = fit_profile(self.config, self.clicks, image_size)
        if self.multi_marker or self.click_opponent:
            self.pipeline = CameraPipeline(self.config, mapping, camera, multi_marker=self.multi_marker, click_opponent=self.click_opponent)
        else:
            self.pipeline = CameraPipeline(self.config, mapping, camera)
        self.assembler = ShadowAssembler(self.contract, self.input_source, self.telemetry, policy=None,
                                         shadow_label=self.shadow_label, max_camera_age_s=self.config.max_detection_age_s,
                                         camera_origin=self.camera_origin, imu_origin=self.imu_origin)
        self.map_note = "" if mapping.valid else f"NO FLOOR MAP ({mapping.reason}): calibrate first"

    def set_policy(self, policy: CheckpointPolicy | None, note: str) -> None:
        self.policy_note = note
        if self.assembler is not None:
            self.assembler.policy = policy

    def take_reference(self, frames: list[np.ndarray], note: str) -> None:
        self.pipeline.set_reference(frames, note)

    def step(self, frame: np.ndarray, t: float, now: float | None = None) -> tuple[CameraSample, Decision]:
        if self.pipeline is None:
            self.start((frame.shape[1], frame.shape[0]))
        sample = self.pipeline.process(frame, t)
        # With the chassis tracker only a pose MEASURED in this frame is put to the tag-switch guard;
        # a predicted pose rests on the measurement that already passed it.
        checked = sample.own_track is None or sample.own_track["state"] == "measured"
        refused = self.guard.check(sample.own, t) if checked else ""
        if refused:
            sample.own = BotPose(reason=refused, visible_ids=sample.own.visible_ids)
            sample.own_at = None                            # the refusal stands at decision time too
        self.assembler.offer(sample)
        decision = self.assembler.decide(t if now is None else max(t, now))
        return sample, decision

    def lines(self, decision: Decision, now: float) -> list[tuple[str, str]]:
        """Everything the screen shows for one decision."""
        out = [(f"CHECKPOINT {self.contract.identifier}  ({self.contract.training.split('.')[0][:70]})", "note"),
               (f"           {self.policy_note}", "note")]
        out += decision.ui_lines()[1:]
        if decision.blocked_by:
            out.append(("BLOCKED BY:", "bad"))
            out += [(f"  {entry}", "bad") for entry in decision.blocked_by]
        elif decision.proposed is None:
            out.append((f"all inputs valid; no proposal because the {self.policy_note}", "note"))
        out += self.telemetry.lines(now)
        if self.pipeline is not None and self.assembler is not None and self.assembler.sample is not None and (self.multi_marker or self.click_opponent):
            from control_center.tools.live_track import tracking_lines
            out += tracking_lines(self.assembler.sample)
        if self.guard.trips:
            out.append((f"TAG LAYOUT FAULT x{self.guard.trips}: {self.guard.message}", "bad"))
        if self.config.placeholders:
            out.append(("PLACEHOLDER config values in use: " + "; ".join(self.config.placeholders), "bad"))
        if self.map_note:
            out.append((self.map_note, "bad"))
        out.append(("Home patch positions are read in the simulator's arena frame (origin = plain corner).", "note"))
        out.append(("Output: screen and trace only. There is no connection to the robot.", "note"))
        return out


def trace_record(decision: Decision, telemetry: LiveTelemetry, now: float) -> dict[str, Any]:
    record = decision.to_trace()
    reading = telemetry.reading
    record["imu_line"] = None if reading is None else {
        "sequence": reading.sequence, "device_us": reading.device_us,
        "pc_receipt_age_ms": None if telemetry.reading_at is None else round(1000 * (now - telemetry.reading_at), 1),
        "clock_note": "device_us is the controller's clock; ages use PC receipt time; the clocks are not synchronised"}
    record["controller"] = telemetry.controller_line(now)[0]
    record["counters"] = {"valid": telemetry.valid_lines, "failed": telemetry.failure_lines,
                          "late_or_repeated": telemetry.out_of_order, "gaps": telemetry.gaps}
    return record


def run_frames(shadow: LiveShadow, frames: Iterable[tuple[float, np.ndarray]],
               before_frame: Callable[[float], None] | None = None) -> list[Decision]:
    """Drive the same loop the camera runner uses from any frame source (tests use synthetic frames)."""
    decisions = []
    for t, frame in frames:
        if before_frame is not None:
            before_frame(t)
        decisions.append(shadow.step(frame, t)[1])
    return decisions


# ---------------------------------------------------------------- the two commands


def imu_check(args: argparse.Namespace) -> int:
    """Console only: no camera, no checkpoint. Shows what the transform makes of the live IMU."""
    mounting = load_mounting(args.mounting)
    telemetry = LiveTelemetry(mounting, f"relay 127.0.0.1:{args.relay_port}")
    tap = TapSubscriber(args.relay_port)
    print(f"IMU axis check. Mounting file: {mounting.path}  verified: {mounting.verified}")
    print("sensor_to_body:", mounting.matrix.astype(int).tolist())
    print("Do, in turn: (1) flat and still, (2) turn it left by hand, (3) lift the nose, (4) lift the left side.")
    print("Each line below says what the readings MEAN in the body frame. It must match what you did. Ctrl+C to stop.\n")
    shown = float("-inf")
    try:
        while True:
            now = time.monotonic()
            for direction, line, _ in tap.poll(now):
                telemetry.feed(direction, line, now)
            if now - shown >= 0.5:
                shown = now
                telemetry.latest(now)
                values = telemetry.body_values()
                fresh = telemetry.reading_at is not None and now - telemetry.reading_at < 0.2
                failed = telemetry.failure_at is not None and (telemetry.reading_at is None or telemetry.failure_at >= telemetry.reading_at)
                if values is None or not fresh or failed:
                    print(f"{time.strftime('%H:%M:%S')}  NO VALID IMU SAMPLE: {telemetry.description}")
                else:
                    gyro, accel = values
                    print(f"{time.strftime('%H:%M:%S')}  body accel {accel[0]:+6.2f} {accel[1]:+6.2f} {accel[2]:+6.2f} m/s^2   "
                          f"gyro {math.degrees(gyro[0]):+7.1f} {math.degrees(gyro[1]):+7.1f} {math.degrees(gyro[2]):+7.1f} deg/s   "
                          + " | ".join(describe_motion(gyro, accel)))
            time.sleep(0.01)
    except KeyboardInterrupt:
        return 0
    finally:
        tap.close()


def run_camera(args: argparse.Namespace) -> dict[str, Any]:
    from control_center.tools import arena_overlay
    from control_center.tools.shadow_stream_demo import _override_markers, annotate, panel

    contract = load_contract(args.checkpoint)
    config = _override_markers(load_config(args.config), args.dictionary, args.tag_size_mm)
    points_path = Path(args.points) if args.points else profile_points_path(args.profile)
    clicks = load_clicks(points_path) if points_path.exists() else ClickSet()
    if clicks.historical:                                   # checked before any camera is opened
        raise ValueError(f"{points_path} is a HISTORICAL footage profile; it is never used with a live camera")
    if not args.points:                                         # an explicit --points file bypasses the profile on purpose
        config = with_arena_setup(config, args.profile, clicks)       # the profile's saved arena and hazards, if it has any
    if config.arena.get("policy_frame_ok") is False:
        raise ValueError("the saved arena layout is not the simulator's, so positions are not in the frame the policy was trained on: "
                         "live shadow is refused. Check the pits and flipper in the ARENA tab.")
    mounting = load_mounting(args.mounting)
    telemetry = LiveTelemetry(mounting, f"relay 127.0.0.1:{args.relay_port}")
    tap = TapSubscriber(args.relay_port)
    source = arena_overlay.CameraSource(args.camera, arena_overlay.camera_settings(args))
    input_source = f"LIVE camera {source.description}, profile '{args.profile}' + LIVE telemetry via the local relay"
    multi_marker, click_opponent = bool(getattr(args, "multi_marker", False)), bool(getattr(args, "click_opponent", False))
    if multi_marker or click_opponent:
        shadow = LiveShadow(contract, config, clicks, telemetry, input_source, multi_marker=multi_marker, click_opponent=click_opponent,
                            camera_origin=SOURCE_LIVE, imu_origin=SOURCE_LIVE)
    else:
        shadow = LiveShadow(contract, config, clicks, telemetry, input_source, camera_origin=SOURCE_LIVE, imu_origin=SOURCE_LIVE)

    policy = CheckpointPolicy(contract)
    shadow.set_policy(None, "checkpoint loading (about a minute)...")
    loaded: dict[str, Any] = {}

    def load() -> None:
        try:
            policy.load()
            loaded["ok"] = True
        except Exception as error:                          # shown on screen; shadow stays without proposals
            loaded["error"] = str(error)

    threading.Thread(target=load, name="checkpoint-load", daemon=True).start()

    out_dir = Path(args.out) / datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    collecting: list[np.ndarray] | None = None
    message = "EMPTY THE FLOOR (no objects, no hands), then press SPACE"
    modes: Counter = Counter()
    blocked: Counter = Counter()
    proposals: Counter = Counter()
    frames_seen = 0
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    queue = None
    if click_opponent:                                      # opt-in: left-click selects the opponent, c clears it
        from control_center.tools.live_track import ClickQueue, apply_clicks
        queue = ClickQueue()
        cv2.setMouseCallback(WINDOW, queue.on_mouse)
        message = "LEFT-CLICK the opponent in the picture (no empty-floor reference is needed)"
    try:
        with (out_dir / "trace.jsonl").open("w", encoding="utf-8") as trace:
            while True:
                ok, frame, t = source.read()
                if frame is None:
                    if not ok:
                        message = "camera stopped returning frames"
                        break
                    continue
                frames_seen += 1
                now = time.monotonic()
                for direction, line, _ in tap.poll(now):
                    telemetry.feed(direction, line, now)
                if "ok" in loaded and shadow.assembler is not None and shadow.assembler.policy is None:
                    shadow.set_policy(policy, f"checkpoint loaded in {policy.load_seconds:.0f} s")
                elif "error" in loaded:
                    shadow.policy_note = f"checkpoint FAILED to load: {loaded['error']}"
                if collecting is not None:
                    collecting.append(frame.copy())
                    if len(collecting) >= 15:
                        try:
                            shadow.take_reference(collecting, "median of 15 live frames taken when SPACE was pressed")
                            message = "reference taken: put the objects on the floor"
                        except ValueError as error:
                            message = f"reference NOT taken: {error}"
                        collecting = None
                if queue is not None:
                    if shadow.pipeline is None:
                        shadow.start((frame.shape[1], frame.shape[0]))
                    queue.frame_width = frame.shape[1]
                    for event in apply_clicks(shadow.pipeline, queue, frame, t):
                        message = ("opponent selected: " if event["accepted"] else "click REFUSED: ") + event["reason"]
                sample, decision = shadow.step(frame, t, time.monotonic())
                modes[decision.mode] += 1
                for entry in decision.blocked_by:
                    blocked[entry.split(":")[0]] += 1
                if decision.proposed is not None:
                    proposals[decision.proposed["tactic"]] += 1
                if frames_seen <= 30000:
                    record = trace_record(decision, telemetry, now)
                    if sample.own_track is not None:            # only with --multi-marker / --click-opponent
                        record["own_track"] = sample.own_track
                    if sample.opponent_track is not None:
                        record["opponent_track"] = sample.opponent_track
                    trace.write(json.dumps(record) + "\n")
                lines = shadow.lines(decision, now) + [
                    (f"markers decoded: {sample.marker_ids or 'none'}  dictionary {config.dictionary}   blobs: {sample.blob_count}", "note"),
                    (message, "note"),
                    ("LEFT-CLICK opponent   c clear   q quit" if click_opponent else "SPACE empty-floor reference   r again   q quit", "note")]
                if queue is not None:
                    queue.shown = (frame, t)
                view = np.hstack([annotate(frame, shadow.pipeline, sample), panel(lines, 640, frame.shape[0], "LIVE SHADOW - read-only, nothing is sent")])
                cv2.imshow(WINDOW, view)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27) or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                    break
                if click_opponent and key == ord("c"):
                    shadow.pipeline.clear_opponent()
                    message = "opponent selection cleared: LEFT-CLICK the opponent"
                if key in (32, ord("r")) and not click_opponent:
                    collecting, message = [], "taking the reference picture: keep the floor empty"
                if args.max_frames is not None and frames_seen >= args.max_frames:
                    break
    finally:
        source.close()
        tap.close()
        cv2.destroyAllWindows()
    summary = {"input_source": input_source, "checkpoint": contract.identifier, "frames": frames_seen, "modes": dict(modes),
               "proposals": dict(proposals), "blocked_by": dict(blocked), "tag_layout_faults": shadow.guard.trips,
               "imu_lines": {"valid": telemetry.valid_lines, "failed": telemetry.failure_lines,
                             "late_or_repeated": telemetry.out_of_order, "gaps": telemetry.gaps},
               "mounting_verified": mounting.verified, "trace": str(out_dir / "trace.jsonl"), "last_message": message,
               "output": "screen and trace only; this program has no connection to the robot"}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Read-only live shadow: the checkpoint proposes from the real camera and IMU. Nothing is sent.")
    parser.add_argument("--imu-check", action="store_true", help="console axis check of the live IMU; no camera, no checkpoint")
    parser.add_argument("--camera", type=int, help="camera index (the C525 has been 1)")
    parser.add_argument("--config", default=str(REPO_ROOT / "config" / "home_test_overlay.yaml"), help="overlay config with the arena and the tag layout")
    parser.add_argument("--profile", default="home_test", help="calibration profile (home_test, or live at the venue)")
    parser.add_argument("--points", help="clicked-points file, instead of the profile's")
    parser.add_argument("--checkpoint", default=str(RECOMMENDED_CHECKPOINT))
    parser.add_argument("--mounting", default=str(DEFAULT_MOUNTING), help="IMU mounting file")
    parser.add_argument("--relay-port", type=int, default=RELAY_PORT, help="local telemetry relay of scripts\\telemetry_link.py")
    parser.add_argument("--dictionary", help="override bot.dictionary, e.g. DICT_4X4_50")
    parser.add_argument("--tag-size-mm", type=float, help="override the tag's black-square side")
    parser.add_argument("--width", type=int)
    parser.add_argument("--height", type=int)
    parser.add_argument("--fps", type=float)
    parser.add_argument("--backend")
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--multi-marker", action="store_true",
                        help="own pose from the multi-marker chassis tracker (consensus, short prediction, MEASURED/PREDICTED/STALE/LOST)")
    parser.add_argument("--click-opponent", action="store_true",
                        help="left-click the opponent in the picture to select it (c clears); no empty-floor reference is needed")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.imu_check:
        return imu_check(args)
    if args.camera is None:
        build_parser().error("--camera N is required (or use --imu-check)")
    summary = run_camera(args)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
