"""Shadow-stream demo: what would the trained checkpoint receive from a camera-like stream? READ-ONLY.

    python -m control_center.tools.shadow_stream_demo --footage <the Bristol mp4>

It runs three things and writes everything to artifacts/shadow-stream/:

  1. CONTRACT   the checkpoint's observation fields, units, frames and where each could come from live.
  2. FOOTAGE    the Bristol recording (historical profile only), fed one frame at a time in timestamp
                order. The blob tracker's empty-floor picture is taken from frames BEFORE the clip.
  3. FIXTURE    a SYNTHETIC tagged bot and an unmarked object: tag dropouts, touching, a camera stall.
                Run twice over the same frames: camera only, then with a SYNTHETIC stand-in IMU stream,
                which is the only run in which the checkpoint is actually asked for a tactic.

Nothing here opens a serial port, the controller link or a real camera, and no proposal is sent anywhere.
A real camera is opened only when YOU pass --camera N (the five-minute home test):

    python -m control_center.tools.shadow_stream_demo --camera 0 --config config/home_test_overlay.yaml --profile home_test

Window keys (camera mode): SPACE take the empty-floor reference picture, r take it again, q quit.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator

import cv2
import numpy as np

from control_center.tools.blob_tracker import LOST, OVERLAP, TRACKED
from control_center.tools.observation_contract import (GROUP_IMU, GRAVITY_M_S2,
                                                       RECOMMENDED_CHECKPOINT, ContractError, ObservationContract,
                                                       format_contract, load_contract)
from control_center.tools.arena_setup import with_arena_setup
from control_center.tools.overlay_core import (DEFAULT_CONFIG_PATH, LIVE_PROFILE, ClickSet, OverlayConfig, densify,
                                               load_clicks, load_config, profile_points_path)
from control_center.tools.overlay_fixture import (FIXTURE_LABEL, PLATE_SIZE_M, BotInstance, FixtureCamera, Scene,
                                                  default_camera, fixture_clicks, render_frame)
from control_center.tools.stream_adapter import (CameraPipeline, CameraSample, CheckpointPolicy, Decision,
                                                 ImuSample, NoTelemetry, RecordedTelemetry, ShadowAssembler, fit_profile)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = REPO_ROOT / "artifacts" / "shadow-stream"
HISTORICAL_PROFILE = "historical_bristol_2026_03_22"
FOOTAGE_ENV = "AUTOBOTS_FOOTAGE_MP4"
CLIP_START_S, CLIP_DURATION_S = 433.0, 24.0
REFERENCE_AT_S, REFERENCE_FRAMES = 330.0, 30       # one second of frames, 102 s before the clip starts
WINDOW = "Capri shadow stream (read-only)"

FIXTURE_SHADOW = "FIXTURE SHADOW (synthetic frames + synthetic IMU)"
FOOTAGE_SOURCE = "recorded footage: Bristol 22 March 2026 (historical profile, approximate map)"
FIXTURE_CAMERA_ONLY = "SYNTHETIC fixture frames, camera only (no telemetry)"
FIXTURE_WITH_IMU = "SYNTHETIC fixture frames + SYNTHETIC stand-in IMU (not live, not the simulator)"


# ----------------------------------------------------------------------------------------------
# The causal loop: frames in timestamp order, decisions at the control rate


@dataclass
class StreamRun:
    name: str
    input_source: str
    samples: list[CameraSample] = field(default_factory=list)
    decisions: dict[str, list[Decision]] = field(default_factory=dict)      # assembler name -> decisions
    events: list[dict[str, Any]] = field(default_factory=list)
    frame_period_s: float = 1 / 30
    notes: dict[str, Any] = field(default_factory=dict)


def run_stream(name: str, frames: Iterable[tuple[float, np.ndarray | None]], pipeline: CameraPipeline,
               assemblers: dict[str, ShadowAssembler], control_hz: float, end_t: float | None = None,
               assumed_latency_s: float = 0.0,
               on_frame: Callable[[np.ndarray, CameraSample], None] | None = None,
               on_decision: Callable[[str, Decision], None] | None = None) -> StreamRun:
    """Feed frames one at a time. A decision at time T sees only frames with timestamp + latency <= T.

    A frame of ``None`` is a frame the camera did not deliver (a stall): time passes, nothing is offered.
    """
    run = StreamRun(name, next(iter(assemblers.values())).input_source if assemblers else "")
    run.decisions = {key: [] for key in assemblers}
    period = 1.0 / control_hz
    tick_t: float | None = None
    last_t: float | None = None

    def decide_until(limit: float, inclusive: bool) -> None:
        nonlocal tick_t
        while tick_t is not None and (tick_t < limit - 1e-9 or (inclusive and tick_t <= limit + 1e-9)):
            for key, assembler in assemblers.items():
                decision = assembler.decide(tick_t)
                run.decisions[key].append(decision)
                if on_decision is not None:
                    on_decision(key, decision)
            tick_t += period

    for t, frame in frames:
        if last_t is not None and t < last_t:
            raise ValueError("frames must be fed in timestamp order")
        last_t = t
        if frame is None:
            continue
        available = t + assumed_latency_s
        if tick_t is None:
            tick_t = available                       # the first decision is made when the first frame is in hand
        decide_until(available, inclusive=False)     # these decisions were made before this frame existed
        sample = pipeline.process(frame, t)
        run.samples.append(sample)
        for assembler in assemblers.values():
            assembler.offer(sample)
        if on_frame is not None:
            on_frame(frame, sample)
    if tick_t is not None:
        decide_until(last_t + assumed_latency_s if end_t is None else end_t, inclusive=True)
    if pipeline.tracker is not None:
        run.events = list(pipeline.tracker.events)
    return run


# ----------------------------------------------------------------------------------------------
# Summaries


def _percentiles(values: list[float]) -> dict[str, float]:
    if not values:
        return {"median": 0.0, "p95": 0.0, "max": 0.0}
    array = np.asarray(values)
    return {"median": round(float(np.median(array)), 2), "p95": round(float(np.percentile(array, 95)), 2),
            "max": round(float(array.max()), 2)}


def _runs_of(flags: list[bool]) -> list[tuple[int, int]]:
    """[start, end) index ranges where flags are True."""
    runs, start = [], None
    for index, flag in enumerate(flags + [False]):
        if flag and start is None:
            start = index
        elif not flag and start is not None:
            runs.append((start, index))
            start = None
    return runs


def summarise(run: StreamRun) -> dict[str, Any]:
    samples = run.samples
    count = len(samples)
    stages = ("markers", "pose", "blobs", "opponent", "total")
    # The first frame pays one-off warm-up costs (OpenCV thread pools, caches); report it apart.
    timing = {stage: _percentiles([s.timing_ms[stage] for s in samples[1:]]) for stage in stages}
    period_ms = 1000 * run.frame_period_s
    over = sum(1 for s in samples[1:] if s.timing_ms["total"] > period_ms)
    own_ok = [s.own.ok for s in samples]
    opponent_ok = [s.opponent.ok for s in samples]
    overlap = [any(track["state"] == OVERLAP for track in s.tracks) for s in samples]
    summary: dict[str, Any] = {
        "name": run.name, "input_source": run.input_source, "frames": count,
        "stream_seconds": round(samples[-1].t - samples[0].t + run.frame_period_s, 3) if samples else 0.0,
        "processing_ms_per_frame": timing,
        "first_frame_ms": round(samples[0].timing_ms["total"], 2) if samples else None,
        "frames_slower_than_one_frame_period": {"count": over, "frame_period_ms": round(period_ms, 2)},
        "processing_note": ("time spent in this software on this computer, per frame, single thread of control. It is NOT "
                            "camera latency (exposure, USB transfer, driver buffering), which a file or a rendered "
                            "picture cannot measure"),
        "own_pose": {"valid_frames": sum(own_ok), "missing_frames": count - sum(own_ok),
                     "missing_reasons": dict(Counter(_short(s.own.reason) for s in samples if not s.own.ok)),
                     "markers_decoded_frames": sum(1 for s in samples if s.marker_ids)},
        "opponent": {"valid_frames": sum(opponent_ok), "unknown_frames": count - sum(opponent_ok),
                     "unknown_reasons": dict(Counter(_short(s.opponent.reason) for s in samples if not s.opponent.ok))},
        "blob_tracks": {"ids_seen": sorted({track["id"] for s in samples for track in s.tracks}),
                        "frames_with_an_overlap": sum(overlap),
                        "overlap_spans_s": [[round(samples[a].t, 2), round(samples[b - 1].t, 2)] for a, b in _runs_of(overlap)],
                        "frames_with_a_lost_track": sum(1 for s in samples if any(t["state"] == LOST for t in s.tracks)),
                        "positions_reported_during_overlap_or_loss": sum(
                            1 for s in samples for t in s.tracks if t["state"] != TRACKED and t["x_m"] is not None),
                        "events": dict(Counter(event["event"] for event in run.events)),
                        "identity_uncertain_tracks": sorted({t["id"] for s in samples for t in s.tracks if t["identity_uncertain"]})},
        "decisions": {}, "notes": run.notes}
    for key, decisions in run.decisions.items():
        blocked = Counter()
        for decision in decisions:
            for group, status in decision.inputs.items():
                if not status.valid:
                    blocked[f"[{status.indices}] {group}"] += 1
        proposals = [d for d in decisions if d.proposed is not None]
        ages = [1000 * (d.t - d.frame_t) for d in decisions if d.frame_t is not None]
        summary["decisions"][key] = {
            "input_source": decisions[0].input_source if decisions else "", "count": len(decisions),
            "modes": dict(Counter(d.mode for d in decisions)),
            "complete_observations": sum(1 for d in decisions if d.observation is not None),
            "proposals": len(proposals), "proposed_tactics": dict(Counter(d.proposed["tactic"] for d in proposals)),
            "decisions_blocked_by_field": dict(blocked),
            "camera_age_ms_at_decision": _percentiles(ages),
            "camera_age_note": "decision time minus the frame's timestamp on the stream clock; real age adds camera latency and processing time",
            "proposal_spans_s": [[round(decisions[a].t, 2), round(decisions[b - 1].t, 2)]
                                 for a, b in _runs_of([d.proposed is not None for d in decisions])]}
    return summary


def _short(reason: str) -> str:
    """Group reasons that differ only in numbers (ids, counts, millimetres)."""
    head = reason.split(" (")[0]
    return "".join("#" if ch.isdigit() else ch for ch in head)


def write_trace(path: Path, decisions: list[Decision]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for decision in decisions:
            handle.write(json.dumps(decision.to_trace()) + "\n")


def write_frames(path: Path, samples: list[CameraSample]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(json.dumps(sample.to_report()) + "\n")


# ----------------------------------------------------------------------------------------------
# Drawing (for the demo video and the camera window). Display only.

_COLOURS = {"ok": (120, 220, 120), "bad": (80, 110, 255), "note": (225, 225, 225)}


def annotate(frame: np.ndarray, pipeline: CameraPipeline, sample: CameraSample, scale: float = 1.0) -> np.ndarray:
    image = frame.copy() if scale == 1.0 else cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    mapping = pipeline.mapping

    def px(point: Any) -> tuple[int, int]:
        return int(round(point[0] * scale)), int(round(point[1] * scale))

    if mapping.valid:
        arena = pipeline.config.arena
        w, h = float(arena["width_m"]), float(arena["height_m"])
        outline = mapping.floor_to_pixel(densify(np.array([[0, 0], [w, 0], [w, h], [0, h], [0, 0]], float), 0.02))
        if np.isfinite(outline).all():
            cv2.polylines(image, [np.round(outline * scale).astype(np.int32)], False, (200, 200, 60), 1, cv2.LINE_AA)
    if pipeline.tracker is not None:
        for track in pipeline.tracker.tracks:
            if track.state != TRACKED or track.pixel is None:
                continue
            is_own = track.track_id == sample.opponent.own_track_id
            is_opponent = sample.opponent.ok and track.track_id == sample.opponent.track_id
            colour = (255, 200, 60) if is_own else (60, 60, 255) if is_opponent else (190, 190, 190)
            label = track.label + (" own" if is_own else " OPPONENT" if is_opponent else "")
            cv2.circle(image, px(track.pixel), max(6, int(9 * scale)), colour, 2, cv2.LINE_AA)
            cv2.putText(image, label, (px(track.pixel)[0] + 12, px(track.pixel)[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1, cv2.LINE_AA)
        for group, position in pipeline.tracker.overlap_at.items():
            pixel = mapping.floor_to_pixel([position])[0]
            if np.isfinite(pixel).all():
                cv2.circle(image, px(pixel), max(12, int(16 * scale)), (0, 165, 255), 2, cv2.LINE_AA)
                cv2.putText(image, "OVERLAP: no position", (px(pixel)[0] + 16, px(pixel)[1] + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 165, 255), 1, cv2.LINE_AA)
    if sample.own.ok and mapping.valid:
        own = sample.own
        anchor = np.array(own.floor_projection_xy or (own.x_m, own.y_m))
        tip = anchor + 0.10 * np.array([math.cos(own.heading_rad), math.sin(own.heading_rad)])
        a, b = mapping.floor_to_pixel([anchor, tip])
        if np.isfinite([a, b]).all():
            cv2.arrowedLine(image, px(a), px(b), (255, 200, 60), 2, cv2.LINE_AA, tipLength=0.3)
    if sample.own_track is not None or sample.opponent_track is not None:      # only with --multi-marker / --click-opponent
        from control_center.tools.live_track import draw_tracking
        draw_tracking(image, pipeline, sample, scale)
    return image


def panel(lines: list[tuple[str, str]], width: int, height: int, title: str) -> np.ndarray:
    image = np.full((height, width, 3), 28, np.uint8)
    cv2.putText(image, title, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
    y, limit = 48, max(20, (width - 20) // 8)
    for text, level in lines:
        words, row = text.split(" "), ""
        rows = []
        for word in words:
            if len(row) + len(word) + 1 > limit and row.strip():
                rows.append(row)
                row = "    " + word
            else:
                row = word if not row else row + " " + word
        rows.append(row)
        for part in rows:
            if y > height - 8:
                return image
            cv2.putText(image, part, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, _COLOURS[level], 1, cv2.LINE_AA)
            y += 17
        y += 5
    return image


class DemoVideo:
    """Writes each camera frame beside the panel of the first decision that could use it."""

    def __init__(self, path: Path | None, pipeline: CameraPipeline, fps: float, scale: float, show: str, title: str, panel_width: int = 600) -> None:
        self.path, self.pipeline, self.fps, self.scale, self.show, self.title = path, pipeline, fps, scale, show, title
        self.panel_width = panel_width
        self.writer: cv2.VideoWriter | None = None
        self.pending: np.ndarray | None = None
        self.last_lines: list[tuple[str, str]] = [("waiting for the first decision", "note")]
        self.last_composed: np.ndarray | None = None

    def on_frame(self, frame: np.ndarray, sample: CameraSample) -> None:
        self._flush()
        self.pending = annotate(frame, self.pipeline, sample, self.scale)

    def on_decision(self, key: str, decision: Decision) -> None:
        if key != self.show:
            return
        self.last_lines = decision.ui_lines()
        self._flush()

    def _flush(self) -> None:
        if self.pending is None:
            return
        composed = np.hstack([self.pending, panel(self.last_lines, self.panel_width, self.pending.shape[0], self.title)])
        self.pending, self.last_composed = None, composed
        if self.path is None:
            return
        if self.writer is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.writer = cv2.VideoWriter(str(self.path), cv2.VideoWriter_fourcc(*"mp4v"), self.fps, (composed.shape[1], composed.shape[0]))
        self.writer.write(composed)

    def close(self) -> None:
        self._flush()
        if self.writer is not None:
            self.writer.release()


# ----------------------------------------------------------------------------------------------
# Part 2: the Bristol recording


def footage_frames(video: Path, start_s: float, duration_s: float) -> tuple[float, Iterator[tuple[float, np.ndarray]]]:
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"could not open video {video}")
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    fps = fps if 1.0 <= fps <= 240.0 else 30.0
    first = int(round(start_s * fps))
    count = max(1, int(round(duration_s * fps)))

    def frames() -> Iterator[tuple[float, np.ndarray]]:
        capture.set(cv2.CAP_PROP_POS_FRAMES, first)
        try:
            for index in range(first, first + count):
                ok, frame = capture.read()
                if not ok or frame is None:
                    return
                yield index / fps, frame          # the frame's own timestamp in the recording
        finally:
            capture.release()

    return fps, frames()


def run_footage(video: Path, contract: ObservationContract, out_dir: Path, start_s: float = CLIP_START_S,
                duration_s: float = CLIP_DURATION_S, reference_at_s: float = REFERENCE_AT_S,
                reference_frames: int = REFERENCE_FRAMES, assumed_latency_s: float = 0.0, save_video: bool = True,
                config: OverlayConfig | None = None) -> dict[str, Any]:
    config = config or load_config()
    clicks = load_clicks(profile_points_path(HISTORICAL_PROFILE))
    if not clicks.historical:
        raise ValueError(f"{HISTORICAL_PROFILE} is not marked historical; refusing to treat it as the footage profile")
    fps, reference_iter = footage_frames(video, reference_at_s, reference_frames / 29.97)
    reference = [frame for _, frame in reference_iter][:reference_frames]
    reference_end_s = reference_at_s + len(reference) / fps
    if not reference or reference_end_s >= start_s:
        raise ValueError("the startup reference must come from frames strictly before the clip starts")
    size = (reference[0].shape[1], reference[0].shape[0])
    mapping, camera, quality = fit_profile(config, clicks, size)
    if not mapping.valid:
        raise ValueError(f"the historical profile does not give a valid map at {size}: {mapping.reason}")
    pipeline = CameraPipeline(config, mapping, camera)
    note = (f"median of {len(reference)} consecutive frames at {reference_at_s:.0f}-{reference_end_s:.1f} s of the recording, "
            f"{start_s - reference_end_s:.0f} s before the clip; no frame at or after the clip start is used")
    pipeline.set_reference(reference, note)
    fps, frames = footage_frames(video, start_s, duration_s)
    assembler = ShadowAssembler(contract, FOOTAGE_SOURCE, NoTelemetry(), policy=None, max_camera_age_s=config.max_detection_age_s)
    recorder = DemoVideo(out_dir / "footage_shadow.mp4" if save_video else None, pipeline, fps, 2.0, "footage",
                         "RECORDED FOOTAGE (historical) - read-only diagnostic")
    run = run_stream("footage", frames, pipeline, {"footage": assembler}, contract.control_hz, assumed_latency_s=assumed_latency_s,
                     on_frame=recorder.on_frame, on_decision=recorder.on_decision)
    recorder.close()
    run.frame_period_s = 1.0 / fps
    run.notes = {
        "video": video.name, "clip_s": [start_s, start_s + duration_s], "fps": round(fps, 3), "frame_size": list(size),
        "profile": HISTORICAL_PROFILE, "map_fit": quality.level, "map_fit_reasons": quality.reasons,
        "startup_reference": note,
        "startup_reference_caveat": ("the arena was never empty in the recording, so robots are in the reference picture: "
                                     "one parked in the near-left corner and one standing by the far-left pit. Each is "
                                     "invisible while it stays put and leaves a false, stationary blob where it stood once "
                                     "it has moved"),
        "live_equivalent_of_reference": ("yes, and better: one second of frames of the EMPTY arena, taken after the camera is "
                                         "fixed and before the robots go in. It must be retaken if the camera or lighting changes"),
        "telemetry": "none: a video file carries no IMU",
        "assumed_latency_s": assumed_latency_s}
    summary = summarise(run)
    write_trace(out_dir / "footage_trace.jsonl", run.decisions["footage"])
    write_frames(out_dir / "footage_frames.jsonl", run.samples)
    if recorder.last_composed is not None:
        cv2.imwrite(str(out_dir / "footage_last_frame.png"), recorder.last_composed)
    summary["files"] = {"trace": "footage_trace.jsonl", "frames": "footage_frames.jsonl",
                        "video": "footage_shadow.mp4" if save_video else None}
    return summary


# ----------------------------------------------------------------------------------------------
# Part 3: the synthetic tagged fixture

OPPONENT_SIZE_M = (0.110, 0.100)
OPPONENT_COLOUR = (40, 40, 220)
MARKER_PLATE_HEIGHT_M = 0.034


@dataclass
class FixtureState:
    t: float
    phase: str
    own: tuple[float, float, float]             # x, y, heading (truth)
    own_yaw_rate: float                         # rad/s during the step that led here
    opponent: tuple[float, float, float]
    tags: list[int] | None                      # visible tag ids; None = all of the top face
    delivered: bool                             # False while the camera is stalled


# (name, seconds, own motion, visible tags, camera delivers frames, opponent speed m/s)
FIXTURE_PHASES: list[tuple[str, float, tuple[str, float, float], list[int] | None, bool, float]] = [
    ("placed, standing still", 1.5, ("drive", 0.0, 0.0), None, True, 0.0),
    ("driving, all four tags", 1.5, ("drive", 0.15, 0.0), None, True, 0.0),
    ("turning, ONE tag covered (id 1)", 1.0, ("drive", 0.10, 0.8), [0, 2, 3], True, 0.0),
    ("driving, all four tags", 0.5, ("drive", 0.10, 0.0), None, True, 0.0),
    ("ALL tags covered", 0.7, ("drive", 0.10, 0.0), [], True, 0.0),
    ("tags uncovered", 0.8, ("drive", 0.10, 0.0), None, True, 0.0),
    ("THREE tags covered (only id 2 left)", 0.7, ("drive", 0.05, -0.5), [2], True, 0.0),
    ("driving at the other object", 3.0, ("seek", 0.20, 2.5), None, True, 0.0),
    ("TOUCHING the other object", 1.2, ("drive", 0.0, 0.0), None, True, 0.0),
    ("backing away", 1.8, ("drive", -0.15, 0.0), None, True, 0.0),
    ("CAMERA STALL (no frames delivered)", 0.6, ("drive", 0.0, 0.0), None, False, 0.0),
    ("frames return, other object drives off", 1.5, ("drive", 0.0, 0.0), None, True, 0.12),
]
FIXTURE_OWN_START = (0.25, 0.30, 0.0)
FIXTURE_OPPONENT_START = (0.45, 0.85, math.radians(20.0))
FIXTURE_CONTACT_M = 0.125                       # centre distance at which the two outlines meet


def fixture_timeline(fps: float = 30.0) -> list[FixtureState]:
    """Truth for every frame time. Simple unicycle steps; `seek` turns toward the other object and stops at contact."""
    dt = 1.0 / fps
    x, y, heading = FIXTURE_OWN_START
    ox, oy, oheading = FIXTURE_OPPONENT_START
    states: list[FixtureState] = []
    t = 0.0
    for name, seconds, (mode, speed, turn), tags, delivered, opponent_speed in FIXTURE_PHASES:
        for _ in range(int(round(seconds * fps))):
            yaw_rate, v = 0.0, speed
            if mode == "seek":
                bearing = math.atan2(oy - y, ox - x)
                error = (bearing - heading + math.pi) % (2 * math.pi) - math.pi
                yaw_rate = max(-turn, min(turn, error / dt))
                v = speed if abs(error) < math.radians(25) and math.hypot(ox - x, oy - y) > FIXTURE_CONTACT_M else 0.0
            elif mode == "drive":
                yaw_rate = turn
            heading += yaw_rate * dt
            x, y = x + v * math.cos(heading) * dt, y + v * math.sin(heading) * dt
            ox, oy = ox + opponent_speed * math.cos(oheading) * dt, oy + opponent_speed * math.sin(oheading) * dt
            states.append(FixtureState(round(t, 6), name, (x, y, heading), yaw_rate, (ox, oy, oheading), tags, delivered))
            t += dt
    return states


def _paint_box(frame: np.ndarray, camera: FixtureCamera, centre: tuple[float, float], heading: float,
               size: tuple[float, float], height: float, colour: tuple[int, int, int]) -> None:
    length, breadth = size
    c, s = math.cos(heading), math.sin(heading)
    corners = np.array([[length / 2, breadth / 2], [length / 2, -breadth / 2], [-length / 2, -breadth / 2], [-length / 2, breadth / 2]])
    world = np.array(centre) + corners @ np.array([[c, s], [-s, c]])
    pixels = camera.project(np.column_stack([world, np.full(4, height)]))
    cv2.fillConvexPoly(frame, np.round(pixels).astype(np.int32), colour, cv2.LINE_AA)


def render_fixture_frame(config: OverlayConfig, camera: FixtureCamera, state: FixtureState | None, seed: int) -> np.ndarray:
    """One SYNTHETIC frame. ``state`` None = the empty floor (for the startup reference)."""
    if state is None:
        return render_frame(config, camera, Scene(), noise_seed=seed)
    x, y, heading = state.own
    if state.tags == []:
        frame = render_frame(config, camera, Scene(), noise_seed=seed)
        _paint_box(frame, camera, (x, y), heading, PLATE_SIZE_M, MARKER_PLATE_HEIGHT_M, (45, 45, 45))   # the bot with every tag covered
    else:
        frame = render_frame(config, camera, Scene([BotInstance(x, y, heading, ids=state.tags)]), noise_seed=seed)
    ox, oy, oheading = state.opponent
    _paint_box(frame, camera, (ox, oy), oheading, OPPONENT_SIZE_M, MARKER_PLATE_HEIGHT_M, OPPONENT_COLOUR)
    return frame


def fixture_imu(states: list[FixtureState], rate_hz: float) -> RecordedTelemetry:
    """SYNTHETIC stand-in IMU: body yaw rate from the fixture's own motion, gravity on +z. No noise, no real sensor."""
    samples, end = [], states[-1].t
    times = [state.t for state in states]
    for index in range(int(end * rate_hz) + 1):
        t = index / rate_hz
        state = states[min(len(states) - 1, int(np.searchsorted(times, t + 1e-9, side="right")) - 1)]
        samples.append(ImuSample(t, (0.0, 0.0, state.own_yaw_rate), (0.0, 0.0, GRAVITY_M_S2)))
    return RecordedTelemetry(samples, "SYNTHETIC stand-in IMU (fixture motion, noise-free)")


def _angle_error_deg(a: float, b: float) -> float:
    return abs(math.degrees((a - b + math.pi) % (2 * math.pi) - math.pi))


def run_fixture(contract: ObservationContract, out_dir: Path, policy: CheckpointPolicy | None, save_video: bool = True,
                assumed_latency_s: float = 0.0, config: OverlayConfig | None = None, fps: float = 30.0) -> dict[str, Any]:
    config = config or load_config()
    camera = default_camera()
    mapping, camera_estimate, quality = fit_profile(config, fixture_clicks(config, camera), camera.image_size)
    pipeline = CameraPipeline(config, mapping, camera_estimate)
    reference = [render_fixture_frame(config, camera, None, seed=9000 + index) for index in range(15)]
    pipeline.set_reference(reference, "median of 15 SYNTHETIC frames of the empty floor, rendered before the stream starts")
    states = fixture_timeline(fps)
    by_time = {state.t: state for state in states}

    def frames() -> Iterator[tuple[float, np.ndarray | None]]:
        for index, state in enumerate(states):
            yield state.t, (render_fixture_frame(config, camera, state, seed=100 + index) if state.delivered else None)

    telemetry = fixture_imu(states, contract.imu_rate_hz)
    assemblers = {
        "camera_only": ShadowAssembler(contract, FIXTURE_CAMERA_ONLY, NoTelemetry(), policy=None,
                                       max_camera_age_s=config.max_detection_age_s),
        "with_synthetic_imu": ShadowAssembler(contract, FIXTURE_WITH_IMU, telemetry, policy=policy, shadow_label=FIXTURE_SHADOW,
                                              max_camera_age_s=config.max_detection_age_s)}
    recorder = DemoVideo(out_dir / "fixture_shadow.mp4" if save_video else None, pipeline, fps, 1.0, "with_synthetic_imu",
                         FIXTURE_LABEL)
    run = run_stream("fixture", frames(), pipeline, assemblers, contract.control_hz, end_t=states[-1].t,
                     assumed_latency_s=assumed_latency_s, on_frame=recorder.on_frame, on_decision=recorder.on_decision)
    recorder.close()
    run.frame_period_s = 1.0 / fps
    run.notes = {"label": FIXTURE_LABEL, "frame_size": list(camera.image_size), "fps": fps, "map_fit": quality.level,
                 "startup_reference": pipeline.reference_note,
                 "what_it_cannot_show": ("real marker pose accuracy, lens distortion, motion blur, lighting, a hand over a "
                                         "tag, or camera latency. The frames are rendered, the map clicks are exact"),
                 "assumed_latency_s": assumed_latency_s}
    summary = summarise(run)

    # ---- against the fixture's ground truth, phase by phase ----
    phases: dict[str, dict[str, Any]] = {}
    position_errors, heading_errors, opponent_errors = [], [], []
    for sample in run.samples:
        state = by_time[round(sample.t, 6)]
        entry = phases.setdefault(state.phase, {"frames": 0, "own_pose_valid": 0, "opponent_valid": 0, "tag_sets": Counter(),
                                                "own_error_mm": [], "opponent_error_mm": []})
        entry["frames"] += 1
        entry["tag_sets"][str(sample.own.used_ids)] += 1
        if sample.own.ok:
            entry["own_pose_valid"] += 1
            error = 1000 * math.hypot(sample.own.x_m - state.own[0], sample.own.y_m - state.own[1])
            entry["own_error_mm"].append(error)
            position_errors.append(error)
            heading_errors.append(_angle_error_deg(sample.own.heading_rad, state.own[2]))
        if sample.opponent.ok:
            entry["opponent_valid"] += 1
            error = 1000 * math.hypot(sample.opponent.xy_m[0] - state.opponent[0], sample.opponent.xy_m[1] - state.opponent[1])
            entry["opponent_error_mm"].append(error)
            opponent_errors.append(error)
    shadow = run.decisions["with_synthetic_imu"]
    for decision in shadow:
        if decision.frame_t is None:
            continue
        state = min(states, key=lambda s: abs(s.t - decision.t))
        entry = phases.setdefault(state.phase, {"frames": 0, "own_pose_valid": 0, "opponent_valid": 0, "tag_sets": Counter(),
                                                "own_error_mm": [], "opponent_error_mm": []})
        entry["decisions"] = entry.get("decisions", 0) + 1
        entry["proposals"] = entry.get("proposals", 0) + (decision.proposed is not None)
    summary["phases"] = [
        {"phase": name, "frames_delivered": e["frames"], "own_pose_valid_frames": e["own_pose_valid"],
         "opponent_valid_frames": e["opponent_valid"], "tag_sets_used": dict(e["tag_sets"]),
         "own_error_mm_max": round(max(e["own_error_mm"]), 1) if e["own_error_mm"] else None,
         "opponent_error_mm_max": round(max(e["opponent_error_mm"]), 1) if e["opponent_error_mm"] else None,
         "decisions": e.get("decisions", 0), "proposals": e.get("proposals", 0)}
        for name, e in phases.items()]
    summary["against_fixture_truth"] = {
        "own_position_error_mm": _percentiles(position_errors), "own_heading_error_deg": _percentiles(heading_errors),
        "opponent_position_error_mm": _percentiles(opponent_errors),
        "note": "rendered pictures with exact map clicks: these numbers check the arithmetic, they are NOT real-camera accuracy"}

    # ---- does the camera-built observation lead to the same choice as the fixture's exact values? ----
    if policy is not None:
        complete = [d for d in shadow if d.observation is not None and d.proposed is not None]
        if complete:
            camera_obs = np.stack([d.observation for d in complete])
            exact = []
            for decision in complete:
                state = by_time[round(decision.frame_t, 6)]
                imu = telemetry.latest(decision.t)
                exact.append(contract.assemble(gyro_rad_s=imu.gyro_rad_s, accel_m_s2=imu.accel_m_s2, own_xy_m=state.own[:2],
                                               own_heading_rad=state.own[2], opponent_xy_m=state.opponent[:2]))
            chosen = np.array([d.proposed["action"] for d in complete])
            exact_actions = policy.actions(np.stack(exact))
            summary["camera_observation_vs_exact_fixture_values"] = {
                "decisions_compared": len(complete), "same_tactic": int((chosen == exact_actions).sum()),
                "largest_difference_in_any_observation_value": round(float(np.abs(camera_obs - np.stack(exact)).max()), 4)}
            summary["imu_sensitivity"] = imu_sensitivity(contract, policy, camera_obs)

    write_trace(out_dir / "fixture_trace_camera_only.jsonl", run.decisions["camera_only"])
    write_trace(out_dir / "fixture_trace_with_synthetic_imu.jsonl", shadow)
    write_frames(out_dir / "fixture_frames.jsonl", run.samples)
    truth = [{"t_s": s.t, "phase": s.phase, "own": [round(v, 5) for v in s.own], "opponent": [round(v, 5) for v in s.opponent],
              "visible_tags": s.tags, "frame_delivered": s.delivered} for s in states]
    (out_dir / "fixture_truth.json").write_text(json.dumps({"label": FIXTURE_LABEL, "frames": truth}), encoding="utf-8")
    summary["files"] = {"trace_camera_only": "fixture_trace_camera_only.jsonl", "trace_with_synthetic_imu": "fixture_trace_with_synthetic_imu.jsonl",
                        "frames": "fixture_frames.jsonl", "truth": "fixture_truth.json",
                        "video": "fixture_shadow.mp4" if save_video else None}
    return summary


def imu_sensitivity(contract: ObservationContract, policy: CheckpointPolicy, observations: np.ndarray, draws: int = 40,
                    seed: int = 0) -> dict[str, Any]:
    """How often the chosen tactic changes when ONLY the six IMU values change. An analysis, not a shadow run.

    The IMU values are replaced by ones the simulator could produce: body rates up to a hard turn and gravity seen
    from a chassis tilted by up to 10 degrees. The other thirteen values are left exactly as the camera built them."""
    indices = contract.indices(GROUP_IMU)
    base = policy.actions(observations)
    rest = observations.copy()
    rest[:, indices] = np.array([0, 0, 0, 0, 0, GRAVITY_M_S2]) / ([contract.gyro_range_rad_s] * 3 + [contract.accel_range_m_s2] * 3)
    at_rest_changed = int((policy.actions(rest) != base).sum())
    rng = np.random.default_rng(seed)
    changed = total = 0
    for _ in range(draws):
        varied = observations.copy()
        gyro = np.column_stack([rng.normal(0, 0.3, len(varied)), rng.normal(0, 0.3, len(varied)), rng.uniform(-4.5, 4.5, len(varied))])
        tilt, direction = np.radians(rng.uniform(0, 10, len(varied))), rng.uniform(0, 2 * math.pi, len(varied))
        accel = GRAVITY_M_S2 * np.column_stack([np.sin(tilt) * np.cos(direction), np.sin(tilt) * np.sin(direction), np.cos(tilt)])
        varied[:, indices] = np.clip(np.column_stack([gyro / contract.gyro_range_rad_s, accel / contract.accel_range_m_s2]), -1, 1)
        changed += int((policy.actions(varied) != base).sum())
        total += len(varied)
    return {"observations": int(len(observations)), "tactic_changes_with_exact_rest_imu": at_rest_changed,
            "random_imu_draws_per_observation": draws, "tactic_changed_share": round(changed / max(1, total), 4),
            "meaning": ("share of decisions whose tactic changes when only the IMU values are replaced by other values the "
                        "simulator could produce (yaw rate within +-4.5 rad/s, tilt up to 10 deg). Not zero means the IMU "
                        "is a real input to this checkpoint, not a formality")}


# ----------------------------------------------------------------------------------------------
# Camera mode: the five-minute home test. A real camera is opened ONLY here, only when the user asks.


def _override_markers(config: OverlayConfig, dictionary: str | None, tag_size_mm: float | None) -> OverlayConfig:
    changes: dict[str, Any] = {}
    if dictionary:
        from control_center.tools.overlay_core import dictionary_id
        dictionary_id(dictionary)
        changes["dictionary"] = dictionary
    if tag_size_mm:
        changes["markers"] = {key: dataclasses.replace(spec, size_m=tag_size_mm / 1000.0) for key, spec in config.markers.items()}
    return dataclasses.replace(config, **changes) if changes else config


def run_camera(args: argparse.Namespace, contract: ObservationContract) -> dict[str, Any]:
    from control_center.tools import arena_overlay          # its camera opener and capture settings, unchanged

    config = _override_markers(load_config(args.config), args.dictionary, args.tag_size_mm)
    points_path = Path(args.points) if args.points else profile_points_path(args.profile)
    clicks = load_clicks(points_path) if points_path.exists() else ClickSet()
    if clicks.historical:                                   # checked before any camera is opened
        raise ValueError(f"{points_path} is a HISTORICAL footage profile; it is never used with a live camera")
    if not args.points:                                         # an explicit --points file bypasses the profile on purpose
        config = with_arena_setup(config, args.profile, clicks)       # the profile's saved arena and hazards, if it has any
    source = arena_overlay.CameraSource(args.camera, arena_overlay.camera_settings(args))
    out_dir = Path(args.out) / ("camera-" + datetime.now().strftime("%Y%m%d-%H%M%S"))
    out_dir.mkdir(parents=True, exist_ok=True)
    pipeline: CameraPipeline | None = None
    assembler: ShadowAssembler | None = None
    collecting: list[np.ndarray] | None = None
    message = "EMPTY THE FLOOR (no objects, no hands), then press SPACE"
    decisions: list[Decision] = []
    frames_seen, processing = 0, []
    input_source = f"LIVE camera {source.description}, profile '{args.profile}' (no telemetry)"
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    multi_marker, click_opponent = bool(getattr(args, "multi_marker", False)), bool(getattr(args, "click_opponent", False))
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
                if pipeline is None:
                    mapping, camera, quality = fit_profile(config, clicks, (frame.shape[1], frame.shape[0]))
                    pipeline = (CameraPipeline(config, mapping, camera, multi_marker=multi_marker, click_opponent=click_opponent)
                                if multi_marker or click_opponent else CameraPipeline(config, mapping, camera))
                    assembler = ShadowAssembler(contract, input_source, NoTelemetry(), policy=None,
                                                max_camera_age_s=config.max_detection_age_s)
                    if not mapping.valid:
                        message = f"NO FLOOR MAP ({mapping.reason}): tag ids only. Calibrate first, see the handoff"
                if queue is not None:
                    queue.frame_width = frame.shape[1]
                    for event in apply_clicks(pipeline, queue, frame, t):
                        message = ("opponent selected: " if event["accepted"] else "click REFUSED: ") + event["reason"]
                if collecting is not None:
                    collecting.append(frame.copy())
                    if len(collecting) >= 15:
                        try:
                            pipeline.set_reference(collecting, "median of 15 live frames taken when SPACE was pressed")
                            message = "reference taken: put the objects on the floor"
                        except ValueError as error:
                            message = f"reference NOT taken: {error}"
                        collecting = None
                sample = pipeline.process(frame, t)
                processing.append(sample.timing_ms["total"])
                assembler.offer(sample)
                decision = assembler.decide(max(t, time.monotonic()) if source.kind == "camera" else t)
                decisions.append(decision)
                if len(decisions) <= 20000:
                    record = decision.to_trace()
                    if sample.own_track is not None:            # only with --multi-marker / --click-opponent
                        record["own_track"] = sample.own_track
                    if sample.opponent_track is not None:
                        record["opponent_track"] = sample.opponent_track
                    trace.write(json.dumps(record) + "\n")
                lines = decision.ui_lines() + [
                    (f"markers decoded: {sample.marker_ids or 'none'}  dictionary {config.dictionary}", "note"),
                    (f"blobs: {sample.blob_count}   software time this frame: {sample.timing_ms['total']:.1f} ms", "note"),
                    (message, "note"), ("SPACE reference   r again   q quit", "note")]
                if multi_marker or click_opponent:
                    from control_center.tools.live_track import tracking_lines
                    keys = "LEFT-CLICK opponent   c clear   q quit" if click_opponent else "SPACE reference   r again   q quit"
                    lines = lines[:-2] + tracking_lines(sample) + [(message, "note"), (keys, "note")]
                view = np.hstack([annotate(frame, pipeline, sample), panel(lines, 600, frame.shape[0], "LIVE CAMERA - read-only, nothing is sent")])
                if queue is not None:
                    queue.shown = (frame, t)
                cv2.imshow(WINDOW, view)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27) or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                    break
                if click_opponent and key == ord("c"):
                    pipeline.clear_opponent()
                    message = "opponent selection cleared: LEFT-CLICK the opponent"
                if key in (32, ord("r")) and not click_opponent:
                    collecting, message = [], "taking the reference picture: keep the floor empty"
                if args.max_frames is not None and frames_seen >= args.max_frames:
                    break
    finally:
        source.close()
        cv2.destroyAllWindows()
    summary = {"input_source": input_source, "frames": frames_seen, "decisions": len(decisions),
               "modes": dict(Counter(d.mode for d in decisions)),
               "software_ms_per_frame": _percentiles(processing[1:]), "trace": str(out_dir / "trace.jsonl"), "last_message": message}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


# ----------------------------------------------------------------------------------------------
# Console report


def _print_stream(summary: dict[str, Any]) -> None:
    timing = summary["processing_ms_per_frame"]
    print(f"  frames fed          {summary['frames']} over {summary['stream_seconds']} s of stream, in timestamp order")
    print(f"  software time/frame median {timing['total']['median']} ms, p95 {timing['total']['p95']} ms, max {timing['total']['max']} ms "
          f"(markers {timing['markers']['median']}, blobs {timing['blobs']['median']}); first frame {summary['first_frame_ms']} ms")
    slow = summary["frames_slower_than_one_frame_period"]
    print(f"  slower than a frame {slow['count']} frame(s) took longer than {slow['frame_period_ms']} ms. Camera latency is NOT in these numbers")
    own, opponent, tracks = summary["own_pose"], summary["opponent"], summary["blob_tracks"]
    print(f"  own pose (ArUco)    valid on {own['valid_frames']} frames, missing on {own['missing_frames']}")
    for reason, count in own["missing_reasons"].items():
        print(f"      {count:>5}  {reason}")
    print(f"  opponent            valid on {opponent['valid_frames']} frames, unknown on {opponent['unknown_frames']}")
    for reason, count in opponent["unknown_reasons"].items():
        print(f"      {count:>5}  {reason}")
    print(f"  blob tracks         ids {tracks['ids_seen']}, overlap on {tracks['frames_with_an_overlap']} frames "
          f"{tracks['overlap_spans_s']}, positions reported during overlap or loss: {tracks['positions_reported_during_overlap_or_loss']}")
    for key, block in summary["decisions"].items():
        print(f"  decisions [{key}]  {block['count']} at the control rate; complete observations {block['complete_observations']}, "
              f"proposals {block['proposals']} {block['proposed_tactics'] or ''}")
        for name, count in block["decisions_blocked_by_field"].items():
            print(f"      {count:>5}  blocked by {name}")


def _print_sample_trace(title: str, decisions_path: Path, picks: int = 3) -> None:
    lines = decisions_path.read_text(encoding="utf-8").splitlines()
    if not lines:
        return
    print(f"  sample from {decisions_path.name} ({title}):")
    step = max(1, len(lines) // picks)
    for index in range(step // 2, len(lines), step):
        record = json.loads(lines[index])
        inputs = "  ".join(f"{group}={'ok' if status['valid'] else 'MISSING'}" for group, status in record["inputs"].items())
        proposed = record["proposed"]["tactic"] if record["proposed"] else "none"
        print(f"    t={record['t_s']:>8.2f}s frame={record['camera_frame']} {inputs}  mode={record['mode'].split(' (')[0]}  proposed={proposed}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0], epilog=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--checkpoint", default=str(RECOMMENDED_CHECKPOINT), help="checkpoint zip (default: the recommended one)")
    parser.add_argument("--footage", default=os.environ.get(FOOTAGE_ENV), help=f"the Bristol recording (default: ${FOOTAGE_ENV})")
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="output folder")
    parser.add_argument("--skip-footage", action="store_true")
    parser.add_argument("--skip-fixture", action="store_true")
    parser.add_argument("--no-video", action="store_true", help="do not write the annotated demo videos")
    parser.add_argument("--no-policy", action="store_true", help="never load the checkpoint (contract and stream diagnostics only)")
    parser.add_argument("--assumed-latency-ms", type=float, default=0.0,
                        help="pretend every frame arrives this long after its timestamp (the real figure is not measured)")
    camera = parser.add_argument_group("camera mode (opens a REAL camera; only when you pass --camera)")
    camera.add_argument("--camera", type=int, metavar="N", help="live camera index")
    camera.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="overlay config (camera mode)")
    camera.add_argument("--profile", default=LIVE_PROFILE, help="calibration profile (camera mode); never a historical one")
    camera.add_argument("--points", default=None, help="clicked-points file, instead of the profile's own (camera mode)")
    camera.add_argument("--dictionary", default=None, help="override the marker dictionary, e.g. DICT_APRILTAG_36h11")
    camera.add_argument("--tag-size-mm", type=float, default=None, help="override the printed tag size (black square side)")
    camera.add_argument("--max-frames", type=int, default=None)
    camera.add_argument("--width", type=int)
    camera.add_argument("--height", type=int)
    camera.add_argument("--fps", type=float)
    camera.add_argument("--backend", choices=["dshow", "msmf", "auto"])
    camera.add_argument("--multi-marker", action="store_true",
                        help="own pose from the multi-marker chassis tracker (consensus, short prediction, MEASURED/PREDICTED/STALE/LOST)")
    camera.add_argument("--click-opponent", action="store_true",
                        help="left-click the opponent in the picture to select it (c clears); no empty-floor reference is needed")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        contract = load_contract(args.checkpoint)
    except ContractError as error:
        print(f"contract error: {error}")
        return 2
    if args.camera is not None:
        try:
            summary = run_camera(args, contract)
        except (ValueError, RuntimeError) as error:
            print(f"error: {error}")
            return 2
        print(json.dumps(summary, indent=2))
        return 0

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    latency = args.assumed_latency_ms / 1000.0
    report: dict[str, Any] = {"generated": datetime.now().isoformat(timespec="seconds"), "contract": contract.to_report(),
                              "output": "none: no serial port, controller link or camera is opened; proposals go to trace files only"}
    print("=" * 100 + "\n1. CONTRACT\n" + "=" * 100)
    print(format_contract(contract))

    print("=" * 100 + "\n2. FOOTAGE (recorded, historical profile, one frame at a time)\n" + "=" * 100)
    if args.skip_footage:
        print("  skipped (--skip-footage)")
    elif not args.footage or not Path(args.footage).is_file():
        print(f"  skipped: no recording given. Pass --footage <mp4> or set {FOOTAGE_ENV}")
        report["footage"] = {"skipped": "recording not available"}
    else:
        report["footage"] = run_footage(Path(args.footage), contract, out_dir, assumed_latency_s=latency, save_video=not args.no_video)
        print(f"  startup reference   {report['footage']['notes']['startup_reference']}")
        _print_stream(report["footage"])
        _print_sample_trace("blocked: nothing is proposed", out_dir / "footage_trace.jsonl")

    print("=" * 100 + f"\n3. FIXTURE ({FIXTURE_LABEL})\n" + "=" * 100)
    if args.skip_fixture:
        print("  skipped (--skip-fixture)")
    else:
        policy = None if args.no_policy else CheckpointPolicy(contract)
        report["fixture"] = fixture = run_fixture(contract, out_dir, policy, save_video=not args.no_video, assumed_latency_s=latency)
        _print_stream(fixture)
        print("  phase by phase:")
        print(f"    {'phase':<40} {'frames':>6} {'own ok':>6} {'opp ok':>6} {'proposals':>10}   tag sets used")
        for row in fixture["phases"]:
            print(f"    {row['phase']:<40} {row['frames_delivered']:>6} {row['own_pose_valid_frames']:>6} {row['opponent_valid_frames']:>6} "
                  f"{str(row['proposals']) + '/' + str(row['decisions']):>10}   {row['tag_sets_used']}")
        truth = fixture["against_fixture_truth"]
        print(f"  against fixture truth (arithmetic check, NOT camera accuracy): own position max {truth['own_position_error_mm']['max']} mm, "
              f"heading max {truth['own_heading_error_deg']['max']} deg, opponent max {truth['opponent_position_error_mm']['max']} mm")
        if "camera_observation_vs_exact_fixture_values" in fixture:
            same = fixture["camera_observation_vs_exact_fixture_values"]
            sensitivity = fixture["imu_sensitivity"]
            print(f"  same tactic as with the fixture's exact values: {same['same_tactic']} of {same['decisions_compared']} decisions")
            print(f"  IMU sensitivity: tactic changes on {100 * sensitivity['tactic_changed_share']:.1f}% of decisions when only the IMU values change")
            print(f"  checkpoint loaded in {policy.load_seconds:.1f} s; used only in the run labelled {FIXTURE_SHADOW}")
        _print_sample_trace("camera only: blocked by the IMU alone when the camera inputs are valid", out_dir / "fixture_trace_camera_only.jsonl", 4)
        _print_sample_trace("synthetic IMU stand-in: the only run with proposals", out_dir / "fixture_trace_with_synthetic_imu.jsonl", 8)

    footage_blocked = report.get("footage", {}).get("decisions", {}).get("footage", {})
    verdict = ["LIVE SHADOW is NOT available. The checkpoint was not run on the recorded footage."]
    if footage_blocked:
        verdict.append("On the footage every decision was blocked by: " + ", ".join(footage_blocked["decisions_blocked_by_field"]))
    verdict.append("Fields [0-5] (gyro, accelerometer) have no live source in this repository: that alone blocks inference.")
    report["verdict"] = verdict
    print("=" * 100 + "\nVERDICT\n" + "=" * 100)
    for line in verdict:
        print("  " + line)
    (out_dir / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (out_dir / "contract.txt").write_text(format_contract(contract), encoding="utf-8")
    print(f"\n  written to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
