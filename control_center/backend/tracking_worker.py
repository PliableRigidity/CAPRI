"""Vision work for the ROBOT TRACKING & LIVE MONITORING screen. READ-ONLY: frames in, display data out.

  TrackingEngine   Qt-free: one frame in -> one TrackingResult out (annotated picture, tracker reports, marker rates,
                   diagnostics, optional shadow decision). Driven directly by the tests with synthetic frames.
  TrackingWorker   QThread around an engine: pulls the NEWEST frame from the hub's capture session (older frames are
                   skipped, never queued), publishes only the latest result, applies commands posted by the GUI.
  ShadowRunner     the display-only shadow: ShadowAssembler (+ LiveShadow's tag-switch guard) on the engine's own samples.
  HubTelemetry     the controller lines the hub already receives, turned into the IMU the policy needs. It never sends.
  CheckpointLoader QThread that loads the checkpoint (30-60 s) off the GUI thread.

There is no camera, serial port or socket here: frames come from ``hub.capture``, and nothing in this module has a way to
command the robot. A policy proposal is a dictionary that is shown on screen.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from collections import deque
from dataclasses import dataclass, field, replace
from typing import Any, Callable

import cv2
import numpy as np
from PySide6.QtCore import QObject, QThread, Signal

from control_center.backend.controller_link import parse_imu_line
from control_center.backend.observation_builder import SOURCE_LIVE, SOURCE_SIMULATED, origin_from_capture_state
from control_center.backend.tracking_model import (DiagnosticsModel, MarkerRateTracker, MarkerRow, arena_problems, marker_spread,
                                                   pit_list)
from control_center.tools.arena_overlay import BOUNDARY, FLIPPER, PIT, draw_arena
from control_center.tools.live_shadow import LIVE_LABEL, TagSwitchGuard
from control_center.tools.live_telemetry import LiveTelemetry
from control_center.tools.live_track import LiveTrack, draw_tracking
from control_center.tools.overlay_core import (BotPose, ClickSet, OverlayConfig, arena_outline, densify, flipper_boundary,
                                               pit_arcs)
from control_center.tools.stream_adapter import CameraSample, ShadowAssembler, fit_profile

log = logging.getLogger(__name__)

PAUSE_REBUILD_AFTER_S = 2.0          # trackers are restarted when the screen was hidden longer than this
MAX_IMU_AGE_S = 0.10                 # the assembler's own limit, so "fresh" on screen means what it means in the decision


# ---------------------------------------------------------------- telemetry from the hub's listen-only monitor


class HubTelemetry(LiveTelemetry):
    """``LiveTelemetry`` fed by the hub's controller lines (listen-only), safe to read from the vision thread.

    The hub's own comma format ``IMU,<ms>,ax,ay,az,gx,gy,gz`` is re-spelled as the strict ``IMU seq= t_us= ...`` line the
    parent parses; the numbers are copied, never made up (the sequence counter is local and only orders them).
    """

    origin = SOURCE_LIVE        # parsed from lines received on the controller link; the CAMERA's origin (capture state) must also be LIVE

    def __init__(self, mounting, source_label: str = "Command Centre controller monitor (listen-only)") -> None:
        super().__init__(mounting, source_label)
        self._lock = threading.RLock()
        self._comma_sequence = 0

    def feed(self, direction: str, line: str, now: float) -> None:
        with self._lock:
            if direction == "RX" and line.startswith("IMU,"):
                sample = parse_imu_line(line)
                if sample is None:
                    self.last_line_at, self.malformed = now, self.malformed + 1
                    return
                self._comma_sequence += 1
                a, g = sample.accel_m_s2, sample.gyro_deg_s
                line = (f"IMU seq={self._comma_sequence} t_us={sample.device_ms * 1000} ok=1 ax={a[0]!r} ay={a[1]!r} az={a[2]!r} "
                        f"gx={g[0]!r} gy={g[1]!r} gz={g[2]!r} unit=m/s2,deg/s")
            super().feed(direction, line, now)

    def latest(self, t: float):
        with self._lock:
            return super().latest(t)

    def lines(self, now: float):
        with self._lock:
            return super().lines(now)

    def report(self, now: float, max_age_s: float = MAX_IMU_AGE_S) -> dict[str, Any]:
        """Present / fresh / mounting, each with the exact reason. Never contains a number that was not received."""
        with self._lock:
            failed = (self.failure_at is not None and (self.reading_at is None or self.failure_at >= self.reading_at))
            if self.last_line_at is None:
                present, why = False, "no controller lines have been received (is the controller telemetry source connected?)"
            elif now - self.last_line_at > 1.0:
                present, why = False, f"controller link silent for {now - self.last_line_at:.1f} s"
            elif failed:
                present, why = False, f"the controller reports the IMU as {self.failure.why} and sends no values"
            elif self.reading is None:
                present, why = False, "controller status is arriving but there is no IMU line: this firmware does not send IMU data"
            else:
                present, why = True, f"IMU lines arriving, seq {self.reading.sequence}"
            age = None if self.reading_at is None else now - self.reading_at
            if age is None:
                fresh, fresh_why = False, "no IMU sample has been received"
            elif age > max_age_s:
                fresh, fresh_why = False, f"newest IMU sample is {1000 * age:.0f} ms old (limit {1000 * max_age_s:.0f} ms)"
            else:
                fresh, fresh_why = True, f"newest IMU sample is {1000 * age:.0f} ms old"
            verified = bool(self.mounting.verified)
            mounting = ("imu.verified is true in " + (self.mounting.path.name if self.mounting.path else "the mounting file")
                        if verified else "NOT verified: imu.verified is false in "
                        + (self.mounting.path.name if self.mounting.path else "the mounting file")
                        + " (do the axis check, docs test IMU-2, then set it)")
            return {"present": present, "present_detail": why, "fresh": fresh and present, "fresh_detail": fresh_why,
                    "mounting_verified": verified, "mounting_detail": mounting}


# ---------------------------------------------------------------- the result handed to the GUI


@dataclass
class TrackingResult:
    seq: int
    frame_index: int
    t: float                                  # capture time (time.monotonic())
    done_at: float                            # when this result was finished (time.monotonic())
    size: tuple[int, int]
    raw: np.ndarray                           # the frame as captured: never drawn on, used for clicks
    annotated: np.ndarray                     # a copy with the overlay and both tracks drawn
    sample: CameraSample | None
    own: dict[str, Any] | None
    opponent: dict[str, Any] | None
    marker_ids: list[int]
    marker_rows: list[MarkerRow]
    marker_frames: int
    evidence: dict[str, Any] | None
    limits: dict[str, float]
    dictionary: str
    placeholders: list[str]
    map_level: str                            # "none" (never calibrated), "invalid", "weak", "ok"
    map_reason: str
    fit_level: str
    fit_reasons: list[str]
    arena_problems: list[str]
    software_ms: float                        # CameraPipeline.process total (markers + tracker + opponent)
    engine_ms: float                          # everything this engine did for the frame, drawing included
    skipped: int
    queue_age_s: float                        # capture -> picked up by the vision thread
    generation: int
    epoch: int
    diagnostics: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    click_events: list[dict[str, Any]] = field(default_factory=list)
    shadow: dict[str, Any] | None = None
    error: str = ""


# ---------------------------------------------------------------- drawing


def arena_shapes(arena: dict[str, Any]) -> list[tuple[str, np.ndarray, tuple[int, int, int]]]:
    """Like ``arena_overlay.overlay_shapes`` but each pit keeps its own radius when the profile has ``pit_radii_m``."""
    shapes: list[tuple[str, np.ndarray, tuple[int, int, int]]] = [("arena boundary", arena_outline(arena), BOUNDARY)]
    for centre, radius in pit_list(arena):
        one = {**arena, "pit_centres_m": [list(centre)], "pit_radius_m": radius}
        shapes += [("pit boundary", arc, PIT) for arc in pit_arcs(one)]
    shapes.append(("fixed flipper boundary", flipper_boundary(arena), FLIPPER))
    return shapes


def draw_arena_overlay(image: np.ndarray, mapping, arena: dict[str, Any]) -> None:
    if arena.get("pit_radii_m") is None:
        draw_arena(image, mapping, arena)               # the overlay tool's own drawing, unchanged
        return
    for label, polyline, colour in arena_shapes(arena):
        pixels = mapping.floor_to_pixel(densify(polyline))
        if len(pixels) == 0 or not np.isfinite(pixels).all():
            continue
        points = np.round(np.clip(pixels, -20000, 20000)).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(image, [points], False, colour, 2, cv2.LINE_AA)
        if label != "arena boundary":
            anchor = points[len(points) // 2, 0]
            cv2.putText(image, "PIT" if label == "pit boundary" else "FLIPPER (fixed)", (int(anchor[0]) + 6, int(anchor[1]) - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, 1, cv2.LINE_AA)


def draw_heading(image: np.ndarray, mapping, own: dict[str, Any] | None, length_m: float = 0.12) -> None:
    """A heading line from the own chassis centre (the footprint ring itself is ``live_track.draw_tracking``)."""
    if own is None or own.get("x_m") is None or own.get("heading_deg") is None or not mapping.valid:
        return
    angle = math.radians(own["heading_deg"])
    pixels = mapping.floor_to_pixel(np.array([[own["x_m"], own["y_m"]],
                                              [own["x_m"] + length_m * math.cos(angle), own["y_m"] + length_m * math.sin(angle)]]))
    if not np.isfinite(pixels).all():
        return
    colour = {"measured": (120, 220, 120), "predicted": (60, 200, 255)}.get(own["state"], (80, 110, 255))
    a, b = np.round(pixels).astype(int)
    cv2.arrowedLine(image, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])), colour, 2, cv2.LINE_AA, tipLength=0.3)


# ---------------------------------------------------------------- shadow (display only)


class ShadowRunner:
    """LiveShadow's contract, tag-switch guard and ShadowAssembler, run on the engine's OWN camera samples.

    ``LiveShadow.step`` would run a second pipeline (a second marker detection and a second opponent selection) on the same
    frame; this reuses its guard and assembler on the engine's sample instead, with the same steps as ``LiveShadow.step``.
    """

    def __init__(self, contract, config: OverlayConfig, clicks: ClickSet, telemetry, input_source: str, policy=None) -> None:
        self.contract, self.config, self.telemetry, self.input_source = contract, config, telemetry, input_source
        self.policy = policy
        self.guard = TagSwitchGuard(config.max_position_disagreement_m)
        self.assembler: ShadowAssembler | None = None
        self.error = ""
        self.last_decision = None
        # Declared by whoever owns the camera: the hub capture session's state (LIVE / MOCK / REPLAY), via the engine. Until
        # it is declared LIVE every decision is SIMULATED, i.e. never valid for control.
        self.camera_origin = SOURCE_SIMULATED

    def attach(self, config: OverlayConfig) -> None:
        """A (re)built pipeline starts a fresh assembler: it insists on timestamp order."""
        self.config = config
        self.guard = TagSwitchGuard(config.max_position_disagreement_m)
        self.assembler = ShadowAssembler(self.contract, self.input_source, self.telemetry, policy=self.policy,
                                         shadow_label=LIVE_LABEL, max_camera_age_s=config.max_detection_age_s,
                                         max_imu_age_s=MAX_IMU_AGE_S, camera_origin=self.camera_origin)

    def set_policy(self, policy) -> None:
        self.policy, self.error = policy, ""
        if self.assembler is not None:
            self.assembler.policy = policy

    def decide(self, sample: CameraSample, t: float, now: float) -> dict[str, Any]:
        if self.assembler is None:
            self.attach(self.config)
        sample = replace(sample)                                  # the guard may refuse the pose; the shared sample stays as it was
        checked = sample.own_track is None or sample.own_track["state"] == "measured"
        refused = self.guard.check(sample.own, t) if checked else ""
        if refused:
            sample.own = BotPose(reason=refused, visible_ids=sample.own.visible_ids)
            sample.own_at = None
        self.assembler.camera_origin = self.camera_origin
        self.assembler.offer(sample)
        try:
            decision = self.assembler.decide(max(t, now))
        except Exception as error:                                # not an inference failure (the assembler blocks those itself)
            log.exception("shadow decision failed")
            self.error = f"{type(error).__name__}: {error}"
            self.assembler.policy = None
            decision = self.assembler.decide(max(t, now))
        self.last_decision = decision
        return {"proposed": decision.proposed, "blocked_by": list(decision.blocked_by), "mode": decision.mode,
                "policy_loaded": self.assembler.policy is not None, "error": self.error or decision.error,
                "source": decision.source, "valid_for_control": decision.valid_for_control,
                "ages_ms": {group: None if age is None else round(1000 * age, 1) for group, age in decision.ages_s.items()},
                "inputs": {group: {"valid": status.valid, "reason": status.reason, "source": status.source}
                           for group, status in decision.inputs.items()},
                "tag_layout_fault": self.guard.message if self.guard.trips else "",
                "output": "none: display only, nothing is sent to the robot"}


# ---------------------------------------------------------------- the engine


def _clicks_for_tracking(clicks: ClickSet | None) -> tuple[ClickSet, str]:
    """The ClickSet the trackers get, and a note when it was refused or absent."""
    if clicks is None:
        return ClickSet(), "no calibration saved for this profile"
    if clicks.historical:
        return ClickSet(), "the saved points are a HISTORICAL footage profile; they are never used for the live view"
    if not clicks.points:
        return ClickSet(), "the profile has no clicked points"
    return clicks, ""


class TrackingEngine:
    """One frame in, one ``TrackingResult`` out. No camera, no Qt, no clock of its own beyond ``clock``."""

    def __init__(self, config: OverlayConfig, clicks: ClickSet | None, profile: str = "live", window_s: float = 5.0,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self.clock = clock
        self.window_s = window_s
        self.generation = 0
        self.epoch = 0                       # set by the GUI with each rebuild; results of an older epoch are not shown
        self.config, self.profile = config, profile
        self.clicks: ClickSet = ClickSet()
        self.clicks_note = ""
        self.track: LiveTrack | None = None
        self.size: tuple[int, int] | None = None
        self.rates = MarkerRateTracker(config.markers, window_s)
        self.diagnostics = DiagnosticsModel()
        self.shadow: ShadowRunner | None = None
        self.camera_state: str | None = None   # the capture session's state, set by TrackingWorker; None = undeclared = SIMULATED
        self.seq = 0
        self._notes: list[str] = []
        self._click_events: list[dict[str, Any]] = []
        self._orientation: str | None = None
        self._fit_reasons: list[str] = []
        self.rebuild(config, clicks, profile, "")

    # ---- commands (called on the thread that processes frames) ----

    def rebuild(self, config: OverlayConfig | None = None, clicks: ClickSet | None = None, profile: str | None = None,
                note: str = "", epoch: int | None = None) -> None:
        """Drop every tracker and the opponent selection; the next frame fits the mapping afresh."""
        if epoch is not None:
            self.epoch = epoch
        if config is not None:
            self.config = config
        if profile is not None:
            self.profile = profile
        if clicks is not None or config is not None or self.generation == 0:
            self.clicks, self.clicks_note = _clicks_for_tracking(clicks)
        self.track, self.size = None, None
        self.rates = MarkerRateTracker(self.config.markers, self.window_s)
        self.diagnostics.reset()
        self._orientation, self._fit_reasons = None, []
        self.generation += 1
        if self.shadow is not None:
            self.shadow.attach(self.config)
        if note:
            self._notes.append(note)

    def set_shadow(self, runner: ShadowRunner | None) -> None:
        self.shadow = runner
        if runner is not None:
            runner.attach(self.config)

    def select_opponent(self, pixel: tuple[float, float], frame: np.ndarray, t: float) -> None:
        """A click, on the frame the operator was looking at. The outcome is reported in the next result."""
        size = (frame.shape[1], frame.shape[0])
        if self.track is None or self.size != size:
            self._build(size)
        result = self.track.pipeline.select_opponent(frame, pixel, t)
        self._click_events.append({"event": "click", "pixel": [float(pixel[0]), float(pixel[1])], "accepted": bool(result.ok),
                                   "reason": result.reason})

    def select_roi(self, rect: tuple[float, float, float, float], frame: np.ndarray, t: float) -> None:
        """A box dragged on the frame the operator was looking at. It replaces any current selection when accepted."""
        size = (frame.shape[1], frame.shape[0])
        if self.track is None or self.size != size:
            self._build(size)
        result = self.track.pipeline.select_roi(frame, rect, t)
        self._click_events.append({"event": "roi", "rect": [float(v) for v in rect], "accepted": bool(result.ok), "reason": result.reason})

    def clear_opponent(self) -> None:
        if self.track is not None and self.track.pipeline is not None:
            self.track.pipeline.clear_opponent()
        self._click_events.append({"event": "clear", "accepted": True, "reason": "selection cleared"})

    # ---- frames ----

    def _build(self, size: tuple[int, int]) -> None:
        self.track = LiveTrack(self.config, self.clicks, opponent_mode="roi")      # the opponent is chosen by dragging a box
        self.track.start(size)
        self.size = size
        self._fit_reasons = []
        if self.track.fit_level == "weak":
            self._fit_reasons = list(fit_profile(self.config, self.clicks, size)[2].reasons)
        if self.shadow is not None:
            self.shadow.attach(self.config)

    def _map_state(self) -> tuple[str, str]:
        pipeline = self.track.pipeline
        if not self.clicks.points:
            return "none", self.clicks_note or "no clicked points"
        if not pipeline.mapping.valid:
            return "invalid", pipeline.mapping.reason
        return ("ok" if self.track.fit_level == "ok" else "weak"), "; ".join(self._fit_reasons)

    def _evidence(self, own: dict[str, Any] | None) -> dict[str, Any] | None:
        """Quantities of the newest frame's fusion (marker fit rms, spread). Read-only look at the chassis tracker."""
        chassis = getattr(self.track.pipeline, "chassis", None)
        frame = getattr(chassis, "_frame", None)
        if own is None or own["state"] != "measured" or frame is None or not getattr(frame, "ok", False):
            return None
        spread, turn = marker_spread(frame.markers, frame.used_ids)
        return {"fit_rms_mm": frame.fit_rms_mm, "spread_mm": spread, "heading_spread_deg": turn}

    def process(self, index: int, t: float, frame: np.ndarray, skipped: int = 0, queue_age_s: float = 0.0) -> TrackingResult:
        started = self.clock()
        size = (frame.shape[1], frame.shape[0])
        if self.track is None:
            self._build(size)
        elif size != self.size:
            self._notes.append(f"camera picture changed size {self.size[0]}x{self.size[1]} -> {size[0]}x{size[1]}: "
                               "trackers and the opponent selection were cleared")
            self.rates = MarkerRateTracker(self.config.markers, self.window_s)
            self._build(size)
        happened, self._click_events = self._click_events, []
        try:
            sample, _ = self.track.finish(frame, t, [])
        except ValueError as error:                           # frames out of timestamp order: start again rather than guess
            self._notes.append(f"tracking restarted: {error}")
            self._build(size)
            sample, _ = self.track.finish(frame, t, [])
        pipeline = self.track.pipeline
        own, opponent = sample.own_track, sample.opponent_track
        if own is not None and own.get("orientation") in ("upright", "inverted"):
            self._orientation = own["orientation"]
        self.rates.update(t, sample.marker_ids)
        expected = {"upright": ("top", "rear"), "inverted": ("bottom", "rear")}.get(self._orientation)
        rows = self.rates.rows(expected)
        shadow = None
        if self.shadow is not None:
            self.shadow.camera_origin = origin_from_capture_state(self.camera_state)
            shadow = self.shadow.decide(sample, t, self.clock())
        image = frame.copy()
        if pipeline.mapping.valid and not arena_problems(self.config.arena):
            draw_arena_overlay(image, pipeline.mapping, self.config.arena)
        draw_tracking(image, pipeline, sample)
        draw_heading(image, pipeline.mapping, own)
        map_level, map_reason = self._map_state()
        done = self.clock()
        engine_ms = 1000.0 * (done - started)
        self.diagnostics.record(done, index, sample.timing_ms["total"], queue_age_s, skipped)
        notes, self._notes = self._notes, []
        self.seq += 1
        return TrackingResult(
            seq=self.seq, frame_index=index, t=t, done_at=done, size=size, raw=frame, annotated=image, sample=sample,
            own=own, opponent=opponent, marker_ids=list(sample.marker_ids), marker_rows=rows,
            marker_frames=self.rates.frames_in_window, evidence=self._evidence(own),
            limits={"position_m": self.config.max_position_disagreement_m, "heading_deg": self.config.max_heading_disagreement_deg,
                    "predict_hold_s": self.track.tracker_settings.predict_hold_s},
            dictionary=self.config.dictionary, placeholders=list(self.config.placeholders), map_level=map_level,
            map_reason=map_reason, fit_level=self.track.fit_level, fit_reasons=list(self._fit_reasons),
            arena_problems=arena_problems(self.config.arena), software_ms=sample.timing_ms["total"], engine_ms=engine_ms,
            skipped=skipped, queue_age_s=queue_age_s, generation=self.generation, epoch=self.epoch,
            diagnostics=self.diagnostics.summary(done), notes=notes, click_events=happened, shadow=shadow)


# ---------------------------------------------------------------- the thread


class TrackingWorker(QThread):
    """Pulls the newest frame from the hub's capture session and runs the engine. The GUI only ever calls ``latest()``."""

    def __init__(self, capture, engine: TrackingEngine, parent=None) -> None:
        super().__init__(parent)
        self.capture, self.engine = capture, engine
        self._halt = threading.Event()
        self._active = threading.Event()
        self._commands: deque[tuple] = deque()
        self._lock = threading.Lock()
        self._latest: TrackingResult | None = None
        self._paused_at: float | None = None
        self.thread_ident: int | None = None           # the thread that ran the engine (tests assert it is not the GUI thread)
        self.processed = 0
        self.last_error = ""

    # ---- GUI-side API (thread-safe) ----

    def post(self, *command: Any) -> None:
        with self._lock:
            self._commands.append(command)

    def latest(self) -> TrackingResult | None:
        with self._lock:
            return self._latest

    def resume(self) -> None:
        self._active.set()

    def pause(self) -> None:
        self._active.clear()

    @property
    def paused(self) -> bool:
        return not self._active.is_set()

    def stop(self) -> None:
        self._halt.set()
        self._active.set()

    # ---- the loop ----

    def _apply(self) -> None:
        with self._lock:
            commands, self._commands = list(self._commands), deque()
        for command in commands:
            kind = command[0]
            try:
                if kind == "rebuild":
                    self.engine.rebuild(command[1], command[2], command[3], command[4], command[5])
                elif kind == "click":
                    self.engine.select_opponent(command[1], command[2], command[3])
                elif kind == "roi":
                    self.engine.select_roi(command[1], command[2], command[3])
                elif kind == "clear":
                    self.engine.clear_opponent()
                elif kind == "shadow":
                    self.engine.set_shadow(command[1])
                elif kind == "policy" and self.engine.shadow is not None:
                    self.engine.shadow.set_policy(command[1])
            except Exception:
                log.exception("tracking command %s failed", kind)

    def run(self) -> None:
        self.thread_ident = threading.get_ident()
        seen = -1
        was_paused_at: float | None = None
        while not self._halt.is_set():
            if not self._active.is_set():
                if was_paused_at is None:
                    was_paused_at = time.monotonic()
                self._active.wait(0.1)
                continue
            if was_paused_at is not None:
                if time.monotonic() - was_paused_at > PAUSE_REBUILD_AFTER_S:
                    self.engine.rebuild(note="tracking restarted: the tab was hidden, so every track and the opponent selection "
                                             "were dropped (click the opponent again)")
                was_paused_at, seen = None, -1
            self._apply()
            try:
                index, t, frame = self.capture.wait_for_frame(seen, 0.1)
            except Exception:
                log.exception("capture wait failed")
                time.sleep(0.1)
                continue
            if frame is None or index == seen:
                continue
            skipped = max(0, index - seen - 1) if 0 <= seen < index else 0
            seen = index
            picked_up = time.monotonic()
            self.engine.camera_state = getattr(self.capture, "state", None)      # LIVE / MOCK / REPLAY decides the decision's source
            try:
                result = self.engine.process(index, t, frame, skipped, max(0.0, picked_up - t))
            except Exception as error:
                log.exception("tracking failed on frame %d", index)
                self.last_error = f"{type(error).__name__}: {error}"
                continue
            self.last_error = ""
            self.processed += 1
            with self._lock:
                self._latest = result


# ---------------------------------------------------------------- checkpoint loading, off the GUI thread


class CheckpointLoader(QObject):
    """Builds and loads the policy (about 30-60 s with torch) on a daemon thread. Emits ``loaded(policy)`` or ``failed(text)``.

    A plain daemon thread, not a QThread: a checkpoint load cannot be interrupted, and a running QThread that is destroyed
    aborts the process, while a daemon thread simply ends with it.
    """

    loaded = Signal(object)
    failed = Signal(str)

    def __init__(self, factory: Callable[[Any], Any], contract, parent=None) -> None:
        super().__init__(parent)
        self.factory, self.contract = factory, contract
        self.started_at = time.monotonic()
        self.thread_ident: int | None = None
        self._thread: threading.Thread | None = None
        self._abandoned = False

    def start(self) -> None:
        self.started_at = time.monotonic()
        self._thread = threading.Thread(target=self._run, name="checkpoint-load", daemon=True)
        self._thread.start()

    def isRunning(self) -> bool:  # noqa: N802 - mirrors QThread
        return self._thread is not None and self._thread.is_alive()

    def abandon(self) -> None:
        """The screen is closing: whatever the load produces is no longer wanted."""
        self._abandoned = True

    def _emit(self, signal, value) -> None:
        if self._abandoned:
            return
        try:
            signal.emit(value)
        except RuntimeError:                         # the owning screen was deleted meanwhile
            pass

    def _run(self) -> None:
        self.thread_ident = threading.get_ident()
        try:
            policy = self.factory(self.contract)
            policy.load()
        except Exception as error:
            log.exception("checkpoint load failed")
            self._emit(self.failed, f"{type(error).__name__}: {error}")
            return
        self._emit(self.loaded, policy)
