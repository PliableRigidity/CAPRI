"""A camera-like stream turned into a checkpoint's observation, one frame at a time. Read-only.

Three stages, each of which only ever sees frames that have already arrived:

  CameraPipeline     one frame -> own-bot pose from ArUco markers (existing overlay_core code) and
                     unmarked blobs on the floor (existing blob_tracker code), then a decision about
                     which blob, if any, is the opponent. Unresolved means ``unknown``, never a guess.
  ShadowAssembler    at a decision time: the newest frame result and the newest telemetry sample, each
                     with its age and validity -> either the complete observation vector or the exact
                     list of fields that block it.
  CheckpointPolicy   a complete observation -> the tactic the checkpoint would choose. Display only.

There is no output path: this module never imports a serial port, the controller link or the command
manager, and a proposal is a dictionary that goes into a trace file. Nothing is held or predicted across
a gap, and no simulator value is ever used to fill a missing input.
"""

from __future__ import annotations

import bisect
import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

import numpy as np

from control_center.backend.observation_builder import (SOURCE_REPLAY, SOURCE_SIMULATED, combine_origins,
                                                        control_validity)
from control_center.tools.blob_tracker import OVERLAP, TRACKED, BlobTracker, median_background
from control_center.tools.observation_contract import (GROUP_IMU, GROUP_OPPONENT, GROUP_OWN, GROUP_PREVIOUS,
                                                       ObservationContract)
from control_center.tools.overlay_core import (UNKNOWN, BotPose, CameraEstimate, ClickSet, FitQuality, FloorMapping,
                                               MarkerDetector, OverlayConfig, PixelModel, assess_fit,
                                               estimate_bot_pose, fit_floor_mapping, resolve_camera)

MODE_BLOCKED = "BLOCKED"
MODE_COMPLETE = "OBSERVATION COMPLETE (no checkpoint loaded)"
OK = "ok"


# ----------------------------------------------------------------------------------------------
# Floor map from a calibration profile (same steps as the overlay tool's refit)


def fit_profile(config: OverlayConfig, clicks: ClickSet, image_size: tuple[int, int]) -> tuple[FloorMapping, CameraEstimate, FitQuality]:
    model = clicks.lens if clicks.lens.active else PixelModel(config.camera_matrix, config.distortion)
    mapping = fit_floor_mapping(config.references, clicks.points, image_size, model, roles=clicks.roles,
                                clicks_image_size=clicks.image_size, max_fit_residual_mm=config.max_fit_residual_mm)
    camera = resolve_camera(config, mapping)
    quality = assess_fit(config.references, clicks.points, clicks.roles, mapping, config.arena, config.warn_check_error_mm,
                         config.weak_predicted_error_mm, config.min_hull_fraction)
    return mapping, camera, quality


# ----------------------------------------------------------------------------------------------
# Telemetry (IMU). Nothing in this repository supplies it live; the interface exists so that a real
# source, or an explicitly synthetic fixture stream, can be connected without touching the rest.


@dataclass(frozen=True)
class ImuSample:
    t: float                                    # stream clock, seconds
    gyro_rad_s: tuple[float, float, float]      # body frame of the contract (x nose, y left, z up)
    accel_m_s2: tuple[float, float, float]


class TelemetrySource(Protocol):
    description: str

    def latest(self, t: float) -> ImuSample | None: ...


class NoTelemetry:
    description = "none connected"
    origin = SOURCE_SIMULATED                     # nothing arrives, so nothing can be live

    def latest(self, t: float) -> ImuSample | None:
        return None


class RecordedTelemetry:
    """Samples with their own timestamps; ``latest(t)`` never returns one from after ``t``."""

    origin = SOURCE_REPLAY                        # recorded samples are a replay, never LIVE (see observation_builder)

    def __init__(self, samples: list[ImuSample], description: str) -> None:
        self.samples = sorted(samples, key=lambda sample: sample.t)
        self._times = [sample.t for sample in self.samples]
        self.description = description

    def latest(self, t: float) -> ImuSample | None:
        index = bisect.bisect_right(self._times, t + 1e-9) - 1
        return self.samples[index] if index >= 0 else None


# ----------------------------------------------------------------------------------------------
# One frame


@dataclass
class OpponentEstimate:
    state: str = UNKNOWN                        # "ok" or "unknown"
    reason: str = "no frame yet"
    xy_m: tuple[float, float] | None = None     # only when state is "ok"
    track_id: int | None = None
    own_track_id: int | None = None             # the blob that is the tagged own bot, when known
    own_identified_by: str = "nothing"          # "tag this frame", "same blob as when the tag was last read", "nothing"
    candidates: int = 0                         # unmarked blobs that are not the own bot
    source: str = ""                            # set by the click-selected tracker; "" = the blob tracker's wording

    @property
    def ok(self) -> bool:
        return self.state == OK


@dataclass
class CameraSample:
    frame_index: int
    t: float
    own: BotPose
    opponent: OpponentEstimate
    marker_ids: list[int] = field(default_factory=list)
    tracks: list[dict[str, Any]] = field(default_factory=list)
    blob_count: int = 0
    timing_ms: dict[str, float] = field(default_factory=dict)
    # Only set by the opt-in trackers (CameraPipeline(multi_marker=True / click_opponent=True)):
    own_track: dict[str, Any] | None = None     # chassis tracker report at the frame's own time
    opponent_track: dict[str, Any] | None = None
    own_at: Callable[[float], Any] | None = None   # chassis tracker state at a later query time

    def to_report(self) -> dict[str, Any]:
        report = self._base_report()
        if self.own_track is not None:
            report["own_track"] = self.own_track
        if self.opponent_track is not None:
            report["opponent_track"] = self.opponent_track
        return report

    def _base_report(self) -> dict[str, Any]:
        own = self.own
        return {"frame": self.frame_index, "t_s": round(self.t, 6),
                "own": {"state": own.state, "reason": own.reason,
                        "x_m": _r(own.x_m) if own.ok else None, "y_m": _r(own.y_m) if own.ok else None,
                        "heading_deg": _r(own.heading_deg, 2) if own.ok else None, "used_ids": own.used_ids,
                        "visible_ids": own.visible_ids, "position_quality": own.position_quality},
                "opponent": {"state": self.opponent.state, "reason": self.opponent.reason,
                             "x_m": _r(self.opponent.xy_m[0]) if self.opponent.ok else None,
                             "y_m": _r(self.opponent.xy_m[1]) if self.opponent.ok else None,
                             "track": self.opponent.track_id, "own_track": self.opponent.own_track_id,
                             "own_identified_by": self.opponent.own_identified_by, "candidates": self.opponent.candidates},
                "tracks": self.tracks, "blobs": self.blob_count,
                "processing_ms": {key: round(value, 3) for key, value in self.timing_ms.items()}}


def _r(value: float | None, digits: int = 4) -> float | None:
    return None if value is None else round(float(value), digits)


class OpponentResolver:
    """Decides which unmarked blob is the opponent. The tag is the only thing that says which blob is ours.

    * The blob under the own bot's tag pose is the own bot. That label stays with the blob only while the
      tracker keeps following the same blob on its own; an overlap or a loss drops it.
    * The opponent is known only when the own bot is accounted for and exactly one other blob is tracked.
    * While the own bot shares a blob with something, the tracker reports no position for either: unknown.
    """

    def __init__(self, own_gate_m: float = 0.10, overlap_gate_m: float = 0.22) -> None:
        self.own_gate_m, self.overlap_gate_m = own_gate_m, overlap_gate_m
        self.own_track_id: int | None = None

    @staticmethod
    def _distance(a: tuple[float, float], b: tuple[float, float]) -> float:
        return math.hypot(a[0] - b[0], a[1] - b[1])

    def resolve(self, tracker: BlobTracker | None, own: BotPose, no_tracker_reason: str = "") -> OpponentEstimate:
        if tracker is None:
            self.own_track_id = None
            return OpponentEstimate(reason=no_tracker_reason or "no blob tracker running")
        tracked = [track for track in tracker.tracks if track.state == TRACKED and track.floor is not None]
        own_in_overlap, own_has_no_blob, identified_by = False, False, "nothing"
        if own.ok:
            # A blob centre is not height-corrected, so compare it with the tag pose's raw floor projection.
            anchor = own.floor_projection_xy or (own.x_m, own.y_m)
            near = sorted((self._distance(track.floor, anchor), track.track_id) for track in tracked
                          if self._distance(track.floor, anchor) <= self.own_gate_m)
            if near:
                self.own_track_id, identified_by = near[0][1], "tag this frame"
            else:
                self.own_track_id = None
                own_in_overlap = any(self._distance(position, anchor) <= self.overlap_gate_m for position in tracker.overlap_at.values())
                own_has_no_blob = not own_in_overlap
                identified_by = "tag this frame"
        elif self.own_track_id is not None:
            if any(track.track_id == self.own_track_id for track in tracked):
                identified_by = "same blob as when the tag was last read"
            else:
                self.own_track_id = None        # that blob merged or vanished: the label does not survive it
        candidates = [track for track in tracked if track.track_id != self.own_track_id]
        estimate = OpponentEstimate(own_track_id=self.own_track_id, own_identified_by=identified_by, candidates=len(candidates))
        own_accounted = self.own_track_id is not None or own_in_overlap or own_has_no_blob
        overlapping = sum(1 for track in tracker.tracks if track.state == OVERLAP)
        if own_in_overlap:
            estimate.reason = ("overlap: the own bot shares one blob with another object, so the tracker gives no "
                               "position for either")
        elif not own_accounted:
            count = len(tracked)
            what = "no unmarked blob is tracked" if not count else f"{count} unmarked blob(s) tracked"
            if overlapping:
                what += f", {overlapping} track(s) merged in an overlap"
            estimate.reason = f"own bot not identified by a tag, so no blob can be called the opponent ({what})"
        elif len(candidates) == 1:
            track = candidates[0]
            estimate.state, estimate.reason = OK, OK
            estimate.xy_m, estimate.track_id = (float(track.floor[0]), float(track.floor[1])), track.track_id
        elif not candidates:
            if overlapping:
                estimate.reason = "overlap: the other object's blob has merged with another blob (no position)"
            else:
                estimate.reason = ("no separate unmarked blob on the floor (the other object is absent, was in the "
                                   "startup picture, or is touching the own bot inside one blob)")
        else:
            estimate.reason = f"{len(candidates)} unmarked blobs besides the own bot: cannot tell which is the opponent"
        return estimate


class CameraPipeline:
    """Per frame: markers -> own pose; blobs -> opponent. Uses only this frame and earlier ones."""

    def __init__(self, config: OverlayConfig, mapping: FloorMapping, camera: CameraEstimate,
                 own_gate_m: float = 0.10, overlap_gate_m: float = 0.22, multi_marker: bool = False,
                 click_opponent: bool = False, tracker_settings: Any = None, opponent_settings: Any = None,
                 opponent_mode: str = "click") -> None:
        """``multi_marker`` / ``click_opponent`` are opt-in; without them this is the pipeline it always was.

        multi_marker    own pose from ``chassis_tracker.ChassisTracker`` (consensus + short prediction)
        click_opponent  opponent from ``opponent_select.ClickedOpponentTracker`` (``select_opponent``); the
                        empty-floor reference picture and the blob tracker are then not needed for it
        opponent_mode   "click" (default): the click-selected colour-segmentation tracker, with its size checks.
                        "roi": ``roi_tracker.RoiOpponentTracker``, selected by dragging a box (``select_roi``) around
                        any object, with no physical size limit. Only used together with ``click_opponent=True``.
        """
        self.config, self.mapping, self.camera = config, mapping, camera
        self.detector = MarkerDetector(config.dictionary)
        self.tracker: BlobTracker | None = None
        self.reference_note = "no startup reference picture: unmarked objects cannot be seen"
        self.resolver = OpponentResolver(own_gate_m, overlap_gate_m)
        self.frames = 0
        self.chassis = None
        self.clicked = None
        if multi_marker:
            from control_center.tools.chassis_tracker import ChassisTracker
            self.chassis = ChassisTracker(config, mapping, camera, tracker_settings)
        if click_opponent and opponent_mode == "roi":
            from control_center.tools.roi_tracker import RoiOpponentTracker
            self.clicked = RoiOpponentTracker(mapping, config.arena, opponent_settings)
        elif click_opponent:
            from control_center.tools.opponent_select import ClickedOpponentTracker
            self.clicked = ClickedOpponentTracker(mapping, config.arena, opponent_settings)
        self._own_estimate: Any = None

    def select_opponent(self, frame: np.ndarray, pixel_xy: tuple[float, float], t: float) -> Any:
        """A click on the camera picture (click_opponent only). Returns the tracker's Selection (ok, reason)."""
        if self.clicked is None:
            raise ValueError("this pipeline was not built with click_opponent=True")
        own = None if self._own_estimate is None else self._own_estimate.to_bot_pose()
        return self.clicked.select(frame, pixel_xy, t, own)

    def select_roi(self, frame: np.ndarray, rect: tuple[float, float, float, float], t: float) -> Any:
        """A box dragged on the camera picture (opponent_mode="roi"). Returns the selection (ok, reason)."""
        if self.clicked is None or not hasattr(self.clicked, "select_roi"):
            raise ValueError("this pipeline was not built with opponent_mode='roi'")
        own = None if self._own_estimate is None else self._own_estimate.to_bot_pose()
        hint = None if self._own_estimate is None or self._own_estimate.x_m is None else (self._own_estimate.x_m, self._own_estimate.y_m)
        return self.clicked.select_roi(frame, rect, t, own, hint)

    def clear_opponent(self) -> None:
        if self.clicked is not None:
            self.clicked.clear()

    def set_reference(self, frames: list[np.ndarray], note: str) -> None:
        """The empty-floor picture the blob tracker compares against. Must be taken BEFORE the stream starts."""
        if not frames:
            raise ValueError("the startup reference needs at least one frame")
        if not self.mapping.valid:
            raise ValueError(f"the blob tracker needs a valid floor map: {self.mapping.reason}")
        self.tracker = BlobTracker(self.mapping, self.config.arena, frames[0] if len(frames) == 1 else median_background(frames))
        self.resolver = OpponentResolver(self.resolver.own_gate_m, self.resolver.overlap_gate_m)
        self.reference_note = note

    def process(self, frame: np.ndarray, t: float) -> CameraSample:
        start = time.perf_counter()
        detections = self.detector.detect_markers(frame)
        after_markers = time.perf_counter()
        estimate = None
        if self.chassis is not None:
            estimate = self._own_estimate = self.chassis.update(detections, t)
            own = estimate.to_bot_pose()
        else:
            own = estimate_bot_pose(detections, self.mapping, self.camera, self.config)
        after_pose = time.perf_counter()
        if self.tracker is not None:
            self.tracker.update(frame)
        after_blobs = time.perf_counter()
        if self.clicked is not None:
            # A stale own pose is not a position, but it still says where NOT to look for the opponent.
            hint = None if estimate is None or own.ok or estimate.x_m is None else (estimate.x_m, estimate.y_m)
            self.clicked.update(frame, t, own, hint)
            opponent = self.clicked.to_opponent_estimate()
        else:
            opponent = self.resolver.resolve(self.tracker, own, self.reference_note)
        end = time.perf_counter()
        sample = CameraSample(
            self.frames, t, own, opponent, sorted(d.tag_id for d in detections),
            [] if self.tracker is None else [track.to_report() for track in self.tracker.tracks],
            0 if self.tracker is None else len(self.tracker.blobs),
            {"markers": 1000 * (after_markers - start), "pose": 1000 * (after_pose - after_markers),
             "blobs": 1000 * (after_blobs - after_pose), "opponent": 1000 * (end - after_blobs), "total": 1000 * (end - start)})
        if self.chassis is not None:
            sample.own_track, sample.own_at = estimate.to_report(), self.chassis.status
        if self.clicked is not None:
            sample.opponent_track = self.clicked.to_report()
        self.frames += 1
        return sample


# ----------------------------------------------------------------------------------------------
# A decision time: what is valid, how old it is, and what blocks the observation


@dataclass
class InputStatus:
    group: str
    indices: str
    valid: bool
    source: str
    age_ms: float | None
    reason: str
    values: dict[str, float] = field(default_factory=dict)

    def to_report(self) -> dict[str, Any]:
        return {"indices": self.indices, "valid": self.valid, "source": self.source,
                "age_ms": None if self.age_ms is None else round(self.age_ms, 1), "reason": self.reason,
                "values": {key: round(float(value), 4) for key, value in self.values.items()}}


@dataclass
class Decision:
    tick: int
    t: float
    mode: str
    input_source: str
    checkpoint: str
    frame_index: int | None
    frame_t: float | None
    inputs: dict[str, InputStatus]
    observation: np.ndarray | None
    blocked_by: list[str]
    proposed: dict[str, Any] | None
    # Provenance (control_center/backend/observation_builder.py). The safe defaults say "not valid for control".
    source: str = SOURCE_SIMULATED               # LIVE only when the camera AND the IMU were declared LIVE
    valid_for_control: bool = False
    ages_s: dict[str, float | None] = field(default_factory=dict)
    error: str = ""                              # an inference failure, when there was one (the decision is then blocked)

    def to_trace(self) -> dict[str, Any]:
        return {"tick": self.tick, "t_s": round(self.t, 6), "mode": self.mode, "input_source": self.input_source,
                "source": self.source, "valid_for_control": self.valid_for_control,
                "ages_ms": {k: None if v is None else round(1000 * v, 1) for k, v in self.ages_s.items()}, "error": self.error,
                "checkpoint": self.checkpoint, "camera_frame": self.frame_index,
                "camera_frame_t_s": None if self.frame_t is None else round(self.frame_t, 6),
                "inputs": {group: status.to_report() for group, status in self.inputs.items()},
                "observation": None if self.observation is None else [round(float(v), 5) for v in self.observation],
                "blocked_by": self.blocked_by, "proposed": self.proposed,
                "output": "none: display and trace only, nothing is sent to the robot"}

    def ui_lines(self) -> list[tuple[str, str]]:
        """What a dashboard should show for this decision: (text, level) with level ok / bad / note."""
        labels = {GROUP_IMU: "IMU", GROUP_PREVIOUS: "previous tactic", GROUP_OWN: "own pose", GROUP_OPPONENT: "opponent"}
        lines = [(f"POLICY   {self.checkpoint}", "note"), (f"INPUT    {self.input_source}", "note")]
        if self.frame_t is None:
            lines.append((f"t = {self.t:.2f} s   no camera frame yet", "bad"))
        else:
            lines.append((f"t = {self.t:.2f} s   frame {self.frame_index}, {1000 * (self.t - self.frame_t):.0f} ms old "
                          "(stream clock; real camera latency not included)", "note"))
        for group, status in self.inputs.items():
            age = "" if status.age_ms is None or not status.valid else f"  age {status.age_ms:.0f} ms"
            word = "OK     " if status.valid else ("UNKNOWN" if group == GROUP_OPPONENT else "MISSING")
            detail = status.source + age if status.valid else status.reason
            lines.append((f"[{status.indices:>5}] {labels[group]:<15} {word}  {detail}", "ok" if status.valid else "bad"))
        if self.proposed is not None:
            lines.append((f"MODE     {self.mode}: proposal only, nothing is sent", "ok"))
            lines.append((f"PROPOSED {self.proposed['tactic']}  (p = {self.proposed['probability']:.2f})", "ok"))
        elif self.observation is not None:
            lines.append((f"MODE     {self.mode}", "note"))
        else:
            spans = ", ".join(entry.split("]")[0].lstrip("[") for entry in self.blocked_by)
            lines.append((f"MODE     {MODE_BLOCKED}: the checkpoint is not run", "bad"))
            lines.append((f"PROPOSED none (missing values {spans})", "bad"))
        return lines


class ShadowAssembler:
    """Builds the observation at a decision time from whatever has arrived, or says what is missing."""

    def __init__(self, contract: ObservationContract, input_source: str, telemetry: TelemetrySource | None = None,
                 policy: "CheckpointPolicy | None" = None, shadow_label: str = "SHADOW",
                 max_camera_age_s: float = 0.25, max_imu_age_s: float = 0.10,
                 camera_origin: str = SOURCE_SIMULATED, imu_origin: str | None = None) -> None:
        """``camera_origin`` / ``imu_origin`` declare where the inputs come from (LIVE / SIMULATED / REPLAY). The default is
        SIMULATED, so nothing is ``valid_for_control`` unless the caller has said its inputs are LIVE. ``imu_origin`` None =
        the telemetry object's own ``origin`` attribute (SIMULATED when it has none)."""
        self.contract, self.input_source = contract, input_source
        self.camera_origin, self.imu_origin = camera_origin, imu_origin
        self.last_error = ""
        self.telemetry = telemetry or NoTelemetry()
        self.policy, self.shadow_label = policy, shadow_label
        self.max_camera_age_s, self.max_imu_age_s = max_camera_age_s, max_imu_age_s
        self.sample: CameraSample | None = None
        self.ticks = 0

    def offer(self, sample: CameraSample) -> None:
        if self.sample is not None and sample.t < self.sample.t:
            raise ValueError("camera samples must arrive in timestamp order")
        self.sample = sample

    def _camera_problem(self, t: float) -> tuple[float | None, str]:
        if self.sample is None:
            return None, "no camera frame has arrived yet"
        age = t - self.sample.t
        if age < -1e-9:
            raise ValueError("a decision cannot be made before the newest frame's timestamp")
        age = max(0.0, age)
        if age > self.max_camera_age_s:
            return age, f"stale: the newest camera frame is {1000 * age:.0f} ms old (limit {1000 * self.max_camera_age_s:.0f} ms)"
        return age, ""

    @staticmethod
    def _own_at(sample: CameraSample, t: float) -> tuple[BotPose, str]:
        """The own pose a decision at ``t`` may use, and the words for where it came from.

        Without the chassis tracker this is the frame's own pose, as before. With it, the tracker is asked for
        its state AT THE DECISION TIME, so a pose that was valid when the frame arrived is predicted forward,
        and is refused once it is stale or lost. A pose replaced after the frame (``own_at`` cleared) stays refused.
        """
        if sample.own_at is None:
            own = sample.own
            return own, f"ArUco ids {own.used_ids}, {own.position_quality} ({own.correction_source})"
        estimate = sample.own_at(t)
        own = estimate.to_bot_pose()
        detail = f"{own.position_quality} ({own.correction_source})"
        if getattr(estimate, "source", "") == "rear marker only":
            detail += ", REAR MARKER ONLY (lower confidence)"
        if estimate.state.value == "predicted":
            return own, f"ArUco ids {own.used_ids} PREDICTED {1000 * estimate.age_s:.0f} ms, {detail}"
        return own, f"ArUco ids {own.used_ids}, {detail}"

    def decide(self, t: float) -> Decision:
        c, sample = self.contract, self.sample
        camera_age, camera_problem = self._camera_problem(t)
        age_ms = None if camera_age is None else 1000 * camera_age
        inputs: dict[str, InputStatus] = {}

        imu = self.telemetry.latest(t)
        if imu is None:
            inputs[GROUP_IMU] = InputStatus(GROUP_IMU, c.span(GROUP_IMU), False, "none", None,
                                            f"no IMU telemetry ({self.telemetry.description})")
        else:
            imu_age = max(0.0, t - imu.t)
            stale = imu_age > self.max_imu_age_s
            inputs[GROUP_IMU] = InputStatus(
                GROUP_IMU, c.span(GROUP_IMU), not stale, self.telemetry.description, 1000 * imu_age,
                f"stale: the newest IMU sample is {1000 * imu_age:.0f} ms old (limit {1000 * self.max_imu_age_s:.0f} ms)" if stale else OK,
                {"gyro_z_rad_s": imu.gyro_rad_s[2], "accel_z_m_s2": imu.accel_m_s2[2]})

        if c.indices(GROUP_PREVIOUS):
            masked = c.previous_action_encoding == "masked_zero_v1"
            inputs[GROUP_PREVIOUS] = InputStatus(
                GROUP_PREVIOUS, c.span(GROUP_PREVIOUS), masked, "constant 0.0 (masked)" if masked else "none", None,
                OK if masked else "this checkpoint reads the tactic it last APPLIED; in shadow nothing is applied, so there is none")

        if c.indices(GROUP_OWN):
            own, own_source = (None, "") if camera_problem else self._own_at(sample, t)
            if camera_problem:
                inputs[GROUP_OWN] = InputStatus(GROUP_OWN, c.span(GROUP_OWN), False, "none", age_ms, camera_problem)
            elif not own.ok:
                inputs[GROUP_OWN] = InputStatus(GROUP_OWN, c.span(GROUP_OWN), False, "none", age_ms, own.reason)
            else:
                inputs[GROUP_OWN] = InputStatus(
                    GROUP_OWN, c.span(GROUP_OWN), True, own_source,
                    age_ms, OK, {"x_m": own.x_m, "y_m": own.y_m, "heading_deg": own.heading_deg})
        if c.indices(GROUP_OPPONENT):
            if camera_problem:
                inputs[GROUP_OPPONENT] = InputStatus(GROUP_OPPONENT, c.span(GROUP_OPPONENT), False, "none", age_ms, camera_problem)
            elif not sample.opponent.ok:
                inputs[GROUP_OPPONENT] = InputStatus(GROUP_OPPONENT, c.span(GROUP_OPPONENT), False, "none", age_ms, sample.opponent.reason)
            else:
                opponent = sample.opponent
                inputs[GROUP_OPPONENT] = InputStatus(
                    GROUP_OPPONENT, c.span(GROUP_OPPONENT), True,
                    opponent.source or f"unmarked blob T{opponent.track_id}, floor-plane centre (approximate, not height-corrected)", age_ms, OK,
                    {"x_m": opponent.xy_m[0], "y_m": opponent.xy_m[1]})

        blocked = [f"[{status.indices}] {group}: {status.reason}" for group, status in inputs.items() if not status.valid]
        observation, proposed, mode = None, None, MODE_BLOCKED
        if not blocked:
            own_values = inputs[GROUP_OWN].values if GROUP_OWN in inputs else {"x_m": 0.0, "y_m": 0.0, "heading_deg": 0.0}
            opponent_values = inputs[GROUP_OPPONENT].values if GROUP_OPPONENT in inputs else {"x_m": 0.0, "y_m": 0.0}
            observation = c.assemble(gyro_rad_s=imu.gyro_rad_s, accel_m_s2=imu.accel_m_s2,
                                     own_xy_m=(own_values["x_m"], own_values["y_m"]),
                                     own_heading_rad=math.radians(own_values["heading_deg"]),
                                     opponent_xy_m=(opponent_values["x_m"], opponent_values["y_m"]))
            mode = MODE_COMPLETE
            if self.policy is not None:
                try:
                    proposed, mode = self.policy.propose(observation), self.shadow_label
                    self.last_error = ""
                except Exception as error:             # inference failed: BLOCKED, no tactic, nothing from an earlier decision
                    self.last_error = f"inference error: {type(error).__name__}: {error}"
                    proposed, mode = None, MODE_BLOCKED
                    blocked.append(self.last_error)
        imu_origin = self.imu_origin or getattr(self.telemetry, "origin", None)
        source = combine_origins(self.camera_origin, imu_origin)
        valid, _why = control_validity(source, imu_origin, inputs[GROUP_IMU].valid, not blocked and proposed is not None)
        decision = Decision(self.ticks, t, mode, self.input_source, c.identifier,
                            None if sample is None else sample.frame_index, None if sample is None else sample.t,
                            inputs, observation, blocked, proposed, source, valid,
                            {group: (None if status.age_ms is None else status.age_ms / 1000.0) for group, status in inputs.items()},
                            self.last_error)
        self.ticks += 1
        return decision


# ----------------------------------------------------------------------------------------------
# The checkpoint itself. Loaded only when asked; proposes, never drives.


class CheckpointPolicy:
    """The checkpoint as the shadow path uses it. The model is NOT loaded here: ``control_center.backend.policy_engine`` owns
    the single per-process copy (cached by path + mtime + sha256), the deterministic prediction and the inference lock."""

    def __init__(self, contract: ObservationContract, device: str = "cpu") -> None:
        self.contract, self.device = contract, device
        self.engine = None
        self.load_seconds: float | None = None

    @property
    def loaded(self) -> bool:
        return self.engine is not None and self.engine.loaded

    @property
    def model(self):                                   # the shared model object (read-only use)
        return None if self.engine is None else self.engine._model

    def load(self) -> None:
        if self.loaded:
            return
        from control_center.backend import policy_engine

        engine = policy_engine.engine_for(self.contract.checkpoint, self.device)
        info = engine.load()
        if not engine.loaded:
            raise ValueError("checkpoint not loaded: " + "; ".join(info.problems))
        if engine.encoding != self.contract.previous_action_encoding:
            raise ValueError(f"checkpoint encoding {engine.encoding} differs from the contract's {self.contract.previous_action_encoding}")
        if info.obs_dim != self.contract.observation_size or len(info.tactics) != self.contract.action_count:
            raise ValueError("the loaded checkpoint's spaces differ from its recorded contract")
        self.engine, self.load_seconds = engine, info.load_seconds

    def actions(self, observations: np.ndarray) -> np.ndarray:
        """Greedy tactic index for each row (the same choice as predict(deterministic=True))."""
        self.load()
        return self.engine.actions(observations)

    def propose(self, observation: np.ndarray) -> dict[str, Any]:
        self.load()
        from autobot.sim.env import tactic_to_drive

        output = self.engine.predict(observation)
        if not output.ok:
            raise RuntimeError(output.error)
        return {"action": output.action, "tactic": output.tactic, "probability": output.probability,
                "probabilities": output.probabilities,
                "simulator_wheel_command": [float(v) for v in tactic_to_drive(output.action, self.contract.action_profile)],
                "note": "what the checkpoint would choose; simulator wheel units (-1..1, negative = forward); not sent anywhere"}
