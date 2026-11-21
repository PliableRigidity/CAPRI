"""Multi-marker, temporal chassis-centre tracker for the own (tagged) robot. Read-only vision.

Built on the existing per-marker maths of ``overlay_core`` (``MarkerSpec.model_corners``,
``FloorMapping.pixel_to_floor``, ``lift_to_height``, ``rigid_fit_2d``). Nothing here opens a camera,
a serial port or a socket: ``ChassisTracker.update`` is handed marker detections and a timestamp.

What it adds to ``estimate_bot_pose`` (which is all-or-nothing on one frame):

  CONSENSUS    every visible configured marker gives its own chassis pose. The largest set of
               mutually consistent markers is fused with one rigid fit; the others are listed as
               rejected, each with a reason. A tie (for example two markers that disagree) is
               broken only by agreement with the track's short-term prediction; without a usable
               track a tie is "inconsistent" and gives no pose. A size-ratio failure or a repeated
               id rejects that marker, not the frame. A marker that a consensus of two or more
               others has just rejected is not trusted on its own for ``suspect_hold_s``.
  FACES        top and bottom markers together: the face whose consensus is a strict majority is
               used only if every marker of the other face fails the consistency test against it;
               otherwise the frame gives no measurement.
  MOTION GATE  a measurement that would need more than ``max_speed_m_s`` / ``max_turn_rate_rad_s``
               (plus a noise floor) from the last accepted one is rejected. If such measurements
               keep agreeing with each other for ``reacquire_frames`` measurement-bearing frames in a
               row (gaps no longer than ``predict_hold_s``), the track starts again at the new place,
               so a wrong lock cannot last.
  STATES       MEASURED   a consistent detection in the newest frame
               PREDICTED  no measurement: constant-velocity extrapolation of recent measurements,
                          velocity clamped, only up to ``predict_hold_s``
               STALE      older than the hold, younger than ``lost_after_s``: the last measured
                          pose is kept FOR DISPLAY ONLY and is not valid
               LOST       no pose at all
               Validity is a pure function of the query time: if frames stop arriving,
               ``status(now)`` still goes MEASURED -> PREDICTED -> STALE -> LOST.

REAR MARKER (``face: rear`` in ``bot.markers``): a marker on the vertical rear face. Its four
corners are at two heights, so each corner is lifted to its own height before the rigid fit. That
needs the camera position; without one the marker is rejected. Upright and inverted are both
tried and the clearly better fit is kept, unless top/bottom markers decide the orientation in the
same frame. THIS PART IS FIXTURE-TESTED ONLY: it has never seen a real camera, and it depends
entirely on the camera position, which is usually DERIVED from the floor mapping under a pinhole
assumption. A rear-only pose is labelled ``source: "rear marker only"``.

Conventions for a rear marker in the config (all values to be measured on the robot):
  forward_m   where the rear face is, toward the nose from the chassis centre (negative)
  left_m      marker centre, to the robot's left, with the robot upright
  height_m    marker CENTRE above the floor, robot upright
  height_inverted_m  the same with the robot upside down (default: height_m)
  yaw_deg     in-plane rotation, counter-clockwise as seen from behind the upright robot;
              0 = the marker's top edge is up
An inverted chassis is taken to be rolled over about its nose axis, so the rear marker is still at
the rear, mirrored left-right and rotated by 180 degrees.

Optional config section (every key optional)::

    tracker:
      predict_hold_s: 0.20
      lost_after_s: 1.0
      max_speed_m_s: 2.5
      max_turn_rate_rad_s: 25.0
      reacquire_frames: 3
      start_confirm_frames: 1   # >1: a track that would start from ONE marker in ONE frame waits for that many agreeing frames

The consistency thresholds are the existing ``pose:`` keys (``max_position_disagreement_m``,
``max_heading_disagreement_deg``, ``max_marker_size_error_fraction``).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field, fields
from enum import Enum
import itertools
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import yaml

from control_center.tools.overlay_core import (POSITION_LABELS, UNKNOWN, BotPose, CameraEstimate,
                                               FloorMapping, MarkerSpec, OverlayConfig,
                                               angle_difference, lift_to_height, rigid_fit_2d)


class TrackState(str, Enum):
    MEASURED = "measured"
    PREDICTED = "predicted"
    STALE = "stale"
    LOST = "lost"


MEASURED, PREDICTED, STALE, LOST = (TrackState.MEASURED, TrackState.PREDICTED, TrackState.STALE,
                                    TrackState.LOST)
#: Added to a size rejection while ``bot.measured`` is false: on the real camera EVERY decoded marker was rejected this way (the traces in
#: artifacts/live-track say 9-16 mm against the 40 mm placeholder), so the operator has to be told what to do about it.
SIZE_PLACEHOLDER_HINT = ("; the configured size is a PLACEHOLDER (bot.measured is false): measure the printed black square "
                         "and set bot.marker_size_m (or size_m per marker) in the overlay config")
REAR_ONLY = "rear marker only"
REAR_LABEL = ("rear marker only: LOWER CONFIDENCE (fixture-tested only; depends on the camera "
              "position, which is derived unless camera.position_m is measured)")


# --------------------------------------------------------------------------------------
# Settings
# --------------------------------------------------------------------------------------

@dataclass(frozen=True)
class TrackerSettings:
    predict_hold_s: float = 0.20
    lost_after_s: float = 1.0
    max_speed_m_s: float = 2.5
    max_turn_rate_rad_s: float = 25.0
    reacquire_frames: int = 3
    start_confirm_frames: int = 1           # a track that STARTS from one marker needs this many agreeing frames (1 = no check;
                                            # the live viewers use LIVE_START_CONFIRM_FRAMES via ``live_tracker_settings``)
    position_noise_m: float = 0.03          # motion-gate noise floor
    heading_noise_deg: float = 15.0
    measured_fresh_s: float = 0.05          # how long the newest measurement reads MEASURED
    velocity_window_s: float = 0.15         # measurements this recent give the velocity
    suspect_hold_s: float = 3.0             # a consensus-rejected marker is not trusted alone
    rear_max_residual_m: float = 0.006      # a rear marker must fit its model this well
    rear_residual_ratio: float = 0.5        # ... and this much better than the other orientation

    def __post_init__(self) -> None:
        for name in ("predict_hold_s", "lost_after_s", "max_speed_m_s", "max_turn_rate_rad_s",
                     "position_noise_m", "heading_noise_deg", "measured_fresh_s",
                     "velocity_window_s", "rear_max_residual_m", "rear_residual_ratio"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"tracker.{name} must be a positive number, not {value!r}")
        if self.lost_after_s <= self.predict_hold_s:
            raise ValueError("tracker.lost_after_s must be longer than tracker.predict_hold_s")
        if self.measured_fresh_s > self.predict_hold_s:
            raise ValueError("tracker.measured_fresh_s must not exceed tracker.predict_hold_s")
        if int(self.reacquire_frames) < 1:
            raise ValueError("tracker.reacquire_frames must be at least 1")
        if int(self.start_confirm_frames) < 1:
            raise ValueError("tracker.start_confirm_frames must be at least 1")

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "TrackerSettings":
        data = dict(data or {})
        known = {f.name: f.type for f in fields(cls)}
        unknown = sorted(set(data) - set(known))
        if unknown:
            raise ValueError(f"unknown tracker setting(s) {unknown}; known: {sorted(known)}")
        values = {key: (int(value) if key in ("reacquire_frames", "start_confirm_frames") else float(value))
                  for key, value in data.items()}
        return cls(**values)

    def to_report(self) -> dict[str, Any]:
        return {f.name: getattr(self, f.name) for f in fields(self)}


def load_tracker_settings(config: OverlayConfig | None = None,
                          path: Path | str | None = None) -> TrackerSettings:
    """The optional ``tracker:`` section of the overlay config file; defaults when absent."""
    source = Path(path) if path else (config.path if config is not None else None)
    if source is None or not Path(source).is_file():
        return TrackerSettings()
    data = yaml.safe_load(Path(source).read_text(encoding="utf-8")) or {}
    return TrackerSettings.from_dict(data.get("tracker"))


#: What the live viewers (LiveTrack, hence the TRACKING screen) use unless the config's ``tracker:`` section says otherwise: a track
#: that would start from one marker in one frame waits for a second agreeing frame. One decoded marker is the weakest evidence there
#: is and a false decode of an id that is in the layout looks exactly like it. The bare ``TrackerSettings()`` default stays 1.
LIVE_START_CONFIRM_FRAMES = 2


def live_tracker_settings(config: OverlayConfig | None = None, path: Path | str | None = None) -> TrackerSettings:
    """``load_tracker_settings`` with the live default for ``start_confirm_frames``; a value in the file wins."""
    source = Path(path) if path else (config.path if config is not None else None)
    data: dict[str, Any] = {}
    if source is not None and Path(source).is_file():
        data = dict((yaml.safe_load(Path(source).read_text(encoding="utf-8")) or {}).get("tracker") or {})
    data.setdefault("start_confirm_frames", LIVE_START_CONFIRM_FRAMES)
    return TrackerSettings.from_dict(data)


# --------------------------------------------------------------------------------------
# One marker -> one chassis pose candidate
# --------------------------------------------------------------------------------------

@dataclass
class MarkerCandidate:
    marker_id: int
    face: str
    orientation: str                 # "upright" or "inverted"
    x: float
    y: float
    theta: float
    model: np.ndarray                # (4, 2) bot frame
    plane: np.ndarray                # (4, 2) arena metres at the marker's own height(s)
    floor: np.ndarray                # (4, 2) raw floor projection
    rms_m: float
    size_ratio: float | None = None

    def detail(self) -> dict[str, Any]:
        return {"id": self.marker_id, "face": self.face, "bot_x_m": self.x, "bot_y_m": self.y,
                "heading_deg": math.degrees(self.theta), "corner_fit_rms_mm": 1000 * self.rms_m,
                "size_ratio": self.size_ratio}


def rear_model(spec: MarkerSpec, orientation: str) -> tuple[np.ndarray, np.ndarray]:
    """Rear-marker corners TL, TR, BR, BL as (forward, left) in the bot frame, and their heights.

    The bot frame of an inverted chassis is the one ``overlay_core`` already uses for bottom
    markers: forward is the nose, left is the left of someone looking down on the upturned robot.
    """
    inverted = orientation == "inverted"
    half = spec.size_m / 2.0
    psi = math.radians(spec.yaw_deg) + (math.pi if inverted else 0.0)
    c, s = math.cos(psi), math.sin(psi)
    centre_left = -spec.left_m if inverted else spec.left_m
    height = spec.height_m
    if inverted and spec.height_inverted_m is not None:
        height = spec.height_inverted_m
    xy, z = [], []
    # (right, up) as seen by someone standing behind the robot and looking toward its nose
    for right, up in ((-half, half), (half, half), (half, -half), (-half, -half)):
        r, u = right * c - up * s, right * s + up * c
        xy.append((spec.forward_m, centre_left - r))
        z.append(height + u)
    return np.asarray(xy, float), np.asarray(z, float)


def drop_to_floor(xy: Any, height_m: float, camera_xyz: Iterable[float]) -> np.ndarray:
    """Inverse of ``lift_to_height``: where a point at ``height_m`` appears on the floor plane."""
    cam = np.asarray(tuple(camera_xyz), float)
    k = height_m / cam[2]
    return (np.asarray(xy, float).reshape(-1, 2) - k * cam[:2]) / (1.0 - k)


# --------------------------------------------------------------------------------------
# One frame's measurement, and the tracker's answer at a query time
# --------------------------------------------------------------------------------------

@dataclass
class FrameMeasurement:
    ok: bool = False
    reason: str = "no configured bot markers visible"
    x: float | None = None
    y: float | None = None
    theta: float | None = None
    orientation: str = UNKNOWN
    used_ids: list[int] = field(default_factory=list)
    rejected: dict[int, str] = field(default_factory=dict)
    visible_ids: list[int] = field(default_factory=list)
    ignored_ids: list[int] = field(default_factory=list)
    position_quality: str = UNKNOWN
    correction_source: str = "none"
    position_label: str = UNKNOWN
    source: str = "none"
    floor_projection_xy: tuple[float, float] | None = None
    fit_rms_mm: float | None = None
    markers: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class ChassisEstimate:
    state: TrackState = LOST
    x_m: float | None = None                 # chassis centre; for STALE: display only
    y_m: float | None = None
    heading_rad: float | None = None
    orientation: str = UNKNOWN               # held from the last measurement while predicting
    age_s: float | None = None               # age of the last accepted measurement
    used_ids: list[int] = field(default_factory=list)   # ids of the measurement the pose rests on
    rejected: dict[int, str] = field(default_factory=dict)       # newest frame: id -> reason
    visible_ids: list[int] = field(default_factory=list)         # newest frame
    ignored_ids: list[int] = field(default_factory=list)
    position_quality: str = UNKNOWN
    correction_source: str = "none"
    position_label: str = UNKNOWN
    source: str = "none"                     # "top markers", "bottom markers", ..., REAR_ONLY
    reason: str = "no consistent detection yet"
    frame_reason: str = ""                   # why the newest frame gave no measurement ("" if it did)
    floor_projection_xy: tuple[float, float] | None = None
    velocity_m_s: tuple[float, float] = (0.0, 0.0)
    turn_rate_rad_s: float = 0.0
    query_t: float | None = None

    @property
    def valid_for_observation(self) -> bool:
        return self.state in (MEASURED, PREDICTED)

    @property
    def heading_deg(self) -> float | None:
        return None if self.heading_rad is None else math.degrees(self.heading_rad)

    @property
    def flipped(self) -> bool | None:
        """True when the bottom markers are the ones in view (the chassis is upside down), False upright, None when never seen.

        Which face is up comes ONLY from which markers were decoded (ids on the ``bottom`` face vs the ``top`` face of the config).
        It is held from the last accepted detection while PREDICTED or STALE and is None when LOST."""
        return {"inverted": True, "upright": False}.get(self.orientation)

    def to_report(self) -> dict[str, Any]:
        def r(value: float | None, digits: int = 4) -> float | None:
            return None if value is None else round(float(value), digits)

        return {"state": self.state.value, "valid_for_observation": self.valid_for_observation,
                "x_m": r(self.x_m), "y_m": r(self.y_m), "heading_deg": r(self.heading_deg, 2),
                "pose_note": ("last measured pose, DISPLAY ONLY, not valid"
                              if self.state == STALE else None),
                "orientation": self.orientation, "flipped": self.flipped, "age_s": r(self.age_s),
                "used_ids": self.used_ids, "visible_ids": self.visible_ids,
                "rejected": {str(key): value for key, value in self.rejected.items()},
                "ignored_ids": self.ignored_ids, "position_quality": self.position_quality,
                "correction_source": self.correction_source, "source": self.source,
                "reason": self.reason, "frame_reason": self.frame_reason,
                "velocity_m_s": [r(self.velocity_m_s[0]), r(self.velocity_m_s[1])],
                "turn_rate_rad_s": r(self.turn_rate_rad_s)}

    def to_bot_pose(self) -> BotPose:
        """For existing consumers: ``ok`` only when the pose is valid for the observation."""
        if not self.valid_for_observation:
            return BotPose(reason=self.reason, orientation=self.orientation,
                           visible_ids=list(self.visible_ids), ignored_ids=list(self.ignored_ids))
        return BotPose(state="ok", reason="ok" if self.state == MEASURED else self.reason,
                       x_m=self.x_m, y_m=self.y_m, heading_rad=self.heading_rad,
                       orientation=self.orientation, position_quality=self.position_quality,
                       correction_source=self.correction_source,
                       position_label=self.position_label,
                       floor_projection_xy=self.floor_projection_xy,
                       visible_ids=list(self.visible_ids), used_ids=list(self.used_ids),
                       ignored_ids=list(self.ignored_ids))


# --------------------------------------------------------------------------------------
# The tracker
# --------------------------------------------------------------------------------------

_EPS = 1e-9                                  # so that an age of exactly the limit is still inside it


def _wrap(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


class ChassisTracker:
    def __init__(self, config: OverlayConfig, mapping: FloorMapping, camera: CameraEstimate,
                 settings: TrackerSettings | None = None) -> None:
        self.config, self.mapping, self.camera = config, mapping, camera
        self.settings = settings or load_tracker_settings(config)
        self._last: FrameMeasurement | None = None       # last ACCEPTED measurement
        self._last_t: float | None = None
        self._history: deque[tuple[float, float, float, float]] = deque()   # t, x, y, unwrapped heading
        self._velocity = (0.0, 0.0, 0.0)
        self._frame = FrameMeasurement(reason="no frame yet")               # newest frame's result
        self._frame_t: float | None = None
        self._pending: list[tuple[float, FrameMeasurement]] = []
        self._candidates: list[tuple[float, FrameMeasurement]] = []      # single-marker frames waiting to start a track
        self._suspect: dict[int, float] = {}
        self.reinitialisations = 0
        self.gate_rejections = 0

    # ---- per-marker candidates -----------------------------------------------------------
    def _face_candidate(self, spec: MarkerSpec, corners: np.ndarray) -> MarkerCandidate:
        floor = self.mapping.pixel_to_floor(corners)
        corrected = self.camera.position_m is not None
        plane = lift_to_height(floor, spec.height_m, self.camera.position_m) if corrected else floor
        model = spec.model_corners()
        theta, centre, rms = rigid_fit_2d(model, plane)
        side = float(np.mean(np.linalg.norm(plane - np.roll(plane, -1, axis=0), axis=1)))
        return MarkerCandidate(spec.marker_id, spec.face,
                               "upright" if spec.face == "top" else "inverted",
                               float(centre[0]), float(centre[1]), theta, model, plane, floor, rms,
                               side / spec.size_m)

    def _rear_candidate(self, spec: MarkerSpec, corners: np.ndarray, orientation: str) -> MarkerCandidate:
        floor = self.mapping.pixel_to_floor(corners)
        model, heights = rear_model(spec, orientation)
        plane = np.vstack([lift_to_height(floor[i:i + 1], float(heights[i]), self.camera.position_m)
                           for i in range(4)])
        theta, centre, rms = rigid_fit_2d(model, plane)
        return MarkerCandidate(spec.marker_id, "rear", orientation, float(centre[0]),
                               float(centre[1]), theta, model, plane, floor, rms)

    def _consistent(self, a: MarkerCandidate, b: MarkerCandidate) -> bool:
        return (math.hypot(a.x - b.x, a.y - b.y) <= self.config.max_position_disagreement_m and
                abs(math.degrees(angle_difference(a.theta, b.theta)))
                <= self.config.max_heading_disagreement_deg)

    def _largest_sets(self, candidates: list[MarkerCandidate]) -> list[tuple[int, ...]]:
        """Every largest set of mutually consistent candidates (indices). More than one = a tie."""
        count = len(candidates)
        ok = [[i == j or self._consistent(candidates[i], candidates[j]) for j in range(count)]
              for i in range(count)]
        for size in range(count, 0, -1):
            found = [combo for combo in itertools.combinations(range(count), size)
                     if all(ok[i][j] for i, j in itertools.combinations(combo, 2))]
            if found:
                return found
        return []

    @staticmethod
    def _fuse(members: list[MarkerCandidate]) -> tuple[float, float, float, float]:
        theta, centre, rms = rigid_fit_2d(np.vstack([m.model for m in members]),
                                          np.vstack([m.plane for m in members]))
        return float(centre[0]), float(centre[1]), theta, rms

    def _away_from(self, candidate: MarkerCandidate, x: float, y: float, theta: float) -> str:
        """"" when the candidate agrees with the pose, else how far off it is."""
        distance = math.hypot(candidate.x - x, candidate.y - y)
        turn = abs(math.degrees(angle_difference(candidate.theta, theta)))
        if (distance <= self.config.max_position_disagreement_m
                and turn <= self.config.max_heading_disagreement_deg):
            return ""
        return f"{1000 * distance:.0f} mm / {turn:.1f} deg"

    # ---- the track's short-term prediction -------------------------------------------------
    def _usable_track(self, t: float) -> bool:
        return self._last_t is not None and 0.0 <= t - self._last_t <= self.settings.predict_hold_s + _EPS

    def _predict(self, t: float) -> tuple[float, float, float]:
        age = max(0.0, t - self._last_t)
        vx, vy, omega = self._velocity
        return self._last.x + vx * age, self._last.y + vy * age, _wrap(self._last.theta + omega * age)

    def _agrees_with_prediction(self, x: float, y: float, theta: float, t: float) -> bool:
        s = self.settings
        age = max(0.0, t - self._last_t)
        px, py, ptheta = self._predict(t)
        return (math.hypot(x - px, y - py)
                <= self.config.max_position_disagreement_m + 0.25 * s.max_speed_m_s * age and
                abs(angle_difference(theta, ptheta))
                <= math.radians(self.config.max_heading_disagreement_deg)
                + 0.25 * s.max_turn_rate_rad_s * age)

    # ---- one frame -> at most one measurement ---------------------------------------------
    def measure(self, detections: list[Any], t: float) -> FrameMeasurement:
        """Consensus over this frame's markers. The track is read only to break a tie; the only state
        this changes is the note of which markers a consensus has just outvoted."""
        config, s = self.config, self.settings
        known = [d for d in detections if d.tag_id in config.markers]
        frame = FrameMeasurement(
            visible_ids=sorted(d.tag_id for d in known),
            ignored_ids=sorted({d.tag_id for d in detections} - set(config.markers)))
        if not known:
            return frame
        if not self.mapping.valid:
            frame.reason = f"no valid floor mapping ({self.mapping.reason})"
            return frame
        counts: dict[int, int] = {}
        for detection in known:
            counts[detection.tag_id] = counts.get(detection.tag_id, 0) + 1
        rejected = frame.rejected
        faces: list[MarkerCandidate] = []
        rear: list[tuple[MarkerSpec, np.ndarray]] = []
        for detection in sorted(known, key=lambda d: d.tag_id):
            spec = config.markers[detection.tag_id]
            if counts[spec.marker_id] > 1:
                rejected[spec.marker_id] = "detected more than once in this frame"
                continue
            corners = np.asarray(detection.corners, float).reshape(4, 2)
            if spec.face == "rear":
                if self.camera.position_m is None:
                    rejected[spec.marker_id] = ("rear marker needs the camera position, which is "
                                                f"not available ({self.camera.detail})")
                else:
                    rear.append((spec, corners))
                continue
            candidate = self._face_candidate(spec, corners)
            frame.markers.append(candidate.detail())
            tolerance = config.max_marker_size_error_fraction
            if not np.isfinite([candidate.x, candidate.y, candidate.theta]).all():
                rejected[spec.marker_id] = "maps to no finite floor position"
            elif tolerance is not None and abs(candidate.size_ratio - 1.0) > tolerance:
                rejected[spec.marker_id] = (
                    f"measures {1000 * candidate.size_ratio * spec.size_m:.0f} mm but is configured "
                    f"as {1000 * spec.size_m:.0f} mm (size, height or dictionary wrong)"
                    + (SIZE_PLACEHOLDER_HINT if "bot markers" in self.config.placeholder_tags else ""))
            else:
                faces.append(candidate)

        chosen: list[MarkerCandidate] = []
        if faces:
            chosen = self._face_consensus(faces, frame, t)
            if not chosen:
                for spec, _ in rear:
                    rejected[spec.marker_id] = "not used: the top/bottom markers gave no consensus"
                return frame
            x, y, theta, _ = self._fuse(chosen)
            for spec, corners in rear:                   # the faces decide which way up it is
                candidate = self._rear_candidate(spec, corners, chosen[0].orientation)
                frame.markers.append(candidate.detail())
                off = self._away_from(candidate, x, y, theta)
                if candidate.rms_m > s.rear_max_residual_m:
                    rejected[spec.marker_id] = (f"does not fit as a rear marker of a(n) "
                                                f"{chosen[0].orientation} chassis (residual "
                                                f"{1000 * candidate.rms_m:.1f} mm)")
                elif off:
                    rejected[spec.marker_id] = (f"disagrees with the {chosen[0].face} markers "
                                                f"{sorted(m.marker_id for m in chosen)} by {off}")
                else:
                    chosen.append(candidate)
        elif rear:
            chosen = self._rear_consensus(rear, frame, t)
            if not chosen:
                return frame
        else:
            frame.reason = "every visible marker was rejected: " + "; ".join(
                f"{key}: {value}" for key, value in sorted(rejected.items()))
            return frame

        return self._finish(frame, chosen, t)

    def _break_tie(self, candidates: list[MarkerCandidate], sets: list[tuple[int, ...]],
                   t: float) -> tuple[int, ...] | None:
        """The one set that agrees with the track's prediction, or None.

        Several sets can lie inside the agreement tolerance when the last detection is a few frames old;
        then the nearest one wins only if every other is clearly farther from the prediction (by at least
        the position-disagreement limit). Anything less clear stays a tie.
        """
        if not self._usable_track(t):
            return None
        px, py, _ = self._predict(t)
        agreeing = []
        for combo in sets:
            x, y, theta, _ = self._fuse([candidates[i] for i in combo])
            if self._agrees_with_prediction(x, y, theta, t):
                agreeing.append((math.hypot(x - px, y - py), combo))
        agreeing.sort(key=lambda item: item[0])
        if len(agreeing) == 1:
            return agreeing[0][1]
        if len(agreeing) > 1 and agreeing[1][0] - agreeing[0][0] >= self.config.max_position_disagreement_m:
            return agreeing[0][1]
        return None

    def _face_consensus(self, faces: list[MarkerCandidate], frame: FrameMeasurement,
                        t: float) -> list[MarkerCandidate]:
        sets = self._largest_sets(faces)
        described = " vs ".join(str(sorted(faces[i].marker_id for i in combo)) for combo in sets)
        if len(sets) > 1:
            pick = self._break_tie(faces, sets, t)
            if pick is None:
                why = ("no usable track to decide" if not self._usable_track(t)
                       else "the track's prediction does not single one out")
                frame.reason = f"inconsistent: markers disagree ({described}); {why}"
                for candidate in faces:
                    frame.rejected[candidate.marker_id] = f"tie between {described}: {why}"
                return []
            tie_note = f"tie {described} broken by the track's prediction"
        else:
            pick, tie_note = sets[0], ""
        chosen = [faces[i] for i in pick]
        outside = [faces[i] for i in range(len(faces)) if i not in pick]
        used = sorted(m.marker_id for m in chosen)
        if len({m.face for m in chosen}) > 1:
            frame.reason = (f"conflict: top and bottom markers {used} agree with each other, so "
                            "which face is up cannot be told")
            for candidate in faces:
                frame.rejected[candidate.marker_id] = "top and bottom markers agree with each other"
            return []
        x, y, theta, _ = self._fuse(chosen)
        other = [m for m in outside if m.face != chosen[0].face]
        if other:
            other_ids = sorted(m.marker_id for m in other)
            agreeing = sorted(m.marker_id for m in other if not self._away_from(m, x, y, theta))
            if len(chosen) <= len(other) or agreeing:
                why = (f"{chosen[0].face} markers {used} are not a strict majority over "
                       f"{other[0].face} markers {other_ids}" if len(chosen) <= len(other) else
                       f"{other[0].face} marker(s) {agreeing} also agree with the "
                       f"{chosen[0].face} consensus {used}")
                frame.reason = f"conflict: top and bottom markers visible together; {why}"
                for candidate in faces:
                    frame.rejected[candidate.marker_id] = "top and bottom markers visible together"
                return []
        now_suspect = [m.marker_id for m in chosen
                       if t - self._suspect.get(m.marker_id, -math.inf) <= self.settings.suspect_hold_s]
        if len(chosen) == 1 and now_suspect:
            marker = chosen[0].marker_id
            frame.reason = (f"marker {marker} is the only usable marker and a consensus rejected it "
                            f"{1000 * (t - self._suspect[marker]):.0f} ms ago: not trusted alone")
            frame.rejected[marker] = "recently rejected by a consensus: not trusted alone"
            for candidate in outside:
                frame.rejected[candidate.marker_id] = "disagrees with the other visible marker"
            return []
        for candidate in outside:
            off = self._away_from(candidate, x, y, theta) or "an incompatible pose"
            frame.rejected[candidate.marker_id] = (
                f"disagrees with the consensus of {used} by {off}" + (f" ({tie_note})" if tie_note else ""))
            if len(chosen) >= 2:
                self._suspect[candidate.marker_id] = t
        if len(chosen) >= 2:
            for candidate in chosen:
                self._suspect.pop(candidate.marker_id, None)
        return chosen

    def _rear_consensus(self, rear: list[tuple[MarkerSpec, np.ndarray]], frame: FrameMeasurement,
                        t: float) -> list[MarkerCandidate]:
        s = self.settings
        usable: list[MarkerCandidate] = []
        for spec, corners in rear:
            options = sorted((self._rear_candidate(spec, corners, orientation)
                              for orientation in ("upright", "inverted")), key=lambda c: c.rms_m)
            best, other = options
            frame.markers.append({**best.detail(), "other_orientation_rms_mm": 1000 * other.rms_m})
            if not np.isfinite([best.x, best.y, best.theta, best.rms_m]).all():
                frame.rejected[spec.marker_id] = "maps to no finite floor position"
            elif best.rms_m > s.rear_max_residual_m:
                frame.rejected[spec.marker_id] = (
                    f"does not fit as a rear marker either way up (best residual "
                    f"{1000 * best.rms_m:.1f} mm; size, heights or camera position wrong)")
            elif best.rms_m > s.rear_residual_ratio * other.rms_m:
                frame.rejected[spec.marker_id] = (
                    f"ambiguous: upright and inverted fit about equally well "
                    f"({1000 * best.rms_m:.1f} / {1000 * other.rms_m:.1f} mm)")
            else:
                usable.append(best)
        if not usable:
            frame.reason = "rear marker not usable: " + "; ".join(
                f"{key}: {value}" for key, value in sorted(frame.rejected.items()))
            return []
        if len(usable) == 1:
            return usable
        sets = self._largest_sets(usable)
        pick = sets[0] if len(sets) == 1 else self._break_tie(usable, sets, t)
        if pick is None or len({usable[i].orientation for i in pick}) > 1:
            frame.reason = "inconsistent: the rear markers disagree with each other"
            for candidate in usable:
                frame.rejected[candidate.marker_id] = "rear markers disagree with each other"
            return []
        for index, candidate in enumerate(usable):
            if index not in pick:
                frame.rejected[candidate.marker_id] = "disagrees with the other rear marker(s)"
        return [usable[i] for i in pick]

    def _finish(self, frame: FrameMeasurement, chosen: list[MarkerCandidate], t: float) -> FrameMeasurement:
        x, y, theta, rms = self._fuse(chosen)
        on_faces = [m for m in chosen if m.face != "rear"]
        frame.ok, frame.reason = True, "ok"
        frame.x, frame.y, frame.theta = x, y, theta
        frame.orientation = chosen[0].orientation
        frame.used_ids = sorted(m.marker_id for m in chosen)
        frame.fit_rms_mm = 1000.0 * rms
        corrected = self.camera.position_m is not None
        frame.correction_source = self.camera.source if corrected else "none"
        frame.position_quality = "height-corrected" if corrected else "approximate"
        frame.position_label = POSITION_LABELS[frame.correction_source]
        if on_faces:
            frame.source = f"{on_faces[0].face} markers" + (" + rear" if len(on_faces) < len(chosen) else "")
            _, floor_centre, _ = rigid_fit_2d(np.vstack([m.model for m in on_faces]),
                                              np.vstack([m.floor for m in on_faces]))
            frame.floor_projection_xy = (float(floor_centre[0]), float(floor_centre[1]))
        else:
            frame.source, frame.position_label = REAR_ONLY, REAR_LABEL
            spec = self.config.markers[chosen[0].marker_id]
            height = spec.height_m
            if chosen[0].orientation == "inverted" and spec.height_inverted_m is not None:
                height = spec.height_inverted_m
            seen = drop_to_floor([[x, y]], height, self.camera.position_m)[0]
            frame.floor_projection_xy = (float(seen[0]), float(seen[1]))
        return frame

    # ---- the track -----------------------------------------------------------------------
    def _gate(self, frame: FrameMeasurement, since_t: float, since: FrameMeasurement, t: float) -> str:
        """"" when moving from ``since`` to ``frame`` is physically possible, else why not."""
        s = self.settings
        dt = max(0.0, t - since_t)
        step = math.hypot(frame.x - since.x, frame.y - since.y)
        turn = abs(angle_difference(frame.theta, since.theta))
        if step > s.position_noise_m + s.max_speed_m_s * dt:
            return (f"a {1000 * step:.0f} mm step in {1000 * dt:.0f} ms would need "
                    f"{step / max(dt, 1e-6):.1f} m/s (limit {s.max_speed_m_s:g} m/s)")
        if turn > math.radians(s.heading_noise_deg) + s.max_turn_rate_rad_s * dt:
            return (f"a {math.degrees(turn):.0f} deg turn in {1000 * dt:.0f} ms would need "
                    f"{turn / max(dt, 1e-6):.0f} rad/s (limit {s.max_turn_rate_rad_s:g} rad/s)")
        return ""

    def _start(self, frame: FrameMeasurement, t: float,
               earlier: Iterable[tuple[float, FrameMeasurement]] = ()) -> None:
        self._history.clear()
        self._velocity = (0.0, 0.0, 0.0)
        self._candidates = []
        for earlier_t, earlier_frame in earlier:        # the agreeing frames that confirmed this start give the first velocity
            unwrapped = earlier_frame.theta
            if self._history:
                unwrapped = self._history[-1][3] + angle_difference(earlier_frame.theta, self._history[-1][3])
            self._history.append((earlier_t, earlier_frame.x, earlier_frame.y, unwrapped))
        self._accept(frame, t)

    def _confirm_start(self, frame: FrameMeasurement, t: float) -> None:
        """No track, and this frame's pose rests on ONE marker: believe it only after ``start_confirm_frames`` agreeing frames.

        One decoded marker is the weakest evidence there is (a false decode of an id that is in the layout looks exactly like it),
        so a track never starts from a single frame of it. The frames must follow each other within the prediction hold and each
        step must be a physically possible move. A frame that does not fit restarts the count. Two or more agreeing markers in one
        frame start a track at once (``update`` does not come here for them).
        """
        s = self.settings
        if self._candidates and (t - self._candidates[-1][0] > s.predict_hold_s + _EPS
                                 or self._gate(frame, self._candidates[-1][0], self._candidates[-1][1], t)):
            self._candidates = []
        self._candidates.append((t, frame))
        if len(self._candidates) >= s.start_confirm_frames:
            self._start(frame, t, earlier=self._candidates[:-1])
            return
        seen = list(frame.used_ids)
        frame.ok = False
        frame.reason = (f"a track cannot start from one marker in one frame (marker {seen}): "
                        f"{len(self._candidates)} of {s.start_confirm_frames} agreeing frames so far")
        for marker_id in seen:
            frame.rejected[marker_id] = "one marker only: waiting for a second agreeing frame before a track starts"
        frame.used_ids = []

    def _accept(self, frame: FrameMeasurement, t: float) -> None:
        s = self.settings
        unwrapped = frame.theta
        if self._history:
            unwrapped = self._history[-1][3] + angle_difference(frame.theta, self._history[-1][3])
        self._history.append((t, frame.x, frame.y, unwrapped))
        while len(self._history) > 2 and t - self._history[0][0] > s.velocity_window_s:
            self._history.popleft()
        first = self._history[0]
        span = t - first[0]
        if len(self._history) >= 2 and 1e-3 < span <= max(s.velocity_window_s, s.predict_hold_s):
            vx, vy = (frame.x - first[1]) / span, (frame.y - first[2]) / span
            omega = (unwrapped - first[3]) / span
            speed = math.hypot(vx, vy)
            if speed > s.max_speed_m_s:
                vx, vy = vx * s.max_speed_m_s / speed, vy * s.max_speed_m_s / speed
            omega = max(-s.max_turn_rate_rad_s, min(s.max_turn_rate_rad_s, omega))
            self._velocity = (vx, vy, omega)
        else:
            self._velocity = (0.0, 0.0, 0.0)             # one measurement: hold, do not extrapolate
        self._last, self._last_t = frame, t
        self._pending = []

    def update(self, detections: list[Any], t: float) -> ChassisEstimate:
        """Feed one frame's marker detections (in timestamp order) and get the state at ``t``."""
        if self._frame_t is not None and t < self._frame_t:
            raise ValueError("frames must be fed in timestamp order")
        s = self.settings
        frame = self.measure(detections, t)
        if frame.ok:
            have_track = self._last_t is not None and t - self._last_t <= s.lost_after_s + _EPS
            blocked = self._gate(frame, self._last_t, self._last, t) if have_track else ""
            if not have_track:
                if len(frame.used_ids) >= 2 or s.start_confirm_frames <= 1:
                    self._start(frame, t)
                else:
                    self._confirm_start(frame, t)
            elif not blocked:
                self._accept(frame, t)
            else:
                self.gate_rejections += 1
                if self._pending and (t - self._pending[-1][0] > s.predict_hold_s
                                      or self._gate(frame, self._pending[-1][0], self._pending[-1][1], t)):
                    self._pending = []                   # not the same new place: start counting again
                self._pending.append((t, frame))
                if len(self._pending) >= s.reacquire_frames:
                    self.reinitialisations += 1
                    self._start(frame, t)
                else:
                    frame.ok = False
                    frame.reason = (f"inconsistent with the track: {blocked}; seen "
                                    f"{len(self._pending)} of {s.reacquire_frames} frames at the new "
                                    "place before the track restarts there")
                    for marker_id in frame.used_ids:
                        frame.rejected[marker_id] = "consistent with each other but not with the track"
                    frame.used_ids = []
        self._frame, self._frame_t = frame, t
        return self.status(t)

    def status(self, now: float) -> ChassisEstimate:
        """The state at ``now``. A pure function of the query time and what has been fed."""
        s, frame = self.settings, self._frame
        estimate = ChassisEstimate(rejected=dict(frame.rejected), visible_ids=list(frame.visible_ids),
                                   ignored_ids=list(frame.ignored_ids), query_t=now,
                                   frame_reason="" if frame.ok else frame.reason)
        why = "" if frame.ok else f"; newest frame: {frame.reason}"
        if self._last is None or self._last_t is None:
            estimate.reason = "lost: no consistent detection yet" + why
            return estimate
        last = self._last
        age = max(0.0, now - self._last_t)
        estimate.age_s = age
        if age > s.lost_after_s + _EPS:
            estimate.reason = (f"lost: the last consistent detection is {1000 * age:.0f} ms old "
                               f"(limit {1000 * s.lost_after_s:.0f} ms)" + why)
            return estimate
        estimate.orientation, estimate.used_ids = last.orientation, list(last.used_ids)
        estimate.position_quality, estimate.correction_source = last.position_quality, last.correction_source
        estimate.position_label, estimate.source = last.position_label, last.source
        estimate.velocity_m_s, estimate.turn_rate_rad_s = self._velocity[:2], self._velocity[2]
        if age > s.predict_hold_s + _EPS:
            estimate.state = STALE
            estimate.x_m, estimate.y_m, estimate.heading_rad = last.x, last.y, last.theta
            estimate.floor_projection_xy = last.floor_projection_xy
            estimate.reason = (f"stale: the last consistent detection is {1000 * age:.0f} ms old "
                               f"(prediction is allowed for {1000 * s.predict_hold_s:.0f} ms); the pose "
                               "shown is the last measured one, for display only" + why)
            return estimate
        measured_now = frame.ok and self._frame_t == self._last_t and age <= s.measured_fresh_s + _EPS
        if measured_now:
            estimate.state, estimate.reason = MEASURED, "ok"
            estimate.x_m, estimate.y_m, estimate.heading_rad = last.x, last.y, last.theta
            estimate.floor_projection_xy = last.floor_projection_xy
            return estimate
        estimate.state = PREDICTED
        x, y, theta = self._predict(now)
        estimate.x_m, estimate.y_m, estimate.heading_rad = x, y, theta
        if last.floor_projection_xy is not None:
            estimate.floor_projection_xy = (last.floor_projection_xy[0] + x - last.x,
                                            last.floor_projection_xy[1] + y - last.y)
        estimate.reason = (f"predicted {1000 * age:.0f} ms from the last consistent detection "
                           f"(ids {last.used_ids})"
                           + (why if not frame.ok else "; no newer frame has been processed"))
        return estimate

    def reset(self) -> None:
        self._last, self._last_t = None, None
        self._history.clear()
        self._velocity = (0.0, 0.0, 0.0)
        self._pending, self._candidates, self._suspect = [], [], {}
        self._frame, self._frame_t = FrameMeasurement(reason="no frame yet"), None
