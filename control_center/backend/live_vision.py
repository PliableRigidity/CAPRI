"""Own-bot pose from the camera, using the overlay tool's floor mapping and marker code unchanged.

One ``LiveVision`` serves the Operate view, the recorder and (offline) the reviewer, so a pose
means the same thing everywhere. There is no hold or prediction in ``pose``: a frame without a valid,
consistent detection is a gap.

Beside it, ``VisionSample.own_track`` carries the multi-marker tracker's report for the same frame (state MEASURED / PREDICTED /
STALE / LOST, age, markers used, flipped or not, confidence inputs): the same ``chassis_tracker`` the TRACKING screen uses. It is
additive and display-only: nothing recorded or decided here reads it, and ``pose`` is exactly what it was.
"""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from control_center.tools.chassis_tracker import ChassisTracker, live_tracker_settings
from control_center.tools.overlay_core import (LIVE_PROFILE, BotPose, CameraEstimate, ClickSet, FitQuality,
                                               FloorMapping, MarkerDetector, OverlayConfig, PixelModel,
                                               assess_fit, estimate_bot_pose, fit_floor_mapping, load_clicks,
                                               profile_points_path, resolve_camera)

TAG_SWITCH_WINDOW_S = 1.0


@dataclass
class VisionSample:
    frame_index: int
    t_mono: float
    pose: BotPose
    detections: list = field(default_factory=list)
    mapping_valid: bool = False
    tag_switch: bool = False            # the set of markers used changed since the last valid pose
    jump_mm: float | None = None        # position change across that switch (marker-offset effect, not motion)
    duplicate_ids: list[int] = field(default_factory=list)
    unknown_ids: list[int] = field(default_factory=list)
    own_track: dict[str, Any] | None = None    # chassis_tracker report at t_mono (None: no valid floor map yet); see the module note


@dataclass
class MapState:
    level: str          # "none", "ok", "weak", "invalid"
    text: str           # short, for the status row
    detail: str = ""


def load_live_clicks(points_path: Path | None = None) -> tuple[ClickSet | None, str]:
    """The live camera's clicked points, or (None, why). A footage profile is never returned."""
    path = points_path or profile_points_path(LIVE_PROFILE)
    if not path.exists():
        return None, "no live calibration saved yet"
    try:
        clicks = load_clicks(path)
    except (OSError, ValueError, KeyError) as error:
        return None, f"live calibration file unreadable: {error}"
    if clicks.historical:
        return None, "the saved points are a HISTORICAL footage profile and are not used for the live camera"
    return clicks, "ok"


class LiveVision:
    def __init__(self, config: OverlayConfig, clicks: ClickSet | None = None, profile: str = LIVE_PROFILE,
                 not_calibrated_reason: str = "no live calibration saved yet") -> None:
        self.config = config
        self.detector = MarkerDetector(config.dictionary)
        self._lock = threading.Lock()
        self.profile = profile
        self.clicks = clicks
        self.not_calibrated_reason = not_calibrated_reason
        self.mapping = FloorMapping()
        self.camera = CameraEstimate()
        self.quality = FitQuality()
        self._fit_size: tuple[int, int] | None = None
        self._dirty = True
        self._last_valid: tuple[float, BotPose] | None = None
        self.chassis: ChassisTracker | None = None

    # ---- calibration ----

    def set_config(self, config: OverlayConfig) -> None:
        """Switch to another profile's overlay config (arena, reference points, marker layout)."""
        with self._lock:
            if config.dictionary != self.config.dictionary:
                self.detector = MarkerDetector(config.dictionary)
            self.config = config
            self._dirty = True
            self._last_valid = None

    def set_clicks(self, clicks: ClickSet | None, profile: str = LIVE_PROFILE, reason: str = "") -> None:
        with self._lock:
            self.clicks, self.profile = clicks, profile
            if clicks is None:
                self.not_calibrated_reason = reason or "no live calibration saved yet"
            self._dirty = True
            self._last_valid = None

    def _refit(self, size: tuple[int, int]) -> None:
        if self.clicks is None:
            self.mapping, self.camera, self.quality = FloorMapping(reason=self.not_calibrated_reason), CameraEstimate(), FitQuality()
        else:
            model = self.clicks.lens if self.clicks.lens.active else PixelModel(self.config.camera_matrix, self.config.distortion)
            self.mapping = fit_floor_mapping(self.config.references, self.clicks.points, size, model,
                                             roles=self.clicks.roles, clicks_image_size=self.clicks.image_size,
                                             max_fit_residual_mm=self.config.max_fit_residual_mm)
            self.camera = resolve_camera(self.config, self.mapping)
            self.quality = assess_fit(self.config.references, self.clicks.points, self.clicks.roles, self.mapping,
                                      self.config.arena, self.config.warn_check_error_mm,
                                      self.config.weak_predicted_error_mm, self.config.min_hull_fraction)
        self._fit_size, self._dirty = size, False
        self.chassis = (ChassisTracker(self.config, self.mapping, self.camera, live_tracker_settings(self.config))
                        if self.mapping.valid else None)          # a new map or layout starts a new track

    def map_state(self) -> MapState:
        with self._lock:
            if self.clicks is None:
                return MapState("none", "Map not calibrated", self.not_calibrated_reason)
            if self._fit_size is None:
                return MapState("none", "Map calibration loaded (waiting for a frame)", "")
            if not self.mapping.valid:
                return MapState("invalid", "Map calibration INVALID", self.mapping.reason)
            label = "" if self.profile == LIVE_PROFILE else f" [{self.profile}]"
            if self.clicks.synthetic:
                label = " [SYNTHETIC FIXTURE]"
            if self.quality.level == "ok":
                return MapState("ok", f"Map calibrated{label}", "")
            return MapState("weak", f"Map calibrated, fit WEAK{label}", "; ".join(self.quality.reasons))

    # ---- per frame ----

    def process(self, frame_index: int, t_mono: float, frame: np.ndarray) -> VisionSample:
        size = (frame.shape[1], frame.shape[0])
        detections = self.detector.detect_markers(frame)
        with self._lock:
            if self._dirty or size != self._fit_size:
                self._refit(size)
            pose = estimate_bot_pose(detections, self.mapping, self.camera, self.config)
            sample = VisionSample(frame_index, t_mono, pose, detections, self.mapping.valid)
            counts: dict[int, int] = {}
            for detection in detections:
                counts[detection.tag_id] = counts.get(detection.tag_id, 0) + 1
            sample.duplicate_ids = sorted(i for i, n in counts.items() if n > 1 and i in self.config.markers)
            sample.unknown_ids = sorted(i for i in counts if i not in self.config.markers)
            sample.own_track = self._own_track(detections, t_mono)
            if pose.ok:
                if self._last_valid is not None:
                    then, last = self._last_valid
                    if t_mono - then <= TAG_SWITCH_WINDOW_S and last.used_ids != pose.used_ids:
                        sample.tag_switch = True
                        sample.jump_mm = 1000.0 * math.hypot(pose.x_m - last.x_m, pose.y_m - last.y_m)
                self._last_valid = (t_mono, pose)
            return sample

    def _own_track(self, detections: list, t_mono: float) -> dict[str, Any] | None:
        """The chassis tracker's report for this frame. Never raises: a tracker problem must not cost the frame its pose."""
        if self.chassis is None:
            return None
        try:
            try:
                return self.chassis.update(detections, t_mono).to_report()
            except ValueError:                                   # frames out of timestamp order: a new stream, start the track again
                self.chassis.reset()
                return self.chassis.update(detections, t_mono).to_report()
        except Exception:                                        # pragma: no cover - defensive
            return None

    # ---- for the round manifest ----

    def calibration_snapshot(self) -> dict[str, Any] | None:
        with self._lock:
            if self.clicks is None:
                return None
            return {
                "profile": self.profile, "kind": self.clicks.kind, "synthetic_fixture": self.clicks.synthetic,
                "image_size": list(self.clicks.image_size) if self.clicks.image_size else None,
                "points_px": {name: list(point) for name, point in self.clicks.points.items()},
                "roles": dict(self.clicks.roles), "saved_at": self.clicks.meta.get("saved_at"),
                "lens": self.clicks.meta.get("lens"),
                "mapping": self.mapping.to_report() if self._fit_size else None,
                "fit_quality": self.quality.to_report() if self._fit_size else None,
                "camera_position": self.camera.to_report(),
            }

    def marker_layout(self) -> dict[str, Any]:
        return {"dictionary": self.config.dictionary,
                "placeholders_not_measured": list(self.config.placeholders),
                "markers": [{"id": spec.marker_id, "face": spec.face, "forward_m": spec.forward_m, "left_m": spec.left_m,
                             "yaw_deg": spec.yaw_deg, "size_m": spec.size_m, "height_m": spec.height_m}
                            for spec in self.config.markers.values()]}


def clicks_from_snapshot(snapshot: dict[str, Any] | None) -> ClickSet | None:
    """Rebuild the clicked points stored in a round manifest (for offline reconstruction)."""
    if not snapshot or not snapshot.get("points_px"):
        return None
    meta = {"profile_kind": snapshot.get("kind", "live"), "profile": snapshot.get("profile"),
            "synthetic_fixture": bool(snapshot.get("synthetic_fixture")), "lens": snapshot.get("lens")}
    size = snapshot.get("image_size")
    return ClickSet({name: (float(p[0]), float(p[1])) for name, p in snapshot["points_px"].items()},
                    dict(snapshot.get("roles") or {}), None if not size else (int(size[0]), int(size[1])), meta)
