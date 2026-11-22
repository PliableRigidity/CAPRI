"""Read-only arena camera overlay: floor mapping check plus ArUco bot pose.

    python -m control_center.tools.arena_overlay --camera 0
    python -m control_center.tools.arena_overlay --image frame.png
    python -m control_center.tools.arena_overlay --video clip.avi --no-window --report out.json
    python -m control_center.tools.arena_overlay --camera 0 --calibrate      (fresh live calibration)
    python -m control_center.tools.arena_overlay --video match.mp4 --profile NAME --start 433 --duration 22 --track

Profiles keep calibrations apart. "live" (the default) is your own camera. A profile made from
recorded footage is marked historical and is refused for a live camera.

This tool only reads frames and draws on them. It never imports the robot link or the command
manager, never opens a serial port and never sends a motor command.

Window keys:
    c      start/stop clicking the floor reference points (left click places the highlighted one)
    f      freeze / unfreeze the picture (same as space): freeze, then click the landmarks you can see
    1-9,0  jump to that landmark in the list (0 = the tenth)   b  previous landmark
    n      skip the highlighted point (also right click)      u  undo the last click
    t      swap the highlighted point between FIT and CHECK   x  clear every click
    s      save the clicked points to the --points file
    p      save a snapshot (overlay PNG + JSON report)        space  pause/resume
    h      full / compact status text                         q/Esc  quit
"""

from __future__ import annotations

import argparse
from collections import deque
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
import time
from typing import Any

import cv2
import numpy as np
import yaml

from control_center.backend.vision import TagDetection
from control_center.tools.overlay_core import (
    DEFAULT_CLICKS_PATH, DEFAULT_CONFIG_PATH, LIVE_PROFILE, REPO_ROOT, UNKNOWN, CameraEstimate,
    ClickSession, ClickSet, FitQuality, FloorMapping, MarkerDetector, OverlayConfig, PixelModel,
    PoseStatus, PoseTracker, arena_outline, assess_fit, densify, estimate_bot_pose,
    fit_floor_mapping, flipper_boundary, load_clicks, load_config, pit_arcs, profile_points_path,
    resolve_camera, save_clicks, topdown_pixel, topdown_view)

WINDOW = "Capri arena overlay (read-only, no motor commands)"
CONTROL_CENTER_CONFIG = REPO_ROOT / "config" / "control_center.yaml"
SNAPSHOT_DIR = REPO_ROOT / "artifacts" / "vision-overlay-snapshots"

BOUNDARY, PIT, FLIPPER = (255, 210, 60), (60, 60, 255), (0, 165, 255)
FIT, CHECK, WARN, GOOD, TEXT, DIM = ((80, 220, 80), (0, 230, 255), (60, 60, 255),
                                     (120, 230, 120), (240, 240, 240), (170, 170, 170))

POSITION_HUD = {
    "measured": "height-corrected (camera position measured)",
    "derived": "height-corrected (camera position DERIVED from mapping, unverified)",
    "none": "APPROXIMATE - raised marker projected to floor, NOT height-corrected",
}

WHEN_TO_RECALIBRATE = [
    "Camera or stand moved, bumped, re-aimed or zoomed: rerun with --calibrate and click the "
    "landmarks again. The saved points are pixels of one camera position.",
    "Camera resolution changed: the tool refuses the old points and says so; rerun with "
    "--calibrate at the new resolution.",
    "Quick test before a match: the drawn outline must sit on the floor edges. If it has "
    "slipped off them, the camera moved.",
    "Arena dimensions re-measured: edit config/arena_overlay.yaml; the clicks stay valid.",
]

HOW_TO_MEASURE_POSITION_ERROR = [
    "Tape 5 to 9 small crosses on the arena floor, including some close to the walls and both "
    "the near and far side from the camera; tape-measure each from the x=0 and y=0 walls.",
    "Mark the bot's centre point (the point the marker offsets in bot.markers are measured "
    "from) and set it over a cross, nose along a known direction such as +x.",
    "Run this tool with --truth X,Y,HEADING_DEG (metres, degrees) for that cross and read the "
    "'truth error' line, or press p to save a snapshot report for each cross.",
    "The 'floor projection' error is what you would get with no height correction; it should "
    "point away from the camera and grow with distance from it. The corrected error is what "
    "remains. A leftover error that still points toward/away from the camera means the marker "
    "height (bot.marker_height_m) or camera position (camera.position_m) is off.",
    "Repeat with the bot upside down to check the bottom markers.",
]


# --------------------------------------------------------------------------------------
# Frame sources: saved image, saved video, live camera
# --------------------------------------------------------------------------------------

class ImageSource:
    kind = "image"
    fps = 0.0

    def __init__(self, path: str) -> None:
        self.description = str(path)
        self._frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if self._frame is None:
            raise RuntimeError(f"could not read image {path}")

    def read(self) -> tuple[bool, np.ndarray | None, float]:
        return True, self._frame.copy(), 0.0

    def now(self) -> float:
        return 0.0

    def close(self) -> None:
        pass


class VideoSource:
    kind = "video"

    def __init__(self, path: str, start_s: float = 0.0, duration_s: float | None = None) -> None:
        self.description = str(path)
        self._cap = cv2.VideoCapture(str(path))
        if not self._cap.isOpened():
            raise RuntimeError(f"could not open video {path}")
        fps = float(self._cap.get(cv2.CAP_PROP_FPS) or 0.0)
        self.fps = fps if 1.0 <= fps <= 240.0 else 30.0
        self._index = max(0, int(round(start_s * self.fps)))
        if self._index:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, self._index)
        self._end = None if duration_s is None else self._index + max(1, int(round(duration_s * self.fps)))
        self._time = self._index / self.fps

    def sample(self, count: int) -> list[np.ndarray]:
        """``count`` frames spread over the whole file (own reader; playback is not disturbed)."""
        cap = cv2.VideoCapture(self.description)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        frames = []
        for index in np.linspace(0, max(0, total - 1), max(1, count)).astype(int):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
            ok, frame = cap.read()
            if ok and frame is not None:
                frames.append(frame)
        cap.release()
        return frames

    def read(self) -> tuple[bool, np.ndarray | None, float]:
        if self._end is not None and self._index >= self._end:
            return False, None, self._time
        ok, frame = self._cap.read()
        if not ok or frame is None:
            return False, None, self._time
        self._time = self._index / self.fps
        self._index += 1
        return True, frame, self._time

    def now(self) -> float:
        return self._time

    def close(self) -> None:
        self._cap.release()


def camera_settings(args: argparse.Namespace) -> dict[str, Any]:
    """Capture settings: the Control Centre's config values unless overridden on the CLI."""
    base: dict[str, Any] = {}
    if CONTROL_CENTER_CONFIG.exists():
        base = yaml.safe_load(CONTROL_CENTER_CONFIG.read_text(encoding="utf-8")) or {}
    return {
        "backend": str(args.backend or base.get("camera_backend", "auto")).lower(),
        "width": int(args.width or base.get("camera_width", 1280)),
        "height": int(args.height or base.get("camera_height", 720)),
        "fps": float(args.fps or base.get("camera_fps", 30)),
        "fourcc": str(base.get("camera_fourcc", "")).strip(),
    }


class CameraSource:
    kind = "camera"

    def __init__(self, index: int, settings: dict[str, Any]) -> None:
        backend_id = {"dshow": cv2.CAP_DSHOW, "msmf": cv2.CAP_MSMF}.get(settings["backend"])

        def open_capture():
            return (cv2.VideoCapture(index) if backend_id is None
                    else cv2.VideoCapture(index, backend_id))

        cap = open_capture()
        if not cap.isOpened():
            cap.release()
            raise RuntimeError(f"camera index {index} did not open (backend "
                               f"{settings['backend']}); try another --camera index")
        # Same order as control_center/backend/camera_worker.py: the C525 over DirectShow only
        # honours MJPG when FOURCC is requested after the size and frame rate.
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(settings["width"]))
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(settings["height"]))
        cap.set(cv2.CAP_PROP_FPS, float(settings["fps"]))
        if len(settings["fourcc"]) == 4:
            cap.set(cv2.CAP_PROP_FOURCC, float(cv2.VideoWriter_fourcc(*settings["fourcc"])))
        ok, frame = cap.read()
        if not ok or frame is None:
            cap.release()
            cap = open_capture()                     # fall back to the camera's default mode
            ok, frame = cap.read()
            if not ok or frame is None:
                cap.release()
                raise RuntimeError(f"camera index {index} opened but returned no frame")
        self._cap = cap
        self._pending: np.ndarray | None = frame
        self._failures = 0
        self.fps = float(settings["fps"])
        self.description = (f"camera {index} ({settings['backend']}, "
                            f"{frame.shape[1]}x{frame.shape[0]})")

    def read(self) -> tuple[bool, np.ndarray | None, float]:
        if self._pending is not None:
            frame, self._pending = self._pending, None
            return True, frame, time.monotonic()
        ok, frame = self._cap.read()
        if not ok or frame is None:
            self._failures += 1
            return self._failures < 30, None, time.monotonic()
        self._failures = 0
        return True, frame, time.monotonic()

    def now(self) -> float:
        return time.monotonic()

    def close(self) -> None:
        self._cap.release()


# --------------------------------------------------------------------------------------
# Drawing
# --------------------------------------------------------------------------------------

def _pixel_polyline(points: np.ndarray) -> np.ndarray | None:
    if len(points) == 0 or not np.isfinite(points).all():
        return None
    return np.round(np.clip(points, -20000, 20000)).astype(np.int32).reshape(-1, 1, 2)


def overlay_shapes(arena: dict[str, Any]) -> list[tuple[str, np.ndarray, tuple[int, int, int]]]:
    """Floor-level polylines drawn over the image: boundary, pit rims, fixed flipper edge."""
    shapes = [("arena boundary", arena_outline(arena), BOUNDARY)]
    shapes += [("pit boundary", arc, PIT) for arc in pit_arcs(arena)]
    shapes.append(("fixed flipper boundary", flipper_boundary(arena), FLIPPER))
    return shapes


def draw_arena(frame: np.ndarray, mapping: FloorMapping, arena: dict[str, Any]) -> None:
    for label, polyline, colour in overlay_shapes(arena):
        pixels = _pixel_polyline(mapping.floor_to_pixel(densify(polyline)))
        if pixels is None:
            continue
        cv2.polylines(frame, [pixels], False, colour, 2, cv2.LINE_AA)
        if label != "arena boundary":
            anchor = pixels[len(pixels) // 2, 0]
            text = "PIT" if label == "pit boundary" else "FLIPPER (fixed)"
            cv2.putText(frame, text, (int(anchor[0]) + 6, int(anchor[1]) - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, 1, cv2.LINE_AA)


def draw_reference_points(frame: np.ndarray, mapping: FloorMapping, session: ClickSession,
                          warn_mm: float) -> None:
    errors = {error.name: error for error in mapping.errors}
    for ref in session.references:
        pixel = session.clicks.points.get(ref.name)
        if pixel is None:
            continue
        role = session.role_of(ref)
        colour = FIT if role == "fit" else CHECK
        centre = (int(round(pixel[0])), int(round(pixel[1])))
        cv2.drawMarker(frame, centre, colour, cv2.MARKER_CROSS, 14, 1, cv2.LINE_AA)
        text = ref.name
        error = errors.get(ref.name)
        if error is not None and mapping.valid:
            if role == "check":
                text = f"{ref.name} CHECK {error.error_mm:.1f} mm"
                if error.error_mm > warn_mm:
                    colour = WARN
                target = mapping.floor_to_pixel([ref.arena_xy])[0]
                if np.isfinite(target).all():
                    cv2.circle(frame, (int(round(target[0])), int(round(target[1]))), 6, colour,
                               1, cv2.LINE_AA)
        cv2.putText(frame, text, (centre[0] + 8, centre[1] + 14), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, colour, 1, cv2.LINE_AA)


def _as_seen(xy: Any, height_m: float, camera: CameraEstimate) -> np.ndarray:
    """Floor point that lies on the camera ray through (x, y, height): for drawing only."""
    point = np.asarray(xy, float)
    if camera.position_m is None or height_m <= 0.0:
        return point
    cam = np.asarray(camera.position_m, float)
    k = height_m / cam[2]
    return (point - k * cam[:2]) / (1.0 - k)


def draw_bot(frame: np.ndarray, mapping: FloorMapping, camera: CameraEstimate,
             status: PoseStatus, detections: list[TagDetection], config: OverlayConfig) -> None:
    for detection in detections:                           # thin outlines, small IDs
        known = detection.tag_id in config.markers
        colour = (70, 210, 120) if known else (150, 150, 150)
        cv2.polylines(frame, [np.round(detection.corners).astype(np.int32)], True, colour, 1,
                      cv2.LINE_AA)
        corner = detection.corners[0]
        cv2.putText(frame, str(detection.tag_id), (int(corner[0]) - 12, int(corner[1]) - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, colour, 1, cv2.LINE_AA)
    pose = status.pose
    if not pose.ok or not mapping.valid:
        return
    height = 0.0
    if pose.position_quality == "height-corrected" and pose.used_ids:
        height = float(np.mean([config.markers[i].height_m for i in pose.used_ids]))
    centre = np.array([pose.x_m, pose.y_m])
    nose = centre + 0.12 * np.array([math.cos(pose.heading_rad), math.sin(pose.heading_rad)])
    pixels = mapping.floor_to_pixel([_as_seen(centre, height, camera),
                                     _as_seen(nose, height, camera), centre])
    if not np.isfinite(pixels).all():
        return
    start, end, floor_point = (tuple(int(round(v)) for v in p) for p in pixels)
    cv2.line(frame, floor_point, start, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.circle(frame, floor_point, 3, (255, 255, 255), -1, cv2.LINE_AA)   # floor under centre
    cv2.arrowedLine(frame, start, end, (255, 60, 255), 3, cv2.LINE_AA, tipLength=0.3)
    cv2.circle(frame, start, 5, (255, 60, 255), -1, cv2.LINE_AA)


def draw_hud(frame: np.ndarray, lines: list[tuple[str, tuple[int, int, int]]],
             prompt: str | None = None) -> None:
    scale, line_height = 0.42, 16
    width = max((cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)[0][0]
                 for text, _ in lines), default=0) + 16
    height = line_height * len(lines) + 6
    width, height = min(width, frame.shape[1]), min(height, frame.shape[0])
    panel = frame[0:height, 0:width]
    panel[:] = (panel * 0.35).astype(np.uint8)
    for row, (text, colour) in enumerate(lines):
        cv2.putText(frame, text, (8, 15 + row * line_height), cv2.FONT_HERSHEY_SIMPLEX, scale,
                    colour, 1, cv2.LINE_AA)
    if prompt:
        y = frame.shape[0] - 34
        band = frame[y - 22:y + 10, :]
        band[:] = (band * 0.3).astype(np.uint8)
        cv2.putText(frame, prompt, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1,
                    cv2.LINE_AA)


def draw_landmark_list(frame: np.ndarray, session: ClickSession) -> None:
    """Named landmarks down the right-hand side while clicking: number, name, placed or not."""
    rows = session.references
    scale, line_height = 0.4, 15
    width = min(frame.shape[1], 300)
    x0, height = frame.shape[1] - width, min(frame.shape[0], line_height * (len(rows) + 1) + 6)
    panel = frame[0:height, x0:]
    panel[:] = (panel * 0.3).astype(np.uint8)
    cv2.putText(frame, "LANDMARKS (press its number, then click it)", (x0 + 6, 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.36, TEXT, 1, cv2.LINE_AA)
    for row, ref in enumerate(rows):
        placed = ref.name in session.clicks.points
        role = session.role_of(ref).upper()
        colour = (FIT if role == "FIT" else CHECK) if placed else DIM
        marker = ">" if row == session.index else " "
        key = str((row + 1) % 10) if row < 10 else "-"
        text = f"{marker}{key} {ref.name} {'[' + role + ']' if placed else '(not placed)'}"
        cv2.putText(frame, text, (x0 + 6, 12 + (row + 1) * line_height), cv2.FONT_HERSHEY_SIMPLEX,
                    scale, (0, 255, 255) if row == session.index else colour, 1, cv2.LINE_AA)


TRACK_COLOURS = [(80, 220, 80), (255, 170, 60), (220, 120, 255), (60, 220, 255), (200, 200, 90)]


def track_colour(track_id: int) -> tuple[int, int, int]:
    return TRACK_COLOURS[(track_id - 1) % len(TRACK_COLOURS)]


def draw_tracks(frame: np.ndarray, tracker) -> None:
    """Blob outlines and track labels on the camera picture."""
    from control_center.tools.blob_tracker import OVERLAP, TRACKED
    for blob in tracker.blobs:
        x, y, w, h = blob.box
        cv2.rectangle(frame, (x, y), (x + w, y + h), (200, 200, 200), 1)
    for track in tracker.tracks:
        if track.state == TRACKED and track.pixel is not None:
            centre = (int(round(track.pixel[0])), int(round(track.pixel[1])))
            cv2.circle(frame, centre, 4, track_colour(track.track_id), -1, cv2.LINE_AA)
            cv2.putText(frame, track.label, (centre[0] + 6, centre[1] - 6), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, track_colour(track.track_id), 1, cv2.LINE_AA)
    for group, position in tracker.overlap_at.items():
        members = "+".join(t.label for t in tracker.tracks if t.state == OVERLAP and t.group == group)
        pixel = tracker.mapping.floor_to_pixel([position])[0]
        if members and np.isfinite(pixel).all():
            cv2.putText(frame, f"{members} OVERLAP", (int(pixel[0]) + 6, int(pixel[1]) - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 165, 255), 1, cv2.LINE_AA)


def track_panel(frame: np.ndarray, mapping: FloorMapping, arena: dict[str, Any], tracker,
                height: int) -> np.ndarray:
    """Top-down view of the floor with the mapped tracks, scaled to ``height`` pixels."""
    from control_center.tools.blob_tracker import OVERLAP, TRACKED
    per_m = 300.0
    view = topdown_view(frame, mapping, arena, per_m)
    to_px = lambda xy: np.round(topdown_pixel(xy, arena, per_m)).astype(np.int32)
    for polyline, colour in ([(arena_outline(arena), BOUNDARY)] + [(arc, PIT) for arc in pit_arcs(arena)]
                             + [(flipper_boundary(arena), FLIPPER)]):
        cv2.polylines(view, [to_px(polyline).reshape(-1, 1, 2)], False, colour, 1, cv2.LINE_AA)
    if tracker is not None:
        for track in tracker.tracks:
            colour = track_colour(track.track_id)
            trail = track.history[-90:]
            if len(trail) >= 2:
                cv2.polylines(view, [to_px(trail).reshape(-1, 1, 2)], False, colour, 1, cv2.LINE_AA)
            if track.state == TRACKED and track.floor is not None:
                centre = tuple(int(v) for v in to_px([track.floor])[0])
                cv2.circle(view, centre, 6, colour, -1, cv2.LINE_AA)
                cv2.putText(view, track.label, (centre[0] + 8, centre[1] - 8), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, colour, 1, cv2.LINE_AA)
        for group, position in tracker.overlap_at.items():
            members = "+".join(t.label for t in tracker.tracks if t.state == OVERLAP and t.group == group)
            if members:
                centre = tuple(int(v) for v in to_px([position])[0])
                cv2.circle(view, centre, 16, (0, 165, 255), 1, cv2.LINE_AA)
                cv2.putText(view, f"{members} overlap", (centre[0] + 18, centre[1] + 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 165, 255), 1, cv2.LINE_AA)
    cv2.putText(view, "mapped floor (top-down, +y up)", (6, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                TEXT, 1, cv2.LINE_AA)
    scale = height / view.shape[0]
    return cv2.resize(view, (int(round(view.shape[1] * scale)), height), interpolation=cv2.INTER_LINEAR)


# --------------------------------------------------------------------------------------
# The overlay application (no window code here, so it runs headless)
# --------------------------------------------------------------------------------------

def parse_truth(text: str | None) -> tuple[float, float, float | None] | None:
    if not text:
        return None
    parts = [float(value) for value in text.replace(" ", "").split(",")]
    if len(parts) not in (2, 3):
        raise ValueError("--truth expects X,Y or X,Y,HEADING_DEG")
    return parts[0], parts[1], (parts[2] if len(parts) == 3 else None)


class OverlayApp:
    def __init__(self, config: OverlayConfig, clicks: ClickSet, source_description: str,
                 truth: tuple[float, float, float | None] | None = None) -> None:
        self.config = config
        self.clicks = clicks
        self.source_description = source_description
        self.truth = truth
        self.detector = MarkerDetector(config.dictionary)
        self.tracker = PoseTracker(config.max_detection_age_s)
        self.session = ClickSession(config.references, clicks)
        # A profile made from footage carries that camera's own lens; otherwise the config's.
        self.pixel_model = clicks.lens if clicks.lens.active else PixelModel(config.camera_matrix, config.distortion)
        self.profile_name = LIVE_PROFILE
        self.calibrating = False
        self.frozen = False
        self.quality = FitQuality()
        self.blob_tracker = None
        self.side_by_side = False
        self.raw_frame: np.ndarray | None = None
        self.mapping = FloorMapping()
        self.camera = CameraEstimate()
        self.click_mode = False
        self.show_full_hud = True
        self.dirty = True
        self.message = ""
        self._frame_size: tuple[int, int] | None = None
        self._clicks_size = clicks.image_size
        self.status = self.tracker.status(0.0)
        self.detections: list[TagDetection] = []

    def refit(self, frame_size: tuple[int, int]) -> None:
        self.mapping = fit_floor_mapping(
            self.config.references, self.clicks.points, frame_size, self.pixel_model,
            roles=self.clicks.roles, clicks_image_size=self._clicks_size,
            max_fit_residual_mm=self.config.max_fit_residual_mm)
        self.camera = resolve_camera(self.config, self.mapping)
        self.quality = assess_fit(
            self.config.references, self.clicks.points, self.clicks.roles, self.mapping,
            self.config.arena, self.config.warn_check_error_mm,
            self.config.weak_predicted_error_mm, self.config.min_hull_fraction)
        self._frame_size, self.dirty = frame_size, False

    def clicks_changed(self) -> None:
        # Points placed in this session belong to the current frame size.
        self._clicks_size = self.clicks.image_size = self._frame_size
        self.dirty = True

    def place_click(self, x: float, y: float) -> str | None:
        if (self._clicks_size is not None and self._frame_size is not None
                and tuple(self._clicks_size) != tuple(self._frame_size)):
            self.session.clear()                  # loaded points were for another resolution
            self.message = "points from another image size were cleared"
        name = self.session.click(x, y)
        if name:
            self.clicks_changed()
        return name

    def process(self, frame: np.ndarray, timestamp: float, source) -> np.ndarray:
        size = (frame.shape[1], frame.shape[0])
        if self.dirty or size != self._frame_size:
            self.refit(size)
        self.raw_frame = frame
        self.detections = self.detector.detect_markers(frame)
        pose = estimate_bot_pose(self.detections, self.mapping, self.camera, self.config)
        self.tracker.update(pose, timestamp)
        self.status = self.tracker.status(source.now())
        if self.blob_tracker is not None and not self.frozen:
            self.blob_tracker.update(frame)
        return self.render(frame)

    # ---- text -------------------------------------------------------------------------
    def truth_error(self) -> dict[str, Any] | None:
        pose = self.status.pose
        if self.truth is None or not pose.ok:
            return None
        tx, ty, heading = self.truth
        result: dict[str, Any] = {
            "truth_m": [tx, ty], "truth_heading_deg": heading,
            "reported_error_mm": 1000.0 * math.hypot(pose.x_m - tx, pose.y_m - ty),
            "floor_projection_error_mm": None, "heading_error_deg": None}
        if pose.floor_projection_xy is not None:
            fx, fy = pose.floor_projection_xy
            result["floor_projection_error_mm"] = 1000.0 * math.hypot(fx - tx, fy - ty)
        if heading is not None:
            result["heading_error_deg"] = (pose.heading_deg - heading + 180.0) % 360.0 - 180.0
        return result

    def mapping_line(self) -> tuple[str, tuple[int, int, int]]:
        mapping = self.mapping
        if not mapping.valid:
            return f"MAP: NOT VALID - {mapping.reason}", WARN
        fit, check = mapping.fit_stats, mapping.check_stats
        if not check["count"]:
            return (f"MAP: fit {fit['count']} pts | NO CHECK POINTS - mapping error UNVERIFIED",
                    CHECK)
        colour = WARN if check["max_mm"] > self.config.warn_check_error_mm else GOOD
        return (f"MAP: fit {fit['count']} pts | held-out check {check['count']} pts: "
                f"max {check['max_mm']:.1f} mm, mean {check['mean_mm']:.1f} mm", colour)

    def profile_line(self) -> tuple[str, tuple[int, int, int]]:
        clicks = self.clicks
        if clicks.historical:
            return (f"PROFILE: {self.profile_name} - HISTORICAL footage, APPROXIMATE map; "
                    "not valid for a live camera", CHECK)
        if not clicks.points:
            return (f"PROFILE: {self.profile_name} - not calibrated yet (c = click landmarks, "
                    "f = freeze)", WARN)
        when = str(clicks.meta.get("saved_at", "this session, not saved"))[:16]
        size = "x".join(str(v) for v in clicks.image_size) if clicks.image_size else "?"
        return (f"PROFILE: {self.profile_name} - calibrated {when} at {size}; recalibrate if the "
                "camera moves", TEXT)

    def fit_line(self) -> tuple[str, tuple[int, int, int]]:
        quality = self.quality
        colour = {"ok": GOOD, "weak": CHECK, "insufficient": WARN}[quality.level]
        if quality.level == "ok":
            detail = (f"{quality.fit_points} fit + {quality.check_points} check points; a 1 px "
                      f"click error moves the map <= {quality.predicted_worst_mm:.0f} mm")
        else:
            detail = quality.reasons[0] if quality.reasons else ""
            if len(quality.reasons) > 1:
                detail += f" (+{len(quality.reasons) - 1} more)"
        return f"FIT: {quality.level.upper()} - {detail}", colour

    def track_lines(self) -> list[tuple[str, tuple[int, int, int]]]:
        from control_center.tools.blob_tracker import LOST, OVERLAP
        tracker = self.blob_tracker
        if tracker is None:
            return []
        parts = []
        for track in tracker.tracks:
            if track.state == OVERLAP:
                parts.append(f"{track.label} OVERLAP (no position)")
            elif track.state == LOST:
                parts.append(f"{track.label} LOST (no position)")
            else:
                still = " still" if track.still_frames >= 15 else ""
                parts.append(f"{track.label} x {track.floor[0]:.2f} y {track.floor[1]:.2f}{still}")
        return [("BLOBS (unmarked, approximate, identity = same blob only, no heading): "
                 + (" | ".join(parts) if parts else "none"), TEXT)]

    def status_lines(self, compact: bool = False) -> list[tuple[str, tuple[int, int, int]]]:
        """HUD text. ``compact`` keeps only the pose, its quality label and the age."""
        status, pose = self.status, self.status.pose
        lines = [(f"READ-ONLY OVERLAY (no motor commands) | {self.source_description}", DIM)]
        if self.clicks.synthetic:
            lines.append(("SYNTHETIC FIXTURE reference points - not real-camera accuracy", CHECK))
        lines.append(self.profile_line())
        lines.append(self.mapping_line())
        lines.append(self.fit_line())
        if self.frozen:
            lines.append(("FROZEN picture (f = back to live)", CHECK))
        lines += self.track_lines()
        if compact:
            lines = []
        if pose.ok:
            lines.append((f"BOT: x {pose.x_m:.3f} m  y {pose.y_m:.3f} m  nose heading "
                          f"{pose.heading_deg:+.1f} deg  {pose.orientation.upper()}", GOOD))
            lines.append((f"POSITION: {POSITION_HUD[pose.correction_source]}",
                          GOOD if pose.correction_source == "measured" else CHECK))
        else:
            lines.append((f"BOT: {UNKNOWN} - {pose.reason}", WARN))
            lines.append((f"POSITION: {UNKNOWN}  heading: {UNKNOWN}  orientation: "
                          f"{pose.orientation}", WARN))
        if status.detection_age_s is None:
            age = "detection age: no valid detection yet"
        elif pose.ok:
            age = f"detection age: {status.detection_age_s * 1000:.0f} ms"
        else:
            age = f"last valid detection: {status.detection_age_s:.2f} s ago (not current)"
        seen = " ".join(str(i) for i in pose.visible_ids) or "none"
        extra = f" | other ids: {pose.ignored_ids}" if pose.ignored_ids else ""
        lines.append((f"{age} | bot markers seen: {seen}{extra}", TEXT))
        if compact:
            return lines
        truth = self.truth_error()
        if truth is not None:
            floor = truth["floor_projection_error_mm"]
            heading = truth["heading_error_deg"]
            lines.append((f"truth error: {truth['reported_error_mm']:.1f} mm | "
                          f"uncorrected floor projection {floor:.1f} mm"
                          + (f" | heading {heading:+.1f} deg" if heading is not None else ""),
                          TEXT))
        if self.config.placeholder_tags:
            lines.append(("PLACEHOLDER config in use: "
                          + ", ".join(self.config.placeholder_tags), CHECK))
        if self.message:
            lines.append((self.message, TEXT))
        return lines

    def prompt(self) -> str | None:
        if not self.click_mode:
            return None
        ref = self.session.current
        if ref is None:
            return "All reference points placed. s = save, u = undo, c = leave click mode"
        placed = len(self.clicks.points)
        verb = "RE-CLICK (already placed)" if ref.name in self.clicks.points else "CLICK"
        return (f"{verb} {ref.name} [{self.session.role_of(ref).upper()}] at "
                f"({ref.arena_xy[0]:.3f}, {ref.arena_xy[1]:.3f}) m, floor level: {ref.note}  "
                f"({placed} placed; n skip, t fit/check, u undo, s save)")

    def render(self, frame: np.ndarray) -> np.ndarray:
        out = frame.copy()
        if self.mapping.valid:
            draw_arena(out, self.mapping, self.config.arena)
        draw_reference_points(out, self.mapping, self.session, self.config.warn_check_error_mm)
        draw_bot(out, self.mapping, self.camera, self.status, self.detections, self.config)
        if self.blob_tracker is not None:
            draw_tracks(out, self.blob_tracker)
        if self.side_by_side:
            return self.compose(frame, out)
        if self.click_mode:
            draw_landmark_list(out, self.session)
        draw_hud(out, self.status_lines(compact=not self.show_full_hud), self.prompt())
        return out

    def compose(self, raw: np.ndarray, annotated: np.ndarray) -> np.ndarray:
        """Camera picture and mapped top-down view side by side, status text on a strip below."""
        factor = 2 if raw.shape[1] < 960 else 1
        left = cv2.resize(annotated, None, fx=factor, fy=factor, interpolation=cv2.INTER_LINEAR)
        if self.mapping.valid:
            right = track_panel(raw, self.mapping, self.config.arena, self.blob_tracker, left.shape[0])
        else:
            right = np.zeros((left.shape[0], left.shape[0], 3), np.uint8)
        lines = self.status_lines(compact=False)[:10]
        strip = np.zeros((16 * 10 + 8, left.shape[1] + right.shape[1], 3), np.uint8)     # fixed size: the saved video needs one frame size
        for row, (text, colour) in enumerate(lines):
            cv2.putText(strip, text, (8, 15 + row * 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, colour, 1, cv2.LINE_AA)
        return np.vstack([np.hstack([left, right]), strip])

    # ---- report -----------------------------------------------------------------------
    def report(self, source_info: dict[str, Any], frames: list[dict[str, Any]]) -> dict[str, Any]:
        pose = self.status.to_report()
        return {
            "tool": "control_center.tools.arena_overlay",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "read_only": True,
            "motor_commands_sent": False,
            "source": source_info,
            "synthetic_fixture_input": self.clicks.synthetic,
            "synthetic_note": ("Reference points come from the SYNTHETIC fixture; these numbers "
                               "are not real-camera accuracy." if self.clicks.synthetic else None),
            "config_file": str(self.config.path) if self.config.path else None,
            "placeholder_values_in_use": self.config.placeholders,
            "profile": {"name": self.profile_name, "kind": self.clicks.kind,
                        "approximate": bool(self.clicks.historical),
                        "calibrated_at": self.clicks.meta.get("saved_at"),
                        "calibrated_on": self.clicks.meta.get("source"),
                        "when_to_recalibrate": WHEN_TO_RECALIBRATE},
            "mapping": self.mapping.to_report(),
            "fit_quality": self.quality.to_report(),
            "tracking": None if self.blob_tracker is None else {
                "kind": "unmarked blobs (offline prototype)", "events": self.blob_tracker.events,
                "limits": ["identity means the same blob, not a known robot", "no heading",
                           "positions are blob centres on the floor plane (approximate)",
                           "overlapping blobs give no positions"]},
            "camera": self.camera.to_report(),
            "bot_pose": pose,
            "truth_check": self.truth_error(),
            "how_to_measure_position_error": HOW_TO_MEASURE_POSITION_ERROR,
            "frames_processed": len(frames),
            "frames": frames,
        }

    def frame_record(self, index: int, timestamp: float) -> dict[str, Any]:
        pose = self.status.pose
        return {"index": index, "t_s": timestamp, "state": pose.state, "reason": pose.reason,
                "x_m": pose.x_m if pose.ok else UNKNOWN, "y_m": pose.y_m if pose.ok else UNKNOWN,
                "heading_deg": pose.heading_deg if pose.ok else UNKNOWN,
                "orientation": pose.orientation, "position_quality": pose.position_quality,
                "detection_age_s": self.status.detection_age_s,
                "visible_ids": pose.visible_ids,
                **({} if self.blob_tracker is None else {"blobs": self.blob_tracker.frame_report()})}


# --------------------------------------------------------------------------------------
# Command line
# --------------------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m control_center.tools.arena_overlay",
        description="Read-only arena overlay and ArUco bot pose (no motor commands).",
        epilog=__doc__.split("Window keys:")[1], formatter_class=argparse.RawTextHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--image", help="saved frame (PNG/JPG)")
    source.add_argument("--video", help="saved video file")
    source.add_argument("--camera", type=int, metavar="N", help="live camera index")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH),
                        help="overlay config (default config/arena_overlay.yaml)")
    parser.add_argument("--profile", default=LIVE_PROFILE,
                        help="calibration profile (default 'live' = your own camera, "
                             "config/arena_overlay_clicks.json). Other names live in "
                             "config/overlay_profiles/NAME/. A historical-footage profile is "
                             "refused for --camera.")
    parser.add_argument("--points", default=None,
                        help="clicked reference-point file to load and to save with key s "
                             "(default: the profile's file)")
    parser.add_argument("--calibrate", action="store_true",
                        help="fresh calibration: ignore any saved points, freeze the first frame "
                             "and start clicking landmarks; s saves (the old file is kept as .bak)")
    parser.add_argument("--start", type=float, default=0.0, help="video: start time in seconds")
    parser.add_argument("--duration", type=float, default=None, help="video: seconds to play")
    parser.add_argument("--track", action="store_true",
                        help="video only: offline unmarked-blob tracker, shown beside a top-down view")
    parser.add_argument("--background-samples", type=int, default=60,
                        help="--track: frames sampled over the whole video for the empty-floor picture")
    parser.add_argument("--save-video", help="write the annotated frames to this video file")
    parser.add_argument("--save-topdown", help="write a top-down view of the last frame to this PNG")
    parser.add_argument("--click", action="store_true", help="start in click mode")
    parser.add_argument("--truth", help="X,Y[,HEADING_DEG]: tape-measured bot centre to compare")
    parser.add_argument("--no-window", action="store_true", help="headless: no window, no keys")
    parser.add_argument("--compact-hud", action="store_true",
                        help="start with the short status text (key h toggles it)")
    parser.add_argument("--save-overlay", help="write the last annotated frame to this PNG")
    parser.add_argument("--report", help="write a JSON report to this file")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="stop after this many frames (headless camera default: 30)")
    parser.add_argument("--width", type=int, help="camera width (default: control_center.yaml)")
    parser.add_argument("--height", type=int, help="camera height (default: control_center.yaml)")
    parser.add_argument("--fps", type=float, help="camera fps (default: control_center.yaml)")
    parser.add_argument("--backend", choices=["dshow", "msmf", "auto"],
                        help="camera backend (default: control_center.yaml)")
    return parser


def open_source(args: argparse.Namespace):
    if args.image:
        return ImageSource(args.image)
    if args.video:
        return VideoSource(args.video, args.start, args.duration)
    return CameraSource(args.camera, camera_settings(args))


def _write_outputs(app: OverlayApp, annotated: np.ndarray | None, overlay_path: str | None,
                   report_path: str | None, source_info: dict[str, Any],
                   frames: list[dict[str, Any]]) -> dict[str, Any]:
    report = app.report(source_info, frames)
    if overlay_path and annotated is not None:
        Path(overlay_path).parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(overlay_path), annotated):
            raise OSError(f"could not write {overlay_path}")
        report["overlay_image"] = str(overlay_path)
    if report_path:
        Path(report_path).parent.mkdir(parents=True, exist_ok=True)
        Path(report_path).write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def _handle_key(key: int, app: OverlayApp, points_path: Path) -> str | None:
    """Apply one window key. Returns "quit", "pause", "snapshot" or None."""
    if key in (ord("q"), 27):
        return "quit"
    if key in (ord(" "), ord("f")):
        return "pause"
    if key == ord("p"):
        return "snapshot"
    if key == ord("h"):
        app.show_full_hud = not app.show_full_hud
    elif key == ord("c"):
        app.click_mode = not app.click_mode
        app.message = "click mode ON" if app.click_mode else "click mode off"
    elif key == ord("s"):
        app.message = save_profile(app, points_path)
    elif app.click_mode and ord("0") <= key <= ord("9"):
        app.session.select((key - ord("1")) % 10)
    elif app.click_mode and key == ord("b"):
        app.session.select(app.session.index - 1)
    elif app.click_mode and key == ord("n"):
        app.session.skip()
    elif app.click_mode and key == ord("u"):
        if app.session.undo():
            app.clicks_changed()
    elif app.click_mode and key == ord("x"):
        app.session.clear()
        app.clicks_changed()
    elif app.click_mode and key == ord("t"):
        if app.session.toggle_role():
            app.dirty = True
    return None


def save_profile(app: OverlayApp, points_path: Path) -> str:
    """Save the clicked points. A fresh calibration never overwrites the old file without a copy."""
    if app.quality.level == "insufficient":
        return f"NOT saved: {app.quality.reasons[0] if app.quality.reasons else 'no valid fit'}"
    note = ""
    if points_path.exists() and app.calibrating:
        backup = points_path.with_name(points_path.name + datetime.now().strftime(".bak-%Y%m%d-%H%M%S"))
        backup.write_bytes(points_path.read_bytes())
        note = f" (previous file kept as {backup.name})"
    if not app.clicks.historical:
        app.clicks.meta.update({"profile_kind": "live", "profile": app.profile_name,
                                "fit_quality_when_saved": app.quality.level})
    save_clicks(points_path, app.clicks, source=app.source_description)
    app.clicks.meta["saved_at"] = datetime.now(timezone.utc).isoformat()
    weak = "" if app.quality.level == "ok" else f" - fit is {app.quality.level.upper()}"
    return f"saved {len(app.clicks.points)} point(s) to {points_path.name}{note}{weak}"


def load_profile_clicks(args: argparse.Namespace, source_kind: str) -> tuple[ClickSet, Path]:
    """Pick the clicked-points file for this run, keeping footage profiles away from a live camera."""
    points_path = Path(args.points) if args.points else profile_points_path(args.profile)
    clicks = load_clicks(points_path) if points_path.exists() else ClickSet()
    if source_kind == "camera" and clicks.historical:
        raise ValueError(
            f"{points_path} is a HISTORICAL footage profile ({clicks.meta.get('source_video', 'recorded footage')}); "
            "it describes another camera position and is not loaded for a live camera. "
            "Run with --camera N --calibrate (profile 'live') to calibrate your own view.")
    if args.calibrate:
        if clicks.historical:
            raise ValueError("--calibrate would replace a historical footage profile; pick another --profile")
        clicks = ClickSet()                    # fresh: nothing from an earlier calibration is reused
    return clicks, points_path


def run(args: argparse.Namespace) -> dict[str, Any]:
    config = load_config(args.config)
    if args.track and not args.video:
        raise ValueError("--track is an offline prototype and needs --video")
    # The profile is checked before any camera is opened: a footage profile never reaches a live view.
    clicks, points_path = load_profile_clicks(args, "camera" if args.camera is not None else "file")
    from control_center.tools.arena_setup import load_arena_setup, with_arena_setup

    if args.calibrate and not args.points and load_arena_setup(args.profile) is not None:
        raise ValueError(f"profile '{args.profile}' has a saved arena set-up whose camera corners belong to the Command Centre: "
                         "use its ARENA tab (Recalibrate camera). Landmark calibration here would replace those corner clicks.")
    if not args.points:                                         # an explicit --points file bypasses the profile on purpose
        config = with_arena_setup(config, args.profile, clicks)  # the profile's saved arena and hazards, if it has any
    source = open_source(args)
    label = source.description if source.kind == "camera" else Path(source.description).name
    app = OverlayApp(config, clicks, f"{source.kind}: {label}", parse_truth(args.truth))
    app.profile_name = args.profile if not args.points else Path(args.points).stem
    app.calibrating = bool(args.calibrate)
    app.click_mode = bool(args.click or args.calibrate)
    app.side_by_side = bool(args.track)
    app.show_full_hud = not args.compact_hud
    source_info = {"kind": source.kind, "description": source.description, "profile": app.profile_name,
                   "start_s": args.start, "duration_s": args.duration,
                   "points_file": str(points_path), "points_file_found": points_path.exists()}
    writer: cv2.VideoWriter | None = None
    writer_size = (0, 0)
    frames: deque[dict[str, Any]] = deque(maxlen=5000)
    headless = bool(args.no_window)
    limit = args.max_frames
    if limit is None and headless:
        limit = {"image": 1, "camera": 30}.get(source.kind)
    annotated: np.ndarray | None = None
    raw: np.ndarray | None = None
    timestamp, count, loops, paused, ended = 0.0, 0, 0, False, False

    def start_tracker(first_frame: np.ndarray) -> None:
        from control_center.tools.blob_tracker import BlobTracker, median_background
        app.refit((first_frame.shape[1], first_frame.shape[0]))
        if not app.mapping.valid:
            raise ValueError(f"--track needs a valid floor mapping: {app.mapping.reason}")
        frames_for_background = source.sample(args.background_samples)
        app.blob_tracker = BlobTracker(app.mapping, config.arena, median_background(frames_for_background))
        source_info["tracker_background"] = (f"median of {len(frames_for_background)} frames spread over the whole "
                                             "video; anything present in most of them is invisible")

    if not headless:
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)

        def on_mouse(event: int, x: int, y: int, _flags: int, _param: Any) -> None:
            if not app.click_mode:
                return
            if event == cv2.EVENT_LBUTTONDOWN:
                app.place_click(x, y)
            elif event == cv2.EVENT_RBUTTONDOWN:
                app.session.skip()

        cv2.setMouseCallback(WINDOW, on_mouse)

    try:
        while True:
            loops += 1
            if not (paused or ended):
                ok, frame, frame_time = source.read()
                if frame is not None:
                    raw, timestamp = frame, frame_time
                    count += 1
                    if args.track and app.blob_tracker is None:
                        start_tracker(raw)
                    annotated = app.process(raw, timestamp, source)
                    frames.append(app.frame_record(count - 1, timestamp))
                    if args.save_video:
                        if writer is None:
                            Path(args.save_video).parent.mkdir(parents=True, exist_ok=True)
                            writer_size = (annotated.shape[1], annotated.shape[0])
                            writer = cv2.VideoWriter(str(args.save_video), cv2.VideoWriter_fourcc(*"mp4v"),
                                                     source.fps or 30.0, writer_size)
                        if (annotated.shape[1], annotated.shape[0]) == writer_size:
                            writer.write(annotated)
                    ended = source.kind == "image"    # a still is read once, then repainted
                    if args.calibrate and count == 1 and not headless:
                        paused = app.frozen = True    # fresh calibration starts on a frozen picture
                elif not ok:
                    ended = True                      # end of video, or the camera stopped
                    if source.kind == "camera":
                        app.message = "camera stopped returning frames"
            elif raw is not None and not headless:
                annotated = app.process(raw, timestamp, source)     # repaint after clicks
            if headless:
                if ended or (limit is not None and count >= limit):
                    break
                continue
            if annotated is not None:
                cv2.imshow(WINDOW, annotated)
            delay = 30 if paused or ended else (
                1 if source.kind == "camera" else max(1, int(1000 / source.fps)))
            action = _handle_key(cv2.waitKey(delay) & 0xFF, app, points_path)
            if action == "quit" or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                break
            if action == "pause":
                paused = not paused
                app.frozen = paused and source.kind != "image"
            elif action == "snapshot" and annotated is not None:
                stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
                _write_outputs(app, annotated, str(SNAPSHOT_DIR / f"overlay-{stamp}.png"),
                               str(SNAPSHOT_DIR / f"overlay-{stamp}.json"), source_info,
                               list(frames)[-1:])
                app.message = f"snapshot saved to {SNAPSHOT_DIR}"
            # --max-frames with a window is for unattended smoke runs: frames for a stream,
            # repaint loops once a still or a finished video is being held on screen.
            if limit is not None and (loops >= limit if ended else count >= limit):
                break
    finally:
        source.close()
        if writer is not None:
            writer.release()
        if not headless:
            cv2.destroyAllWindows()
    if args.save_topdown and raw is not None and app.mapping.valid:
        Path(args.save_topdown).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(args.save_topdown), track_panel(raw, app.mapping, config.arena, app.blob_tracker, 720))
    return _write_outputs(app, annotated, args.save_overlay, args.report, source_info,
                          list(frames))


def print_summary(report: dict[str, Any]) -> None:
    mapping, pose, camera = report["mapping"], report["bot_pose"], report["camera"]
    print("Arena overlay (read-only; no motor commands sent)")
    print(f"  source: {report['source']['kind']} {report['source']['description']}")
    if report["synthetic_fixture_input"]:
        print("  NOTE: SYNTHETIC fixture reference points; not real-camera accuracy")
    profile, quality = report["profile"], report["fit_quality"]
    print(f"  profile: {profile['name']} ({profile['kind']}"
          + (", APPROXIMATE historical footage map" if profile["approximate"] else "") + ")")
    print(f"  fit quality: {quality['level'].upper()}" + ("" if not quality["reasons"] else " - " + "; ".join(quality["reasons"])))
    if mapping["valid"]:
        check = mapping["check_points"]
        if check["count"]:
            print(f"  mapping: fit {mapping['fit_points']['count']} pts; held-out check "
                  f"{check['count']} pts: max {check['max_mm']:.2f} mm, "
                  f"mean {check['mean_mm']:.2f} mm")
        else:
            print(f"  mapping: fit {mapping['fit_points']['count']} pts; NO held-out check "
                  "points, mapping error unverified")
    else:
        print(f"  mapping: NOT VALID - {mapping['reason']}")
    print(f"  camera position: {camera['source']} {camera['position_m']} ({camera['detail']})")
    if pose["state"] == "ok":
        print(f"  bot: x {pose['x_m']:.3f} m, y {pose['y_m']:.3f} m, nose heading "
              f"{pose['heading_deg']:+.1f} deg, {pose['orientation']}, "
              f"detection age {pose['detection_age_s'] * 1000:.0f} ms")
        print(f"  position: {pose['position_label']}")
    else:
        print(f"  bot: {UNKNOWN} - {pose['reason']} (orientation: {pose['orientation']})")
    if report["truth_check"]:
        truth = report["truth_check"]
        print(f"  truth error: reported {truth['reported_error_mm']:.1f} mm, floor projection "
              f"{truth['floor_projection_error_mm']:.1f} mm")
    if report["frames_processed"] > 1:
        ok = sum(1 for frame in report["frames"] if frame["state"] == "ok")
        print(f"  frames: {report['frames_processed']} processed, {ok} with a valid pose, "
              f"{report['frames_processed'] - ok} unknown")
    for item in report["placeholder_values_in_use"]:
        print(f"  PLACEHOLDER still in use: {item}")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = run(args)
    except (RuntimeError, OSError, ValueError) as error:
        print(f"arena_overlay: {error}", file=sys.stderr)
        return 2
    print_summary(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
