"""Arena floor mapping and ArUco bot-pose maths for the read-only camera overlay tool.

Pure numpy/OpenCV: no Qt, no serial port, no robot link, no motor commands. Everything here
works on points and frames that are handed in, so it runs headless in the offline tests.

Reused from the existing backend: ``TagDetection`` and ``AprilTagTracker.detect`` from
``control_center.backend.vision`` (marker detection and corner extraction).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
import yaml

from control_center.backend.marker_settings import (LIVE_PARAMETER_SET, MARKER_DICTIONARY, build_detector,  # noqa: F401 - dictionary_id re-exported
                                                    dictionary_id)
from control_center.backend.vision import AprilTagTracker, TagDetection

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "config" / "arena_overlay.yaml"
DEFAULT_CLICKS_PATH = REPO_ROOT / "config" / "arena_overlay_clicks.json"
# Calibration profiles. "live" is the user's own camera (DEFAULT_CLICKS_PATH). Any other profile
# lives in its own folder and says what it was made from; a profile made from someone else's
# footage ("historical") is never loaded for a live camera.
PROFILES_DIR = REPO_ROOT / "config" / "overlay_profiles"
LIVE_PROFILE = "live"

UNKNOWN = "unknown"


# --------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------

@dataclass(frozen=True)
class ReferencePoint:
    name: str
    arena_xy: tuple[float, float]
    role: str            # "fit" or "check"
    measured: bool
    note: str = ""


@dataclass(frozen=True)
class MarkerSpec:
    marker_id: int
    face: str            # "top", "bottom" or "rear" (vertical rear face; chassis_tracker only)
    forward_m: float
    left_m: float
    yaw_deg: float
    size_m: float
    height_m: float
    height_inverted_m: float | None = None   # rear markers only: centre height when the chassis is inverted

    def model_corners(self) -> np.ndarray:
        """Marker corners in the bot frame (forward, left), in ArUco order TL, TR, BR, BL."""
        psi = math.radians(self.yaw_deg)
        up = np.array([math.cos(psi), math.sin(psi)])
        right = np.array([math.sin(psi), -math.cos(psi)])   # `up` turned 90 deg clockwise
        centre = np.array([self.forward_m, self.left_m])
        half = self.size_m / 2.0
        return np.array([centre + half * (up - right), centre + half * (up + right),
                         centre + half * (-up + right), centre + half * (-up - right)])


@dataclass
class OverlayConfig:
    path: Path | None
    arena: dict[str, Any]
    references: list[ReferencePoint]
    dictionary: str
    markers: dict[int, MarkerSpec]
    camera_position_m: tuple[float, float, float] | None
    camera_matrix: np.ndarray | None
    distortion: np.ndarray | None
    height_correction: str
    max_position_disagreement_m: float
    max_heading_disagreement_deg: float
    max_marker_size_error_fraction: float | None
    max_detection_age_s: float
    warn_check_error_mm: float
    max_fit_residual_mm: float
    weak_predicted_error_mm: float = 25.0
    min_hull_fraction: float = 0.25
    placeholders: list[str] = field(default_factory=list)
    placeholder_tags: list[str] = field(default_factory=list)      # short names for the HUD

    def reference(self, name: str) -> ReferencePoint:
        return next(ref for ref in self.references if ref.name == name)


def parse_config(data: dict[str, Any], path: Path | None = None) -> OverlayConfig:
    arena = dict(data.get("arena") or {})
    for key in ("width_m", "height_m", "pit_radius_m", "pit_centres_m", "flipper_vertices_m"):
        if key not in arena:
            raise ValueError(f"arena.{key} is missing from the overlay config")
    references: list[ReferencePoint] = []
    for item in data.get("reference_points") or []:
        role = str(item.get("role", "fit")).lower()
        if role not in ("fit", "check"):
            raise ValueError(f"reference point {item.get('name')!r}: role must be fit or check")
        xy = item["arena_m"]
        references.append(ReferencePoint(str(item["name"]), (float(xy[0]), float(xy[1])), role,
                                         bool(item.get("measured", False)),
                                         str(item.get("note", ""))))
    names = [ref.name for ref in references]
    if len(set(names)) != len(names):
        raise ValueError("reference point names must be unique")

    bot = data.get("bot") or {}
    dictionary = str(bot.get("dictionary", MARKER_DICTIONARY))
    dictionary_id(dictionary)
    default_size = float(bot.get("marker_size_m", 0.04))
    heights = bot.get("marker_height_m") or {}
    markers: dict[int, MarkerSpec] = {}
    for item in bot.get("markers") or []:
        marker_id = int(item["id"])
        face = str(item["face"]).lower()
        if face not in ("top", "bottom", "rear"):
            raise ValueError(f"marker {marker_id}: face must be top or bottom (or rear)")
        if marker_id in markers:
            raise ValueError(f"marker id {marker_id} is configured twice")
        height = item.get("height_m", heights.get(face))
        if height is None:
            raise ValueError(f"marker {marker_id}: no height (bot.marker_height_m.{face})")
        markers[marker_id] = MarkerSpec(marker_id, face, float(item.get("forward_m", 0.0)),
                                        float(item.get("left_m", 0.0)),
                                        float(item.get("yaw_deg", 0.0)),
                                        float(item.get("size_m", default_size)), float(height),
                                        (None if face != "rear" or item.get("height_inverted_m") is None
                                         else float(item["height_inverted_m"])))

    camera = data.get("camera") or {}
    position = camera.get("position_m")
    matrix = camera.get("matrix")
    distortion = camera.get("distortion")
    pose = data.get("pose") or {}
    mode = str(pose.get("height_correction", "auto")).lower()
    if mode not in ("auto", "measured", "derived", "off"):
        raise ValueError("pose.height_correction must be auto, measured, derived or off")
    size_tolerance = pose.get("max_marker_size_error_fraction", 0.30)
    mapping = data.get("mapping") or {}

    placeholders: list[str] = []
    tags: list[str] = []
    if not arena.get("measured", False):
        placeholders.append("arena geometry (arena.*)")
        tags.append("arena")
    unmeasured = [ref.name for ref in references if not ref.measured]
    if unmeasured:
        placeholders.append(f"{len(unmeasured)} reference point position(s) (reference_points)")
        tags.append(f"{len(unmeasured)} ref points")
    if not bot.get("measured", False):
        placeholders.append("marker size/height/offsets (bot.*; the dictionary is not a placeholder)")
        tags.append("bot markers")
    if position is None:
        placeholders.append("camera position (camera.position_m)")
        tags.append("camera position")

    return OverlayConfig(
        path=path, arena=arena, references=references, dictionary=dictionary, markers=markers,
        camera_position_m=(None if position is None else tuple(float(v) for v in position)),
        camera_matrix=(None if matrix is None else np.asarray(matrix, float).reshape(3, 3)),
        distortion=(None if distortion is None else np.asarray(distortion, float).ravel()),
        height_correction=mode,
        max_position_disagreement_m=float(pose.get("max_position_disagreement_m", 0.02)),
        max_heading_disagreement_deg=float(pose.get("max_heading_disagreement_deg", 10.0)),
        max_marker_size_error_fraction=(None if size_tolerance is None else float(size_tolerance)),
        max_detection_age_s=float(pose.get("max_detection_age_s", 0.25)),
        warn_check_error_mm=float(mapping.get("warn_check_error_mm", 10.0)),
        max_fit_residual_mm=float(mapping.get("max_fit_residual_mm", 30.0)),
        weak_predicted_error_mm=float(mapping.get("weak_predicted_error_mm", 25.0)),
        min_hull_fraction=float(mapping.get("min_hull_fraction", 0.25)),
        placeholders=placeholders, placeholder_tags=tags)


def load_config(path: Path | str | None = None) -> OverlayConfig:
    config_path = Path(path) if path else DEFAULT_CONFIG_PATH
    data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    return parse_config(data, config_path)


# --------------------------------------------------------------------------------------
# Fixed arena geometry (floor level): outer boundary, pit rims, fixed flipper boundary
# --------------------------------------------------------------------------------------

def arena_outline(arena: dict[str, Any]) -> np.ndarray:
    w, h = float(arena["width_m"]), float(arena["height_m"])
    return np.array([[0.0, 0.0], [w, 0.0], [w, h], [0.0, h], [0.0, 0.0]])


def pit_radii(arena: dict[str, Any]) -> list[float]:
    """One radius per pit: ``pit_radii_m`` when the arena set-up supplies them, else the single ``pit_radius_m``."""
    centres = arena["pit_centres_m"]
    listed = arena.get("pit_radii_m")
    if listed is not None and len(listed) == len(centres):
        return [float(r) for r in listed]
    return [float(arena["pit_radius_m"])] * len(centres)


def pit_arcs(arena: dict[str, Any], samples: int = 720) -> list[np.ndarray]:
    """The part of each pit circle that lies inside the arena, as an open polyline."""
    w, h = float(arena["width_m"]), float(arena["height_m"])
    radii = pit_radii(arena)
    eps = 1e-9
    arcs: list[np.ndarray] = []
    for (cx, cy), radius in zip(arena["pit_centres_m"], radii):
        angles = np.linspace(0.0, 2.0 * math.pi, samples, endpoint=False)
        points = np.column_stack([cx + radius * np.cos(angles), cy + radius * np.sin(angles)])
        inside = ((points[:, 0] >= -eps) & (points[:, 0] <= w + eps) &
                  (points[:, 1] >= -eps) & (points[:, 1] <= h + eps))
        if inside.all():
            arcs.append(np.vstack([points, points[:1]]))
            continue
        if not inside.any():
            continue
        start = int(np.argmin(inside))            # an outside sample, so runs do not wrap
        order = np.roll(np.arange(samples), -start)
        run: list[np.ndarray] = []
        for index in order:
            if inside[index]:
                run.append(points[index])
            elif run:
                arcs.append(np.array(run))
                run = []
        if run:
            arcs.append(np.array(run))
    return arcs


def flipper_boundary(arena: dict[str, Any]) -> np.ndarray:
    vertices = np.asarray(arena["flipper_vertices_m"], float).reshape(-1, 2)
    return np.vstack([vertices, vertices[:1]])


def densify(polyline: np.ndarray, step_m: float = 0.02) -> np.ndarray:
    """Insert points along straight segments so lens distortion can bend the drawn line."""
    points = [polyline[0]]
    for a, b in zip(polyline[:-1], polyline[1:]):
        count = max(1, int(math.ceil(float(np.linalg.norm(b - a)) / step_m)))
        points.extend(a + (b - a) * (i / count) for i in range(1, count + 1))
    return np.asarray(points)


# --------------------------------------------------------------------------------------
# Pixel model and image-to-floor mapping
# --------------------------------------------------------------------------------------

def _points(values: Any) -> np.ndarray:
    return np.asarray(values, dtype=float).reshape(-1, 2)


def apply_homography(H: np.ndarray, points: Any) -> np.ndarray:
    pts = _points(points)
    mapped = np.column_stack([pts, np.ones(len(pts))]) @ H.T
    return mapped[:, :2] / mapped[:, 2:3]


class PixelModel:
    """Optional lens intrinsics. Without them pixels are used as an ideal pinhole image.

    ``model`` is "pinhole" (OpenCV's standard distortion coefficients) or "fisheye" (OpenCV's
    equidistant fisheye model, for action-camera footage; zero coefficients are still a lens).
    """

    def __init__(self, matrix: Any = None, distortion: Any = None, model: str = "pinhole") -> None:
        if model not in ("pinhole", "fisheye"):
            raise ValueError("lens model must be pinhole or fisheye")
        self.model = model
        self.matrix = None if matrix is None else np.asarray(matrix, float).reshape(3, 3)
        self.distortion = None if distortion is None else np.asarray(distortion, float).ravel()
        if model == "fisheye" and self.matrix is not None:
            padded = np.zeros(4)
            if self.distortion is not None:
                padded[:min(4, len(self.distortion))] = self.distortion[:4]
            self.distortion = padded

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "PixelModel":
        if not data or data.get("matrix") is None:
            return cls()
        return cls(data["matrix"], data.get("distortion"), str(data.get("model", "pinhole")))

    def to_dict(self) -> dict[str, Any] | None:
        if self.matrix is None:
            return None
        return {"model": self.model, "matrix": self.matrix.tolist(),
                "distortion": None if self.distortion is None else self.distortion.tolist()}

    @property
    def active(self) -> bool:
        if self.matrix is None:
            return False
        if self.model == "fisheye":
            return True
        return self.distortion is not None and bool(np.any(self.distortion != 0))

    def undistort(self, points: Any) -> np.ndarray:
        pts = _points(points)
        if not self.active:
            return pts.copy()
        if self.model == "fisheye":
            out = cv2.fisheye.undistortPoints(pts.reshape(-1, 1, 2), self.matrix, self.distortion,
                                              P=self.matrix)
            return out.reshape(-1, 2)
        out = cv2.undistortPoints(pts.reshape(-1, 1, 2), self.matrix, self.distortion,
                                  P=self.matrix)
        return out.reshape(-1, 2)

    def distort(self, points: Any) -> np.ndarray:
        pts = _points(points)
        if not self.active:
            return pts.copy()
        fx, fy = self.matrix[0, 0], self.matrix[1, 1]
        cx, cy = self.matrix[0, 2], self.matrix[1, 2]
        if self.model == "fisheye":
            rays = np.column_stack([(pts[:, 0] - cx) / fx, (pts[:, 1] - cy) / fy])
            out = cv2.fisheye.distortPoints(rays.reshape(-1, 1, 2), self.matrix, self.distortion)
            return out.reshape(-1, 2)
        rays = np.column_stack([(pts[:, 0] - cx) / fx, (pts[:, 1] - cy) / fy, np.ones(len(pts))])
        out, _ = cv2.projectPoints(rays.reshape(-1, 1, 3), np.zeros(3), np.zeros(3),
                                   self.matrix, self.distortion)
        return out.reshape(-1, 2)


@dataclass
class PointError:
    name: str
    role: str
    arena_xy: tuple[float, float]
    pixel: tuple[float, float]
    mapped_xy: tuple[float, float]
    error_mm: float
    error_px: float


@dataclass
class FloorMapping:
    valid: bool = False
    reason: str = "no reference points clicked"
    image_size: tuple[int, int] | None = None
    H: np.ndarray | None = None          # ideal pixel -> arena floor metres
    H_inv: np.ndarray | None = None      # arena floor metres -> ideal pixel
    pixel_model: PixelModel = field(default_factory=PixelModel)
    errors: list[PointError] = field(default_factory=list)

    def pixel_to_floor(self, points: Any) -> np.ndarray:
        return apply_homography(self.H, self.pixel_model.undistort(points))

    def floor_to_pixel(self, points: Any) -> np.ndarray:
        return self.pixel_model.distort(apply_homography(self.H_inv, points))

    def _stats(self, role: str) -> dict[str, Any]:
        values = [e.error_mm for e in self.errors if e.role == role]
        if not values:
            return {"count": 0, "max_mm": None, "mean_mm": None, "rms_mm": None}
        array = np.asarray(values)
        return {"count": len(values), "max_mm": float(array.max()),
                "mean_mm": float(array.mean()), "rms_mm": float(np.sqrt(np.mean(array ** 2)))}

    @property
    def fit_stats(self) -> dict[str, Any]:
        return self._stats("fit")

    @property
    def check_stats(self) -> dict[str, Any]:
        return self._stats("check")

    def to_report(self) -> dict[str, Any]:
        check = self.check_stats
        return {
            "valid": self.valid, "reason": self.reason,
            "image_size": list(self.image_size) if self.image_size else None,
            "lens_intrinsics_used": self.pixel_model.active,
            "lens_model": self.pixel_model.model if self.pixel_model.active else None,
            "fit_points": self.fit_stats, "check_points": check,
            "held_out_check_available": bool(self.valid and check["count"]),
            "note": ("fit-point residuals are not accuracy evidence (4 points always fit "
                     "exactly); only check-point errors are held out"),
            "points": [{"name": e.name, "role": e.role, "arena_m": list(e.arena_xy),
                        "pixel": list(e.pixel), "mapped_m": list(e.mapped_xy),
                        "error_mm": e.error_mm, "error_px": e.error_px} for e in self.errors],
            "homography_pixel_to_arena_m": None if self.H is None else self.H.tolist(),
        }


def _dlt_degenerate(src: np.ndarray, dst: np.ndarray) -> bool:
    """True when the point layout cannot determine a homography (e.g. collinear points)."""
    def normalise(points: np.ndarray) -> np.ndarray:
        centred = points - points.mean(axis=0)
        scale = math.sqrt(float(np.mean(np.sum(centred ** 2, axis=1)))) or 1.0
        return centred / scale
    s, d = normalise(src), normalise(dst)
    rows = []
    for (x, y), (u, v) in zip(s, d):
        rows.append([-x, -y, -1, 0, 0, 0, u * x, u * y, u])
        rows.append([0, 0, 0, -x, -y, -1, v * x, v * y, v])
    singular = np.linalg.svd(np.asarray(rows), compute_uv=False)
    return len(singular) < 8 or singular[7] / singular[0] < 1e-6


def fit_floor_mapping(references: Iterable[ReferencePoint], clicks: dict[str, Any],
                      image_size: tuple[int, int], pixel_model: PixelModel | None = None,
                      roles: dict[str, str] | None = None,
                      clicks_image_size: tuple[int, int] | None = None,
                      max_fit_residual_mm: float = 30.0) -> FloorMapping:
    """Fit pixel -> floor from the clicked fit points; measure error at the check points."""
    model = pixel_model or PixelModel()
    roles = roles or {}
    mapping = FloorMapping(image_size=tuple(image_size), pixel_model=model)
    if clicks_image_size is not None and tuple(clicks_image_size) != tuple(image_size):
        mapping.reason = (f"reference points were clicked on a {clicks_image_size[0]}x"
                          f"{clicks_image_size[1]} image but this frame is {image_size[0]}x"
                          f"{image_size[1]}; click them again at this resolution")
        return mapping
    used = [(ref, roles.get(ref.name, ref.role), tuple(float(v) for v in clicks[ref.name]))
            for ref in references if ref.name in clicks]
    fit = [(ref, pixel) for ref, role, pixel in used if role == "fit"]
    if len(fit) < 4:
        mapping.reason = f"need at least 4 clicked fit points, have {len(fit)}"
        return mapping
    src = model.undistort([pixel for _, pixel in fit])
    dst = np.array([ref.arena_xy for ref, _ in fit], float)
    if _dlt_degenerate(src, dst):
        mapping.reason = "fit points are degenerate (collinear or repeated); pick spread-out points"
        return mapping
    H, _ = cv2.findHomography(src, dst, 0)
    if H is None or not np.isfinite(H).all() or abs(np.linalg.det(H)) < 1e-15:
        mapping.reason = "homography calculation failed"
        return mapping
    mapping.H, mapping.H_inv = H, np.linalg.inv(H)
    for ref, role, pixel in used:
        mapped = mapping.pixel_to_floor([pixel])[0]
        back = mapping.floor_to_pixel([ref.arena_xy])[0]
        mapping.errors.append(PointError(
            ref.name, role, ref.arena_xy, pixel, (float(mapped[0]), float(mapped[1])),
            1000.0 * float(np.linalg.norm(mapped - np.asarray(ref.arena_xy))),
            float(np.linalg.norm(back - np.asarray(pixel)))))
    fit_rms = mapping.fit_stats["rms_mm"]
    if fit_rms is None or fit_rms > max_fit_residual_mm:
        mapping.reason = (f"fit residual {fit_rms:.1f} mm exceeds {max_fit_residual_mm:.1f} mm; "
                          "a point is mis-clicked or its arena_m value is wrong")
        return mapping
    # A camera looking down at the floor sees +x -> +y as clockwise in the image (v is down).
    centre = dst.mean(axis=0)
    p0, px, py = apply_homography(mapping.H_inv, [centre, centre + [0.01, 0.0],
                                                  centre + [0.0, 0.01]])
    dx, dy = px - p0, py - p0
    if float(dx[0] * dy[1] - dx[1] * dy[0]) >= 0.0:
        mapping.reason = ("mapping is mirrored: the clicked points do not match the arena frame "
                          "(x and y swapped, or a mirrored camera image)")
        return mapping
    mapping.valid, mapping.reason = True, "ok"
    return mapping


# --------------------------------------------------------------------------------------
# How far to trust a fit: point count, spread, and what a one-pixel click error would do
# --------------------------------------------------------------------------------------

@dataclass
class FitQuality:
    level: str = "insufficient"          # "ok", "weak" or "insufficient"
    reasons: list[str] = field(default_factory=list)
    fit_points: int = 0
    check_points: int = 0
    hull_fraction: float | None = None   # share of the arena inside the fit points' hull
    predicted_worst_mm: float | None = None      # map shift for a 1 px click error, worst place
    predicted_worst_at_m: tuple[float, float] | None = None
    predicted_centre_mm: float | None = None

    def to_report(self) -> dict[str, Any]:
        return {"level": self.level, "reasons": self.reasons, "fit_points": self.fit_points,
                "check_points": self.check_points, "fit_hull_share_of_arena": self.hull_fraction,
                "predicted_shift_for_1px_click_error_mm": {
                    "worst": self.predicted_worst_mm, "worst_at_m": self.predicted_worst_at_m,
                    "centre": self.predicted_centre_mm},
                "note": "predicted shift is a sensitivity estimate, not a measured accuracy"}


def assess_fit(references: Iterable[ReferencePoint], clicks: dict[str, Any],
               roles: dict[str, str] | None, mapping: FloorMapping, arena: dict[str, Any],
               warn_check_error_mm: float = 10.0, weak_predicted_error_mm: float = 25.0,
               min_hull_fraction: float = 0.25, noise_px: float = 1.0, samples: int = 80,
               seed: int = 0) -> FitQuality:
    """Grade a fit. Weak selections are shown as weak; they are never silently accepted."""
    roles = roles or {}
    used = [(ref, roles.get(ref.name, ref.role)) for ref in references if ref.name in clicks]
    fit = [ref for ref, role in used if role == "fit"]
    quality = FitQuality(fit_points=len(fit), check_points=len(used) - len(fit))
    if not mapping.valid:
        quality.reasons = [mapping.reason]
        return quality
    reasons: list[str] = []
    w, h = float(arena["width_m"]), float(arena["height_m"])
    hull = cv2.convexHull(np.array([ref.arena_xy for ref in fit], np.float32))
    quality.hull_fraction = float(cv2.contourArea(hull) / (w * h))
    if quality.hull_fraction < min_hull_fraction:
        reasons.append(f"fit points span only {100 * quality.hull_fraction:.0f}% of the arena; "
                       "add a point on the opposite side")
    # Sensitivity: jitter the fit clicks by `noise_px` and see how far the map moves on a grid.
    grid = np.array([[x, y] for x in np.linspace(0.05 * w, 0.95 * w, 5)
                     for y in np.linspace(0.05 * h, 0.95 * h, 5)])
    grid_pixels = mapping.floor_to_pixel(grid)
    usable = np.isfinite(grid_pixels).all(axis=1)
    src = np.array([clicks[ref.name] for ref in fit], float)
    dst = np.array([ref.arena_xy for ref in fit], float)
    rng = np.random.default_rng(seed)
    squared = np.zeros(len(grid))
    count = 0
    for _ in range(samples):
        noisy = mapping.pixel_model.undistort(src + rng.normal(0.0, noise_px, src.shape))
        H, _ = cv2.findHomography(noisy, dst, 0)
        if H is None or not np.isfinite(H).all():
            continue
        moved = apply_homography(H, mapping.pixel_model.undistort(grid_pixels[usable]))
        squared[usable] += np.sum((moved - grid[usable]) ** 2, axis=1)
        count += 1
    if count and usable.any():
        rms = 1000.0 * np.sqrt(squared / count)
        rms[~usable] = np.nan
        worst = int(np.nanargmax(rms))
        quality.predicted_worst_mm = float(rms[worst])
        quality.predicted_worst_at_m = (float(grid[worst, 0]), float(grid[worst, 1]))
        quality.predicted_centre_mm = float(rms[len(grid) // 2])
        if quality.predicted_worst_mm > weak_predicted_error_mm:
            x, y = quality.predicted_worst_at_m
            reasons.append(f"a {noise_px:g} px click error moves the map by about "
                           f"{quality.predicted_worst_mm:.0f} mm near ({x:.2f}, {y:.2f}) m; "
                           "add or spread fit points toward that side")
    check = mapping.check_stats
    if not check["count"]:
        reasons.append("no check point clicked: nothing independent confirms the fit"
                       + (" (4 fit points always fit exactly)" if len(fit) == 4 else ""))
    elif check["max_mm"] > warn_check_error_mm:
        worst_check = max((e for e in mapping.errors if e.role == "check"),
                          key=lambda e: e.error_mm)
        reasons.append(f"check point {worst_check.name} is off by {worst_check.error_mm:.0f} mm")
    quality.reasons = reasons
    quality.level = "weak" if reasons else "ok"
    return quality


def topdown_view(frame: np.ndarray, mapping: FloorMapping, arena: dict[str, Any],
                 pixels_per_m: float = 400.0, margin_m: float = 0.08) -> np.ndarray:
    """The frame resampled onto the floor plane, +y up, with a margin around the arena."""
    w, h = float(arena["width_m"]), float(arena["height_m"])
    cols = int(round((w + 2 * margin_m) * pixels_per_m))
    rows = int(round((h + 2 * margin_m) * pixels_per_m))
    xs = (np.arange(cols) + 0.5) / pixels_per_m - margin_m
    ys = h + margin_m - (np.arange(rows) + 0.5) / pixels_per_m
    gx, gy = np.meshgrid(xs, ys)
    pixels = mapping.floor_to_pixel(np.column_stack([gx.ravel(), gy.ravel()]))
    pixels = np.where(np.isfinite(pixels), pixels, -1.0).astype(np.float32)
    return cv2.remap(frame, pixels[:, 0].reshape(rows, cols), pixels[:, 1].reshape(rows, cols),
                     cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(40, 40, 40))


def topdown_pixel(points_xy: Any, arena: dict[str, Any], pixels_per_m: float = 400.0,
                  margin_m: float = 0.08) -> np.ndarray:
    """Arena metres -> pixel coordinates in a ``topdown_view`` image."""
    pts = _points(points_xy)
    return np.column_stack([(pts[:, 0] + margin_m) * pixels_per_m,
                            (float(arena["height_m"]) + margin_m - pts[:, 1]) * pixels_per_m])


# --------------------------------------------------------------------------------------
# Camera position (needed to correct for the raised marker)
# --------------------------------------------------------------------------------------

@dataclass
class CameraEstimate:
    position_m: tuple[float, float, float] | None = None
    source: str = "none"                 # "measured", "derived" or "none"
    focal_px: float | None = None
    detail: str = "camera position not available"

    def to_report(self) -> dict[str, Any]:
        return {"position_m": list(self.position_m) if self.position_m else None,
                "source": self.source, "focal_px": self.focal_px, "detail": self.detail}


def derive_camera_from_homography(H_floor_to_pixel: np.ndarray, image_size: tuple[int, int],
                                  camera_matrix: np.ndarray | None = None) -> CameraEstimate:
    """Recover the camera centre from the floor homography of an ideal pinhole camera.

    Without a camera matrix this assumes square pixels, zero skew and the principal point at
    the image centre, and solves the focal length from the orthonormality of the first two
    rotation columns. With a camera matrix it only solves the pose.
    """
    H = np.asarray(H_floor_to_pixel, float)
    H = H / np.linalg.norm(H)
    if camera_matrix is not None:
        K = np.asarray(camera_matrix, float).reshape(3, 3)
        focal = float((K[0, 0] + K[1, 1]) / 2.0)
        basis = "configured camera matrix"
    else:
        cx, cy = image_size[0] / 2.0, image_size[1] / 2.0
        a = np.array([H[0, 0] - cx * H[2, 0], H[1, 0] - cy * H[2, 0], H[2, 0]])
        b = np.array([H[0, 1] - cx * H[2, 1], H[1, 1] - cy * H[2, 1], H[2, 1]])
        p1, q1 = a[0] * b[0] + a[1] * b[1], a[2] * b[2]
        p2 = (a[0] ** 2 + a[1] ** 2) - (b[0] ** 2 + b[1] ** 2)
        q2 = a[2] ** 2 - b[2] ** 2
        denominator = p1 * p1 + p2 * p2
        if denominator <= 0.0:
            return CameraEstimate(detail="camera derivation failed: degenerate homography")
        inverse_f2 = -(p1 * q1 + p2 * q2) / denominator
        if not np.isfinite(inverse_f2) or inverse_f2 <= 0.0:
            return CameraEstimate(detail="camera derivation failed: no positive focal length "
                                         "(view too close to straight-down, or bad mapping)")
        focal = float(1.0 / math.sqrt(inverse_f2))
        if not 0.2 * image_size[0] <= focal <= 20.0 * image_size[0]:
            return CameraEstimate(detail=f"camera derivation failed: implausible focal length "
                                         f"{focal:.0f} px")
        K = np.array([[focal, 0.0, cx], [0.0, focal, cy], [0.0, 0.0, 1.0]])
        basis = "pinhole assumption (principal point at image centre, no lens distortion)"
    M = np.linalg.inv(K) @ H
    scale = 2.0 / (np.linalg.norm(M[:, 0]) + np.linalg.norm(M[:, 1]))
    r1, r2, t = M[:, 0] * scale, M[:, 1] * scale, M[:, 2] * scale
    U, _, Vt = np.linalg.svd(np.column_stack([r1, r2, np.cross(r1, r2)]))
    R = U @ Vt
    centre = -R.T @ t
    centre[2] = abs(centre[2])               # the two sign choices mirror the camera in the floor
    if not np.isfinite(centre).all() or not 0.15 <= centre[2] <= 8.0:
        return CameraEstimate(detail=f"camera derivation failed: implausible camera height "
                                     f"{centre[2]:.2f} m")
    return CameraEstimate(tuple(float(v) for v in centre), "derived", focal,
                          f"derived from the floor homography using {basis}")


def resolve_camera(config: OverlayConfig, mapping: FloorMapping) -> CameraEstimate:
    """Pick the camera position used for height correction, according to the config mode."""
    mode = config.height_correction
    if mode == "off":
        return CameraEstimate(detail="height correction disabled (pose.height_correction: off)")
    if mode in ("auto", "measured") and config.camera_position_m is not None:
        return CameraEstimate(config.camera_position_m, "measured", None,
                              "measured lens position from config camera.position_m")
    if mode == "measured":
        return CameraEstimate(detail="pose.height_correction is 'measured' but "
                                     "camera.position_m is not set")
    if not mapping.valid or mapping.image_size is None:
        return CameraEstimate(detail="camera position cannot be derived without a valid mapping")
    # With a lens model the mapping works on ideal pixels of that lens's own camera matrix.
    matrix = mapping.pixel_model.matrix if mapping.pixel_model.active else config.camera_matrix
    return derive_camera_from_homography(mapping.H_inv, mapping.image_size, matrix)


def lift_to_height(floor_xy: Any, height_m: float, camera_xyz: Iterable[float]) -> np.ndarray:
    """Where the ray camera -> floor point crosses the horizontal plane at ``height_m``."""
    cam = np.asarray(tuple(camera_xyz), float)
    pts = _points(floor_xy)
    return pts + (height_m / cam[2]) * (cam[:2] - pts)


# --------------------------------------------------------------------------------------
# Marker detection (reuses the existing backend detector code path)
# --------------------------------------------------------------------------------------

class MarkerDetector(AprilTagTracker):
    """``AprilTagTracker`` with any cv2.aruco dictionary and sub-pixel corner refinement.

    The existing tracker only maps AprilTag family names and silently falls back to 36h11, so
    only the constructor is replaced here; ``detect`` is the existing code.
    """

    def __init__(self, dictionary: str, parameter_set: str = LIVE_PARAMETER_SET) -> None:  # noqa: D107 - intentionally not calling super
        self._detector = build_detector(dictionary, parameter_set)         # the repository's one named parameter set
        self.dictionary = dictionary
        self.parameter_set = parameter_set
        self.self_tag_id = -1
        self.tag_size_m = None
        self.camera_matrix = None
        self.distortion = None

    def detect_markers(self, frame_bgr: np.ndarray) -> list[TagDetection]:
        return self.detect(frame_bgr)[0]


# --------------------------------------------------------------------------------------
# Bot pose from markers
# --------------------------------------------------------------------------------------

@dataclass
class BotPose:
    state: str = UNKNOWN                 # "ok" or "unknown"
    reason: str = "no detection yet"
    x_m: float | None = None             # bot centre on the floor
    y_m: float | None = None
    heading_rad: float | None = None     # nose direction, CCW from arena +x
    orientation: str = UNKNOWN           # "upright", "inverted" or "unknown"
    position_quality: str = UNKNOWN      # "height-corrected", "approximate" or "unknown"
    correction_source: str = "none"      # "measured", "derived" or "none"
    position_label: str = UNKNOWN
    floor_projection_xy: tuple[float, float] | None = None   # raised markers treated as floor
    parallax_shift_mm: float | None = None
    visible_ids: list[int] = field(default_factory=list)
    used_ids: list[int] = field(default_factory=list)
    ignored_ids: list[int] = field(default_factory=list)
    position_spread_mm: float | None = None
    heading_spread_deg: float | None = None
    marker_fit_rms_mm: float | None = None
    markers: list[dict[str, Any]] = field(default_factory=list)
    rear_ids: list[int] = field(default_factory=list)   # visible rear-face markers (not used here)

    @property
    def ok(self) -> bool:
        return self.state == "ok"

    @property
    def heading_deg(self) -> float | None:
        return None if self.heading_rad is None else math.degrees(self.heading_rad)

    def to_report(self) -> dict[str, Any]:
        return {
            "state": self.state, "reason": self.reason,
            "x_m": self.x_m if self.ok else UNKNOWN, "y_m": self.y_m if self.ok else UNKNOWN,
            "heading_rad": self.heading_rad if self.ok else UNKNOWN,
            "heading_deg": self.heading_deg if self.ok else UNKNOWN,
            "orientation": self.orientation,
            "position_quality": self.position_quality,
            "position_label": self.position_label,
            "correction_source": self.correction_source,
            "floor_projection_xy_m": (list(self.floor_projection_xy)
                                      if self.floor_projection_xy else None),
            "parallax_shift_mm": self.parallax_shift_mm,
            "visible_ids": self.visible_ids, "used_ids": self.used_ids,
            "ignored_ids": self.ignored_ids,
            "position_spread_mm": self.position_spread_mm,
            "heading_spread_deg": self.heading_spread_deg,
            "marker_fit_rms_mm": self.marker_fit_rms_mm, "markers": self.markers,
        }


def rigid_fit_2d(model: np.ndarray, observed: np.ndarray) -> tuple[float, np.ndarray, float]:
    """Rotation + translation (no scale) taking bot-frame points onto arena points."""
    model_centre, observed_centre = model.mean(axis=0), observed.mean(axis=0)
    m, o = model - model_centre, observed - observed_centre
    theta = math.atan2(float(np.sum(m[:, 0] * o[:, 1] - m[:, 1] * o[:, 0])),
                       float(np.sum(m[:, 0] * o[:, 0] + m[:, 1] * o[:, 1])))
    c, s = math.cos(theta), math.sin(theta)
    R = np.array([[c, -s], [s, c]])
    translation = observed_centre - R @ model_centre
    residual = observed - (model @ R.T + translation)
    return theta, translation, float(np.sqrt(np.mean(np.sum(residual ** 2, axis=1))))


def angle_difference(a: float, b: float) -> float:
    return (a - b + math.pi) % (2.0 * math.pi) - math.pi


POSITION_LABELS = {
    "measured": "height-corrected (camera position measured: camera.position_m)",
    "derived": ("height-corrected (camera position DERIVED from the floor mapping, "
                "pinhole assumption; verify at the arena)"),
    "none": "APPROXIMATE: raised marker projected to the floor, no height correction",
}


def estimate_bot_pose(detections: list[TagDetection], mapping: FloorMapping,
                      camera: CameraEstimate, config: OverlayConfig) -> BotPose:
    """Fuse the visible configured markers into one bot pose, or say ``unknown`` and why."""
    # Rear-face markers are vertical: this single-frame estimator leaves them to chassis_tracker.
    rear = sorted(d.tag_id for d in detections
                  if d.tag_id in config.markers and config.markers[d.tag_id].face == "rear")
    known = [d for d in detections
             if d.tag_id in config.markers and config.markers[d.tag_id].face != "rear"]
    pose = BotPose(visible_ids=sorted(d.tag_id for d in known),
                   ignored_ids=sorted({d.tag_id for d in detections} - set(config.markers)),
                   rear_ids=rear)
    if not known:
        pose.reason = "no configured bot markers visible"
        if rear:
            pose.reason = (f"only rear marker(s) {rear} visible: not used by the single-frame "
                           "estimator (see chassis_tracker)")
        return pose
    counts: dict[int, int] = {}
    for detection in known:
        counts[detection.tag_id] = counts.get(detection.tag_id, 0) + 1
    repeated = sorted(marker_id for marker_id, count in counts.items() if count > 1)
    if repeated:
        pose.reason = f"conflict: marker id(s) {repeated} detected more than once"
        return pose
    faces = {face: sorted(d.tag_id for d in known if config.markers[d.tag_id].face == face)
             for face in ("top", "bottom")}
    if faces["top"] and faces["bottom"]:
        pose.reason = (f"conflict: top markers {faces['top']} and bottom markers "
                       f"{faces['bottom']} visible together")
        return pose
    orientation = "upright" if faces["top"] else "inverted"
    if not mapping.valid:
        pose.orientation = orientation          # the ID set alone tells which face is up
        pose.reason = f"no valid floor mapping ({mapping.reason})"
        return pose

    corrected = camera.position_m is not None
    model_all, corrected_all, floor_all, per_marker = [], [], [], []
    size_problem: str | None = None
    for detection in sorted(known, key=lambda d: d.tag_id):
        spec = config.markers[detection.tag_id]
        floor = mapping.pixel_to_floor(detection.corners)
        plane = lift_to_height(floor, spec.height_m, camera.position_m) if corrected else floor
        model = spec.model_corners()
        theta, centre, rms = rigid_fit_2d(model, plane)
        side = float(np.mean(np.linalg.norm(plane - np.roll(plane, -1, axis=0), axis=1)))
        ratio = side / spec.size_m
        tolerance = config.max_marker_size_error_fraction
        if tolerance is not None and abs(ratio - 1.0) > tolerance and size_problem is None:
            size_problem = (f"conflict: marker {spec.marker_id} measures {side * 1000:.0f} mm "
                            f"but is configured as {spec.size_m * 1000:.0f} mm "
                            "(check bot.marker_size_m, marker height and dictionary)")
        per_marker.append({"id": spec.marker_id, "face": spec.face,
                           "bot_x_m": float(centre[0]), "bot_y_m": float(centre[1]),
                           "heading_deg": math.degrees(theta), "theta": theta,
                           "apparent_size_mm": side * 1000.0, "size_ratio": ratio,
                           "corner_fit_rms_mm": rms * 1000.0,
                           "height_m": spec.height_m})
        model_all.append(model)
        corrected_all.append(plane)
        floor_all.append(floor)

    position_spread = max((math.hypot(a["bot_x_m"] - b["bot_x_m"], a["bot_y_m"] - b["bot_y_m"])
                           for a in per_marker for b in per_marker), default=0.0)
    heading_spread = max((abs(angle_difference(a["theta"], b["theta"]))
                          for a in per_marker for b in per_marker), default=0.0)
    for item in per_marker:
        del item["theta"]
    pose.markers = per_marker
    pose.position_spread_mm = position_spread * 1000.0
    pose.heading_spread_deg = math.degrees(heading_spread)
    if size_problem:
        pose.reason = size_problem
        return pose
    if position_spread > config.max_position_disagreement_m:
        pose.reason = (f"conflict: markers disagree on the bot centre by "
                       f"{position_spread * 1000:.0f} mm "
                       f"(limit {config.max_position_disagreement_m * 1000:.0f} mm)")
        return pose
    if math.degrees(heading_spread) > config.max_heading_disagreement_deg:
        pose.reason = (f"conflict: markers disagree on heading by "
                       f"{math.degrees(heading_spread):.1f} deg "
                       f"(limit {config.max_heading_disagreement_deg:.1f} deg)")
        return pose

    model_points = np.vstack(model_all)
    theta, centre, rms = rigid_fit_2d(model_points, np.vstack(corrected_all))
    _, floor_centre, _ = rigid_fit_2d(model_points, np.vstack(floor_all))
    pose.state, pose.reason = "ok", "ok"
    pose.x_m, pose.y_m, pose.heading_rad = float(centre[0]), float(centre[1]), theta
    pose.orientation = orientation
    pose.used_ids = [item["id"] for item in per_marker]
    pose.marker_fit_rms_mm = rms * 1000.0
    pose.floor_projection_xy = (float(floor_centre[0]), float(floor_centre[1]))
    pose.correction_source = camera.source if corrected else "none"
    pose.position_quality = "height-corrected" if corrected else "approximate"
    pose.position_label = POSITION_LABELS[pose.correction_source]
    if corrected:
        pose.parallax_shift_mm = 1000.0 * float(np.linalg.norm(centre - floor_centre))
    return pose


@dataclass
class PoseStatus:
    pose: BotPose                        # what to show now ("unknown" when stale or missing)
    detection_age_s: float | None        # age of the last valid detection; None if never
    last_valid: BotPose | None           # last valid pose, for reference only (not current)

    def to_report(self) -> dict[str, Any]:
        report = self.pose.to_report()
        report["detection_age_s"] = self.detection_age_s
        report["last_valid_pose"] = None
        if self.last_valid is not None and not self.pose.ok:
            report["last_valid_pose"] = {"x_m": self.last_valid.x_m, "y_m": self.last_valid.y_m,
                                         "heading_deg": self.last_valid.heading_deg,
                                         "orientation": self.last_valid.orientation,
                                         "note": "last valid detection, NOT the current pose"}
        return report


class PoseTracker:
    """Keeps the newest pose result and how old the last valid detection is.

    There is deliberately no hold or prediction: a frame without a valid, consistent detection
    reads ``unknown`` straight away, and a valid pose turns ``unknown`` once it is older than
    ``max_age_s`` (for example when the camera stops delivering frames).
    """

    def __init__(self, max_age_s: float) -> None:
        self.max_age_s = max_age_s
        self._latest = BotPose()
        self._last_valid: BotPose | None = None
        self._last_valid_time: float | None = None

    def update(self, pose: BotPose, timestamp: float) -> None:
        self._latest = pose
        if pose.ok:
            self._last_valid, self._last_valid_time = pose, timestamp

    def status(self, now: float) -> PoseStatus:
        age = None if self._last_valid_time is None else max(0.0, now - self._last_valid_time)
        pose = self._latest
        if pose.ok and age is not None and age > self.max_age_s:
            pose = BotPose(reason=(f"stale: last detection is {age * 1000:.0f} ms old "
                                   f"(limit {self.max_age_s * 1000:.0f} ms)"),
                           visible_ids=[], ignored_ids=[])
        return PoseStatus(pose, age, self._last_valid)


# --------------------------------------------------------------------------------------
# Clicked reference points: load, save and the click-through session
# --------------------------------------------------------------------------------------

@dataclass
class ClickSet:
    points: dict[str, tuple[float, float]] = field(default_factory=dict)
    roles: dict[str, str] = field(default_factory=dict)      # per-set-up fit/check overrides
    image_size: tuple[int, int] | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def synthetic(self) -> bool:
        return bool(self.meta.get("synthetic_fixture", False))

    @property
    def kind(self) -> str:
        """"live" (the user's own camera) unless the file says it came from other footage."""
        return str(self.meta.get("profile_kind", "live"))

    @property
    def historical(self) -> bool:
        return self.kind != "live"

    @property
    def lens(self) -> PixelModel:
        return PixelModel.from_dict(self.meta.get("lens"))


def profile_points_path(name: str) -> Path:
    """Where a profile keeps its clicked points. "live" is the default live-camera file."""
    if name == LIVE_PROFILE:
        return DEFAULT_CLICKS_PATH
    if not name or any(ch in name for ch in "/\\:") or name.startswith("."):
        raise ValueError(f"bad profile name {name!r}")
    return PROFILES_DIR / name / "clicks.json"


def load_clicks(path: Path | str) -> ClickSet:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    size = data.get("image_size")
    meta = {key: value for key, value in data.items()
            if key not in ("points", "roles", "image_size")}
    return ClickSet({name: (float(p[0]), float(p[1])) for name, p in data["points"].items()},
                    {name: str(role) for name, role in (data.get("roles") or {}).items()},
                    None if size is None else (int(size[0]), int(size[1])), meta)


def save_clicks(path: Path | str, clicks: ClickSet, source: str = "") -> None:
    payload = dict(clicks.meta)
    payload.update({
        "image_size": list(clicks.image_size) if clicks.image_size else None,
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "source": source or clicks.meta.get("source", ""),
        "roles": clicks.roles,
        "points": {name: [round(p[0], 2), round(p[1], 2)] for name, p in clicks.points.items()},
    })
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")


class ClickSession:
    """Steps through the configured reference points; one left click places the current one."""

    def __init__(self, references: list[ReferencePoint], clicks: ClickSet) -> None:
        self.references = references
        self.clicks = clicks
        self.index = 0
        self._history: list[str] = []
        self._seek_unclicked(0)

    @property
    def current(self) -> ReferencePoint | None:
        return self.references[self.index] if 0 <= self.index < len(self.references) else None

    def role_of(self, ref: ReferencePoint) -> str:
        return self.clicks.roles.get(ref.name, ref.role)

    def _seek_unclicked(self, start: int) -> None:
        count = len(self.references)
        for step in range(count):
            candidate = (start + step) % count
            if self.references[candidate].name not in self.clicks.points:
                self.index = candidate
                return
        self.index = count                      # everything is placed

    def click(self, u: float, v: float) -> str | None:
        ref = self.current
        if ref is None:
            return None
        self.clicks.points[ref.name] = (float(u), float(v))
        self._history.append(ref.name)
        self._seek_unclicked(self.index + 1)
        return ref.name

    def select(self, index: int) -> ReferencePoint | None:
        """Jump straight to a named landmark (to place it, or to click it again)."""
        if not self.references:
            return None
        self.index = index % len(self.references)
        return self.references[self.index]

    def skip(self) -> None:
        if self.references:
            count = len(self.references)
            self.index = (self.index + 1) % count if self.index < count else 0

    def undo(self) -> str | None:
        while self._history:
            name = self._history.pop()
            if name in self.clicks.points:
                del self.clicks.points[name]
                self.index = [ref.name for ref in self.references].index(name)
                return name
        return None

    def clear(self) -> None:
        self.clicks.points.clear()
        self._history.clear()
        self.index = 0

    def toggle_role(self) -> str | None:
        ref = self.current
        if ref is None:
            return None
        new_role = "check" if self.role_of(ref) == "fit" else "fit"
        if new_role == ref.role:
            self.clicks.roles.pop(ref.name, None)
        else:
            self.clicks.roles[ref.name] = new_role
        return new_role
