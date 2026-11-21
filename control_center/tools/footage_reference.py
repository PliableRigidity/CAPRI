"""Calibrate ONE reference frame of recorded footage against the arena drawing (offline, read-only).

    python -m control_center.tools.footage_reference --video MATCH.mp4 --profile historical_bristol_2026_03_22

This is for testing the overlay on someone else's recording. It writes a HISTORICAL profile
(config/overlay_profiles/<profile>/clicks.json) that the overlay tool refuses to use with a live
camera. Nothing here is copied into the live profile: a different camera position needs its own
fresh calibration (``arena_overlay --camera N --calibrate``).

What it does, driven by config/overlay_profiles/<profile>/source.yaml:
  1. builds a median frame around the reference time, so moving robots drop out;
  2. traces the outline of the arena floor by colour;
  3. estimates the lens (fisheye focal length) from how straight the straight floor edges become;
     this step does not use the drawing's dimensions;
  4. places each landmark where two traced edges meet (for example pit rim and wall line);
  5. fits the image-to-floor mapping from the landmarks marked ``fit`` and reports the error at
     the ``check`` landmarks, leave-one-out errors, and how far the traced edges lie from the
     drawing's outline;
  6. repeats the edge trace at other times to see whether the camera stayed put.

The drawing is a candidate layout. Without a tape-measured reference in the footage, agreement
between the drawing and the traced edges is a consistency check, not an accuracy figure.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
from typing import Any

import cv2
import numpy as np
import yaml

from control_center.tools.overlay_core import (
    PROFILES_DIR, REPO_ROOT, ClickSet, PixelModel, arena_outline, assess_fit, densify,
    fit_floor_mapping, flipper_boundary, load_config, pit_arcs, profile_points_path, save_clicks,
    topdown_pixel, topdown_view)

ARTIFACTS = REPO_ROOT / "artifacts" / "vision-footage"


# --------------------------------------------------------------------------------------
# Frames and the floor outline
# --------------------------------------------------------------------------------------

def median_frame(video: str, time_s: float, window_s: float = 24.0, step_s: float = 2.0) -> np.ndarray:
    """Per-pixel median of frames around ``time_s``: static scenery stays, moving robots go."""
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"could not open video {video}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    frames = []
    for offset in np.arange(-window_s, window_s + 1e-9, step_s):
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(round((time_s + offset) * fps))))
        ok, frame = cap.read()
        if ok and frame is not None:
            frames.append(frame)
    cap.release()
    if len(frames) < 3:
        raise RuntimeError(f"too few frames around {time_s:.0f} s in {video}")
    return np.median(np.stack(frames), axis=0).astype(np.uint8)


def floor_outline(image: np.ndarray, samples: list[list[int]], chroma_tolerance: float = 7.0) -> np.ndarray:
    """Outer contour (N x 2 pixels) of the region coloured like the floor samples."""
    lab = cv2.cvtColor(cv2.GaussianBlur(image, (3, 3), 0), cv2.COLOR_BGR2LAB).astype(np.float32)
    refs = [lab[max(0, y - 3):y + 4, max(0, x - 3):x + 4].reshape(-1, 3).mean(axis=0) for x, y in samples]
    chroma = np.min(np.stack([np.hypot(lab[..., 1] - r[1], lab[..., 2] - r[2]) for r in refs]), axis=0)
    lows, highs = min(r[0] for r in refs) - 40.0, max(r[0] for r in refs) + 25.0
    mask = ((chroma < chroma_tolerance) & (lab[..., 0] > lows) & (lab[..., 0] < highs)).astype(np.uint8) * 255
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)           # paint on the floor (blue squares, logo)
    paint = ((hsv[..., 0] > 95) & (hsv[..., 0] < 115) & (hsv[..., 1] > 120) & (hsv[..., 2] > 120))
    mask = cv2.morphologyEx(cv2.bitwise_or(mask, paint.astype(np.uint8) * 255), cv2.MORPH_OPEN,
                            np.ones((3, 3), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if count < 2:
        raise RuntimeError("no floor-coloured region found; check floor_samples_px")
    floor = (labels == 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))).astype(np.uint8) * 255
    flood = floor.copy()
    h, w = floor.shape
    scratch = np.zeros((h + 2, w + 2), np.uint8)
    for seed in ((0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1), (w // 2, 0)):
        if flood[seed[1], seed[0]] == 0:
            cv2.floodFill(flood, scratch, seed, 128)
    floor[flood == 0] = 255                                # holes (robots, glare) belong to the floor
    contours, _ = cv2.findContours(floor, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return max(contours, key=cv2.contourArea).reshape(-1, 2).astype(float)


def image_shift(reference: np.ndarray, other: np.ndarray, crop: list[int]) -> dict[str, float]:
    """Translation (px) of ``other`` relative to ``reference`` inside ``crop`` (x0, y0, x1, y1), by
    phase correlation. The crop should hold static floor paint; only its position is compared,
    its arena coordinates are never used."""
    x0, y0, x1, y1 = crop
    a = cv2.cvtColor(reference[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY).astype(np.float32)
    b = cv2.cvtColor(other[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY).astype(np.float32)
    (dx, dy), response = cv2.phaseCorrelate(a, b, cv2.createHanningWindow((x1 - x0, y1 - y0), cv2.CV_32F))
    return {"dx_px": float(dx), "dy_px": float(dy), "shift_px": float(math.hypot(dx, dy)), "response": float(response)}


def in_boxes(points: np.ndarray, boxes: list[list[float]]) -> np.ndarray:
    keep = np.zeros(len(points), bool)
    for x0, y0, x1, y1 in boxes:
        keep |= ((points[:, 0] >= x0) & (points[:, 0] <= x1) & (points[:, 1] >= y0) & (points[:, 1] <= y1))
    return points[keep]


def smooth_inliers(points: np.ndarray, along: int, tolerance_px: float = 1.0, iterations: int = 400,
                   seed: int = 0) -> np.ndarray:
    """Points consistent with one gently curved edge; drops places where the colour mask leaked."""
    if len(points) < 8:
        return points
    rng = np.random.default_rng(seed)
    t, v = points[:, along], points[:, 1 - along]
    best = None
    for _ in range(iterations):
        pick = rng.choice(len(points), 3, replace=False)
        if len(set(t[pick].tolist())) < 3:
            continue
        coefficients = np.polyfit(t[pick], v[pick], 2)
        inliers = np.abs(np.polyval(coefficients, t) - v) < tolerance_px
        if best is None or inliers.sum() > best.sum():
            best = inliers
    if best is None or best.sum() < 8:
        return points
    coefficients = np.polyfit(t[best], v[best], 2)
    return points[np.abs(np.polyval(coefficients, t) - v) < tolerance_px]


# --------------------------------------------------------------------------------------
# Lens from edge straightness; lines, conics and their intersections
# --------------------------------------------------------------------------------------

def fisheye_model(focal_px: float, image_size: tuple[int, int]) -> PixelModel:
    w, h = image_size
    return PixelModel([[focal_px, 0.0, w / 2.0], [0.0, focal_px, h / 2.0], [0.0, 0.0, 1.0]], [0, 0, 0, 0], "fisheye")


def line_residuals(points: np.ndarray) -> np.ndarray:
    centre = points.mean(axis=0)
    return (points - centre) @ np.linalg.svd(points - centre)[2][1]


def straightness_rms(edges: dict[str, np.ndarray], model: PixelModel | None) -> float:
    residuals = np.concatenate([line_residuals(pts if model is None else model.undistort(pts)) for pts in edges.values()])
    return float(np.sqrt(np.mean(residuals ** 2)))


def estimate_fisheye_focal(edges: dict[str, np.ndarray], image_size: tuple[int, int],
                           search: tuple[float, float] = (0.3, 4.0)) -> dict[str, Any]:
    """Focal length (px) of an equidistant fisheye, centred on the image, that best straightens
    edges known to be straight. Returns the fit and how sharply it is determined."""
    width = image_size[0]
    candidates = width * np.geomspace(search[0], search[1], 60)
    costs = np.array([straightness_rms(edges, fisheye_model(f, image_size)) for f in candidates])
    best = int(np.argmin(costs))
    low, high = candidates[max(0, best - 1)], candidates[min(len(candidates) - 1, best + 1)]
    for _ in range(40):                                       # golden-section refinement
        a, b = low + 0.382 * (high - low), low + 0.618 * (high - low)
        if straightness_rms(edges, fisheye_model(a, image_size)) < straightness_rms(edges, fisheye_model(b, image_size)):
            high = b
        else:
            low = a
    focal = float((low + high) / 2.0)
    best_cost = straightness_rms(edges, fisheye_model(focal, image_size))
    plausible = candidates[costs <= best_cost + 0.1]         # within 0.1 px rms of the best
    return {"model": "fisheye (equidistant), principal point at the image centre", "focal_px": focal,
            "straightness_rms_px": best_cost, "straightness_rms_px_uncorrected": straightness_rms(edges, None),
            "focal_range_within_0.1px_rms": [float(min(plausible.min(), focal)), float(max(plausible.max(), focal))],
            "at_search_limit": bool(best in (0, len(candidates) - 1))}


def fit_line(points: np.ndarray) -> np.ndarray:
    centre = points.mean(axis=0)
    normal = np.linalg.svd(points - centre)[2][1]
    return np.array([normal[0], normal[1], -float(normal @ centre)])       # a x + b y + c = 0


class Conic:
    """Robust conic through points (the image of a circular pit rim)."""

    def __init__(self, points: np.ndarray, tolerance_px: float = 0.8, iterations: int = 600, seed: int = 1) -> None:
        self.centre, self.scale = points.mean(axis=0), float(points.std()) or 1.0
        q = (points - self.centre) / self.scale
        design = np.column_stack([q[:, 0] ** 2, q[:, 0] * q[:, 1], q[:, 1] ** 2, q[:, 0], q[:, 1], np.ones(len(q))])
        rng = np.random.default_rng(seed)
        best = None
        for _ in range(iterations):
            trial = np.linalg.svd(design[rng.choice(len(q), 5, replace=False)])[2][-1]
            inliers = self._distance(trial, q) < tolerance_px
            if best is None or inliers.sum() > best.sum():
                best = inliers
        self.coefficients = np.linalg.svd(design[best])[2][-1]
        self.inliers = self._distance(self.coefficients, q) < tolerance_px
        if self.inliers.sum() >= 5:
            self.coefficients = np.linalg.svd(design[self.inliers])[2][-1]
        distance = self._distance(self.coefficients, q)
        self.rms_px = float(np.sqrt(np.mean(distance[self.inliers] ** 2))) if self.inliers.any() else float("nan")

    def _distance(self, c: np.ndarray, q: np.ndarray) -> np.ndarray:
        value = c[0] * q[:, 0] ** 2 + c[1] * q[:, 0] * q[:, 1] + c[2] * q[:, 1] ** 2 + c[3] * q[:, 0] + c[4] * q[:, 1] + c[5]
        gx = 2 * c[0] * q[:, 0] + c[1] * q[:, 1] + c[3]
        gy = c[1] * q[:, 0] + 2 * c[2] * q[:, 1] + c[4]
        return np.abs(value) / np.maximum(np.hypot(gx, gy), 1e-9) * self.scale

    def meet_line(self, line: np.ndarray, near: np.ndarray) -> np.ndarray | None:
        a, b, c = line
        direction = np.array([-b, a])
        origin = -c * np.array([a, b]) / (a * a + b * b)
        q0, qd = (origin - self.centre) / self.scale, direction / self.scale
        A, B, C, D, E, F = self.coefficients
        roots = np.roots([A * qd[0] ** 2 + B * qd[0] * qd[1] + C * qd[1] ** 2,
                          2 * A * q0[0] * qd[0] + B * (q0[0] * qd[1] + q0[1] * qd[0]) + 2 * C * q0[1] * qd[1] + D * qd[0] + E * qd[1],
                          A * q0[0] ** 2 + B * q0[0] * q0[1] + C * q0[1] ** 2 + D * q0[0] + E * q0[1] + F])
        real = roots[np.abs(roots.imag) < 1e-9].real
        if not len(real):
            return None
        return min((origin + t * direction for t in real), key=lambda p: float(np.linalg.norm(p - near)))


def meet_lines(first: np.ndarray, second: np.ndarray) -> np.ndarray | None:
    point = np.cross(first, second)
    return None if abs(point[2]) < 1e-12 else point[:2] / point[2]


# --------------------------------------------------------------------------------------
# One frame: trace, primitives, landmarks
# --------------------------------------------------------------------------------------

def trace_frame(image: np.ndarray, source: dict[str, Any], model: PixelModel | None = None) -> dict[str, Any]:
    """Traced edges of one frame. With ``model`` also the landmark pixels where edges meet."""
    outline = floor_outline(image, source["floor_samples_px"], float(source.get("chroma_tolerance", 7.0)))
    edges, arcs = {}, {}
    for name, spec in source["edges"].items():
        points = in_boxes(outline, spec["boxes"])
        if spec["type"] == "line":
            edges[name] = smooth_inliers(points, 1 if spec["along"] == "y" else 0)
        else:
            arcs[name] = points
    trace: dict[str, Any] = {"outline": outline, "edges": edges, "arcs": arcs, "landmarks": {}, "primitives": {}}
    if model is None:
        return trace
    lines = {name: fit_line(model.undistort(points)) for name, points in edges.items() if len(points) >= 8}
    conics = {name: Conic(model.undistort(points)) for name, points in arcs.items() if len(points) >= 12}
    trace["primitives"] = {**{name: {"type": "line", "points": int(len(edges[name])),
                                     "rms_px": float(np.sqrt(np.mean(line_residuals(model.undistort(edges[name])) ** 2)))} for name in lines},
                           **{name: {"type": "arc", "points": int(len(arcs[name])), "inliers": int(conic.inliers.sum()),
                                     "rms_px": conic.rms_px} for name, conic in conics.items()}}
    for name, spec in source["landmarks"].items():
        first, second = spec["from"]
        near = model.undistort([spec["near_px"]])[0]
        if first in lines and second in lines:
            point = meet_lines(lines[first], lines[second])
        elif first in lines and second in conics:
            point = conics[second].meet_line(lines[first], near)
        elif first in conics and second in lines:
            point = conics[first].meet_line(lines[second], near)
        else:
            point = None
        if point is not None and np.isfinite(point).all():
            trace["landmarks"][name] = tuple(float(v) for v in model.distort([point])[0])
    return trace


def boundary_checks(trace: dict[str, Any], mapping, arena: dict[str, Any], source: dict[str, Any]) -> dict[str, Any]:
    """Distance (mm, on the floor) of each traced edge from the drawing's matching boundary."""
    w, h = float(arena["width_m"]), float(arena["height_m"])
    flipper = np.asarray(arena["flipper_vertices_m"], float)
    start, end = flipper[0], flipper[2]
    normal = np.array([end[1] - start[1], start[0] - end[0]])
    normal /= np.linalg.norm(normal)
    fit_names = {name for name, spec in source["landmarks"].items() if spec.get("role", "fit") == "fit"}
    result = {}
    for name, spec in source["edges"].items():
        points = trace["edges"].get(name) if spec["type"] == "line" else trace["arcs"].get(name)
        if points is None or len(points) < 5:
            continue
        floor = mapping.pixel_to_floor(points)
        target = spec["drawing"]
        if target == "x=0": signed = floor[:, 0]
        elif target == "x=W": signed = floor[:, 0] - w
        elif target == "y=0": signed = floor[:, 1]
        elif target == "y=H": signed = floor[:, 1] - h
        elif target == "flipper": signed = (floor - start) @ normal
        else:
            centre = np.asarray(arena["pit_centres_m"][int(target.split(":")[1])], float)
            signed = np.linalg.norm(floor - centre, axis=1) - float(arena["pit_radius_m"])
        anchors = sorted(lm for lm, lspec in source["landmarks"].items() if name in lspec["from"] and lm in fit_names)
        result[name] = {"drawing": target, "points": int(len(points)), "mean_mm": float(1000 * signed.mean()),
                        "min_mm": float(1000 * signed.min()), "max_mm": float(1000 * signed.max()),
                        "rms_mm": float(1000 * np.sqrt(np.mean(signed ** 2))),
                        "floor_extent_m": [[float(floor[:, 0].min()), float(floor[:, 1].min())], [float(floor[:, 0].max()), float(floor[:, 1].max())]],
                        "fit_landmarks_on_this_edge": anchors,
                        "independence": ("pinned at both ends by fit landmarks" if len(anchors) >= 2 else
                                         "pinned at one end by a fit landmark; its direction and length are independent" if anchors else
                                         "not used in the fit")}
    return result


# --------------------------------------------------------------------------------------
# Pictures
# --------------------------------------------------------------------------------------

def draw_outline(image: np.ndarray, mapping, arena: dict[str, Any], scale: int) -> None:
    for polyline, colour in ([(arena_outline(arena), (255, 210, 60))] + [(arc, (60, 60, 255)) for arc in pit_arcs(arena)]
                             + [(flipper_boundary(arena), (0, 165, 255))]):
        pixels = mapping.floor_to_pixel(densify(polyline)) * scale
        if np.isfinite(pixels).all():
            cv2.polylines(image, [np.round(pixels).astype(np.int32).reshape(-1, 1, 2)], False, colour, 1, cv2.LINE_AA)


def annotated_frame(image: np.ndarray, trace: dict[str, Any], clicks: ClickSet, mapping, arena: dict[str, Any],
                    order: list[str], title: str, scale: int = 3) -> np.ndarray:
    out = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    for points in list(trace["edges"].values()) + list(trace["arcs"].values()):
        for x, y in points:
            cv2.circle(out, (int(round(x * scale + scale / 2)), int(round(y * scale + scale / 2))), 1, (80, 255, 80), -1)
    if mapping is not None and mapping.valid:
        draw_outline(out, mapping, arena, scale)
    for number, name in enumerate(order, 1):
        if name not in clicks.points:
            continue
        x, y = clicks.points[name]
        role = clicks.roles.get(name, "fit")
        colour = (80, 220, 80) if role == "fit" else (0, 230, 255)
        centre = (int(round(x * scale)), int(round(y * scale)))
        cv2.drawMarker(out, centre, colour, cv2.MARKER_TILTED_CROSS, 18, 2, cv2.LINE_AA)
        cv2.putText(out, f"{number} {name} [{role.upper()}]", (min(centre[0] + 10, out.shape[1] - 330), centre[1] + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(out, f"{number} {name} [{role.upper()}]", (min(centre[0] + 10, out.shape[1] - 330), centre[1] + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1, cv2.LINE_AA)
    cv2.rectangle(out, (0, out.shape[0] - 26), (out.shape[1], out.shape[0]), (0, 0, 0), -1)
    cv2.putText(out, title, (8, out.shape[0] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
    return out


def topdown_preview(image: np.ndarray, mapping, arena: dict[str, Any], title: str) -> np.ndarray:
    view = topdown_view(image, mapping, arena)
    for polyline, colour in ([(arena_outline(arena), (255, 210, 60))] + [(arc, (60, 60, 255)) for arc in pit_arcs(arena)]
                             + [(flipper_boundary(arena), (0, 165, 255))]):
        cv2.polylines(view, [np.round(topdown_pixel(polyline, arena)).astype(np.int32).reshape(-1, 1, 2)], False, colour, 1, cv2.LINE_AA)
    cv2.rectangle(view, (0, 0), (view.shape[1], 20), (0, 0, 0), -1)
    cv2.putText(view, title, (6, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1, cv2.LINE_AA)
    return view


# --------------------------------------------------------------------------------------
# The calibration run
# --------------------------------------------------------------------------------------

def calibrate(video: str, profile: str, config_path: str | None = None, out_dir: Path | None = None) -> dict[str, Any]:
    config = load_config(config_path)
    profile_dir = PROFILES_DIR / profile
    source = yaml.safe_load((profile_dir / "source.yaml").read_text(encoding="utf-8"))
    if source.get("profile_kind", "historical") == "live":
        raise ValueError("footage_reference only writes historical profiles")
    out_dir = Path(out_dir) if out_dir else ARTIFACTS / profile
    out_dir.mkdir(parents=True, exist_ok=True)
    reference_time = float(source["reference_time_s"])
    image = median_frame(video, reference_time, float(source.get("median_window_s", 24.0)))
    size = (image.shape[1], image.shape[0])
    lens_fit = estimate_fisheye_focal(trace_frame(image, source)["edges"], size)
    model = fisheye_model(lens_fit["focal_px"], size)
    trace = trace_frame(image, source, model)
    order = list(source["landmarks"])
    roles = {name: str(spec.get("role", "fit")) for name, spec in source["landmarks"].items()}
    references = [ref for ref in config.references if ref.name in trace["landmarks"]]

    def fit(use: dict[str, str]):
        return fit_floor_mapping(references, trace["landmarks"], size, model, roles=use, max_fit_residual_mm=1e9)

    mapping = fit(roles)
    quality = assess_fit(references, trace["landmarks"], roles, mapping, config.arena, config.warn_check_error_mm,
                         config.weak_predicted_error_mm, config.min_hull_fraction)
    leave_one_out = {}
    for held in trace["landmarks"]:
        trial = fit({name: ("check" if name == held else "fit") for name in trace["landmarks"]})
        error = next((e for e in trial.errors if e.name == held), None)
        leave_one_out[held] = None if error is None or trial.H is None else {
            "error_mm": error.error_mm, "error_px": error.error_px,
            "offset_mm": [1000 * (error.mapped_xy[0] - error.arena_xy[0]), 1000 * (error.mapped_xy[1] - error.arena_xy[1])]}
    all_fit = fit({name: "fit" for name in trace["landmarks"]})

    clicks = ClickSet(dict(trace["landmarks"]), roles, size, {
        "profile_kind": "historical", "profile": profile, "approximate": True,
        "do_not_use_for_live_camera": "made from recorded footage shot from another camera position",
        "source_video": Path(video).name, "reference_time_s": reference_time,
        "points_obtained_by": "intersections of floor edges traced on a median frame (see footage_reference.py); hints in source.yaml were picked by eye",
        "lens": model.to_dict(), "lens_fit": lens_fit,
        "drawing": "candidate layout from config/arena_overlay.yaml (not measured on this arena)"})
    save_clicks(profile_points_path(profile), clicks, source=f"{Path(video).name} @ {reference_time:.0f} s (median frame)")

    title = f"HISTORICAL footage {Path(video).name[:40]}... @ {reference_time:.0f}s | approximate map | green dots = traced floor edge"
    cv2.imwrite(str(out_dir / "reference_frame.png"), image)
    cv2.imwrite(str(out_dir / "reference_annotated.png"), annotated_frame(image, trace, clicks, mapping, config.arena, order, title))
    cv2.imwrite(str(out_dir / "topdown_preview.png"),
                topdown_preview(image, mapping, config.arena, f"top-down preview, historical profile, APPROXIMATE ({reference_time:.0f} s)"))

    stability = {}
    for other in source.get("check_times_s", []):
        frame = median_frame(video, float(other), float(source.get("median_window_s", 24.0)))
        other_trace = trace_frame(frame, source, model)
        # A trace is only trusted where it looks like the reference trace: people and tools at the
        # arena edge contaminate some edges at some times.
        trusted = set()
        for name, primitive in other_trace["primitives"].items():
            base = trace["primitives"].get(name)
            if base and primitive["points"] >= 0.6 * base["points"] and primitive["rms_px"] <= max(1.0, 1.5 * base["rms_px"]) \
                    and primitive.get("inliers", primitive["points"]) >= 0.5 * primitive["points"]:
                trusted.add(name)
        shifts = {name: float(np.linalg.norm(np.subtract(other_trace["landmarks"][name], trace["landmarks"][name])))
                  for name in trace["landmarks"] if name in other_trace["landmarks"] and set(source["landmarks"][name]["from"]) <= trusted}
        edges = boundary_checks(other_trace, mapping, config.arena, source)
        for name, entry in edges.items():
            entry["trace_trusted"] = name in trusted
        stability[str(int(other))] = {
            "floor_paint_shift": image_shift(image, frame, source["stability_crop_px"]) if source.get("stability_crop_px") else None,
            "landmark_shift_px": shifts, "largest_shift_px": max(shifts.values()) if shifts else None,
            "untrusted_traces": sorted(set(other_trace["primitives"]) - trusted),
            "edges_against_reference_map_mm": edges}
        cv2.imwrite(str(out_dir / f"check_{int(other)}s_annotated.png"),
                    annotated_frame(frame, other_trace, ClickSet({}, {}, size), mapping, config.arena, order,
                                    f"frame at {int(other)} s with the map fitted at {reference_time:.0f} s (outline) and this frame's traced edge (green)"))

    report = {
        "tool": "control_center.tools.footage_reference", "generated_at": datetime.now(timezone.utc).isoformat(),
        "profile": profile, "profile_kind": "historical", "video": Path(video).name, "image_size": list(size),
        "reference_time_s": reference_time, "map_label": "APPROXIMATE: no measured reference in the footage",
        "lens": lens_fit, "primitives": trace["primitives"],
        "landmarks_px": {name: list(point) for name, point in trace["landmarks"].items()},
        "by_eye_hints_px": {name: spec["near_px"] for name, spec in source["landmarks"].items()},
        "hint_to_fitted_px": {name: float(np.linalg.norm(np.subtract(trace["landmarks"][name], source["landmarks"][name]["near_px"])))
                              for name in trace["landmarks"]},
        "landmarks_not_visible": source.get("not_visible", {}), "roles": roles,
        "mapping": mapping.to_report(), "fit_quality": quality.to_report(),
        "leave_one_out": leave_one_out,
        "fit_with_every_landmark": {e.name: e.error_mm for e in all_fit.errors},
        "edges_against_drawing_mm": boundary_checks(trace, mapping, config.arena, source),
        "edges_against_drawing_mm_all_landmarks_fitted": boundary_checks(trace, all_fit, config.arena, source),
        "camera_stability": stability,
        "outputs": {"clicks": str(profile_points_path(profile)), "reference_frame": str(out_dir / "reference_frame.png"),
                    "annotated": str(out_dir / "reference_annotated.png"), "topdown": str(out_dir / "topdown_preview.png")}}
    (out_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def print_summary(report: dict[str, Any]) -> None:
    lens, mapping = report["lens"], report["mapping"]
    print(f"Historical footage calibration: profile {report['profile']} ({report['map_label']})")
    print(f"  lens: fisheye focal {lens['focal_px']:.0f} px (range {lens['focal_range_within_0.1px_rms'][0]:.0f}-"
          f"{lens['focal_range_within_0.1px_rms'][1]:.0f}); edge straightness {lens['straightness_rms_px_uncorrected']:.2f} -> "
          f"{lens['straightness_rms_px']:.2f} px rms")
    for name, point in report["landmarks_px"].items():
        print(f"  landmark {name:24s} px ({point[0]:6.1f}, {point[1]:6.1f})  role {report['roles'][name]}  "
              f"{report['hint_to_fitted_px'][name]:.1f} px from the by-eye hint")
    check = mapping["check_points"]
    print(f"  mapping valid: {mapping['valid']}; fit {mapping['fit_points']['count']} landmarks; held-out check "
          f"{check['count']}: " + (f"max {check['max_mm']:.0f} mm" if check["count"] else "none"))
    print("  leave-one-out error mm: " + ", ".join(f"{name} {value['error_mm']:.0f}" for name, value in report["leave_one_out"].items() if value))
    print(f"  fit quality: {report['fit_quality']['level'].upper()} " + "; ".join(report["fit_quality"]["reasons"]))
    for name, edge in report["edges_against_drawing_mm"].items():
        print(f"  traced {name:9s} vs drawing {edge['drawing']:8s}: mean {edge['mean_mm']:+.0f} mm, range {edge['min_mm']:+.0f}..{edge['max_mm']:+.0f} "
              f"({edge['independence']})")
    for time_s, entry in report["camera_stability"].items():
        trusted = [e for e in entry["edges_against_reference_map_mm"].values() if e["trace_trusted"]]
        worst = max((abs(e["mean_mm"]) for e in trusted), default=float("nan"))
        paint = entry["floor_paint_shift"]
        moved = "n/a" if entry["largest_shift_px"] is None else f"{entry['largest_shift_px']:.1f} px"
        print(f"  camera at {time_s:>5s} s: floor paint shifted " + ("n/a" if paint is None else f"({paint['dx_px']:+.1f}, {paint['dy_px']:+.1f}) px")
              + f"; trusted landmarks moved up to {moved}; trusted edges sit up to {worst:.0f} mm (mean) from the reference map"
              + (f"; untrusted traces: {', '.join(entry['untrusted_traces'])}" if entry["untrusted_traces"] else ""))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m control_center.tools.footage_reference", description=__doc__.split("\n\n")[0])
    parser.add_argument("--video", required=True, help="recorded footage (read only)")
    parser.add_argument("--profile", required=True, help="historical profile folder under config/overlay_profiles")
    parser.add_argument("--config", default=None, help="overlay config with the drawing (default config/arena_overlay.yaml)")
    parser.add_argument("--out", default=None, help="folder for pictures and report (default artifacts/vision-footage/<profile>)")
    args = parser.parse_args(argv)
    try:
        report = calibrate(args.video, args.profile, args.config, None if args.out is None else Path(args.out))
    except (RuntimeError, OSError, ValueError, KeyError) as error:
        print(f"footage_reference: {error}", file=sys.stderr)
        return 2
    print_summary(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
