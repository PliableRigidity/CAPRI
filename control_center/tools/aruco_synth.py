"""SYNTHETIC ArUco frames with known ground truth and controlled degradations. NOT real camera footage.

Used by tests/test_aruco_tracking_conditions.py and tests/test_aruco_dictionary_consistency.py, and by the
detector-parameter comparison in docs/aruco-tracking-audit.md. Nothing here opens a camera, a serial port
or a socket. Everything is rendered with ``cv2.aruco.generateImageMarker`` and perspective-warped through a
known homography onto a textured floor, so the true corners of every marker are known exactly.

What it can imitate (each is an explicit, parameterised function, so a test says what it applied):
  motion blur (a straight streak), partial occlusion (a rectangle over part of a marker), glare (a
  saturated blob that washes out part of the picture), sensor noise, JPEG compression, markers of another
  dictionary, ids outside the layout, a repeated id, markers visible only some of the time.

What it cannot imitate, so nothing measured here is a statement about the real camera: the real lens, the
real lighting, rolling shutter, the real print quality and the real marker size and mounting.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence

import cv2
import numpy as np

SYNTHETIC_LABEL = "SYNTHETIC ARUCO FRAME - NOT REAL CAMERA FOOTAGE"
SUPERSAMPLE = 3


def dictionary_by_name(name: str):
    if not hasattr(cv2.aruco, name):
        raise ValueError(f"unknown dictionary {name!r}")
    return cv2.aruco.getPredefinedDictionary(int(getattr(cv2.aruco, name)))


def marker_patch(dictionary_name: str, marker_id: int, cell_px: int = 16, quiet_cells: int = 1) -> np.ndarray:
    """The marker with a white quiet zone of ``quiet_cells`` cells around the black border (BGR)."""
    dictionary = dictionary_by_name(dictionary_name)
    cells = int(dictionary.markerSize) + 2
    marker = cv2.aruco.generateImageMarker(dictionary, marker_id, cells * cell_px)
    margin = quiet_cells * cell_px
    patch = np.full((cells * cell_px + 2 * margin,) * 2, 255, np.uint8)
    patch[margin:-margin or None, margin:-margin or None] = marker
    return cv2.cvtColor(patch, cv2.COLOR_GRAY2BGR)


def quiet_grow(dictionary_name: str, quiet_cells: int = 1) -> float:
    """How much bigger the printed patch is than the black square (so a quad can be given for the black square)."""
    cells = int(dictionary_by_name(dictionary_name).markerSize) + 2
    return (cells + 2 * quiet_cells) / cells


_FLOORS: dict[tuple, np.ndarray] = {}


def textured_floor(size: tuple[int, int] = (1280, 720), seed: int = 5, base: int = 150) -> np.ndarray:
    """Floor-like picture: mid grey, low-frequency mottling, fine grain and a few dark scuffs and light tape strips."""
    key = (tuple(size), seed, base)
    if key not in _FLOORS:
        if len(_FLOORS) > 8:
            _FLOORS.clear()
        _FLOORS[key] = _make_floor(size, seed, base)
    return _FLOORS[key].copy()


def _make_floor(size: tuple[int, int], seed: int, base: int) -> np.ndarray:
    width, height = size
    rng = np.random.default_rng(seed)
    coarse = rng.normal(0, 1, (max(2, height // 60), max(2, width // 60))).astype(np.float32)
    mottling = cv2.resize(coarse, (width, height), interpolation=cv2.INTER_CUBIC) * 9.0
    grain = rng.normal(0, 3.5, (height, width)).astype(np.float32)
    gray = np.clip(base + mottling + grain, 0, 255).astype(np.uint8)
    image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    for _ in range(14):                                          # scuffs and tape: edges that are not markers
        a = (int(rng.integers(0, width)), int(rng.integers(0, height)))
        b = (int(np.clip(a[0] + rng.integers(-120, 120), 0, width - 1)), int(np.clip(a[1] + rng.integers(-90, 90), 0, height - 1)))
        colour = int(rng.choice([55, 70, 215, 235]))
        cv2.line(image, a, b, (colour,) * 3, int(rng.integers(1, 4)), cv2.LINE_AA)
    return image


@dataclass
class PlacedMarker:
    """A marker whose BLACK SQUARE has these four image corners (TL, TR, BR, BL of the marker, i.e. ArUco order)."""
    marker_id: int
    corners_px: np.ndarray               # (4, 2)
    dictionary: str = "DICT_4X4_50"
    visible: bool = True

    def centre(self) -> np.ndarray:
        return np.asarray(self.corners_px, float).mean(axis=0)

    def side_px(self) -> float:
        c = np.asarray(self.corners_px, float)
        return float(np.mean(np.linalg.norm(c - np.roll(c, -1, axis=0), axis=1)))


def square_corners(centre: tuple[float, float], side_px: float, angle_deg: float = 0.0,
                   tilt: tuple[float, float] = (0.0, 0.0)) -> np.ndarray:
    """Corners TL, TR, BR, BL of a square rotated by ``angle_deg`` (clockwise in the image), with an optional
    perspective ``tilt`` (fractional foreshortening of the left/right and top/bottom edges, 0 = fronto-parallel)."""
    h = side_px / 2.0
    base = np.array([[-h, -h], [h, -h], [h, h], [-h, h]], float)
    kx, ky = tilt
    base[:, 0] *= 1.0 - kx * np.sign(base[:, 1]) * 0.5
    base[:, 1] *= 1.0 - ky * np.sign(base[:, 0]) * 0.5
    a = math.radians(angle_deg)
    rot = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
    return base @ rot.T + np.asarray(centre, float)


def paint_marker(canvas: np.ndarray, placed: PlacedMarker, quiet_cells: int = 1, cell_px: int = 16) -> None:
    """Warp the marker onto ``canvas`` (in place) through the homography that takes its patch to ``corners_px``."""
    patch = marker_patch(placed.dictionary, placed.marker_id, cell_px, quiet_cells)
    grow = quiet_grow(placed.dictionary, quiet_cells)
    corners = np.asarray(placed.corners_px, np.float32)
    centre = corners.mean(axis=0)
    quad = (centre + (corners - centre) * grow).astype(np.float32)
    size = patch.shape[0]
    source = np.array([[-0.5, -0.5], [size - 0.5, -0.5], [size - 0.5, size - 0.5], [-0.5, size - 0.5]], np.float32)
    matrix = cv2.getPerspectiveTransform(source, quad)
    shape = (canvas.shape[1], canvas.shape[0])
    warped = cv2.warpPerspective(patch, matrix, shape, flags=cv2.INTER_AREA)
    mask = cv2.warpPerspective(np.full(patch.shape[:2], 255, np.uint8), matrix, shape, flags=cv2.INTER_AREA)
    alpha = (mask.astype(np.float32) / 255.0)[..., None]
    canvas[:] = np.clip(canvas * (1.0 - alpha) + warped * alpha, 0, 255).astype(np.uint8)


def render(markers: Iterable[PlacedMarker], size: tuple[int, int] = (1280, 720), seed: int = 5, floor: np.ndarray | None = None,
           supersample: int = SUPERSAMPLE) -> np.ndarray:
    """Floor + markers, rendered at ``supersample`` times the resolution and shrunk with area averaging (soft, antialiased edges)."""
    width, height = size
    frame = (textured_floor(size, seed) if floor is None else floor).copy()
    for placed in markers:
        if not placed.visible:
            continue
        corners = np.asarray(placed.corners_px, float)
        grow = quiet_grow(placed.dictionary)
        centre = corners.mean(axis=0)
        outer = centre + (corners - centre) * grow
        x0, y0 = np.maximum(np.floor(outer.min(axis=0)).astype(int) - 3, 0)
        x1, y1 = np.minimum(np.ceil(outer.max(axis=0)).astype(int) + 4, [width, height])
        if x1 <= x0 or y1 <= y0:
            continue
        # Render only this marker's neighbourhood at ``supersample`` times the resolution, then shrink it with area
        # averaging (soft antialiased edges), and paste it back: the same picture as a whole-frame supersample, much cheaper.
        crop = frame[y0:y1, x0:x1]
        big = cv2.resize(crop, ((x1 - x0) * supersample, (y1 - y0) * supersample), interpolation=cv2.INTER_LINEAR)
        local = (corners - [x0, y0]) * supersample + (supersample - 1) / 2.0
        paint_marker(big, PlacedMarker(placed.marker_id, local, placed.dictionary), cell_px=16)
        frame[y0:y1, x0:x1] = cv2.resize(big, (x1 - x0, y1 - y0), interpolation=cv2.INTER_AREA)
    return frame


# ---------------------------------------------------------------------------------------- degradations


def motion_blur(image: np.ndarray, length_px: float, angle_deg: float = 0.0) -> np.ndarray:
    """A straight-line blur of ``length_px`` (the streak a point leaves during the exposure)."""
    if length_px <= 1.0:
        return image.copy()
    size = int(math.ceil(length_px)) | 1
    kernel = np.zeros((size, size), np.float32)
    centre = size // 2
    a = math.radians(angle_deg)
    half = (length_px - 1) / 2.0
    cv2.line(kernel, (int(round(centre - half * math.cos(a))), int(round(centre - half * math.sin(a)))),
             (int(round(centre + half * math.cos(a))), int(round(centre + half * math.sin(a)))), 1.0, 1, cv2.LINE_AA)
    kernel /= max(float(kernel.sum()), 1e-6)
    return cv2.filter2D(image, -1, kernel, borderType=cv2.BORDER_REFLECT)


def occlude(image: np.ndarray, corners_px: np.ndarray, fraction: float, side: str = "right", colour: int = 120) -> np.ndarray:
    """Cover ``fraction`` of a marker's width from one ``side`` ('left', 'right', 'top', 'bottom') with a flat rectangle."""
    out = image.copy()
    c = np.asarray(corners_px, float)
    x0, y0 = c.min(axis=0)
    x1, y1 = c.max(axis=0)
    w, h = x1 - x0, y1 - y0
    box = {"right": (x1 - fraction * w, y0 - 2, x1 + 2, y1 + 2), "left": (x0 - 2, y0 - 2, x0 + fraction * w, y1 + 2),
           "bottom": (x0 - 2, y1 - fraction * h, x1 + 2, y1 + 2), "top": (x0 - 2, y0 - 2, x1 + 2, y0 + fraction * h)}[side]
    cv2.rectangle(out, (int(round(box[0])), int(round(box[1]))), (int(round(box[2])), int(round(box[3]))), (colour,) * 3, -1)
    return out


def glare(image: np.ndarray, centre: tuple[float, float], radius_px: float, strength: float = 1.0) -> np.ndarray:
    """A bright, soft-edged blob (a reflection): adds up to ``strength`` * 255 at the centre, saturating the marker under it."""
    height, width = image.shape[:2]
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    distance = np.sqrt((xx - centre[0]) ** 2 + (yy - centre[1]) ** 2)
    blob = np.clip(1.0 - distance / max(radius_px, 1.0), 0.0, 1.0) ** 0.5
    return np.clip(image.astype(np.float32) + 255.0 * strength * 2.0 * blob[..., None], 0, 255).astype(np.uint8)


def add_noise(image: np.ndarray, sigma: float, seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.clip(image.astype(np.float32) + rng.normal(0, sigma, image.shape), 0, 255).astype(np.uint8)


def jpeg(image: np.ndarray, quality: int = 70) -> np.ndarray:
    ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    if not ok:
        raise RuntimeError("jpeg encode failed")
    return cv2.imdecode(encoded, cv2.IMREAD_COLOR)


def camera_like(image: np.ndarray, blur_px: float = 0.0, blur_angle_deg: float = 0.0, noise_sigma: float = 2.0,
                jpeg_quality: int | None = 75, seed: int = 1) -> np.ndarray:
    """The usual camera chain applied to a clean render: streak, sensor noise, compression."""
    out = motion_blur(image, blur_px, blur_angle_deg) if blur_px > 1.0 else image
    out = add_noise(out, noise_sigma, seed) if noise_sigma > 0 else out
    return jpeg(out, jpeg_quality) if jpeg_quality else out


def corner_error_px(detected: np.ndarray, truth: np.ndarray) -> float:
    """Largest distance between matching corners (a marker's corner order is fixed, so no matching is needed)."""
    return float(np.max(np.linalg.norm(np.asarray(detected, float).reshape(4, 2) - np.asarray(truth, float).reshape(4, 2), axis=1)))


def find(detections: Sequence, marker_id: int):
    return [d for d in detections if d.tag_id == marker_id]
