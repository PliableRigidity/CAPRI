"""Is the camera good enough, and does it see the whole arena? Pure checks for the camera step.

Nothing here opens a camera: it judges numbers the hub already has (capture state, frame size, the saved
corner clicks) and returns ``Readiness`` values with the reason and what to do.
"""

from __future__ import annotations

import math
from typing import Any

from control_center.backend.readiness import FAILED, NOT_CONFIGURED, READY, WARNING, Readiness

MIN_FPS = 10.0
EDGE_MARGIN_FRACTION = 0.02          # a corner closer to the picture edge than this is "at the edge"
MIN_ARENA_AREA_FRACTION = 0.12       # the arena should fill at least this share of the picture
MIN_MARKER_PX = 14.0                 # a marker narrower than this (at the far side) is decoded unreliably
MAX_FRAME_AGE_S = 1.0


def polygon_area(points: list[tuple[float, float]]) -> float:
    return abs(sum(x1 * y2 - x2 * y1 for (x1, y1), (x2, y2) in zip(points, points[1:] + points[:1]))) / 2.0


def field_of_view(corners_px: list[tuple[float, float]] | None, frame_size: tuple[int, int] | None, side_lengths_m: tuple[float, float] | None,
                  marker_size_m: float | None = None, marker_size_is_placeholder: bool = True) -> Readiness:
    """Does the saved arena fit in the picture, fill enough of it, and leave the markers big enough to decode?"""
    if corners_px is None or frame_size is None or side_lengths_m is None:
        return Readiness(WARNING, "Field of view not checked", "It is checked against the arena corners, which are not clicked yet.",
                         "Calibrate the arena (ARENA tab); this check then runs by itself.")
    width, height = frame_size
    margin = EDGE_MARGIN_FRACTION * min(width, height)
    outside = [i + 1 for i, (u, v) in enumerate(corners_px) if not (0.0 <= u <= width and 0.0 <= v <= height)]
    if outside:
        return Readiness(FAILED, "Part of the arena is outside the picture",
                         f"corner(s) {outside} fall outside the {width} x {height} frame",
                         "Move the camera back or higher, or zoom out, until all four corners are inside; then recalibrate the camera.")
    close = [i + 1 for i, (u, v) in enumerate(corners_px) if min(u, width - u, v, height - v) < margin]
    area = polygon_area(corners_px) / float(width * height)
    sides_px = [math.hypot(a[0] - b[0], a[1] - b[1]) for a, b in zip(corners_px, corners_px[1:] + corners_px[:1])]
    sides_m = [side_lengths_m[0], side_lengths_m[1], side_lengths_m[0], side_lengths_m[1]]
    px_per_m = min(px / m for px, m in zip(sides_px, sides_m) if m > 0)
    notes: list[str] = []
    level = READY
    if close:
        level = WARNING
        notes.append(f"corner(s) {close} are within {100 * EDGE_MARGIN_FRACTION:.0f}% of the picture edge: a small bump of the camera would lose them")
    if area < MIN_ARENA_AREA_FRACTION:
        level = WARNING
        notes.append(f"the arena fills only {100 * area:.0f}% of the picture (aim for at least {100 * MIN_ARENA_AREA_FRACTION:.0f}%); move the camera closer or zoom in")
    detail = f"arena fills {100 * area:.0f}% of the picture; farthest side {px_per_m:.0f} px per metre"
    if marker_size_m:
        apparent = marker_size_m * px_per_m
        detail += f"; a {marker_size_m * 1000:.0f} mm marker is about {apparent:.0f} px wide there"
        if apparent < MIN_MARKER_PX:
            level = WARNING
            notes.append(f"markers would be only {apparent:.0f} px wide at the far side (under {MIN_MARKER_PX:.0f} px they are often missed): "
                         "move the camera closer, or print larger markers")
        if marker_size_is_placeholder:
            notes.append("the marker size in the config is a placeholder, so this is only an estimate")
    if notes:
        return Readiness(level, "Field of view usable, with warnings" if level == WARNING else "Field of view good",
                         detail + ". " + "; ".join(notes), "Adjust the camera, then recalibrate the camera corners." if level == WARNING else "")
    return Readiness(READY, "Field of view good", detail)


def camera_state(capture_state: str, detail: str, fps: float, frame_age_s: float | None, mock: bool = False,
                 replay: bool = False) -> Readiness:
    """The camera itself: open, delivering frames, fast enough."""
    if capture_state in ("OPENING", "RECONNECTING"):
        return Readiness(WARNING, "Camera is opening", detail, "Wait a few seconds; if it stays, check the index or unplug and replug the camera.")
    if capture_state not in ("LIVE", "MOCK", "REPLAY"):
        return Readiness(FAILED if capture_state else NOT_CONFIGURED, f"Camera {capture_state.lower() or 'not open'}",
                         detail or "no frames", "Close any other program using the camera (the Logitech app, a browser tab), choose the right index below and press Open camera.")
    if frame_age_s is not None and frame_age_s > MAX_FRAME_AGE_S:
        return Readiness(FAILED, "The camera stopped delivering frames", f"the newest frame is {frame_age_s:.1f} s old",
                         "Unplug and replug the camera, then press Open camera.")
    notes = []
    if mock:
        notes.append("MOCK: a synthetic picture, not a camera")
    if replay:
        notes.append("REPLAY: a recorded video, not a live camera")
    if fps and fps < MIN_FPS:
        notes.append(f"only {fps:.1f} fps (tracking needs about {MIN_FPS:.0f} or more): lower the resolution or close other programs")
    if notes:
        return Readiness(WARNING, "Camera working, with warnings", "; ".join(notes))
    return Readiness(READY, "Camera delivering frames", f"{fps:.1f} fps" if fps else "frames arriving")
