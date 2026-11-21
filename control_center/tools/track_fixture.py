"""SYNTHETIC fixture helpers for the chassis tracker and the click-selected opponent tracker.

NOT real camera footage. Everything is rendered through the ideal pinhole camera of
``overlay_fixture``; numbers obtained from it check the arithmetic and the state logic, nothing
about a real camera, lens, lighting or motion blur.

The rear marker is drawn here from its own explicit geometry (not from ``chassis_tracker``), so a
test that renders with this module and estimates with the tracker compares two independent
statements of the convention.
"""

from __future__ import annotations

import math
from typing import Any

import cv2
import numpy as np
import yaml

from control_center.tools.overlay_core import DEFAULT_CONFIG_PATH, OverlayConfig, dictionary_id, parse_config
from control_center.tools.overlay_fixture import BotInstance, FixtureCamera, Scene, render_frame

REAR_MARKER = {"id": 8, "face": "rear", "forward_m": -0.075, "left_m": 0.0, "yaw_deg": 0.0,
               "size_m": 0.040, "height_m": 0.026}


def nine_marker_config(rear: dict[str, Any] | None = None, tracker: dict[str, Any] | None = None,
                       marker_overrides: dict[int, dict[str, Any]] | None = None) -> OverlayConfig:
    """The placeholder arena config (top ids 0-3, bottom ids 4-7) plus one rear marker, id 8."""
    data = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    markers = [dict(item) for item in data["bot"]["markers"]]
    for item in markers:
        item.update((marker_overrides or {}).get(int(item["id"]), {}))
    markers.append({**REAR_MARKER, **(rear or {})})
    data["bot"]["markers"] = markers
    if tracker is not None:
        data["tracker"] = tracker
    return parse_config(data)


def rear_corners_world(spec, x: float, y: float, heading: float, orientation: str = "upright") -> np.ndarray:
    """World (x, y, z) of the rear marker's corners TL, TR, BR, BL for a chassis pose."""
    inverted = orientation == "inverted"
    forward = np.array([math.cos(heading), math.sin(heading)])
    left = np.array([-math.sin(heading), math.cos(heading)])
    half = spec.size_m / 2.0
    centre_left = -spec.left_m if inverted else spec.left_m
    height = spec.height_inverted_m if inverted and spec.height_inverted_m is not None else spec.height_m
    spin = math.radians(spec.yaw_deg) + (math.pi if inverted else 0.0)
    corners = []
    # Someone behind the robot looking at its rear face sees the robot's left on their own left.
    for right, up in ((-half, half), (half, half), (half, -half), (-half, -half)):
        r = right * math.cos(spin) - up * math.sin(spin)
        u = right * math.sin(spin) + up * math.cos(spin)
        xy = np.array([x, y]) + spec.forward_m * forward + (centre_left - r) * left
        corners.append([xy[0], xy[1], height + u])
    return np.asarray(corners)


def paint_rear_marker(frame: np.ndarray, config: OverlayConfig, camera: FixtureCamera, marker_id: int,
                      x: float, y: float, heading: float, orientation: str = "upright") -> bool:
    """Draw the vertical rear marker if the camera is behind the robot. Returns whether it was drawn."""
    spec = config.markers[marker_id]
    world = rear_corners_world(spec, x, y, heading, orientation)
    face_centre = world.mean(axis=0)
    backward = -np.array([math.cos(heading), math.sin(heading)])
    if float(np.dot(camera.position[:2] - face_centre[:2], backward)) <= 0.0:
        return False                                         # the rear face points away from the camera
    dictionary = cv2.aruco.getPredefinedDictionary(dictionary_id(config.dictionary))
    cells = int(dictionary.markerSize) + 2
    cell_px = 12
    marker = cv2.aruco.generateImageMarker(dictionary, marker_id, cells * cell_px)
    patch = np.full(((cells + 2) * cell_px, (cells + 2) * cell_px), 255, np.uint8)
    patch[cell_px:-cell_px, cell_px:-cell_px] = marker
    patch = cv2.cvtColor(patch, cv2.COLOR_GRAY2BGR)
    grow = (cells + 2) / cells                               # the white border around the black square
    quad_world = face_centre + (world - face_centre) * grow
    quad = camera.project(quad_world).astype(np.float32)
    size = patch.shape[0]
    source = np.array([[-0.5, -0.5], [size - 0.5, -0.5], [size - 0.5, size - 0.5], [-0.5, size - 0.5]], np.float32)
    matrix = cv2.getPerspectiveTransform(source, quad)
    shape = (frame.shape[1], frame.shape[0])
    warped = cv2.warpPerspective(patch, matrix, shape, flags=cv2.INTER_AREA)
    mask = cv2.warpPerspective(np.full(patch.shape[:2], 255, np.uint8), matrix, shape,
                               flags=cv2.INTER_AREA).astype(np.float32)[..., None] / 255.0
    frame[:] = (frame * (1.0 - mask) + warped * mask).astype(np.uint8)
    return True


def render_bot(config: OverlayConfig, camera: FixtureCamera, x: float, y: float, heading: float,
               orientation: str = "upright", ids: list[int] | None = None, rear: bool = False,
               seed: int = 3, strays: list | None = None, plate: bool = True) -> np.ndarray:
    """One SYNTHETIC frame of the tagged chassis. ``ids`` = the top/bottom markers left visible."""
    bots = [] if ids == [] and not plate else [BotInstance(x, y, heading, orientation=orientation, ids=ids, plate=plate)]
    frame = render_frame(config, camera, Scene(bots, list(strays or [])), noise_seed=seed)
    if ids == [] and plate:                                  # every top/bottom marker covered: the bare plate
        paint_box(frame, camera, (x, y), heading, (0.150, 0.124), 0.034, (45, 45, 45))
    if rear:
        for spec in config.markers.values():
            if spec.face == "rear":
                paint_rear_marker(frame, config, camera, spec.marker_id, x, y, heading, orientation)
    return frame


def paint_box(frame: np.ndarray, camera: FixtureCamera, centre: tuple[float, float], heading: float,
              size: tuple[float, float], height: float, colour: tuple[int, int, int]) -> None:
    """A flat coloured slab at ``height`` (the same stand-in object the shadow-stream fixture uses)."""
    length, breadth = size
    c, s = math.cos(heading), math.sin(heading)
    corners = np.array([[length / 2, breadth / 2], [length / 2, -breadth / 2], [-length / 2, -breadth / 2], [-length / 2, breadth / 2]])
    world = np.array(centre) + corners @ np.array([[c, s], [-s, c]])
    pixels = camera.project(np.column_stack([world, np.full(4, height)]))
    cv2.fillConvexPoly(frame, np.round(pixels).astype(np.int32), colour, cv2.LINE_AA)
