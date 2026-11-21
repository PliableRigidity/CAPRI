"""SYNTHETIC test fixture for the arena overlay tool. NOT real camera footage.

Renders the nominal arena floor and ArUco markers through an ideal pinhole camera with a known
pose, so the mapping and pose maths can be checked against exact ground truth offline. The
rendered scene has no walls, no lens distortion, no motion blur and no lighting variation, so
numbers obtained from it say nothing about real-camera accuracy.

    python -m control_center.tools.overlay_fixture --out artifacts/vision-overlay-fixture
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import json
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from control_center.tools.overlay_core import (ClickSet, OverlayConfig, dictionary_id,
                                               load_config, save_clicks)

FIXTURE_LABEL = "SYNTHETIC TEST FIXTURE - NOT REAL CAMERA FOOTAGE"
PLATE_SIZE_M = (0.150, 0.124)          # fixture-only dark plate under the markers (length, width)
_TEXTURE_MARGIN_MM = 150
_SUPERSAMPLE = 2
_FLOOR_CACHE: dict[Any, np.ndarray] = {}


@dataclass
class FixtureCamera:
    K: np.ndarray
    R: np.ndarray                        # world -> camera rotation (x right, y down, z forward)
    position: np.ndarray                 # camera centre in the arena frame (x, y, z)
    image_size: tuple[int, int]

    def project(self, points_xyz: Any) -> np.ndarray:
        pts = np.asarray(points_xyz, float).reshape(-1, 3)
        cam = (pts - self.position) @ self.R.T
        pix = cam @ self.K.T
        return pix[:, :2] / pix[:, 2:3]

    def plane_homography(self, height_m: float = 0.0) -> np.ndarray:
        """Homography taking arena (x, y) on the plane z = height to pixels."""
        offset = self.R @ (np.array([0.0, 0.0, height_m]) - self.position)
        return self.K @ np.column_stack([self.R[:, 0], self.R[:, 1], offset])

    def to_report(self) -> dict[str, Any]:
        return {"camera_matrix": self.K.tolist(), "position_m": self.position.tolist(),
                "rotation_world_to_camera": self.R.tolist(), "image_size": list(self.image_size)}


def make_camera(position: tuple[float, float, float], target: tuple[float, float, float],
                focal_px: float, image_size: tuple[int, int] = (1280, 720),
                principal: tuple[float, float] | None = None) -> FixtureCamera:
    centre = np.asarray(position, float)
    forward = np.asarray(target, float) - centre
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    cx, cy = principal if principal is not None else (image_size[0] / 2.0, image_size[1] / 2.0)
    K = np.array([[focal_px, 0.0, cx], [0.0, focal_px, cy], [0.0, 0.0, 1.0]])
    return FixtureCamera(K, np.vstack([right, down, forward]), centre, tuple(image_size))


def default_camera() -> FixtureCamera:
    """Off-axis oblique view from outside the y=0 wall; the whole floor is in frame."""
    return make_camera((0.40, -0.55, 1.30), (0.60, 0.42, 0.0), focal_px=930.0)


@dataclass
class MarkerPlacement:
    marker_id: int
    centre_xy: tuple[float, float]
    up_angle_rad: float                  # direction of the marker's top edge in the arena frame
    size_m: float
    height_m: float


@dataclass
class BotInstance:
    x: float
    y: float
    heading_rad: float
    orientation: str = "upright"         # "upright" shows top markers, "inverted" bottom ones
    ids: list[int] | None = None         # subset of that face's markers; None = all of them
    plate: bool = True


@dataclass
class Scene:
    bots: list[BotInstance] = field(default_factory=list)
    strays: list[MarkerPlacement] = field(default_factory=list)
    expect: str = "ok"                   # "ok" or "unknown"
    note: str = ""


def _rotation(theta: float) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s], [s, c]])


def bot_marker_placements(config: OverlayConfig, bot: BotInstance) -> list[MarkerPlacement]:
    face = "top" if bot.orientation == "upright" else "bottom"
    placements = []
    for spec in config.markers.values():
        if spec.face != face or (bot.ids is not None and spec.marker_id not in bot.ids):
            continue
        centre = np.array([bot.x, bot.y]) + _rotation(bot.heading_rad) @ [spec.forward_m,
                                                                          spec.left_m]
        placements.append(MarkerPlacement(spec.marker_id, (float(centre[0]), float(centre[1])),
                                          bot.heading_rad + math.radians(spec.yaw_deg),
                                          spec.size_m, spec.height_m))
    return placements


def _floor_texture(arena: dict[str, Any], seed: int = 7) -> tuple[np.ndarray, np.ndarray]:
    """Top-down floor image at 1 px per mm and the arena-metres -> texture-pixel transform."""
    w_mm, h_mm = int(round(arena["width_m"] * 1000)), int(round(arena["height_m"] * 1000))
    margin = _TEXTURE_MARGIN_MM
    to_texture = np.array([[1000.0, 0.0, margin], [0.0, -1000.0, margin + h_mm],
                           [0.0, 0.0, 1.0]])

    def tex(point: Any) -> tuple[int, int]:
        x, y = point
        return int(round(1000 * x + margin)), int(round(margin + h_mm - 1000 * y))

    rng = np.random.default_rng(seed)
    texture = np.full((h_mm + 2 * margin, w_mm + 2 * margin, 3), 70, np.uint8)
    floor = np.clip(150 + rng.normal(0, 4, (h_mm, w_mm, 1)), 0, 255).astype(np.uint8)
    texture[margin:margin + h_mm, margin:margin + w_mm] = floor
    radius_px = int(round(arena["pit_radius_m"] * 1000))
    for centre in arena["pit_centres_m"]:
        cv2.circle(texture, tex(centre), radius_px, (12, 12, 12), -1, cv2.LINE_AA)
    cv2.fillConvexPoly(texture, np.array([tex(v) for v in arena["flipper_vertices_m"]], np.int32),
                       (235, 235, 235), cv2.LINE_AA)
    # Everything outside the arena boundary (including the pit circle overspill) is "wall".
    outside = np.ones(texture.shape[:2], bool)
    outside[margin:margin + h_mm, margin:margin + w_mm] = False
    texture[outside] = (205, 215, 220)
    return texture, to_texture


def _warp_patch(canvas: np.ndarray, patch: np.ndarray, quad: np.ndarray) -> None:
    size = patch.shape[0]
    source = np.array([[-0.5, -0.5], [size - 0.5, -0.5], [size - 0.5, size - 0.5],
                       [-0.5, size - 0.5]], np.float32)
    # Warp into the quad's bounding box only (same result as a full-canvas warp, much cheaper).
    x0, y0 = np.maximum(np.floor(quad.min(axis=0)).astype(int) - 2, 0)
    x1 = min(int(math.ceil(quad[:, 0].max())) + 3, canvas.shape[1])
    y1 = min(int(math.ceil(quad[:, 1].max())) + 3, canvas.shape[0])
    if x1 <= x0 or y1 <= y0:
        return
    matrix = cv2.getPerspectiveTransform(source, (quad - [x0, y0]).astype(np.float32))
    shape = (x1 - x0, y1 - y0)
    warped = cv2.warpPerspective(patch, matrix, shape, flags=cv2.INTER_LINEAR)
    mask = cv2.warpPerspective(np.full(patch.shape[:2], 255, np.uint8), matrix, shape,
                               flags=cv2.INTER_LINEAR).astype(np.float32)[..., None] / 255.0
    region = canvas[y0:y1, x0:x1]
    region[:] = (region * (1.0 - mask) + warped * mask).astype(np.uint8)


def render_frame(config: OverlayConfig, camera: FixtureCamera, scene: Scene | None = None,
                 label: bool = True, noise_seed: int = 3) -> np.ndarray:
    """Render the arena floor plus any bots/stray markers. Always burnt-in labelled SYNTHETIC."""
    scene = scene or Scene()
    ss = _SUPERSAMPLE
    width, height = camera.image_size
    scale = np.array([[ss, 0.0, (ss - 1) / 2.0], [0.0, ss, (ss - 1) / 2.0], [0.0, 0.0, 1.0]])
    cache_key = (camera.K.tobytes(), camera.R.tobytes(), camera.position.tobytes(),
                 camera.image_size, json.dumps(config.arena, sort_keys=True))
    if cache_key not in _FLOOR_CACHE:
        texture, to_texture = _floor_texture(config.arena)
        floor_h = scale @ camera.plane_homography(0.0) @ np.linalg.inv(to_texture)
        _FLOOR_CACHE.clear()                                  # keep one rendered floor only
        _FLOOR_CACHE[cache_key] = cv2.warpPerspective(
            texture, floor_h, (width * ss, height * ss), flags=cv2.INTER_LINEAR,
            borderValue=(55, 50, 45))
    canvas = _FLOOR_CACHE[cache_key].copy()

    def to_canvas(points_xy: np.ndarray, height_m: float) -> np.ndarray:
        xyz = np.column_stack([points_xy, np.full(len(points_xy), height_m)])
        pix = camera.project(xyz)
        return pix * ss + (ss - 1) / 2.0

    dictionary = cv2.aruco.getPredefinedDictionary(dictionary_id(config.dictionary))
    cells = int(dictionary.markerSize) + 2
    cell_px = 12
    placements: list[MarkerPlacement] = list(scene.strays)
    for bot in scene.bots:
        bot_markers = bot_marker_placements(config, bot)
        placements.extend(bot_markers)
        if bot.plate and bot_markers:
            plate_height = bot_markers[0].height_m
            length, breadth = PLATE_SIZE_M
            rotation = _rotation(bot.heading_rad)
            corners = np.array([[length / 2, breadth / 2], [length / 2, -breadth / 2],
                                [-length / 2, -breadth / 2], [-length / 2, breadth / 2]])
            plate = np.array([bot.x, bot.y]) + corners @ rotation.T
            cv2.fillConvexPoly(canvas, np.round(to_canvas(plate, plate_height)).astype(np.int32),
                               (45, 45, 45), cv2.LINE_AA)
            nose = np.array([[length / 2, 0.012], [length / 2 + 0.018, 0.0],
                             [length / 2, -0.012]])
            nose = np.array([bot.x, bot.y]) + nose @ rotation.T
            cv2.fillConvexPoly(canvas, np.round(to_canvas(nose, plate_height)).astype(np.int32),
                               (40, 150, 240), cv2.LINE_AA)
    for placement in placements:
        marker = cv2.aruco.generateImageMarker(dictionary, placement.marker_id,
                                               (cells) * cell_px)
        patch = np.full(((cells + 2) * cell_px, (cells + 2) * cell_px), 255, np.uint8)
        patch[cell_px:-cell_px, cell_px:-cell_px] = marker
        patch = cv2.cvtColor(patch, cv2.COLOR_GRAY2BGR)
        half = placement.size_m * (cells + 2) / cells / 2.0
        up = np.array([math.cos(placement.up_angle_rad), math.sin(placement.up_angle_rad)])
        right = np.array([up[1], -up[0]])
        centre = np.asarray(placement.centre_xy)
        quad = np.array([centre + half * (up - right), centre + half * (up + right),
                         centre + half * (-up + right), centre + half * (-up - right)])
        _warp_patch(canvas, patch, to_canvas(quad, placement.height_m))

    frame = cv2.resize(canvas, (width, height), interpolation=cv2.INTER_AREA)
    rng = np.random.default_rng(noise_seed)
    frame = np.clip(frame.astype(np.float32) + rng.normal(0, 1.5, frame.shape), 0, 255)
    frame = frame.astype(np.uint8)
    if label:
        cv2.rectangle(frame, (0, height - 26), (width, height), (0, 0, 0), -1)
        cv2.putText(frame, FIXTURE_LABEL, (10, height - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (0, 255, 255), 1, cv2.LINE_AA)
    return frame


def reference_pixels(config: OverlayConfig, camera: FixtureCamera, noise_px: float = 0.0,
                     seed: int = 0) -> dict[str, tuple[float, float]]:
    """Exact (optionally noised) image positions of every configured floor reference point."""
    rng = np.random.default_rng(seed)
    pixels = {}
    for ref in config.references:
        u, v = camera.project([[ref.arena_xy[0], ref.arena_xy[1], 0.0]])[0]
        if noise_px:
            du, dv = rng.normal(0.0, noise_px, 2)
            u, v = u + du, v + dv
        pixels[ref.name] = (float(u), float(v))
    return pixels


def fixture_clicks(config: OverlayConfig, camera: FixtureCamera, noise_px: float = 0.0,
                   seed: int = 0) -> ClickSet:
    return ClickSet(points=reference_pixels(config, camera, noise_px, seed),
                    image_size=camera.image_size,
                    meta={"synthetic_fixture": True, "label": FIXTURE_LABEL,
                          "click_noise_px": noise_px})


def demo_scenes() -> list[Scene]:
    """A short labelled sequence: driving upright, absent, inverted, then two conflict cases."""
    scenes: list[Scene] = []
    for step in range(8):                                   # upright bot on a curved path
        angle = math.radians(20 + 18 * step)
        scenes.append(Scene([BotInstance(0.575 + 0.30 * math.cos(angle),
                                         0.575 + 0.30 * math.sin(angle),
                                         angle + math.pi / 2)], note="upright, driving"))
    scenes += [Scene(expect="unknown", note="no bot in view")] * 2
    for step in range(4):                                   # inverted bot
        scenes.append(Scene([BotInstance(0.35 + 0.10 * step, 0.40, math.radians(-30 + 25 * step),
                                         orientation="inverted")], note="inverted"))
    upright = BotInstance(0.70, 0.60, math.radians(100))
    scenes.append(Scene([upright], [MarkerPlacement(5, (0.30, 0.30), 0.3, 0.040, 0.034)],
                        expect="unknown", note="top markers plus a stray bottom marker"))
    scenes.append(Scene([BotInstance(0.70, 0.60, math.radians(100), ids=[0, 3])],
                        [MarkerPlacement(1, (0.35, 0.80), 1.0, 0.040, 0.034)],
                        expect="unknown", note="marker 1 far from where markers 0 and 3 put it"))
    scenes.append(Scene([BotInstance(0.88, 0.88, math.radians(-135))], note="upright, far side"))
    scenes.append(Scene([BotInstance(0.12, 0.12, math.radians(45))], note="upright, near corner"))
    return scenes


def write_fixture(out_dir: Path | str, config: OverlayConfig | None = None,
                  camera: FixtureCamera | None = None, fps: float = 10.0) -> dict[str, Any]:
    """Write a labelled synthetic frame, video, click file and ground truth to ``out_dir``."""
    config = config or load_config()
    camera = camera or default_camera()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = {"frame": out / "synthetic_frame.png", "video": out / "synthetic_video.avi",
             "clicks": out / "synthetic_clicks.json", "truth": out / "synthetic_truth.json"}
    still = Scene([BotInstance(0.80, 0.45, math.radians(35.0))], note="single upright bot")
    if not cv2.imwrite(str(paths["frame"]), render_frame(config, camera, still)):
        raise OSError(f"could not write {paths['frame']}")
    writer = cv2.VideoWriter(str(paths["video"]), cv2.VideoWriter_fourcc(*"MJPG"), fps,
                             camera.image_size)
    if not writer.isOpened():
        raise OSError("could not open an MJPG video writer for the synthetic fixture")
    scenes = demo_scenes()
    for index, scene in enumerate(scenes):
        writer.write(render_frame(config, camera, scene, noise_seed=100 + index))
    writer.release()
    save_clicks(paths["clicks"], fixture_clicks(config, camera), source=FIXTURE_LABEL)

    def scene_truth(scene: Scene) -> dict[str, Any]:
        bot = scene.bots[0] if scene.bots else None
        return {"expect": scene.expect, "note": scene.note,
                "bot": None if bot is None else {
                    "x_m": bot.x, "y_m": bot.y, "heading_deg": math.degrees(bot.heading_rad),
                    "orientation": bot.orientation}}

    truth = {"label": FIXTURE_LABEL, "fps": fps, "camera": camera.to_report(),
             "frame": scene_truth(still), "video_frames": [scene_truth(s) for s in scenes]}
    paths["truth"].write_text(json.dumps(truth, indent=2), encoding="utf-8")
    return {"paths": {key: str(value) for key, value in paths.items()}, "truth": truth}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default="artifacts/vision-overlay-fixture")
    parser.add_argument("--config", default=None, help="overlay config (default config/arena_overlay.yaml)")
    args = parser.parse_args(argv)
    result = write_fixture(args.out, load_config(args.config))
    print(FIXTURE_LABEL)
    for key, value in result["paths"].items():
        print(f"  {key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
