"""Interactive arena and hazard set-up: the pure model behind the Command Centre's ARENA tab.

Read-only vision configuration. Nothing here opens a camera, a serial port or a robot link, and nothing
sends a command. The window code lives in ``control_center/screens/arena.py``; this module holds the
maths, the checks, the click-by-click session and the files, so all of it runs headless in tests.

What it does
------------
1. Four clicked arena corners + the arena's real size  -> the image-to-floor mapping (the existing
   ``fit_floor_mapping``, so the old overlay, tracker and shadow tools read the result unchanged).
2. Pit centres and the flipper's vertices are clicked on the image and converted to METRES with that
   mapping. Pit radii are typed. All hazard geometry is stored in metres, never in pixels.
3. The geometry is checked (``validate``); an incomplete, impossible or inconsistent set-up cannot be saved.

Two files per profile, kept apart on purpose
--------------------------------------------
* ``arena.json``  the PHYSICAL arena (size, pits, flipper, in metres). It does not depend on the camera.
* ``clicks.json`` the CAMERA's four corner clicks, in pixels (the existing clicks file).

So when the camera is moved you re-click only the corners (``mode="recalibrate"``) and the hazards stay
exactly where they were; when the arena changes you edit the hazards (``mode="edit"``) and the corners can
stay. A change of arena size gives the set-up a new ``setup_id``; corner clicks made for the old size are
then refused rather than silently reused.

Frames
------
The corners are clicked in the order the picture shows them (near-left, near-right, far-right, far-left),
which defines the SET-UP FRAME: origin at the first corner, +x toward the second, +y toward the fourth.
Hazard positions are never inferred from screen orientation. From the clicked hazards the tool works out
which corner is which and, when the layout is the simulator's (two pits on opposite corners, the flipper
on a third), re-expresses everything in the SIMULATOR'S frame: origin at the plain corner, pits at
(0, H) and (W, 0), flipper at (W, H). That is the frame the trained policy's observation uses, so the saved
file records it as the ACTIVE frame. A layout that cannot be expressed that way stays in the set-up frame and
is flagged ``policy_frame_ok: false``.

Profiles
--------
``live`` and ``home_test`` (and any other name) have separate files. Nothing from one is read for another;
the profile name is written into both files and checked on load.
"""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
import math
import re
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from control_center.tools.arena_design import DesignError, EffectiveDesign, Overrides
from control_center.tools.overlay_core import (
    DEFAULT_CLICKS_PATH, LIVE_PROFILE, PROFILES_DIR, REPO_ROOT, ClickSet, FitQuality, FloorMapping, OverlayConfig,
    PixelModel, ReferencePoint, assess_fit, fit_floor_mapping, profile_points_path, resolve_camera)

SETUP_VERSION = 2                       # 2 adds the "design" block; version-1 files (typed dimensions) still load
READABLE_VERSIONS = (1, 2)
LIVE_ARENA_PATH = REPO_ROOT / "config" / "arena_overlay_arena.json"
CORNER_IDS = ("c1", "c2", "c3", "c4")
CORNER_PREFIX = "arena_corner_"

# Plausibility limits (metres)
MIN_SIDE_M, MAX_SIDE_M = 0.30, 3.00
MIN_PIT_RADIUS_M = 0.03
MAX_PIT_RADIUS_FRACTION = 0.50          # of the shorter side
OUTSIDE_TOLERANCE_M = 0.05              # a click this far outside the arena is still a click on its edge
WALL_SNAP_M = 0.04                      # a flipper vertex this close to a wall is put exactly on it
PIT_CORNER_SNAP_M = 0.06                # a pit centre this close to a corner is put exactly on it
MIN_EDGE_M = 0.02
MIN_FLIPPER_AREA_M2 = 0.001
MAX_FLIPPER_FRACTION = 0.30
MAX_FLIPPER_VERTICES = 8
RIM_CHECK_OK_MM = 15.0
MIN_CORNER_SEPARATION_PX = 30.0

MAX_HAZARD_CLICK_OFFSET_M = 0.30        # a pit/flipper click farther than this from every corner is not a click on a corner feature
WARN_HAZARD_CLICK_OFFSET_M = 0.12       # a pit/flipper click this far from its corner is probably a slip: shown as a warning
MAX_MM_PER_PIXEL = 10.0                 # coarser than this on the far side: the floor position cannot be pinned down well
MAX_SCALE_RATIO = 4.0                   # near side more than this many times sharper than the far side: a very oblique view

DRAWING_SIDE_M = 1.150                  # the competition drawing's size, offered as a default for LIVE only
DRAWING_PIT_RADIUS_M = 0.250

NEAR_LEFT, NEAR_RIGHT, FAR_RIGHT, FAR_LEFT = "near-left", "near-right", "far-right", "far-left"
NEW_CORNER_LABELS = {
    "c1": "NEAR-LEFT corner (bottom-left of the picture)",
    "c2": "NEAR-RIGHT corner (bottom-right of the picture)",
    "c3": "FAR-RIGHT corner (top-right of the picture)",
    "c4": "FAR-LEFT corner (top-left of the picture)",
}


class ArenaSetupError(ValueError):
    """A saved set-up is unusable or does not belong where it was loaded; the message says what to do."""


# ---------------------------------------------------------------------------------------------------
# Paths and profiles

def check_profile_name(profile: str) -> str:
    if not profile or any(ch in profile for ch in "/\\:") or profile.startswith("."):
        raise ArenaSetupError(f"bad profile name {profile!r}")
    return profile


def profile_arena_path(profile: str) -> Path:
    """Where a profile keeps its PHYSICAL arena. ``live`` has its own file; every other name its own folder."""
    check_profile_name(profile)
    if profile == LIVE_PROFILE:
        return LIVE_ARENA_PATH
    return PROFILES_DIR / profile / "arena.json"


def guard_profile_and_config(profile: str, config: OverlayConfig | None) -> None:
    """The real-arena profile must never be paired with the home-test configuration's arena (the A4 patch)."""
    if profile == LIVE_PROFILE and config is not None and config.path is not None and "home_test" in config.path.stem.lower():
        raise ArenaSetupError(
            f"profile '{LIVE_PROFILE}' is the real arena but the overlay config is {config.path.name}, the HOME TEST patch: "
            "its dimensions and hazards are not the competition arena. Use config/arena_overlay.yaml with profile 'live', "
            "or profile 'home_test' with this config.")


# ---------------------------------------------------------------------------------------------------
# Lengths typed by a person

def parse_length(text: str) -> float:
    """'1.15' or '1150' or '1150 mm' or '1.15 m' -> metres. A bare number above 10 is millimetres."""
    cleaned = text.strip().lower().replace(",", ".")
    match = re.fullmatch(r"([0-9]*\.?[0-9]+)\s*(mm|cm|m)?", cleaned)
    if match is None:
        raise ValueError(f"'{text}' is not a length: type 1.15 (metres) or 1150 (millimetres)")
    value, unit = float(match.group(1)), match.group(2)
    if unit == "mm":
        return value / 1000.0
    if unit == "cm":
        return value / 100.0
    if unit == "m":
        return value
    return value / 1000.0 if value > 10.0 else value


# ---------------------------------------------------------------------------------------------------
# Geometry

@dataclass(frozen=True)
class Pit:
    centre_m: tuple[float, float]
    radius_m: float


@dataclass
class Geometry:
    """The arena in the SET-UP frame (origin = corner c1, +x toward c2, +y toward c4), metres."""
    width_m: float
    height_m: float
    pits: list[Pit] = field(default_factory=list)
    flipper_m: list[tuple[float, float]] = field(default_factory=list)

    def corners(self) -> list[tuple[float, float]]:
        w, h = self.width_m, self.height_m
        return [(0.0, 0.0), (w, 0.0), (w, h), (0.0, h)]


def _dist(a: Iterable[float], b: Iterable[float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def polygon_area(points: list[tuple[float, float]]) -> float:
    """Signed area (positive when counter-clockwise)."""
    total = 0.0
    for (x1, y1), (x2, y2) in zip(points, points[1:] + points[:1]):
        total += x1 * y2 - x2 * y1
    return total / 2.0


def _point_in_polygon(point: tuple[float, float], polygon: list[tuple[float, float]]) -> bool:
    x, y = point
    inside = False
    for (x1, y1), (x2, y2) in zip(polygon, polygon[1:] + polygon[:1]):
        if (y1 > y) != (y2 > y) and x < (x2 - x1) * (y - y1) / (y2 - y1) + x1:
            inside = not inside
    return inside


def _orient(a, b, c) -> float:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _segments_cross(a, b, c, d) -> bool:
    """Proper crossing of segment ab with segment cd (touching at a shared end does not count)."""
    o1, o2, o3, o4 = _orient(a, b, c), _orient(a, b, d), _orient(c, d, a), _orient(c, d, b)
    return (o1 * o2 < 0.0) and (o3 * o4 < 0.0)


def polygon_is_simple(points: list[tuple[float, float]]) -> bool:
    count = len(points)
    for i in range(count):
        for j in range(i + 1, count):
            if j == i + 1 or (i == 0 and j == count - 1):
                continue
            if _segments_cross(points[i], points[(i + 1) % count], points[j], points[(j + 1) % count]):
                return False
    return True


@dataclass
class CanonicalFrame:
    """The simulator's arena frame, found from the hazards: origin at the plain corner."""
    available: bool
    reason: str = ""
    origin_index: int = 0                 # which set-up corner (0..3) is the plain corner
    width_m: float = 0.0
    height_m: float = 0.0
    corner_names: dict[str, str] = field(default_factory=dict)       # c1.. -> "plain" / "pit_1" / "pit_2" / "flipper"
    _origin: tuple[float, float] = (0.0, 0.0)
    _e1: tuple[float, float] = (1.0, 0.0)
    _e2: tuple[float, float] = (0.0, 1.0)

    def transform(self, point: tuple[float, float]) -> tuple[float, float]:
        dx, dy = point[0] - self._origin[0], point[1] - self._origin[1]
        return (dx * self._e1[0] + dy * self._e1[1], dx * self._e2[0] + dy * self._e2[1])


def corner_roles(geometry: Geometry) -> tuple[dict[str, str], str]:
    """Which set-up corner holds which feature, from the clicked hazards. Returns (roles, problem)."""
    corners = geometry.corners()
    if len(geometry.pits) != 2 or len(geometry.flipper_m) < 3:
        return {}, "pits or flipper not defined yet"
    pit_corner = [min(range(4), key=lambda i: _dist(corners[i], pit.centre_m)) for pit in geometry.pits]
    if pit_corner[0] == pit_corner[1] or (pit_corner[0] - pit_corner[1]) % 4 != 2:
        return {}, "the two pits are not on opposite corners"
    centroid = (sum(p[0] for p in geometry.flipper_m) / len(geometry.flipper_m),
                sum(p[1] for p in geometry.flipper_m) / len(geometry.flipper_m))
    flipper_corner = min(range(4), key=lambda i: _dist(corners[i], centroid))
    if flipper_corner in pit_corner:
        return {}, "the flipper is on a pit corner"
    if _dist(corners[flipper_corner], centroid) > 0.45 * min(geometry.width_m, geometry.height_m):
        return {}, "the flipper is not in a corner"
    plain_corner = (flipper_corner + 2) % 4
    roles = {CORNER_IDS[plain_corner]: "plain", CORNER_IDS[flipper_corner]: "flipper",
             CORNER_IDS[pit_corner[0]]: "pit_1", CORNER_IDS[pit_corner[1]]: "pit_2"}
    return roles, ""


def canonical_frame(geometry: Geometry) -> CanonicalFrame:
    roles, problem = corner_roles(geometry)
    if problem:
        return CanonicalFrame(False, f"{problem}: not the simulator's layout")
    corners = geometry.corners()
    plain = next(i for i, cid in enumerate(CORNER_IDS) if roles[cid] == "plain")
    origin = corners[plain]
    toward_x, toward_y = corners[(plain + 1) % 4], corners[(plain - 1) % 4]
    width, height = _dist(origin, toward_x), _dist(origin, toward_y)
    e1 = ((toward_x[0] - origin[0]) / width, (toward_x[1] - origin[1]) / width)
    e2 = ((toward_y[0] - origin[0]) / height, (toward_y[1] - origin[1]) / height)
    return CanonicalFrame(True, "", plain, width, height, roles, origin, e1, e2)


def to_canonical_geometry(geometry: Geometry, frame: CanonicalFrame, sort_pits: bool = False) -> Geometry:
    """The geometry in the simulator frame. ``sort_pits`` puts the pit at (0, H) first and the one at (W, 0)
    second, the order of ``config/sim.yaml`` (used for design-based set-ups, where pit 1/2 are click names)."""
    pits = [Pit(tuple(round(v, 6) for v in frame.transform(pit.centre_m)), pit.radius_m) for pit in geometry.pits]
    if sort_pits:
        pits.sort(key=lambda pit: pit.centre_m[0])
    return Geometry(frame.width_m, frame.height_m, pits, [tuple(round(v, 6) for v in frame.transform(p)) for p in geometry.flipper_m])


def quadrant_label(point: tuple[float, float], geometry: Geometry) -> str:
    """'bottom-left' etc. of a set-up-frame point, for the (advisory) hint check only."""
    horizontal = "left" if point[0] < geometry.width_m / 2.0 else "right"
    vertical = "bottom" if point[1] < geometry.height_m / 2.0 else "top"
    return f"{vertical}-{horizontal}"


# ---------------------------------------------------------------------------------------------------
# Validation

@dataclass
class Validation:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    geometry: Geometry | None = None          # the checked, possibly snapped, geometry (None if there are errors)

    @property
    def ok(self) -> bool:
        return not self.errors


def validate_geometry(geometry: Geometry, *, expect_pits: int = 2) -> Validation:
    """Reject incomplete, impossible or inconsistent geometry. Returns a (possibly snapped) copy when it is fine."""
    result = Validation()
    err, warn = result.errors, result.warnings
    w, h = geometry.width_m, geometry.height_m
    for name, value in (("width", w), ("height", h)):
        if not (isinstance(value, (int, float)) and math.isfinite(value)):
            err.append(f"arena {name} is not a number")
        elif not MIN_SIDE_M <= value <= MAX_SIDE_M:
            err.append(f"arena {name} {value:.3f} m is outside the plausible {MIN_SIDE_M:g} to {MAX_SIDE_M:g} m")
    if err:
        return result
    if len(geometry.pits) != expect_pits:
        err.append(f"{expect_pits} pits are required, {len(geometry.pits)} defined")
    if len(geometry.flipper_m) < 3:
        err.append(f"the flipper needs at least 3 vertices, {len(geometry.flipper_m)} defined")
    elif len(geometry.flipper_m) > MAX_FLIPPER_VERTICES:
        err.append(f"the flipper has {len(geometry.flipper_m)} vertices; at most {MAX_FLIPPER_VERTICES}")
    if err:
        return result

    corners = geometry.corners()
    pits: list[Pit] = []
    for index, pit in enumerate(geometry.pits, start=1):
        x, y = pit.centre_m
        r = pit.radius_m
        if not all(math.isfinite(v) for v in (x, y, r)):
            err.append(f"pit {index}: position or radius is not a number")
            continue
        if not (-OUTSIDE_TOLERANCE_M <= x <= w + OUTSIDE_TOLERANCE_M and -OUTSIDE_TOLERANCE_M <= y <= h + OUTSIDE_TOLERANCE_M):
            err.append(f"pit {index}: centre ({x:.3f}, {y:.3f}) m is outside the {w:.3f} x {h:.3f} m arena")
            continue
        limit = MAX_PIT_RADIUS_FRACTION * min(w, h)
        if not MIN_PIT_RADIUS_M <= r <= limit:
            err.append(f"pit {index}: radius {r * 1000:.0f} mm is outside {MIN_PIT_RADIUS_M * 1000:.0f} to {limit * 1000:.0f} mm")
            continue
        nearest = min(corners, key=lambda c: _dist(c, (x, y)))
        gap = _dist(nearest, (x, y))
        if 1e-9 < gap <= PIT_CORNER_SNAP_M:
            warn.append(f"pit {index}: centre was {gap * 1000:.0f} mm from an arena corner and was put exactly on it "
                        "(a quarter-circle pit is centred on its corner)")
            x, y = nearest
        pits.append(Pit((x, y), r))
    if err:
        return result
    for i in range(len(pits)):
        for j in range(i + 1, len(pits)):
            if _dist(pits[i].centre_m, pits[j].centre_m) < pits[i].radius_m + pits[j].radius_m:
                err.append(f"pits {i + 1} and {j + 1} overlap")

    flipper: list[tuple[float, float]] = []
    snapped = 0
    for index, (x, y) in enumerate(geometry.flipper_m, start=1):
        if not (math.isfinite(x) and math.isfinite(y)):
            err.append(f"flipper vertex {index} is not a number")
            continue
        if not (-OUTSIDE_TOLERANCE_M <= x <= w + OUTSIDE_TOLERANCE_M and -OUTSIDE_TOLERANCE_M <= y <= h + OUTSIDE_TOLERANCE_M):
            err.append(f"flipper vertex {index} ({x:.3f}, {y:.3f}) m is outside the arena")
            continue
        nx, ny = x, y
        if abs(x) <= WALL_SNAP_M:
            nx = 0.0
        elif abs(x - w) <= WALL_SNAP_M:
            nx = w
        if abs(y) <= WALL_SNAP_M:
            ny = 0.0
        elif abs(y - h) <= WALL_SNAP_M:
            ny = h
        if (nx, ny) != (x, y):
            snapped += 1
        flipper.append((nx, ny))
    if err:
        return result
    if snapped:
        warn.append(f"{snapped} flipper vertex(es) within {WALL_SNAP_M * 1000:.0f} mm of a wall were put exactly on it")
    for a, b in zip(flipper, flipper[1:] + flipper[:1]):
        if _dist(a, b) < MIN_EDGE_M:
            err.append(f"two flipper vertices are only {_dist(a, b) * 1000:.0f} mm apart (a duplicate click?)")
    if err:
        return result
    if not polygon_is_simple(flipper):
        err.append("the flipper outline crosses itself: click its vertices in order around the shape")
        return result
    area = polygon_area(flipper)
    if abs(area) < MIN_FLIPPER_AREA_M2:
        err.append(f"the flipper's area is {abs(area) * 1e4:.1f} cm2: its vertices are almost in a line")
    elif abs(area) > MAX_FLIPPER_FRACTION * w * h:
        err.append(f"the flipper covers {100 * abs(area) / (w * h):.0f}% of the arena, which is implausible")
    if err:
        return result
    if area < 0:
        flipper.reverse()                      # stored counter-clockwise
    # The flipper must not sit in a pit.
    xs, ys = [p[0] for p in flipper], [p[1] for p in flipper]
    for gx in np.linspace(min(xs), max(xs), 40):
        for gy in np.linspace(min(ys), max(ys), 40):
            if _point_in_polygon((float(gx), float(gy)), flipper):
                for index, pit in enumerate(pits, start=1):
                    if _dist((gx, gy), pit.centre_m) < pit.radius_m - 0.01:
                        err.append(f"the flipper overlaps pit {index}")
                        return result
    if not err:
        result.geometry = Geometry(w, h, pits, flipper)
    return result


# ---------------------------------------------------------------------------------------------------
# The saved arena (camera independent)

def setup_id_for(width_m: float, height_m: float) -> str:
    """Names the corner geometry the camera's corner clicks were made for."""
    key = f"{round(width_m * 1000)}x{round(height_m * 1000)}|{'/'.join(CORNER_IDS)}"
    return hashlib.sha1(key.encode("ascii")).hexdigest()[:10]


@dataclass
class ArenaSetup:
    profile: str
    geometry: Geometry                          # set-up frame
    canonical: CanonicalFrame
    dimensions_source: str = "entered"          # "entered", "confirmed default", "command line"
    warnings: list[str] = field(default_factory=list)
    hints: dict[str, str] = field(default_factory=dict)
    saved_at: str = ""
    rim_check_mm: list[float] = field(default_factory=list)
    setup_id: str = ""
    design: dict[str, Any] | None = None        # EffectiveDesign.to_dict() when the dimensions came from the design
    click_checks: dict[str, Any] = field(default_factory=dict)   # how far the hazard clicks were from the design positions (mm)

    @property
    def design_used(self) -> bool:
        return bool(self.design and self.design.get("used"))

    def __post_init__(self) -> None:
        if not self.setup_id:
            self.setup_id = setup_id_for(self.geometry.width_m, self.geometry.height_m)

    # ---- the frame everything downstream uses
    @property
    def policy_frame_ok(self) -> bool:
        return self.canonical.available

    def active_geometry(self) -> Geometry:
        return (to_canonical_geometry(self.geometry, self.canonical, sort_pits=self.design_used)
                if self.canonical.available else self.geometry)

    def active_corner_coords(self) -> dict[str, tuple[float, float]]:
        corners = dict(zip(CORNER_IDS, self.geometry.corners()))
        if not self.canonical.available:
            return corners
        return {cid: tuple(round(v, 6) for v in self.canonical.transform(point)) for cid, point in corners.items()}

    def corner_references(self) -> list[ReferencePoint]:
        names = self.canonical.corner_names
        return [ReferencePoint(f"{CORNER_PREFIX}{cid}", xy, "fit", True,
                               f"arena corner {cid}" + (f" ({names[cid].replace('_', ' ')} corner)" if cid in names else ""))
                for cid, xy in self.active_corner_coords().items()]

    def arena_dict(self) -> dict[str, Any]:
        """A ``config.arena`` block for the existing overlay, tracker and shadow tools."""
        geometry = self.active_geometry()
        return {"measured": True, "width_m": geometry.width_m, "height_m": geometry.height_m,
                "pit_radius_m": max(p.radius_m for p in geometry.pits),
                "pit_radii_m": [p.radius_m for p in geometry.pits],
                "pit_centres_m": [list(p.centre_m) for p in geometry.pits],
                "flipper_vertices_m": [list(v) for v in geometry.flipper_m],
                "frame": "simulator" if self.canonical.available else "setup",
                "policy_frame_ok": self.canonical.available, "source": "arena setup file"}

    # ---- file form
    def to_json(self) -> dict[str, Any]:
        def block(geometry: Geometry) -> dict[str, Any]:
            return {"width_m": round(geometry.width_m, 6), "height_m": round(geometry.height_m, 6),
                    "pits": [{"name": f"pit_{i}", "centre_m": [round(v, 6) for v in p.centre_m], "radius_m": round(p.radius_m, 6)}
                             for i, p in enumerate(geometry.pits, start=1)],
                    "flipper_vertices_m": [[round(v, 6) for v in p] for p in geometry.flipper_m]}

        canonical: dict[str, Any] = {"available": self.canonical.available, "reason": self.canonical.reason}
        if self.canonical.available:
            canonical.update({"origin_corner": CORNER_IDS[self.canonical.origin_index], "corner_roles": self.canonical.corner_names,
                              **block(self.active_geometry())})
        return {"kind": "arena_setup", "version": SETUP_VERSION, "profile": self.profile, "setup_id": self.setup_id,
                "saved_at": self.saved_at or datetime.now(timezone.utc).isoformat(),
                "dimensions_source": self.dimensions_source,
                "frame_note": ("setup_frame: origin = corner c1 (first corner clicked), +x toward c2, +y toward c4, as the picture "
                               "showed them. canonical_frame: the simulator's frame (origin at the plain corner, pits at (0,H) and "
                               "(W,0), flipper at (W,H)), derived from the clicked hazards, never from screen orientation."),
                "active_frame": "canonical" if self.canonical.available else "setup",
                "policy_frame_ok": self.canonical.available,
                "corner_ids": list(CORNER_IDS), "setup_frame": block(self.geometry), "canonical_frame": canonical,
                "hints_given": self.hints, "warnings": self.warnings, "rim_check_mm": [round(v, 2) for v in self.rim_check_mm],
                "design": self.design if self.design_used else {"used": False}, "click_checks": self.click_checks,
                "measured": True, "units": "metres"}

    @classmethod
    def from_json(cls, data: dict[str, Any], expect_profile: str | None = None) -> "ArenaSetup":
        if data.get("kind") != "arena_setup":
            raise ArenaSetupError("not an arena set-up file")
        if int(data.get("version", 0)) not in READABLE_VERSIONS:
            raise ArenaSetupError(f"arena set-up file version {data.get('version')} is not supported (expected one of {READABLE_VERSIONS})")
        profile = str(data.get("profile", ""))
        if expect_profile is not None and profile != expect_profile:
            raise ArenaSetupError(f"this arena file belongs to profile '{profile}', not '{expect_profile}': profiles never share arena files")
        try:
            block = data["setup_frame"]
            geometry = Geometry(float(block["width_m"]), float(block["height_m"]),
                                [Pit((float(p["centre_m"][0]), float(p["centre_m"][1])), float(p["radius_m"])) for p in block["pits"]],
                                [(float(v[0]), float(v[1])) for v in block["flipper_vertices_m"]])
        except (KeyError, TypeError, ValueError, IndexError) as error:
            raise ArenaSetupError(f"arena set-up file is incomplete or damaged ({type(error).__name__}: {error})") from None
        checked = validate_geometry(geometry)
        if not checked.ok:
            raise ArenaSetupError("saved arena geometry is not valid: " + "; ".join(checked.errors))
        setup = cls(profile, checked.geometry, canonical_frame(checked.geometry), str(data.get("dimensions_source", "entered")),
                    list(data.get("warnings") or []), dict(data.get("hints_given") or {}), str(data.get("saved_at", "")),
                    [float(v) for v in data.get("rim_check_mm") or []])
        design = data.get("design")
        if isinstance(design, dict) and design.get("used"):
            setup.design = design
        if isinstance(data.get("click_checks"), dict):
            setup.click_checks = data["click_checks"]
        if data.get("setup_id") and data["setup_id"] != setup.setup_id:
            raise ArenaSetupError("arena set-up file is inconsistent (its setup_id does not match its dimensions)")
        return setup

    def summary_lines(self) -> list[str]:
        g = self.active_geometry()
        lines = [f"profile {self.profile}: {g.width_m:.3f} x {g.height_m:.3f} m ({self.dimensions_source})"]
        if self.design_used and self.design.get("overrides"):
            lines.append("advanced override in use: " + ", ".join(f"{k} {v}" for k, v in self.design["overrides"].items() if k != "reason"))
        for i, pit in enumerate(g.pits, start=1):
            lines.append(f"pit {i}: centre ({pit.centre_m[0]:.3f}, {pit.centre_m[1]:.3f}) m, radius {pit.radius_m * 1000:.0f} mm")
        lines.append("flipper: " + ", ".join(f"({x:.3f}, {y:.3f})" for x, y in g.flipper_m))
        lines.append("frame: " + ("simulator frame (origin at the plain corner)" if self.canonical.available
                                  else f"SET-UP frame, NOT the simulator's layout ({self.canonical.reason})"))
        return lines


def load_arena_setup(profile: str, path: Path | str | None = None) -> ArenaSetup | None:
    """The saved arena for ``profile``, or None when none was ever saved. A damaged file raises."""
    target = Path(path) if path else profile_arena_path(profile)
    if not target.exists():
        return None
    try:
        data = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ArenaSetupError(f"{target} is unreadable: {error}") from None
    return ArenaSetup.from_json(data, expect_profile=profile)


def _backup(path: Path) -> Path | None:
    if not path.exists():
        return None
    stamp = datetime.now().strftime(".bak-%Y%m%d-%H%M%S")
    target = path.with_name(path.name + stamp)
    counter = 1
    while target.exists():
        target = path.with_name(path.name + f"{stamp}-{counter}")
        counter += 1
    shutil.copy2(path, target)
    return target


@dataclass
class SaveReport:
    arena_path: Path | None = None
    clicks_path: Path | None = None
    arena_backup: Path | None = None
    clicks_backup: Path | None = None
    wrote_arena: bool = False

    def lines(self) -> list[str]:
        out = []
        if self.wrote_arena and self.arena_path:
            out.append(f"arena geometry saved: {self.arena_path}" + (f" (previous kept as {self.arena_backup.name})" if self.arena_backup else ""))
        elif self.arena_path:
            out.append(f"arena geometry left unchanged: {self.arena_path}")
        if self.clicks_path:
            out.append(f"camera corners saved: {self.clicks_path}" + (f" (previous kept as {self.clicks_backup.name})" if self.clicks_backup else ""))
        return out


def corner_clickset(profile: str, setup: ArenaSetup, corners_px: dict[str, tuple[float, float]], image_size: tuple[int, int],
                    previous: ClickSet | None = None, source: str = "") -> ClickSet:
    meta: dict[str, Any] = {"profile_kind": "live", "profile": profile, "arena_setup_id": setup.setup_id,
                            "arena_corner_clicks": True, "source": source}
    if previous is not None and previous.lens.active:
        meta["lens"] = previous.meta.get("lens")
    return ClickSet({f"{CORNER_PREFIX}{cid}": (float(u), float(v)) for cid, (u, v) in corners_px.items()},
                    {}, (int(image_size[0]), int(image_size[1])), meta)


def save_setup(profile: str, setup: ArenaSetup, clicks: ClickSet, *, write_arena: bool = True,
               arena_path: Path | str | None = None, clicks_path: Path | str | None = None) -> SaveReport:
    """Write the arena (optional) and the camera corners. Old files are kept as .bak copies. Both files are
    written to temporary names first, so a failure leaves the previous pair untouched."""
    check_profile_name(profile)
    if setup.profile != profile:
        raise ArenaSetupError(f"set-up belongs to profile '{setup.profile}', refusing to save it as '{profile}'")
    if clicks.meta.get("arena_setup_id") != setup.setup_id:
        raise ArenaSetupError("the corner clicks were made for a different arena size than this set-up")
    arena_target = Path(arena_path) if arena_path else profile_arena_path(profile)
    clicks_target = Path(clicks_path) if clicks_path else profile_points_path(profile)
    report = SaveReport(arena_target, clicks_target, wrote_arena=write_arena)
    setup.saved_at = datetime.now(timezone.utc).isoformat()
    arena_tmp = arena_target.with_name(arena_target.name + ".tmp")
    clicks_tmp = clicks_target.with_name(clicks_target.name + ".tmp")
    from control_center.tools.overlay_core import save_clicks
    try:
        arena_target.parent.mkdir(parents=True, exist_ok=True)
        clicks_target.parent.mkdir(parents=True, exist_ok=True)
        if write_arena:
            arena_tmp.write_text(json.dumps(setup.to_json(), indent=2), encoding="utf-8")
        save_clicks(clicks_tmp, clicks, source=str(clicks.meta.get("source", "")))
        if write_arena:
            report.arena_backup = _backup(arena_target)
            arena_tmp.replace(arena_target)
        report.clicks_backup = _backup(clicks_target)
        clicks_tmp.replace(clicks_target)
    finally:
        for temp in (arena_tmp, clicks_tmp):
            if temp.exists():
                temp.unlink()
    return report


# ---------------------------------------------------------------------------------------------------
# Feeding the saved arena to the existing tools

def with_arena_setup(config: OverlayConfig, profile: str, clicks: ClickSet | None = None,
                     arena_path: Path | str | None = None) -> OverlayConfig:
    """``config`` with the profile's saved arena applied, or ``config`` unchanged when the profile has none.

    The returned config's arena, pit radii and reference points come from the saved set-up and from nothing
    else, so a profile cannot pick up another profile's (or the A4 test patch's) dimensions or hazards.
    Raises ``ArenaSetupError`` when the saved set-up is damaged, belongs to another profile, or when the
    camera corner clicks were made for a different arena size.
    """
    guard_profile_and_config(profile, config)
    setup = load_arena_setup(profile, arena_path)
    if setup is None:
        return config
    if clicks is not None and clicks.points and any(name.startswith(CORNER_PREFIX) for name in clicks.points):
        saved = clicks.meta.get("arena_setup_id")
        if saved != setup.setup_id:
            raise ArenaSetupError(
                "the saved camera corner clicks were made for a different arena size than the saved arena. "
                "Re-click the four corners (ARENA tab > Recalibrate camera).")
        owner = clicks.meta.get("profile")
        if owner and owner != profile:
            raise ArenaSetupError(f"the corner clicks belong to profile '{owner}', not '{profile}'")
    placeholders = [item for item in config.placeholders if not item.startswith(("arena geometry", "reference point"))]
    tags = [item for item in config.placeholder_tags if item != "arena" and not item.endswith("ref points")]
    return dataclasses.replace(config, arena=setup.arena_dict(), references=setup.corner_references(),
                               placeholders=placeholders, placeholder_tags=tags)


# ---------------------------------------------------------------------------------------------------
# The click-by-click session

POINT, NUMBER = "point", "number"


@dataclass
class Slot:
    key: str
    kind: str                # POINT or NUMBER
    label: str
    required: bool = True


@dataclass
class _Located:
    """A hazard point: clicked on the image (pixel) or loaded from a file (metres, locked until re-clicked)."""
    pixel: tuple[float, float] | None = None
    metres: tuple[float, float] | None = None       # set-up frame

    @property
    def placed(self) -> bool:
        return self.pixel is not None or self.metres is not None

    @property
    def locked(self) -> bool:
        return self.pixel is None and self.metres is not None


@dataclass
class Solution:
    """Everything the screen needs after each change."""
    complete: bool = False
    mapping: FloorMapping = field(default_factory=FloorMapping)
    quality: FitQuality = field(default_factory=FitQuality)
    geometry: Geometry | None = None                 # unchecked, as clicked (set-up frame)
    validation: Validation | None = None
    setup: ArenaSetup | None = None                  # present only when validation passed
    problems: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    rim_check_mm: list[float] = field(default_factory=list)
    level: str = "insufficient"                      # "ok", "weak", "insufficient"
    accuracy: list[str] = field(default_factory=list)       # plain statements about how well the picture pins the floor down
    unreliable: list[str] = field(default_factory=list)     # reasons the mapping should not be trusted (also folded into level)
    click_checks: dict[str, Any] = field(default_factory=dict)


MODES = ("new", "recalibrate", "edit")


class ArenaSetupSession:
    """One calibration conversation. Pure state: feed it clicks and numbers, ask it what to do next."""

    def __init__(self, profile: str, mode: str = "new", saved: ArenaSetup | None = None, saved_clicks: ClickSet | None = None,
                 image_size: tuple[int, int] | None = None, default_size_m: tuple[float, float] | None = None,
                 default_radius_m: float | None = None, hints: dict[str, str] | None = None,
                 design: EffectiveDesign | None = None) -> None:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        check_profile_name(profile)
        if mode != "new" and saved is None:
            raise ArenaSetupError(f"nothing saved for profile '{profile}' to {'recalibrate' if mode == 'recalibrate' else 'edit'}: run a full set-up first")
        if saved is not None and saved.profile != profile:
            raise ArenaSetupError("saved set-up belongs to a different profile")
        self.profile, self.mode, self.saved, self.saved_clicks = profile, mode, saved, saved_clicks
        self.image_size = image_size
        self.default_size_m, self.default_radius_m = default_size_m, default_radius_m
        self.hints = dict(hints or {})
        # With a design the dimensions, pit radius and flipper size are KNOWN: the operator only clicks where things are.
        self.design = design
        self.design_flow = design is not None and mode != "recalibrate"
        self.corners: dict[str, tuple[float, float] | None] = {cid: None for cid in CORNER_IDS}
        self.width_m: float | None = None
        self.height_m: float | None = None
        self.dimensions_source = "entered"
        self.pit_centres = [_Located(), _Located()]
        self.pit_radii: list[float | None] = [None, None]
        self.rim_points: list[tuple[float, float] | None] = [None, None]
        self.flipper: list[_Located] = []
        self.flipper_closed = False
        self._history: list[tuple[str, Any]] = []
        self.current: str = "c1"
        self.message = ""
        if self.design_flow:
            self._apply_design()
        if saved is not None:
            self._load_saved(saved, saved_clicks if mode == "edit" else None)
        self._seek()

    def _apply_design(self) -> None:
        self.width_m, self.height_m = self.design.width_m, self.design.height_m
        self.pit_radii = [self.design.pit_radius_m, self.design.pit_radius_m]
        self.dimensions_source = "design" if not self.design.overrides.any else "design with override"

    def set_overrides(self, overrides: Overrides) -> None:
        """Advanced: a measured difference from the design. Raises ``DesignError`` when implausible. The clicked
        points stay; only the sizes change, so the picture-to-floor mapping is refitted by the next ``solve``."""
        if not self.design_flow:
            raise DesignError("overrides apply to a design-based set-up only")
        self.design = self.design.with_overrides(overrides)
        self._apply_design()

    # ---- loading
    def _load_saved(self, saved: ArenaSetup, clicks: ClickSet | None) -> None:
        g = saved.geometry
        if not self.design_flow:
            self.width_m, self.height_m = g.width_m, g.height_m
            self.dimensions_source = saved.dimensions_source
        for i, pit in enumerate(g.pits):
            self.pit_centres[i] = _Located(metres=pit.centre_m)
            if not self.design_flow:
                self.pit_radii[i] = pit.radius_m
        if self.design_flow:
            count = len(g.flipper_m)
            self.flipper = [_Located(metres=(sum(p[0] for p in g.flipper_m) / count, sum(p[1] for p in g.flipper_m) / count))]
        else:
            self.flipper = [_Located(metres=p) for p in g.flipper_m]
        self.flipper_closed = True
        if clicks is not None:
            for cid in CORNER_IDS:
                point = clicks.points.get(f"{CORNER_PREFIX}{cid}")
                if point is not None and clicks.meta.get("arena_setup_id") == saved.setup_id:
                    self.corners[cid] = (float(point[0]), float(point[1]))
            if clicks.image_size and self.image_size is None:
                self.image_size = clicks.image_size

    # ---- slots
    def slots(self) -> list[Slot]:
        slots = [Slot(cid, POINT, self.corner_label(cid)) for cid in CORNER_IDS]
        if self.mode == "recalibrate":
            return slots
        if self.design_flow:
            slots += [Slot("pit1_centre", POINT, self.hazard_label("pit_1", "centre")),
                      Slot("pit2_centre", POINT, self.hazard_label("pit_2", "centre")),
                      Slot("flipper1", POINT, "the FLIPPER: click its middle (its corner and direction are worked out for you)"),
                      Slot("rim1", POINT, "OPTIONAL check: a point on the RIM of pit 1, at floor level", required=False),
                      Slot("rim2", POINT, "OPTIONAL check: a point on the RIM of pit 2, at floor level", required=False)]
            return slots
        slots += [Slot("width", NUMBER, "arena WIDTH (distance from the first to the second corner)"),
                  Slot("height", NUMBER, "arena HEIGHT (distance from the first to the last corner)"),
                  Slot("pit1_centre", POINT, self.hazard_label("pit_1", "centre")), Slot("pit1_radius", NUMBER, "radius of PIT 1"),
                  Slot("pit2_centre", POINT, self.hazard_label("pit_2", "centre")), Slot("pit2_radius", NUMBER, "radius of PIT 2")]
        count = max(3, len(self.flipper) + (0 if self.flipper_closed else 1))
        slots += [Slot(f"flipper{i + 1}", POINT, f"flipper vertex {i + 1}", required=i < 3) for i in range(min(count, MAX_FLIPPER_VERTICES))]
        slots += [Slot("rim1", POINT, "OPTIONAL check: a point on the RIM of pit 1, at floor level", required=False),
                  Slot("rim2", POINT, "OPTIONAL check: a point on the RIM of pit 2, at floor level", required=False)]
        return slots

    def corner_label(self, cid: str) -> str:
        if self.mode == "new" or self.saved is None:
            return NEW_CORNER_LABELS[cid]
        names = self.saved.canonical.corner_names if self.saved.canonical.available else {}
        role = names.get(cid)
        shown = {"plain": "PLAIN corner (no pit, no flipper)", "flipper": "FLIPPER corner", "pit_1": "PIT 1 corner", "pit_2": "PIT 2 corner"}.get(role, "")
        x, y = self.saved.geometry.corners()[CORNER_IDS.index(cid)]
        return f"{shown or 'corner ' + cid} at set-up coordinates ({x:.3f}, {y:.3f}) m"

    def hazard_label(self, which: str, what: str) -> str:
        hint = self.hints.get(which)
        number = which[-1]
        return f"CENTRE of PIT {number}" + (f" (expected {hint} in this camera view)" if hint else "") + ", at floor level"

    def slot(self, key: str) -> Slot | None:
        return next((s for s in self.slots() if s.key == key), None)

    def value_of(self, key: str) -> Any:
        if key in CORNER_IDS:
            return self.corners[key]
        if key == "width":
            return self.width_m
        if key == "height":
            return self.height_m
        if key == "pit1_centre":
            return self.pit_centres[0].pixel or self.pit_centres[0].metres
        if key == "pit2_centre":
            return self.pit_centres[1].pixel or self.pit_centres[1].metres
        if key == "pit1_radius":
            return self.pit_radii[0]
        if key == "pit2_radius":
            return self.pit_radii[1]
        if key.startswith("flipper"):
            index = int(key[7:]) - 1
            return self.flipper[index].pixel or self.flipper[index].metres if index < len(self.flipper) else None
        if key in ("rim1", "rim2"):
            return self.rim_points[int(key[3]) - 1]
        return None

    def _seek(self) -> None:
        for slot in self.slots():
            if slot.required and self.value_of(slot.key) is None:
                self.current = slot.key
                return
        self.current = "done"

    @property
    def done(self) -> bool:
        return self.current == "done"

    def goto(self, key: str) -> bool:
        """Jump to any slot of this mode to place it again (the correction of an individual point)."""
        if self.slot(key) is None:
            return False
        self.current = key
        return True

    def current_slot(self) -> Slot | None:
        return self.slot(self.current)

    def prompt(self) -> str:
        slot = self.current_slot()
        if slot is None:
            return "All required items are placed. Check the overlay, then Save."
        if slot.kind == POINT:
            verb = "RE-CLICK" if self.value_of(slot.key) is not None else "CLICK"
            return f"{verb}: {slot.label}"
        default = ""
        if slot.key in ("width", "height") and self.default_size_m:
            default = f" (Enter accepts {self.default_size_m[0 if slot.key == 'width' else 1]:.3f} m)"
        if slot.key.endswith("radius") and self.default_radius_m:
            default = f" (Enter accepts {self.default_radius_m * 1000:.0f} mm)"
        return f"TYPE the {slot.label} in metres or millimetres{default}, then Enter"

    # ---- editing
    def _push(self, what: str, previous: Any) -> None:
        self._history.append((what, previous))

    def click(self, u: float, v: float) -> str:
        slot = self.current_slot()
        if slot is None or slot.kind != POINT:
            return ""
        key = slot.key
        previous = copy.deepcopy(self._state_of(key))
        if key in CORNER_IDS:
            self.corners[key] = (float(u), float(v))
        elif key in ("pit1_centre", "pit2_centre"):
            self.pit_centres[int(key[3]) - 1] = _Located(pixel=(float(u), float(v)))
        elif key.startswith("flipper"):
            index = int(key[7:]) - 1
            while len(self.flipper) <= index:
                self.flipper.append(_Located())
            self.flipper[index] = _Located(pixel=(float(u), float(v)))
        elif key in ("rim1", "rim2"):
            self.rim_points[int(key[3]) - 1] = (float(u), float(v))
        self._push(key, previous)
        self._advance(key)
        return key

    def _state_of(self, key: str) -> Any:
        if key.startswith("flipper"):
            return ("flipper", copy.deepcopy(self.flipper), self.flipper_closed)
        return ("value", self.value_of(key)) if key not in ("pit1_centre", "pit2_centre") else ("pit", copy.deepcopy(self.pit_centres[int(key[3]) - 1]))

    def _advance(self, key: str) -> None:
        if key.startswith("flipper"):
            index = int(key[7:]) - 1
            if index >= 2 and index == len(self.flipper) - 1 and not self.flipper_closed:
                self.current = f"flipper{index + 2}" if index + 2 <= MAX_FLIPPER_VERTICES else "done"
                self.message = "Click the next flipper vertex, or press Finish flipper"
                return
        self._seek_after(key)

    def _seek_after(self, key: str) -> None:
        order = [s.key for s in self.slots()]
        start = order.index(key) + 1 if key in order else 0
        for slot in order[start:] + order[:start]:
            spec = self.slot(slot)
            if spec is not None and spec.required and self.value_of(slot) is None:
                self.current = slot
                return
        self.current = "done"

    def finish_flipper(self) -> bool:
        placed = [loc for loc in self.flipper if loc.placed]
        if len(placed) < 3:
            self.message = "The flipper needs at least 3 vertices"
            return False
        self.flipper = placed
        self.flipper_closed = True
        self._seek()
        return True

    def reopen_flipper(self) -> None:
        """Allow another flipper vertex to be added to an already finished flipper."""
        self.flipper_closed = False
        if len(self.flipper) < MAX_FLIPPER_VERTICES:
            self.current = f"flipper{len(self.flipper) + 1}"

    def submit_number(self, key: str, value: float) -> None:
        slot = self.slot(key)
        if slot is None or slot.kind != NUMBER:
            raise ValueError(f"{key} is not a number slot")
        if not (isinstance(value, (int, float)) and math.isfinite(value) and value > 0):
            raise ValueError("the value must be a positive number")
        previous = self.value_of(key)
        if key == "width":
            self.width_m = float(value)
            self.dimensions_source = "entered"
        elif key == "height":
            self.height_m = float(value)
            self.dimensions_source = "entered"
        elif key == "pit1_radius":
            self.pit_radii[0] = float(value)
        elif key == "pit2_radius":
            self.pit_radii[1] = float(value)
        self._push(key, ("value", previous))
        self._seek_after(key)

    def accept_default(self) -> bool:
        """Enter on an empty number box: confirm the offered default. Returns False when there is none."""
        slot = self.current_slot()
        if slot is None or slot.kind != NUMBER:
            return False
        if slot.key in ("width", "height") and self.default_size_m:
            self.submit_number(slot.key, self.default_size_m[0 if slot.key == "width" else 1])
            self.dimensions_source = "confirmed default"
            return True
        if slot.key.endswith("radius") and self.default_radius_m:
            self.submit_number(slot.key, self.default_radius_m)
            return True
        return False

    def undo(self) -> str | None:
        if not self._history:
            return None
        key, previous = self._history.pop()
        kind = previous[0]
        if key in CORNER_IDS:
            self.corners[key] = previous[1]
        elif key in ("pit1_centre", "pit2_centre"):
            self.pit_centres[int(key[3]) - 1] = previous[1]
        elif key.startswith("flipper"):
            self.flipper, self.flipper_closed = previous[1], previous[2]
        elif key in ("rim1", "rim2"):
            self.rim_points[int(key[3]) - 1] = previous[1]
        elif key == "width":
            self.width_m = previous[1]
        elif key == "height":
            self.height_m = previous[1]
        elif key == "pit1_radius":
            self.pit_radii[0] = previous[1]
        elif key == "pit2_radius":
            self.pit_radii[1] = previous[1]
        self.current = key if self.slot(key) is not None else "done"
        return key

    def remove_flipper_vertex(self, number: int) -> bool:
        index = number - 1
        if not 0 <= index < len(self.flipper):
            return False
        self._push(f"flipper{number}", ("flipper", copy.deepcopy(self.flipper), self.flipper_closed))
        del self.flipper[index]
        self.flipper_closed = len(self.flipper) >= 3 and self.flipper_closed
        self._seek()
        return True

    def reset(self) -> None:
        """Start again. In recalibrate mode the saved hazards stay; in new/edit mode everything is cleared
        (edit mode reloads the saved arena, so Reset there means 'discard my changes')."""
        profile, mode, size, dsize, dradius, hints = self.profile, self.mode, self.image_size, self.default_size_m, self.default_radius_m, self.hints
        saved = self.saved if mode != "new" else None
        self.__init__(profile, mode, saved, self.saved_clicks if mode != "new" else None, size, dsize, dradius, hints)
        self.message = "reset"

    # ---- progress that survives a restart
    def to_draft(self) -> dict[str, Any]:
        def located(item: _Located) -> dict[str, Any]:
            return {"pixel": item.pixel, "metres": item.metres}
        return {"profile": self.profile, "mode": self.mode, "image_size": list(self.image_size) if self.image_size else None,
                "corners": {cid: point for cid, point in self.corners.items()}, "width_m": self.width_m, "height_m": self.height_m,
                "dimensions_source": self.dimensions_source, "pits": [located(p) for p in self.pit_centres],
                "radii": list(self.pit_radii), "rim": list(self.rim_points), "flipper": [located(f) for f in self.flipper],
                "flipper_closed": self.flipper_closed, "setup_id": None if self.saved is None else self.saved.setup_id,
                "design_flow": self.design_flow, "overrides": self.design.overrides.to_dict() if self.design_flow else None}

    def restore_draft(self, data: dict[str, Any]) -> bool:
        """Continue an unfinished set-up. Refused (False) for another profile or mode, or when the saved arena changed."""
        if data.get("profile") != self.profile or data.get("mode") != self.mode:
            return False
        if data.get("setup_id") != (None if self.saved is None else self.saved.setup_id):
            return False
        if bool(data.get("design_flow")) != self.design_flow:
            return False
        if self.design_flow and data.get("overrides"):
            try:
                self.set_overrides(Overrides.from_dict(data["overrides"]))
            except DesignError:
                return False

        def pair(value: Any) -> tuple[float, float] | None:
            return None if value is None else (float(value[0]), float(value[1]))

        def located(item: dict[str, Any]) -> _Located:
            return _Located(pair(item.get("pixel")), pair(item.get("metres")))
        try:
            self.corners = {cid: pair((data.get("corners") or {}).get(cid)) for cid in CORNER_IDS}
            if not self.design_flow:
                self.width_m = None if data.get("width_m") is None else float(data["width_m"])
                self.height_m = None if data.get("height_m") is None else float(data["height_m"])
            if not self.design_flow:
                self.dimensions_source = str(data.get("dimensions_source", "entered"))
            self.pit_centres = [located(item) for item in data["pits"]][:2]
            if not self.design_flow:
                self.pit_radii = [None if r is None else float(r) for r in data["radii"]][:2]
            self.rim_points = [pair(r) for r in data.get("rim") or [None, None]][:2]
            self.flipper = [located(item) for item in data["flipper"]]
            self.flipper_closed = bool(data.get("flipper_closed"))
            size = data.get("image_size")
            self.image_size = None if not size else (int(size[0]), int(size[1]))
        except (KeyError, TypeError, ValueError, IndexError):
            return False
        self._history.clear()
        self._seek()
        return True

    # ---- solving
    def corner_pixels(self) -> dict[str, tuple[float, float]] | None:
        if any(point is None for point in self.corners.values()):
            return None
        return {cid: tuple(point) for cid, point in self.corners.items()}

    def _dimensions(self) -> tuple[float, float] | None:
        if self.width_m is None or self.height_m is None:
            return None
        return self.width_m, self.height_m

    def solve(self, image_size: tuple[int, int] | None = None, pixel_model: PixelModel | None = None,
              config: OverlayConfig | None = None) -> Solution:
        out = Solution()
        size = image_size or self.image_size
        dims = self._dimensions()
        pixels = self.corner_pixels()
        if pixels is None:
            out.problems.append("click all four arena corners")
            return out
        if dims is None:
            out.problems.append("enter the arena width and height")
            return out
        for name, value in zip(("width", "height"), dims):
            if not MIN_SIDE_M <= value <= MAX_SIDE_M:
                out.problems.append(f"arena {name} {value:.3f} m is outside the plausible {MIN_SIDE_M:g} to {MAX_SIDE_M:g} m: retype it")
        if out.problems:
            return out
        if size is None:
            out.problems.append("no camera frame size is known yet")
            return out
        side_px = [math.hypot(a[0] - b[0], a[1] - b[1]) for a, b in zip(pixels.values(), list(pixels.values())[1:] + [pixels["c1"]])]
        if min(side_px) < MIN_CORNER_SEPARATION_PX:
            out.problems.append("two corner clicks are almost on the same spot: the corners must be the four different corners")
            return out
        if self.design_flow:
            staged = self._design_stage(out, pixels, size, pixel_model)
            if staged is None:
                return out
            dims, pits, flipper = staged
        else:
            geometry_corners = Geometry(dims[0], dims[1]).corners()
            references = [ReferencePoint(f"{CORNER_PREFIX}{cid}", xy, "fit", True, "") for cid, xy in zip(CORNER_IDS, geometry_corners)]
            clicks = {f"{CORNER_PREFIX}{cid}": pixels[cid] for cid in CORNER_IDS}
            mapping = fit_floor_mapping(references, clicks, tuple(size), pixel_model, max_fit_residual_mm=30.0)
            out.mapping = mapping
            if not mapping.valid:
                out.problems.append("corner mapping failed: " + mapping.reason + (" Click the corners in the order shown." if "mirror" in mapping.reason else ""))
                return out
            arena_for_quality = {"width_m": dims[0], "height_m": dims[1]}
            quality = assess_fit(references, clicks, {}, mapping, arena_for_quality)
            quality.reasons = [r for r in quality.reasons if not r.startswith("no check point")]
            out.quality = quality
            # Convert everything the user placed into the set-up frame, in metres.
            def metres(located: _Located) -> tuple[float, float] | None:
                if located.pixel is not None:
                    x, y = mapping.pixel_to_floor([located.pixel])[0]
                    return (float(x), float(y))
                return located.metres

            pits, missing = [], []
            for i in range(2):
                centre, radius = metres(self.pit_centres[i]), self.pit_radii[i]
                if centre is None:
                    missing.append(f"click the centre of pit {i + 1}")
                if radius is None:
                    missing.append(f"enter the radius of pit {i + 1}")
                if centre is not None and radius is not None:
                    pits.append(Pit(centre, radius))
            flipper = [m for m in (metres(loc) for loc in self.flipper) if m is not None]
            if len(flipper) < 3:
                missing.append("click at least 3 flipper vertices")
            if not self.flipper_closed and len(flipper) >= 3:
                missing.append("press Finish flipper")
            out.problems.extend(missing)
            if self.mode == "recalibrate" and self.saved is not None:
                pits, flipper = list(self.saved.geometry.pits), list(self.saved.geometry.flipper_m)
                out.problems = []
        mapping, quality = out.mapping, out.quality
        out.geometry = Geometry(dims[0], dims[1], pits, flipper)
        if out.problems:
            out.level = "insufficient"
            return out
        validation = validate_geometry(out.geometry)
        out.validation = validation
        out.warnings = list(validation.warnings)
        if not validation.ok:
            out.problems.extend(validation.errors)
            return out
        # Optional held-out checks: a clicked rim point should be one radius from the pit centre.
        for i, rim in enumerate(self.rim_points):
            if rim is not None and i < len(validation.geometry.pits):
                x, y = mapping.pixel_to_floor([rim])[0]
                pit = validation.geometry.pits[i]
                out.rim_check_mm.append(1000.0 * abs(_dist((x, y), pit.centre_m) - pit.radius_m))
        # Soft consistency checks and hints.
        if config is not None:
            camera = resolve_camera(config, mapping)
            if camera.position_m is None:
                out.warnings.append("the corners and the entered size do not give a plausible camera position "
                                    f"({camera.detail}); check the size you entered and the corner clicks")
        for which, hint, located in (("pit_1", self.hints.get("pit_1"), validation.geometry.pits[0].centre_m),
                                     ("pit_2", self.hints.get("pit_2"), validation.geometry.pits[1].centre_m)):
            if hint and self.mode == "new" and quadrant_label(located, validation.geometry) != hint:
                out.warnings.append(f"{which.replace('_', ' ')} was expected {hint} in this view but you placed it {quadrant_label(located, validation.geometry)}")
        centroid = (sum(p[0] for p in validation.geometry.flipper_m) / len(validation.geometry.flipper_m),
                    sum(p[1] for p in validation.geometry.flipper_m) / len(validation.geometry.flipper_m))
        flipper_hint = self.hints.get("flipper")
        if flipper_hint and self.mode == "new" and quadrant_label(centroid, validation.geometry) != flipper_hint:
            out.warnings.append(f"the flipper was expected {flipper_hint} in this view but you placed it {quadrant_label(centroid, validation.geometry)}")
        canonical = canonical_frame(validation.geometry)
        if not canonical.available:
            out.warnings.append(f"layout is not the simulator's ({canonical.reason}); positions stay in the set-up frame and the "
                                "policy's observation cannot use them")
        elif abs(canonical.width_m - DRAWING_SIDE_M) > 0.05 or abs(canonical.height_m - DRAWING_SIDE_M) > 0.05:
            out.warnings.append(f"arena is {canonical.width_m:.3f} x {canonical.height_m:.3f} m; the simulator and the policy were trained on "
                                f"{DRAWING_SIDE_M:.3f} x {DRAWING_SIDE_M:.3f} m")
        setup = ArenaSetup(self.profile, validation.geometry, canonical, self.dimensions_source, list(out.warnings), dict(self.hints),
                           rim_check_mm=list(out.rim_check_mm), design=self.design.to_dict() if self.design_flow else None,
                           click_checks=dict(out.click_checks))
        out.setup, out.complete = setup, True
        reasons = list(quality.reasons)
        if not out.rim_check_mm:
            reasons.append("no independent check: the four corners always fit exactly. Click a rim point of each pit (optional) "
                           "or check the overlay against the floor by eye")
        elif max(out.rim_check_mm) > RIM_CHECK_OK_MM:
            reasons.append(f"a pit-rim check is off by {max(out.rim_check_mm):.0f} mm (limit {RIM_CHECK_OK_MM:.0f} mm)")
        reasons.extend(out.unreliable)
        quality.reasons = reasons
        out.level = "ok" if not reasons and not out.warnings else "weak"
        quality.level = out.level
        return out

    # ---- design flow: dimensions known, the operator only says where things are
    def _fit(self, dims: tuple[float, float], pixels: dict[str, tuple[float, float]], size: tuple[int, int], pixel_model: PixelModel | None):
        references = [ReferencePoint(f"{CORNER_PREFIX}{cid}", xy, "fit", True, "") for cid, xy in zip(CORNER_IDS, Geometry(*dims).corners())]
        clicks = {f"{CORNER_PREFIX}{cid}": pixels[cid] for cid in CORNER_IDS}
        mapping = fit_floor_mapping(references, clicks, tuple(size), pixel_model, max_fit_residual_mm=30.0)
        return references, clicks, mapping

    def _design_stage(self, out: Solution, pixels: dict[str, tuple[float, float]], size: tuple[int, int],
                      pixel_model: PixelModel | None) -> tuple[tuple[float, float], list[Pit], list[tuple[float, float]]] | None:
        """Corners -> mapping; pit and flipper clicks -> which corner each belongs to; hazards from the design.
        Returns (dims, pits, flipper) or None with ``out.problems`` filled."""
        design = self.design
        dims = (design.width_m, design.height_m)
        placement = None
        for _attempt in range(2):
            references, clicks, mapping = self._fit(dims, pixels, size, pixel_model)
            out.mapping = mapping
            if not mapping.valid:
                out.problems.append("corner mapping failed: " + mapping.reason + (" Click the corners in the order shown." if "mirror" in mapping.reason else ""))
                return None
            placement = self._place_hazards(mapping, dims)
            if placement["problems"]:
                break
            # A rectangle that is not square has two possible side assignments; the plain corner decides which is right.
            plain = (placement["flipper_corner"] + 2) % 4
            wanted = (design.width_m, design.height_m) if plain % 2 == 0 else (design.height_m, design.width_m)
            if all(abs(a - b) < 1e-9 for a, b in zip(wanted, dims)):
                break
            dims = wanted
        quality = assess_fit(references, clicks, {}, mapping, {"width_m": dims[0], "height_m": dims[1]})
        quality.reasons = [r for r in quality.reasons if not r.startswith("no check point")]
        out.quality = quality
        out.geometry = Geometry(dims[0], dims[1], [], [])
        out.accuracy, out.unreliable = self._accuracy(mapping, dims, size)
        if placement["problems"]:
            out.problems.extend(placement["problems"])
            out.level = "insufficient"
            return None
        out.warnings.extend(placement["warnings"])
        out.click_checks = placement["checks"]
        return dims, placement["pits"], placement["flipper"]

    def _place_hazards(self, mapping: FloorMapping, dims: tuple[float, float]) -> dict[str, Any]:
        design = self.design
        corners = Geometry(*dims).corners()
        problems: list[str] = []
        warnings: list[str] = []
        checks: dict[str, Any] = {}

        def metres(located: _Located) -> tuple[float, float] | None:
            if located.pixel is not None:
                x, y = mapping.pixel_to_floor([located.pixel])[0]
                return (float(x), float(y))
            return located.metres

        def nearest(point: tuple[float, float]) -> tuple[int, float]:
            index = min(range(4), key=lambda i: _dist(corners[i], point))
            return index, _dist(corners[index], point)

        pit_corner: list[int | None] = []
        for number in (1, 2):
            point = metres(self.pit_centres[number - 1])
            if point is None:
                problems.append(f"click the centre of pit {number}")
                pit_corner.append(None)
                continue
            index, offset = nearest(point)
            checks[f"pit_{number}_click_offset_mm"] = round(offset * 1000.0, 1)
            checks[f"pit_{number}_corner"] = CORNER_IDS[index]
            if offset > MAX_HAZARD_CLICK_OFFSET_M:
                problems.append(f"the pit {number} click is {offset * 1000:.0f} mm from the nearest arena corner: click the middle of the pit's "
                                "quarter circle, where the two walls would meet at floor level")
            elif offset > WARN_HAZARD_CLICK_OFFSET_M:
                warnings.append(f"pit {number}: the click was {offset * 1000:.0f} mm from corner {CORNER_IDS[index]}, where the design puts the pit "
                                "centre; it was placed on the corner. Re-click it if you meant another spot")
            pit_corner.append(index)
        if None not in pit_corner:
            if pit_corner[0] == pit_corner[1]:
                problems.append(f"both pit clicks are nearest the same corner ({CORNER_IDS[pit_corner[0]]}): the two pits are on opposite corners")
            elif (pit_corner[0] - pit_corner[1]) % 4 != 2:
                problems.append(f"pit 1 is nearest corner {CORNER_IDS[pit_corner[0]]} and pit 2 nearest {CORNER_IDS[pit_corner[1]]}: those are "
                                "next to each other, but the design has the pits on OPPOSITE corners")
        flipper_corner = None
        flipper_point = metres(self.flipper[0]) if self.flipper else None
        if flipper_point is None:
            problems.append("click the flipper")
        else:
            flipper_corner, offset = nearest(flipper_point)
            checks["flipper_click_offset_mm"] = round(offset * 1000.0, 1)
            checks["flipper_corner"] = CORNER_IDS[flipper_corner]
            if offset > max(MAX_HAZARD_CLICK_OFFSET_M, 1.2 * design.flipper_leg_m):
                problems.append(f"the flipper click is {offset * 1000:.0f} mm from the nearest arena corner: click on the flipper, the chamfered corner")
                flipper_corner = None
            elif None not in pit_corner and flipper_corner in pit_corner:
                problems.append(f"the flipper click is nearest corner {CORNER_IDS[flipper_corner]}, which is a pit corner: click on the flipper")
                flipper_corner = None
        result: dict[str, Any] = {"problems": problems, "warnings": warnings, "checks": checks, "pits": [], "flipper": [],
                                  "flipper_corner": flipper_corner}
        if problems or flipper_corner is None:
            return result
        result["pits"] = [Pit(corners[k], design.pit_radius_m) for k in pit_corner]
        corner = corners[flipper_corner]
        legs = []
        for neighbour in (corners[(flipper_corner + 1) % 4], corners[(flipper_corner - 1) % 4]):
            length = _dist(corner, neighbour)
            legs.append((corner[0] + design.flipper_leg_m * (neighbour[0] - corner[0]) / length,
                         corner[1] + design.flipper_leg_m * (neighbour[1] - corner[1]) / length))
        result["flipper"] = [legs[0], corner, legs[1]]
        return result

    @staticmethod
    def _accuracy(mapping: FloorMapping, dims: tuple[float, float], size: tuple[int, int]) -> tuple[list[str], list[str]]:
        """How sharply the picture pins the floor down, from the mapping itself (nothing is measured on the robot)."""
        corners = Geometry(*dims).corners()
        pixels = np.asarray(mapping.floor_to_pixel(corners), float)
        if not np.isfinite(pixels).all():
            return [], ["the mapping gives no finite picture positions for the arena corners"]
        sides = [(math.hypot(*(pixels[i] - pixels[(i + 1) % 4])), _dist(corners[i], corners[(i + 1) % 4])) for i in range(4)]
        per_metre = [px / m for px, m in sides if m > 0]
        if not per_metre or min(per_metre) <= 0:
            return [], ["the arena corners collapse to a line in the picture"]
        coarse, fine = 1000.0 / min(per_metre), 1000.0 / max(per_metre)
        lines = [f"picture scale: {fine:.1f} mm per pixel on the near side, {coarse:.1f} mm per pixel on the far side"]
        reasons = []
        if coarse > MAX_MM_PER_PIXEL:
            reasons.append(f"the far side is only {1000.0 / coarse:.0f} pixels per metre ({coarse:.1f} mm per pixel): positions there are coarse. "
                           "Move the camera closer or raise the resolution")
        if max(per_metre) / min(per_metre) > MAX_SCALE_RATIO:
            reasons.append(f"a very oblique view (the near side is {max(per_metre) / min(per_metre):.1f} times sharper than the far side): "
                           "raise the camera or move it more toward the middle")
        return lines, reasons
