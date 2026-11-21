"""Click-selected tracker for one UNMARKED opponent. Read-only vision: frames in, an estimate out.

The user clicks the opponent once. From then on the tracker follows that object by

  * what stands out from the floor near where the object should be (no empty-floor reference
    picture is needed: the floor colour is the median of the floor pixels around the object),
  * an appearance model taken at the click (coarse hue/brightness histogram and floor-plane size),
  * a constant-velocity motion model with a gate on speed.

The installed OpenCV (4.13.0, no contrib) has no CSRT/KCF tracker, so none is used: everything
here is plain colour segmentation and is deterministic on the same frames.

It answers with one of four states and never guesses:

  TRACKED       exactly one acceptable match; a position is reported
  AMBIGUOUS     no position: nothing acceptable this frame (occluded or merged), more than one
                similar candidate within reach, or the object is touching / within contact
                distance of the own tagged robot
  LOST          no single clear match for longer than ``hold_s`` (nothing acceptable, or rivals),
                or the object left the arena. It is NEVER picked up again
                automatically: ``select()`` must be called again
  NOT_SELECTED  nothing has been clicked (or the selection was cleared)

The own robot is always excluded: its footprint (from the tag pose and ``own_radius_m``) is cut out
of the picture before anything is looked for, and a match that touches it is never TRACKED.

The reported position is the centre of the object's picture region projected onto the floor plane
with the floor mapping. It is APPROXIMATE: not height-corrected (a raised surface reads farther
from the camera than it is), the same approximation the blob tracker makes.

Nothing here has been run on a real camera. The thresholds are starting values.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from enum import Enum
import math
from typing import Any

import cv2
import numpy as np

from control_center.tools.blob_tracker import floor_mask
from control_center.tools.overlay_core import UNKNOWN, FloorMapping
from control_center.tools.stream_adapter import OK, OpponentEstimate

POSITION_QUALITY = "approximate (centre of the object's picture region on the floor plane, not height-corrected)"


class OpponentState(str, Enum):
    NOT_SELECTED = "not_selected"
    TRACKED = "tracked"
    AMBIGUOUS = "ambiguous"
    UNCERTAIN = "uncertain"       # drag-selected tracker only (roi_tracker.py): confidence too low to give a position
    LOST = "lost"


NOT_SELECTED, TRACKED, AMBIGUOUS, LOST, UNCERTAIN = (OpponentState.NOT_SELECTED, OpponentState.TRACKED,
                                                     OpponentState.AMBIGUOUS, OpponentState.LOST, OpponentState.UNCERTAIN)


@dataclass(frozen=True)
class OpponentSettings:
    difference_threshold: float = 30.0      # largest BGR channel difference from the floor colour
    min_size_m: float = 0.04                # the object's longer side on the floor plane, at least
    max_size_m: float = 0.28                # ... and at most (0.06-0.20 m robot + oblique stretch)
    own_radius_m: float = 0.09              # own chassis outline radius, cut out of the picture
    contact_margin_m: float = 0.03          # centre closer than own radius + this = contact
    max_speed_m_s: float = 2.5
    gate_floor_m: float = 0.06              # match gate = this + max speed x time since last match
    max_gate_m: float = 0.25
    hold_s: float = 0.5                     # no match / unresolved identity for longer = LOST
    min_appearance: float = 0.6             # histogram similarity 0..1 a match must reach
    appearance_margin: float = 0.15         # a rival this close in similarity makes it ambiguous
    min_size_ratio: float = 0.5             # floor-plane area relative to the selected object
    max_size_ratio: float = 1.7             # (two like-sized objects merged into one region read about 2x)
    arena_margin_m: float = 0.03
    model_update: float = 0.05              # how fast the appearance model follows the object
    own_memory_s: float = 1.0               # how long a no-longer-current own pose still masks

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "OpponentSettings":
        data = dict(data or {})
        unknown = sorted(set(data) - {f.name for f in fields(cls)})
        if unknown:
            raise ValueError(f"unknown opponent setting(s) {unknown}")
        return cls(**{key: float(value) for key, value in data.items()})


@dataclass
class Candidate:
    pixel: tuple[float, float]
    floor: tuple[float, float]
    area_px: float
    area_m2: float
    box: tuple[int, int, int, int]                  # x, y, w, h in the full picture
    histogram: np.ndarray
    dims_m: tuple[float, float]                     # longer, shorter side on the floor plane
    touches_own: bool
    touches_window: bool
    label: int = 0                                  # component label inside the search window
    appearance: float = 0.0
    size_ratio: float = 0.0
    distance_m: float = 0.0


@dataclass
class Selection:
    ok: bool
    reason: str

    def __bool__(self) -> bool:
        return self.ok


@dataclass
class OpponentTrack:
    state: OpponentState = NOT_SELECTED
    reason: str = "no opponent selected: click it in the camera view"
    xy_m: tuple[float, float] | None = None         # only when TRACKED
    box: tuple[int, int, int, int] | None = None    # picture box of the match, when there is one
    appearance: float | None = None
    candidates: int = 0                             # objects standing out from the floor in reach
    selection_id: int = 0
    tracked_age_s: float | None = None              # since the last TRACKED frame
    position_quality: str = UNKNOWN
    selection_note: str = ""
    # Only the drag-selected tracker (roi_tracker.py) fills these; the click-selected one leaves them None.
    confidence: float | None = None                 # appearance similarity 0..1 (not a probability)
    centre_px: tuple[float, float] | None = None    # centre of the tracked box in the picture
    backend: str = ""                               # which tracker produced the box
    diagnostics: dict[str, Any] | None = None       # optional extras (apparent floor size): never used to accept anything
    note: str = ""

    def to_report(self) -> dict[str, Any]:
        report = self._base_report()
        if self.backend or self.confidence is not None:
            report.update({"confidence": None if self.confidence is None else round(self.confidence, 3), "centre_px": list(self.centre_px) if self.centre_px else None,
                           "backend": self.backend, "diagnostics": self.diagnostics, "note": self.note, "image_based": True})
        return report

    def _base_report(self) -> dict[str, Any]:
        tracked = self.state == TRACKED and self.xy_m is not None
        return {"state": self.state.value, "reason": self.reason,
                "x_m": round(self.xy_m[0], 4) if tracked else None,
                "y_m": round(self.xy_m[1], 4) if tracked else None,
                "box": list(self.box) if self.box else None,
                "appearance": None if self.appearance is None else round(self.appearance, 3),
                "candidates": self.candidates, "selection": self.selection_id,
                "tracked_age_s": None if self.tracked_age_s is None else round(self.tracked_age_s, 4),
                "position_quality": self.position_quality if tracked else None,
                "selection_note": self.selection_note}

    def to_opponent_estimate(self) -> OpponentEstimate:
        if self.state == TRACKED and self.xy_m is not None:
            what = "drag-selected object" if self.backend else "clicked object"
            quality = "image-based, approximate, not height-corrected" if self.backend else POSITION_QUALITY
            return OpponentEstimate(state=OK, reason=OK, xy_m=self.xy_m, track_id=self.selection_id,
                                    candidates=self.candidates,
                                    source=f"{what} #{self.selection_id}, {quality}")
        return OpponentEstimate(reason=f"{'selected' if self.backend else 'clicked'} opponent {self.state.value.upper()}: {self.reason}",
                                candidates=self.candidates)


def colour_histogram(hsv_pixels: np.ndarray) -> np.ndarray:
    """Coarse appearance: 6 brightness bins for grey pixels, 12 hues x 3 brightness for coloured."""
    h = hsv_pixels[:, 0].astype(np.int32)
    s = hsv_pixels[:, 1].astype(np.int32)
    v = hsv_pixels[:, 2].astype(np.int32)
    bins = np.where(s < 50, np.minimum(v // 43, 5), 6 + np.minimum(h * 12 // 180, 11) * 3 + np.minimum(v // 86, 2))
    histogram = np.bincount(bins, minlength=42).astype(np.float64)
    total = histogram.sum()
    return histogram / total if total else histogram


def similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Bhattacharyya coefficient: 1 = the same distribution, 0 = nothing in common."""
    return float(np.sum(np.sqrt(a * b)))


class ClickedOpponentTracker:
    def __init__(self, mapping: FloorMapping, arena: dict[str, Any], settings: OpponentSettings | None = None) -> None:
        self.mapping, self.arena = mapping, arena
        self.settings = settings or OpponentSettings()
        self.state = NOT_SELECTED
        self.reason = OpponentTrack().reason
        self.selection_id = 0
        self.selection_note = ""
        self._mask: np.ndarray | None = None
        self._size: tuple[int, int] | None = None
        self._xy: np.ndarray | None = None              # internal position; reported only when TRACKED
        self._velocity = np.zeros(2)
        self._t: float | None = None                    # time of the last match
        self._histogram: np.ndarray | None = None
        self._area_m2 = 0.0
        self._radius_m = 0.05
        self._floor_colour: np.ndarray | None = None
        self._miss_since: float | None = None           # start of the current stretch without one clear match
        self._tracked_t: float | None = None
        self._own: tuple[float, tuple[float, float], tuple[float, float] | None] | None = None
        self._estimate = OpponentTrack()

    # ---- geometry helpers ----------------------------------------------------------------
    def _prepare(self, frame: np.ndarray) -> str:
        if not self.mapping.valid:
            return f"no valid floor map ({self.mapping.reason})"
        size = (frame.shape[1], frame.shape[0])
        if self._size != size:
            self._mask, self._size = floor_mask(self.mapping, self.arena, size, inset_m=0.0), size
        return ""

    def _in_arena(self, xy: Any) -> bool:
        margin = self.settings.arena_margin_m
        return (np.isfinite(xy).all() and -margin <= xy[0] <= float(self.arena["width_m"]) + margin
                and -margin <= xy[1] <= float(self.arena["height_m"]) + margin)

    def _window(self, centre_xy: Any, radius_m: float) -> tuple[int, int, int, int] | None:
        """Picture box (x0, y0, x1, y1) holding a floor circle and whatever stands on it."""
        angles = np.linspace(0.0, 2.0 * math.pi, 16, endpoint=False)
        ring = np.asarray(centre_xy, float) + radius_m * np.column_stack([np.cos(angles), np.sin(angles)])
        pixels = self.mapping.floor_to_pixel(ring)
        if not np.isfinite(pixels).all():
            return None
        x0, y0 = np.floor(pixels.min(axis=0)).astype(int)
        x1, y1 = np.ceil(pixels.max(axis=0)).astype(int)
        y0 -= (y1 - y0) // 2                                  # things on the floor rise in the picture
        width, height = self._size
        x0, y0, x1, y1 = max(0, int(x0)), max(0, int(y0)), min(width, int(x1) + 1), min(height, int(y1) + 1)
        return (x0, y0, x1, y1) if x1 - x0 >= 8 and y1 - y0 >= 8 else None

    def _own_mask(self, window: tuple[int, int, int, int]) -> np.ndarray:
        x0, y0, x1, y1 = window
        mask = np.zeros((y1 - y0, x1 - x0), np.uint8)
        if self._own is None:
            return mask
        _, xy, seen = self._own
        angles = np.linspace(0.0, 2.0 * math.pi, 24, endpoint=False)
        circle = self.settings.own_radius_m * np.column_stack([np.cos(angles), np.sin(angles)])
        points = [np.asarray(xy) + circle]
        if seen is not None:
            points.append(np.asarray(seen) + circle)       # the raised top surface as the camera sees it
        pixels = self.mapping.floor_to_pixel(np.vstack(points))
        if np.isfinite(pixels).all():
            hull = cv2.convexHull(np.round(pixels - [x0, y0]).astype(np.int32))
            cv2.fillConvexPoly(mask, hull, 255)
        return mask

    def _remember_own(self, own_pose: Any, own_hint_xy: Any, t: float) -> None:
        if own_pose is not None and getattr(own_pose, "ok", False):
            seen = getattr(own_pose, "floor_projection_xy", None)
            self._own = (t, (float(own_pose.x_m), float(own_pose.y_m)), None if seen is None else (float(seen[0]), float(seen[1])))
        elif own_hint_xy is not None:
            self._own = (t, (float(own_hint_xy[0]), float(own_hint_xy[1])), None)
        elif self._own is not None and t - self._own[0] > self.settings.own_memory_s:
            self._own = None

    # ---- what stands out from the floor in a window ----------------------------------------
    def _candidates(self, frame: np.ndarray, window: tuple[int, int, int, int]) -> tuple[list[Candidate], np.ndarray, str]:
        s = self.settings
        x0, y0, x1, y1 = window
        patch = cv2.GaussianBlur(frame[y0:y1, x0:x1], (3, 3), 0)
        on_floor = self._mask[y0:y1, x0:x1] > 0
        own = self._own_mask(window)
        usable = on_floor & (own == 0)
        if int(usable.sum()) >= 200:
            self._floor_colour = np.median(patch[usable][::2], axis=0)
        elif self._floor_colour is None:
            return [], np.zeros(patch.shape[:2], np.int32), "too little floor in view around that point to tell an object from it"
        difference = np.abs(patch.astype(np.int16) - self._floor_colour.astype(np.int16)).max(axis=2)
        foreground = ((difference > s.difference_threshold) & usable).astype(np.uint8) * 255
        foreground = cv2.morphologyEx(foreground, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        gap = max(5, int(round(0.012 * self._size[1])) | 1)
        foreground = cv2.morphologyEx(foreground, cv2.MORPH_CLOSE, np.ones((gap, gap), np.uint8))
        foreground[own > 0] = 0
        count, labels, stats, centroids = cv2.connectedComponentsWithStats(foreground, 8)
        hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
        near_own = cv2.dilate(own, np.ones((7, 7), np.uint8)) > 0
        candidates: list[Candidate] = []
        for index in range(1, count):
            area = float(stats[index, cv2.CC_STAT_AREA])
            if area < 25.0:
                continue
            left, top = int(stats[index, cv2.CC_STAT_LEFT]), int(stats[index, cv2.CC_STAT_TOP])
            wide, tall = int(stats[index, cv2.CC_STAT_WIDTH]), int(stats[index, cv2.CC_STAT_HEIGHT])
            pixel = (float(centroids[index][0]) + x0, float(centroids[index][1]) + y0)
            floor = self.mapping.pixel_to_floor([pixel])[0]
            if not np.isfinite(floor).all():
                continue
            component = labels == index
            east, south = self.mapping.pixel_to_floor([(pixel[0] + 1.0, pixel[1]), (pixel[0], pixel[1] + 1.0)]) - floor
            scale = abs(float(east[0] * south[1] - east[1] * south[0]))
            contours, _ = cv2.findContours(component.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            outline = self.mapping.pixel_to_floor(np.vstack([c.reshape(-1, 2) for c in contours]) + [x0, y0])
            dims = (0.0, 0.0)
            if np.isfinite(outline).all() and len(outline) >= 3:
                (_, _), (a, b), _ = cv2.minAreaRect(outline.astype(np.float32))
                dims = (float(max(a, b)), float(min(a, b)))
            edge = left <= 0 or top <= 0 or left + wide >= x1 - x0 or top + tall >= y1 - y0
            candidates.append(Candidate(
                pixel, (float(floor[0]), float(floor[1])), area, area * scale if math.isfinite(scale) else 0.0,
                (left + x0, top + y0, wide, tall), colour_histogram(hsv[component]), dims,
                bool((component & near_own).any()), edge, index))
        return candidates, labels, ""

    # ---- select / clear --------------------------------------------------------------------
    def clear(self, reason: str = "selection cleared: click the opponent in the camera view") -> None:
        self.state, self.reason = NOT_SELECTED, reason
        self._xy, self._t, self._histogram = None, None, None
        self._velocity = np.zeros(2)
        self._miss_since = self._tracked_t = None
        self._estimate = OpponentTrack(reason=reason, selection_id=self.selection_id, selection_note=self.selection_note)

    def select(self, frame: np.ndarray, pixel_xy: Any, t: float, own_pose: Any = None) -> Selection:
        """Start following the object under ``pixel_xy``. A refused click leaves any current track alone."""
        result = self._select(frame, pixel_xy, t, own_pose)
        self.selection_note = ("selected" if result.ok else "click REFUSED") + f": {result.reason}"
        self._estimate.selection_note = self.selection_note
        return result

    def _select(self, frame: np.ndarray, pixel_xy: Any, t: float, own_pose: Any) -> Selection:
        s = self.settings
        problem = self._prepare(frame)
        if problem:
            return Selection(False, problem)
        u, v = float(pixel_xy[0]), float(pixel_xy[1])
        if not (0 <= u < self._size[0] and 0 <= v < self._size[1]):
            return Selection(False, "the click is outside the camera picture")
        floor = self.mapping.pixel_to_floor([(u, v)])[0]
        if not self._in_arena(floor):
            return Selection(False, "the clicked point is off the arena floor")
        if own_pose is not None:
            self._remember_own(own_pose, None, t)
        window = self._window(floor, s.max_size_m)
        if window is None:
            return Selection(False, "the clicked point is too close to the edge of the picture")
        x0, y0, _, _ = window
        own_checked = self._own is not None
        column, row = int(round(u)) - x0, int(round(v)) - y0
        own_mask = self._own_mask(window)
        if own_checked and 0 <= row < own_mask.shape[0] and 0 <= column < own_mask.shape[1] and own_mask[row, column]:
            return Selection(False, "that is the own (tagged) robot, not the opponent")
        candidates, labels, problem = self._candidates(frame, window)
        if problem:
            return Selection(False, problem)
        label = 0
        if 0 <= row < labels.shape[0] and 0 <= column < labels.shape[1]:
            near = labels[max(0, row - 2):row + 3, max(0, column - 2):column + 3]
            label = int(near[near > 0][0]) if (near > 0).any() else 0
        picked = next((c for c in candidates if label and c.label == label), None)
        if picked is None:
            return Selection(False, "nothing at the clicked point stands out from the floor (empty floor, or too small)")
        longer = picked.dims_m[0]
        if picked.touches_window or longer > s.max_size_m:
            return Selection(False, f"the region under the click is implausibly large for a robot ({1000 * longer:.0f} mm across; "
                                    f"limit {1000 * s.max_size_m:.0f} mm): it may be touching something else")
        if longer < s.min_size_m:
            return Selection(False, f"the region under the click is implausibly small for a robot ({1000 * longer:.0f} mm across; "
                                    f"limit {1000 * s.min_size_m:.0f} mm)")
        if picked.touches_own:
            return Selection(False, "the object under the click is touching the own robot: select it when they are apart")
        if not self._in_arena(picked.floor):
            return Selection(False, "the object under the click is off the arena floor")
        self.selection_id += 1
        self.state, self.reason = TRACKED, "ok"
        self._xy, self._velocity, self._t = np.asarray(picked.floor), np.zeros(2), t
        self._histogram, self._area_m2 = picked.histogram.copy(), picked.area_m2
        self._radius_m = max(0.03, min(0.14, longer / 2.0))
        self._miss_since = None
        self._tracked_t = t
        self._estimate = OpponentTrack(TRACKED, "ok", picked.floor, picked.box, 1.0, len(candidates), self.selection_id,
                                       0.0, POSITION_QUALITY)
        note = f"object {1000 * picked.dims_m[0]:.0f} x {1000 * picked.dims_m[1]:.0f} mm on the floor plane"
        return Selection(True, note + ("" if own_checked else "; own pose unknown at the click, so it was not checked against the own robot"))

    # ---- per frame -------------------------------------------------------------------------
    def update(self, frame: np.ndarray, t: float, own_pose: Any = None, own_hint_xy: Any = None) -> OpponentTrack:
        """Follow the selected object in this frame. ``own_hint_xy``: a no-longer-valid own position, used only to mask."""
        s = self.settings
        self._remember_own(own_pose, own_hint_xy, t)
        if self.state in (NOT_SELECTED, LOST):
            self._estimate = OpponentTrack(self.state, self.reason, selection_id=self.selection_id, selection_note=self.selection_note)
            return self._estimate
        problem = self._prepare(frame)
        if problem:
            return self._lose(f"lost: {problem}")
        if t < self._t:
            raise ValueError("frames must be fed in timestamp order")
        dt = t - self._t
        predicted = self._xy + self._velocity * min(dt, s.hold_s)
        gate = min(s.gate_floor_m + s.max_speed_m_s * dt, s.max_gate_m)
        window = self._window(predicted, gate + 0.75 * s.max_size_m)
        candidates: list[Candidate] = []
        if window is not None:
            candidates, _, _ = self._candidates(frame, window)
        for candidate in candidates:
            candidate.appearance = similarity(candidate.histogram, self._histogram)
            candidate.size_ratio = candidate.area_m2 / self._area_m2 if self._area_m2 > 0 else 0.0
            candidate.distance_m = float(np.linalg.norm(np.asarray(candidate.floor) - predicted))
        reach = [c for c in candidates if c.distance_m <= gate]
        # An object the own robot's cut-out has clipped is allowed to look smaller: it can only keep the
        # internal track alive, because a match that touches the own robot is never reported as a position.
        plausible = [c for c in reach if c.appearance >= s.min_appearance and c.size_ratio <= s.max_size_ratio
                     and (c.size_ratio >= s.min_size_ratio or c.touches_own)]
        age = None if self._tracked_t is None else t - self._tracked_t
        # One clock for every frame without a single, clear match (nothing acceptable, or rivals): it is
        # reset only by such a match, so the unresolved stretch as a whole is what has to stay short.
        unresolved = 0.0 if self._miss_since is None else t - self._miss_since
        if not plausible:
            self._miss_since = t if self._miss_since is None else self._miss_since
            missing = unresolved
            why = "nothing stands out from the floor where it should be"
            if reach:
                nearest = min(reach, key=lambda c: c.distance_m)
                why = (f"the nearest object does not match (appearance {nearest.appearance:.2f}, needs {s.min_appearance:.2f}; "
                       f"size {nearest.size_ratio:.1f}x the selected one, allowed {s.min_size_ratio:g}-{s.max_size_ratio:g}x): "
                       "occluded, merged with something, or a different object")
            if missing > s.hold_s:
                return self._lose(f"no acceptable match for {1000 * missing:.0f} ms ({why}): click the opponent again")
            return self._report(AMBIGUOUS, f"no acceptable match this frame, held for {1000 * missing:.0f} of {1000 * s.hold_s:.0f} ms: {why}",
                                None, len(candidates), age)
        best = max(plausible, key=lambda c: c.appearance - 0.3 * c.distance_m / gate)
        rivals = [c for c in plausible if c is not best and c.appearance >= best.appearance - s.appearance_margin]
        if rivals:
            self._miss_since = t if self._miss_since is None else self._miss_since
            unsure = unresolved
            if unsure > s.hold_s:
                return self._lose(f"{len(rivals) + 1} similar objects within reach for {1000 * unsure:.0f} ms, so which one is the "
                                  "opponent is no longer known: click the opponent again")
            return self._report(AMBIGUOUS, f"{len(rivals) + 1} similar objects within reach (appearance {best.appearance:.2f} vs "
                                           f"{max(c.appearance for c in rivals):.2f}): cannot tell which is the opponent", None, len(candidates), age)
        self._miss_since = None
        if not self._in_arena(np.asarray(best.floor)):
            return self._lose("the object left the arena: click the opponent again if it returns")
        if dt > 1e-6:                                         # follow the match, whether or not it is reported
            velocity = 0.5 * self._velocity + 0.5 * (np.asarray(best.floor) - self._xy) / dt
            speed = float(np.linalg.norm(velocity))
            self._velocity = velocity if speed <= s.max_speed_m_s else velocity * s.max_speed_m_s / speed
        self._xy, self._t = np.asarray(best.floor), t
        if self._own is not None:
            own_xy = np.asarray(self._own[1])
            distance = float(np.linalg.norm(self._xy - own_xy))
            limit = s.own_radius_m + s.contact_margin_m
            if best.touches_own or distance <= limit:
                how = "its outline touches the own robot's" if best.touches_own else f"its centre is {1000 * distance:.0f} mm from the own robot's (limit {1000 * limit:.0f} mm)"
                return self._report(AMBIGUOUS, f"in contact with the own robot ({how}): no position is reported", best, len(candidates), age)
        rate = s.model_update
        self._histogram = (1.0 - rate) * self._histogram + rate * best.histogram
        self._area_m2 = (1.0 - rate) * self._area_m2 + rate * best.area_m2
        self._tracked_t = t
        note = "ok" if len(reach) == 1 else f"ok ({len(reach) - 1} other object(s) in reach do not look like it)"
        return self._report(TRACKED, note, best, len(candidates), 0.0)

    def _report(self, state: OpponentState, reason: str, match: Candidate | None, candidates: int, age: float | None) -> OpponentTrack:
        self.state, self.reason = state, reason
        tracked = state == TRACKED and match is not None
        self._estimate = OpponentTrack(state, reason, match.floor if tracked else None, None if match is None else match.box,
                                       None if match is None else match.appearance, candidates, self.selection_id, age,
                                       POSITION_QUALITY if tracked else UNKNOWN, self.selection_note)
        return self._estimate

    def _lose(self, reason: str) -> OpponentTrack:
        self.state, self.reason = LOST, reason
        self._xy, self._t, self._histogram = None, None, None
        self._velocity = np.zeros(2)
        self._miss_since = None
        self._estimate = OpponentTrack(LOST, reason, selection_id=self.selection_id, selection_note=self.selection_note)
        return self._estimate

    # ---- for the pipeline ------------------------------------------------------------------
    @property
    def estimate(self) -> OpponentTrack:
        return self._estimate

    def to_opponent_estimate(self) -> OpponentEstimate:
        return self._estimate.to_opponent_estimate()

    def to_report(self) -> dict[str, Any]:
        return self._estimate.to_report()
