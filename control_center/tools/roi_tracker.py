"""Free-form opponent tracking: drag a rectangle around ANY object, then follow it. Read-only vision.

The operator drags a box on the camera picture around a phone, a ball, an antweight robot or anything else.
That box is the initial region of interest (ROI). From then on the object is followed from frame to frame by

  * a tracker proposing where the box moved. With OpenCV's CSRT or KCF if the installed build has them
    (``cv2.TrackerCSRT_create`` / ``cv2.legacy``); the build on this PC (4.13, no contrib) has neither, so the
    built-in tracker is used: multi-scale template matching in a search window around the predicted position,
    with a slowly adapting appearance template. (OpenCV's own MIL tracker reports no confidence and drifts;
    Nano, Vit, DaSiamRPN and GOTURN need model files that are not part of this repository.)
  * an appearance CONFIDENCE, computed the same way for either tracker: how well the current box looks like the
    selected object (normalised correlation with the box as first selected and with the slowly adapted
    template). It is a similarity measure, not a probability.

There is NO physical size limit anywhere in this path. A selection is refused only when the rectangle is
empty (smaller than ``min_side_px`` after clipping to the picture), not a number, entirely outside the picture,
or sits on our own tagged robot. The object's apparent size on the floor is reported as a DIAGNOSTIC and never
decides whether anything is accepted.

States (never a guess):

  TRACKED       confidence at or above ``confident``: a position is reported
  UNCERTAIN     confidence between ``floor`` and ``confident``, the box overlaps our own robot, or the object is
                briefly hidden: NO position is reported (the last good one is not shown as current). Recovers by
                itself when the object is seen again within ``uncertain_hold_s``
  LOST          uncertain for longer than ``uncertain_hold_s``. It is NEVER picked up again automatically: drag a
                new box (``select_roi``), which works at any time and replaces the selection
  NOT_SELECTED  nothing selected, or the selection was cleared

The reported position is the centre of the tracked box projected onto the FLOOR PLANE with the floor mapping.
That is an image-based estimate, not a physical measurement: an object raised above the floor appears displaced
away from the camera's nadir, so the position of anything that is not flat on the floor is approximate, and it
says so. Nothing here has been run on a real camera; the thresholds are starting values.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields
from typing import Any, Callable

import cv2
import numpy as np

from control_center.tools.opponent_select import (LOST, NOT_SELECTED, TRACKED, UNCERTAIN, OpponentSettings, OpponentState,
                                                  OpponentTrack, Selection)
from control_center.tools.overlay_core import UNKNOWN, FloorMapping

POSITION_QUALITY = ("image-based: centre of the tracked box projected onto the floor plane. NOT height-corrected and not a "
                    "physical measurement; an object raised above the floor appears displaced away from the camera")
FLAT_TOLERANCE = 40                  # a pixel within this of a flat object's colour (largest channel difference) counts as the object
OWN_OUTLINE_MARGIN = 1.25            # our robot's outline is drawn this much larger than its radius when keeping the opponent off it
DISTINCT_FULL_AT = 0.15              # the best match must beat the runner-up place by this much to count in full ...
DISTINCT_NOT_NEEDED_ABOVE = 0.90     # ... unless the match is nearly perfect
FLAT_OBJECT_STD = 14.0               # the object pixels of a box with background in it: flatter than this = a one-colour object
FLAT_STD = 6.0                       # a template flatter than this is compared by colour difference, not correlation


@dataclass(frozen=True)
class RoiSettings:
    min_side_px: float = 4.0              # an empty rectangle: the ONLY size rule, in picture pixels, never physical
    work_px: float = 72.0                 # longest template side while matching (speed only; larger boxes are shrunk)
    scales: tuple[float, ...] = (0.88, 0.94, 1.0, 1.07, 1.14)    # size change tolerated per frame
    confident: float = 0.60               # similarity needed to be TRACKED
    floor: float = 0.35                   # below this the object is treated as hidden / not found
    uncertain_hold_s: float = 0.8         # UNCERTAIN longer than this becomes LOST
    adapt_rate: float = 0.12              # how fast the template follows the object while it is clearly visible
    adapt_min_confidence: float = 0.75    # ... and never below this similarity
    search_margin_px: float = 16.0        # search window: box grown by this + 0.75 x its size + motion since the last match
    own_overlap_fraction: float = 0.5     # share of the box on our own robot that makes it UNCERTAIN / refuses a selection
    own_radius_m: float = 0.09            # OUR robot's outline radius (not an opponent size); excluded from the opponent
    own_memory_s: float = 1.0             # a no-longer-current own pose still marks where our robot is for this long
    arena_margin_m: float = 0.03

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "RoiSettings":
        data = dict(data or {})
        unknown = sorted(set(data) - {f.name for f in fields(cls)})
        if unknown:
            raise ValueError(f"unknown ROI tracker setting(s) {unknown}")
        return cls(**{key: (tuple(float(v) for v in value) if key == "scales" else float(value)) for key, value in data.items()})


# ---------------------------------------------------------------------------------------------------
# Rectangle checks

def validate_rect(rect: Any, image_size: tuple[int, int], min_side_px: float = 4.0) -> tuple[tuple[float, float, float, float] | None, str]:
    """(clipped x, y, w, h) and a note, or (None, why). Only genuinely invalid rectangles are refused."""
    try:
        x0, y0, x1, y1 = (float(v) for v in (rect[0], rect[1], rect[0] + rect[2], rect[1] + rect[3]))
    except (TypeError, ValueError, IndexError):
        return None, "the rectangle is not four numbers (x, y, width, height)"
    if not all(math.isfinite(v) for v in (x0, y0, x1, y1)):
        return None, "the rectangle has a coordinate that is not a finite number"
    width, height = image_size
    left, right = min(x0, x1), max(x0, x1)
    top, bottom = min(y0, y1), max(y0, y1)
    if right <= 0 or bottom <= 0 or left >= width or top >= height:
        return None, f"the rectangle is outside the {width} x {height} picture"
    cl, ct, cr, cb = max(0.0, left), max(0.0, top), min(float(width), right), min(float(height), bottom)
    note = "" if (cl, ct, cr, cb) == (left, top, right, bottom) else "the part of the rectangle outside the picture was cut off. "
    if cr - cl < min_side_px or cb - ct < min_side_px:
        return None, f"the rectangle is empty: it is {cr - cl:.0f} x {cb - ct:.0f} px, drag a box at least {min_side_px:g} px on each side"
    return (cl, ct, cr - cl, cb - ct), note


# ---------------------------------------------------------------------------------------------------
# Appearance

def _work_size(width: float, height: float, work_px: float) -> tuple[int, int]:
    scale = min(1.0, work_px / max(width, height))
    return max(3, int(round(width * scale))), max(3, int(round(height * scale)))


def _crop(frame: np.ndarray, box: tuple[float, float, float, float]) -> np.ndarray | None:
    h, w = frame.shape[:2]
    x0, y0 = max(0, int(round(box[0]))), max(0, int(round(box[1])))
    x1, y1 = min(w, int(round(box[0] + box[2]))), min(h, int(round(box[1] + box[3])))
    if x1 - x0 < 2 or y1 - y0 < 2:
        return None
    return frame[y0:y1, x0:x1]


def similarity(patch: np.ndarray, template: np.ndarray, flat: bool) -> float:
    """0..1: how much ``patch`` (resized to the template) looks like ``template``."""
    if patch is None:
        return 0.0
    size = (template.shape[1], template.shape[0])
    resized = cv2.resize(patch, size, interpolation=cv2.INTER_AREA)
    method = cv2.TM_SQDIFF_NORMED if flat else cv2.TM_CCOEFF_NORMED
    value = float(np.nan_to_num(cv2.matchTemplate(resized, template, method)[0, 0], nan=0.0, posinf=0.0, neginf=0.0))
    return float(np.clip(1.0 - value if flat else value, 0.0, 1.0))


class TemplateBackend:
    """Built-in tracker: multi-scale template matching around the predicted position."""

    name = "built-in template matching (OpenCV has no CSRT/KCF in this build)"

    def __init__(self, settings: RoiSettings) -> None:
        self.s = settings
        self.template: np.ndarray | None = None
        self.flat = False
        self.scale = 1.0
        self.base_size = (3, 3)
        self.base_box_wh = (1.0, 1.0)

    def init(self, frame: np.ndarray, box: tuple[float, float, float, float], template: np.ndarray, flat: bool,
             colour: np.ndarray | None = None, area: float | None = None) -> None:
        self.template, self.flat = template, flat
        self.base_size = (template.shape[1], template.shape[0])
        self.base_box_wh = (box[2], box[3])
        self.scale = 1.0
        self.last_distinct: float | None = None
        self.mean_colour = template.reshape(-1, template.shape[2]).mean(axis=0) if colour is None else np.asarray(colour, float)
        self.base_area = float(box[2] * box[3]) if area is None else float(area)

    def adapt(self, patch: np.ndarray, rate: float) -> None:
        resized = cv2.resize(patch, self.base_size, interpolation=cv2.INTER_AREA)
        self.template = cv2.addWeighted(self.template, 1.0 - rate, resized, rate, 0.0)

    def _update_flat(self, frame: np.ndarray, predicted: tuple[float, float, float, float], margin: float) -> tuple[float, float, float, float] | None:
        """A one-colour object has nothing for correlation to lock onto, so follow the blob of that colour instead
        (whatever its size): the colour region that overlaps the predicted box the most."""
        h, w = frame.shape[:2]
        cx, cy = predicted[0] + predicted[2] / 2.0, predicted[1] + predicted[3] / 2.0
        x0, y0 = max(0, int(cx - predicted[2] / 2.0 - margin)), max(0, int(cy - predicted[3] / 2.0 - margin))
        x1, y1 = min(w, int(cx + predicted[2] / 2.0 + margin) + 1), min(h, int(cy + predicted[3] / 2.0 + margin) + 1)
        if x1 - x0 < 4 or y1 - y0 < 4:
            return None
        crop = frame[y0:y1, x0:x1].astype(np.int16)
        mask = (np.abs(crop - self.mean_colour.astype(np.int16)).max(axis=2) < FLAT_TOLERANCE).astype(np.uint8)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        best, best_overlap = None, 0.0
        px0, py0, px1, py1 = predicted[0] - x0, predicted[1] - y0, predicted[0] + predicted[2] - x0, predicted[1] + predicted[3] - y0
        for label in range(1, count):
            bx, by, bw, bh, area = (int(v) for v in stats[label])
            if area < 0.2 * self.base_area * self.scale ** 2 or area < 6:
                continue
            overlap = max(0.0, min(px1, bx + bw) - max(px0, bx)) * max(0.0, min(py1, by + bh) - max(py0, by))
            if overlap > best_overlap:
                best, best_overlap = (bx, by, bw, bh), overlap
        if best is None:
            return None
        self.scale = float(np.clip(math.sqrt(best[2] * best[3] / max(self.base_area, 1.0)), 0.2, 5.0))
        return (x0 + best[0], y0 + best[1], float(best[2]), float(best[3]))

    def update(self, frame: np.ndarray, predicted: tuple[float, float, float, float], margin: float,
               freeze_scale: bool = False) -> tuple[float, float, float, float] | None:
        if self.flat:
            return self._update_flat(frame, predicted, margin)
        h, w = frame.shape[:2]
        cx, cy = predicted[0] + predicted[2] / 2.0, predicted[1] + predicted[3] / 2.0
        half_w, half_h = predicted[2] / 2.0 + margin, predicted[3] / 2.0 + margin
        x0, y0 = max(0, int(cx - half_w)), max(0, int(cy - half_h))
        x1, y1 = min(w, int(cx + half_w) + 1), min(h, int(cy + half_h) + 1)
        if x1 - x0 < 4 or y1 - y0 < 4:
            return None
        ds = self.base_size[0] / self.base_box_wh[0]                       # picture pixels -> working pixels at scale 1
        crop = cv2.resize(frame[y0:y1, x0:x1], None, fx=ds, fy=ds, interpolation=cv2.INTER_AREA) if ds < 1.0 else frame[y0:y1, x0:x1]
        ds = ds if ds < 1.0 else 1.0
        best: tuple[float, tuple[int, int], tuple[int, int], float, np.ndarray] | None = None
        for factor in ((1.0,) if freeze_scale else self.s.scales):          # while the object is not verified the size is not allowed to wander
            scale = float(np.clip(self.scale * factor, 0.4, 2.5))
            size = (max(3, int(round(self.base_size[0] * scale))), max(3, int(round(self.base_size[1] * scale))))
            if size[0] > crop.shape[1] or size[1] > crop.shape[0]:
                continue
            template = cv2.resize(self.template, size, interpolation=cv2.INTER_AREA)
            response = cv2.matchTemplate(crop, template, cv2.TM_SQDIFF_NORMED if self.flat else cv2.TM_CCOEFF_NORMED)
            response = np.nan_to_num(response, nan=-1.0, posinf=-1.0, neginf=-1.0)
            if self.flat:
                low, _, location, _ = cv2.minMaxLoc(response)
                score = 1.0 - low
            else:
                _, high, _, location = cv2.minMaxLoc(response)
                score = high
            if best is None or score > best[0]:
                best = (float(score), location, size, scale, response)
        if best is None:
            return None
        peak, location, size, scale, response = best
        # How much better is the best place than every other place in the window? A real match stands out; a patch of
        # plain floor that merely resembles a box that held some floor does not.
        masked = response.copy()
        masked[max(0, location[1] - size[1] // 2):location[1] + size[1] // 2 + 1, max(0, location[0] - size[0] // 2):location[0] + size[0] // 2 + 1] = -1.0
        runner_up = float(masked.max()) if masked.size and masked.max() > -1.0 else 0.0
        self.last_distinct = float(peak - max(runner_up, 0.0))
        self.scale = scale
        return (x0 + location[0] / ds, y0 + location[1] / ds, size[0] / ds, size[1] / ds)


class CvTrackerBackend:
    """OpenCV's own CSRT or KCF tracker, when this build has one. ``factory`` is injectable for tests."""

    def __init__(self, factory: Callable[[], Any], name: str) -> None:
        self.factory, self.name = factory, name
        self.tracker: Any = None

    def init(self, frame: np.ndarray, box: tuple[float, float, float, float], template: np.ndarray, flat: bool,
             colour: np.ndarray | None = None, area: float | None = None) -> None:
        self.tracker = self.factory()
        self.tracker.init(frame, tuple(int(round(v)) for v in box))

    def adapt(self, patch: np.ndarray, rate: float) -> None:
        pass

    def update(self, frame: np.ndarray, predicted: tuple[float, float, float, float], margin: float,
               freeze_scale: bool = False) -> tuple[float, float, float, float] | None:
        ok, box = self.tracker.update(frame)
        return tuple(float(v) for v in box) if ok else None


def available_cv_backends() -> list[tuple[str, Callable[[], Any]]]:
    """CSRT and KCF in whichever place this OpenCV build keeps them. Empty on a build without contrib."""
    found = []
    for label, attribute in (("CSRT", "TrackerCSRT_create"), ("KCF", "TrackerKCF_create")):
        factory = getattr(cv2, attribute, None) or getattr(getattr(cv2, "legacy", None), attribute, None)
        if factory is not None:
            found.append((f"OpenCV {label}", factory))
    return found


def make_backend(settings: RoiSettings, prefer_builtin: bool = False) -> Any:
    if not prefer_builtin:
        for name, factory in available_cv_backends():
            return CvTrackerBackend(factory, name)
    return TemplateBackend(settings)


def flat_object(frame: np.ndarray, box: tuple[float, float, float, float]) -> tuple[bool, tuple[float, float, float, float], np.ndarray | None, float]:
    """Is the dragged box a one-colour object (maybe with background around it)? Returns (flat, box snapped to the
    object's colour region, its colour, its area in pixels). A textured object returns (False, the box, None, area).

    The background is the median colour of a ring just outside the box. The object is whatever inside the box differs
    from it; if that is one flat colour the tracker follows the colour region rather than correlating texture."""
    x, y, w, h = (int(round(v)) for v in box)
    w, h = max(1, w), max(1, h)
    patch = frame[y:y + h, x:x + w]
    area = float(w * h)
    if patch.size == 0:
        return False, box, None, area
    pad = max(3, int(0.15 * max(w, h)))
    fh, fw = frame.shape[:2]
    rx0, ry0, rx1, ry1 = max(0, x - pad), max(0, y - pad), min(fw, x + w + pad), min(fh, y + h + pad)
    region = frame[ry0:ry1, rx0:rx1]
    ring = np.ones(region.shape[:2], bool)
    ring[y - ry0:y - ry0 + h, x - rx0:x - rx0 + w] = False
    if ring.sum() >= 24:
        background = np.median(region[ring].reshape(-1, 3), axis=0)
        foreground = np.abs(patch.astype(np.int16) - background.astype(np.int16)).max(axis=2) > FLAT_TOLERANCE * 0.75
        share = float(foreground.mean())
        if 0.08 <= share <= 0.95:
            pixels = patch[foreground].reshape(-1, 3)
            if float(pixels.std(axis=0).mean()) < FLAT_OBJECT_STD:
                colour = np.median(pixels, axis=0)
                count, labels, stats, _ = cv2.connectedComponentsWithStats(foreground.astype(np.uint8), connectivity=8)
                if count > 1:
                    best = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
                    bx, by, bw, bh, barea = (int(v) for v in stats[best])
                    return True, (float(x + bx), float(y + by), float(bw), float(bh)), colour, float(barea)
    if float(patch.reshape(-1, 3).std(axis=0).mean()) < FLAT_STD:
        return True, box, patch.reshape(-1, 3).mean(axis=0), area
    return False, box, None, area


# ---------------------------------------------------------------------------------------------------
# The tracker

class RoiOpponentTracker:
    """Same surface as ``ClickedOpponentTracker`` (``update``, ``clear``, ``to_report``, ``to_opponent_estimate``)
    so the pipeline can use either; the selection is ``select_roi`` (a dragged box) instead of a click."""

    def __init__(self, mapping: FloorMapping, arena: dict[str, Any], settings: RoiSettings | None = None,
                 backend_factory: Callable[[RoiSettings], Any] | None = None) -> None:
        self.mapping, self.arena = mapping, arena
        self.settings = settings or RoiSettings()
        self._backend_factory = backend_factory or make_backend
        self.state: OpponentState = NOT_SELECTED
        self.reason = "no opponent selected: drag a box around it in the camera view"
        self.selection_id = 0
        self.selection_note = ""
        self.backend: Any = None
        self._reset_track()
        self._estimate = OpponentTrack(reason=self.reason, selection_id=0)
        self._own_xy: tuple[float, float] | None = None
        self._own_t: float | None = None

    # ---- selection -------------------------------------------------------------------------
    def _reset_track(self) -> None:
        self._box: tuple[float, float, float, float] | None = None
        self._last_good_t: float | None = None
        self._prev_centre: tuple[float, float] | None = None
        self._velocity = np.zeros(2)
        self._reference: np.ndarray | None = None
        self._adaptive: np.ndarray | None = None
        self._flat = False
        self._uncertain_since: float | None = None
        self._t: float | None = None

    def clear(self, reason: str = "selection cleared: drag a box around the opponent") -> None:
        self.state, self.reason = NOT_SELECTED, reason
        self.backend = None
        self._reset_track()
        self._estimate = OpponentTrack(reason=reason, selection_id=self.selection_id, selection_note=self.selection_note)

    def select(self, frame: np.ndarray, pixel_xy: Any, t: float, own_pose: Any = None) -> Selection:
        """A plain click is not a selection here: the object's extent is the box the operator drags."""
        result = Selection(False, "a click does not say how big the object is: drag a box around it")
        self.selection_note = f"click REFUSED: {result.reason}"
        self._estimate.selection_note = self.selection_note
        return result

    def select_roi(self, frame: np.ndarray, rect: Any, t: float, own_pose: Any = None, own_hint_xy: Any = None) -> Selection:
        """Start following whatever is inside ``rect`` = (x, y, width, height) in picture pixels. A refused box leaves any
        current selection alone; an accepted one replaces it at once."""
        result = self._select_roi(frame, rect, t, own_pose, own_hint_xy)
        self.selection_note = ("selected" if result.ok else "box REFUSED") + f": {result.reason}"
        self._estimate.selection_note = self.selection_note
        return result

    def _select_roi(self, frame: np.ndarray, rect: Any, t: float, own_pose: Any, own_hint_xy: Any) -> Selection:
        if not isinstance(frame, np.ndarray) or frame.ndim != 3 or frame.shape[2] != 3 or frame.size == 0:
            return Selection(False, "there is no usable camera picture to select on")
        size = (frame.shape[1], frame.shape[0])
        box, note = validate_rect(rect, size, self.settings.min_side_px)
        if box is None:
            return Selection(False, note)
        self._remember_own(own_pose, own_hint_xy, t)
        overlap = self._own_overlap(box, t, own_pose)
        if overlap is not None and overlap >= self.settings.own_overlap_fraction:
            return Selection(False, f"{100 * overlap:.0f}% of that box is on our own tagged robot: draw the box around the opponent")
        patch = _crop(frame, box)
        if patch is None:
            return Selection(False, "the box holds no picture pixels")
        flat, snapped, colour, area = flat_object(frame, box)
        snap_note = ""
        if flat and isinstance(self._backend_factory(self.settings), TemplateBackend) and snapped != box:
            box, patch = snapped, _crop(frame, snapped)                      # follow the object's own colour region, not the background in the box
            snap_note = " The box was snapped to the object's colour region."
        tw, th = _work_size(box[2], box[3], self.settings.work_px)
        template = cv2.resize(patch, (tw, th), interpolation=cv2.INTER_AREA)
        backend = self._backend_factory(self.settings)
        try:
            backend.init(frame, box, template, flat, colour, area)
        except Exception as error:                                         # a tracker that cannot start must not strand the screen
            return Selection(False, f"the tracker could not start on that box ({type(error).__name__}: {error})")
        self.selection_id += 1
        self.backend = backend
        self._reset_track()
        self._reference, self._adaptive, self._flat = template, template.copy(), flat
        if isinstance(backend, TemplateBackend):
            backend.template = self._adaptive
        self._box, self._last_good_t, self._t = box, t, t
        self._prev_centre = (box[0] + box[2] / 2.0, box[1] + box[3] / 2.0)
        extra = " Low texture: followed by colour, so a similar-coloured object nearby can mislead it." if flat else ""
        unchecked = "" if overlap is not None else " (our own robot's position is unknown, so the box could not be checked against it)"
        self._set(TRACKED, f"following the box you drew ({box[2]:.0f} x {box[3]:.0f} px) with {backend.name}", 1.0, box, t)
        return Selection(True, f"{note}{box[2]:.0f} x {box[3]:.0f} px box accepted, tracker: {backend.name}.{extra}{snap_note}{unchecked}")

    # ---- our own robot ---------------------------------------------------------------------
    def _remember_own(self, own_pose: Any, hint_xy: Any, t: float) -> None:
        if own_pose is not None and getattr(own_pose, "ok", False) and own_pose.x_m is not None:
            self._own_xy, self._own_t = (float(own_pose.x_m), float(own_pose.y_m)), t
        elif hint_xy is not None:
            self._own_xy, self._own_t = (float(hint_xy[0]), float(hint_xy[1])), t

    def _own_polygon(self, t: float) -> np.ndarray | None:
        if self._own_xy is None or self._own_t is None or t - self._own_t > self.settings.own_memory_s or not self.mapping.valid:
            return None
        angles = np.linspace(0.0, 2.0 * math.pi, 36, endpoint=False)
        # The outline is drawn on the floor plane but the chassis is raised, so it appears displaced in the picture: be generous.
        ring = np.array(self._own_xy) + OWN_OUTLINE_MARGIN * self.settings.own_radius_m * np.column_stack([np.cos(angles), np.sin(angles)])
        pixels = self.mapping.floor_to_pixel(ring)
        return pixels if np.isfinite(pixels).all() else None

    def _own_overlap(self, box: tuple[float, float, float, float], t: float, own_pose: Any = None) -> float | None:
        """Share of ``box`` that lies on our own robot, or None when our robot's position is not known."""
        polygon = self._own_polygon(t)
        if polygon is None:
            return None
        x, y, w, h = (int(round(v)) for v in box)
        w, h = max(1, w), max(1, h)
        mask = np.zeros((h, w), np.uint8)
        cv2.fillPoly(mask, [np.round(polygon - np.array([x, y])).astype(np.int32)], 1)
        share = float(mask.mean())
        centre_inside = cv2.pointPolygonTest(np.round(polygon).astype(np.float32).reshape(-1, 1, 2), (x + w / 2.0, y + h / 2.0), False) >= 0
        return max(share, self.settings.own_overlap_fraction) if centre_inside else share      # a box centred on our robot is on it

    # ---- frames ----------------------------------------------------------------------------
    def update(self, frame: np.ndarray, t: float, own_pose: Any = None, own_hint_xy: Any = None) -> OpponentTrack:
        if self.state in (NOT_SELECTED, LOST) or self._box is None or self.backend is None:
            return self._estimate
        self._remember_own(own_pose, own_hint_xy, t)
        s = self.settings
        dt = max(0.0, t - (self._last_good_t if self._last_good_t is not None else t))
        predicted = self._predicted_box(dt)
        margin = s.search_margin_px + 0.75 * max(predicted[2], predicted[3]) + float(np.linalg.norm(self._velocity)) * min(dt, 0.5)
        candidate = self.backend.update(frame, predicted, margin, freeze_scale=self.state != TRACKED)
        if candidate is None:
            return self._uncertain_or_lost("the tracker found nothing in its search window", None, candidate, t)
        candidate = self._clip(candidate, frame.shape[1], frame.shape[0])
        if candidate is None:
            return self._uncertain_or_lost("the tracked box left the picture", None, None, t)
        patch = _crop(frame, candidate)
        confidence = 0.6 * similarity(patch, self._adaptive, self._flat) + 0.4 * similarity(patch, self._reference, self._flat)
        distinct = getattr(self.backend, "last_distinct", None)
        if distinct is not None and confidence < DISTINCT_NOT_NEEDED_ABOVE:
            confidence *= float(np.clip(distinct / DISTINCT_FULL_AT, 0.4, 1.0))
        overlap = self._own_overlap(candidate, t)
        if overlap is not None and overlap >= s.own_overlap_fraction:
            return self._uncertain_or_lost(f"the tracked box is {100 * overlap:.0f}% on our own robot (not updating the appearance, no "
                                           "position reported)", confidence, candidate, t, keep_search_at_last_good=True)
        if confidence < s.floor:
            return self._uncertain_or_lost(f"confidence {confidence:.2f} is below {s.floor:.2f}: the object is hidden or has changed "
                                           "appearance", confidence, candidate, t)
        if confidence < s.confident:
            return self._uncertain_or_lost(f"TRACKING UNCERTAIN: confidence {confidence:.2f} is below {s.confident:.2f}", confidence,
                                           candidate, t)
        centre = (candidate[0] + candidate[2] / 2.0, candidate[1] + candidate[3] / 2.0)
        if self._prev_centre is not None and self._last_good_t is not None and t > self._last_good_t:
            instant = (np.array(centre) - np.array(self._prev_centre)) / (t - self._last_good_t)
            self._velocity = 0.6 * self._velocity + 0.4 * instant
        self._prev_centre, self._last_good_t, self._box, self._uncertain_since = centre, t, candidate, None
        if confidence >= s.adapt_min_confidence and patch is not None:
            self.backend.adapt(patch, s.adapt_rate)
            if not isinstance(self.backend, TemplateBackend):
                resized = cv2.resize(patch, (self._adaptive.shape[1], self._adaptive.shape[0]), interpolation=cv2.INTER_AREA)
                self._adaptive = cv2.addWeighted(self._adaptive, 1.0 - s.adapt_rate, resized, s.adapt_rate, 0.0)
            else:
                self._adaptive = self.backend.template
        return self._set(TRACKED, "tracking", confidence, candidate, t)

    def _predicted_box(self, dt: float) -> tuple[float, float, float, float]:
        x, y, w, h = self._box
        step = self._velocity * min(dt, 0.5)
        return (x + float(step[0]), y + float(step[1]), w, h)

    @staticmethod
    def _clip(box: tuple[float, float, float, float], width: int, height: int) -> tuple[float, float, float, float] | None:
        x0, y0 = max(0.0, box[0]), max(0.0, box[1])
        x1, y1 = min(float(width), box[0] + box[2]), min(float(height), box[1] + box[3])
        return (x0, y0, x1 - x0, y1 - y0) if x1 - x0 >= 3 and y1 - y0 >= 3 else None

    def _uncertain_or_lost(self, why: str, confidence: float | None, candidate: Any, t: float, keep_search_at_last_good: bool = False) -> OpponentTrack:
        if self._uncertain_since is None:
            self._uncertain_since = t
        held = t - self._uncertain_since
        if held > self.settings.uncertain_hold_s:
            self.state, self.reason = LOST, f"TRACKING LOST: {why}, for {held:.1f} s. Drag a new box around the opponent"
            self.backend = None
            self._estimate = OpponentTrack(LOST, self.reason, None, None, confidence, 0, self.selection_id, None, UNKNOWN,
                                           self.selection_note, confidence, None, "", {"lost_after_s": round(held, 2)}, "")
            self._box = None
            return self._estimate
        shown = candidate if candidate is not None else self._box
        self._t = t
        track = self._set(UNCERTAIN, f"TRACKING UNCERTAIN ({held:.1f} s so far): {why}", confidence, shown, t)
        return track

    # ---- reporting -------------------------------------------------------------------------
    def _floor_of(self, pixel: tuple[float, float]) -> tuple[float, float] | None:
        if not self.mapping.valid:
            return None
        xy = self.mapping.pixel_to_floor([pixel])[0]
        return (float(xy[0]), float(xy[1])) if np.isfinite(xy).all() else None

    def _diagnostics(self, box: tuple[float, float, float, float]) -> dict[str, Any]:
        out: dict[str, Any] = {"box_px": [round(v, 1) for v in box], "diagnostic_only": "sizes never decide whether anything is accepted"}
        if self.mapping.valid:
            corners = np.array([[box[0], box[1]], [box[0] + box[2], box[1]], [box[0] + box[2], box[1] + box[3]], [box[0], box[1] + box[3]]])
            floor = self.mapping.pixel_to_floor(corners)
            if np.isfinite(floor).all():
                extent = floor.max(axis=0) - floor.min(axis=0)
                out["approx_floor_extent_mm"] = [round(1000.0 * float(extent[0])), round(1000.0 * float(extent[1]))]
                out["approx_floor_extent_note"] = "the box's footprint on the floor plane; for a raised object this is stretched and shifted, so treat it as a rough guide"
        return out

    def _set(self, state: OpponentState, reason: str, confidence: float | None, box: tuple[float, float, float, float] | None, t: float) -> OpponentTrack:
        self.state, self.reason = state, reason
        centre = None if box is None else (box[0] + box[2] / 2.0, box[1] + box[3] / 2.0)
        tracked = state == TRACKED and centre is not None
        xy = self._floor_of(centre) if tracked else None
        notes = ""
        if tracked and xy is None:
            reason = reason + " (no valid floor map: image position only)"
            self.reason = reason
        if tracked and xy is not None:
            margin = self.settings.arena_margin_m
            width, height = float(self.arena["width_m"]), float(self.arena["height_m"])
            if not (-margin <= xy[0] <= width + margin and -margin <= xy[1] <= height + margin):
                notes = "outside the arena rectangle (shown anyway; the object may be on a table beside it)"
        age = None if self._last_good_t is None else max(0.0, t - self._last_good_t)
        diagnostics = None if box is None else self._diagnostics(box)
        if diagnostics is not None and notes:
            diagnostics["arena_note"] = notes
        self._estimate = OpponentTrack(
            state, reason, xy if tracked else None, None if box is None else tuple(int(round(v)) for v in box), confidence, 0,
            self.selection_id, age, POSITION_QUALITY if xy is not None else UNKNOWN, self.selection_note, confidence,
            None if centre is None else (round(centre[0], 1), round(centre[1], 1)),
            getattr(self.backend, "name", ""), diagnostics, notes)
        return self._estimate

    @property
    def estimate(self) -> OpponentTrack:
        return self._estimate

    def to_opponent_estimate(self):
        return self._estimate.to_opponent_estimate()

    def to_report(self) -> dict[str, Any]:
        return self._estimate.to_report()
