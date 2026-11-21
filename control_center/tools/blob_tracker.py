"""Offline prototype: track unmarked robots on the arena floor as moving blobs. Read-only vision.

It answers one question per frame for each blob: where on the floor is it, roughly? It does NOT
know which robot is which, cannot see heading, and says so:

  * a track's label (T1, T2, ...) is only "the same blob as before", never "my bot";
  * while two tracks share one blob the state is ``overlap`` and neither gets a position;
  * a track whose blob has gone is ``lost`` from that frame and gets no position (nothing is
    predicted); it is picked up again only by a blob nobody else owns, near where it vanished;
  * after an overlap the tracks are told apart by colour; if the colours are too alike the
    tracks are flagged ``identity_uncertain`` instead of being quietly assigned.

Everything outside the arena floor (people, tools, the pits, the fixed flipper corner and the
flipper beyond it) is masked out before anything is detected.

A blob is compared with a picture of the empty floor. An object that is present in that picture
(for example a robot parked for the whole background window) cannot be seen.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import itertools
import math
from typing import Any

import cv2
import numpy as np

from control_center.tools.overlay_core import FloorMapping, densify, pit_arcs, pit_radii

TRACKED, OVERLAP, LOST = "tracked", "overlap", "lost"


def arena_floor_polygon(arena: dict[str, Any], inset_m: float = 0.0) -> np.ndarray:
    """Playable floor in arena metres: the square minus both pit cut-outs and the fixed flipper corner."""
    w, h = float(arena["width_m"]), float(arena["height_m"])
    flipper = np.asarray(arena["flipper_vertices_m"], float)
    arcs = {tuple(np.round(centre, 6)): arc for centre, arc in zip(map(tuple, arena["pit_centres_m"]), pit_arcs(arena))}
    radius_at = {tuple(np.round(centre, 6)): radius for centre, radius in zip(map(tuple, arena["pit_centres_m"]), pit_radii(arena))}
    if not {(round(w, 6), 0.0), (0.0, round(h, 6))} <= set(arcs):
        # Not the simulator's layout (pits at (w, 0) and (0, h)): the playable floor is only known as the rectangle.
        rectangle = np.array([[0.0, 0.0], [w, 0.0], [w, h], [0.0, h]])
        return rectangle if inset_m <= 0.0 else np.array([w / 2.0, h / 2.0]) + (rectangle - [w / 2.0, h / 2.0]) * (1.0 - 2.0 * inset_m / min(w, h))

    def arc_from(centre: tuple[float, float], start: tuple[float, float]) -> list[np.ndarray]:
        arc = arcs[tuple(np.round(centre, 6))]
        if np.linalg.norm(arc[0] - start) > np.linalg.norm(arc[-1] - start):
            arc = arc[::-1]
        return list(arc)

    points: list[np.ndarray] = [np.array([0.0, 0.0])]
    points += arc_from((w, 0.0), (w - radius_at[(round(w, 6), 0.0)], 0.0))   # along y=0, round the pit at (w, 0)
    on_x_wall = min(flipper, key=lambda v: (abs(v[0] - w), v[1]))       # diagonal's end on the x=w wall
    on_y_wall = min(flipper, key=lambda v: (abs(v[1] - h), v[0]))       # ... and on the y=h wall
    points += [on_x_wall, on_y_wall]                                    # up x=w, across the flipper diagonal
    points += arc_from((0.0, h), (radius_at[(0.0, round(h, 6))], h))    # along y=h, round the pit at (0, h)
    polygon = np.asarray(points)
    if inset_m > 0.0:
        centre = np.array([w / 2.0, h / 2.0])
        polygon = centre + (polygon - centre) * (1.0 - 2.0 * inset_m / min(w, h))
    return polygon


def floor_mask(mapping: FloorMapping, arena: dict[str, Any], image_size: tuple[int, int], inset_m: float = 0.02) -> np.ndarray:
    """Image mask (uint8 0/255) of the playable floor, slightly shrunk so edges do not flicker."""
    polygon = arena_floor_polygon(arena, inset_m)
    pixels = mapping.floor_to_pixel(densify(np.vstack([polygon, polygon[:1]]), 0.01))
    mask = np.zeros((image_size[1], image_size[0]), np.uint8)
    if np.isfinite(pixels).all():
        cv2.fillPoly(mask, [np.round(pixels).astype(np.int32)], 255)
    return mask


def median_background(frames: list[np.ndarray]) -> np.ndarray:
    return np.median(np.stack(frames), axis=0).astype(np.uint8)


@dataclass
class Blob:
    pixel: tuple[float, float]
    floor: tuple[float, float]
    area_px: float
    box: tuple[int, int, int, int]
    colour: np.ndarray                      # mean Lab of the blob's pixels
    area_m2: float = 0.0                    # the blob's picture area scaled to the floor at its centre


@dataclass
class Track:
    track_id: int
    state: str = TRACKED
    floor: tuple[float, float] | None = None        # None unless state is "tracked"
    pixel: tuple[float, float] | None = None
    last_floor: tuple[float, float] = (0.0, 0.0)    # where it was last seen on its own (for matching only)
    colour: np.ndarray | None = None
    area_px: float = 0.0
    area_m2: float = 0.0
    missing: int = 0
    age: int = 0
    group: int | None = None                        # overlap group id while state is "overlap"
    identity_uncertain: bool = False
    still_frames: int = 0
    history: list[tuple[float, float]] = field(default_factory=list)

    @property
    def label(self) -> str:
        return f"T{self.track_id}" + ("?" if self.identity_uncertain else "")

    def to_report(self) -> dict[str, Any]:
        return {"id": self.track_id, "state": self.state,
                "x_m": self.floor[0] if self.floor else None, "y_m": self.floor[1] if self.floor else None,
                "identity_uncertain": self.identity_uncertain, "heading": "not available from a blob",
                "position_quality": "approximate (blob centre on the floor plane, not height-corrected)" if self.floor else None,
                "stationary_frames": self.still_frames}


class BlobTracker:
    def __init__(self, mapping: FloorMapping, arena: dict[str, Any], background: np.ndarray,
                 difference_threshold: float = 30.0, min_area_fraction: float = 0.0010, gate_m: float = 0.20,
                 confirm_frames: int = 8, drop_after: int = 90, revive_within: int = 30,
                 colour_margin: float = 6.0, still_m: float = 0.004) -> None:
        self.mapping, self.arena = mapping, arena
        self.size = (background.shape[1], background.shape[0])
        self.background = cv2.GaussianBlur(background, (3, 3), 0).astype(np.int16)
        self.mask = floor_mask(mapping, arena, self.size)
        self.threshold, self.gate, self.confirm = difference_threshold, gate_m, confirm_frames
        self.min_area = min_area_fraction * self.size[0] * self.size[1]
        self.drop_after, self.colour_margin, self.still_m, self.revive_within = drop_after, colour_margin, still_m, revive_within
        self.tracks: list[Track] = []
        self.blobs: list[Blob] = []
        self.overlap_at: dict[int, tuple[float, float]] = {}       # group id -> shared blob position (floor)
        self._next_id, self._next_group = 1, 1
        self._candidates: list[tuple[tuple[float, float], int]] = []
        self.events: list[dict[str, Any]] = []
        self.frame_index = -1

    # ---- detection ----------------------------------------------------------------------
    def detect(self, frame: np.ndarray) -> list[Blob]:
        if (frame.shape[1], frame.shape[0]) != self.size:
            raise ValueError("frame size differs from the background the tracker was built with")
        difference = np.abs(cv2.GaussianBlur(frame, (3, 3), 0).astype(np.int16) - self.background).max(axis=2)
        foreground = ((difference > self.threshold) & (self.mask > 0)).astype(np.uint8) * 255
        # Drop thin lines first (paint edges show up when the camera drifts a pixel), then close gaps
        # of about 3.5% of the picture height so the parts of one robot mostly stay one blob.
        foreground = cv2.morphologyEx(foreground, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        gap = max(7, int(round(0.036 * self.size[1])) | 1)
        foreground = cv2.morphologyEx(foreground, cv2.MORPH_CLOSE, np.ones((gap, gap), np.uint8))
        count, labels, stats, centroids = cv2.connectedComponentsWithStats(foreground, 8)
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        blobs = []
        for index in range(1, count):
            area = float(stats[index, cv2.CC_STAT_AREA])
            if area < self.min_area:
                continue
            pixel = (float(centroids[index][0]), float(centroids[index][1]))
            floor = self.mapping.pixel_to_floor([pixel])[0]
            if not np.isfinite(floor).all():
                continue
            box = tuple(int(stats[index, key]) for key in (cv2.CC_STAT_LEFT, cv2.CC_STAT_TOP, cv2.CC_STAT_WIDTH, cv2.CC_STAT_HEIGHT))
            # Floor area per pixel here, so sizes can be compared across the picture (near things look bigger).
            east, south = self.mapping.pixel_to_floor([(pixel[0] + 1.0, pixel[1]), (pixel[0], pixel[1] + 1.0)]) - floor
            scale = abs(float(east[0] * south[1] - east[1] * south[0]))
            blobs.append(Blob(pixel, (float(floor[0]), float(floor[1])), area, box, lab[labels == index].reshape(-1, 3).mean(axis=0),
                              area * scale if math.isfinite(scale) else 0.0))
        return blobs

    # ---- association --------------------------------------------------------------------
    @staticmethod
    def _distance(a: tuple[float, float], b: tuple[float, float]) -> float:
        return math.hypot(a[0] - b[0], a[1] - b[1])

    def _best_assignment(self, rows: list[Any], columns: list[Any], cost) -> tuple[dict[int, int], float, float]:
        """Cheapest one-to-one assignment of rows to columns (tiny problems: brute force). Also the runner-up cost."""
        if not rows or not columns:
            return {}, 0.0, float("inf")
        size = min(len(rows), len(columns))
        totals = []
        for chosen_rows in itertools.combinations(range(len(rows)), size):
            for chosen_columns in itertools.permutations(range(len(columns)), size):
                totals.append((sum(cost(rows[r], columns[c]) for r, c in zip(chosen_rows, chosen_columns)), dict(zip(chosen_rows, chosen_columns))))
        totals.sort(key=lambda item: item[0])
        return totals[0][1], totals[0][0], (totals[1][0] if len(totals) > 1 else float("inf"))

    def update(self, frame: np.ndarray) -> list[Track]:
        self.frame_index += 1
        blobs = self.blobs = self.detect(frame)
        # An entity is one track on its own, or the tracks sharing one blob (an overlap group).
        # Lost tracks do not compete for blobs: they can only be picked up again by a blob nobody owns.
        entities: list[tuple[tuple[float, float], list[Track]]] = [(t.last_floor, [t]) for t in self.tracks if t.state == TRACKED]
        for group in sorted({t.group for t in self.tracks if t.state == OVERLAP}):
            entities.append((self.overlap_at[group], [t for t in self.tracks if t.state == OVERLAP and t.group == group]))
        # 1. every entity takes its nearest blob, one blob each
        pairs, _, _ = self._best_assignment(entities, blobs, lambda e, b: min(self._distance(e[0], b.floor), 10 * self.gate))
        owners: dict[int, list[Track]] = {}
        unplaced = []
        for row, (position, members) in enumerate(entities):
            column = pairs.get(row)
            if column is not None and self._distance(position, blobs[column].floor) <= self.gate:
                owners.setdefault(column, []).extend(members)
            else:
                unplaced.append((position, members))
        # 2. an entity left without a blob, next to a blob something else took, has merged into it
        for position, members in unplaced:
            near = [(self._distance(position, blobs[c].floor), c) for c in owners if self._distance(position, blobs[c].floor) <= self.gate]
            if near:
                owners[min(near)[1]].extend(members)
        # 3. a blob shared by several tracks with unowned blobs beside it: the overlap has come apart
        for column in list(owners):
            members = owners[column]
            if len(members) < 2:
                continue
            # When robots part, the shared blob's centre jumps, so look twice as far as usual.
            centre = blobs[column].floor
            smallest = min(t.area_m2 for t in members)
            free = [c for c in range(len(blobs)) if c not in owners and self._distance(centre, blobs[c].floor) <= 2 * self.gate
                    and blobs[c].area_m2 >= 0.5 * smallest]       # a fragment (a fork, a loose part) is not a robot leaving
            if free:
                del owners[column]
                self._split(members, [column] + free, owners)
        # 4. apply
        seen = set()
        for column, members in owners.items():
            if len(members) == 1:
                self._follow(members[0], blobs[column])
            else:
                self._overlap(members, blobs[column])
            seen.update(id(t) for t in members)
        # 4b. a lost track is picked up again only by a blob nobody owns, close to where it vanished
        for track in sorted((t for t in self.tracks if id(t) not in seen and t.missing < self.revive_within), key=lambda t: t.missing):
            near = [(self._distance(track.last_floor, blobs[c].floor), c) for c in range(len(blobs))
                    if c not in owners and self._distance(track.last_floor, blobs[c].floor) <= self.gate / 2
                    and blobs[c].area_m2 >= 0.5 * track.area_m2]
            if near:
                column = min(near)[1]
                owners[column] = [track]
                self._follow(track, blobs[column])
                seen.add(id(track))
        for track in self.tracks:
            if id(track) not in seen:
                track.missing += 1
                if track.state != LOST:                 # no blob this frame: no position is reported
                    self._event("lost", track)
                    track.state, track.floor, track.pixel, track.group, track.still_frames = LOST, None, None, None, 0
            track.age += 1
        self.tracks = [t for t in self.tracks if t.missing <= self.drop_after]
        self.overlap_at = {g: p for g, p in self.overlap_at.items() if any(t.group == g for t in self.tracks)}
        # 5. a blob nobody owns, away from every live track, that persists for a few frames is a new track
        live = [t.last_floor if t.state != OVERLAP else self.overlap_at[t.group] for t in self.tracks if t.state != LOST]
        live += [blobs[c].floor for c, members in owners.items() if len(members) > 1 for _ in (0, 1)]
        candidates = []
        for column, blob in enumerate(blobs):
            if column in owners or any(self._distance(position, blob.floor) <= self.gate for position in live):
                continue
            count = 1 + max((n for position, n in self._candidates if self._distance(position, blob.floor) <= self.gate / 2), default=0)
            if count >= self.confirm:
                track = Track(self._next_id, floor=blob.floor, pixel=blob.pixel, last_floor=blob.floor, colour=blob.colour.copy(),
                              area_px=blob.area_px, area_m2=blob.area_m2)
                self._next_id += 1
                self.tracks.append(track)
                self._event("new", track)
            else:
                candidates.append((blob.floor, count))
        self._candidates = candidates
        return self.tracks

    def _follow(self, track: Track, blob: Blob) -> None:
        moved = self._distance(track.last_floor, blob.floor)
        if track.state == OVERLAP:
            self._event("separated", track)
            moved = 0.0
        elif track.state == LOST:
            self._event("found again", track)
        track.still_frames = track.still_frames + 1 if moved < self.still_m and track.state == TRACKED else 0
        track.state, track.floor, track.pixel, track.last_floor = TRACKED, blob.floor, blob.pixel, blob.floor
        track.group, track.missing = None, 0
        track.colour = blob.colour.copy() if track.colour is None else 0.9 * track.colour + 0.1 * blob.colour
        track.area_px = blob.area_px if not track.area_px else 0.9 * track.area_px + 0.1 * blob.area_px
        track.area_m2 = blob.area_m2 if not track.area_m2 else 0.9 * track.area_m2 + 0.1 * blob.area_m2
        track.history.append(blob.floor)

    def _overlap(self, owners: list[Track], blob: Blob) -> None:
        groups = {t.group for t in owners if t.state == OVERLAP and t.group is not None}
        group = min(groups) if groups else self._next_group
        if not groups:
            self._next_group += 1
        for track in owners:
            if track.state != OVERLAP:
                self._event("overlap begins", track)
            track.state, track.group, track.floor, track.pixel, track.missing, track.still_frames = OVERLAP, group, None, None, 0, 0
        self.overlap_at[group] = blob.floor

    def _split(self, members: list[Track], columns: list[int], owners: dict[int, list[Track]]) -> None:
        """Several blobs where an overlap was: hand them back by colour, and say so if colour cannot tell."""
        blobs = [self.blobs[c] for c in columns]
        cost = lambda track, blob: float(np.linalg.norm((track.colour - blob.colour)[1:])) + 0.25 * abs(float(track.colour[0] - blob.colour[0]))
        pairs, best, runner_up = self._best_assignment(members, blobs, cost)
        uncertain = (runner_up - best) < self.colour_margin
        for row, track in enumerate(members):
            column = pairs.get(row)
            if column is None:                       # more tracks than blobs: the rest stay together on the largest blob
                column = int(np.argmax([blob.area_m2 for blob in blobs]))
            elif uncertain:
                track.identity_uncertain = True
            owners.setdefault(columns[column], []).append(track)
        self.events.append({"frame": self.frame_index, "event": "overlap ends", "tracks": [t.track_id for t in members],
                            "colour_margin": None if not math.isfinite(runner_up) else round(runner_up - best, 1), "identity_uncertain": bool(uncertain)})

    def _event(self, name: str, track: Track) -> None:
        self.events.append({"frame": self.frame_index, "event": name, "track": track.track_id})

    def frame_report(self) -> dict[str, Any]:
        return {"frame": self.frame_index, "blobs": len(self.blobs), "tracks": [t.to_report() for t in self.tracks],
                "overlaps": [{"group": g, "shared_blob_at_m": list(p), "tracks": [t.track_id for t in self.tracks if t.group == g]} for g, p in self.overlap_at.items()]}
