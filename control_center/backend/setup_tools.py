"""Pure logic behind the Setup screen: no Qt, no camera, no serial port, nothing sent to the robot.

* ``CalibrationModel``: click-through floor calibration of the LIVE profile, built on the overlay
  tool's own ``ClickSession`` / ``fit_floor_mapping`` / ``assess_fit`` so a fit means the same thing
  here as in ``control_center.tools.arena_overlay``.
* ``list_profiles``: the other calibration profiles, for display only.
* ``SingleTagCheck``: per-tag bot-centre estimates of a stationary bot, and the spread between them.
* ``marker_report``, ``camera_rows``, ``controller_summary``, ``imu_report``, ``RawLineLog``: text for
  the diagnostics panels. A missing stream is always spelled out, never left blank.
"""

from __future__ import annotations

import copy
import math
import os
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from control_center.backend import status as st
from control_center.backend.live_vision import load_live_clicks
from control_center.tools.overlay_core import (LIVE_PROFILE, PROFILES_DIR, CameraEstimate, ClickSession, ClickSet,
                                               FitQuality, FloorMapping, OverlayConfig, PixelModel, ReferencePoint,
                                               assess_fit, fit_floor_mapping, load_clicks, profile_points_path,
                                               resolve_camera, save_clicks)

LIVE_PROFILE_LABEL = "LIVE profile (your camera)"
FLOOR_LEVEL_NOTE = ("Click every landmark AT FLOOR LEVEL: where the feature meets the floor, never the top edge of a "
                    "wall. Skip a landmark you cannot see.")
WHEN_TO_RECALIBRATE = ("Recalibrate whenever the camera or its stand is moved, bumped, re-aimed or zoomed, or the "
                       "camera resolution changes. The saved points are pixels of one camera position. Quick check "
                       "before a match: the drawn outline must sit on the floor edges.")
MOCK_SAVE_BLOCK = ("Saving is disabled in MOCK mode: points clicked on the synthetic picture must never be written "
                   "into the live calibration file. Restart without --mock to calibrate your camera.")


# ---------------------------------------------------------------- floor calibration (live profile)


def _norm(path: Path | str) -> str:
    return os.path.normcase(str(Path(path).resolve()))


def _is_inside(path: Path, folder: Path) -> bool:
    try:
        Path(_norm(path)).relative_to(_norm(folder))
        return True
    except ValueError:
        return False


@dataclass(frozen=True)
class LandmarkRow:
    index: int
    name: str
    arena_xy: tuple[float, float]
    role: str                           # "fit" or "check", for this camera set-up
    role_overridden: bool               # differs from the role in config/arena_overlay.yaml
    placed: bool
    pixel: tuple[float, float] | None
    fit_residual_mm: float | None       # fit points only; not accuracy evidence
    check_error_mm: float | None        # check points only; held out of the fit
    note: str
    current: bool                       # the landmark the next click places


@dataclass(frozen=True)
class SaveResult:
    ok: bool
    message: str
    path: Path | None = None
    backup: Path | None = None


class CalibrationModel:
    """The live profile's clicked floor points, refitted after every change.

    Only the live profile is edited here. A HISTORICAL footage profile is never adopted, and the
    real live file is never written in MOCK mode or from synthetic points.
    """

    def __init__(self, config: OverlayConfig, target_path: Path | str | None = None, mock: bool = False,
                 clicks: ClickSet | None = None, profiles_dir: Path | str | None = None) -> None:
        self.config = config
        self.live_path = profile_points_path(LIVE_PROFILE)
        self.target_path = Path(target_path) if target_path is not None else self.live_path
        self.profiles_dir = Path(profiles_dir) if profiles_dir is not None else PROFILES_DIR
        self.mock = bool(mock)
        self.extra_block = None                 # optional callable -> reason saving is impossible (e.g. another profile is active)
        self.clicks = ClickSet()
        self.session = ClickSession(config.references, self.clicks)
        self.frame_size: tuple[int, int] | None = None
        self.mapping = FloorMapping()
        self.camera = CameraEstimate()
        self.quality = FitQuality()
        self.message = ""
        self.unsaved = False
        if clicks is not None:
            self.adopt(clicks)
        self.refit()

    # ---- which points are being edited ----

    @property
    def targets_real_live_file(self) -> bool:
        return _norm(self.target_path) == _norm(self.live_path)

    def adopt(self, clicks: ClickSet | None) -> tuple[bool, str]:
        """Start editing from a copy of ``clicks``. A HISTORICAL footage profile is refused."""
        if clicks is None:
            return False, "no points to load"
        if clicks.historical:
            self.message = ("refused: that is a HISTORICAL footage profile; it describes another camera position "
                            "and is never loaded for the live camera")
            return False, self.message
        self.clicks = copy.deepcopy(clicks)
        self.session = ClickSession(self.config.references, self.clicks)
        self.unsaved = False
        self.refit()
        self.message = f"loaded {len(self.clicks.points)} point(s)"
        return True, self.message

    def load_profile(self, name: str) -> tuple[bool, str]:
        """Other profiles are display-only: nothing but the live profile is ever loaded here."""
        if name == LIVE_PROFILE:
            return self.reload_saved()
        self.message = (f"refused: profile {name!r} is not the live profile. Footage profiles are never loaded for "
                        "the live camera; only the LIVE profile is edited on this screen")
        return False, self.message

    def reload_saved(self) -> tuple[bool, str]:
        """Throw away unsaved clicks and reload the saved live points (never a footage profile)."""
        if self.mock and self.targets_real_live_file:
            self.message = ("MOCK mode: the saved live calibration belongs to your real camera and is not loaded "
                            "over the synthetic picture")
            return False, self.message
        clicks, reason = load_live_clicks(self.target_path)
        if clicks is None:
            self.message = f"nothing loaded: {reason}"
            return False, self.message
        return self.adopt(clicks)

    # ---- fitting ----

    def set_frame_size(self, size: tuple[int, int] | None) -> None:
        size = None if size is None else (int(size[0]), int(size[1]))
        if size != self.frame_size:
            self.frame_size = size
            self.refit()

    def clicks_match_frame(self) -> bool:
        return (self.frame_size is None or self.clicks.image_size is None
                or tuple(self.clicks.image_size) == tuple(self.frame_size))

    def refit(self) -> None:
        size = self.frame_size or self.clicks.image_size
        if size is None:
            self.mapping = FloorMapping(reason="no frame yet: freeze a camera frame, then click the landmarks")
            self.camera, self.quality = CameraEstimate(), FitQuality(reasons=[self.mapping.reason])
            return
        model = self.clicks.lens if self.clicks.lens.active else PixelModel(self.config.camera_matrix, self.config.distortion)
        self.mapping = fit_floor_mapping(self.config.references, self.clicks.points, size, model,
                                         roles=self.clicks.roles, clicks_image_size=self.clicks.image_size,
                                         max_fit_residual_mm=self.config.max_fit_residual_mm)
        self.camera = resolve_camera(self.config, self.mapping)
        self.quality = assess_fit(self.config.references, self.clicks.points, self.clicks.roles, self.mapping,
                                  self.config.arena, self.config.warn_check_error_mm,
                                  self.config.weak_predicted_error_mm, self.config.min_hull_fraction)

    def _changed(self) -> None:
        if self.frame_size is not None:
            self.clicks.image_size = self.frame_size       # points placed now belong to this frame size
        self.unsaved = True
        self.refit()

    # ---- editing ----

    @property
    def current(self) -> ReferencePoint | None:
        return self.session.current

    def select(self, index: int) -> ReferencePoint | None:
        return self.session.select(index)

    def select_name(self, name: str) -> ReferencePoint | None:
        names = [ref.name for ref in self.config.references]
        return self.session.select(names.index(name)) if name in names else None

    def place(self, x: float, y: float) -> str | None:
        """Place the current landmark at image pixel (x, y). Returns its name, or None if nothing was placed."""
        if self.frame_size is None:
            self.message = "freeze a camera frame before clicking landmarks"
            return None
        if self.session.current is None:
            self.message = "all landmarks are placed: select one in the list to click it again"
            return None
        cleared = ""
        if self.clicks.points and not self.clicks_match_frame():
            self.session.clear()                            # the old points were for another resolution
            cleared = " (points from another image size were cleared)"
        name = self.session.click(x, y)
        if name:
            self._changed()
            self.message = f"placed {name} at ({x:.1f}, {y:.1f}) px{cleared}"
        return name

    def undo(self) -> str | None:
        name = self.session.undo()
        if name:
            self._changed()
            self.message = f"removed the click on {name}"
        else:
            self.message = "nothing to undo (only clicks made in this session can be undone)"
        return name

    def clear(self) -> None:
        self.session.clear()
        self.clicks.roles.clear()
        self._changed()
        self.message = "all clicks cleared"

    def skip(self) -> None:
        self.session.skip()
        ref = self.session.current
        self.message = f"skipped; next is {ref.name}" if ref is not None else "skipped"

    def toggle_role(self, index: int | None = None) -> str | None:
        """Swap one landmark between FIT and CHECK for this camera set-up."""
        if index is not None:
            self.session.select(index)
        ref = self.session.current
        role = self.session.toggle_role()
        if role and ref is not None:
            self._changed()
            self.message = f"{ref.name} is now a {role.upper()} point"
        return role

    # ---- what to show ----

    def rows(self) -> list[LandmarkRow]:
        errors = {error.name: error for error in self.mapping.errors}
        rows = []
        for index, ref in enumerate(self.config.references):
            role = self.session.role_of(ref)
            pixel = self.clicks.points.get(ref.name)
            error = errors.get(ref.name)
            rows.append(LandmarkRow(
                index, ref.name, ref.arena_xy, role, role != ref.role, pixel is not None, pixel,
                error.error_mm if error is not None and role == "fit" else None,
                error.error_mm if error is not None and role == "check" else None,
                ref.note, index == self.session.index))
        return rows

    def fit_line(self) -> str:
        quality = self.quality
        if quality.level == "ok":
            detail = f"{quality.fit_points} fit + {quality.check_points} check point(s)"
            if quality.predicted_worst_mm is not None:
                detail += f"; a 1 px click error moves the map by at most about {quality.predicted_worst_mm:.0f} mm"
        else:
            detail = "; ".join(quality.reasons) if quality.reasons else "no valid fit"
        return f"FIT: {quality.level.upper()} - {detail}"

    def error_summary(self) -> str:
        fit, check = self.mapping.fit_stats, self.mapping.check_stats
        if not fit["count"]:
            return "No fit yet. Fit residuals and CHECK errors appear once at least 4 fit points are clicked."
        lines = [f"Fit residual: rms {fit['rms_mm']:.1f} mm, max {fit['max_mm']:.1f} mm over {fit['count']} fit point(s) "
                 "(not accuracy evidence: 4 fit points always fit exactly)."]
        if check["count"]:
            lines.append(f"Held-out CHECK error: max {check['max_mm']:.1f} mm, mean {check['mean_mm']:.1f} mm over "
                         f"{check['count']} check point(s) (warning above {self.config.warn_check_error_mm:.0f} mm).")
        else:
            lines.append("Held-out CHECK error: no check point clicked, so the mapping error is UNVERIFIED.")
        return "\n".join(lines)

    def prompt(self) -> str:
        ref = self.session.current
        if ref is None:
            return "All landmarks are placed. Select one in the list to click it again."
        verb = "Re-click" if ref.name in self.clicks.points else "Next click places"
        return (f"{verb}: {ref.name} [{self.session.role_of(ref).upper()}] at ({ref.arena_xy[0]:.3f}, "
                f"{ref.arena_xy[1]:.3f}) m, floor level" + (f" - {ref.note}" if ref.note else ""))

    def status_line(self) -> str:
        placed = len(self.clicks.points)
        size = "x".join(str(v) for v in self.clicks.image_size) if self.clicks.image_size else "no frame size yet"
        if self.unsaved:
            state = "UNSAVED changes"
        elif self.clicks.meta.get("saved_at"):
            state = f"saved {str(self.clicks.meta['saved_at'])[:16]}"
        else:
            state = "nothing saved from this session"
        return f"{placed} of {len(self.config.references)} landmarks placed ({size}) - {state}"

    # ---- saving (the rules of arena_overlay.save_profile, plus the mock / footage guards) ----

    def permanent_save_block(self) -> str | None:
        """A reason saving can never work in this session, whatever is clicked. None = it can."""
        if self.extra_block is not None:
            reason = self.extra_block()
            if reason:
                return reason
        if self.targets_real_live_file and self.mock:
            return MOCK_SAVE_BLOCK
        if _is_inside(self.target_path, self.profiles_dir) or _is_inside(self.target_path, PROFILES_DIR):
            return ("NOT saved: the overlay_profiles folder holds footage profiles; a live calibration is never "
                    "saved there")
        return None

    def save_block_reason(self) -> str | None:
        reason = self.permanent_save_block()
        if reason:
            return reason
        if self.targets_real_live_file and self.clicks.synthetic:
            return "NOT saved: these are SYNTHETIC fixture points and are never written to the live calibration file"
        if self.clicks.historical:
            return "NOT saved: these points are a HISTORICAL footage profile, not a live calibration"
        if self.quality.level == "insufficient":
            return f"NOT saved: fit is INSUFFICIENT - {self.quality.reasons[0] if self.quality.reasons else 'no valid fit'}"
        if self.target_path.exists():
            try:
                existing = load_clicks(self.target_path)
            except (OSError, ValueError, KeyError, TypeError):
                existing = None                             # unreadable: it is still backed up before the save
            if existing is not None and existing.historical:
                return (f"NOT saved: {self.target_path.name} holds a HISTORICAL footage profile, which a live "
                        "calibration never overwrites")
        return None

    def _backup_path(self) -> Path:
        stamp = datetime.now().strftime(".bak-%Y%m%d-%H%M%S")
        backup = self.target_path.with_name(self.target_path.name + stamp)
        counter = 1
        while backup.exists():                              # two saves in one second: keep both backups
            backup = self.target_path.with_name(f"{self.target_path.name}{stamp}-{counter}")
            counter += 1
        return backup

    def save(self, source: str = "") -> SaveResult:
        self.refit()
        reason = self.save_block_reason()
        if reason:
            self.message = reason
            return SaveResult(False, reason)
        backup: Path | None = None
        try:
            if self.target_path.exists():
                backup = self._backup_path()
                backup.write_bytes(self.target_path.read_bytes())
            self.clicks.meta.update({"profile_kind": "live", "profile": LIVE_PROFILE,
                                     "fit_quality_when_saved": self.quality.level})
            if self.mock:
                self.clicks.meta.update({"synthetic_fixture": True,
                                         "label": "clicked on the MOCK synthetic arena, not a camera"})
            save_clicks(self.target_path, self.clicks, source=source or "Command Centre Setup screen")
        except OSError as error:
            self.message = f"NOT saved: {error}"
            return SaveResult(False, self.message)
        self.clicks.meta["saved_at"] = datetime.now(timezone.utc).isoformat()
        self.unsaved = False
        note = f" (previous file kept as {backup.name})" if backup is not None else ""
        weak = "" if self.quality.level == "ok" else f" - fit is {self.quality.level.upper()}"
        self.message = f"saved {len(self.clicks.points)} point(s) to {self.target_path.name}{note}{weak}"
        return SaveResult(True, self.message, self.target_path, backup)

    def clicks_copy(self) -> ClickSet:
        """An independent copy to hand to the hub, so later edits here never touch the live vision."""
        return copy.deepcopy(self.clicks)


# ---------------------------------------------------------------- other profiles (display only)


@dataclass(frozen=True)
class ProfileInfo:
    name: str
    kind: str
    historical: bool
    points: int
    path: Path
    error: str = ""

    def label(self) -> str:
        if self.error:
            return f"{self.name} - unreadable ({self.error})"
        if self.historical:
            return f"{self.name} — {self.kind.upper()} footage, never used for the live camera"
        return f"{self.name} — {self.kind} profile, shown for reference only (this screen edits the LIVE profile only)"


def list_profiles(profiles_dir: Path | str | None = None) -> list[ProfileInfo]:
    """Profiles under ``config/overlay_profiles/*/clicks.json``. Read for their labels only."""
    folder = Path(profiles_dir) if profiles_dir is not None else PROFILES_DIR
    found: list[ProfileInfo] = []
    if not folder.is_dir():
        return found
    for path in sorted(folder.glob("*/clicks.json")):
        name = path.parent.name
        try:
            clicks = load_clicks(path)
        except (OSError, ValueError, KeyError, TypeError) as error:
            found.append(ProfileInfo(name, "unknown", True, 0, path, str(error)))
            continue
        found.append(ProfileInfo(name, clicks.kind, clicks.historical, len(clicks.points), path))
    return found


# ---------------------------------------------------------------- single-tag check (stationary bot)


SPREAD_CAUSE = "marker offset / yaw configuration error"
SPREAD_STATEMENT = ("Differences between tags are marker offset / yaw configuration error (bot.markers in "
                    "config/arena_overlay.yaml), NOT bot movement. A change of the tag in view must not be read as "
                    "motion.")
SINGLE_TAG_HOWTO = ("Keep the bot still. Cover every configured tag except one, press Capture, and wait for the "
                    "count to finish. Repeat for each tag without touching the bot.")


@dataclass(frozen=True)
class TagEstimate:
    """Bot centre and nose heading estimated from ONE tag, averaged over ``samples`` frames."""

    tag_id: int
    x_m: float
    y_m: float
    heading_deg: float
    samples: int
    position_scatter_mm: float          # rms frame-to-frame scatter inside this capture
    heading_scatter_deg: float          # largest deviation from the mean inside this capture


@dataclass(frozen=True)
class TagSpread:
    tags: tuple[int, ...]
    position_mm: float                  # largest distance between two tags' bot-centre estimates
    position_pair: tuple[int, int]
    heading_deg: float                  # largest heading difference between two tags' estimates
    heading_pair: tuple[int, int]
    cause: str = SPREAD_CAUSE


def _angle_difference(a: float, b: float) -> float:
    return (a - b + math.pi) % (2.0 * math.pi) - math.pi


class SingleTagCheck:
    """Averages N valid single-tag poses per tag. Feed it every ``VisionSample``; it ignores the rest."""

    def __init__(self, marker_ids: Iterable[int], samples_needed: int = 20) -> None:
        self.marker_ids = sorted(int(i) for i in marker_ids)
        self.samples_needed = max(1, int(samples_needed))
        self.estimates: dict[int, TagEstimate] = {}
        self.status = "Not capturing."
        self.version = 0                    # bumps whenever the results change
        self._active = False
        self._tag: int | None = None
        self._buffer: list[tuple[float, float, float]] = []
        self._last_frame: int | None = None
        self._restart_note = ""

    @property
    def capturing(self) -> bool:
        return self._active

    @property
    def progress(self) -> tuple[int, int, int | None]:
        return len(self._buffer), self.samples_needed, self._tag

    def start(self) -> None:
        self._active, self._tag, self._buffer, self._last_frame, self._restart_note = True, None, [], None, ""
        self.status = f"Capturing: waiting for a valid pose from exactly one configured tag (0 of {self.samples_needed})."

    def cancel(self) -> None:
        if self._active:
            self._active, self._tag, self._buffer = False, None, []
            self.status = "Capture cancelled."

    def reset(self) -> None:
        self._active, self._tag, self._buffer, self._last_frame = False, None, [], None
        self.estimates.clear()
        self.status = "Reset: no tag captured."
        self.version += 1

    def add(self, sample: Any) -> TagEstimate | None:
        """Take one vision sample. Returns the finished estimate when a capture completes."""
        if not self._active:
            return None
        if self._last_frame is not None and sample.frame_index == self._last_frame:
            return None
        self._last_frame = sample.frame_index
        pose = sample.pose
        visible = [i for i in pose.visible_ids if i in self.marker_ids]
        have = f"{len(self._buffer)} of {self.samples_needed}"
        if len(set(visible)) != 1 or len(visible) != 1:
            if not visible:
                self.status = f"Capturing ({have}): no configured tag in view."
            else:
                self.status = (f"Capturing ({have}): {len(visible)} configured tags in view "
                               f"({', '.join(str(i) for i in visible)}); cover all but one.")
            return None
        if not pose.ok or len(pose.used_ids) != 1:
            self.status = f"Capturing ({have}): tag {visible[0]} in view but no valid pose - {pose.reason}"
            return None
        tag = int(pose.used_ids[0])
        if self._tag is not None and tag != self._tag:
            self._buffer = []
            self._restart_note = f" The tag in view changed from {self._tag} to {tag}, so the count restarted."
        self._tag = tag
        self._buffer.append((float(pose.x_m), float(pose.y_m), float(pose.heading_rad)))
        if len(self._buffer) < self.samples_needed:
            self.status = (f"Capturing tag {tag}: {len(self._buffer)} of {self.samples_needed} valid samples."
                           + self._restart_note)
            return None
        estimate = self._finish(tag)
        self.estimates[tag] = estimate
        self._active, self._tag, self._buffer = False, None, []
        self.version += 1
        self.status = (f"Tag {tag} captured from {estimate.samples} samples. Cover it, uncover another tag and "
                       "press Capture again.")
        return estimate

    def _finish(self, tag: int) -> TagEstimate:
        count = len(self._buffer)
        x = sum(item[0] for item in self._buffer) / count
        y = sum(item[1] for item in self._buffer) / count
        heading = math.atan2(sum(math.sin(item[2]) for item in self._buffer),
                             sum(math.cos(item[2]) for item in self._buffer))
        scatter = math.sqrt(sum((item[0] - x) ** 2 + (item[1] - y) ** 2 for item in self._buffer) / count)
        heading_scatter = max(abs(_angle_difference(item[2], heading)) for item in self._buffer)
        return TagEstimate(tag, x, y, math.degrees(heading), count, 1000.0 * scatter, math.degrees(heading_scatter))

    def spread(self) -> TagSpread | None:
        """Largest disagreement between the captured tags, or None with fewer than two tags."""
        items = [self.estimates[tag] for tag in sorted(self.estimates)]
        if len(items) < 2:
            return None
        position, position_pair, heading, heading_pair = -1.0, (items[0].tag_id, items[1].tag_id), -1.0, (items[0].tag_id, items[1].tag_id)
        for i, a in enumerate(items):
            for b in items[i + 1:]:
                distance = 1000.0 * math.hypot(a.x_m - b.x_m, a.y_m - b.y_m)
                turn = abs(math.degrees(_angle_difference(math.radians(a.heading_deg), math.radians(b.heading_deg))))
                if distance > position:
                    position, position_pair = distance, (a.tag_id, b.tag_id)
                if turn > heading:
                    heading, heading_pair = turn, (a.tag_id, b.tag_id)
        return TagSpread(tuple(item.tag_id for item in items), position, position_pair, heading, heading_pair)

    def spread_text(self) -> str:
        spread = self.spread()
        if spread is None:
            if not self.estimates:
                return "No tag captured yet. The spread between tags needs at least two captured tags."
            only = next(iter(self.estimates))
            return f"One tag captured (id {only}). Capture at least one more tag to see the spread between tags."
        return (f"Spread between {len(spread.tags)} tags: bot centre {spread.position_mm:.1f} mm "
                f"(tags {spread.position_pair[0]} and {spread.position_pair[1]}), heading {spread.heading_deg:.1f} deg "
                f"(tags {spread.heading_pair[0]} and {spread.heading_pair[1]}). Cause: {spread.cause}.")


# ---------------------------------------------------------------- markers panel


def _ids(values: Iterable[int]) -> str:
    return ", ".join(str(v) for v in values)


@dataclass
class MarkerReport:
    rows: list[tuple[str, str, str]]        # (label, text, level) with level ok / warn / bad / off

    def text(self) -> str:
        return "\n".join(f"{label}: {value}" for label, value, _ in self.rows)

    def value(self, label: str) -> str:
        return next(value for name, value, _ in self.rows if name == label)


MARKER_ROW_LABELS = ("Vision", "Detected IDs", "Configured IDs in view", "Missing for the visible face",
                     "Duplicate IDs", "Unknown IDs", "Pose", "Position quality")


def marker_report(config: OverlayConfig, sample: Any, now: float | None = None) -> MarkerReport:
    """What the newest vision sample says about the bot's markers. Every row always has text."""
    if sample is None:
        rows = [("Vision", "No vision sample yet - no camera frame has been processed", "off")]
        rows += [(label, "unknown (no vision sample)", "off") for label in MARKER_ROW_LABELS[1:]]
        return MarkerReport(rows)
    pose = sample.pose
    faces = {face: sorted(i for i, spec in config.markers.items() if spec.face == face) for face in ("top", "bottom")}
    detected = sorted(d.tag_id for d in sample.detections) if sample.detections else sorted(list(pose.visible_ids) + list(pose.ignored_ids))
    counts: dict[int, int] = {}
    for tag_id in detected:
        counts[tag_id] = counts.get(tag_id, 0) + 1
    duplicates = sorted(set(sample.duplicate_ids) | {i for i, n in counts.items() if n > 1})
    unknown = sorted(set(sample.unknown_ids) | {i for i in counts if i not in config.markers})
    configured = sorted(i for i in counts if i in config.markers)
    seen_faces = [face for face in ("top", "bottom") if any(i in faces[face] for i in configured)]

    age = None if now is None else now - sample.t_mono
    if age is not None and age > 1.0:
        rows = [("Vision", f"STALE: the last processed frame is {age:.1f} s old - the rows below are NOT current", "bad")]
    else:
        rows = [("Vision", f"frame {sample.frame_index} processed", "ok")]
    shown = [f"{i} (x{counts[i]})" if counts[i] > 1 else str(i) for i in sorted(counts)]
    rows.append(("Detected IDs", ", ".join(shown) if shown else "none - no marker detected in the frame", "ok" if shown else "off"))
    if not configured:
        rows.append(("Configured IDs in view", "none", "off"))
        rows.append(("Missing for the visible face",
                     f"cannot tell: no configured marker in view (top face: {_ids(faces['top']) or 'none configured'}; "
                     f"bottom face: {_ids(faces['bottom']) or 'none configured'})", "off"))
    elif len(seen_faces) == 2:
        rows.append(("Configured IDs in view", f"{_ids(configured)} (top AND bottom markers together: conflict)", "bad"))
        rows.append(("Missing for the visible face", "cannot tell: top and bottom markers are visible together", "bad"))
    else:
        face = seen_faces[0]
        missing = [i for i in faces[face] if i not in configured]
        rows.append(("Configured IDs in view", f"{_ids(configured)} ({face} face)", "ok"))
        if missing:
            rows.append(("Missing for the visible face",
                         f"{_ids(missing)} not seen ({face} face is configured with {_ids(faces[face])})", "warn"))
        else:
            rows.append(("Missing for the visible face", f"none - all {len(faces[face])} {face}-face markers seen", "ok"))
    rows.append(("Duplicate IDs", f"{_ids(duplicates)} - seen more than once in one frame" if duplicates else "none",
                 "bad" if duplicates else "ok"))
    rows.append(("Unknown IDs", f"{_ids(unknown)} - not in the configured layout" if unknown else "none",
                 "warn" if unknown else "ok"))
    if pose.ok:
        rows.append(("Pose", f"VALID - markers {_ids(pose.used_ids)} used; x {pose.x_m:.3f} m, y {pose.y_m:.3f} m, "
                             f"nose heading {pose.heading_deg:+.1f} deg, {pose.orientation.upper()}", "ok"))
        rows.append(("Position quality", pose.position_label, "ok" if pose.correction_source == "measured" else "warn"))
    else:
        rows.append(("Pose", f"NOT VALID - {pose.reason}", "bad"))
        rows.append(("Position quality", "unknown (no valid pose)", "off"))
    return MarkerReport(rows)


def marker_layout_rows(config: OverlayConfig) -> list[tuple[str, ...]]:
    """id, face, forward_m, left_m, yaw_deg, size_m, height_m as text, one row per configured marker."""
    return [(str(spec.marker_id), spec.face, f"{spec.forward_m:+.3f}", f"{spec.left_m:+.3f}", f"{spec.yaw_deg:+.1f}",
             f"{spec.size_m:.3f}", f"{spec.height_m:.3f}") for _, spec in sorted(config.markers.items())]


def marker_placeholder_note(config: OverlayConfig) -> tuple[str, str]:
    """(text, level): whether the marker layout is measured or still placeholder values."""
    bot = [item for item in config.placeholders if "bot." in item]
    if bot:
        return ("PLACEHOLDER values, NOT measured on the robot: " + "; ".join(bot) + ". Measure them and set "
                "bot.measured: true in config/arena_overlay.yaml.", "warn")
    return "Marked as measured in config/arena_overlay.yaml (bot.measured: true).", "ok"


# ---------------------------------------------------------------- camera panel


def camera_rows(capture: Any, options: Any) -> list[tuple[str, str, str]]:
    """(label, text, level) rows for the camera panel, from the hub's capture session."""
    info = capture.source_info or {}
    kind = info.get("kind")
    source = {"camera": "Camera (live)", "mock": "MOCK synthetic source (not a camera)",
              "replay": "REPLAY of a video file (not live)"}.get(kind, "not opened yet" if kind is None else str(kind))
    rows = [("Source", source, "ok" if kind == "camera" else "warn" if kind else "off"),
            ("Description", info.get("description") or "none (source not open)", "off" if not info else "plain")]
    if info.get("index") is not None:
        rows.append(("Index", str(info["index"]), "ok"))
    elif kind in ("mock", "replay") or getattr(options, "mock", False) or getattr(options, "video", None) is not None:
        rows.append(("Index", "not used (this session is not reading a camera)", "off"))
    else:
        rows.append(("Index", f"{options.camera_index} (configured; camera not open)", "off"))
    size = capture.frame_size
    requested = info.get("requested") or {}
    wanted = (f" (requested {requested.get('width')} x {requested.get('height')} at {requested.get('fps')} fps"
              f"{' ' + str(requested['fourcc']) if requested.get('fourcc') else ''}, backend {requested.get('backend')})"
              if requested else "")
    rows.append(("Resolution", (f"{size[0]} x {size[1]}" if size else "no frame received yet") + wanted, "ok" if size else "off"))
    fps = float(capture.measured_fps or 0.0)
    rows.append(("Measured fps", f"{fps:.1f}" if fps > 0 else "not measured yet (needs 1 s of frames)", "ok" if fps > 0 else "off"))
    level = {"LIVE": "ok", "MOCK": "warn", "REPLAY": "warn", "OPENING": "warn"}.get(capture.state, "bad")
    rows.append(("State", capture.state + (f" - {capture.detail}" if capture.detail else ""), level))
    return rows


# ---------------------------------------------------------------- controller & IMU panel (read-only)


NO_IMU_TEXT = ("No IMU stream — the controller is not sending IMU lines (the handoff bench sketch has no IMU)")
ONE_PORT_NOTE = ("Only one program can hold the COM port. While the PC drive client "
                 "(scripts/handoff_bench_client.py --log file) is driving, follow its log file here instead of "
                 "opening the port.")
READ_ONLY_NOTE = "This screen only listens. It has no control that sends anything to the robot."


class RawLineLog:
    """The newest raw controller lines, and the newest line of each kind."""

    KINDS = ("ST", "CH", "HELLO", "IMU")

    def __init__(self, keep: int = 12) -> None:
        self.tail: deque[tuple[str, str]] = deque(maxlen=keep)
        self.latest: dict[str, str] = {}
        self.count = 0
        self.imu_lines = 0

    def add(self, direction: str, text: str) -> None:
        self.count += 1
        if direction == "RX" and text.startswith("IMU,"):
            # An IMU stream is tens of lines a second: it would push every ST line out of the tail.
            self.imu_lines += 1
            self.latest["IMU"] = text
            return
        self.tail.append((direction, text))
        if direction != "RX":
            return
        for kind in self.KINDS:
            if text.startswith(kind + " "):
                self.latest[kind] = text
                return

    def clear(self) -> None:
        self.tail.clear()
        self.latest.clear()
        self.count = 0
        self.imu_lines = 0

    def latest_text(self, kind: str) -> str:
        return self.latest.get(kind) or f"no {kind} line received"

    def tail_text(self) -> str:
        left_out = f"({self.imu_lines} IMU line(s) received; they are left out of this list)" if self.imu_lines else ""
        if not self.tail:
            return left_out or "no lines received from a controller telemetry source"
        lines = [f"{direction}  {text}" for direction, text in self.tail]
        return "\n".join(lines + ([left_out] if left_out else []))


def controller_summary(monitor: Any, link_label: str | None, link_error: str | None, now: float) -> tuple[str, str, str]:
    """(source text, state text, level) for the telemetry source."""
    item = st.robot_link_item(monitor.authority(now), link_label, link_error)
    source = link_label or "none - no controller telemetry source connected"
    state = item.text + (f" - {link_error}" if link_error and link_label else "")
    return source, state, item.level


def firmware_text(monitor: Any) -> str:
    firmware = monitor.firmware()
    if firmware:
        extra = ", ".join(f"{key}={value}" for key, value in monitor.hello.items() if key != "fw")
        return firmware + (f" ({extra})" if extra else "")
    return "unknown - no HELLO line received (the controller sends it once at start-up)"


def imu_report(monitor: Any, now: float) -> tuple[str, str]:
    """(text, level). Values are the controller's raw sensor axes; nothing is rotated or filtered."""
    if monitor.imu_present(now) and monitor.last_imu is not None:
        sample = monitor.last_imu
        rate = monitor.imu_rate_hz(now)
        ax, ay, az = sample.accel_m_s2
        gx, gy, gz = sample.gyro_deg_s
        return ("IMU stream live - sensor axes, raw (no rotation into a robot frame)\n"
                f"accel X/Y/Z (m/s^2):  {ax:+.3f}   {ay:+.3f}   {az:+.3f}\n"
                f"gyro  X/Y/Z (deg/s):  {gx:+.3f}   {gy:+.3f}   {gz:+.3f}\n"
                + (f"rate: {rate:.0f} Hz" if rate else "rate: measuring (needs 0.5 s of samples)")
                + f"   controller time: {sample.device_ms} ms"), "ok"
    text = NO_IMU_TEXT
    if monitor.last_imu is not None and monitor.last_imu_at is not None:
        text += (f"\nThe last IMU line arrived {now - monitor.last_imu_at:.1f} s ago; its values are stale and are "
                 "not shown as current.")
    return text, "off"
