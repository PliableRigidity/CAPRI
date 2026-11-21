"""The ARENA tab's logic without the window: start, click, type, undo, save, reload, verify, readiness.

Pure Python plus OpenCV drawing, so every step runs in the tests without Qt, a camera or a robot link.
It reads and writes only the active profile's own files (through ``control_center.tools.arena_setup``) and
the Command Centre's progress file. It has no connection to the robot.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from control_center.backend.app_state import AppState
from control_center.backend.profiles import LIVE_PROFILE, LoadedProfile, load_profile, profile_label
from control_center.backend.readiness import FAILED, NOT_CONFIGURED, READY, WARNING, Readiness
from control_center.tools import arena_setup as A
from control_center.tools.arena_design import DesignError, EffectiveDesign, Overrides, design_for_profile
from control_center.tools.arena_overlay import draw_arena
from control_center.tools.overlay_core import ClickSet, FloorMapping, arena_outline, densify, pit_arcs

# The competition camera's current view (advisory hints only; the geometry never depends on them).
LIVE_VIEW_HINTS = {"pit_1": "top-left", "pit_2": "bottom-right", "flipper": "bottom-left"}

COLOUR = {"corner": (80, 220, 80), "pit": (60, 60, 255), "flipper": (0, 165, 255), "outline": (255, 210, 60),
          "current": (0, 255, 255), "locked": (200, 160, 255), "text": (240, 240, 240), "rim": (255, 120, 120)}
MODE_TITLES = {"new": "New set-up (everything)", "recalibrate": "Recalibrate camera (corners only; hazards kept)",
               "edit": "Edit arena (hazards and size; corners kept)"}


@dataclass
class SlotRow:
    key: str
    label: str
    state: str            # "empty", "placed", "saved" (locked from file), "current"
    value: str
    required: bool


def _fmt_point(point: tuple[float, float] | None) -> str:
    return "" if point is None else f"({point[0]:.0f}, {point[1]:.0f}) px"


class ArenaWorkflow:
    def __init__(self, profile: str, state: AppState | None = None, *, arena_path: Path | str | None = None,
                 clicks_path: Path | str | None = None, config_path: Path | str | None = None,
                 use_design: bool | None = None, design_path: Path | str | None = None) -> None:
        """The three paths exist so tests never touch the real files; the application leaves them None.

        ``use_design``: None = automatic (the LIVE competition arena uses the design drawing's dimensions, so the
        operator only clicks; other profiles ask for their own dimensions), True/False to force either."""
        A.check_profile_name(profile)
        self.profile = profile
        self.state = state or AppState(None)
        self.arena_path = Path(arena_path) if arena_path else None
        self.clicks_path = Path(clicks_path) if clicks_path else None
        self._config_path = config_path
        self.loaded: LoadedProfile = self._load()
        self.session: A.ArenaSetupSession | None = None
        self.solution = A.Solution()
        self.frame_size: tuple[int, int] | None = None
        self.message = ""
        self.last_save: A.SaveReport | None = None
        self.verified_message = ""
        self._use_design = use_design
        self._design_path = design_path

    # ---- the design (what the software already knows)
    def design_for(self, saved: A.ArenaSetup | None) -> EffectiveDesign | None:
        """The design to use for the next set-up, or None when this profile types its own dimensions.
        An existing saved arena decides: a typed-dimension arena stays typed until a new set-up replaces it."""
        wanted = self.profile == LIVE_PROFILE if self._use_design is None else self._use_design
        if not wanted:
            return None
        if saved is not None and saved.design_used:
            return EffectiveDesign.from_dict(saved.design, design_for_profile(LIVE_PROFILE, self._design_path).design
                                             if self._design_path or self.profile == LIVE_PROFILE else None)
        if saved is not None:
            return None
        from control_center.tools.arena_design import load_design
        return EffectiveDesign(load_design(self._design_path))

    @property
    def design(self) -> EffectiveDesign | None:
        return self.session.design if self.session is not None and self.session.design_flow else None

    def set_overrides(self, overrides: Overrides) -> tuple[bool, str]:
        """Advanced overrides (measured differences from the design). Refused with the reason when implausible."""
        session = self._require()
        try:
            session.set_overrides(overrides)
        except DesignError as error:
            return False, str(error)
        self.message = "Advanced override applied: " + "; ".join(session.design.lines()[:1])
        self._changed()
        return True, self.message

    # ---- what is saved
    def _load(self) -> LoadedProfile:
        return load_profile(self.profile, config_path=self._config_path, clicks_path=self.clicks_path, arena_path=self.arena_path)

    def reload(self) -> LoadedProfile:
        """Read the profile's files again from disk (also the verification step after saving)."""
        self.loaded = self._load()
        return self.loaded

    def defaults(self) -> tuple[tuple[float, float] | None, float | None]:
        if self.profile == LIVE_PROFILE and self._use_design is False:
            return (A.DRAWING_SIDE_M, A.DRAWING_SIDE_M), A.DRAWING_PIT_RADIUS_M
        return None, None

    @property
    def has_draft(self) -> bool:
        return self.state.arena_draft(self.profile) is not None

    # ---- starting
    def begin(self, mode: str, frame_size: tuple[int, int] | None) -> tuple[bool, str]:
        self.reload()
        if self.loaded.problem and mode != "new":
            return False, f"Cannot {mode}: {self.loaded.problem}"
        size, radius = self.defaults()
        hints = LIVE_VIEW_HINTS if self.profile == LIVE_PROFILE else {}
        try:
            saved = self.loaded.arena if mode != "new" else None
            design = self.design_for(saved)
            if design is not None:
                size, radius = None, None
            self.session = A.ArenaSetupSession(self.profile, mode, saved,
                                               self.loaded.clicks if mode == "edit" else None, frame_size, size, radius, hints, design)
        except (A.ArenaSetupError, DesignError) as error:
            self.session = None
            return False, str(error)
        self.frame_size = frame_size
        self.message = f"{MODE_TITLES[mode]}: {self.session.prompt()}"
        self.resolve()
        return True, self.message

    def resume_draft(self, frame_size: tuple[int, int] | None) -> tuple[bool, str]:
        draft = self.state.arena_draft(self.profile)
        if draft is None:
            return False, "There is no unfinished set-up to continue."
        ok, message = self.begin(str(draft.get("mode", "new")), frame_size)
        if not ok:
            return False, message
        if not self.session.restore_draft(draft):
            self.session = None
            self.state.set_arena_draft(self.profile, None)
            return False, "The unfinished set-up no longer matches the saved arena and was discarded."
        if frame_size is not None and self.session.image_size not in (None, tuple(frame_size)):
            self.session.corners = {cid: None for cid in A.CORNER_IDS}
            self.session.image_size = frame_size
            self.message = "The camera picture size changed since those clicks, so the four corners were cleared; click them again."
        else:
            self.message = "Continuing the unfinished set-up: check the clicks against the picture."
        self.frame_size = frame_size or self.session.image_size
        self.resolve()
        return True, self.message

    def discard_draft(self) -> None:
        self.state.set_arena_draft(self.profile, None)

    # ---- input
    def _require(self) -> A.ArenaSetupSession:
        if self.session is None:
            raise RuntimeError("no set-up in progress")
        return self.session

    def _changed(self) -> None:
        self.resolve()
        if self.session is not None:
            self.state.set_arena_draft(self.profile, self.session.to_draft())

    def click(self, u: float, v: float) -> str:
        session = self._require()
        key = session.click(u, v)
        self.message = session.prompt() if key else "The current step is typed, not clicked."
        self._changed()
        return key

    def type_value(self, text: str) -> tuple[bool, str]:
        """Text typed for the current number step. Empty text accepts the offered default."""
        session = self._require()
        slot = session.current_slot()
        if slot is None or slot.kind != A.NUMBER:
            return False, "The current step is a click, not a number."
        try:
            if not text.strip():
                if not session.accept_default():
                    return False, "Type a value (there is no default for this step)."
            else:
                session.submit_number(slot.key, A.parse_length(text))
        except ValueError as error:
            return False, str(error)
        self.message = session.prompt()
        self._changed()
        return True, self.message

    def undo(self) -> str | None:
        key = self._require().undo()
        self.message = f"Undid {key}" if key else "Nothing to undo."
        self._changed()
        return key

    def reset(self) -> None:
        self._require().reset()
        self.message = "Reset."
        self.state.set_arena_draft(self.profile, None)
        self.resolve()

    def goto(self, key: str) -> bool:
        ok = self._require().goto(key)
        self.message = self.session.prompt() if ok else f"No such step: {key}"
        return ok

    def finish_flipper(self) -> bool:
        ok = self._require().finish_flipper()
        self.message = self.session.prompt() if ok else self.session.message
        self._changed()
        return ok

    def reopen_flipper(self) -> None:
        self._require().reopen_flipper()
        self.message = self.session.prompt()
        self._changed()

    def remove_flipper_vertex(self, number: int) -> bool:
        ok = self._require().remove_flipper_vertex(number)
        self.message = f"Removed flipper vertex {number}" if ok else "No such flipper vertex."
        self._changed()
        return ok

    def set_frame_size(self, size: tuple[int, int]) -> None:
        if self.frame_size != size:
            self.frame_size = size
            if self.session is not None:
                self.session.image_size = size
                self.resolve()

    # ---- results
    def resolve(self) -> A.Solution:
        if self.session is None:
            self.solution = A.Solution()
        else:
            self.solution = self.session.solve(self.frame_size, None, self.loaded.config if self.profile == LIVE_PROFILE else None)
        return self.solution

    def rows(self) -> list[SlotRow]:
        session = self.session
        if session is None:
            return []
        rows = []
        for slot in session.slots():
            value = session.value_of(slot.key)
            located = None
            if slot.key in ("pit1_centre", "pit2_centre"):
                located = session.pit_centres[int(slot.key[3]) - 1]
            elif slot.key.startswith("flipper") and int(slot.key[7:]) - 1 < len(session.flipper):
                located = session.flipper[int(slot.key[7:]) - 1]
            if value is None:
                state, text = "empty", ""
            elif located is not None and located.locked:
                state, text = "saved", f"({located.metres[0]:.3f}, {located.metres[1]:.3f}) m  [saved, click to replace]"
            elif slot.kind == A.NUMBER:
                state = "placed"
                text = f"{value * 1000:.0f} mm" if slot.key not in ("width", "height") else f"{value:.3f} m"
            else:
                state, text = "placed", _fmt_point(value)
            if slot.key == session.current:
                state = "current"
            rows.append(SlotRow(slot.key, slot.label, state, text, slot.required))
        return rows

    def accuracy_lines(self) -> list[str]:
        """What the calibration can and cannot be trusted for: fit, picture scale, click offsets, independent checks."""
        solution = self.solution
        lines = list(solution.accuracy)
        worst = solution.quality.predicted_worst_mm if solution.mapping.valid else None
        if worst is not None:
            lines.append(f"a 1-pixel error in a corner click moves the floor map by up to {worst:.1f} mm")
        for key, value in solution.click_checks.items():
            if key.endswith("_offset_mm"):
                lines.append(f"{key[:-10].replace('_', ' ')} click was {value:.0f} mm from where the design puts it")
        if solution.rim_check_mm:
            lines.append("pit-rim check: " + ", ".join(f"{v:.0f} mm" for v in solution.rim_check_mm) + f" (limit {A.RIM_CHECK_OK_MM:.0f} mm)")
        lines += [f"UNRELIABLE: {item}" for item in solution.unreliable]
        return lines

    def hazard_lines(self) -> list[str]:
        setup = self.solution.setup
        return setup.summary_lines() if setup is not None else []

    def can_save(self) -> tuple[bool, str]:
        session = self.session
        if session is None:
            return False, "Start a set-up first."
        solution = self.solution
        if not solution.complete or solution.setup is None:
            return False, "Not complete: " + "; ".join(solution.problems[:3]) if solution.problems else "Not complete."
        if self.frame_size is None:
            return False, "No camera frame size is known."
        if self.loaded.problem and session.mode == "new":
            return True, "ok (the existing damaged files will be kept as .bak copies)"
        return True, "ok"

    def save(self) -> tuple[bool, list[str]]:
        ok, reason = self.can_save()
        if not ok:
            return False, [reason]
        session, solution = self._require(), self.solution
        corners = session.corner_pixels()
        clicks = A.corner_clickset(self.profile, solution.setup, corners, self.frame_size,
                                   self.loaded.clicks, source=f"Command Centre ARENA tab ({profile_label(self.profile)})")
        try:
            report = A.save_setup(self.profile, solution.setup, clicks, write_arena=session.mode != "recalibrate",
                                  arena_path=self.arena_path, clicks_path=self.clicks_path)
        except (A.ArenaSetupError, OSError) as error:
            return False, [f"NOT saved: {error}"]
        self.last_save = report
        self.state.set_arena_draft(self.profile, None)
        self.state.clear_arena_verified(self.profile)         # whatever was confirmed by eye is void after a new save
        self.reload()
        lines = report.lines()
        if solution.level != "ok":
            lines.append("Quality: WEAK. " + "; ".join(solution.quality.reasons[:2]))
        return True, lines

    # ---- verification of what is on disk
    def verify_report(self) -> dict[str, Any]:
        """Reload from disk and say what the overlay will be built from. Visual confirmation is the operator's."""
        loaded = self.reload()
        out: dict[str, Any] = {"problem": loaded.problem, "arena_file": str(loaded.arena_path), "clicks_file": str(loaded.clicks_path),
                               "arena_saved": loaded.arena_saved, "corners_saved": loaded.corners_saved, "lines": []}
        if loaded.arena is not None:
            out["lines"] += loaded.arena.summary_lines()
        return out

    def mark_verified(self) -> bool:
        loaded = self.reload()
        if loaded.arena is None or not loaded.corners_saved or loaded.problem:
            return False
        self.state.set_arena_verified(self.profile, loaded.arena.saved_at, str(loaded.clicks.meta.get("saved_at", "")))
        return True

    def is_verified(self, loaded: LoadedProfile | None = None) -> bool:
        loaded = loaded or self.loaded
        mark = self.state.arena_verified(self.profile)
        if not mark or loaded.arena is None or loaded.clicks is None:
            return False
        return mark.get("arena_saved_at") == loaded.arena.saved_at and mark.get("clicks_saved_at") == str(loaded.clicks.meta.get("saved_at", ""))

    # ---- readiness
    def readiness(self, map_level: str | None = None, map_reason: str = "", restore: Any = None) -> Readiness:
        """``map_level`` is the live vision's own verdict: "ok", "weak", "invalid", "none" or None when unknown.
        ``restore`` is the hub's ``startup_restore.RestoreReport`` for a real camera session (None in MOCK/REPLAY): the
        remembered calibration is trusted only after it was checked against the picture AND confirmed in this run."""
        loaded = self.loaded
        label = profile_label(self.profile)
        if loaded.problem:
            return Readiness(FAILED, "Saved arena files cannot be used", loaded.problem,
                             "Open ARENA, choose 'New set-up' and calibrate again; the damaged files are kept as .bak copies.")
        if loaded.arena is None:
            return Readiness(NOT_CONFIGURED, f"No arena saved for {label}",
                             "The arena size and hazards in the config files are drawing or test values, not a calibration of this arena.",
                             "Open ARENA and run a New set-up: four corners, size, both pits, flipper.")
        if not loaded.corners_saved:
            return Readiness(NOT_CONFIGURED, "Arena saved but the camera corners are missing",
                             "The physical arena is saved; this camera position has not been calibrated against it.",
                             "Open ARENA > Recalibrate camera and click the four corners.")
        if map_level == "invalid":
            return Readiness(FAILED, "The camera mapping is not valid", map_reason,
                             "Open ARENA > Recalibrate camera (the camera moved, or the picture size changed).")
        if restore is not None:
            from control_center.backend.startup_restore import TrustLevel
            if restore.trust is TrustLevel.NEEDS_RECALIBRATION:
                return Readiness(FAILED, "Recalibration needed: " + restore.headline, restore.detail,
                                 restore.next_action or "Recalibrate the camera corners (ARENA > Recalibrate camera).")
            if restore.trust is TrustLevel.NEEDS_CHECK:
                return Readiness(WARNING, "Calibration loaded; alignment not confirmed this run", restore.detail or restore.headline,
                                 restore.next_action or "Press Quick alignment check, compare the overlay with the floor, then Alignment OK.")
        reasons = []
        if not loaded.arena.policy_frame_ok:
            reasons.append("the layout is not the simulator's, so positions cannot feed the policy: " + loaded.arena.canonical.reason)
        if restore is None and not self.is_verified(loaded):
            reasons.append("the overlay has not been confirmed against the floor since it was saved")
        if map_level in (None, "none"):
            reasons.append("the live map has not been built from these files yet (no camera frame, or the files were not loaded into the live view)")
        if reasons:
            return Readiness(WARNING, "Arena saved; confirmation outstanding", "; ".join(reasons),
                             "Open ARENA > 'Check saved overlay', compare the outline with the floor, then tick 'Overlay matches'.")
        checks = loaded.arena.rim_check_mm
        numeric = f"pit-rim check max {max(checks):.0f} mm" if checks else "no numeric check (corners fit exactly); confirmed by eye"
        return Readiness(READY, f"Arena calibrated and confirmed ({label})", f"{loaded.arena.summary_lines()[0]}; {numeric}")


# ---------------------------------------------------------------------------------------------------
# Drawing

def _pixels(mapping: FloorMapping, points_m: Any) -> np.ndarray | None:
    pixels = mapping.floor_to_pixel(densify(np.asarray(points_m, float), 0.02))
    if not np.isfinite(pixels).all():
        return None
    return np.round(np.clip(pixels, -20000, 20000)).astype(np.int32).reshape(-1, 1, 2)


def _label(frame: np.ndarray, text: str, at: tuple[float, float], colour, scale: float = 0.5) -> None:
    cv2.putText(frame, text, (int(at[0]) + 7, int(at[1]) - 7), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(frame, text, (int(at[0]) + 7, int(at[1]) - 7), cv2.FONT_HERSHEY_SIMPLEX, scale, colour, 1, cv2.LINE_AA)


def render_setup_view(frame: np.ndarray, workflow: ArenaWorkflow) -> np.ndarray:
    """A copy of ``frame`` with the corners, the arena outline and the hazards clicked so far."""
    out = frame.copy()
    session, solution = workflow.session, workflow.solution
    if session is None:
        return out
    corners = [session.corners[cid] for cid in A.CORNER_IDS]
    if all(c is not None for c in corners):
        cv2.polylines(out, [np.round(np.array(corners)).astype(np.int32).reshape(-1, 1, 2)], True, COLOUR["outline"], 1, cv2.LINE_AA)
    for cid, point in zip(A.CORNER_IDS, corners):
        if point is not None:
            cv2.drawMarker(out, (int(round(point[0])), int(round(point[1]))), COLOUR["corner"], cv2.MARKER_CROSS, 18, 2, cv2.LINE_AA)
            _label(out, f"{cid[1]}: {session.corner_label(cid).split(' (')[0].split(' at ')[0]}", point, COLOUR["corner"], 0.45)
    mapping, geometry = solution.mapping, solution.geometry
    if mapping.valid and geometry is not None:
        shown = solution.validation.geometry if solution.validation is not None and solution.validation.geometry is not None else geometry
        outline = _pixels(mapping, arena_outline({"width_m": shown.width_m, "height_m": shown.height_m}))
        if outline is not None:
            cv2.polylines(out, [outline], False, COLOUR["outline"], 2, cv2.LINE_AA)
        for index, pit in enumerate(shown.pits, start=1):
            arena = {"width_m": shown.width_m, "height_m": shown.height_m, "pit_centres_m": [list(pit.centre_m)],
                     "pit_radii_m": [pit.radius_m], "pit_radius_m": pit.radius_m}
            for arc in pit_arcs(arena):
                pixels = _pixels(mapping, arc)
                if pixels is not None:
                    cv2.polylines(out, [pixels], False, COLOUR["pit"], 2, cv2.LINE_AA)
            centre = mapping.floor_to_pixel([pit.centre_m])[0]
            if np.isfinite(centre).all():
                cv2.drawMarker(out, (int(centre[0]), int(centre[1])), COLOUR["pit"], cv2.MARKER_CROSS, 14, 2, cv2.LINE_AA)
                _label(out, f"PIT {index}  r {pit.radius_m * 1000:.0f} mm", centre, COLOUR["pit"])
        if len(shown.flipper_m) >= 3:
            polygon = _pixels(mapping, list(shown.flipper_m) + [shown.flipper_m[0]])
            if polygon is not None:
                cv2.polylines(out, [polygon], False, COLOUR["flipper"], 2, cv2.LINE_AA)
                _label(out, "FLIPPER", polygon[0, 0], COLOUR["flipper"])
    # Points exactly as clicked (including ones the maths has not accepted yet).
    for index, located in enumerate(session.pit_centres, start=1):
        if located.pixel is not None:
            cv2.drawMarker(out, (int(round(located.pixel[0])), int(round(located.pixel[1]))), COLOUR["pit"], cv2.MARKER_TILTED_CROSS, 12, 1, cv2.LINE_AA)
    for number, located in enumerate(session.flipper, start=1):
        if located.pixel is not None:
            p = (int(round(located.pixel[0])), int(round(located.pixel[1])))
            cv2.circle(out, p, 5, COLOUR["flipper"], 1, cv2.LINE_AA)
            _label(out, str(number), p, COLOUR["flipper"], 0.45)
    for index, rim in enumerate(session.rim_points, start=1):
        if rim is not None:
            cv2.drawMarker(out, (int(round(rim[0])), int(round(rim[1]))), COLOUR["rim"], cv2.MARKER_DIAMOND, 10, 1, cv2.LINE_AA)
    slot = session.current_slot()
    banner = session.prompt()
    band = out[0:26, :]
    band[:] = (band * 0.3).astype(np.uint8)
    cv2.putText(out, banner[:150], (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOUR["current"] if slot else (120, 230, 120), 1, cv2.LINE_AA)
    return out


def render_saved_view(frame: np.ndarray, loaded: LoadedProfile, mapping: FloorMapping) -> np.ndarray:
    """What the other tools will draw from the saved files: the saved arena over the live picture."""
    out = frame.copy()
    if mapping.valid and loaded.arena is not None:
        draw_arena(out, mapping, loaded.config.arena)
    band = out[0:26, :]
    band[:] = (band * 0.3).astype(np.uint8)
    text = "SAVED ARENA reloaded from disk: does the outline sit on the floor edges, the pits and the flipper?" if mapping.valid else \
        "No valid map to draw the saved arena on: " + mapping.reason
    cv2.putText(out, text[:150], (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (120, 230, 120) if mapping.valid else (60, 60, 255), 1, cv2.LINE_AA)
    return out
