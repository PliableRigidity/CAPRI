"""What happens on every reopening: load the saved arena calibration, check it still fits, ask the operator once.

Workflow this serves: unplug the laptop and the USB camera between rounds, keep the camera physically fixed,
plug back in, open the Command Centre, and get the LIVE arena calibration back WITHOUT recalibrating. The only
things asked of the operator are a quick look at the alignment overlay and, for a new round, the opponent.

Everything here is pure: it works on values handed in (the store, the loaded profile, the camera signature, one
frame) and returns a ``RestoreReport`` the screen shows. It opens no camera, no serial port and no socket.

Trust levels (``TrustLevel``)
  NOT_CALIBRATED          nothing usable is saved for this profile
  NEEDS_RECALIBRATION     the saved pixel calibration cannot be used: the picture size changed, the corner/arena
                          files are inconsistent, or the stored reference shows the camera has moved a lot
  NEEDS_CHECK             calibration loaded; the operator has not confirmed it this session (always the state right
                          after a restart, even when every automatic check passed), or something changed in a way
                          that needs a look (camera signature changed, minor shift, alignment could not be verified)
  CONFIRMED_THIS_SESSION  set ONLY by ``confirm_alignment`` (the operator pressed "Alignment OK")

Confirmation is SESSION-ONLY. ``world_coordinates_trusted`` is True only for CONFIRMED_THIS_SESSION. It is never
written to disk; a fresh start always lands in NEEDS_CHECK. ``record_confirmation`` persists only the informational
"last confirmed at" time, which is shown ("last time...") and never counts as trust.

Typical integration (hub / ARENA screen)
  1. on startup / camera LIVE:   plan = camera_open_settings(profile, store, config_defaults)   (before opening the camera)
  2. first frame arrives:        sig = CameraSignature.from_capture_info(capture.source_info, capture.frame_size)
                                 report = restore_calibration(profile, store, load_profile(profile), sig, frame)
  3. show report.headline / detail / lines; "Check alignment now" calls step 2 again with a fresh frame.
  4. operator presses Alignment OK:   if report.can_confirm: report = confirm_alignment(report); record_confirmation(...)
     (if report.needs_reference_frame: also record_calibration(...) with the current frame, so next time there is a
     reference to compare with)
  5. whenever the calibration is saved:   record_calibration(profile, store, sig, corner_pixels, arena_setup_id, frame)
  6. capture events:   policy.on_event("LOST"/"FAILED"/"RECONNECTING"/"LIVE"); on LIVE after an outage:
                       verdict = policy.on_reconnected(new_sig, saved_sig); report = apply_reconnect_verdict(report, verdict)

Software behaviour only. None of this has been run against a real camera, floor or lighting.
"""

from __future__ import annotations

import dataclasses
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Mapping

import numpy as np

from control_center.backend import alignment_check as ac
from control_center.backend import camera_signature as cs
from control_center.backend.camera_signature import CameraSignature, SignatureComparison
from control_center.backend.profile_settings import ProfileSettingsStore, SaveResult

CORNER_PREFIX = "arena_corner_"          # same as arena_setup.CORNER_PREFIX (a test keeps them equal)
CORNER_TOLERANCE_PX = 1.0                # stored corners vs the loaded ones: beyond this the reference belongs to an older calibration
BACKOFF_S = (1.0, 2.0, 4.0, 5.0)         # then 5 s for ever
ATTEMPT_TIMEOUT_S = 10.0                 # an attempt with no answer for this long counts as failed
RECHECK_AFTER_OUTAGE_S = 30.0            # an outage this long suggests the camera was handled: recommend a re-check


class TrustLevel(str, Enum):
    NOT_CALIBRATED = "NOT_CALIBRATED"
    NEEDS_RECALIBRATION = "NEEDS_RECALIBRATION"
    NEEDS_CHECK = "NEEDS_CHECK"
    CONFIRMED_THIS_SESSION = "CONFIRMED_THIS_SESSION"


class ConfirmationRefused(ValueError):
    """``confirm_alignment`` was called when confirming is not allowed; the message says why."""


# ---------------------------------------------------------------------------------------------- small helpers

def _aware(now: datetime | None) -> datetime:
    value = now or datetime.now(timezone.utc)
    return value if value.tzinfo else value.replace(tzinfo=timezone.utc)


def _when(iso: str | None, now: datetime) -> str:
    """'2026-11-21 10:03 UTC (2 h ago)' for a stored ISO time; the raw text when it cannot be read."""
    if not iso:
        return "unknown time"
    try:
        stamp = datetime.fromisoformat(iso)
    except ValueError:
        return str(iso)
    stamp = stamp if stamp.tzinfo else stamp.replace(tzinfo=timezone.utc)
    seconds = (now - stamp).total_seconds()
    if seconds < 90:
        ago = "just now"
    elif seconds < 5400:
        ago = f"{seconds / 60:.0f} min ago"
    elif seconds < 48 * 3600:
        ago = f"{seconds / 3600:.0f} h ago"
    else:
        ago = f"{seconds / 86400:.0f} days ago"
    return f"{stamp.astimezone(timezone.utc):%Y-%m-%d %H:%M} UTC ({ago})"


def _corner_dict(clicks: Any) -> dict[str, tuple[float, float]]:
    points = getattr(clicks, "points", None) or {}
    return {name: (float(p[0]), float(p[1])) for name, p in sorted(points.items()) if str(name).startswith(CORNER_PREFIX)}


def _as_point_list(corners: Any) -> list[tuple[float, float]] | None:
    if corners is None:
        return None
    values = [corners[k] for k in sorted(corners)] if isinstance(corners, Mapping) else list(corners)
    try:
        return [(float(p[0]), float(p[1])) for p in values]
    except (TypeError, ValueError, IndexError):
        return None


def _same_corners(stored: Any, loaded: list[tuple[float, float]]) -> bool:
    saved = _as_point_list(stored)
    if saved is None or len(saved) != len(loaded):
        return False
    return all(abs(a[0] - b[0]) <= CORNER_TOLERANCE_PX and abs(a[1] - b[1]) <= CORNER_TOLERANCE_PX for a, b in zip(saved, loaded))


def _size_text(size: tuple[int, int]) -> str:
    return f"{size[0]} x {size[1]}"


# ---------------------------------------------------------------------------------------------- the report

@dataclass(frozen=True)
class RestoreReport:
    """Everything the screen needs after a (re)opening. Immutable: ``confirm_alignment`` returns a new report."""

    profile: str
    trust: TrustLevel
    calibration_loaded: bool
    calibration_problem: str = ""                 # why the calibration is not loaded / not usable ("" when loaded)
    recalibration_reason: str = ""                # set for NEEDS_RECALIBRATION
    settings_match: SignatureComparison | None = None
    alignment: ac.AlignmentResult | None = None
    alignment_skipped_reason: str = ""            # why no alignment check was made ("" when one was)
    needs_reference_frame: bool = False           # loaded, but no usable reference picture: store one after the operator confirms
    calibrated_at: str | None = None              # when the corners were saved (ISO)
    last_confirmed_at: str | None = None          # a PREVIOUS run's confirmation: information only
    confirmed_at: str | None = None               # this session's confirmation (never persisted as trust)
    generated_at: str = ""
    store_problem: str = ""                       # the settings file had a load problem
    headline: str = ""
    detail: str = ""
    next_action: str = ""
    lines: tuple[str, ...] = ()

    @property
    def world_coordinates_trusted(self) -> bool:
        """True only after the operator confirmed alignment in THIS run. Never for NEEDS_RECALIBRATION / NOT_CALIBRATED."""
        return self.trust is TrustLevel.CONFIRMED_THIS_SESSION

    @property
    def can_confirm(self) -> bool:
        return self.trust is TrustLevel.NEEDS_CHECK

    @property
    def confirm_refusal(self) -> str:
        """Why 'Alignment OK' is not allowed now ("" when it is allowed or already given)."""
        if self.trust is TrustLevel.NOT_CALIBRATED:
            return f"There is no calibration to confirm. {self.calibration_problem or 'Calibrate the arena first.'}".strip()
        if self.trust is TrustLevel.NEEDS_RECALIBRATION:
            return f"The alignment cannot be confirmed: {self.recalibration_reason or self.calibration_problem}. Recalibrate the arena corners."
        return ""

    @property
    def auto_check_passed(self) -> bool:
        """The automatic alignment check matched and the camera signature did not change in a way that matters."""
        return (self.alignment is not None and self.alignment.status == ac.MATCH
                and (self.settings_match is None or self.settings_match.level not in (cs.CHANGED, cs.INCOMPATIBLE)))

    def to_dict(self) -> dict[str, Any]:
        return {"profile": self.profile, "trust": self.trust.value, "world_coordinates_trusted": self.world_coordinates_trusted,
                "calibration_loaded": self.calibration_loaded, "calibration_problem": self.calibration_problem,
                "recalibration_reason": self.recalibration_reason, "calibrated_at": self.calibrated_at,
                "last_confirmed_at": self.last_confirmed_at, "confirmed_at": self.confirmed_at, "generated_at": self.generated_at,
                "settings_match": None if self.settings_match is None else self.settings_match.to_dict(),
                "alignment": None if self.alignment is None else self.alignment.to_dict(),
                "alignment_skipped_reason": self.alignment_skipped_reason, "needs_reference_frame": self.needs_reference_frame,
                "store_problem": self.store_problem, "headline": self.headline, "detail": self.detail,
                "next_action": self.next_action, "lines": list(self.lines)}


def _finalize(report: RestoreReport, now: datetime) -> RestoreReport:
    """Fill in headline / detail / lines / next_action from the facts in the report."""
    r = report
    lines: list[str] = []

    # calibration
    if r.calibration_loaded:
        lines.append(f"Calibration: loaded (saved {_when(r.calibrated_at, now)}).")
    else:
        lines.append(f"Calibration: NOT usable. {r.calibration_problem}".strip())

    # camera
    match = r.settings_match
    if r.calibration_loaded or match is not None:
        if match is None:
            lines.append("Camera: not open yet, so it could not be compared with the calibration.")
        elif match.level == cs.MATCH:
            minor = len(match.differences)
            lines.append("Camera: same as when the arena was calibrated" + (f" ({minor} minor difference{'s' if minor != 1 else ''}, no effect on positions)." if minor else "."))
        elif match.level == cs.UNKNOWN:
            lines.append(f"Camera: could not be compared. {match.summary}")
        else:
            lines.append("Camera: DIFFERENT from the one used for the calibration.")
            for d in match.non_info:
                lines.append(f"    - {d.message}")

    # alignment
    if r.alignment is not None:
        a = r.alignment
        if a.status == ac.MATCH:
            lines.append(f"Alignment check: PASSED. {a.summary_line()}.")
        elif a.status == ac.SHIFTED:
            lines.append(f"Alignment check: MOVED ({a.severity}). {a.reason}.")
        else:
            lines.append(f"Alignment check: could not be judged. {a.reason}.")
    elif r.calibration_loaded and r.alignment_skipped_reason:
        lines.append(f"Alignment check: not done. {r.alignment_skipped_reason}")

    # confirmation history
    if r.confirmed_at:
        lines.append(f"Alignment confirmed by the operator this session ({_when(r.confirmed_at, now)}).")
    elif r.last_confirmed_at and r.calibration_loaded:
        lines.append(f"Last confirmed by the operator {_when(r.last_confirmed_at, now)}. It must be confirmed again after every restart.")
    if r.store_problem:
        lines.append(f"Saved settings: {r.store_problem}")

    trust = r.trust
    if trust is TrustLevel.NOT_CALIBRATED:
        headline, detail = "Arena not calibrated", r.calibration_problem
        action = "Calibrate the arena (ARENA tab: click the four corners)."
    elif trust is TrustLevel.NEEDS_RECALIBRATION:
        headline, detail = "Recalibrate the arena", r.recalibration_reason or r.calibration_problem
        action = "Recalibrate the arena corners (ARENA tab). Positions must not be used until then."
    elif trust is TrustLevel.CONFIRMED_THIS_SESSION:
        headline = "Alignment confirmed for this session"
        detail = "The operator confirmed the overlay. This confirmation is not saved: it is asked again after a restart."
        action = "Select the opponent and monitor."
    else:
        if r.auto_check_passed:
            headline = "Calibration loaded and the camera looks unchanged: confirm the alignment"
            detail = "The arena corners are where they were calibrated and the camera is the same. Look at the overlay once to confirm."
            action = "Look at the overlay. If its lines sit on the arena edges press Alignment OK."
        else:
            headline = "Calibration loaded: check the alignment"
            reasons = []
            if r.alignment is not None and r.alignment.status == ac.SHIFTED:
                reasons.append(r.alignment.reason)
            elif r.alignment is not None and r.alignment.status == ac.UNVERIFIABLE:
                reasons.append(f"the automatic check could not judge it ({r.alignment.reason})")
            elif r.alignment is None and r.alignment_skipped_reason:
                reasons.append(f"no automatic check was possible ({r.alignment_skipped_reason})")
            if match is not None and match.level in (cs.CHANGED, cs.UNKNOWN):
                reasons.append(match.summary)
            detail = " ".join(reasons) or "It has not been confirmed in this session."
            action = "Look at the overlay by eye. If its lines sit on the arena edges press Alignment OK; otherwise recalibrate."
    lines.append(f"Next: {action}")
    return dataclasses.replace(r, headline=headline, detail=detail, next_action=action, lines=tuple(lines))


# ---------------------------------------------------------------------------------------------- restore

def restore_calibration(profile: str, store: ProfileSettingsStore, loaded_profile: Any, camera_signature: CameraSignature | None,
                        frame: np.ndarray | None, now: datetime | None = None) -> RestoreReport:
    """Decide how far the saved calibration of ``profile`` can be trusted right now. Pure: nothing is written or opened.

    ``store``           the ``ProfileSettingsStore`` (read only here).
    ``loaded_profile``  ``profiles.LoadedProfile`` (or any object with ``.arena``, ``.clicks``, ``.problem``, ``.reason``,
                        ``.corners_saved``; ``.name`` is checked against ``profile`` when present).
    ``camera_signature``  the camera in use now (``CameraSignature.from_capture_info``), or None when it is not open yet.
    ``frame``           one current camera picture (BGR or grey, full size), or None when none has arrived yet.

    The returned trust is NOT_CALIBRATED, NEEDS_RECALIBRATION or NEEDS_CHECK; never CONFIRMED_THIS_SESSION (that is
    ``confirm_alignment``). The alignment check is attempted only when a frame, a stored reference picture belonging to
    THIS calibration, and a matching picture size all exist; otherwise ``alignment_skipped_reason`` says why.
    """
    stamp = _aware(now)
    name = getattr(loaded_profile, "name", profile)
    if name != profile:
        raise ValueError(f"loaded profile is '{name}' but the restore was asked for '{profile}': profiles never share settings")
    settings = store.load(profile)
    context = settings.sections["calibration_context"]
    has_context = "calibration_context" in settings.present
    clicks = getattr(loaded_profile, "clicks", None)
    arena = getattr(loaded_profile, "arena", None)
    problem = str(getattr(loaded_profile, "problem", "") or "")
    reason = str(getattr(loaded_profile, "reason", "") or "")
    corners = _corner_dict(clicks) if clicks is not None else {}
    meta = dict(getattr(clicks, "meta", None) or {})
    calibrated_at = context.get("calibrated_at") or meta.get("saved_at") or getattr(arena, "saved_at", None) or None
    base = RestoreReport(profile=profile, trust=TrustLevel.NOT_CALIBRATED, calibration_loaded=False,
                         calibrated_at=calibrated_at, last_confirmed_at=context.get("last_confirmed_at"),
                         generated_at=stamp.isoformat(), store_problem=settings.load_problem)

    # ---- is there a usable calibration at all
    unusable = ""
    if problem:
        unusable = problem
    elif clicks is None or not corners:
        return _finalize(dataclasses.replace(base, calibration_problem=reason or f"no arena corners are saved for profile '{profile}'"), stamp)
    elif arena is None:
        return _finalize(dataclasses.replace(base, calibration_problem="the arena corners are saved but the arena itself (size, pits, flipper) is not"), stamp)
    else:
        saved_id, arena_id = meta.get("arena_setup_id"), getattr(arena, "setup_id", None)
        if saved_id and arena_id and saved_id != arena_id:
            unusable = "the saved corner clicks were made for a different arena size than the saved arena (the files do not belong together)"
    if unusable:
        return _finalize(dataclasses.replace(base, trust=TrustLevel.NEEDS_RECALIBRATION, calibration_problem=unusable,
                                             recalibration_reason=unusable), stamp)

    loaded = dataclasses.replace(base, calibration_loaded=True)
    corner_list = list(corners.values())

    # ---- picture size: the calibration is in pixels
    calibration_size = None
    if getattr(clicks, "image_size", None):
        calibration_size = (int(clicks.image_size[0]), int(clicks.image_size[1]))
    elif context.get("resolution"):
        calibration_size = (int(context["resolution"][0]), int(context["resolution"][1]))
    current_size = None
    if frame is not None and getattr(frame, "ndim", 0) >= 2:
        current_size = (int(frame.shape[1]), int(frame.shape[0]))
    elif camera_signature is not None and camera_signature.resolution:
        current_size = camera_signature.resolution

    saved_signature = CameraSignature.from_dict(context["camera_signature"]) if context.get("camera_signature") else None
    settings_match = cs.compare(saved_signature, camera_signature) if camera_signature is not None else None
    loaded = dataclasses.replace(loaded, settings_match=settings_match)

    resolution_problem = ""
    if calibration_size and current_size and calibration_size != current_size:
        resolution_problem = (f"the camera picture is {_size_text(current_size)} but the arena was calibrated at {_size_text(calibration_size)}; "
                              "the calibration is in pixels and no longer applies")
    elif settings_match is not None and settings_match.level == cs.INCOMPATIBLE:
        resolution_problem = settings_match.summary
    if resolution_problem:
        return _finalize(dataclasses.replace(loaded, trust=TrustLevel.NEEDS_RECALIBRATION, recalibration_reason=resolution_problem,
                                             calibration_problem=resolution_problem,
                                             alignment_skipped_reason="the picture size differs from the calibration"), stamp)

    # ---- alignment against the stored reference picture
    skipped = ""
    alignment = None
    reference = None
    stale = ""
    if not has_context:
        skipped = "no reference picture was stored with this calibration (it was saved without one)"
    else:
        if context.get("arena_setup_id") and getattr(arena, "setup_id", None) and context["arena_setup_id"] != arena.setup_id:
            stale = "the stored reference picture was made for a different arena set-up"
        elif not _same_corners(context.get("corner_pixels"), corner_list):
            stale = "the stored reference picture belongs to an earlier calibration (the corners have been clicked again since)"
        if stale:
            skipped = stale
        else:
            reference = store.load_reference_frame(profile)
            if reference is None:
                skipped = store.reference_problem(profile) or "no reference picture is stored for this calibration"
            elif calibration_size and tuple(reference.original_size) != calibration_size:
                skipped, reference = "the stored reference picture has a different size from the calibration", None
    if reference is not None:
        if frame is None:
            skipped = "no camera picture has arrived yet"
        else:
            alignment = ac.check_alignment(reference.image, frame, corner_list, reference_original_size=reference.original_size)

    needs_reference = reference is None and frame is not None
    loaded = dataclasses.replace(loaded, alignment=alignment, alignment_skipped_reason=skipped if alignment is None else "",
                                 needs_reference_frame=needs_reference)
    if alignment is not None and alignment.major:
        reason_text = f"the camera has moved since the calibration: {alignment.reason}"
        return _finalize(dataclasses.replace(loaded, trust=TrustLevel.NEEDS_RECALIBRATION, recalibration_reason=reason_text,
                                             calibration_problem=reason_text), stamp)
    return _finalize(dataclasses.replace(loaded, trust=TrustLevel.NEEDS_CHECK), stamp)


def confirm_alignment(report: RestoreReport, now: datetime | None = None) -> RestoreReport:
    """The operator pressed 'Alignment OK'. Returns the report with trust CONFIRMED_THIS_SESSION and
    ``world_coordinates_trusted`` True. Allowed only from NEEDS_CHECK (idempotent when already confirmed);
    raises ``ConfirmationRefused`` with the reason for NOT_CALIBRATED / NEEDS_RECALIBRATION.

    Session-only: this changes nothing on disk. Call ``record_confirmation`` to store the informational time."""
    if report.trust is TrustLevel.CONFIRMED_THIS_SESSION:
        return report
    if not report.can_confirm:
        raise ConfirmationRefused(report.confirm_refusal)
    stamp = _aware(now)
    return _finalize(dataclasses.replace(report, trust=TrustLevel.CONFIRMED_THIS_SESSION, confirmed_at=stamp.isoformat()), stamp)


def record_confirmation(profile: str, store: ProfileSettingsStore, report: RestoreReport) -> SaveResult:
    """Persist ONLY the informational 'last confirmed at' time of a confirmed report. It never restores trust."""
    if report.trust is not TrustLevel.CONFIRMED_THIS_SESSION or not report.confirmed_at:
        return SaveResult(False, "nothing to record: the alignment has not been confirmed", None)
    if not store.has_section(profile, "calibration_context"):
        return SaveResult(False, "no saved calibration context to attach the confirmation time to; call record_calibration first", None)
    return store.update_section(profile, "calibration_context", {"last_confirmed_at": report.confirmed_at})


# ---------------------------------------------------------------------------------------------- recording

def record_calibration(profile: str, store: ProfileSettingsStore, camera_signature: CameraSignature | None,
                       corner_pixels: Mapping[str, Any] | list[Any], arena_setup_id: str | None, frame: np.ndarray | None,
                       now: datetime | None = None) -> SaveResult:
    """Call when a calibration is saved (and after an operator-confirmed legacy calibration with no reference):
    stores the camera signature, picture size, corner pixels, arena set-up id and a downscaled greyscale reference
    picture, in ONE atomic write. ``frame`` should be the full-size picture the corners were clicked on (or a current one
    when the operator has just confirmed alignment). With ``frame=None`` the context is saved without a reference picture
    and any older reference is dropped."""
    stamp = _aware(now)
    size = None
    if frame is not None and getattr(frame, "ndim", 0) >= 2:
        size = (int(frame.shape[1]), int(frame.shape[0]))
    elif camera_signature is not None and camera_signature.resolution:
        size = camera_signature.resolution
    signature = camera_signature
    if signature is not None and frame is not None and size is not None:
        signature = dataclasses.replace(signature, width=size[0], height=size[1])          # what actually arrived wins
    if isinstance(corner_pixels, Mapping):
        corners: Any = {str(k): [float(v[0]), float(v[1])] for k, v in corner_pixels.items()}
    else:
        corners = [[float(p[0]), float(p[1])] for p in corner_pixels]
    context = {"camera_signature": None if signature is None else signature.to_dict(),
               "resolution": None if size is None else [size[0], size[1]], "corner_pixels": corners,
               "arena_setup_id": arena_setup_id, "calibrated_at": stamp.isoformat(), "last_confirmed_at": None}
    return store.save_calibration_bundle(profile, context, frame)


def save_camera_preferences(profile: str, store: ProfileSettingsStore, camera_signature: CameraSignature) -> SaveResult:
    """Remember how the camera was opened (index, name if known, backend, size, fps, format, extra controls)."""
    return store.set_section(profile, "camera", camera_signature.to_dict())


def load_camera_preferences(profile: str, store: ProfileSettingsStore) -> CameraSignature:
    """The remembered camera mode as a signature (all None when nothing was saved)."""
    return CameraSignature.from_dict(store.get_section(profile, "camera"))


@dataclass(frozen=True)
class CameraOpenPlan:
    index: int | None                 # None: use the configured default index
    settings: dict[str, Any]          # for capture.open_camera(index, settings): backend, width, height, fps, fourcc (+ extra controls)
    notes: tuple[str, ...] = ()       # where each value came from, for the log


def camera_open_settings(profile: str, store: ProfileSettingsStore, config_defaults: Mapping[str, Any],
                         prefer_calibration: bool = True) -> CameraOpenPlan:
    """The settings to open the camera with so it comes back in the SAME MODE as when the arena was calibrated.

    Priority for backend / size / fps / format / extra controls: the calibration's signature (when
    ``prefer_calibration``), then the remembered camera preferences, then ``config_defaults`` (e.g.
    ``capture.camera_settings_from_config(config)``). The index comes from the remembered preferences first (the camera
    was last found there), then the calibration, then ``config_defaults['index']``/``['camera_index']``.
    Opens nothing."""
    settings = {key: config_defaults[key] for key in ("backend", "width", "height", "fps", "fourcc") if key in config_defaults}
    extra = dict(config_defaults.get("extra") or {})
    notes: list[str] = []
    prefs = load_camera_preferences(profile, store)
    context = store.get_section(profile, "calibration_context")
    calibration = CameraSignature.from_dict(context["camera_signature"]) if context.get("camera_signature") else CameraSignature()
    layers = [("remembered camera settings", prefs)]
    if prefer_calibration:
        layers.append(("the calibration", calibration))
    else:
        layers.insert(0, ("the calibration", calibration))
    for label, signature in layers:
        for key in ("backend", "width", "height", "fps", "fourcc"):
            value = getattr(signature, key)
            if value is not None:
                if key in settings and settings[key] != value:
                    notes.append(f"{key}: {value} from {label} (config said {settings[key]})")
                settings[key] = value
        extra.update(signature.extra)
    settings.update(extra)
    index = prefs.index if prefs.index is not None else calibration.index
    if index is None:
        index = config_defaults.get("index", config_defaults.get("camera_index"))
    return CameraOpenPlan(index, settings, tuple(notes))


# ---------------------------------------------------------------------------------------------- reconnecting

LIVE, LOST, RECONNECTING, FAILED, IDLE = "LIVE", "LOST", "RECONNECTING", "FAILED", "IDLE"
_EVENT_ALIASES = {"LIVE": LIVE, "MOCK": LIVE, "REPLAY": LIVE, "LOST": LOST, "UNAVAILABLE": FAILED, "FAILED": FAILED,
                  "RECONNECTING": RECONNECTING, "OPENING": RECONNECTING}

VALID, NEEDS_CHECK, NEEDS_RECALIBRATION, UNKNOWN = "valid", "needs_check", "needs_recalibration", "unknown"


@dataclass(frozen=True)
class ReconnectVerdict:
    """What a reconnected camera means for the saved calibration."""

    outcome: str                                # "valid", "needs_check", "needs_recalibration" or "unknown" (treated as needs_check)
    comparison: SignatureComparison | None
    outage_s: float | None
    message: str

    @property
    def calibration_valid(self) -> bool:
        return self.outcome != NEEDS_RECALIBRATION

    @property
    def keeps_trust(self) -> bool:
        """World coordinates stay trusted only when the camera came back exactly as it was."""
        return self.outcome == VALID

    @property
    def needs_check(self) -> bool:
        return self.outcome in (NEEDS_CHECK, UNKNOWN)

    @property
    def needs_recalibration(self) -> bool:
        return self.outcome == NEEDS_RECALIBRATION

    @property
    def recheck_alignment_recommended(self) -> bool:
        """True after a change, or after an outage long enough that the camera may have been handled."""
        return self.outcome != VALID or (self.outage_s is not None and self.outage_s >= RECHECK_AFTER_OUTAGE_S)


class CameraReconnectPolicy:
    """When to retry a lost camera, what to tell the operator, and what a reconnection means for the calibration.

    Pure and injectable: ``clock`` returns seconds (default ``time.monotonic``); nothing is opened. Feed it the capture
    session's state changes with ``on_event`` (LIVE, LOST, RECONNECTING, FAILED; the capture names MOCK/REPLAY,
    UNAVAILABLE and OPENING are accepted too). Poll ``poll()`` from a timer: it returns True exactly when a retry is due
    (then the caller asks the capture session to reopen). Backoff after the camera is lost: 1, 2, 4, 5, 5, ... seconds,
    one step per failed attempt. ``manual_reconnect()`` resets it and makes the next ``poll()`` return True.
    """

    def __init__(self, clock: Callable[[], float] = time.monotonic, backoff: tuple[float, ...] = BACKOFF_S,
                 attempt_timeout_s: float = ATTEMPT_TIMEOUT_S) -> None:
        if not backoff or any(b <= 0 for b in backoff):
            raise ValueError("backoff must be a non-empty tuple of positive seconds")
        self._clock = clock
        self._backoff = tuple(backoff)
        self._attempt_timeout = attempt_timeout_s
        self.state = IDLE
        self.detail = ""
        self.failures = 0                       # consecutive failed/lost events since the last LIVE
        self.attempts = 0                       # retries issued by poll() since the camera was lost
        self.ever_live = False
        self._next_retry_at: float | None = None
        self._attempt_started: float | None = None
        self._down_since: float | None = None

    # ---- events
    def _delay(self, failures: int) -> float:
        return self._backoff[min(max(failures, 1) - 1, len(self._backoff) - 1)]

    def on_event(self, event: str, detail: str = "") -> None:
        kind = _EVENT_ALIASES.get(str(event).upper())
        if kind is None:
            raise ValueError(f"unknown camera event {event!r}")
        now = self._clock()
        self.detail = detail
        if kind == LIVE:
            self.state, self.ever_live = LIVE, True
            self.failures, self._next_retry_at, self._attempt_started = 0, None, None
        elif kind in (LOST, FAILED):
            if self._down_since is None:
                self._down_since = now
            self.failures += 1
            self.state = LOST
            self._next_retry_at = now + self._delay(self.failures)
            self._attempt_started = None
        else:                                   # RECONNECTING
            if self.state != LIVE:
                self.state = RECONNECTING
                self._attempt_started = self._attempt_started if self._attempt_started is not None else now

    # ---- retrying
    def poll(self) -> bool:
        """True exactly once when a retry is due: the caller should ask the capture session to reopen the camera."""
        now = self._clock()
        if self.state == RECONNECTING and self._attempt_started is not None and now - self._attempt_started >= self._attempt_timeout:
            self.on_event(FAILED, "no answer from the camera")
        if self.state == LOST and self._next_retry_at is not None and now >= self._next_retry_at:
            self.state, self._attempt_started, self._next_retry_at = RECONNECTING, now, None
            self.attempts += 1
            return True
        return False

    def next_retry_in(self) -> float | None:
        """Seconds until the next retry (0.0 when due), or None when no retry is scheduled."""
        if self.state != LOST or self._next_retry_at is None:
            return None
        return max(0.0, self._next_retry_at - self._clock())

    def manual_reconnect(self) -> bool:
        """The operator pressed 'Reconnect': reset the backoff and make the next ``poll()`` return True.
        Returns False when the camera is live (nothing to do)."""
        if self.state in (LIVE, IDLE):
            return False
        self.failures = 0
        self.state = LOST
        self._attempt_started = None
        self._next_retry_at = self._clock()
        if self._down_since is None:
            self._down_since = self._clock()
        return True

    @property
    def outage_s(self) -> float | None:
        return None if self._down_since is None else max(0.0, self._clock() - self._down_since)

    def status_text(self) -> str:
        """One line for the screen."""
        if self.state == IDLE:
            return "Camera not opened yet."
        if self.state == LIVE:
            return "Camera connected."
        away = self.outage_s or 0.0
        if self.state == RECONNECTING:
            return f"Reconnecting to the camera (attempt {max(self.attempts, 1)}, lost {away:.0f} s ago)..."
        wait = self.next_retry_in()
        head = "Camera lost" if self.ever_live else "Camera not found"
        when = "retrying now" if wait is not None and wait <= 0.05 else (f"next try in {wait:.0f} s" if wait is not None else "waiting")
        return (f"{head} ({away:.0f} s): {when} (attempt {self.attempts + 1}). Check the USB cable, or press Reconnect to try now."
                + (f" [{self.detail}]" if self.detail else ""))

    # ---- what a reconnection means
    def on_reconnected(self, new_signature: CameraSignature | None, saved_signature: CameraSignature | None) -> ReconnectVerdict:
        """Call after the camera is LIVE again. ``saved_signature`` is the one the calibration was made with
        (``calibration_context.camera_signature``), or, when none is stored, the signature the session was using before
        the camera went away. Same picture size, index and device -> the calibration stays valid and trusted; any other
        non-INFO change -> needs_check (world coordinates untrusted until the operator confirms again); a different
        picture size -> needs_recalibration. Missing information -> "unknown", treated as needs_check."""
        outage = self.outage_s
        self._down_since = None
        comparison = cs.compare(saved_signature, new_signature)
        if comparison.level == cs.INCOMPATIBLE:
            outcome, message = NEEDS_RECALIBRATION, comparison.summary
        elif comparison.level == cs.CHANGED:
            outcome, message = NEEDS_CHECK, comparison.summary + " Check the alignment before trusting positions again."
        elif comparison.level == cs.UNKNOWN:
            outcome = UNKNOWN
            message = comparison.summary + " Positions are not trusted until the alignment is confirmed again."
        else:
            outcome = VALID
            message = "The camera is back and matches the calibration; the saved calibration still applies."
            if outage is not None and outage >= RECHECK_AFTER_OUTAGE_S:
                message += f" It was away for {outage:.0f} s: if it may have been touched, re-check the alignment."
        return ReconnectVerdict(outcome, comparison, outage, message)


def apply_reconnect_verdict(report: RestoreReport, verdict: ReconnectVerdict, now: datetime | None = None) -> RestoreReport:
    """Update a report after a reconnection. A camera that came back with a changed signature takes a CONFIRMED report back
    to NEEDS_CHECK (so ``world_coordinates_trusted`` becomes False until confirmed again); a different picture size takes
    NEEDS_CHECK / CONFIRMED to NEEDS_RECALIBRATION. NOT_CALIBRATED and NEEDS_RECALIBRATION are left as they are, and a
    camera that came back unchanged keeps the report (and its confirmation) as it was."""
    stamp = _aware(now)
    if verdict.keeps_trust or report.trust in (TrustLevel.NOT_CALIBRATED, TrustLevel.NEEDS_RECALIBRATION):
        return report
    updated = dataclasses.replace(report, settings_match=verdict.comparison)
    if verdict.needs_recalibration:
        updated = dataclasses.replace(updated, trust=TrustLevel.NEEDS_RECALIBRATION, confirmed_at=None,
                                      recalibration_reason=verdict.message, calibration_problem=verdict.message)
    else:
        updated = dataclasses.replace(updated, trust=TrustLevel.NEEDS_CHECK, confirmed_at=None)
    return _finalize(updated, stamp)
