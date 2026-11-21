"""Which camera, in which mode, was the arena calibrated with? And is it the same one now?

The arena calibration is a set of PIXEL positions. It stays valid only while the camera delivers the same
pixels for the same scene: same picture size, same lens and mount, same field of view. This module turns
"the camera we are using" into a small comparable ``CameraSignature`` and compares a saved signature with the
current one, producing plain-language differences the operator screen can show verbatim.

Pure data: no camera is opened, no Qt, no files. A signature is only as good as what the caller knows. OpenCV
does not give a device name, so ``device_name`` is usually None; anything None on either side is UNKNOWN for that
field and is NEVER counted as a mismatch (and never as a confirmation either).

Rules (``compare``):

  * picture size (width x height) differs          -> BLOCKING: every saved corner is in pixels, the calibration is void
  * device name differs                            -> WARNING: another camera may have another lens or mount
  * index differs                                  -> INFO if the device name matches, otherwise WARNING (cannot confirm
                                                      it is the same camera after a replug)
  * fourcc (video format) differs                  -> WARNING: some webcams crop or zoom differently per format
  * fps or backend (driver) differs                -> INFO
  * autofocus now ON where it was saved OFF        -> WARNING: focus breathing changes the image scale
  * other camera controls (exposure, focus, ...)   -> INFO

Levels: INCOMPATIBLE when anything is BLOCKING; CHANGED when anything is a WARNING; MATCH when nothing differs
above INFO (INFO differences are still listed); UNKNOWN when either signature is missing or the picture size is
unknown on either side, so it cannot be said whether the pixels still mean the same thing.

None of this is hardware verification: it compares numbers handed to it. Whether the picture really is the same
is what ``alignment_check`` (and the operator's eyes) are for.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

MATCH, CHANGED, INCOMPATIBLE, UNKNOWN = "MATCH", "CHANGED", "INCOMPATIBLE", "UNKNOWN"
LEVELS = (MATCH, CHANGED, INCOMPATIBLE, UNKNOWN)
BLOCKING, WARNING, INFO = "BLOCKING", "WARNING", "INFO"
SEVERITIES = (BLOCKING, WARNING, INFO)

FPS_TOLERANCE = 0.5                 # frames per second: below this is not a difference
NUMBER_TOLERANCE = 1e-6
_CORE_KEYS = ("index", "device_name", "backend", "width", "height", "fps", "fourcc")
_AUTOFOCUS_KEYS = ("autofocus", "auto_focus")


# ---------------------------------------------------------------------------------------------- normalising

def _text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _positive_int(value: Any) -> int | None:
    try:
        if isinstance(value, bool) or value is None:
            return None
        number = int(round(float(value)))
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


def _positive_float(value: Any) -> float | None:
    try:
        if isinstance(value, bool) or value is None:
            return None
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number > 0 and number == number else None       # 0 and NaN mean "unknown" for a driver


def _index(value: Any) -> int | None:
    try:
        if isinstance(value, bool) or value is None:
            return None
        number = int(value)
    except (TypeError, ValueError):
        return None
    return number if number >= 0 else None


def _as_bool(value: Any) -> bool | None:
    """True/False for a camera on/off value in the forms drivers and config files use; None when unclear."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        word = value.strip().lower()
        if word in ("1", "true", "on", "yes", "auto"):
            return True
        if word in ("0", "false", "off", "no", "manual"):
            return False
    return None


@dataclass(frozen=True)
class CameraSignature:
    """The camera and mode the pixels came from. Every field may be None (= not known)."""

    index: int | None = None
    device_name: str | None = None          # None when the OS does not say (OpenCV does not)
    backend: str | None = None              # "dshow", "msmf", ...
    width: int | None = None                # ACTUAL delivered picture size when known, else the requested one
    height: int | None = None
    fps: float | None = None
    fourcc: str | None = None               # "MJPG", "YUY2", ...
    extra: dict[str, Any] = field(default_factory=dict)       # exposure, auto_exposure, focus, autofocus, brightness ...

    def __post_init__(self) -> None:
        # frozen dataclass: normalise through object.__setattr__ so every constructor path gives the same shape
        object.__setattr__(self, "index", _index(self.index))
        object.__setattr__(self, "device_name", _text(self.device_name))
        backend = _text(self.backend)
        object.__setattr__(self, "backend", None if backend is None else backend.lower())
        object.__setattr__(self, "width", _positive_int(self.width))
        object.__setattr__(self, "height", _positive_int(self.height))
        object.__setattr__(self, "fps", _positive_float(self.fps))
        object.__setattr__(self, "fourcc", _text(self.fourcc))
        object.__setattr__(self, "extra", {str(k): v for k, v in dict(self.extra or {}).items() if v is not None})

    # ---- constructors
    @classmethod
    def from_settings(cls, settings: Mapping[str, Any] | None, *, index: int | None = None, device_name: str | None = None,
                      actual_size: tuple[int, int] | None = None, extra: Mapping[str, Any] | None = None) -> "CameraSignature":
        """From a settings dict such as ``capture.camera_settings_from_config(...)``
        (keys backend, width, height, fps, fourcc; ``index`` may be in it or passed separately).

        ``actual_size`` = (width, height) of a frame that really arrived (``capture.frame_size``). It wins over the
        requested size, because a driver may deliver something else than was asked for and the calibration is about
        what arrived. Settings keys other than the core ones (exposure, focus, autofocus, brightness, ...) and the
        ``extra`` argument are kept as ``extra``.
        """
        settings = dict(settings or {})
        merged: dict[str, Any] = {key: value for key, value in settings.items() if key not in _CORE_KEYS and key != "extra"}
        merged.update(dict(settings.get("extra") or {}))
        merged.update(dict(extra or {}))
        width, height = settings.get("width"), settings.get("height")
        if actual_size is not None:
            width, height = actual_size
        return cls(index=index if index is not None else settings.get("index"),
                   device_name=device_name if device_name is not None else settings.get("device_name"),
                   backend=settings.get("backend"), width=width, height=height, fps=settings.get("fps"),
                   fourcc=settings.get("fourcc"), extra=merged)

    @classmethod
    def from_capture_info(cls, source_info: Mapping[str, Any] | None, frame_size: tuple[int, int] | None = None,
                          device_name: str | None = None) -> "CameraSignature":
        """From ``CaptureSession.source_info`` ({"index", "requested": settings, ...}) and ``CaptureSession.frame_size``."""
        info = dict(source_info or {})
        return cls.from_settings(info.get("requested") or {}, index=info.get("index"), device_name=device_name, actual_size=frame_size)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> "CameraSignature":
        """Inverse of ``to_dict``; also accepts the flat ``camera`` section of the profile settings file."""
        data = dict(data or {})
        return cls(index=data.get("index"), device_name=data.get("device_name"), backend=data.get("backend"),
                   width=data.get("width"), height=data.get("height"), fps=data.get("fps"), fourcc=data.get("fourcc"),
                   extra=dict(data.get("extra") or {}))

    def to_dict(self) -> dict[str, Any]:
        return {"index": self.index, "device_name": self.device_name, "backend": self.backend, "width": self.width,
                "height": self.height, "fps": self.fps, "fourcc": self.fourcc, "extra": dict(self.extra)}

    # ---- helpers
    @property
    def resolution(self) -> tuple[int, int] | None:
        return (self.width, self.height) if self.width and self.height else None

    @property
    def is_empty(self) -> bool:
        return all(getattr(self, key) is None for key in _CORE_KEYS) and not self.extra

    def describe(self) -> str:
        """One line for a log or a tooltip."""
        parts = [self.device_name or "camera (name not available)"]
        if self.index is not None:
            parts.append(f"index {self.index}")
        if self.resolution:
            parts.append(f"{self.width}x{self.height}")
        if self.fps:
            parts.append(f"{self.fps:g} fps")
        if self.fourcc:
            parts.append(self.fourcc)
        if self.backend:
            parts.append(self.backend)
        return ", ".join(parts)


@dataclass(frozen=True)
class Difference:
    field: str
    saved: Any
    current: Any
    severity: str                  # BLOCKING, WARNING or INFO
    message: str                   # plain language, safe to show as it is

    def to_dict(self) -> dict[str, Any]:
        return {"field": self.field, "saved": self.saved, "current": self.current, "severity": self.severity, "message": self.message}


@dataclass(frozen=True)
class SignatureComparison:
    level: str                                     # MATCH, CHANGED, INCOMPATIBLE or UNKNOWN
    differences: tuple[Difference, ...] = ()
    summary: str = ""
    unknown_fields: tuple[str, ...] = ()           # fields that could not be compared because one side is None

    @property
    def blocking(self) -> tuple[Difference, ...]:
        return tuple(d for d in self.differences if d.severity == BLOCKING)

    @property
    def warnings(self) -> tuple[Difference, ...]:
        return tuple(d for d in self.differences if d.severity == WARNING)

    @property
    def non_info(self) -> tuple[Difference, ...]:
        return tuple(d for d in self.differences if d.severity != INFO)

    @property
    def calibration_void(self) -> bool:
        """True when the saved pixel calibration cannot apply any more (picture size changed)."""
        return self.level == INCOMPATIBLE

    def to_dict(self) -> dict[str, Any]:
        return {"level": self.level, "summary": self.summary, "unknown_fields": list(self.unknown_fields),
                "differences": [d.to_dict() for d in self.differences]}


def _size_text(w: int, h: int) -> str:
    return f"{w} x {h}"


def _same_number(a: float, b: float, tolerance: float) -> bool:
    return abs(a - b) <= tolerance


def compare(saved: CameraSignature | Mapping[str, Any] | None, current: CameraSignature | Mapping[str, Any] | None) -> SignatureComparison:
    """Compare the signature saved with a calibration against the camera in use now. See the module docstring for the rules."""
    if isinstance(saved, Mapping):
        saved = CameraSignature.from_dict(saved)
    if isinstance(current, Mapping):
        current = CameraSignature.from_dict(current)
    if saved is None or saved.is_empty:
        return SignatureComparison(UNKNOWN, (), "No camera details were saved with the calibration, so the camera cannot be compared. "
                                                "Check the alignment by eye.", ("everything",))
    if current is None or current.is_empty:
        return SignatureComparison(UNKNOWN, (), "The camera is not open yet, so it cannot be compared with the calibration.", ("everything",))

    diffs: list[Difference] = []
    unknown: list[str] = []

    # ---- picture size: the one that voids a pixel calibration
    if saved.resolution is None or current.resolution is None:
        unknown.append("resolution")
    elif saved.resolution != current.resolution:
        sw, sh = saved.resolution
        cw, ch = current.resolution
        diffs.append(Difference(
            "resolution", [sw, sh], [cw, ch], BLOCKING,
            f"The camera now gives {_size_text(cw, ch)} pictures, but the arena was calibrated at {_size_text(sw, sh)}. "
            "The calibration is in pixels, so it no longer applies. Set the camera back to the calibrated size, or calibrate the arena again."))

    # ---- which camera
    names_known = saved.device_name is not None and current.device_name is not None
    names_equal = names_known and saved.device_name.casefold() == current.device_name.casefold()
    if not names_known:
        unknown.append("device_name")
    elif not names_equal:
        diffs.append(Difference(
            "device_name", saved.device_name, current.device_name, WARNING,
            f'The camera is "{current.device_name}" now, but the calibration was made with "{saved.device_name}". '
            "A different camera can have a different lens or mount, so check the alignment before trusting positions."))
    if saved.index is None or current.index is None:
        unknown.append("index")
    elif saved.index != current.index:
        if names_equal:
            diffs.append(Difference("index", saved.index, current.index, INFO,
                                    f"The camera is on slot {current.index} now (it was {saved.index}). It has the same name, so this is normal after replugging."))
        else:
            why = "a different camera name" if names_known else "its name is not available to confirm it is the same camera"
            diffs.append(Difference("index", saved.index, current.index, WARNING,
                                    f"The camera is on slot {current.index} now (it was {saved.index}) and {why}. Check the alignment before trusting positions."))

    # ---- mode
    if saved.fourcc is None or current.fourcc is None:
        unknown.append("fourcc")
    elif saved.fourcc.casefold() != current.fourcc.casefold():
        diffs.append(Difference("fourcc", saved.fourcc, current.fourcc, WARNING,
                                f"The video format changed from {saved.fourcc} to {current.fourcc}. Some webcams change the field of view (crop or zoom) per format, "
                                "so check the alignment."))
    if saved.fps is None or current.fps is None:
        unknown.append("fps")
    elif not _same_number(saved.fps, current.fps, FPS_TOLERANCE):
        diffs.append(Difference("fps", saved.fps, current.fps, INFO,
                                f"The frame rate is {current.fps:g} fps now (it was {saved.fps:g}). The picture itself is not affected."))
    if saved.backend is None or current.backend is None:
        unknown.append("backend")
    elif saved.backend != current.backend:
        diffs.append(Difference("backend", saved.backend, current.backend, INFO,
                                f"The camera driver changed from {saved.backend} to {current.backend}. The picture is normally the same."))

    # ---- camera controls
    for key in sorted(set(saved.extra) | set(current.extra)):
        before, now = saved.extra.get(key), current.extra.get(key)
        if before is None or now is None:
            unknown.append(f"extra.{key}")
            continue
        if key in _AUTOFOCUS_KEYS or key.startswith("auto"):
            was, is_ = _as_bool(before), _as_bool(now)
            if was is None or is_ is None:
                unknown.append(f"extra.{key}")
            elif was != is_:
                if key in _AUTOFOCUS_KEYS and is_ and not was:
                    diffs.append(Difference(f"extra.{key}", before, now, WARNING,
                                            "Autofocus is ON now but was OFF when the arena was calibrated. Autofocus changes the image scale slightly "
                                            "as it refocuses, so turn it off again or check the alignment."))
                else:
                    label = key.replace("_", " ")
                    diffs.append(Difference(f"extra.{key}", before, now, INFO,
                                            f"The camera's {label} setting is {'ON' if is_ else 'OFF'} now (it was {'ON' if was else 'OFF'}). "
                                            "Brightness may look different; the geometry is not affected."))
            continue
        if isinstance(before, (int, float)) and isinstance(now, (int, float)) and not isinstance(before, bool) and not isinstance(now, bool):
            different = not _same_number(float(before), float(now), NUMBER_TOLERANCE)
        else:
            different = str(before).strip().casefold() != str(now).strip().casefold()
        if different:
            label = key.replace("_", " ")
            diffs.append(Difference(f"extra.{key}", before, now, INFO,
                                    f"The camera's {label} is {now} now (it was {before}). Brightness or focus may look different; the geometry is not affected."))

    if any(d.severity == BLOCKING for d in diffs):
        level = INCOMPATIBLE
        summary = next(d.message for d in diffs if d.severity == BLOCKING)
    elif any(d.severity == WARNING for d in diffs):
        level = CHANGED
        summary = ("The camera differs from the one the arena was calibrated with: "
                   + " ".join(d.message for d in diffs if d.severity == WARNING))
    elif "resolution" in unknown:
        level = UNKNOWN
        summary = "The picture size is not known on both sides, so the calibration cannot be confirmed to apply. Check the alignment by eye."
    else:
        level = MATCH
        minor = [d for d in diffs if d.severity == INFO]
        summary = "The camera matches the calibration." + (f" Minor differences: {len(minor)} (they do not affect positions)." if minor else "")
        if unknown:
            summary += " Not available to compare: " + ", ".join(sorted(set(unknown))) + "."
    return SignatureComparison(level, tuple(diffs), summary, tuple(sorted(set(unknown))))
