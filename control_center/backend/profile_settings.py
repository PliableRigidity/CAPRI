"""Per-profile settings that are slow to redo and safe to remember, stored so a restart loses nothing.

Why: the operator unplugs the laptop and the USB camera between rounds and must NOT have to recalibrate when
everything is plugged back in. This store keeps what the Command Centre needs to pick up where it left off:

  camera               how the camera was opened (index, name if known, backend, size, fps, format, extra controls)
  calibration_context  what the saved arena corners were made with: the camera signature and picture size at that
                       time, the corner pixels, which arena set-up they belong to, the reference-picture file name
  markers              ArUco dictionary, own tag id, marker layout name/path/hash, measured offsets (opaque here)
  wifi                 host, port, preferred SSID, auto_connect (default False), when the endpoint last answered
  preferences          anything else cheap to repeat, as a free dictionary

Layout: ONE JSON file per profile, ``<directory>/<profile>.json`` (default ``config/profile_settings/``), plus
``<profile>.json.bak`` (the previous good file) and ``<profile>.reference-<hash>.png`` (the reference picture).
A profile never reads or writes another profile's files, and a file whose embedded profile name differs from the
name it is loaded under is refused.

NEVER persisted (see ``FORBIDDEN_KEYS``): anything about the live situation. Opponent selection and tracks,
tracking state, control mode, control authority, armed / motors-running flags, Wi-Fi link status, observations,
the alignment-confirmed / world-coordinates-trusted flag, and credentials. A new run must find all of that in its
safe default state; ``set_section`` raises ``ValueError`` for these keys at any depth, so nothing can slip in by
accident.

Safety of the file itself:
  * writes go to a temporary file in the same directory, are flushed and fsynced, and only then ``os.replace``d
    over the real file, so after an unexpected shutdown the file is either the old one or the new one, never half;
  * the previous GOOD file is kept as ``.bak`` (one rolling copy; a damaged file never overwrites a good ``.bak``);
  * loading never raises: a missing/damaged/wrong-version/wrong-profile file is reported in ``load_problem``, the
    ``.bak`` is tried, and only then defaults are used. A never-saved profile is NOT a problem (``status == "fresh"``).

This module opens no camera and no socket and drives nothing. Tests of it are software tests with a temporary
directory; they say nothing about hardware.
"""

from __future__ import annotations

import copy
import hashlib
import itertools
import json
import logging
import math
import os
import re
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

import cv2
import numpy as np

log = logging.getLogger(__name__)

SCHEMA_VERSION = 1
DEFAULT_DIRECTORY = Path(__file__).resolve().parents[2] / "config" / "profile_settings"
MAX_REFERENCE_WIDTH = 640

SECTION_NAMES = ("camera", "calibration_context", "markers", "wifi", "preferences")

# Keys that describe the LIVE situation or secrets. Matched case-insensitively at any depth, in any section.
FORBIDDEN_KEYS = frozenset({
    # opponent / tracking
    "opponent_roi", "opponent_track", "opponent_tracks", "opponent", "opponent_id", "roi", "track", "tracks",
    "tracking_state", "tracker_state", "target", "selected_opponent",
    # control
    "control_mode", "authority", "control_authority", "armed", "arm", "arming", "motors_running", "motor_command",
    "drive_command", "drive_enabled", "motors_enabled",
    # link / observation
    "wifi_status", "link_state", "link_status", "connected", "observation", "last_observation", "rssi",
    # session-only trust (the operator re-confirms alignment on every start)
    "world_coordinates_trusted", "alignment_confirmed", "trust", "trust_level", "confirmed_this_session",
    # credentials
    "password", "passphrase", "psk", "wifi_password", "token", "secret", "api_key",
})

_DEFAULTS: dict[str, dict[str, Any]] = {
    "camera": {"index": None, "device_name": None, "backend": None, "width": None, "height": None, "fps": None,
               "fourcc": None, "extra": {}},
    "calibration_context": {"camera_signature": None, "resolution": None, "frame_reference": None, "corner_pixels": None,
                            "arena_setup_id": None, "calibrated_at": None, "last_confirmed_at": None},
    "markers": {"aruco_dictionary": None, "self_tag_id": None, "layout_name": None, "layout_path": None,
                "layout_hash": None, "measured_offsets": {}, "tag_size_m": None},
    "wifi": {"host": None, "port": None, "preferred_ssid": None, "auto_connect": False, "last_endpoint_ok_at": None},
    "preferences": {},
}


class ProfileSettingsError(ValueError):
    """A bad profile name, an unknown section or key, a forbidden key, or a value of the wrong type."""


# ---------------------------------------------------------------------------------------------- names and values

_BAD_NAME_CHARS = re.compile(r'[\x00-\x1f<>:"/\\|?*]')


def check_profile_name(profile: str) -> str:
    """Same rule as ``arena_setup.check_profile_name`` (no path separators or colon, no leading dot), plus the
    characters Windows file names cannot hold. Raises ``ProfileSettingsError``."""
    if (not isinstance(profile, str) or not profile or profile.startswith(".") or _BAD_NAME_CHARS.search(profile)
            or profile.endswith((" ", ".")) or len(profile) > 64):
        raise ProfileSettingsError(f"bad profile name {profile!r}")
    return profile


def _json_value(value: Any, where: str) -> Any:
    """Plain JSON data (numpy scalars and tuples converted); raises for anything JSON cannot hold, and for NaN/inf."""
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, (int,)):
        return int(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ProfileSettingsError(f"{where}: not a finite number")
        return float(value)
    if isinstance(value, np.generic):
        return _json_value(value.item(), where)
    if isinstance(value, np.ndarray):
        return _json_value(value.tolist(), where)
    if isinstance(value, Mapping):
        return {str(k): _json_value(v, f"{where}.{k}") for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(v, f"{where}[{i}]") for i, v in enumerate(value)]
    raise ProfileSettingsError(f"{where}: {type(value).__name__} cannot be stored")


def find_forbidden_keys(value: Any, where: str = "") -> list[str]:
    """Paths of every forbidden key anywhere inside ``value`` (dicts and lists, any depth)."""
    found: list[str] = []
    if isinstance(value, Mapping):
        for key, inner in value.items():
            path = f"{where}.{key}" if where else str(key)
            if str(key).strip().lower() in FORBIDDEN_KEYS:
                found.append(path)
            found.extend(find_forbidden_keys(inner, path))
    elif isinstance(value, (list, tuple)):
        for i, inner in enumerate(value):
            found.extend(find_forbidden_keys(inner, f"{where}[{i}]"))
    return found


def _opt(kind: str, check: Callable[[Any], bool], numeric: bool = False) -> Callable[[Any, str], Any]:
    def validate(value: Any, where: str) -> Any:
        if value is None:
            return None
        if numeric and isinstance(value, bool):                  # True is an int in Python; it is not a port or a size
            raise ProfileSettingsError(f"{where}: expected {kind}, got a boolean")
        if not check(value):
            raise ProfileSettingsError(f"{where}: expected {kind}, got {value!r}")
        return value
    return validate


_str = _opt("text", lambda v: isinstance(v, str))
_posint = _opt("a positive whole number", lambda v: isinstance(v, int) and v > 0, True)
_index = _opt("a whole number >= 0", lambda v: isinstance(v, int) and v >= 0, True)
_posnum = _opt("a positive number", lambda v: isinstance(v, (int, float)) and v > 0, True)
_port = _opt("a port 1..65535", lambda v: isinstance(v, int) and 1 <= v <= 65535, True)
_dict = _opt("an object", lambda v: isinstance(v, dict))
_resolution = _opt("[width, height]", lambda v: isinstance(v, list) and len(v) == 2 and all(isinstance(n, int) and n > 0 for n in v))


def _bool(value: Any, where: str) -> bool:
    if not isinstance(value, bool):
        raise ProfileSettingsError(f"{where}: expected true or false, got {value!r}")
    return value


def _corners(value: Any, where: str) -> Any:
    """Corner pixels as a list of [x, y] or a {name: [x, y]} mapping."""
    if value is None:
        return None

    def point(p: Any, at: str) -> list[float]:
        if not (isinstance(p, list) and len(p) == 2 and all(isinstance(n, (int, float)) and not isinstance(n, bool) for n in p)):
            raise ProfileSettingsError(f"{at}: expected [x, y]")
        return [float(p[0]), float(p[1])]

    if isinstance(value, dict):
        return {k: point(p, f"{where}.{k}") for k, p in value.items()}
    if isinstance(value, list):
        return [point(p, f"{where}[{i}]") for i, p in enumerate(value)]
    raise ProfileSettingsError(f"{where}: expected a list or mapping of [x, y]")


_VALIDATORS: dict[str, dict[str, Callable[[Any, str], Any]]] = {
    "camera": {"index": _index, "device_name": _str, "backend": _str, "width": _posint, "height": _posint, "fps": _posnum,
               "fourcc": _str, "extra": lambda v, w: _dict(v, w) or {}},
    "calibration_context": {"camera_signature": _dict, "resolution": _resolution, "frame_reference": _str,
                            "corner_pixels": _corners, "arena_setup_id": _str, "calibrated_at": _str, "last_confirmed_at": _str},
    "markers": {"aruco_dictionary": _str, "self_tag_id": _index, "layout_name": _str, "layout_path": _str, "layout_hash": _str,
                "measured_offsets": lambda v, w: _dict(v, w) or {}, "tag_size_m": _posnum},
    "wifi": {"host": _str, "port": _port, "preferred_ssid": _str, "auto_connect": _bool, "last_endpoint_ok_at": _str},
}


def normalise_section(name: str, data: Mapping[str, Any]) -> dict[str, Any]:
    """Validated, JSON-clean copy of a section with every default filled in. Raises ``ProfileSettingsError``."""
    if name not in SECTION_NAMES:
        raise ProfileSettingsError(f"unknown section {name!r}; sections are {', '.join(SECTION_NAMES)}")
    if not isinstance(data, Mapping):
        raise ProfileSettingsError(f"section {name!r} must be a mapping")
    forbidden = find_forbidden_keys(data, name)
    if forbidden:
        raise ProfileSettingsError("not stored, because it is live state or a secret and is never persisted: " + ", ".join(forbidden))
    clean = _json_value(data, name)
    if name == "preferences":
        return clean
    validators = _VALIDATORS[name]
    unknown = sorted(set(clean) - set(validators))
    if unknown:
        raise ProfileSettingsError(f"unknown key(s) in section {name!r}: {', '.join(unknown)}")
    result = copy.deepcopy(_DEFAULTS[name])
    for key, value in clean.items():
        result[key] = validators[key](value, f"{name}.{key}")
    return result


def _default_section(name: str) -> dict[str, Any]:
    return copy.deepcopy(_DEFAULTS[name])


# ---------------------------------------------------------------------------------------------- result types

@dataclass(frozen=True)
class SaveResult:
    ok: bool
    error: str = ""
    path: Path | None = None

    def __bool__(self) -> bool:
        return self.ok


@dataclass(frozen=True)
class ReferenceFrame:
    """The stored reference picture: greyscale, at most 640 px wide. Corner pixels are in ORIGINAL-frame coordinates;
    ``original_size`` and ``scale`` (= stored width / original width) say how to relate the two."""

    image: np.ndarray
    stored_size: tuple[int, int]
    original_size: tuple[int, int]
    scale: float
    saved_at: str
    file: str


@dataclass
class LoadedSettings:
    profile: str
    sections: dict[str, dict[str, Any]]          # every section, defaults filled in
    saved_at: dict[str, str | None]              # per section; None when never saved
    present: frozenset[str]                      # sections that were really saved
    reference: dict[str, Any] | None = None      # the reference-frame record (file, sizes, scale, saved_at, sha256)
    load_problem: str = ""                       # "" when all is well (including a never-saved profile)
    status: str = "fresh"                        # "fresh", "ok", "restored_from_backup" or "defaults_after_problem"
    used_backup: bool = False
    document: dict[str, Any] = field(default_factory=dict, repr=False)


# ---------------------------------------------------------------------------------------------- the store

class ProfileSettingsStore:
    """Per-profile settings files. See the module docstring.

    ``directory``  default ``config/profile_settings``. Tests pass a temporary directory.
    ``clock``      callable returning an aware ``datetime`` (default: now, UTC); stamps every section's ``saved_at``.
    ``fsync`` / ``replace``  injectable for failure tests (default ``os.fsync`` / ``os.replace``).
    """

    def __init__(self, directory: Path | str | None = None, *, clock: Callable[[], datetime] | None = None,
                 fsync: Callable[[int], None] | None = None, replace: Callable[[Any, Any], None] | None = None) -> None:
        self.directory = Path(directory) if directory is not None else DEFAULT_DIRECTORY
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._fsync = fsync or os.fsync
        self._replace = replace or os.replace
        self._lock = threading.RLock()
        self._counter = itertools.count()

    # ---- paths
    def path_for(self, profile: str) -> Path:
        return self.directory / f"{check_profile_name(profile)}.json"

    def backup_path_for(self, profile: str) -> Path:
        return self.directory / f"{check_profile_name(profile)}.json.bak"

    def _stamp(self) -> str:
        return self._clock().astimezone(timezone.utc).isoformat()

    # ---- reading
    def _read_document(self, path: Path, profile: str) -> tuple[dict[str, Any] | None, str]:
        """(document, "") when the file is a good settings file of this profile, else (None, why not)."""
        try:
            raw = path.read_bytes()
        except FileNotFoundError:
            return None, f"{path.name} is missing"
        except OSError as error:
            return None, f"{path.name} could not be read ({error})"
        try:
            doc = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, ValueError) as error:
            return None, f"{path.name} is damaged ({type(error).__name__}: {error})"
        if not isinstance(doc, dict):
            return None, f"{path.name} is damaged (not a settings object)"
        if doc.get("version") != SCHEMA_VERSION:
            return None, f"{path.name} has format version {doc.get('version')!r}, expected {SCHEMA_VERSION}"
        if doc.get("profile") != profile:
            return None, (f"{path.name} belongs to profile {doc.get('profile')!r}, not {profile!r}: "
                          "profiles never share settings files")
        if not isinstance(doc.get("sections"), dict):
            return None, f"{path.name} is damaged (no sections)"
        return doc, ""

    def _clean_document(self, doc: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
        """Drop individual sections that fail validation (the others survive); returns (doc, notes)."""
        notes: list[str] = []
        sections: dict[str, Any] = {}
        for name, entry in doc["sections"].items():
            try:
                if name not in SECTION_NAMES:
                    raise ProfileSettingsError("unknown section")
                if not isinstance(entry, dict) or not isinstance(entry.get("data"), dict):
                    raise ProfileSettingsError("not a section record")
                sections[name] = {"saved_at": entry.get("saved_at") if isinstance(entry.get("saved_at"), str) else None,
                                  "data": normalise_section(name, entry["data"])}
            except ProfileSettingsError as error:
                notes.append(f"section '{name}' ignored ({error})")
        reference = doc.get("reference_frame")
        if reference is not None and not (isinstance(reference, dict) and isinstance(reference.get("file"), str)):
            notes.append("reference-frame record ignored (damaged)")
            reference = None
        return {"version": SCHEMA_VERSION, "profile": doc["profile"], "saved_at": doc.get("saved_at"),
                "sections": sections, "reference_frame": reference}, notes

    def load(self, profile: str) -> LoadedSettings:
        """Everything saved for ``profile``. Never raises for file problems (a bad profile NAME still raises)."""
        path, backup = self.path_for(profile), self.backup_path_for(profile)
        with self._lock:
            doc, main_problem = self._read_document(path, profile)
            used_backup = False
            problem = ""
            if doc is None:
                backup_doc, backup_problem = self._read_document(backup, profile)
                if backup_doc is not None:
                    doc, used_backup = backup_doc, True
                    problem = (f"{main_problem}; restored the previous good copy ({backup.name}). "
                               "The most recent change may be missing.")
                elif path.exists() or backup.exists():
                    problem = f"{main_problem}; no usable backup ({backup_problem}). Starting from defaults."
                # neither file exists: a never-saved profile, not a problem
            if doc is None:
                empty = {"version": SCHEMA_VERSION, "profile": profile, "saved_at": None, "sections": {}, "reference_frame": None}
                return self._build(profile, empty, problem, "defaults_after_problem" if problem else "fresh", False)
            clean, notes = self._clean_document(doc)
            if notes:
                problem = "; ".join(filter(None, [problem, *notes]))
            status = "restored_from_backup" if used_backup else "ok"        # "ok" with a problem = a damaged section was dropped
            return self._build(profile, clean, problem, status, used_backup)

    def _build(self, profile: str, doc: dict[str, Any], problem: str, status: str, used_backup: bool) -> LoadedSettings:
        sections = {name: _default_section(name) for name in SECTION_NAMES}
        saved_at: dict[str, str | None] = {name: None for name in SECTION_NAMES}
        for name, entry in doc["sections"].items():
            sections[name] = copy.deepcopy(entry["data"])
            saved_at[name] = entry.get("saved_at")
        if problem:
            log.warning("profile settings '%s': %s", profile, problem)
        return LoadedSettings(profile, sections, saved_at, frozenset(doc["sections"]), copy.deepcopy(doc.get("reference_frame")),
                              problem, status, used_backup, doc)

    def load_problem(self, profile: str) -> str:
        return self.load(profile).load_problem

    def get_section(self, profile: str, name: str) -> dict[str, Any]:
        """A copy of one section (defaults for anything not saved). Never raises for file problems."""
        if name not in SECTION_NAMES:
            raise ProfileSettingsError(f"unknown section {name!r}")
        return self.load(profile).sections[name]

    def has_section(self, profile: str, name: str) -> bool:
        return name in self.load(profile).present

    def section_saved_at(self, profile: str, name: str) -> str | None:
        """ISO-8601 time that section was last written, or None if it never was."""
        if name not in SECTION_NAMES:
            raise ProfileSettingsError(f"unknown section {name!r}")
        return self.load(profile).saved_at[name]

    # ---- writing
    def set_section(self, profile: str, name: str, data: Mapping[str, Any]) -> SaveResult:
        """Replace one section (keys not given go back to their defaults). Other sections are untouched.

        Raises ``ProfileSettingsError`` (a ``ValueError``) for an unknown section or key, a wrong type, or a
        forbidden key at any depth. A disk failure does not raise: it returns ``SaveResult(ok=False, error=...)``
        and the old file stays exactly as it was.
        """
        return self.set_sections(profile, {name: data})

    def update_section(self, profile: str, name: str, changes: Mapping[str, Any]) -> SaveResult:
        """Merge ``changes`` into one section (keys not mentioned keep their saved values)."""
        with self._lock:
            merged = self.get_section(profile, name)
            merged.update(dict(changes))
            return self.set_sections(profile, {name: merged})

    def set_sections(self, profile: str, sections: Mapping[str, Mapping[str, Any]]) -> SaveResult:
        """Replace several sections in ONE atomic file write."""
        clean = {name: normalise_section(name, data) for name, data in sections.items()}
        stamp = self._stamp()

        def mutate(doc: dict[str, Any]) -> None:
            for name, data in clean.items():
                doc["sections"][name] = {"saved_at": stamp, "data": data}

        return self._commit(profile, mutate)

    def clear_section(self, profile: str, name: str) -> SaveResult:
        """Forget one section (it reads as defaults again)."""
        if name not in SECTION_NAMES:
            raise ProfileSettingsError(f"unknown section {name!r}")
        return self._commit(profile, lambda doc: doc["sections"].pop(name, None))

    def _commit(self, profile: str, mutate: Callable[[dict[str, Any]], None],
                after_success: Callable[[], None] | None = None) -> SaveResult:
        path = self.path_for(profile)
        with self._lock:
            loaded = self.load(profile)
            doc = copy.deepcopy(loaded.document)
            doc.setdefault("sections", {})
            doc.setdefault("reference_frame", None)
            mutate(doc)
            doc["version"], doc["profile"], doc["saved_at"] = SCHEMA_VERSION, profile, self._stamp()
            try:
                payload = json.dumps(doc, indent=2, sort_keys=True, allow_nan=False).encode("utf-8")
            except (TypeError, ValueError) as error:
                return SaveResult(False, f"could not encode the settings ({error})", path)
            result = self._write_with_backup(path, profile, payload)
            if result.ok and after_success is not None:
                try:
                    after_success()
                except OSError as error:                       # clean-up only; the save itself succeeded
                    log.warning("profile settings clean-up failed: %s", error)
            return result

    def _temp_name(self, path: Path) -> Path:
        return path.with_name(f".{path.name}.{os.getpid()}.{next(self._counter)}.tmp")

    def _atomic_write(self, path: Path, payload: bytes) -> None:
        """temp file in the same directory -> flush -> fsync -> os.replace. Leaves no temp file behind on failure."""
        temp = self._temp_name(path)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(temp, "wb") as handle:
                handle.write(payload)
                handle.flush()
                self._fsync(handle.fileno())
            self._replace(temp, path)
        finally:
            try:
                if temp.exists():
                    temp.unlink()
            except OSError:
                pass

    def _write_with_backup(self, path: Path, profile: str, payload: bytes) -> SaveResult:
        try:
            # Keep the previous GOOD file as .bak before replacing it. A damaged current file never overwrites a good .bak.
            current, _ = self._read_document(path, profile)
            if current is not None:
                try:
                    self._atomic_write(self.backup_path_for(profile), path.read_bytes())
                except OSError as error:
                    log.warning("could not refresh %s: %s", self.backup_path_for(profile).name, error)
            self._atomic_write(path, payload)
        except Exception as error:                              # disk full, permission, injected failure ...
            message = f"could not save {path.name}: {error}"
            log.error(message)
            return SaveResult(False, message, path)
        return SaveResult(True, "", path)

    # ---- reference frame
    def _reference_files(self, profile: str) -> list[Path]:
        pattern = re.compile(rf"{re.escape(check_profile_name(profile))}\.reference-[0-9a-f]{{12}}\.png")
        if not self.directory.exists():
            return []
        return [p for p in self.directory.iterdir() if pattern.fullmatch(p.name)]

    @staticmethod
    def _grey(frame: np.ndarray) -> np.ndarray:
        if not isinstance(frame, np.ndarray) or frame.ndim not in (2, 3) or frame.size == 0:
            raise ProfileSettingsError("the reference frame must be a non-empty image array")
        image = frame
        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)
        if image.ndim == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY if image.shape[2] == 4 else cv2.COLOR_BGR2GRAY)
        return image

    def _stage_reference(self, profile: str, frame: np.ndarray) -> tuple[dict[str, Any], Path, bytes]:
        grey = self._grey(frame)
        height, width = grey.shape
        if width > MAX_REFERENCE_WIDTH:
            small = cv2.resize(grey, (MAX_REFERENCE_WIDTH, max(1, int(round(height * MAX_REFERENCE_WIDTH / width)))), interpolation=cv2.INTER_AREA)
        else:
            small = grey
        ok, encoded = cv2.imencode(".png", small)
        if not ok:
            raise ProfileSettingsError("the reference frame could not be encoded")
        data = bytes(encoded.tobytes())
        digest = hashlib.sha256(data).hexdigest()
        name = f"{profile}.reference-{digest[:12]}.png"
        record = {"file": name, "stored_size": [int(small.shape[1]), int(small.shape[0])], "original_size": [int(width), int(height)],
                  "scale": float(small.shape[1]) / float(width), "saved_at": self._stamp(), "sha256": digest}
        return record, self.directory / name, data

    def save_reference_frame(self, profile: str, bgr_array: np.ndarray) -> SaveResult:
        """Store ``bgr_array`` (BGR, BGRA or grey) as this profile's reference picture: greyscale, at most 640 px wide,
        written atomically next to the JSON, and recorded with its stored size and the ORIGINAL frame size."""
        return self._save_reference_and_sections(profile, bgr_array, {})

    def _save_reference_and_sections(self, profile: str, frame: np.ndarray | None, sections: Mapping[str, Mapping[str, Any]],
                                     fill_reference_name: str | None = None) -> SaveResult:
        clean = {name: normalise_section(name, data) for name, data in sections.items()}
        with self._lock:
            old_files = set(self._reference_files(profile))
            record = None
            if frame is not None:
                record, png_path, png_bytes = self._stage_reference(profile, frame)
                try:
                    if not png_path.exists():
                        self._atomic_write(png_path, png_bytes)
                except Exception as error:
                    return SaveResult(False, f"could not save the reference picture: {error}", png_path)
                if fill_reference_name and fill_reference_name in clean:
                    clean[fill_reference_name]["frame_reference"] = record["file"]
            stamp = self._stamp()

            def mutate(doc: dict[str, Any]) -> None:
                for name, data in clean.items():
                    doc["sections"][name] = {"saved_at": stamp, "data": data}
                doc["reference_frame"] = record

            def clean_up() -> None:
                keep = None if record is None else self.directory / record["file"]
                for old in old_files | set(self._reference_files(profile)):
                    if old != keep:
                        old.unlink(missing_ok=True)

            return self._commit(profile, mutate, after_success=clean_up)

    def save_calibration_bundle(self, profile: str, context: Mapping[str, Any], frame: np.ndarray | None) -> SaveResult:
        """Save a ``calibration_context`` section AND its reference picture in ONE atomic settings write.

        The context's ``frame_reference`` is filled in with the stored picture's file name. With ``frame=None`` the
        context is saved without a reference picture and any older reference is dropped (it would not match the new corners)."""
        return self._save_reference_and_sections(profile, frame, {"calibration_context": context}, "calibration_context")

    def load_reference_frame(self, profile: str) -> ReferenceFrame | None:
        """The stored reference picture, or None (never saved, file missing, or its hash no longer matches the record).
        ``reference_problem`` says which."""
        frame, _ = self._load_reference(profile)
        return frame

    def reference_problem(self, profile: str) -> str:
        """Why ``load_reference_frame`` returned None ("" when there is no reference at all, or it loaded fine)."""
        return self._load_reference(profile)[1]

    def _load_reference(self, profile: str) -> tuple[ReferenceFrame | None, str]:
        record = self.load(profile).reference
        if record is None:
            return None, ""
        try:
            file = str(record["file"])
            if "/" in file or "\\" in file or not file.startswith(f"{profile}.reference-"):
                return None, "the reference-picture record points at a file that does not belong to this profile"
            data = (self.directory / file).read_bytes()
        except FileNotFoundError:
            return None, "the stored reference picture file is missing"
        except (OSError, KeyError, TypeError) as error:
            return None, f"the stored reference picture could not be read ({error})"
        if hashlib.sha256(data).hexdigest() != record.get("sha256"):
            return None, "the stored reference picture does not match its record (damaged)"
        image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_GRAYSCALE)
        if image is None:
            return None, "the stored reference picture could not be decoded"
        try:
            original = (int(record["original_size"][0]), int(record["original_size"][1]))
            stored = (int(image.shape[1]), int(image.shape[0]))
            scale = float(record.get("scale") or stored[0] / original[0])
        except (KeyError, TypeError, ValueError, IndexError, ZeroDivisionError):
            return None, "the reference-picture record is incomplete"
        return ReferenceFrame(image, stored, original, scale, str(record.get("saved_at") or ""), file), ""
