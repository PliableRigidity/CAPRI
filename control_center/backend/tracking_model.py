"""Pure (Qt-free, camera-free) model behind the ROBOT TRACKING & LIVE MONITORING screen.

Everything here is arithmetic and wording on numbers the trackers already measured. Nothing in this
file opens a camera, a port or a socket, and nothing in it can command the robot.

  MarkerRateTracker   which ids decode, how often over the last ~5 s (the number that quantifies intermittent detection)
  tracking_confidence a label derived ONLY from measured quantities, returned together with those quantities
  marker_spread       disagreement between the markers the own pose was fused from
  WarningMonitor      debounced warnings (so a flickering condition does not spam the event log)
  TrailBuffer         short position trails whose gaps stay gaps
  DiagnosticsModel    software ms per frame, processed fps, skipped frames, queue age
  shadow_*            the prerequisites of the display-only SHADOW MODE and the words for its result
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

# ---------------------------------------------------------------- vocabulary

LEVELS = ("ok", "warn", "bad", "off", "plain")

OWN_STATE_LEVEL = {"measured": "ok", "predicted": "warn", "stale": "bad", "lost": "bad"}
OPPONENT_STATE_LEVEL = {"tracked": "ok", "ambiguous": "warn", "uncertain": "warn", "lost": "bad", "not_selected": "off"}

NOT_CALIBRATED_TEXT = "NOT CONFIGURED: calibrate the arena first (ARENA tab)"
NO_IMU_STATEMENT = ("The baseline firmware sends no IMU data, so on the competition robot SHADOW MODE will currently stay "
                    "BLOCKED. Nothing here ever invents an IMU value.")
SHADOW_ONLY_STATEMENT = "SHADOW MODE is display only: the policy's choice is shown here and is never sent to the robot."
OPPONENT_HINT = ("DRAG a box around ANY object in the camera view to follow it (a robot, a ball, a phone): there is no size limit. "
                  "TRACKING UNCERTAIN gives no position and may recover by itself; LOST needs a new box. Drag again at any time "
                  "to change the selection. A box on your own tagged robot is refused. Positions are image-based and approximate.")
CONFIDENCE_NOTE = ("Confidence is a label computed only from the measured inputs shown beside it (state, age, markers "
                   "agreeing, fit rms, spread). It is not a probability and not an accuracy figure.")

#: A confidence input counts against the pose once it uses more than this share of the tracker's own agreement limit.
DOWNGRADE_FRACTION = 0.5


# ---------------------------------------------------------------- marker table


@dataclass(frozen=True)
class MarkerRow:
    marker_id: int
    face: str                  # "top" / "bottom" / "rear", or "?" for an id that is not configured
    configured: bool
    seen_now: bool
    frames_seen: int           # frames in the window that contained it
    frames_total: int          # frames in the window
    rate: float | None         # frames_seen / frames_total; None before any frame
    ever_seen: bool
    status: str
    level: str                 # one of LEVELS

    @property
    def rate_text(self) -> str:
        if self.rate is None or not self.frames_total:
            return "-"
        return f"{100.0 * self.rate:.0f}%  ({self.frames_seen}/{self.frames_total})"


class MarkerRateTracker:
    """Detection rate per id over the newest ``window_s`` seconds of frames (counted per frame, not per second)."""

    INTERMITTENT_BELOW = 0.90
    MIN_FRAMES_FOR_RATE_FLAG = 15

    def __init__(self, markers: Mapping[int, Any], window_s: float = 5.0) -> None:
        self.markers = dict(markers)
        self.window_s = float(window_s)
        self._frames: deque[tuple[float, frozenset[int]]] = deque()
        self._ever: set[int] = set()

    def reset(self) -> None:
        self._frames.clear()
        self._ever.clear()

    def update(self, t: float, ids: Iterable[int]) -> None:
        seen = frozenset(int(i) for i in ids)
        if self._frames and t < self._frames[-1][0]:
            self._frames.clear()                                  # time went backwards: a new stream, start the window again
        self._frames.append((float(t), seen))
        self._ever.update(seen)
        while len(self._frames) > 1 and t - self._frames[0][0] > self.window_s:
            self._frames.popleft()

    @property
    def frames_in_window(self) -> int:
        return len(self._frames)

    def rows(self, expected_faces: Sequence[str] | None = None) -> list[MarkerRow]:
        """Configured ids first (ascending), then any unknown id that has been seen.

        ``expected_faces``: the faces that can be in view for the robot's current orientation (e.g. ("top", "rear")
        while upright). A configured id on another face that was never seen is then "not expected" rather than a fault.
        """
        total = len(self._frames)
        latest = self._frames[-1][1] if self._frames else frozenset()
        counts: dict[int, int] = {}
        for _, ids in self._frames:
            for i in ids:
                counts[i] = counts.get(i, 0) + 1
        rows: list[MarkerRow] = []
        for marker_id in sorted(self.markers):
            face = str(getattr(self.markers[marker_id], "face", "?"))
            rows.append(self._row(marker_id, face, True, marker_id in latest, counts.get(marker_id, 0), total, expected_faces))
        for marker_id in sorted(self._ever - set(self.markers)):
            rows.append(self._row(marker_id, "?", False, marker_id in latest, counts.get(marker_id, 0), total, expected_faces))
        return rows

    def _row(self, marker_id: int, face: str, configured: bool, seen_now: bool, seen: int, total: int,
             expected_faces: Sequence[str] | None) -> MarkerRow:
        rate = None if not total else seen / total
        ever = marker_id in self._ever
        expected = expected_faces is None or face in expected_faces
        if not configured:
            status, level = "UNKNOWN ID: not in the marker layout", "bad"
        elif seen_now:
            intermittent = (rate is not None and total >= self.MIN_FRAMES_FOR_RATE_FLAG and rate < self.INTERMITTENT_BELOW)
            status, level = ("seen (INTERMITTENT in the last few seconds)", "warn") if intermittent else ("seen", "ok")
        elif not ever:
            status, level = (("never seen since tracking started", "warn") if expected
                             else ("never seen (its face is not expected to be in view)", "off"))
        else:
            status, level = (("not seen this frame", "warn") if expected else ("not seen (its face is not expected in view)", "off"))
        return MarkerRow(marker_id, face, configured, seen_now, seen, total, rate, ever, status, level)

    def unknown_ids(self) -> list[int]:
        return sorted(self._ever - set(self.markers))

    def never_seen_ids(self) -> list[int]:
        return sorted(set(self.markers) - self._ever)


# ---------------------------------------------------------------- confidence


@dataclass(frozen=True)
class Confidence:
    level: str                                   # "HIGH", "MEDIUM", "LOW" or "NONE"
    inputs: list[tuple[str, str]]                # (name, value text) of every measured quantity used
    reasons: list[str] = field(default_factory=list)

    @property
    def ui_level(self) -> str:
        return {"HIGH": "ok", "MEDIUM": "warn", "LOW": "bad", "NONE": "bad"}[self.level]

    def text(self) -> str:
        return self.level + ("" if not self.reasons else "  (" + "; ".join(self.reasons) + ")")


def marker_spread(details: Sequence[Mapping[str, Any]], used_ids: Sequence[int]) -> tuple[float | None, float | None]:
    """(position spread mm, heading spread deg) between the per-marker pose estimates of the used face markers.

    Each marker gives its own estimate of the chassis centre and heading; with correct offsets they agree. The spread is the
    largest pairwise distance / angle. None with fewer than two markers: one marker has nothing to disagree with.
    """
    chosen = [d for d in details if d.get("id") in set(used_ids) and d.get("bot_x_m") is not None]
    if len(chosen) < 2:
        return None, None
    position = max(math.hypot(a["bot_x_m"] - b["bot_x_m"], a["bot_y_m"] - b["bot_y_m"]) for a in chosen for b in chosen)
    heading = 0.0
    for a in chosen:
        for b in chosen:
            turn = abs((a["heading_deg"] - b["heading_deg"] + 180.0) % 360.0 - 180.0)
            heading = max(heading, turn)
    return 1000.0 * position, heading


def tracking_confidence(own: Mapping[str, Any] | None, evidence: Mapping[str, Any] | None, position_limit_m: float,
                        heading_limit_deg: float, predict_hold_s: float) -> Confidence:
    """A label from measured quantities only. ``evidence``: the newest frame's fit_rms_mm / spread_mm / heading_spread_deg."""
    inputs: list[tuple[str, str]] = []
    if own is None:
        return Confidence("NONE", [("tracker", "no report yet")], ["no tracker report"])
    state = own["state"]
    used = list(own.get("used_ids") or [])
    age = own.get("age_s")
    inputs.append(("tracker state", state.upper()))
    inputs.append(("age of last measurement", "-" if age is None else f"{1000 * age:.0f} ms"))
    inputs.append(("markers agreeing", f"{len(used)} {used}" if used else "0"))
    evidence = evidence or {}
    rms, spread, turn = evidence.get("fit_rms_mm"), evidence.get("spread_mm"), evidence.get("heading_spread_deg")
    measured_now = state == "measured"
    inputs.append(("marker fit rms", "-" if rms is None else f"{rms:.1f} mm") if measured_now
                  else ("marker fit rms", "n/a (not measured this frame)"))
    inputs.append(("marker position spread", ("n/a (one marker)" if spread is None else f"{spread:.1f} mm") if measured_now
                   else "n/a (not measured this frame)"))
    inputs.append(("marker heading spread", ("n/a (one marker)" if turn is None else f"{turn:.1f} deg") if measured_now
                   else "n/a (not measured this frame)"))
    inputs.append(("position quality", str(own.get("position_quality", "unknown"))))
    if state == "lost":
        return Confidence("NONE", inputs, ["no current estimate"])
    if state == "stale":
        return Confidence("NONE", inputs, ["stale: last measured pose, display only"])
    order = ["LOW", "MEDIUM", "HIGH"]
    level = 2 if state == "measured" else 1
    reasons: list[str] = []
    if state == "predicted":
        reasons.append("predicted, not measured")
    position_limit_mm = 1000.0 * position_limit_m
    if len(used) <= 1:
        level -= 1
        reasons.append("only one marker, nothing to agree with")
    if own.get("source") == "rear marker only":
        level -= 1
        reasons.append("rear marker only")
    if measured_now and rms is not None and rms > DOWNGRADE_FRACTION * position_limit_mm:
        level -= 1
        reasons.append(f"fit rms {rms:.1f} mm > {DOWNGRADE_FRACTION * position_limit_mm:.0f} mm")
    if measured_now and spread is not None and spread > DOWNGRADE_FRACTION * position_limit_mm:
        level -= 1
        reasons.append(f"markers disagree by {spread:.1f} mm")
    if measured_now and turn is not None and turn > DOWNGRADE_FRACTION * heading_limit_deg:
        level -= 1
        reasons.append(f"markers disagree by {turn:.1f} deg")
    if state == "predicted" and age is not None and age > DOWNGRADE_FRACTION * predict_hold_s:
        level -= 1
        reasons.append(f"prediction is {1000 * age:.0f} ms old")
    return Confidence(order[max(0, level)], inputs, reasons)


# ---------------------------------------------------------------- warnings


@dataclass(frozen=True)
class Condition:
    key: str
    text: str
    severity: str            # "bad" or "warn"


@dataclass(frozen=True)
class WarningEvent:
    key: str
    text: str
    severity: str


class WarningMonitor:
    """Turns "this is wrong right now" into a banner and at most one event-log line per episode.

    A condition becomes ACTIVE only after it has held for ``on_s``; it stays active until it has been absent for
    ``off_s``; and a key is logged at most once per ``cooldown_s``. A flickering condition therefore shows one banner
    and one log line, not one per flicker.
    """

    def __init__(self, on_s: float = 0.4, off_s: float = 1.0, cooldown_s: float = 15.0) -> None:
        self.on_s, self.off_s, self.cooldown_s = on_s, off_s, cooldown_s
        self._since: dict[str, float] = {}
        self._last_true: dict[str, float] = {}
        self._active: dict[str, Condition] = {}
        self._logged_at: dict[str, float] = {}
        self.log: list[WarningEvent] = []

    def reset(self) -> None:
        self._since.clear(), self._last_true.clear(), self._active.clear(), self._logged_at.clear()

    @property
    def active(self) -> list[Condition]:
        return sorted(self._active.values(), key=lambda c: (c.severity != "bad", c.key))

    def update(self, now: float, conditions: Mapping[str, Condition]) -> list[WarningEvent]:
        events: list[WarningEvent] = []
        for key, condition in conditions.items():
            self._last_true[key] = now
            if key in self._active:
                self._active[key] = condition              # wording may change while it stays active
                continue
            since = self._since.setdefault(key, now)
            if now - since >= self.on_s:
                self._active[key] = condition
                last = self._logged_at.get(key)
                if last is None or now - last >= self.cooldown_s:
                    self._logged_at[key] = now
                    events.append(WarningEvent(key, condition.text, condition.severity))
        for key in list(self._since):
            if key not in conditions and key not in self._active:
                del self._since[key]                         # it went away before it was ever raised
        for key in list(self._active):
            if key not in conditions and now - self._last_true.get(key, now) >= self.off_s:
                del self._active[key]
                self._since.pop(key, None)
        self.log.extend(events)
        return events


# ---------------------------------------------------------------- trails


class TrailBuffer:
    """Recent positions as separate runs. ``None`` is a gap: the line is never drawn across it."""

    def __init__(self, keep_s: float = 10.0, max_points: int = 600) -> None:
        self.keep_s, self.max_points = keep_s, max_points
        self._points: deque[tuple[float, tuple[float, float] | None]] = deque()

    def clear(self) -> None:
        self._points.clear()

    def add(self, t: float, xy: tuple[float, float] | None) -> None:
        if xy is None and (not self._points or self._points[-1][1] is None):
            return                                           # one gap marker is enough
        self._points.append((t, None if xy is None else (float(xy[0]), float(xy[1]))))
        while self._points and (t - self._points[0][0] > self.keep_s or len(self._points) > self.max_points):
            self._points.popleft()

    def runs(self) -> list[list[tuple[float, float]]]:
        runs: list[list[tuple[float, float]]] = []
        current: list[tuple[float, float]] = []
        for _, xy in self._points:
            if xy is None:
                if current:
                    runs.append(current)
                current = []
            else:
                current.append(xy)
        if current:
            runs.append(current)
        return runs


# ---------------------------------------------------------------- diagnostics


class DiagnosticsModel:
    """Software timing, processed rate and skipped frames. Wall time is passed in so tests need no clock."""

    def __init__(self, window_s: float = 3.0, keep: int = 120) -> None:
        self.window_s = window_s
        self._ms: deque[float] = deque(maxlen=keep)
        self._done: deque[float] = deque()
        self.processed = 0
        self.skipped_total = 0
        self.queue_age_s: float | None = None
        self.last_index: int | None = None

    def reset(self) -> None:
        self._ms.clear(), self._done.clear()
        self.processed = self.skipped_total = 0
        self.queue_age_s = self.last_index = None

    def record(self, done_at: float, frame_index: int, software_ms: float, queue_age_s: float, skipped: int) -> None:
        self._ms.append(float(software_ms))
        self._done.append(float(done_at))
        while self._done and done_at - self._done[0] > self.window_s:
            self._done.popleft()
        self.processed += 1
        self.skipped_total += max(0, int(skipped))
        self.queue_age_s, self.last_index = float(queue_age_s), frame_index

    def summary(self, now: float) -> dict[str, float | int | None]:
        recent = [t for t in self._done if now - t <= self.window_s]
        fps = None
        if len(recent) >= 2 and recent[-1] > recent[0]:
            fps = (len(recent) - 1) / (recent[-1] - recent[0])
        values = sorted(self._ms)
        return {"ms_mean": (sum(values) / len(values)) if values else None,
                "ms_p95": values[min(len(values) - 1, int(math.ceil(0.95 * len(values))) - 1)] if values else None,
                "ms_last": self._ms[-1] if self._ms else None,
                "fps": fps, "processed": self.processed, "skipped_total": self.skipped_total,
                "queue_age_ms": None if self.queue_age_s is None else 1000.0 * self.queue_age_s}


# ---------------------------------------------------------------- arena geometry


def pit_list(arena: Mapping[str, Any]) -> list[tuple[tuple[float, float], float]]:
    """[(centre, radius)] per pit: ``pit_radii_m`` when the profile has one radius per pit, else ``pit_radius_m`` for all."""
    centres = [tuple(float(v) for v in c) for c in arena.get("pit_centres_m") or []]
    radii = arena.get("pit_radii_m")
    if radii is not None and len(list(radii)) == len(centres):
        return [(c, float(r)) for c, r in zip(centres, radii)]
    radius = float(arena.get("pit_radius_m", 0.0))
    return [(c, radius) for c in centres]


def arena_problems(arena: Mapping[str, Any] | None) -> list[str]:
    """Why the arena geometry cannot be drawn or used; empty when it is usable."""
    if not arena:
        return ["no arena geometry"]
    problems: list[str] = []
    for key in ("width_m", "height_m"):
        try:
            if not float(arena[key]) > 0.0:
                problems.append(f"arena.{key} must be positive")
        except (KeyError, TypeError, ValueError):
            problems.append(f"arena.{key} is missing")
    if not pit_list(arena):
        problems.append("no pits in the arena geometry")
    for centre, radius in pit_list(arena):
        if not radius > 0.0:
            problems.append("a pit radius is not positive")
            break
    vertices = arena.get("flipper_vertices_m")
    if vertices is None or len(list(vertices)) < 3:
        problems.append("the fixed flipper outline has fewer than 3 vertices")
    return problems


# ---------------------------------------------------------------- shadow mode


@dataclass(frozen=True)
class Prerequisite:
    key: str
    label: str
    ok: bool
    detail: str
    cheap: bool = True        # decided before the checkpoint is loaded; the rest are live inputs

    @property
    def word(self) -> str:
        return "Ready" if self.ok else "MISSING"


def frame_compatibility(arena: Mapping[str, Any] | None, profile: str, contract_size_m: tuple[float, float] | None,
                        tolerance_m: float = 0.005) -> tuple[bool, str]:
    """Is this profile's arena the simulator's frame? A False ``frame_ok`` / ``policy_frame_ok`` flag refuses shadow."""
    arena = arena or {}
    for key in ("policy_frame_ok", "frame_ok"):
        if key in arena and not bool(arena[key]):
            return False, (f"profile '{profile}' is not laid out in the simulator's arena frame ({key} is false): the policy "
                           "would be fed positions from another layout, so shadow is refused")
    for key in ("policy_frame_ok", "frame_ok"):
        if key in arena:
            return True, f"profile '{profile}' is flagged as the simulator's arena frame ({key})"
    if contract_size_m is None:
        return False, "the checkpoint's arena size is unknown, so the profile layout cannot be compared"
    try:
        width, height = float(arena["width_m"]), float(arena["height_m"])
    except (KeyError, TypeError, ValueError):
        return False, "the profile has no arena size"
    if abs(width - contract_size_m[0]) > tolerance_m or abs(height - contract_size_m[1]) > tolerance_m:
        return False, (f"profile '{profile}' arena is {width:.3f} x {height:.3f} m, the checkpoint was trained on "
                       f"{contract_size_m[0]:.3f} x {contract_size_m[1]:.3f} m: not the simulator's frame")
    return True, (f"profile '{profile}' arena size matches the checkpoint's ({width:.3f} x {height:.3f} m); "
                  "no frame flag in the profile, so the origin convention is assumed")


def shadow_prerequisites(*, frame_ok: bool, frame_detail: str, mapping_ok: bool, mapping_detail: str, contract_ok: bool,
                         contract_detail: str, imu_present: bool, imu_present_detail: str, imu_fresh: bool, imu_fresh_detail: str,
                         mounting_verified: bool, mounting_detail: str, own_ok: bool, own_detail: str, opponent_ok: bool,
                         opponent_detail: str, checkpoint_state: str, checkpoint_detail: str) -> list[Prerequisite]:
    return [
        Prerequisite("imu_present", "IMU telemetry present", imu_present, imu_present_detail),
        Prerequisite("imu_fresh", "IMU telemetry fresh", imu_fresh, imu_fresh_detail),
        Prerequisite("mounting", "IMU mounting verified", mounting_verified, mounting_detail),
        Prerequisite("mapping", "Arena mapping valid", mapping_ok, mapping_detail),
        Prerequisite("frame", "Profile layout matches the simulator frame", frame_ok, frame_detail),
        Prerequisite("contract", "Checkpoint record readable", contract_ok, contract_detail),
        Prerequisite("own", "Own pose valid", own_ok, own_detail, cheap=False),
        Prerequisite("opponent", "Opponent position valid", opponent_ok, opponent_detail, cheap=False),
        Prerequisite("checkpoint", "Checkpoint loaded", checkpoint_state == "loaded", checkpoint_detail, cheap=False),
    ]


def shadow_checkbox_state(prereqs: Sequence[Prerequisite]) -> tuple[bool, str]:
    """(enabled, why not). Only the cheap prerequisites gate the checkbox; the live inputs gate each proposal."""
    missing = [p for p in prereqs if p.cheap and not p.ok]
    if not missing:
        return True, "all prerequisites that can be checked before loading hold"
    return False, "blocked: " + "; ".join(f"{p.label}: {p.detail}" for p in missing)


def shadow_headline(enabled: bool, load_state: str, summary: Mapping[str, Any] | None) -> tuple[str, str, list[str]]:
    """(headline, level, detail lines) for the shadow result.

    ``summary`` is what the worker saw at its newest decision: {"proposed": {...}|None, "blocked_by": [...],
    "policy_loaded": bool, "error": str}. "PROPOSED" appears only when the assembler produced a proposal.
    """
    if not enabled:
        return "SHADOW OFF (default)", "off", []
    if summary is None:
        return "SHADOW ON: waiting for the first decision", "warn", []
    if summary.get("error"):
        return "SHADOW ERROR: " + str(summary["error"]), "bad", []
    proposed = summary.get("proposed")
    if proposed is not None:
        return (f"PROPOSED {proposed['tactic']} (p={float(proposed['probability']):.2f})", "ok",
                ["display only: this is what the checkpoint would choose; nothing is sent to the robot"])
    blocked = list(summary.get("blocked_by") or [])
    if blocked:
        return "BLOCKED: no proposal", "bad", blocked
    if load_state == "loading":
        return "NO PROPOSAL: all inputs valid, the checkpoint is still loading", "warn", []
    if load_state == "failed":
        return "NO PROPOSAL: all inputs valid, the checkpoint failed to load", "bad", []
    if not summary.get("policy_loaded"):
        return "NO PROPOSAL: all inputs valid, but no checkpoint is loaded", "warn", []
    return "NO PROPOSAL: all inputs valid, waiting for the checkpoint's answer", "warn", []


# ---------------------------------------------------------------- text for the panels


ORIENTATION_TEXT = {"upright": ("UPRIGHT (top markers in view)", "plain"),
                    "inverted": ("UPSIDE DOWN / FLIPPED (bottom markers in view)", "warn")}


def age_own_report(own: Mapping[str, Any] | None, elapsed_s: float, predict_hold_s: float, lost_after_s: float) -> dict[str, Any] | None:
    """The own-robot report as it must read ``elapsed_s`` after it was made, if no newer one has arrived.

    The tracker computes a report once per camera frame; a screen that keeps painting that report while the frames have stopped
    (camera unplugged, vision thread stalled) would show an old pose as if it were current. This applies the tracker's OWN limits to
    the real age: past ``predict_hold_s`` the pose is STALE (the last position stays visible, marked display-only), past
    ``lost_after_s`` it is LOST and has no position at all. Within the hold it is unchanged except that the age keeps growing.
    A report that is already stale or lost can only get older. Nothing is invented: no position is ever extrapolated here.
    """
    if own is None:
        return None
    out = dict(own)
    elapsed = max(0.0, float(elapsed_s))
    age = own.get("age_s")
    state = own["state"]
    if age is None or state == "lost":
        return out
    total = float(age) + elapsed
    out["age_s"] = total
    if total > lost_after_s:
        out.update(state="lost", valid_for_observation=False, x_m=None, y_m=None, heading_deg=None, pose_note=None,
                   reason=f"lost: the last consistent detection is {1000 * total:.0f} ms old (limit {1000 * lost_after_s:.0f} ms); "
                          f"no new camera result for {1000 * elapsed:.0f} ms")
    elif total > predict_hold_s and state != "stale":
        out.update(state="stale", valid_for_observation=False, pose_note="last measured pose, DISPLAY ONLY, not valid",
                   reason=f"stale: the last consistent detection is {1000 * total:.0f} ms old (prediction is allowed for "
                          f"{1000 * predict_hold_s:.0f} ms); no new camera result for {1000 * elapsed:.0f} ms; the pose shown is for display only")
    return out


def own_rows(own: Mapping[str, Any] | None) -> list[tuple[str, str, str]]:
    """[(name, value, level)] for the AutoBot panel: state and age first, then pose, orientation and the markers behind it."""
    if own is None:
        return [("State", "NO TRACKER REPORT", "off")]
    state = own["state"]
    age = own.get("age_s")
    age_text = "no detection yet" if age is None else f"{1000 * age:.0f} ms since the last accepted detection"
    meaning = {"measured": "a marker detection in the newest frame", "predicted": "no marker this frame: extrapolated from recent motion",
               "stale": "too old to use: the last measured pose is shown for display only", "lost": "no pose"}.get(state, "")
    rows = [("State", state.upper() + ("  (display only, not valid)" if state == "stale" else ""), OWN_STATE_LEVEL.get(state, "plain")),
            ("Age", age_text + (f"  ({meaning})" if meaning else ""), "ok" if state == "measured" and age is not None and age < 0.1 else
             ("plain" if age is not None and age < 0.2 else "warn"))]
    if own.get("x_m") is not None:
        suffix = "  [last measured, display only]" if state == "stale" else ("  [predicted]" if state == "predicted" else "")
        rows.append(("Position x, y", f"{own['x_m']:.3f}, {own['y_m']:.3f} m{suffix}", "ok" if state == "measured" else "warn"))
        rows.append(("Heading", f"{own['heading_deg']:.1f} deg{suffix}", "ok" if state == "measured" else "warn"))
    else:
        rows.append(("Position x, y", "-", "off"))
        rows.append(("Heading", "-", "off"))
    orientation = str(own.get("orientation", "unknown"))
    text, level = ORIENTATION_TEXT.get(orientation, ("unknown (no detection yet)" if orientation == "unknown" else orientation, "off"))
    rows.append(("Orientation", text + ("  [held from the last detection]" if state in ("predicted", "stale") and orientation in ORIENTATION_TEXT else ""),
                 level if state in ("measured", "predicted") else "off"))
    quality = str(own.get("position_quality", "unknown"))
    rows.append(("Position quality", {"height-corrected": "height-corrected", "approximate": "approximate (not height-corrected)"}
                 .get(quality, quality), "ok" if quality == "height-corrected" else "warn"))
    visible = list(own.get("visible_ids") or [])
    rows.append(("Marker IDs in the newest frame", str(visible) if visible else "none (no configured marker decoded)",
                 "plain" if visible else "warn"))
    rows.append(("Marker IDs the pose rests on", str(own.get("used_ids") or "none"), "plain"))
    if own.get("ignored_ids"):
        rows.append(("Marker IDs ignored", f"{own['ignored_ids']} (decoded, but not in the marker layout)", "warn"))
    if own.get("source") not in (None, "none"):
        rows.append(("Source", str(own["source"]), "warn" if own["source"] == "rear marker only" else "plain"))
    if own.get("rejected"):
        rows.append(("Markers rejected", "; ".join(f"{k}: {v}" for k, v in own["rejected"].items()), "warn"))
    if state != "measured":
        rows.append(("Why", str(own.get("reason", "")), OWN_STATE_LEVEL.get(state, "plain")))
    return rows


def opponent_status_line(opponent: Mapping[str, Any] | None) -> str:
    """One line for the opponent status label."""
    if opponent is None:
        return "TRACKING STATUS: no report yet"
    state = opponent["state"]
    words = {"tracked": "TRACKED", "uncertain": "TRACKING UNCERTAIN", "lost": "LOST: drag a new box", "ambiguous": "AMBIGUOUS",
             "not_selected": "no opponent selected: drag a box around it"}.get(state, state.upper())
    confidence = opponent.get("confidence")
    return f"TRACKING STATUS: {words}" + ("" if confidence is None else f"   confidence {confidence:.2f}")


def opponent_rows(opponent: Mapping[str, Any] | None) -> list[tuple[str, str, str]]:
    if opponent is None:
        return [("State", "NO OPPONENT TRACKER REPORT", "off")]
    state = opponent["state"]
    rows = [("State", state.upper().replace("_", " "), OPPONENT_STATE_LEVEL.get(state, "plain"))]
    if opponent.get("confidence") is not None or opponent.get("image_based"):
        confidence = opponent.get("confidence")
        rows.append(("Tracking confidence", "not available" if confidence is None else f"{confidence:.2f}  (appearance similarity 0 to 1, not a probability)",
                     "off" if confidence is None else ("ok" if confidence >= 0.6 else "warn")))
    if opponent.get("x_m") is not None:
        how = "IMAGE-BASED, approximate, not height-corrected" if opponent.get("image_based") else "approximate, not height-corrected"
        rows.append(("Position x, y", f"{opponent['x_m']:.3f}, {opponent['y_m']:.3f} m  ({how})", "ok"))
    else:
        rows.append(("Position x, y", "none reported" + (" (not shown as current while uncertain or lost)" if opponent.get("image_based") else ""), "off"))
    diagnostics = opponent.get("diagnostics") or {}
    if diagnostics.get("approx_floor_extent_mm"):
        extent = diagnostics["approx_floor_extent_mm"]
        rows.append(("Apparent size (diagnostic only)", f"about {extent[0]} x {extent[1]} mm on the floor plane; never used to accept or refuse a selection", "plain"))
    if opponent.get("backend"):
        rows.append(("Tracker", str(opponent["backend"]), "plain"))
    rows.append(("Why", str(opponent.get("reason", "")), OPPONENT_STATE_LEVEL.get(state, "plain")))
    if opponent.get("selection_note"):
        rows.append(("Last click", str(opponent["selection_note"]), "bad" if "REFUSED" in opponent["selection_note"] else "plain"))
    return rows


def camera_summary(state: str, detail: str, fps: float, frame_age_s: float | None) -> tuple[str, str]:
    """(text, level) for the camera line."""
    age = "no frame yet" if frame_age_s is None else f"newest frame {1000 * frame_age_s:.0f} ms old"
    level = {"LIVE": "ok", "MOCK": "warn", "REPLAY": "warn", "OPENING": "warn"}.get(state, "bad")
    return f"camera {state}  {fps:.1f} fps  {age}" + (f"  ({detail})" if detail and state not in ("LIVE", "MOCK", "REPLAY") else ""), level
