"""Review screen: scrub through one recorded round.

It answers, for the selected moment: what did the camera see, who had control, what was requested,
and what motion was actually observed. Everything shown comes from the round folder through
``RoundData``; nothing here talks to a camera, a port or the robot.

Rules the screen keeps:
  * stale or missing controller telemetry is shown as STALE / UNKNOWN, never as the last value;
  * a position gap stays a gap (on the map and in the traces);
  * a track estimated during the round is solid and labelled live, a track reconstructed after the
    round is dashed and labelled "reconstructed offline";
  * IMU values are raw sensor axes against PC receipt time; nothing is integrated or fused.

Stand-alone:  python -m control_center.screens.review <round folder>
"""

from __future__ import annotations

import html
import logging
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from PySide6.QtCore import QSignalBlocker, Qt, QThread, QTimer, Signal
from PySide6.QtGui import QBrush, QColor, QFont, QKeySequence, QShortcut
from PySide6.QtWidgets import (QAbstractItemView, QApplication, QComboBox, QFileDialog, QGridLayout, QHBoxLayout, QLabel,
                               QProgressBar, QPushButton, QScrollArea, QSlider, QSplitter, QStyle, QStyleOptionSlider,
                               QTabWidget, QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget)

from control_center.backend.round_reader import PoseTrack, RoundData, reconstruct_pose_offline
from control_center.widgets.arena_map import ArenaMapWidget, Marker, Trail
from control_center.widgets.common import Panel
from control_center.widgets.frame_view import FrameView
from control_center.widgets.trace_plot import Series, TracePlot

log = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[2]

LIVE_COLOUR = "#55c889"
OFFLINE_COLOUR = "#6fa8dc"
OPPONENT_COLOUR = "#ef6b73"
MUTED = "#9eabb8"
WARN = "#e8b85b"
BAD = "#ef6b73"
OK = "#55c889"
OFF = "#7f8996"

TRACK_TEXT = {"live": "live (estimated during the round)",
              "offline": "reconstructed offline (after the round, from the video)"}
ARENA_KEYS = ("width_m", "height_m", "pit_radius_m", "pit_centres_m", "flipper_vertices_m")
IMU_MAX_AGE_S = 0.5
TRACE_GAP_S = 0.5
SPEEDS = (("0.25x", 0.25), ("0.5x", 0.5), ("1x", 1.0), ("2x", 2.0))

EVENT_STYLE = {                         # kind -> (colour, bold)
    "round_start": (WARN, True),
    "mode_requested": (OFFLINE_COLOUR, True),
    "source_confirmed": (OFFLINE_COLOUR, True),
    "tracking_lost": (BAD, True),
    "fault": (BAD, False),
    "tracking_gap": (OFF, False),
}

REVIEW_STYLESHEET = """
QLabel { background: transparent; }
QLabel#reviewCaption { color: #9eabb8; font-size: 8pt; font-weight: 700; letter-spacing: 1px; }
QLabel#reviewValue { font-size: 12pt; font-weight: 600; }
QLabel#reviewPosition { font-size: 10pt; font-weight: 700; }
QLabel#reviewSource { font-size: 26pt; font-weight: 800; }
QLabel#reviewNote { color: #9eabb8; font-size: 9pt; }
QLabel#reviewWarning { color: #e8b85b; }
QLabel#reviewTime { font-family: Consolas; font-size: 13pt; font-weight: 700; }
QLabel#reviewMock { color: #11161d; background: #e8b85b; font-weight: 800; padding: 4px 10px; }
QTabWidget::pane { border: 1px solid #2b3542; }
QTabBar::tab { background: #171e27; color: #9eabb8; padding: 6px 14px; border: 1px solid #2b3542; margin-right: 2px; }
QTabBar::tab:selected { background: #263241; color: #d9dee5; }
QTreeWidget { background: #0d1218; border: none; alternate-background-color: #121922; }
QTreeWidget::item:selected { background: #2c3b4e; }
QHeaderView::section { background: #171e27; color: #9eabb8; border: none; padding: 4px 6px; font-weight: 700; }
QScrollArea#reviewScroll { background: transparent; border: none; }
QWidget#reviewScrollBody { background: transparent; }
QProgressBar { background: #0d1218; border: 1px solid #2b3542; border-radius: 3px; text-align: center; }
QProgressBar::chunk { background: #3c6f9c; }
QComboBox { background: #263241; border: 1px solid #3b4a5d; border-radius: 4px; padding: 5px 8px; }
"""


# ---------------------------------------------------------------- pure helpers (no Qt)


def format_time(seconds: float) -> str:
    """mm:ss.s, truncated (never rounds up past the end of the round)."""
    tenths = int(max(0.0, seconds) * 10.0 + 1e-6)
    return f"{tenths // 600:02d}:{(tenths // 10) % 60:02d}.{tenths % 10}"


def latest_round_folder(root: Path | str) -> Path | None:
    """The newest round folder under ``root`` (folder names start with the recording timestamp)."""
    root = Path(root)
    if not root.is_dir():
        return None
    rounds = sorted((p for p in root.iterdir() if p.is_dir() and (p / "manifest.json").is_file()), key=lambda p: p.name)
    return rounds[-1] if rounds else None


def round_duration(data: RoundData) -> float:
    """End of the timeline: the latest time in any stream (an unclosed round has no manifest duration)."""
    ends = [float(data.duration_s)]
    for times in (data.frame_t, data.imu_t, data.pose_live.t, data.pose_offline.t):
        finite = np.asarray(times, float)
        finite = finite[np.isfinite(finite)]
        if len(finite):
            ends.append(float(finite.max()))
    for records in (data.events, data.controller_status, data.pc_sent, data.policy):
        ends += [float(r["t_s"]) for r in records if isinstance(r.get("t_s"), (int, float))]
    return max(0.0, max(ends))


def arena_for(data: RoundData | None) -> tuple[dict[str, Any] | None, str]:
    """(arena geometry, note). The round's own arena if it stored one, else the overlay config's."""
    arena = (data.manifest.get("arena") if data is not None else None) or {}
    if isinstance(arena, dict) and all(key in arena for key in ARENA_KEYS):
        return arena, ""
    try:
        from control_center.tools.overlay_core import load_config

        note = "" if data is None else "arena outline from config/arena_overlay.yaml (this round did not store one)"
        return dict(load_config().arena), note
    except Exception as error:  # missing or unreadable config
        return None, f"arena geometry unavailable: {error}"


def control_readout(data: RoundData, t: float) -> dict[str, str]:
    """Controller-reported state at ``t`` as display text. Not fresh = the word STALE or UNKNOWN everywhere."""
    if not data.controller_status:
        word = "UNKNOWN"
        return {"freshness": word, "source": word, "requested": word, "armed": word, "flags": word, "motors": word,
                "telemetry": "No control telemetry in this round"}
    state = data.control_at(t)
    word = state["freshness"]
    if word != "FRESH":
        if word == "STALE":
            telemetry = (f"STALE: the last controller report is {state['age_s']:.1f} s before this time; "
                         "it is not shown as current")
        else:
            word = "UNKNOWN"
            telemetry = "UNKNOWN: no controller report at or before this time"
        return {"freshness": word, "source": word, "requested": word, "armed": word, "flags": word, "motors": word,
                "telemetry": telemetry}
    record = state["record"] or {}
    armed = state["armed"]
    telemetry = f"fresh: controller report {state['age_s']:.2f} s before this time"
    if record.get("link") is False:
        telemetry += " · controller reports the RADIO LINK DOWN"
    return {
        "freshness": "FRESH",
        "source": str(state["source"]) if state["source"] else "UNKNOWN",
        "requested": str(state["requested"]) if state["requested"] else "UNKNOWN",
        "armed": "UNKNOWN" if armed is None else ("ARMED" if armed else "DISARMED"),
        "flags": f"block {record.get('block', '?')} · fault {record.get('fault', '?')} · stop {record.get('stop', '?')}",
        "motors": f"L {record.get('left', '?')}   R {record.get('right', '?')}",
        "telemetry": telemetry,
    }


def pc_sent_readout(data: RoundData, t: float) -> str:
    if not data.pc_sent:
        return "No PC-sent commands in this round"
    record = data.pc_sent_at(t)
    if record is None:
        return "none in the 1.0 s before this time"
    return (f"throttle {record.get('throttle', '?')}   steer {record.get('steer', '?')}   "
            f"(token {record.get('token', '?')}, sent {max(0.0, t - float(record['t_s'])):.2f} s earlier)")


def _short(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.3f}"
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_short(v) for v in value) + "]"
    return str(value)


def policy_readout(data: RoundData, t: float) -> str:
    if not data.policy:
        return "No policy stream in this round"
    record = data.policy_at(t)
    if record is None:
        return "no policy record before this time"
    parts = [f"mode {record.get('mode', '?')}"]
    if record.get("checkpoint"):
        parts.append(f"checkpoint {record['checkpoint']}")
    if record.get("input_source"):
        parts.append(f"input {record['input_source']}")
    proposed = record.get("proposed")
    if record.get("age_s", 0.0) > 1.0:
        # The mode holds until the next record; a proposed action that old is not "at this moment".
        action = f"none in the 1.0 s before this time (last record {record['age_s']:.1f} s earlier)"
    elif isinstance(proposed, dict):
        action = ", ".join(f"{key} {_short(value)}" for key, value in proposed.items()) or "empty"
    elif proposed is None:
        action = "no action in this record"
    else:
        action = _short(proposed)
    return " · ".join(parts) + f"\nproposed: {action}"


def imu_readout(data: RoundData, t: float) -> str:
    if len(data.imu_t) == 0:
        return "No IMU stream in this round"
    i = int(np.searchsorted(data.imu_t, t, side="right")) - 1
    if i < 0 or t - data.imu_t[i] > IMU_MAX_AGE_S:
        return f"no IMU sample in the {IMU_MAX_AGE_S:.1f} s before this time"
    accel = "  ".join(f"{data.imu[name][i]:+.2f}" for name in ("ax_m_s2", "ay_m_s2", "az_m_s2"))
    gyro = "  ".join(f"{data.imu[name][i]:+.1f}" for name in ("gx_deg_s", "gy_deg_s", "gz_deg_s"))
    return f"accel  {accel}  m/s²\ngyro   {gyro}  deg/s"


def pose_readout(data: RoundData, t: float) -> dict[str, Any]:
    """Observed own-bot pose at ``t``: per-track samples, the one to show, and the text for it."""
    samples: dict[str, dict[str, Any] | None] = {}
    for track in (data.pose_live, data.pose_offline):
        if len(track):
            samples[track.source] = track.at(t)
    valid = {source: sample for source, sample in samples.items()
             if sample is not None and sample["valid"] and math.isfinite(sample["x_m"]) and math.isfinite(sample["y_m"])}
    if valid:
        source = "live" if "live" in valid else "offline"
        pose = valid[source]
        heading = f"{pose['heading_deg']:.1f}°" if math.isfinite(pose["heading_deg"]) else "unknown"
        return {"known": True, "source": source, "pose": pose, "valid": valid,
                "text": f"x {pose['x_m']:.3f} m   y {pose['y_m']:.3f} m   heading {heading}",
                "source_text": TRACK_TEXT[source]}
    if not samples:
        reason = "no own-bot track in this round"
        if not data.calibrated:
            reason += " (no calibration in this round)"
        source_text = "no track"
    else:
        source = "live" if "live" in samples else "offline"
        sample = samples[source]
        if sample is None:
            reason = "no pose sample within 0.25 s of this time"
        else:
            reason = sample.get("reason") or "no valid detection"
        source_text = f"{TRACK_TEXT[source]}: gap"
    return {"known": False, "source": None, "pose": None, "valid": {}, "reason": reason,
            "text": f"position unknown: {reason}", "source_text": source_text}


def stream_summary(data: RoundData) -> list[tuple[str, bool, str]]:
    """(name, present, detail) for every stream, from what actually loaded."""
    written = int(data.frame_written.sum())
    def rows(n: int) -> str:
        return f"{n} row{'s' if n != 1 else ''}"
    return [
        ("video", data.video_path is not None and written > 0, f"{written} frames"),
        ("frame times", len(data.frame_t) > 0, rows(len(data.frame_t))),
        ("own-bot pose, live", len(data.pose_live) > 0, f"{rows(len(data.pose_live))}, {int(data.pose_live.valid.sum())} valid"),
        ("own-bot pose, reconstructed offline", len(data.pose_offline) > 0,
         f"{rows(len(data.pose_offline))}, {int(data.pose_offline.valid.sum())} valid"),
        ("opponent track", len(data.opponent) > 0, rows(len(data.opponent))),
        ("IMU", len(data.imu_t) > 0, rows(len(data.imu_t))),
        ("controller status", len(data.controller_status) > 0, rows(len(data.controller_status))),
        ("PC-sent commands", len(data.pc_sent) > 0, rows(len(data.pc_sent))),
        ("policy", len(data.policy) > 0, rows(len(data.policy))),
        ("events", len(data.events) > 0, rows(len(data.events))),
    ]


def event_rows(data: RoundData) -> list[dict[str, Any]]:
    """Recorded events plus tracking gaps derived from the pose tracks, in time order."""
    rows: list[dict[str, Any]] = []
    for record in data.events:
        try:
            t = float(record["t_s"])
        except (TypeError, ValueError):
            continue
        rows.append({"t": t, "kind": str(record.get("kind", "?")), "text": str(record.get("text", "")), "derived": False})
    for track in (data.pose_live, data.pose_offline):
        for start, end in track.gaps():
            rows.append({"t": start, "kind": "tracking_gap", "derived": True,
                         "text": f"no valid own-bot pose for {end - start:.1f} s, until {format_time(end)} "
                                 f"({'live track' if track.source == 'live' else 'offline reconstruction'}; derived from the pose rows)"})
    rows.sort(key=lambda row: row["t"])
    return rows


def opponent_runs(rows: list[dict[str, str]], t_end: float) -> dict[str, list[np.ndarray]]:
    """Per track id: runs of consecutive positions up to ``t_end``; a row without a position ends a run."""
    out: dict[str, list[np.ndarray]] = {}
    current: dict[str, list[tuple[float, float]]] = {}
    for row in rows:
        try:
            t = float(row.get("t_s", "nan"))
        except (TypeError, ValueError):
            continue
        if not t <= t_end:
            continue
        track = str(row.get("track_id", ""))
        try:
            x, y = float(row.get("x_m", "")), float(row.get("y_m", ""))
        except (TypeError, ValueError):
            x = y = float("nan")
        if math.isfinite(x) and math.isfinite(y):
            current.setdefault(track, []).append((x, y))
        elif current.get(track):
            out.setdefault(track, []).append(np.array(current.pop(track)))
    for track, points in current.items():
        if points:
            out.setdefault(track, []).append(np.array(points))
    return out


# ---------------------------------------------------------------- offline reconstruction thread


class ReconstructWorker(QThread):
    progress = Signal(int, int)
    done = Signal(object)

    def __init__(self, folder: Path, parent=None) -> None:
        super().__init__(parent)
        self.folder = folder

    def run(self) -> None:
        try:
            result = reconstruct_pose_offline(self.folder, progress=lambda n, total: self.progress.emit(int(n), int(total)))
        except Exception as error:  # never let a bad video take the dashboard down
            log.exception("offline reconstruction failed")
            result = {"ok": False, "reason": f"{type(error).__name__}: {error}"}
        self.done.emit(result)


class SeekSlider(QSlider):
    """A slider that jumps to where it is clicked (the default is to move one page towards the click)."""

    def mousePressEvent(self, event) -> None:  # noqa: N802 - Qt API
        if event.button() == Qt.MouseButton.LeftButton and self.maximum() > self.minimum():
            option = QStyleOptionSlider()
            self.initStyleOption(option)
            style = self.style()
            groove = style.subControlRect(QStyle.ComplexControl.CC_Slider, option, QStyle.SubControl.SC_SliderGroove, self)
            handle = style.subControlRect(QStyle.ComplexControl.CC_Slider, option, QStyle.SubControl.SC_SliderHandle, self)
            if not handle.contains(event.position().toPoint()):
                offset = int(event.position().x() - groove.x() - handle.width() / 2.0)
                self.setValue(QStyle.sliderValueFromPosition(self.minimum(), self.maximum(), offset,
                                                             max(1, groove.width() - handle.width())))
        super().mousePressEvent(event)


# ---------------------------------------------------------------- the screen


class ReviewScreen(QWidget):
    """One scrubber drives the video frame, the map, the read-outs and the traces."""

    time_changed = Signal(float)

    def __init__(self, hub=None, parent=None) -> None:
        super().__init__(parent)
        self.hub = hub
        options = getattr(hub, "options", None)
        self.rounds_root = Path(getattr(options, "rounds_root", None) or ROOT / "rounds")
        self.data: RoundData | None = None
        self.duration = 0.0
        self.current_frame_row: int | None = None     # frames.csv row whose image is on screen (None = placeholder)
        self._t = 0.0
        self._events: list[dict[str, Any]] = []
        self._event_items: list[QTreeWidgetItem] = []
        self._current_event = -1
        self._shown_frame: tuple[Any, ...] | None = None
        self._arena_note = ""
        self._worker: ReconstructWorker | None = None
        self._play_anchor = (0.0, 0.0)                # (wall clock, round time) when playback (re)started
        self._play_timer = QTimer(self)
        self._play_timer.setInterval(30)
        self._play_timer.timeout.connect(self._play_tick)

        self.setStyleSheet(REVIEW_STYLESHEET)
        self._build()
        self._show_no_round("No round open. Open a round folder to review it.")
        signal = getattr(hub, "recording_finished", None)
        if signal is not None:
            try:
                signal.connect(self._on_round_saved)
            except (AttributeError, TypeError, RuntimeError):
                pass

    # ---- construction ----

    def _build(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)
        root.addWidget(self._build_header())

        left = QSplitter(Qt.Orientation.Vertical)
        top = QWidget()
        top_layout = QVBoxLayout(top)
        top_layout.setContentsMargins(0, 0, 0, 0)
        top_layout.setSpacing(8)
        views = QSplitter(Qt.Orientation.Horizontal)
        views.addWidget(self._build_video())
        views.addWidget(self._build_map())
        views.setStretchFactor(0, 3)
        views.setStretchFactor(1, 2)
        views.setChildrenCollapsible(False)
        top_layout.addWidget(views, 1)
        top_layout.addWidget(self._build_transport())
        left.addWidget(top)
        left.addWidget(self._build_tabs())
        left.setStretchFactor(0, 3)
        left.setStretchFactor(1, 1)
        left.setChildrenCollapsible(False)

        body = QSplitter(Qt.Orientation.Horizontal)
        body.addWidget(left)
        body.addWidget(self._build_moment())
        body.setStretchFactor(0, 4)
        body.setStretchFactor(1, 1)
        body.setChildrenCollapsible(False)
        root.addWidget(body, 1)

        for key, action in ((Qt.Key.Key_Space, self.toggle_play), (Qt.Key.Key_Left, lambda: self.step_frame(-1)),
                            (Qt.Key.Key_Right, lambda: self.step_frame(1))):
            shortcut = QShortcut(QKeySequence(key), self)
            shortcut.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
            shortcut.activated.connect(action)

    def _build_header(self) -> QWidget:
        panel = Panel("ROUND REVIEW · recorded data, not live")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(4)
        row = QHBoxLayout()
        row.setSpacing(8)
        row.addWidget(panel.title)
        self.mock_badge = QLabel("MOCK round — not hardware")
        self.mock_badge.setObjectName("reviewMock")
        self.mock_badge.hide()
        row.addWidget(self.mock_badge)
        row.addStretch(1)
        self.open_button = QPushButton("Open round…")
        self.open_button.clicked.connect(self.choose_round)
        self.latest_button = QPushButton("Open latest")
        self.latest_button.setToolTip(f"Newest round folder under {self.rounds_root}")
        self.latest_button.clicked.connect(self.open_latest)
        self.reconstruct_button = QPushButton("Reconstruct own-bot track from video (offline)")
        self.reconstruct_button.setToolTip("Re-runs marker detection on the recorded video and saves the result as a separate, "
                                           "dashed track in derived/pose_offline.csv. The live track is not changed.")
        self.reconstruct_button.clicked.connect(self.reconstruct_offline)
        self.progress = QProgressBar()
        self.progress.setFixedWidth(160)
        self.progress.hide()
        for widget in (self.open_button, self.latest_button, self.reconstruct_button, self.progress):
            row.addWidget(widget)
        layout.addLayout(row)

        self.summary_label = QLabel()
        self.streams_label = QLabel()
        self.warnings_label = QLabel()
        self.warnings_label.setObjectName("reviewWarning")
        self.status_label = QLabel()
        self.status_label.setObjectName("reviewNote")
        for label in (self.summary_label, self.streams_label, self.warnings_label, self.status_label):
            label.setWordWrap(True)
            label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            layout.addWidget(label)
        self.streams_label.setTextFormat(Qt.TextFormat.RichText)
        return panel

    def _build_video(self) -> QWidget:
        panel = Panel("CAMERA · raw recorded frame, no overlays")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(4)
        layout.addWidget(panel.title)
        self.frame_view = FrameView()
        self.frame_view.set_placeholder("No round open")
        layout.addWidget(self.frame_view, 1)
        self.video_caption = QLabel()
        self.video_caption.setObjectName("reviewNote")
        self.video_caption.setWordWrap(True)
        layout.addWidget(self.video_caption)
        return panel

    def _build_map(self) -> QWidget:
        panel = Panel("MAP · own-bot position from the camera")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(4)
        layout.addWidget(panel.title)
        self._map_layout = QVBoxLayout()
        self._map_layout.setContentsMargins(0, 0, 0, 0)
        layout.addLayout(self._map_layout, 1)
        self.map: ArenaMapWidget | None = None
        self._map_placeholder: QLabel | None = None
        self.map_position = QLabel()
        self.map_position.setObjectName("reviewPosition")
        self.map_position.setWordWrap(True)
        self.map_position.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        self.map_position.setMinimumHeight(38)        # room for a reason that wraps onto a second line
        self.map_legend = QLabel()
        self.map_legend.setObjectName("reviewNote")
        self.map_legend.setTextFormat(Qt.TextFormat.RichText)
        self.map_legend.setWordWrap(True)
        self.opponent_label = QLabel()
        self.opponent_label.setObjectName("reviewNote")
        self.opponent_label.setWordWrap(True)
        for label in (self.map_position, self.map_legend, self.opponent_label):
            layout.addWidget(label)
        self._set_arena(None)
        return panel

    def _build_transport(self) -> QWidget:
        panel = Panel("")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(4)
        row = QHBoxLayout()
        row.setSpacing(6)
        self.prev_event_button = QPushButton("|◀ Event")
        self.prev_event_button.setToolTip("Jump to the previous event")
        self.prev_event_button.clicked.connect(lambda: self.jump_event(-1))
        self.back_button = QPushButton("◀ Frame")
        self.back_button.setToolTip("One camera frame back (Left)")
        self.back_button.clicked.connect(lambda: self.step_frame(-1))
        self.play_button = QPushButton("▶ Play")
        self.play_button.setToolTip("Play in real time from the recorded frame times (Space)")
        self.play_button.setMinimumWidth(96)
        self.play_button.clicked.connect(self.toggle_play)
        self.forward_button = QPushButton("Frame ▶")
        self.forward_button.setToolTip("One camera frame forward (Right)")
        self.forward_button.clicked.connect(lambda: self.step_frame(1))
        self.next_event_button = QPushButton("Event ▶|")
        self.next_event_button.setToolTip("Jump to the next event")
        self.next_event_button.clicked.connect(lambda: self.jump_event(1))
        self.speed_box = QComboBox()
        for text, factor in SPEEDS:
            self.speed_box.addItem(text, factor)
        self.speed_box.setCurrentIndex(2)
        self.speed_box.setToolTip("Playback speed")
        self.speed_box.currentIndexChanged.connect(lambda _: self._anchor_playback())
        self.time_label = QLabel("00:00.0 / 00:00.0  frame –")
        self.time_label.setObjectName("reviewTime")
        for widget in (self.prev_event_button, self.back_button, self.play_button, self.forward_button,
                       self.next_event_button, self.speed_box):
            row.addWidget(widget)
        row.addStretch(1)
        row.addWidget(self.time_label)
        layout.addLayout(row)
        self.slider = SeekSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, 0)
        self.slider.setSingleStep(100)
        self.slider.setPageStep(1000)
        self.slider.setTracking(True)
        self.slider.valueChanged.connect(self._on_slider)
        layout.addWidget(self.slider)
        self._transport = [self.prev_event_button, self.back_button, self.play_button, self.forward_button,
                           self.next_event_button, self.speed_box, self.slider]
        return panel

    def _build_tabs(self) -> QWidget:
        self.tabs = QTabWidget()
        events = QWidget()
        events_layout = QVBoxLayout(events)
        events_layout.setContentsMargins(6, 6, 6, 6)
        events_layout.setSpacing(4)
        key = QLabel(f'Click a row to jump there. &nbsp; <span style="color:{WARN}"><b>round start</b></span> &nbsp; '
                     f'<span style="color:{OFFLINE_COLOUR}"><b>handoff: mode requested / source confirmed</b></span> &nbsp; '
                     f'<span style="color:{BAD}"><b>tracking lost</b></span> &nbsp; '
                     f'<span style="color:{OFF}"><i>tracking gap (derived from the pose rows)</i></span>')
        key.setObjectName("reviewNote")
        key.setTextFormat(Qt.TextFormat.RichText)
        key.setWordWrap(True)
        events_layout.addWidget(key)
        self.events_view = QTreeWidget()
        self.events_view.setColumnCount(3)
        self.events_view.setHeaderLabels(["Time", "Kind", "What happened"])
        self.events_view.setRootIsDecorated(False)
        self.events_view.setAlternatingRowColors(True)
        self.events_view.setUniformRowHeights(True)
        self.events_view.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.events_view.setColumnWidth(0, 90)
        self.events_view.setColumnWidth(1, 170)
        self.events_view.itemClicked.connect(self._on_event_clicked)
        self.events_view.itemActivated.connect(self._on_event_clicked)
        events_layout.addWidget(self.events_view, 1)
        self.events_note = QLabel()
        self.events_note.setObjectName("reviewNote")
        events_layout.addWidget(self.events_note)
        self.tabs.addTab(events, "Events")

        traces = QWidget()
        traces_layout = QVBoxLayout(traces)
        traces_layout.setContentsMargins(6, 6, 6, 6)
        traces_layout.setSpacing(4)
        self.traces_note = QLabel(
            "IMU: raw sensor axes exactly as the controller sent them (not rotated into a robot frame), plotted against PC "
            "receipt time; the controller clock is not synchronised with the camera. Camera position is a separate "
            "measurement: nothing here is integrated, and IMU and camera are not fused. Click a plot to move the scrubber.")
        self.traces_note.setObjectName("reviewNote")
        self.traces_note.setWordWrap(True)
        traces_layout.addWidget(self.traces_note)
        grid = QGridLayout()
        grid.setSpacing(6)
        self.accel_plot = TracePlot("Accelerometer · sensor axes", "m/s²", "No IMU stream in this round")
        self.gyro_plot = TracePlot("Gyro · sensor axes", "deg/s", "No IMU stream in this round")
        self.position_plot = TracePlot("Camera position", "m", "No camera position in this round")
        self.heading_plot = TracePlot("Camera heading", "deg", "No camera heading in this round")
        self.plots = [self.accel_plot, self.gyro_plot, self.position_plot, self.heading_plot]
        for index, plot in enumerate(self.plots):
            plot.time_clicked.connect(self.seek)
            grid.addWidget(plot, index // 2, index % 2)
        traces_layout.addLayout(grid, 1)
        self.tabs.addTab(traces, "Traces · IMU and camera position")
        return self.tabs

    def _build_moment(self) -> QWidget:
        panel = Panel("AT THIS MOMENT")
        outer = QVBoxLayout(panel)
        outer.setContentsMargins(10, 8, 10, 8)
        outer.setSpacing(6)
        outer.addWidget(panel.title)
        scroll = QScrollArea()
        scroll.setObjectName("reviewScroll")
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        body = QWidget()
        body.setObjectName("reviewScrollBody")
        layout = QVBoxLayout(body)
        layout.setContentsMargins(0, 0, 4, 0)
        layout.setSpacing(2)
        self.moment_captions: dict[str, QLabel] = {}

        def add(key: str, caption: str, object_name: str = "reviewValue") -> QLabel:
            title = QLabel(caption)
            title.setObjectName("reviewCaption")
            title.setWordWrap(True)
            value = QLabel("–")
            value.setObjectName(object_name)
            value.setWordWrap(True)
            value.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            layout.addWidget(title)
            layout.addWidget(value)
            layout.addSpacing(6)
            self.moment_captions[key] = title
            return value

        self.source_value = add("source", "CONTROL SOURCE (confirmed by the controller)", "reviewSource")
        self.requested_value = add("requested", "REQUESTED MODE (RadioMaster switch)")
        self.armed_value = add("armed", "ARMED / DISARMED (controller-reported)")
        self.telemetry_value = add("telemetry", "CONTROLLER TELEMETRY", "reviewNote")
        self.flags_value = add("flags", "BLOCK · FAULT · STOP (controller-reported)")
        self.motors_value = add("motors", "MOTOR OUTPUTS L / R (controller-reported)")
        self.pc_sent_value = add("pc_sent", "PC sent (not confirmed applied)")
        self.policy_value = add("policy", "policy proposed (display only)")
        self.pose_value = add("pose", "OBSERVED POSE (camera)")
        self.pose_source_value = add("pose_source", "POSE SOURCE", "reviewNote")
        self.imu_value = add("imu", "IMU raw (sensor axes, PC receipt time)")
        mono = QFont("Consolas")
        mono.setPointSize(10)
        self.imu_value.setFont(mono)
        layout.addStretch(1)
        scroll.setWidget(body)
        outer.addWidget(scroll, 1)
        panel.setMinimumWidth(300)
        return panel

    # ---- opening a round ----

    def choose_round(self) -> None:
        start = self.rounds_root if self.rounds_root.is_dir() else ROOT
        folder = QFileDialog.getExistingDirectory(self, "Open a recorded round folder", str(start))
        if folder:
            self.open_round(folder)

    def open_latest(self) -> tuple[bool, str]:
        folder = latest_round_folder(self.rounds_root)
        if folder is None:
            message = f"No recorded rounds under {self.rounds_root}"
            self.status_label.setText(message)
            return False, message
        return self.open_round(folder)

    def open_round(self, folder) -> tuple[bool, str]:
        """Load a round folder and show its start. On failure the round already open (if any) stays."""
        return self._load(Path(folder), keep_time=False)

    def _load(self, folder: Path, keep_time: bool) -> tuple[bool, str]:
        if self._worker is not None:
            message = "Offline reconstruction is running; wait for it to finish before opening a round"
            self.status_label.setText(message)
            return False, message
        self.pause()
        try:
            data = RoundData(folder)
        except Exception as error:  # not a round folder, unreadable manifest, damaged frames.csv ...
            message = f"Could not open {folder}: {error}"
            log.warning(message)
            self.status_label.setText(message)
            return False, message
        previous_t = self._t if keep_time else 0.0
        if self.data is not None:
            self.data.close()
        self.data = data
        self.duration = round_duration(data)
        self._shown_frame = None
        self._events = event_rows(data)
        with QSignalBlocker(self.slider):
            self.slider.setRange(0, int(math.ceil(self.duration * 1000.0)))
        arena, self._arena_note = arena_for(data)
        self._set_arena(arena)
        self._fill_header()
        self._fill_events()
        self._fill_legend()
        self._fill_traces()
        self.frame_view.badge = "MOCK RECORDING · not hardware" if data.manifest.get("mock") else "RECORDED · not live"
        for widget in self._transport:
            widget.setEnabled(True)
        self.reconstruct_button.setEnabled(True)
        self.seek(previous_t)
        message = f"Opened {data.folder}"
        self.status_label.setText(message)
        return True, message

    def _show_no_round(self, text: str) -> None:
        self.summary_label.setText(text)
        self.streams_label.setText("")
        self.warnings_label.setText("")
        self.warnings_label.hide()
        self.status_label.setText(f"Rounds folder: {self.rounds_root}")
        self.mock_badge.hide()
        self.video_caption.setText("")
        self.map_position.setText("")
        self.map_legend.setText("")
        self.opponent_label.setText("")
        self.events_note.setText("No round open")
        for widget in self._transport:
            widget.setEnabled(False)
        self.reconstruct_button.setEnabled(False)
        for plot in self.plots:
            plot.set_series([])
            plot.set_empty_message("No round open")
        if self.map is not None:
            self.map.set_content(trails=[], markers=[], caption="", note="No round open")

    def _set_arena(self, arena: dict[str, Any] | None) -> None:
        if arena is None:
            arena, note = arena_for(None)
            self._arena_note = note
        for widget in (self.map, self._map_placeholder):
            if widget is not None:
                self._map_layout.removeWidget(widget)
                widget.hide()
                widget.setParent(None)
                widget.deleteLater()
        self.map, self._map_placeholder = None, None
        try:
            if arena is None:
                raise ValueError(self._arena_note or "no arena geometry")
            self.map = ArenaMapWidget(arena)
            self.map.setMinimumSize(180, 180)
            self._map_layout.addWidget(self.map, 1)
        except Exception as error:  # geometry present but unusable
            self._map_placeholder = QLabel(f"Map unavailable: {error}")
            self._map_placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self._map_placeholder.setWordWrap(True)
            self._map_layout.addWidget(self._map_placeholder, 1)

    def _fill_header(self) -> None:
        data = self.data
        manifest = data.manifest
        video = manifest.get("video") or {}
        rows = len(data.frame_t)
        written = int(data.frame_written.sum())
        if "frames_written" in video:
            frames = (f"frames received {video.get('frames_received', rows)}, written {video.get('frames_written')}, "
                      f"dropped {video.get('frames_dropped', '?')}")
            if rows - written > int(video.get("frames_dropped") or 0):
                frames += f" ({rows - written} of the received frames are not in the video)"
        else:
            frames = (f"frames received {rows}, written {written}, not in the video {rows - written} "
                      "(counted from frames.csv; the manifest has no video summary)")
        errors = manifest.get("write_errors")
        if errors is None:
            error_text = "write errors: unknown (manifest incomplete)"
        else:
            error_text = f"write errors: {len(errors)}"
        label = manifest.get("label") or "(no label)"
        started = manifest.get("started_local") or "start time unknown"
        self.summary_label.setText(f"{data.folder}\nLabel: {label} · started {started} · duration {self.duration:.1f} s "
                                   f"({format_time(self.duration)}) · {frames} · {error_text}")
        self.mock_badge.setVisible(bool(manifest.get("mock")))

        present, absent = [], []
        for name, there, detail in stream_summary(data):
            if there:
                present.append(f"{html.escape(name)} <span style='color:{MUTED}'>({html.escape(detail)})</span>")
            else:
                absent.append(html.escape(name))
        text = f"<span style='color:{OK}'><b>Present:</b></span> " + (" · ".join(present) if present else "nothing")
        text += f"<br><span style='color:{WARN}'><b>Absent:</b></span> " + (" · ".join(absent) if absent else "none")
        self.streams_label.setText(text)

        warnings = list(data.warnings)
        if manifest.get("mock"):
            warnings.append(str(manifest.get("mock_note") or "MOCK round: synthetic camera and replayed controller lines, not hardware"))
        if errors:
            warnings.append("write errors during recording: " + "; ".join(str(e) for e in errors[:3])
                            + (f" (+{len(errors) - 3} more)" if len(errors) > 3 else ""))
        if video and video.get("verified_after_stop") is False:
            warnings.append(f"the video did not verify after recording: {video.get('verify_detail', 'no detail')}")
        if not data.calibrated:
            reason = manifest.get("calibration_note")
            warnings.append("no calibration in this round: no camera positions" + (f" ({reason})" if reason else ""))
        if self._arena_note:
            warnings.append(self._arena_note)
        self.warnings_label.setText("\n".join(f"⚠ {w}" for w in warnings))
        self.warnings_label.setVisible(bool(warnings))

    def _fill_events(self) -> None:
        self.events_view.clear()
        self._event_items = []
        self._current_event = -1
        for row in self._events:
            item = QTreeWidgetItem([format_time(row["t"]), row["kind"] + (" (derived)" if row["derived"] else ""), row["text"]])
            item.setData(0, Qt.ItemDataRole.UserRole, float(row["t"]))
            colour, bold = EVENT_STYLE.get(row["kind"], (None, False))
            for column in range(3):
                if colour:
                    item.setForeground(column, QBrush(QColor(colour)))
                font = item.font(column)
                font.setBold(bold)
                font.setItalic(row["derived"])
                item.setFont(column, font)
            item.setToolTip(2, row["text"])
            self.events_view.addTopLevelItem(item)
            self._event_items.append(item)
        recorded = sum(1 for row in self._events if not row["derived"])
        if not self._events:
            self.events_note.setText("No events in this round")
        else:
            self.events_note.setText(f"{recorded} recorded event(s), {len(self._events) - recorded} derived tracking gap(s)")
        self.prev_event_button.setEnabled(bool(self._events))
        self.next_event_button.setEnabled(bool(self._events))

    def _fill_legend(self) -> None:
        data = self.data
        parts = []
        if len(data.pose_live):
            parts.append(f"<span style='color:{LIVE_COLOUR}'><b>——— solid ●</b></span> {TRACK_TEXT['live']}")
        else:
            parts.append("No live own-bot track in this round")
        if len(data.pose_offline):
            made = str((data.pose_offline_meta or {}).get("created_utc") or "")
            parts.append(f"<span style='color:{OFFLINE_COLOUR}'><b>- - - dashed ○</b></span> reconstructed offline, after the round"
                         + (f" (made {html.escape(made[:16].replace('T', ' '))} UTC)" if made else ""))
        else:
            parts.append("No offline reconstruction for this round")
        self.map_legend.setText("<br>".join(parts))
        if data.opponent:
            self.opponent_label.setText(f"Opponent track: {len(data.opponent)} recorded rows (red)")
        else:
            self.opponent_label.setText("No opponent track in this round")

    def _trace_track(self) -> PoseTrack | None:
        """The track to plot: live if it has any valid pose, else the offline one if that has."""
        for track in (self.data.pose_live, self.data.pose_offline):
            if len(track) and bool(track.valid.any()):
                return track
        return None

    def _fill_traces(self) -> None:
        data = self.data
        span = (0.0, max(self.duration, 1e-3))
        if len(data.imu_t):
            self.accel_plot.set_series([Series(axis, data.imu_t, data.imu[name], max_gap_s=TRACE_GAP_S)
                                        for axis, name in (("X", "ax_m_s2"), ("Y", "ay_m_s2"), ("Z", "az_m_s2"))], span)
            self.gyro_plot.set_series([Series(axis, data.imu_t, data.imu[name], max_gap_s=TRACE_GAP_S)
                                       for axis, name in (("X", "gx_deg_s"), ("Y", "gy_deg_s"), ("Z", "gz_deg_s"))], span)
        else:
            for plot in (self.accel_plot, self.gyro_plot):
                plot.set_series([], span)
                plot.set_empty_message("No IMU stream in this round")
        track = self._trace_track()
        if track is None:
            reason = "No camera position in this round"
            if not data.calibrated:
                reason += " (no calibration)"
            for plot in (self.position_plot, self.heading_plot):
                plot.set_series([], span)
                plot.set_empty_message(reason)
            self.position_plot.title, self.heading_plot.title = "Camera position", "Camera heading"
            return
        offline = track.source == "offline"
        suffix = "reconstructed offline" if offline else "live"
        self.position_plot.title = f"Camera position · {suffix}"
        self.heading_plot.title = f"Camera heading · {suffix}"
        x = np.where(track.valid, track.x, np.nan)
        y = np.where(track.valid, track.y, np.nan)
        heading = np.where(track.valid, track.heading_deg, np.nan)
        # A step across +/-180 degrees is the angle wrapping, not a turn: do not draw it as a line.
        with np.errstate(invalid="ignore"):
            wrap = np.abs(np.diff(heading, prepend=heading[:1])) > 180.0
        heading_t = np.insert(track.t, np.flatnonzero(wrap), track.t[wrap])
        heading_y = np.insert(heading, np.flatnonzero(wrap), np.nan)
        self.position_plot.set_series([Series("x", track.t, x, dashed=offline, max_gap_s=TRACE_GAP_S),
                                       Series("y", track.t, y, dashed=offline, max_gap_s=TRACE_GAP_S)], span)
        self.heading_plot.set_series([Series("heading", heading_t, heading_y, dashed=offline, max_gap_s=TRACE_GAP_S)], span)

    # ---- the timeline ----

    @property
    def current_time(self) -> float:
        return self._t

    @property
    def playing(self) -> bool:
        return self._play_timer.isActive()

    @property
    def reconstructing(self) -> bool:
        return self._worker is not None

    def seek(self, t_s: float) -> None:
        """Move the scrubber (and everything it drives) to ``t_s`` seconds from the start of the recording."""
        if self.data is None:
            return
        self._t = min(max(0.0, float(t_s)), self.duration)
        with QSignalBlocker(self.slider):
            self.slider.setValue(int(round(self._t * 1000.0)))
        if self.playing:
            self._anchor_playback()
        self._refresh()

    def _on_slider(self, value: int) -> None:
        if self.data is None:
            return
        self._t = min(value / 1000.0, self.duration)
        if self.playing:
            self._anchor_playback()
        self._refresh()

    def _frame_row(self, t: float) -> int | None:
        """frames.csv row of the newest camera frame received at or before ``t``, written or not (the first if earlier)."""
        times = self.data.frame_t
        if len(times) == 0:
            return None
        return max(0, int(np.searchsorted(times, t + 1e-9, side="right")) - 1)

    def step_frame(self, direction: int) -> None:
        """One camera frame back (-1) or forward (+1), by the recorded frame times."""
        if self.data is None or len(self.data.frame_t) == 0:
            return
        self.pause()
        times = self.data.frame_t
        row = int(np.searchsorted(times, self._t + 1e-9, side="right")) - 1
        target = min(max(row + (1 if direction > 0 else -1), 0), len(times) - 1)
        self.seek(float(times[target]))

    def jump_event(self, direction: int) -> None:
        """To the nearest event before (-1) or after (+1) the selected time."""
        if self.data is None or not self._events:
            return
        self.pause()
        times = [row["t"] for row in self._events]
        if direction > 0:
            later = [t for t in times if t > self._t + 1e-4]
            if later:
                self.seek(later[0])
        else:
            earlier = [t for t in times if t < self._t - 1e-4]
            if earlier:
                self.seek(earlier[-1])

    def _on_event_clicked(self, item: QTreeWidgetItem, _column: int = 0) -> None:
        t = item.data(0, Qt.ItemDataRole.UserRole)
        if t is not None:
            self.pause()
            self.seek(float(t))

    # ---- playback ----

    def play(self) -> None:
        if self.data is None or self.duration <= 0.0:
            return
        if self._t >= self.duration - 1e-3:
            self.seek(0.0)
        self._anchor_playback()
        self._play_timer.start()
        self.play_button.setText("Pause")

    def pause(self) -> None:
        self._play_timer.stop()
        self.play_button.setText("▶ Play")

    def toggle_play(self) -> None:
        if self.playing:
            self.pause()
        else:
            self.play()

    def _anchor_playback(self) -> None:
        self._play_anchor = (time.monotonic(), self._t)

    def _play_tick(self) -> None:
        """Real time: the selected time follows the wall clock, and the frame shown is the one recorded for it."""
        wall, start = self._play_anchor
        t = start + (time.monotonic() - wall) * float(self.speed_box.currentData() or 1.0)
        if t >= self.duration:
            self._play_timer.stop()
            self.play_button.setText("▶ Play")
            t = self.duration
        self._t = t
        with QSignalBlocker(self.slider):
            self.slider.setValue(int(round(t * 1000.0)))
        self._refresh()

    # ---- drawing the selected moment ----

    def _refresh(self) -> None:
        if self.data is None:
            return
        t = self._t
        frame_text = self._update_video(t)
        self.time_label.setText(f"{format_time(t)} / {format_time(self.duration)}  frame {frame_text}")
        pose = pose_readout(self.data, t)
        self._update_map(t, pose)
        self._update_moment(t, pose)
        for plot in self.plots:
            plot.set_cursor(t)
        self._highlight_event(t)
        self.time_changed.emit(t)

    def _update_video(self, t: float) -> str:
        """Show the frame for ``t``. Returns the camera frame number for the time label ('–' if none)."""
        data = self.data
        row = self._frame_row(t)
        if row is None:
            key = ("none",)
            if key != self._shown_frame:
                self._shown_frame = key
                self.current_frame_row = None
                self.frame_view.set_placeholder("No frames in this round")
                self.frame_view.set_frame(None)
                self.video_caption.setText("No video frames were recorded")
            return "–"
        number = int(data.frame_index[row])
        received = float(data.frame_t[row])
        if ("row", row) != (self._shown_frame or ())[:2]:
            # frame_at() only ever answers with a frame that is in the video; a dropped frame gets no stand-in image.
            frame = data.read_frame(row) if data.frame_at(max(t, received)) == row else None
            self._shown_frame = ("row", row, frame is not None)
            if frame is None:
                self.current_frame_row = None
                if not data.frame_written[row]:
                    why = f"Frame {number} is not in the video\n(dropped or refused during recording)"
                elif data.video_path is None:
                    why = "No video file in this round"
                else:
                    why = f"Frame {number} could not be read from {data.video_path.name}"
                self.frame_view.set_placeholder(why)
                self.frame_view.set_frame(None)
            else:
                self.current_frame_row = row
                self.frame_view.set_frame(frame)
        caption = f"camera frame {number}, received by the PC at {received:.3f} s"
        if not self._shown_frame[2]:
            caption += " · no image for this frame"
        age = t - received
        if age > 0.25:
            caption += f" · no newer frame: this one is {age:.1f} s before the selected time"
        elif age < -1e-6:
            caption += " · first frame of the recording (selected time is before it)"
        self.video_caption.setText(caption)
        return str(number)

    def _update_map(self, t: float, pose: dict[str, Any]) -> None:
        data = self.data
        self.map_position.setText(pose["text"] + (f"   [{pose['source']}]" if pose["known"] else ""))
        self.map_position.setStyleSheet("" if pose["known"] else f"color: {WARN};")
        if self.map is None:
            return
        trails: list[Trail] = []
        any_valid = False
        for track, colour, dashed, label in ((data.pose_live, LIVE_COLOUR, False, "live"),
                                             (data.pose_offline, OFFLINE_COLOUR, True, "reconstructed offline")):
            if len(track):
                any_valid = any_valid or bool(track.valid.any())
                trails.append(Trail(track.segments(t), colour, dashed, label, 2.0))
        if data.opponent:
            for track_id, runs in opponent_runs(data.opponent, t).items():
                trails.append(Trail(runs, OPPONENT_COLOUR, False, f"opponent {track_id}", 2.0))
        markers = []
        both = len(pose["valid"]) == 2            # two labels at nearly the same place would overprint: use one
        for source, sample in pose["valid"].items():
            offline = source == "offline"
            if both:
                label = "" if offline else "● live   ○ reconstructed offline"
            else:
                label = "reconstructed offline" if offline else "live"
            markers.append(Marker(sample["x_m"], sample["y_m"], sample["heading_deg"], OFFLINE_COLOUR if offline else LIVE_COLOUR,
                                  label, hollow=offline))
        note = ""
        if not any_valid:
            if not data.calibrated:
                note = "No calibration in this round: no camera positions"
            elif not len(data.pose_live) and not len(data.pose_offline):
                note = "No own-bot track in this round"
            else:
                note = "No valid own-bot position in this round"
        caption = "a break in a trail = position unknown (gaps are never bridged)" if trails else ""
        self.map.set_content(trails=trails, markers=markers, caption=caption, note=note)

    def _update_moment(self, t: float, pose: dict[str, Any]) -> None:
        data = self.data
        control = control_readout(data, t)
        fresh = control["freshness"] == "FRESH"
        source_colour = {"RADIO": OK, "PC": WARN, "NONE": OFF, "STALE": BAD, "UNKNOWN": WARN}.get(control["source"], "#d9dee5")
        self.source_value.setText(control["source"])
        self.source_value.setStyleSheet(f"color: {source_colour};")
        self.requested_value.setText(control["requested"])
        self.armed_value.setText(control["armed"])
        armed_colour = {"ARMED": WARN, "DISARMED": OK, "STALE": BAD, "UNKNOWN": WARN}.get(control["armed"], "#d9dee5")
        self.armed_value.setStyleSheet(f"color: {armed_colour};")
        stale_colour = "" if fresh else f"color: {BAD if control['freshness'] == 'STALE' else WARN};"
        for label, key in ((self.requested_value, "requested"), (self.flags_value, "flags"), (self.motors_value, "motors")):
            label.setText(control[key])
            label.setStyleSheet(stale_colour)
        self.telemetry_value.setText(control["telemetry"])
        self.telemetry_value.setStyleSheet(stale_colour)
        self.pc_sent_value.setText(pc_sent_readout(data, t))
        self.policy_value.setText(policy_readout(data, t))
        self.pose_value.setText(pose["text"])
        self.pose_value.setStyleSheet("" if pose["known"] else f"color: {WARN};")
        self.pose_source_value.setText(pose["source_text"])
        self.imu_value.setText(imu_readout(data, t))

    def _highlight_event(self, t: float) -> None:
        index = -1
        for i, row in enumerate(self._events):
            if row["t"] <= t + 1e-4:
                index = i
            else:
                break
        if index == self._current_event:
            return
        self._current_event = index
        with QSignalBlocker(self.events_view):
            if index < 0:
                self.events_view.setCurrentItem(None)
            else:
                self.events_view.setCurrentItem(self._event_items[index])
                self.events_view.scrollToItem(self._event_items[index])

    # ---- offline reconstruction ----

    def reconstruct_offline(self) -> bool:
        """Start re-detecting the own-bot track from the recorded video in a background thread."""
        if self.data is None or self._worker is not None:
            return False
        self.pause()
        self.reconstruct_button.setEnabled(False)
        self.open_button.setEnabled(False)
        self.latest_button.setEnabled(False)
        self.progress.setRange(0, 0)
        self.progress.show()
        self.status_label.setText("Reconstructing the own-bot track from the recorded video (offline)…")
        self._worker = ReconstructWorker(self.data.folder, self)
        self._worker.progress.connect(self._on_reconstruct_progress)
        self._worker.done.connect(self._on_reconstruct_done)
        self._worker.start()
        return True

    def _on_reconstruct_progress(self, done: int, total: int) -> None:
        self.progress.setRange(0, max(1, total))
        self.progress.setValue(done)

    def _on_reconstruct_done(self, result: dict[str, Any]) -> None:
        worker, self._worker = self._worker, None
        if worker is not None:
            worker.wait(5000)
            worker.deleteLater()
        self.progress.hide()
        self.open_button.setEnabled(True)
        self.latest_button.setEnabled(True)
        self.reconstruct_button.setEnabled(self.data is not None)
        if not result.get("ok"):
            self.status_label.setText(f"Offline reconstruction not done: {result.get('reason', 'no reason given')}")
            return
        folder = self.data.folder
        ok, message = self._load(folder, keep_time=True)
        if ok:
            self.status_label.setText(f"Reconstructed OFFLINE from the video: {result.get('frames_processed', '?')} frames, "
                                      f"{result.get('valid_poses', '?')} valid poses, saved as a separate dashed track "
                                      "(derived/pose_offline.csv). The live track is unchanged.")
        else:
            self.status_label.setText(f"Reconstruction finished but the round did not reload: {message}")

    # ---- housekeeping ----

    def _on_round_saved(self, summary) -> None:
        folder = getattr(summary, "folder", None)
        if folder is not None:
            self.status_label.setText(f"New round saved: {folder} (use Open latest to review it)")

    def shutdown(self) -> None:
        """Stop playback, wait for a running reconstruction and release the video file."""
        self.pause()
        if self._worker is not None:
            self._worker.wait(60000)
        if self.data is not None:
            self.data.close()

    def hideEvent(self, event) -> None:  # noqa: N802 - Qt API
        self.pause()
        super().hideEvent(event)

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt API
        self.shutdown()
        super().closeEvent(event)


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    app = QApplication.instance() or QApplication(sys.argv[:1])
    try:
        from control_center.app import STYLESHEET

        app.setStyleSheet(STYLESHEET)
    except Exception:  # the look is optional for stand-alone review
        log.warning("dashboard stylesheet not available; using the default look")
    screen = ReviewScreen()
    screen.setWindowTitle("Capri Command Centre · Round review (recorded data)")
    screen.resize(1500, 900)
    if arguments:
        ok, message = screen.open_round(arguments[0])
        if not ok:
            print(message, file=sys.stderr)
    screen.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
