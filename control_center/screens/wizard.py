"""GUIDED SETUP: the competition-day checklist in one place.

Seven steps in the order of a competition day, each with a status (Not configured / Ready / Warning / Failed /
Blocked), what is wrong, and what to do:

    1 CAMERA SETUP          pick the camera, see the preview, set the picture size, check the field of view
    2 ARENA CALIBRATION     the ARENA tab: restore the saved calibration and check the alignment, or click the corners and hazards
    3 ROBOT TRACKING        the TRACKING tab: ArUco markers, own robot position and heading
    4 OPPONENT SELECTION    the TRACKING tab: drag a box around the opponent (chosen again every round)
    5 CONNECTIONS           the CONNECTIONS tab: Wi-Fi to the robot, acceptance record
    6 SAFETY CHECKS         the ELRS & CONTROL tab: manual control, safety checks, motion check
    7 OPERATE               the MATCH tab: MANUAL / AI SHADOW / AUTONOMOUS, Start Round

The statuses come from the screens themselves (each answers ``readiness()``); this page only shows them and
jumps to the right tab. Nothing here starts a test, moves the robot, or changes a setting except the camera
index / picture size you apply on this page. The step you were on and the last status of each step are
remembered per profile.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (QComboBox, QFrame, QGridLayout, QHBoxLayout, QLabel, QPushButton, QScrollArea, QSpinBox,
                               QVBoxLayout, QWidget)

from control_center.backend import camera_check
from control_center.backend.app_state import STEP_IDS
from control_center.backend.profiles import profile_label
from control_center.backend.readiness import BLOCKED, CHIP_COLOURS, FAILED, NOT_CONFIGURED, READY, WARNING, Readiness, worst
from control_center.backend.setup_tools import camera_rows
from control_center.tools.arena_setup import CORNER_IDS, CORNER_PREFIX
from control_center.widgets.frame_view import FrameView

log = logging.getLogger(__name__)
REFRESH_MS = 1000
PREVIEW_MS = 100

STEP_TITLES = {
    "camera": "1  CAMERA SETUP", "arena": "2  ARENA CALIBRATION", "tracking": "3  ROBOT TRACKING", "opponent": "4  OPPONENT SELECTION",
    "connections": "5  CONNECTIONS (WI-FI)", "safety": "6  SAFETY CHECKS", "operate": "7  OPERATE",
}
STEP_TABS = {"camera": None, "arena": "arena", "tracking": "tracking", "opponent": "tracking", "connections": "connections",
             "safety": "control", "operate": "match"}
STEP_BUTTONS = {"camera": "Camera settings below", "arena": "Open ARENA", "tracking": "Open TRACKING", "opponent": "Open TRACKING",
                "connections": "Open CONNECTIONS", "safety": "Open ELRS & CONTROL", "operate": "Open MATCH"}
RESOLUTIONS = ((640, 480), (800, 600), (1280, 720), (1920, 1080))
FPS_CHOICES = (15, 30, 60)
STYLE = """
QLabel { background: transparent; }
QFrame#step { background: #171e27; border: 1px solid #2b3542; border-radius: 6px; }
QFrame#step[current="true"] { border: 2px solid #6fa8dc; }
QLabel#stepTitle { font-size: 11pt; font-weight: 800; letter-spacing: 1px; }
QLabel#chip { font-weight: 800; padding: 3px 10px; border-radius: 4px; color: #11161d; }
QLabel#detail { color: #b9c3ce; }
QLabel#fix { color: #e8b85b; }
QLabel#summary { font-size: 13pt; font-weight: 800; }
"""


def _label(text: str = "", name: str | None = None) -> QLabel:
    label = QLabel(text)
    label.setTextFormat(Qt.TextFormat.PlainText)
    label.setWordWrap(True)
    if name:
        label.setObjectName(name)
    return label


class StepCard(QFrame):
    def __init__(self, step: str, on_open: Callable[[str], None], on_retry: Callable[[str], None], on_select: Callable[[str], None]) -> None:
        super().__init__()
        self.step = step
        self.setObjectName("step")
        self.setProperty("current", False)
        layout = QGridLayout(self)
        layout.setContentsMargins(12, 8, 12, 8)
        self.title = _label(STEP_TITLES[step], "stepTitle")
        self.chip = QLabel(NOT_CONFIGURED.upper())
        self.chip.setObjectName("chip")
        self.headline = _label("")
        self.headline.setStyleSheet("font-weight: 700;")
        self.detail = _label("", "detail")
        self.fix = _label("", "fix")
        self.open_button = QPushButton(STEP_BUTTONS[step])
        self.open_button.setEnabled(STEP_TABS[step] is not None)
        self.open_button.clicked.connect(lambda: on_open(step))
        self.retry_button = QPushButton("Retry check")
        self.retry_button.clicked.connect(lambda: on_retry(step))
        self.select_button = QPushButton("Here")
        self.select_button.setToolTip("Make this the current step")
        self.select_button.clicked.connect(lambda: on_select(step))
        layout.addWidget(self.title, 0, 0)
        layout.addWidget(self.chip, 0, 1, Qt.AlignmentFlag.AlignRight)
        layout.addWidget(self.headline, 1, 0, 1, 2)
        layout.addWidget(self.detail, 2, 0, 1, 2)
        layout.addWidget(self.fix, 3, 0, 1, 2)
        buttons = QHBoxLayout()
        for button in (self.open_button, self.retry_button, self.select_button):
            buttons.addWidget(button)
        buttons.addStretch(1)
        layout.addLayout(buttons, 4, 0, 1, 2)
        layout.setColumnStretch(0, 1)

    def show_readiness(self, readiness: Readiness, remembered: str = "") -> None:
        self.chip.setText(readiness.level.upper())
        self.chip.setStyleSheet(f"background: {CHIP_COLOURS[readiness.level]};")
        self.headline.setText(readiness.headline)
        self.detail.setText(readiness.detail + (f"\n{remembered}" if remembered else ""))
        self.detail.setVisible(bool(readiness.detail or remembered))
        self.fix.setText(("Do this: " + readiness.fix) if readiness.fix else "")
        self.fix.setVisible(bool(readiness.fix))

    def set_current(self, current: bool) -> None:
        self.setProperty("current", current)
        self.style().unpolish(self)
        self.style().polish(self)


class GuidedSetupScreen(QWidget):
    def __init__(self, hub, navigate: Callable[[str], None] | None = None, providers: dict[str, Callable[[], Readiness | None]] | None = None,
                 parent=None) -> None:
        super().__init__(parent)
        self.hub = hub
        self.navigate = navigate or (lambda _tab: None)
        self.providers = providers if providers is not None else {}
        self.setStyleSheet(STYLE)
        self.statuses: dict[str, Readiness] = {}

        root = QHBoxLayout(self)
        root.setContentsMargins(10, 8, 10, 8)
        left = QVBoxLayout()
        self.summary = _label("", "summary")
        left.addWidget(self.summary)
        self.profile_note = _label("")
        left.addWidget(self.profile_note)
        self.cards: dict[str, StepCard] = {}
        body = QWidget()
        column = QVBoxLayout(body)
        column.setSpacing(8)
        for step in STEP_IDS:
            card = StepCard(step, self.open_step, self.retry_step, self.select_step)
            self.cards[step] = card
            column.addWidget(card)
        column.addStretch(1)
        area = QScrollArea()
        area.setWidgetResizable(True)
        area.setWidget(body)
        area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        left.addWidget(area, 1)
        nav = QHBoxLayout()
        self.back_button = QPushButton("◀ Back")
        self.next_button = QPushButton("Next ▶")
        self.save_button = QPushButton("Save progress")
        self.save_button.setToolTip("Progress and the last status of each step are also saved automatically")
        self.back_button.clicked.connect(lambda: self.move(-1))
        self.next_button.clicked.connect(lambda: self.move(1))
        self.save_button.clicked.connect(self.save_progress)
        for button in (self.back_button, self.next_button, self.save_button):
            nav.addWidget(button)
        nav.addStretch(1)
        self.save_note = _label("")
        nav.addWidget(self.save_note, 1)
        left.addLayout(nav)
        root.addLayout(left, 3)
        root.addWidget(self._build_camera_panel(), 2)

        self._timer = QTimer(self)
        self._timer.setInterval(REFRESH_MS)
        self._timer.timeout.connect(self.refresh)
        self._preview_timer = QTimer(self)
        self._preview_timer.setInterval(PREVIEW_MS)
        self._preview_timer.timeout.connect(self._paint_preview)
        if hasattr(hub, "profile_changed"):
            hub.profile_changed.connect(lambda _name: self.refresh())
        self.refresh()

    # ================================================================ camera step panel
    def _build_camera_panel(self) -> QWidget:
        frame = QFrame()
        frame.setObjectName("step")
        layout = QVBoxLayout(frame)
        heading = _label("CAMERA PREVIEW AND SETTINGS", "stepTitle")
        layout.addWidget(heading)
        self.preview = FrameView()
        self.preview.setMinimumSize(360, 220)
        layout.addWidget(self.preview, 1)
        row = QHBoxLayout()
        row.addWidget(QLabel("Camera index"))
        self.index_spin = QSpinBox()
        self.index_spin.setRange(0, 9)
        self.index_spin.setValue(int(getattr(self.hub.options, "camera_index", 1)))
        row.addWidget(self.index_spin)
        row.addWidget(QLabel("Size"))
        self.size_combo = QComboBox()
        for width, height in RESOLUTIONS:
            self.size_combo.addItem(f"{width} x {height}")
        row.addWidget(self.size_combo)
        row.addWidget(QLabel("fps"))
        self.fps_combo = QComboBox()
        for fps in FPS_CHOICES:
            self.fps_combo.addItem(str(fps))
        row.addWidget(self.fps_combo)
        layout.addLayout(row)
        self._load_camera_choices()
        buttons = QHBoxLayout()
        self.apply_button = QPushButton("Open camera with these settings")
        self.apply_button.clicked.connect(self.apply_camera)
        buttons.addWidget(self.apply_button)
        buttons.addStretch(1)
        layout.addLayout(buttons)
        self.camera_message = _label("")
        layout.addWidget(self.camera_message)
        self.camera_info = _label("")
        self.camera_info.setObjectName("detail")
        layout.addWidget(self.camera_info)
        mock = bool(getattr(self.hub.options, "mock", False)) or getattr(self.hub.options, "video", None) is not None
        if mock:
            for widget in (self.index_spin, self.size_combo, self.fps_combo, self.apply_button):
                widget.setEnabled(False)
            self.camera_message.setText("This session uses a mock or replay source, so there is no camera to configure. Restart without --mock / --video.")
        return frame

    def _load_camera_choices(self) -> None:
        config = getattr(self.hub, "config", {}) or {}
        size = (int(config.get("camera_width", 1280)), int(config.get("camera_height", 720)))
        if size in RESOLUTIONS:
            self.size_combo.setCurrentIndex(RESOLUTIONS.index(size))
        fps = int(config.get("camera_fps", 30))
        if fps in FPS_CHOICES:
            self.fps_combo.setCurrentIndex(FPS_CHOICES.index(fps))

    def apply_camera(self) -> tuple[bool, str]:
        width, height = RESOLUTIONS[max(0, self.size_combo.currentIndex())]
        fps = FPS_CHOICES[max(0, self.fps_combo.currentIndex())]
        index = self.index_spin.value()
        settings = {"camera_index": index, "camera_width": width, "camera_height": height, "camera_fps": fps}
        self.hub.config.update(settings)                      # the hub's opener reads these when it opens the camera
        ok, message = self.hub.reopen_camera(index)
        if ok:
            state = getattr(self.hub, "app_state", None)
            if state is not None:
                state.set_camera_settings(settings)
        self.camera_message.setText((f"Opening camera {index} at {width} x {height}, {fps} fps. " if ok else "Not applied: ") + message)
        self.camera_message.setStyleSheet("color: #55c889;" if ok else "color: #ef6b73;")
        return ok, message

    def _paint_preview(self) -> None:
        try:
            _index, _t, frame = self.hub.capture.latest()
        except Exception:
            frame = None
        if frame is None:
            self.preview.set_frame(None)
            self.preview.set_placeholder("No camera frame yet")
        else:
            self.preview.set_frame(frame)

    # ================================================================ statuses
    def _corner_pixels(self) -> list[tuple[float, float]] | None:
        clicks = getattr(getattr(self.hub, "vision", None), "clicks", None)
        if clicks is None:
            return None
        points = [clicks.points.get(f"{CORNER_PREFIX}{cid}") for cid in CORNER_IDS]
        return None if any(p is None for p in points) else [(float(p[0]), float(p[1])) for p in points]

    def camera_readiness(self) -> Readiness:
        capture = self.hub.capture
        options = self.hub.options
        base = camera_check.camera_state(capture.state, capture.detail, float(capture.measured_fps or 0.0), capture.last_frame_age(),
                                         mock=bool(getattr(options, "mock", False)), replay=getattr(options, "video", None) is not None)
        loaded = getattr(self.hub, "loaded_profile", None)
        arena = None if loaded is None else loaded.arena
        sides = None if arena is None else (arena.geometry.width_m, arena.geometry.height_m)
        config = self.hub.vision.config
        marker = next(iter(config.markers.values()), None)
        fov = camera_check.field_of_view(self._corner_pixels(), capture.frame_size, sides, marker.size_m if marker else None,
                                         "bot markers" in config.placeholder_tags)
        level = worst([base, fov])
        if base.level in (FAILED, NOT_CONFIGURED):
            return base
        if fov.level == FAILED:
            return fov
        detail = " | ".join(part for part in (base.detail, fov.detail) if part)
        headline = base.headline if base.level != READY else fov.headline
        return Readiness(level, headline, detail, base.fix or fov.fix)

    def monitoring_readiness(self) -> Readiness:
        arena, tracking = self.statuses.get("arena"), self.statuses.get("tracking")
        if arena is None or not arena.usable:
            return Readiness(NOT_CONFIGURED, "Waiting for the arena calibration", "The arena overlay and both tracks need a calibrated arena.", "Complete step 2.")
        if tracking is None or tracking.level in (NOT_CONFIGURED, FAILED):
            return Readiness(NOT_CONFIGURED, "Waiting for robot tracking", "Both robots must be tracked to be shown on the arena.", "Complete step 3.")
        try:
            missing = list(self.hub.live_shadow_missing())
        except Exception:
            missing = ["shadow prerequisites could not be read"]
        shadow = "AI SHADOW: available" if not missing else "AI SHADOW blocked: " + "; ".join(missing)
        detail = shadow + ". AUTONOMOUS mode is disabled in this build: no verified laptop-to-robot command path."
        if arena.level == WARNING or tracking.level == WARNING:
            return Readiness(WARNING, "Monitoring available, with warnings from steps 2 or 3", detail, "Read the warnings of those steps.")
        return Readiness(READY, "Both robots can be shown on the calibrated arena", detail)

    def _status_of(self, step: str) -> Readiness:
        if step == "camera":
            return self.camera_readiness()
        provider = self.providers.get(step)
        if provider is None:
            return Readiness(NOT_CONFIGURED, "This screen is not available in this build", "", "See the log in logs/ for the reason.")
        try:
            result = provider()
        except Exception as error:                            # a broken screen must not break the checklist
            log.exception("readiness of %s failed", step)
            return Readiness(FAILED, "The status check itself failed", f"{type(error).__name__}: {error}", "Open the tab and read its message; see logs/.")
        return result or Readiness(NOT_CONFIGURED, "No status available yet")

    def refresh(self) -> None:
        profile = getattr(self.hub, "profile_name", "live")
        state = getattr(self.hub, "app_state", None)
        current = state.wizard_step(profile) if state is not None else STEP_IDS[0]
        for step in STEP_IDS:
            try:
                self.statuses[step] = self._status_of(step)
            except Exception as error:
                self.statuses[step] = Readiness(FAILED, "Status check failed", str(error))
            remembered = ""
            self.cards[step].show_readiness(self.statuses[step], remembered)
            self.cards[step].set_current(step == current)
            if state is not None:
                state.record_step(profile, step, self.statuses[step].level, self.statuses[step].headline)
        bad = [s for s in STEP_IDS if self.statuses[s].level in (FAILED, NOT_CONFIGURED)]
        warn = [s for s in STEP_IDS if self.statuses[s].level == WARNING]
        blocked = [s for s in STEP_IDS if self.statuses[s].level == BLOCKED]
        if not bad and not warn and blocked:
            self.summary.setText(f"{len(blocked)} step(s) BLOCKED until a physical prerequisite is met: " + ", ".join(STEP_TITLES[s][3:].title() for s in blocked))
            self.summary.setStyleSheet(f"color: {CHIP_COLOURS[BLOCKED]};")
        elif not bad and not warn:
            self.summary.setText(f"All {len(STEP_IDS)} steps are Ready.")
            self.summary.setStyleSheet(f"color: {CHIP_COLOURS[READY]};")
        elif bad:
            self.summary.setText(f"{len(bad)} step(s) need attention: " + ", ".join(STEP_TITLES[s][3:].title() for s in bad))
            self.summary.setStyleSheet(f"color: {CHIP_COLOURS[FAILED]};")
        else:
            self.summary.setText(f"Ready with {len(warn)} warning(s): " + ", ".join(STEP_TITLES[s][3:].title() for s in warn))
            self.summary.setStyleSheet(f"color: {CHIP_COLOURS[WARNING]};")
        self.profile_note.setText(f"Profile: {profile_label(profile)}. Each profile keeps its own calibration and progress; "
                                  "nothing carries over between HOME TEST and LIVE.")
        self.back_button.setEnabled(STEP_IDS.index(current) > 0)
        self.next_button.setEnabled(STEP_IDS.index(current) < len(STEP_IDS) - 1)
        self._update_camera_info()

    def _update_camera_info(self) -> None:
        try:
            rows = camera_rows(self.hub.capture, self.hub.options)
            self.camera_info.setText("\n".join(f"{name}: {text}" for name, text, _level in rows))
        except Exception:
            self.camera_info.setText("")

    # ================================================================ navigation
    def open_step(self, step: str) -> None:
        self.select_step(step)
        tab = STEP_TABS.get(step)
        if tab:
            self.navigate(tab)

    def retry_step(self, step: str) -> None:
        self.refresh()

    def select_step(self, step: str) -> None:
        state = getattr(self.hub, "app_state", None)
        if state is not None:
            state.set_wizard_step(getattr(self.hub, "profile_name", "live"), step)
        self.refresh()

    def move(self, direction: int) -> None:
        state = getattr(self.hub, "app_state", None)
        profile = getattr(self.hub, "profile_name", "live")
        current = state.wizard_step(profile) if state is not None else STEP_IDS[0]
        target = STEP_IDS[max(0, min(len(STEP_IDS) - 1, STEP_IDS.index(current) + direction))]
        self.open_step(target)

    def save_progress(self) -> bool:
        state = getattr(self.hub, "app_state", None)
        if state is None or state.path is None:
            self.save_note.setText("Nothing saved: this session keeps no progress file (mock mode).")
            return False
        self.refresh()
        ok = state.save()
        self.save_note.setText(f"Saved to {state.path}" if ok else "NOT saved: the progress file could not be written (see logs/).")
        return ok

    # ================================================================ lifecycle
    def showEvent(self, event) -> None:  # noqa: N802 - Qt API
        self._timer.start()
        self._preview_timer.start()
        self.refresh()
        super().showEvent(event)

    def hideEvent(self, event) -> None:  # noqa: N802 - Qt API
        self._timer.stop()
        self._preview_timer.stop()
        super().hideEvent(event)

    def shutdown(self) -> None:
        self._timer.stop()
        self._preview_timer.stop()

    def readiness(self) -> Readiness:
        levels = [s.level for s in self.statuses.values()]
        if not levels:
            return Readiness(NOT_CONFIGURED, "No status yet")
        return Readiness(worst(list(self.statuses.values())), "Guided setup")
