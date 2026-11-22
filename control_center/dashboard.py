"""The Capri Command Centre window: one application for the whole competition-day workflow.

    GUIDED SETUP    the checklist: camera, arena, tracking, hardware and safety, live monitoring
    OPERATE         camera or map, control source, robot link, recording
    ARENA           click the arena corners, pits and flipper; save per profile
    TRACKING        markers, own robot, click the opponent, AI shadow (display only)
    ELRS & CONTROL  who has control, link health, safety checks, motion check
    SETUP           camera, floor-landmark calibration, markers, controller and IMU details
    REVIEW          one scrubber over a recorded round
    DIAGNOSTICS     sim-to-real comparison and policy shadow checks

The profile selector (HOME TEST / LIVE) at the top decides which calibration every tab uses. Nothing in
this window arms, hands over or drives the robot; the RadioMaster does that, with or without this window.
The first version of the dashboard is still available unchanged (``--legacy``).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QComboBox, QHBoxLayout, QLabel, QMainWindow, QTabWidget, QVBoxLayout, QWidget)

from control_center.backend.hub import HubOptions, OperationsHub
from control_center.backend.profiles import PROFILE_ORDER, profile_label
from control_center.screens.operate import OperateScreen

log = logging.getLogger(__name__)
TABS = ("status", "guided", "arena", "tracking", "connections", "ai", "control", "match", "operate", "setup", "review", "diagnostics")
TAB_TITLES = {"status": "SYSTEM STATUS", "guided": "GUIDED SETUP", "arena": "ARENA", "tracking": "TRACKING", "connections": "CONNECTIONS",
              "ai": "AI / MODEL", "control": "ELRS & CONTROL", "match": "MATCH", "operate": "MONITOR", "setup": "SETUP",
              "review": "REVIEW", "diagnostics": "DIAGNOSTICS"}
PROFILE_BADGE = {"live": ("LIVE · competition arena", "#d34e59"), "home_test": ("HOME TEST · A4 patch, not the arena", "#3c6f9c")}


def _placeholder(title: str, error: Exception) -> QWidget:
    """Shown instead of a screen that failed to load, so the rest of the dashboard still runs."""
    widget = QWidget()
    layout = QVBoxLayout(widget)
    label = QLabel(f"{title} is unavailable in this build.\n\n{type(error).__name__}: {error}\n\nThe other tabs still work. Details are in logs/.")
    label.setAlignment(Qt.AlignmentFlag.AlignCenter)
    label.setWordWrap(True)
    label.setStyleSheet("color: #e8b85b; font-size: 12pt;")
    layout.addWidget(label)
    return widget


class DashboardWindow(QMainWindow):
    def __init__(self, config: dict[str, Any], options: HubOptions, start_tab: str | None = None,
                 review_folder: Path | None = None) -> None:
        super().__init__()
        self.hub = OperationsHub(config, options, self)
        self.setWindowTitle("Capri Command Centre" + ("  —  MOCK (not hardware)" if options.mock else ""))
        self.resize(1600, 950)
        self.setMinimumSize(1180, 720)

        central = QWidget()
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 6, 0, 0)
        root.setSpacing(4)
        banner = QHBoxLayout()
        banner.setContentsMargins(12, 0, 12, 0)
        title = QLabel("CAPRI COMMAND CENTRE")
        title.setObjectName("appTitle")
        banner.addWidget(title)
        banner.addSpacing(16)
        banner.addWidget(QLabel("Profile:"))
        self.profile_box = QComboBox()
        for name in PROFILE_ORDER:
            self.profile_box.addItem(profile_label(name), name)
        self.profile_box.setToolTip("Each profile keeps its own calibration, arena and progress. They never share data.")
        banner.addWidget(self.profile_box)
        self.profile_badge = QLabel()
        banner.addWidget(self.profile_badge)
        self.profile_message = QLabel("")
        self.profile_message.setStyleSheet("color: #e8b85b;")
        banner.addWidget(self.profile_message)
        banner.addStretch(1)
        if options.mock:
            badge = QLabel("MOCK MODE · synthetic camera and replayed controller lines · nothing here is from hardware")
            badge.setObjectName("demoBadge")
            banner.addWidget(badge)
        elif options.video is not None:
            badge = QLabel(f"REPLAY · {options.video.name} · not a live camera")
            badge.setObjectName("demoBadge")
            banner.addWidget(badge)
        root.addLayout(banner)

        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)
        self.operate = OperateScreen(self.hub)
        self.connections = self._load("Connections", "control_center.screens.connections", "ConnectionsScreen",
                                      service=getattr(self.hub, "link_service", None))
        self.arena = self._load("Arena", "control_center.screens.arena", "ArenaScreen")
        self.tracking = self._load("Tracking", "control_center.screens.tracking", "TrackingScreen")
        # AI / MODEL and MATCH are built after TRACKING and CONNECTIONS: they read its results and drive the one link service.
        self.ai = self._load("AI / Model", "control_center.screens.ai_model", "AiModelScreen", tracking=self.tracking)
        self.match = self._load("Match", "control_center.screens.match", "MatchScreen", tracking=self.tracking, ai=self.ai,
                                service=getattr(self.hub, "link_service", None))
        self.control = self._load("ELRS & Control", "control_center.screens.control", "ControlScreen")
        self.setup = self._load("Setup", "control_center.screens.setup", "SetupScreen")
        self.review = self._load("Review", "control_center.screens.review", "ReviewScreen")
        self.diagnostics = self._load("Diagnostics", "control_center.screens.diagnostics", "DiagnosticsScreen")
        self.guided = self._load("Guided setup", "control_center.screens.wizard", "GuidedSetupScreen",
                                 navigate=self.go_to, providers=self._providers())
        self.status = self._load("System status", "control_center.screens.status", "SystemStatusScreen",
                                 navigate=self.go_to, providers=self._status_providers(), safe_state=self._safe_state_text)
        self.screens = {"status": self.status, "guided": self.guided, "arena": self.arena, "tracking": self.tracking,
                        "connections": self.connections, "ai": self.ai, "control": self.control, "match": self.match,
                        "operate": self.operate, "setup": self.setup, "review": self.review, "diagnostics": self.diagnostics}
        for key in TABS:
            self.tabs.addTab(self.screens[key], TAB_TITLES[key])
        root.addWidget(self.tabs, 1)
        self.setCentralWidget(central)

        self.operate.review_requested.connect(self.open_review)
        self._sync_profile_box()
        self.profile_box.currentIndexChanged.connect(self._on_profile_chosen)
        self.hub.profile_changed.connect(lambda _name: self._sync_profile_box())
        self.hub.start()
        state = self.hub.app_state
        if review_folder is not None:
            self.open_review(str(review_folder))
        else:
            wanted = start_tab or state.last_tab or "status"
            self.tabs.setCurrentIndex(TABS.index(wanted if wanted in TABS else "status"))
        self.tabs.currentChanged.connect(lambda index: state.set_last_tab(TABS[index]))
        if state.load_problem:
            self.profile_message.setText(state.load_problem)

    # ---- tabs
    def _load(self, title: str, module_name: str, class_name: str, **kwargs: Any) -> QWidget:
        try:
            module = __import__(module_name, fromlist=[class_name])
            return getattr(module, class_name)(self.hub, **kwargs)
        except Exception as error:
            log.exception("%s screen failed to load", title)
            return _placeholder(title, error)

    def _providers(self) -> dict[str, Any]:
        """What the guided page asks for each step's status. Resolved late: the screens are built after this."""
        def of(attribute: str):
            def ask():
                screen = getattr(self, attribute, None)
                readiness = getattr(screen, "readiness", None)
                return readiness() if readiness is not None else None
            return ask
        def opponent():
            from control_center.backend import system_status as ss

            result = getattr(getattr(self, "tracking", None), "_result", None)
            fresh = result is not None and getattr(self.tracking, "readiness_for", None) is not None
            level = getattr(result, "map_level", "none") if result is not None else "none"
            return ss.opponent_readiness(result.opponent if fresh else None, calibrated=level not in ("none", "invalid") or result is None)
        return {"arena": of("arena"), "tracking": of("tracking"), "opponent": opponent, "connections": of("connections"),
                "safety": of("control"), "operate": of("match")}

    def _status_providers(self) -> dict[str, Any]:
        """What SYSTEM STATUS asks for each row. Resolved late: the screens are built after this."""
        def of(attribute: str, method: str = "readiness"):
            def ask():
                screen = getattr(self, attribute, None)
                function = getattr(screen, method, None)
                return function() if function is not None else None
            return ask
        guided = self._providers()
        return {"camera": of("guided", "camera_readiness"), "arena": guided["arena"], "tracking": guided["tracking"],
                "opponent": guided["opponent"], "connections": of("connections"), "control": of("control"),
                "ai": of("ai"), "autonomous": of("match", "autonomous_readiness")}

    def _safe_state_text(self) -> str:
        service = getattr(self.hub, "link_service", None)
        try:
            snap = service.latest() if service is not None else None
        except Exception:
            snap = None
        if snap is None:
            return "SAFE STATE: MANUAL control (RadioMaster). This application is not transmitting motor commands."
        if snap.transmitting:
            return f"TRANSMITTING: {snap.mode.mode.value}, authority {snap.authority.authority.value}. Manual recovery: RadioMaster ARM switch."
        return (f"SAFE STATE: {snap.mode.mode.value}, authority {snap.authority.authority.value}. Not transmitting motor commands"
                + (" (SIMULATED session)." if snap.simulated else "."))

    def go_to(self, key: str) -> None:
        if key in TABS:
            self.tabs.setCurrentIndex(TABS.index(key))

    def open_review(self, folder: str) -> None:
        self.tabs.setCurrentIndex(TABS.index("review"))
        opener = getattr(self.review, "open_round", None)
        if opener is None:
            return
        try:
            result = opener(Path(folder))
        except Exception:
            log.exception("could not open round %s", folder)
            return
        if isinstance(result, tuple) and not result[0]:
            log.error("could not open round %s: %s", folder, result[1])

    # ---- profile
    def _sync_profile_box(self) -> None:
        name = self.hub.profile_name
        self.profile_box.blockSignals(True)
        self.profile_box.setCurrentIndex(max(0, self.profile_box.findData(name)))
        self.profile_box.blockSignals(False)
        text, colour = PROFILE_BADGE.get(name, (name, "#7b8794"))
        self.profile_badge.setText(text)
        self.profile_badge.setStyleSheet(f"background: {colour}; color: white; font-weight: 800; padding: 3px 10px; border-radius: 4px;")
        self.profile_box.setEnabled(not self.hub.options.mock)

    def _on_profile_chosen(self, _index: int) -> None:
        name = self.profile_box.currentData()
        ok, message = self.hub.set_profile(name)
        self.profile_message.setText("" if ok else message)
        self._sync_profile_box()

    # ---- shutdown
    def closeEvent(self, event) -> None:  # noqa: N802 - Qt API
        log.info("Command Centre stopping")
        # Screens first (their timers, worker threads and any console they started), then the hub that owns the camera.
        for screen in (self.match, self.connections, self.ai, self.status, self.tracking, self.control, self.guided, self.arena,
                       self.diagnostics, self.review, self.setup):
            shutdown = getattr(screen, "shutdown", None)
            if shutdown is not None:
                try:
                    shutdown()
                except Exception:
                    log.exception("%s did not shut down cleanly", type(screen).__name__)
        self.hub.close()
        event.accept()
