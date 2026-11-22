"""Application startup, configuration, and styling."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys

import yaml
from PySide6.QtWidgets import QApplication

from control_center.backend.logging_manager import configure_logging


ROOT = Path(__file__).resolve().parent.parent


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as stream:
        data = yaml.safe_load(stream) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Configuration root must be a mapping: {path}")
    return data


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Capri Command Centre")
    parser.add_argument("--config", type=Path, default=ROOT / "config" / "control_center.yaml")
    parser.add_argument("--camera-index", type=int, help="Override the configured OpenCV camera index")
    parser.add_argument("--mock", action="store_true",
                        help="No hardware: synthetic camera and replayed controller lines, labelled MOCK everywhere")
    parser.add_argument("--video", type=Path, help="Replay a video file in place of the camera (labelled REPLAY)")
    parser.add_argument("--controller-port", help="Listen (read-only) to the controller on this serial port, e.g. COM5")
    parser.add_argument("--controller-log", type=Path,
                        help="Follow the file written by scripts/handoff_bench_client.py --log (when that client holds the port)")
    parser.add_argument("--controller-relay", nargs="?", type=int, const=4211, metavar="PORT",
                        help="Listen to the local telemetry relay of scripts/telemetry_link.py (default port 4211): status and IMU")
    parser.add_argument("--rounds-dir", type=Path, help="Where round folders are written (default: rounds/)")
    parser.add_argument("--review", type=Path, metavar="ROUND_FOLDER", help="Open a recorded round for review; no camera is opened")
    parser.add_argument("--tab", choices=("status", "guided", "arena", "tracking", "connections", "ai", "control", "match", "operate", "setup", "review", "diagnostics"),
                        help="Tab to open (default: the one used last time, or SYSTEM STATUS the first time)")
    parser.add_argument("--state-file", type=Path, help="Where the profile, last tab and setup progress are remembered (default: config/command_centre_state.json)")
    parser.add_argument("--settings-dir", type=Path,
                        help="Where each profile's saved camera, calibration context and settings go (default: config/profile_settings/)")
    parser.add_argument("--profile", choices=("live", "home_test"),
                        help="Calibration profile: live = the competition arena, home_test = the test patch (default: the one used last time)")
    parser.add_argument("--legacy", action="store_true", help="Open the first version of the dashboard instead")
    parser.add_argument("--demo", action="store_true", help="(legacy window only) show clearly labelled simulated telemetry")
    return parser


def hub_options(args: argparse.Namespace, config: dict):
    """Command line and config -> what the dashboard opens. Nothing is opened here."""
    from control_center.backend.hub import HubOptions

    rounds = args.rounds_dir or Path(config.get("rounds_dir", "rounds"))
    return HubOptions(
        camera_index=int(config.get("camera_index", 1)), mock=bool(args.mock), video=args.video,
        controller_port=args.controller_port or config.get("controller_port") or None,
        controller_log=args.controller_log, controller_relay=args.controller_relay, rounds_root=rounds if rounds.is_absolute() else ROOT / rounds,
        codec=str(config.get("recording_codec", "MJPG")),
        start_camera=args.review is None or bool(args.mock) or args.video is not None,
        profile=args.profile, state_path=None if args.mock else (args.state_file or ROOT / "config" / "command_centre_state.json"),
        settings_dir=None if args.mock or args.video is not None else (args.settings_dir or ROOT / "config" / "profile_settings"),
        camera_index_explicit=args.camera_index is not None,
        # logs follow the state file's folder when one is given (tests and --state-file runs), else logs/sessions
        session_log_dir=None if args.review is not None else (args.state_file.parent / "logs" / "sessions" if args.state_file else ROOT / "logs" / "sessions"))


def apply_saved_camera_settings(args: argparse.Namespace, config: dict) -> dict:
    """Camera index, picture size and fps chosen in GUIDED SETUP last time (an explicit --camera-index wins)."""
    if args.mock or args.video is not None:
        return config
    from control_center.backend.app_state import AppState

    saved = AppState(args.state_file or ROOT / "config" / "command_centre_state.json").camera_settings()
    for key in ("camera_index", "camera_width", "camera_height", "camera_fps"):
        if key in saved and not (key == "camera_index" and args.camera_index is not None):
            config[key] = saved[key]
    return config


def run(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    log_path = configure_logging(ROOT / "logs")
    try:
        config = load_config(args.config)
    except Exception as exc:
        logging.getLogger(__name__).exception("Could not load configuration")
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2
    if args.camera_index is not None:
        config["camera_index"] = args.camera_index
    apply_saved_camera_settings(args, config)
    app = QApplication(sys.argv[:1])
    app.setApplicationName("Capri Command Centre")
    app.setStyle("Fusion")
    app.setStyleSheet(STYLESHEET)
    if args.legacy:
        from control_center.main_window import MainWindow

        window = MainWindow(config, demo=args.demo)
    else:
        from control_center.dashboard import DashboardWindow

        window = DashboardWindow(config, hub_options(args, config), start_tab=args.tab, review_folder=args.review)
    window.show()
    logging.getLogger(__name__).info("Control Center started; log=%s", log_path)
    code = app.exec()
    logging.getLogger(__name__).info("Control Center stopped")
    return code


STYLESHEET = """
QWidget { background: #11161d; color: #d9dee5; font-family: 'Segoe UI'; font-size: 10pt; }
QMainWindow { background: #0b0f14; }
QFrame#panel { background: #171e27; border: 1px solid #2b3542; border-radius: 5px; }
QFrame#panel QLabel, QFrame#panel QCheckBox { background: transparent; border: none; }
QLabel#appTitle { font-size: 18pt; font-weight: 700; letter-spacing: 1px; }
QLabel#panelTitle { color: #9eabb8; font-size: 10pt; font-weight: 700; letter-spacing: 1px; }
QLabel#modeBadge { color: #6fa8dc; font-weight: 700; padding: 5px 10px; border: 1px solid #3c6f9c; }
QLabel#demoBadge { color: #11161d; background: #e8b85b; font-weight: 800; padding: 5px 10px; }
QPushButton { background: #263241; border: 1px solid #3b4a5d; border-radius: 4px; padding: 7px 10px; font-size: 10pt; font-weight: 650; }
QPushButton:hover { background: #324256; }
QPushButton:disabled { color: #66717f; background: #1b232d; }
QPushButton#warningButton { background: #694b20; border-color: #a9782e; }
QPushButton#criticalButton { background: #8c2931; border-color: #d34e59; font-size: 11pt; padding: 12px; }
QPlainTextEdit { background: #0d1218; border: 1px solid #2b3542; color: #b8c2cc; font-family: Consolas; }
QSplitter::handle { background: #0b0f14; width: 5px; height: 5px; }
QScrollArea#sidebarScroll { background: transparent; border: none; }
QWidget#sidebarContent { background: transparent; }
QSlider::groove:horizontal { height: 5px; background: #2b3542; }
QSlider::handle:horizontal { background: #6fa8dc; width: 15px; margin: -5px 0; border-radius: 7px; }
QTabWidget::pane { border: none; border-top: 1px solid #2b3542; }
QTabBar::tab { background: #171e27; color: #9eabb8; padding: 9px 11px; margin-right: 2px; font-weight: 700; letter-spacing: 1px; }
QTabBar::tab:selected { background: #263241; color: #ffffff; border-bottom: 2px solid #6fa8dc; }
QTabBar::tab:hover { color: #d9dee5; }
QPushButton:checked { background: #3c6f9c; border-color: #6fa8dc; color: #ffffff; }
QPushButton#recordButton { background: #8c2931; border-color: #d34e59; font-size: 13pt; font-weight: 800; }
QPushButton#recordButton:hover { background: #a5323b; }
QPushButton#recordButton:disabled { background: #3a2327; color: #8d6b6f; border-color: #5a3237; }
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox { background: #0d1218; border: 1px solid #2b3542; border-radius: 3px; padding: 5px; }
QListWidget, QTableWidget, QTreeWidget { background: #0d1218; border: 1px solid #2b3542; alternate-background-color: #121922; }
QHeaderView::section { background: #171e27; color: #9eabb8; border: none; border-bottom: 1px solid #2b3542; padding: 4px; font-weight: 700; }
QTableWidget::item:selected, QListWidget::item:selected { background: #2f4a66; color: #ffffff; }
QCheckBox { spacing: 6px; }
QToolTip { background: #263241; color: #d9dee5; border: 1px solid #3b4a5d; }
QScrollArea { border: none; }
"""
