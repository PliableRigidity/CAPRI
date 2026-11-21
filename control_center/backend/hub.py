"""The dashboard's single owner of camera, vision, controller telemetry, policy state and recording.

Screens read from the hub and ask it to do things; they never open a camera or a port themselves.
The hub has no way to command the robot: its controller sources are listen-only.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from PySide6.QtCore import QObject, QThread, QTimer, Signal

from control_center.backend import status as st
from control_center.backend.app_state import AppState
from control_center.backend.profiles import PROFILE_DEFS, LoadedProfile, load_profile
from control_center.backend.capture import (CaptureSession, MockArenaSource, VideoFileSource, VisionThread,
                                            camera_settings_from_config, open_camera)
from control_center.backend.controller_link import (ClientLogSource, ControllerMonitor, ScriptedSource, SerialLineSource,
                                                    demo_script, parse_imu_line, parse_pc_command, parse_status_line)
from control_center.backend.camera_signature import CameraSignature
from control_center.backend.live_vision import LiveVision, VisionSample, load_live_clicks
from control_center.backend.profile_settings import ProfileSettingsStore
from control_center.backend.session_log import SessionLog
from control_center.backend import startup_restore as sr
from control_center.backend.round_recorder import RoundRecorder, RoundSummary, new_round_folder, pose_row, software_version
from control_center.tools.overlay_core import LIVE_PROFILE, load_config as load_overlay_config

log = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[2]


@dataclass
class HubOptions:
    camera_index: int = 1
    mock: bool = False                      # synthetic camera + replayed controller lines, labelled MOCK
    video: Path | None = None               # replay a video file instead of a camera, labelled REPLAY
    controller_port: str | None = None      # listen-only serial port, e.g. COM5
    controller_log: Path | None = None      # follow the PC drive client's --log file
    controller_relay: int | None = None     # listen to the local relay of scripts/telemetry_link.py on this port
    rounds_root: Path = ROOT / "rounds"
    codec: str = "MJPG"
    start_camera: bool = True
    profile: str | None = None              # "live" or "home_test"; default: the one used last time
    state_path: Path | None = None          # where progress is remembered; None = in memory only (mock mode, tests)
    settings_dir: Path | None = None        # per-profile saved camera / markers / Wi-Fi / calibration context; None = nothing saved
    camera_index_explicit: bool = False     # --camera-index given: it wins over the camera remembered with the calibration
    session_log_dir: Path | None = None     # where session / round logs go (logs/sessions); None = no session log (tests)


class ControllerLinkThread(QThread):
    """Polls a listen-only line source. Emits (generation, [(direction, text, t_mono), ...])."""

    lines = Signal(object)
    failed = Signal(str)

    def __init__(self, source, generation: int = 0, parent=None) -> None:
        super().__init__(parent)
        self.source = source
        self.generation = generation        # lets the hub ignore lines still queued from a source it has replaced
        self._running = True

    def stop(self) -> None:
        self._running = False

    def run(self) -> None:
        while self._running:
            try:
                batch = self.source.read()
            except Exception as error:
                self.failed.emit(str(error))
                break
            if batch:
                now = time.monotonic()
                self.lines.emit((self.generation, [(direction, text, now) for direction, text in batch]))
            time.sleep(0.01)
        try:
            self.source.close()
        except Exception:
            log.exception("closing controller source")


class OperationsHub(QObject):
    snapshot_changed = Signal(object)        # status.Snapshot, 10 Hz
    event_added = Signal(str, str, str)      # "HH:MM:SS", kind, text
    sample_ready = Signal(object)            # VisionSample
    recording_finished = Signal(object)      # RoundSummary
    controller_line = Signal(str, str)       # direction, text (for the Setup screen's raw view)
    controller_source_changed = Signal(str)  # new source label, "" when disconnected
    profile_changed = Signal(str)            # profile name: another profile was loaded, or its saved set-up changed
    restore_changed = Signal(object)         # startup_restore.RestoreReport (or None): calibration trust changed

    def __init__(self, config: dict[str, Any], options: HubOptions, parent=None) -> None:
        super().__init__(parent)
        self.config = config
        self.options = options
        self.app_state = AppState(options.state_path)
        name = options.profile or self.app_state.profile or LIVE_PROFILE
        self.profile_name = name if name in PROFILE_DEFS else LIVE_PROFILE
        if options.profile and self.app_state.profile != self.profile_name and not options.mock:
            self.app_state.set_profile(self.profile_name)        # --profile is remembered for the next start
        self.loaded_profile: LoadedProfile | None = None
        self.overlay_config = load_overlay_config()
        self.events: deque[tuple[str, str, str]] = deque(maxlen=400)
        self.monitor = ControllerMonitor()
        self.link_label: str | None = None
        self.link_error: str | None = None
        self.link_thread: ControllerLinkThread | None = None
        self._link_generation = 0
        self.recorder: RoundRecorder | None = None
        self.last_summary: RoundSummary | None = None
        self._stopping = False
        self.policy: dict[str, Any] = {"mode": "OFF", "input": None, "checkpoint": None, "reason": "", "proposed": None}
        self.latest_sample: VisionSample | None = None
        self._tracking: bool | None = None
        self._last_valid_t: float | None = None
        self._lost_since: float | None = None
        self._last_tag_switch_event = 0.0
        self.mock_source: MockArenaSource | None = None
        # What is remembered per profile between runs (camera mode, calibration context, reference picture). Never the
        # opponent, control authority, armed state, tracking observations or link status (the store refuses those keys).
        self.settings_store = ProfileSettingsStore(options.settings_dir) if options.settings_dir is not None else None
        self.restore_report: sr.RestoreReport | None = None
        self.reconnect_policy = sr.CameraReconnectPolicy()
        self._restore_pending = True
        self._camera_outage = False
        # One log folder per application run: camera / tracking / decisions / command plans / transmissions / mode changes / faults.
        self.session_log = SessionLog(options.session_log_dir, label=self.profile_name) if options.session_log_dir is not None else None
        # The ONE Wi-Fi link / mode manager / acceptance record. Created here, started in start(), stopped in close().
        # It starts disconnected in MANUAL; nothing is transmitted until the operator connects AND every gate is open.
        self.link_service = self._make_link_service()

        if options.mock:
            self.mock_source = MockArenaSource(self.overlay_config)
            self.vision = LiveVision(self.overlay_config, self.mock_source.clicks(), profile="synthetic fixture")
        else:
            self.loaded_profile = load_profile(self.profile_name)
            self.overlay_config = self.loaded_profile.config
            self.vision = LiveVision(self.overlay_config, self.loaded_profile.clicks, self.profile_name,
                                     not_calibrated_reason=self.loaded_profile.reason or "no calibration saved yet")
        startup_note = self._startup_note()          # may change options.camera_index: the camera it was calibrated on
        self.capture = CaptureSession(self._opener(options.camera_index), self)
        self.vision_thread = VisionThread(self.capture, self.vision, self)
        self.vision_thread.listeners.append(self._log_sample)
        self.vision_thread.sample_ready.connect(self._on_sample)
        self.capture.state_changed.connect(self._on_camera_state)
        self.add_event("restore", startup_note)

        self.recording_finished.connect(self._on_recording_finished)
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)

    # ---- start / stop ----

    def _make_link_service(self):
        try:
            from control_center.backend.link_service import LinkService

            return LinkService(store=self.settings_store, profile=self.profile_name, mock_session=bool(self.options.mock),
                               parent=self)
        except Exception:
            log.exception("the Wi-Fi link service could not be created; Wi-Fi control is unavailable")
            return None

    def start(self) -> None:
        if self.link_service is not None:
            self.link_service.start()
        self.add_event("app", "Command Centre started" + (" in MOCK mode: nothing shown is from hardware" if self.options.mock else ""))
        if self.options.start_camera:
            self.capture.start()
            self.vision_thread.start()
        if self.options.mock:
            self.connect_controller(ScriptedSource(demo_script(with_imu=True), loop=True))
        elif self.options.controller_port:
            self.connect_controller_port(self.options.controller_port)
        elif self.options.controller_log:
            self.connect_controller_log(self.options.controller_log)
        elif self.options.controller_relay:
            from control_center.backend.controller_link import RelaySource

            self.connect_controller(RelaySource(self.options.controller_relay))
        self._timer.start(100)

    def close(self) -> None:
        self._timer.stop()
        if self.link_service is not None:
            self.link_service.shutdown()                 # neutral if anything was ever sent, then the worker thread stops
        if self.recorder is not None:
            self.stop_recording(wait=True)
        self.disconnect_controller()
        self.vision_thread.stop()
        self.capture.stop()
        for thread in (self.vision_thread, self.capture):
            if thread.isRunning() and not thread.wait(3000):
                log.error("%s did not stop within 3 s", type(thread).__name__)
        if self.session_log is not None:
            self.session_log.close()

    # ---- camera ----

    def _opener(self, index: int):
        if self.options.mock:
            return lambda: self.mock_source
        if self.options.video is not None:
            return lambda: VideoFileSource(self.options.video)
        settings = camera_settings_from_config(self.config)
        plan = self._camera_plan()
        if plan is not None:
            settings = {**settings, **plan.settings}
        return lambda: open_camera(index, settings)

    def _camera_plan(self) -> "sr.CameraOpenPlan | None":
        if self.settings_store is None or self.options.mock or self.options.video is not None:
            return None
        try:
            return sr.camera_open_settings(self.profile_name, self.settings_store, {**camera_settings_from_config(self.config),
                                                                                     "index": self.options.camera_index})
        except Exception:
            log.exception("could not read the remembered camera settings; using the configured ones")
            return None

    def _startup_note(self) -> str:
        plan = self._camera_plan()
        if plan is None:
            return "Nothing is remembered between runs in this session" if self.settings_store is None else "No remembered camera settings"
        if plan.index is not None and not self.options.camera_index_explicit and plan.index != self.options.camera_index:
            self.options.camera_index = int(plan.index)
        return "Remembered camera settings applied" + (": " + "; ".join(plan.notes) if plan.notes else "")

    def reopen_camera(self, index: int) -> tuple[bool, str]:
        if self.recorder is not None:
            return False, "Stop the recording before changing the camera"
        if self.options.mock or self.options.video is not None:
            return False, "This session uses a mock or replay source; restart without --mock / --video to use a camera"
        self.options.camera_index = index
        self.capture.request_reopen(self._opener(index))
        self.add_event("camera", f"Camera index set to {index}; reopening")
        return True, "reopening"

    def _on_camera_state(self, state: str, detail: str) -> None:
        self.add_event("camera", f"Camera {state}: {detail}" if detail else f"Camera {state}")
        self.reconnect_policy.on_event(state, detail)
        if state in ("LOST", "UNAVAILABLE"):
            self._camera_outage = True
        elif state == "LIVE":
            if self._camera_outage:
                self._camera_outage = False
                self._restore_pending = True            # the camera came back: look at the picture again before trusting it

    def camera_status_text(self) -> str:
        """Plain statement for the screens: opening, lost, retrying in N s, or live."""
        return self.reconnect_policy.status_text()

    def reconnect_camera(self) -> tuple[bool, str]:
        """Manual reconnect (the camera was unplugged and plugged back in). Resets the retry back-off."""
        self.reconnect_policy.manual_reconnect()
        return self.reopen_camera(self.options.camera_index)

    # ---- remembered calibration: restore, quick alignment check, confirmation ----

    def current_signature(self) -> CameraSignature | None:
        info = getattr(self.capture, "source_info", None)
        if not info or getattr(self.capture, "frame_size", None) is None:
            return None
        return CameraSignature.from_capture_info(info, self.capture.frame_size)

    def evaluate_restore(self) -> "sr.RestoreReport | None":
        """How far the saved calibration can be trusted now (saved? same camera mode? picture still aligned?).
        Needs the settings store and a real camera; MOCK and REPLAY sessions have nothing remembered to restore."""
        if self.settings_store is None or self.options.mock or self.options.video is not None or self.loaded_profile is None:
            return None
        previous = self.restore_report
        _index, _t, frame = self.capture.latest()
        report = sr.restore_calibration(self.profile_name, self.settings_store, self.loaded_profile, self.current_signature(), frame)
        if previous is not None and previous.world_coordinates_trusted and report.trust is sr.TrustLevel.NEEDS_CHECK and frame is not None \
                and not self._camera_outage and report.settings_match is not None and report.settings_match.level == "MATCH" \
                and (report.alignment is None or report.alignment.status == "MATCH"):
            report = sr.confirm_alignment(report)         # a refresh of what was already confirmed this run, nothing changed
        self.restore_report = report
        self._restore_pending = False
        self.restore_changed.emit(report)
        self.add_event("restore", f"{report.trust.value}: {report.headline}")
        return report

    def _maybe_restore(self) -> None:
        if self._restore_pending and self.settings_store is not None and self.capture.latest()[2] is not None:
            self.evaluate_restore()

    @property
    def world_coordinates_trusted(self) -> bool | None:
        """True after the operator confirmed alignment this run; False when a remembered calibration is not (yet) trusted;
        None when this session has no remembered calibration to judge (MOCK, REPLAY, no settings folder)."""
        if self.restore_report is None:
            return None
        return self.restore_report.world_coordinates_trusted

    def confirm_alignment(self) -> tuple[bool, str]:
        """The operator compared the overlay with the real arena and pressed 'Alignment OK'."""
        report = self.restore_report
        if report is None:
            return False, "There is no remembered calibration to confirm in this session"
        try:
            confirmed = sr.confirm_alignment(report)
        except sr.ConfirmationRefused as error:
            return False, str(error)
        if self.settings_store is not None:
            sr.record_confirmation(self.profile_name, self.settings_store, confirmed)
            if report.needs_reference_frame:
                self._store_reference()
        self.restore_report = confirmed
        self.restore_changed.emit(confirmed)
        self.add_event("restore", "Alignment confirmed by the operator; world coordinates trusted for this run")
        return True, "Alignment confirmed"

    def _store_reference(self) -> None:
        loaded = self.loaded_profile
        _index, _t, frame = self.capture.latest()
        if self.settings_store is None or loaded is None or loaded.clicks is None or loaded.arena is None or frame is None:
            return
        corners = [loaded.clicks.points[name] for name in sorted(loaded.clicks.points) if name.startswith("arena_corner_")]
        sr.record_calibration(self.profile_name, self.settings_store, self.current_signature(), corners, loaded.arena.setup_id, frame)

    def calibration_saved(self, frame, corners_px) -> None:
        """Called by the ARENA tab after a successful Save: remembers the camera mode and a reference picture."""
        if self.settings_store is None or frame is None or self.options.mock:
            return
        loaded = self.loaded_profile
        setup_id = loaded.arena.setup_id if loaded is not None and loaded.arena is not None else ""
        sr.record_calibration(self.profile_name, self.settings_store, self.current_signature(), list(corners_px), setup_id, frame)
        self.remember_markers()
        self._restore_pending = True

    def remember_markers(self) -> None:
        """The ArUco dictionary, the marker layout and any measured values in force, remembered for this profile.
        Values that were never measured stay None (a placeholder is not remembered as a measurement)."""
        config = getattr(self.vision, "config", None)
        if self.settings_store is None or config is None or self.options.mock:
            return
        import hashlib
        import json

        try:
            layout = {str(k): {"size_m": getattr(m, "size_m", None)} for k, m in sorted(config.markers.items())}
            digest = hashlib.sha256(json.dumps(layout, sort_keys=True, default=str).encode()).hexdigest()[:16]
            measured = "bot markers" not in getattr(config, "placeholder_tags", [])
            self.settings_store.update_section(self.profile_name, "markers", {
                "aruco_dictionary": config.dictionary, "layout_name": "bot.markers", "layout_path": str(getattr(config, "path", "") or ""),
                "layout_hash": digest, "measured_offsets": layout if measured else {}})
        except Exception:
            log.exception("could not remember the marker settings")

    def save_camera_preferences(self) -> None:
        """The index, picture size and fps the operator chose, remembered for this profile."""
        signature = self.current_signature()
        if self.settings_store is not None and signature is not None:
            sr.save_camera_preferences(self.profile_name, self.settings_store, signature)

    # ---- calibration ----

    def reload_calibration(self) -> str:
        """Read the active profile's files again (after saving, or to verify what was saved)."""
        if self.options.mock:
            return "MOCK mode uses the synthetic fixture's calibration"
        loaded = load_profile(self.profile_name)
        self._apply_loaded(loaded)
        self._restore_pending = True
        if loaded.problem:
            return loaded.problem
        return f"profile '{self.profile_name}' loaded" if loaded.clicks is not None else loaded.reason

    def _apply_loaded(self, loaded: LoadedProfile) -> None:
        self.loaded_profile = loaded
        self.overlay_config = loaded.config
        self.vision.set_config(loaded.config)
        self.vision.set_clicks(loaded.clicks, loaded.name, loaded.problem or loaded.reason)
        self.add_event("calibration", f"Profile '{loaded.name}': " + (loaded.problem or (
            "map calibration loaded" if loaded.clicks is not None else f"map not calibrated: {loaded.reason}")))
        self.profile_changed.emit(loaded.name)

    def set_profile(self, name: str) -> tuple[bool, str]:
        """Switch between HOME TEST and LIVE. Each loads only its own files; nothing carries over."""
        if name not in PROFILE_DEFS:
            return False, f"unknown profile {name!r}"
        if self.options.mock:
            return False, "MOCK mode uses the synthetic fixture; restart without --mock to use a profile"
        if self.recorder is not None:
            return False, "Stop the recording before changing profile"
        if name == self.profile_name:
            return True, "already active"
        self.profile_name = name
        self.app_state.set_profile(name)
        self.restore_report = None
        self._restore_pending = True
        self._apply_loaded(load_profile(name))
        return True, f"profile '{name}' loaded"

    def apply_clicks(self, clicks, reason: str = "") -> None:
        self.vision.set_clicks(clicks, self.profile_name, reason)
        self.add_event("calibration", "Map calibration loaded" if clicks is not None else f"Map not calibrated: {reason}")

    # ---- controller telemetry (listen-only) ----

    def connect_controller(self, source) -> None:
        self.disconnect_controller()
        self.monitor = ControllerMonitor()
        self.link_label, self.link_error = source.label, None
        self._link_generation += 1
        self.link_thread = ControllerLinkThread(source, self._link_generation, self)
        self.link_thread.lines.connect(self._on_lines)
        self.link_thread.failed.connect(self._on_link_failed)
        self.link_thread.start()
        self.add_event("controller_link", f"Listening to controller: {source.label}", record_link=True)
        self.controller_source_changed.emit(source.label)

    def connect_controller_port(self, port: str) -> tuple[bool, str]:
        try:
            self.connect_controller(SerialLineSource(port))
            return True, f"listening on {port}"
        except Exception as error:
            self.link_label, self.link_error = f"{port} (listen-only)", str(error)
            self.add_event("controller_link", f"Could not open {port}: {error}")
            return False, str(error)

    def connect_controller_log(self, path: Path | str) -> tuple[bool, str]:
        try:
            self.connect_controller(ClientLogSource(path))
            return True, f"following {path}"
        except Exception as error:
            self.link_label, self.link_error = f"client log {Path(path).name}", str(error)
            self.add_event("controller_link", f"Could not follow {path}: {error}")
            return False, str(error)

    def disconnect_controller(self) -> None:
        if self.link_thread is not None:
            self.link_thread.stop()
            self.link_thread.wait(2000)
            self.link_thread = None
            self._link_generation += 1
            self.add_event("controller_link", "Controller telemetry source closed", record_link=True)
            self.controller_source_changed.emit("")
        self.link_label = None

    def _on_link_failed(self, message: str) -> None:
        self.link_error = message
        self.add_event("controller_link", f"Controller telemetry source failed: {message}", record_link=True)

    def _on_lines(self, payload: tuple[int, list[tuple[str, str, float]]]) -> None:
        generation, batch = payload
        if generation != self._link_generation:
            return                           # queued by a source that has since been closed or replaced
        recorder = self.recorder
        for direction, text, t in batch:
            for kind, message, data in self.monitor.feed(text, t, direction):
                self.add_event(kind, message, t_mono=t, **data)
            if recorder is not None:
                self._record_line(recorder, direction, text, t)
            self.controller_line.emit(direction, text)

    @staticmethod
    def _record_line(recorder: RoundRecorder, direction: str, text: str, t: float) -> None:
        if direction == "TX":
            command = parse_pc_command(text)
            if command is not None:
                recorder.log_control(t, "pc_sent", {"token": command.token, "throttle": command.throttle, "steer": command.steer,
                                                    "note": "sent by the PC client; not a confirmation that it was applied"})
            return
        status = parse_status_line(text)
        if status is not None:
            recorder.log_control(t, "controller_status", {
                "device_ms": status.device_ms, "link": status.link, "armed": status.armed, "requested": status.requested,
                "active": status.active, "pc_ok": status.pc_ok, "block": status.block, "fault": status.fault,
                "stop": status.stop, "left": status.left, "right": status.right, "pc_age_ms": status.pc_age_ms,
                "token": status.token})
            return
        imu = parse_imu_line(text)
        if imu is not None:
            recorder.log_imu(t, imu.device_ms, imu.accel_m_s2, imu.gyro_deg_s, imu.sequence, imu.device_us)

    # ---- vision ----

    def _log_sample(self, sample: VisionSample) -> None:
        """Runs in the vision thread: every processed frame gets a pose row, valid or not."""
        recorder = self.recorder
        if recorder is not None:
            recorder.log_pose(sample.t_mono, sample.frame_index, pose_row(sample.pose, sample.tag_switch))

    def _on_sample(self, sample: VisionSample) -> None:
        self.latest_sample = sample
        if sample.pose.ok:
            if self._tracking is False and self._lost_since is not None:
                self.add_event("tracking_reacquired", f"Own-bot tracking reacquired after {sample.t_mono - self._lost_since:.1f} s",
                               t_mono=sample.t_mono, gap_s=round(sample.t_mono - self._lost_since, 2))
            self._tracking, self._last_valid_t, self._lost_since = True, sample.t_mono, None
            if sample.tag_switch and sample.jump_mm >= 10.0 and sample.t_mono - self._last_tag_switch_event > 5.0:
                self._last_tag_switch_event = sample.t_mono
                self.add_event("tag_switch", f"Own-bot markers in use changed to {sample.pose.used_ids}; the "
                                             f"{sample.jump_mm:.0f} mm step is a marker-offset effect, not movement",
                               t_mono=sample.t_mono, used_ids=sample.pose.used_ids, jump_mm=round(sample.jump_mm, 1))
        elif self._tracking and self._last_valid_t is not None and sample.t_mono - self._last_valid_t >= st.TRACKING_LOST_AFTER_S:
            self._tracking, self._lost_since = False, self._last_valid_t
            self.add_event("tracking_lost", f"Own-bot tracking lost: {sample.pose.reason}", t_mono=sample.t_mono, reason=sample.pose.reason)
        self.sample_ready.emit(sample)

    # ---- policy (display only) ----

    def live_shadow_missing(self) -> list[str]:
        """Inputs a live observation would need that this dashboard cannot supply. Empty = available."""
        now = time.monotonic()
        missing = ["a live observation adapter (camera pose + IMU -> the policy's observation vector) does not exist in this repository"]
        if not self.monitor.imu_present(now):
            missing.append("gyro and accelerometer: the controller is not sending an IMU stream")
        if self.vision.map_state().level not in ("ok", "weak"):
            missing.append("own position and heading: the live camera map is not calibrated")
        if self.world_coordinates_trusted is False:
            missing.append("world coordinates are not trusted yet: " + (self.restore_report.headline if self.restore_report else ""))
        missing.append("opponent position: there is no live opponent tracker")
        return missing

    def set_policy(self, mode: str, input_source: str | None = None, checkpoint: str | None = None,
                   proposed: dict[str, Any] | None = None, reason: str = "") -> tuple[bool, str]:
        if mode not in ("OFF", "SIM", "LIVE SHADOW"):
            return False, f"unknown policy mode {mode}"
        if mode == "LIVE SHADOW":
            missing = self.live_shadow_missing()
            if missing:
                return False, "LIVE SHADOW unavailable: " + "; ".join(missing)
        changed = (mode, checkpoint) != (self.policy["mode"], self.policy["checkpoint"])
        self.policy = {"mode": mode, "input": input_source, "checkpoint": checkpoint, "reason": reason, "proposed": proposed}
        if changed:
            self.add_event("policy", f"Policy {mode}" + (f" · {checkpoint} · input {input_source}" if mode != "OFF" else ""))
        recorder = self.recorder
        if recorder is not None and (changed or proposed is not None):
            recorder.log_policy(time.monotonic(), {"mode": mode, "input_source": input_source, "checkpoint": checkpoint,
                                                   "proposed": proposed, "note": "proposed by a policy; display only, never sent"})
        return True, "ok"

    # ---- events ----

    def add_event(self, kind: str, text: str, t_mono: float | None = None, record_link: bool = False, **data: Any) -> None:
        stamp = datetime.now().strftime("%H:%M:%S")
        self.events.append((stamp, kind, text))
        log.info("[%s] %s", kind, text)
        if getattr(self, "session_log", None) is not None:
            from control_center.backend.session_log import KINDS

            self.session_log.log(kind if kind in KINDS else "note", text, source=kind if kind not in KINDS else "hub")
        recorder = self.recorder
        if recorder is not None and kind != "app":
            when = time.monotonic() if t_mono is None else t_mono
            recorder.log_event(when, kind, text, **data)
            if record_link:
                recorder.log_control(when, "link", {"text": text, "source": self.link_label})
        self.event_added.emit(stamp, kind, text)

    # ---- recording ----

    def build_manifest(self, label: str) -> dict[str, Any]:
        now = time.monotonic()
        authority = self.monitor.authority(now)
        return {
            "label": label,
            "software": software_version(ROOT),
            "mock": bool(self.options.mock),
            "mock_note": "MOCK session: synthetic camera and replayed controller lines, not hardware" if self.options.mock else None,
            "camera": {**self.capture.source_info, "state_at_start": self.capture.state,
                       "frame_size": list(self.capture.frame_size) if self.capture.frame_size else None,
                       "measured_fps_at_start": round(self.capture.measured_fps, 2),
                       "configured": camera_settings_from_config(self.config)},
            "calibration": self.vision.calibration_snapshot(),
            "calibration_note": None if self.vision.clicks is not None else self.vision.not_calibrated_reason,
            "marker_layout": self.vision.marker_layout(),
            "arena": dict(self.overlay_config.arena),
            "controller": {"source": self.link_label, "telemetry_at_start": authority.freshness,
                           "firmware": self.monitor.firmware(), "imu_stream_at_start": self.monitor.imu_present(now)},
            "imu": {"units": {"accel": "m/s^2", "gyro": "deg/s"},
                    "axes": "sensor axes exactly as the controller sends them (IMU,<device_ms>,ax,ay,az,gx,gy,gz); "
                            "no rotation into a robot frame is applied and the mounting orientation is not recorded",
                    "note": "the in-repo handoff bench sketch sends no IMU line; this stream is present only if a firmware sends one"},
            "pose": {"frame": "arena-fixed frame of config/arena_overlay.yaml: origin at the plain corner, metres, "
                              "heading in degrees counter-clockwise from +x",
                     "gaps": "an invalid or missing detection is a row with valid=0 and no numbers; the last good position is never repeated"},
            "policy": {"mode": self.policy["mode"], "input_source": self.policy["input"], "checkpoint": self.policy["checkpoint"]},
            "opponent": {"live_tracker": False, "note": "no live opponent tracker; the footage blob tracker is offline only"},
        }

    def start_recording(self, label: str = "") -> tuple[bool, str]:
        if self.recorder is not None or self._stopping:
            return False, "already recording"
        if self.capture.frame_size is None:
            return False, "no camera frames yet: recording needs video"
        try:
            folder = new_round_folder(self.options.rounds_root, label)
            recorder = RoundRecorder(folder, self.build_manifest(label), fps=self.capture.nominal_fps, codec=self.options.codec)
            recorder.start()
        except Exception as error:
            log.exception("could not start recording")
            return False, f"could not start recording: {error}"
        # What the controller was already reporting goes in at t_s = 0, so a review of the first
        # moments shows it instead of UNKNOWN.
        authority = self.monitor.authority(time.monotonic())
        start = recorder.t0_mono
        recorder.log_control(start, "link", {"text": f"telemetry {authority.freshness} at recording start", "source": self.link_label})
        if authority.status is not None and authority.fresh:
            self._record_line(recorder, "RX", self._status_text(authority.status), start)
        self.add_event("recording_started", f"Recording to {folder}")      # screen only: the recorder logs its own start
        self.recorder = recorder
        self.capture.set_recorder(recorder)
        return True, str(folder)

    @staticmethod
    def _status_text(status) -> str:
        from control_center.backend.controller_link import status_line
        return status_line(status.device_ms, status.token, int(status.link), int(status.armed), status.requested, status.active,
                           int(status.pc_ok), status.block, status.fault, status.stop, status.left, status.right, status.pc_age_ms)

    def mark_round_start(self) -> tuple[bool, str]:
        """Adds a timestamped event. It does not arm, enable or move anything."""
        recorder = self.recorder
        if recorder is None:
            return False, "not recording: start Record Round first"
        recorder.mark_round_start(time.monotonic())
        stamp = datetime.now().strftime("%H:%M:%S")
        text = f"ROUND START marked at {st.format_elapsed(recorder.elapsed_s())}"
        self.events.append((stamp, "round_start", text))
        self.event_added.emit(stamp, "round_start", text)
        return True, text

    def stop_recording(self, wait: bool = False) -> None:
        recorder = self.recorder
        if recorder is None:
            return
        self.capture.set_recorder(None)
        self.recorder = None
        self._stopping = True
        self.add_event("recording_stopping", "Stopping recording, saving the round")

        def finish() -> None:
            try:
                summary = recorder.stop()
            except Exception as error:
                log.exception("stopping the recorder failed")
                summary = RoundSummary(recorder.folder, recorder.elapsed_s(), recorder.frames_received, recorder.frames_written,
                                       recorder.frames_dropped, recorder.errors + [f"stop failed: {error}"], recorder.video_file,
                                       False, "stop failed")
            self.last_summary = summary
            self._stopping = False
            self.recording_finished.emit(summary)

        if wait:
            finish()
        else:
            threading.Thread(target=finish, name="recorder-stop", daemon=True).start()

    def _on_recording_finished(self, summary: RoundSummary) -> None:
        self.add_event("recording_saved", summary.headline())

    # ---- snapshot ----

    def snapshot(self) -> st.Snapshot:
        now = time.monotonic()
        authority = self.monitor.authority(now)
        camera = st.camera_item(self.capture.state, self.capture.detail, self.capture.measured_fps, self.capture.last_frame_age())
        map_state = self.vision.map_state()
        sample = self.latest_sample
        camera_ok = self.capture.state in ("LIVE", "MOCK", "REPLAY")
        since_valid = None if self._last_valid_t is None else now - self._last_valid_t
        own = st.own_bot_item(camera_ok, map_state.level, None if sample is None else sample.pose.ok,
                              "" if sample is None else sample.pose.reason,
                              [] if sample is None else sample.pose.visible_ids,
                              [] if sample is None else sample.pose.used_ids, since_valid)
        track = None if sample is None else getattr(sample, "own_track", None)
        if camera_ok and track and track.get("state"):
            state = str(track["state"]).upper()
            age = track.get("age_s")
            ids = track.get("used_ids") or track.get("visible_ids") or []
            text = f"{state}" + (f" · age {1000 * age:.0f} ms" if isinstance(age, (int, float)) else "") + (
                " · markers " + " ".join(str(i) for i in ids) if ids else " · no markers")
            if track.get("flipped"):
                text += " · FLIPPED"
            level = {"MEASURED": "ok", "PREDICTED": "warn", "STALE": "bad", "LOST": "bad"}.get(state, "warn")
            if state == "STALE":
                text += " (last pose NOT current)"
            own = st.Item(text, level, str(track.get("reason") or ""))
        recorder = self.recorder
        stats = recorder.stats() if recorder is not None else None
        recording = st.recording_item(recorder is not None, stats["elapsed_s"] if stats else 0.0,
                                      stats["frames_dropped"] if stats else 0, len(stats["errors"]) if stats else 0,
                                      str(self.last_summary.folder) if self.last_summary else None)
        if self._stopping:
            recording = st.Item("Saving the round…", "warn")
        policy = st.policy_item(self.policy["mode"], self.policy["input"], self.policy["checkpoint"], self.policy["reason"])
        map_shown = st.map_item(map_state.level, map_state.text, map_state.detail)
        if self.world_coordinates_trusted is False and map_shown.level in ("ok", "warn"):
            report = self.restore_report
            map_shown = st.Item("Map loaded, NOT confirmed this run", "bad" if report.trust is sr.TrustLevel.NEEDS_RECALIBRATION else "warn",
                                f"{report.headline}. {report.next_action}".strip())
        return st.Snapshot(
            control=st.control_display(authority),
            robot_link=st.robot_link_item(authority, self.link_label, self.link_error),
            camera=camera, map=map_shown, own_bot=own,
            opponent=st.opponent_item(False), policy=policy,
            imu=(st.Item(f"IMU NOT VALID: {self.monitor.imu_problem}", "bad", "The controller reports this and sends no values")
                 if self.monitor.imu_problem and self.monitor.imu_problem_at is not None and now - self.monitor.imu_problem_at < 1.5
                 else st.imu_item(self.monitor.imu_present(now), self.monitor.imu_rate_hz(now), self.link_label is not None)),
            recording=recording, camera_map_caption=st.camera_map_caption(self.capture.state, map_state.text),
            recording_active=recorder is not None, recording_elapsed_s=stats["elapsed_s"] if stats else 0.0,
            mock=bool(self.options.mock))

    def _tick(self) -> None:
        now = time.monotonic()
        self._maybe_restore()
        for kind, message, data in self.monitor.poll(now):
            self.add_event(kind, message, **data)
        self.snapshot_changed.emit(self.snapshot())
