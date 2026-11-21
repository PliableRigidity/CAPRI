"""One capture session shared by preview, vision and recording.

Exactly one object owns the camera. Every consumer gets the same frames with the same index and
receipt timestamp: the recorder gets every frame, the vision thread and the display take the
newest one. Nothing else in the dashboard opens the camera.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np
from PySide6.QtCore import QThread, Signal

from control_center.tools.overlay_core import OverlayConfig

log = logging.getLogger(__name__)

RETRY_S = 3.0


# ---------------------------------------------------------------- sources


def open_camera(index: int, settings: dict[str, Any]):
    """The overlay tool's camera opener (same backend / size / MJPG order as the first dashboard)."""
    from control_center.tools.arena_overlay import CameraSource

    source = CameraSource(index, settings)
    source.index = index
    source.settings = dict(settings)
    return source


def camera_settings_from_config(config: dict[str, Any]) -> dict[str, Any]:
    return {"backend": str(config.get("camera_backend", "auto")).lower(),
            "width": int(config.get("camera_width", 1280)), "height": int(config.get("camera_height", 720)),
            "fps": float(config.get("camera_fps", 30)), "fourcc": str(config.get("camera_fourcc", "")).strip()}


class VideoFileSource:
    """REPLAY of a video file at its own frame rate, looping. Not a live camera."""

    kind = "replay"

    def __init__(self, path: Path | str, loop: bool = True) -> None:
        self.path = Path(path)
        self._cap = cv2.VideoCapture(str(self.path))
        if not self._cap.isOpened():
            raise RuntimeError(f"could not open video {self.path}")
        self.fps = float(self._cap.get(cv2.CAP_PROP_FPS) or 30.0) or 30.0
        self.description = f"REPLAY of {self.path.name} (not live)"
        self._loop = loop
        self._next = time.monotonic()

    def read(self) -> tuple[bool, np.ndarray | None, float]:
        delay = self._next - time.monotonic()
        if delay > 0:
            time.sleep(delay)
        self._next = max(self._next + 1.0 / self.fps, time.monotonic())
        ok, frame = self._cap.read()
        if (not ok or frame is None) and self._loop:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self._cap.read()
        return bool(ok and frame is not None), frame if ok else None, time.monotonic()

    def close(self) -> None:
        self._cap.release()


class MockArenaSource:
    """MOCK camera: a rendered arena with a marked bot driving a circle. Every frame says SYNTHETIC.

    The bot disappears for ``hide_s`` seconds in every ``period_s`` (a tracking dropout), and for a
    short stretch only one marker is shown (a tag switch that is not real movement).
    """

    kind = "mock"

    def __init__(self, config: OverlayConfig, fps: float = 15.0, period_s: float = 16.0,
                 hide: tuple[float, float] = (6.0, 8.0), single_tag: tuple[float, float] = (11.0, 13.0),
                 image_size: tuple[int, int] = (960, 540), realtime: bool = True) -> None:
        from control_center.tools import overlay_fixture as fixture

        self._fixture = fixture
        self.config = config
        self.fps = fps
        self.period_s, self.hide, self.single_tag = period_s, hide, single_tag
        scale = image_size[0] / 1280.0
        self.camera = fixture.make_camera((0.40, -0.55, 1.30), (0.60, 0.42, 0.0), focal_px=930.0 * scale, image_size=image_size)
        self.description = "MOCK synthetic arena (not a camera)"
        self._realtime = realtime
        self._start = time.monotonic()
        self._count = 0
        self._top_ids = sorted(i for i, spec in config.markers.items() if spec.face == "top")

    def clicks(self):
        return self._fixture.fixture_clicks(self.config, self.camera)

    def truth(self, elapsed: float) -> tuple[float, float, float] | None:
        """Where the rendered bot is (x, y, heading), or None while it is hidden."""
        phase = elapsed % self.period_s
        if self.hide[0] <= phase < self.hide[1]:
            return None
        angle = 2.0 * math.pi * phase / self.period_s
        return 0.575 + 0.25 * math.cos(angle), 0.575 + 0.25 * math.sin(angle), angle + math.pi / 2.0

    def frame_for(self, step: int) -> np.ndarray:
        """Rendered on demand (about 30 ms at 960x540); markers are too small to detect reliably below that."""
        elapsed = (step % max(1, int(round(self.period_s * self.fps)))) / self.fps
        truth = self.truth(elapsed)
        scene = self._fixture.Scene()
        if truth is not None:
            phase = elapsed % self.period_s
            ids = self._top_ids[:1] if self.single_tag[0] <= phase < self.single_tag[1] else None
            scene.bots.append(self._fixture.BotInstance(truth[0], truth[1], truth[2], ids=ids))
        return self._fixture.render_frame(self.config, self.camera, scene)

    def read(self) -> tuple[bool, np.ndarray | None, float]:
        if self._realtime:
            delay = self._start + self._count / self.fps - time.monotonic()
            if delay > 0:
                time.sleep(delay)
        frame = self.frame_for(self._count)
        self._count += 1
        return True, frame, time.monotonic()

    def close(self) -> None:
        pass


# ---------------------------------------------------------------- the one capture thread


class CaptureSession(QThread):
    """Owns the camera. ``state_changed(state, detail)``: OPENING, LIVE, MOCK, REPLAY, LOST, UNAVAILABLE."""

    state_changed = Signal(str, str)

    def __init__(self, opener: Callable[[], Any], parent=None) -> None:
        super().__init__(parent)
        self._opener = opener
        self._running = True
        self._reopen = False
        self._lock = threading.Lock()
        self._new_frame = threading.Condition(self._lock)
        self._latest: tuple[int, float, np.ndarray | None] = (-1, 0.0, None)
        self._recorder = None
        self.state = "OPENING"
        self.detail = ""
        self.source_kind: str | None = None
        self.source_info: dict[str, Any] = {}
        self.nominal_fps = 30.0
        self.measured_fps = 0.0
        self.frame_size: tuple[int, int] | None = None
        self.frames = 0

    # ---- consumers ----

    def latest(self) -> tuple[int, float, np.ndarray | None]:
        with self._lock:
            return self._latest

    def wait_for_frame(self, after_index: int, timeout: float = 0.5) -> tuple[int, float, np.ndarray | None]:
        with self._new_frame:
            if self._latest[0] <= after_index:
                self._new_frame.wait(timeout)
            return self._latest

    def set_recorder(self, recorder) -> None:
        with self._lock:
            self._recorder = recorder

    def last_frame_age(self) -> float | None:
        with self._lock:
            index, t, _ = self._latest
        return None if index < 0 else time.monotonic() - t

    def request_reopen(self, opener: Callable[[], Any]) -> None:
        self._opener = opener
        self._reopen = True

    def stop(self) -> None:
        self._running = False
        with self._new_frame:
            self._new_frame.notify_all()

    # ---- thread ----

    def _set_state(self, state: str, detail: str) -> None:
        if (state, detail) != (self.state, self.detail):
            self.state, self.detail = state, detail
            self.state_changed.emit(state, detail)

    def run(self) -> None:
        source = None
        index = -1
        while self._running:
            if source is None:
                self._set_state("OPENING", "opening camera")
                try:
                    source = self._opener()
                except Exception as error:  # camera missing, busy, bad index ...
                    self._set_state("UNAVAILABLE", str(error))
                    self._sleep(RETRY_S)
                    continue
                self._reopen = False
                self.source_kind = getattr(source, "kind", "camera")
                self.nominal_fps = float(getattr(source, "fps", 30.0) or 30.0)
                self.source_info = {"kind": self.source_kind, "description": getattr(source, "description", ""),
                                    "index": getattr(source, "index", None), "requested": getattr(source, "settings", None)}
                window_start, window_frames = time.monotonic(), 0
            if self._reopen:
                source.close()
                source = None
                continue
            try:
                ok, frame, t = source.read()
            except Exception as error:
                ok, frame, t = False, None, time.monotonic()
                log.exception("camera read failed: %s", error)
            if frame is None:
                if not ok:
                    self._set_state("LOST", "camera stopped delivering frames")
                    source.close()
                    source = None
                    self._sleep(RETRY_S)
                continue
            index += 1
            self.frames = index + 1
            self.frame_size = (frame.shape[1], frame.shape[0])
            with self._new_frame:
                self._latest = (index, t, frame)
                recorder = self._recorder
                self._new_frame.notify_all()
            if recorder is not None:
                recorder.add_frame(index, t, frame)          # raw frame; consumers must not draw on it
            window_frames += 1
            if t - window_start >= 1.0:
                self.measured_fps = window_frames / (t - window_start)
                window_start, window_frames = t, 0
            label = {"camera": "LIVE", "mock": "MOCK", "replay": "REPLAY"}.get(self.source_kind, "LIVE")
            self._set_state(label, getattr(source, "description", ""))
        if source is not None:
            source.close()

    def _sleep(self, seconds: float) -> None:
        end = time.monotonic() + seconds
        while self._running and not self._reopen and time.monotonic() < end:
            time.sleep(0.05)


class VisionThread(QThread):
    """Runs ``LiveVision`` on the newest frame. Frames it cannot keep up with are skipped, never queued."""

    sample_ready = Signal(object)

    def __init__(self, capture: CaptureSession, vision, parent=None) -> None:
        super().__init__(parent)
        self.capture = capture
        self.vision = vision
        self._running = True
        self.listeners: list[Callable[[Any], None]] = []     # called in this thread, before the signal
        self.processed = 0
        self.last_ms = 0.0

    def stop(self) -> None:
        self._running = False

    def run(self) -> None:
        seen = -1
        while self._running:
            index, t, frame = self.capture.wait_for_frame(seen, 0.25)
            if frame is None or index <= seen:
                continue
            seen = index
            started = time.perf_counter()
            try:
                sample = self.vision.process(index, t, frame)
            except Exception:
                log.exception("vision failed on frame %d", index)
                continue
            self.last_ms = 1000.0 * (time.perf_counter() - started)
            self.processed += 1
            for listener in self.listeners:
                try:
                    listener(sample)
                except Exception:
                    log.exception("vision listener failed")
            self.sample_ready.emit(sample)
