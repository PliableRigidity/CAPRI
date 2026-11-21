"""One folder per round: the raw camera video plus timestamped telemetry and events.

The format is described in ``docs/round-recording-format.md``. Summary:

    <round folder>/
        manifest.json     setup, which streams were present, counts, errors
        video.avi         the camera frames exactly as captured (no overlays)
        frames.csv        frame_index, t_s, t_unix, written
        imu.csv           raw accel + gyro (only if the controller sent any)
        pose.csv          camera-derived own-bot pose, one row per processed frame, gaps included
        opponent.csv      opponent tracks (only if a tracker produced any)
        control.jsonl     controller status, PC-sent commands, link changes
        policy.jsonl      policy mode / proposed action (only if a policy ran)
        events.jsonl      operator events and round markers

``t_s`` is seconds since recording started on the PC's monotonic clock, for every stream.
Recording needs only frames: every other stream is optional and its absence is recorded.

All disk writes happen on one background thread. Callers never wait for the disk; if the video
queue is full the frame is dropped, counted, and marked ``written=0`` in ``frames.csv``.
"""

from __future__ import annotations

import csv
import json
import queue
import subprocess
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

FORMAT_VERSION = 1
FORMAT_DOC = "docs/round-recording-format.md"

FRAME_COLUMNS = ["frame_index", "t_s", "t_unix", "written"]
IMU_COLUMNS = ["t_s", "t_unix", "device_ms", "ax_m_s2", "ay_m_s2", "az_m_s2", "gx_deg_s", "gy_deg_s", "gz_deg_s", "seq", "device_us"]
POSE_COLUMNS = ["t_s", "frame_index", "valid", "x_m", "y_m", "heading_deg", "orientation", "position_quality",
                "used_ids", "visible_ids", "tag_switch", "reason"]
OPPONENT_COLUMNS = ["t_s", "frame_index", "track_id", "state", "x_m", "y_m", "identity_uncertain"]

CSV_STREAMS = {"frames": ("frames.csv", FRAME_COLUMNS), "imu": ("imu.csv", IMU_COLUMNS),
               "pose": ("pose.csv", POSE_COLUMNS), "opponent": ("opponent.csv", OPPONENT_COLUMNS)}
JSON_STREAMS = {"control": "control.jsonl", "policy": "policy.jsonl", "events": "events.jsonl"}
CODECS = {"MJPG": ("video.avi", "MJPG"), "mp4v": ("video.mp4", "mp4v")}

TIMESTAMP_NOTES = {
    "t_s": "seconds since recording start, PC monotonic clock (time.monotonic); shared by every stream",
    "t_unix": "PC wall clock at the same instant (time.time); for matching against other logs only",
    "frames": "t_s is when the frame was RECEIVED from the camera driver (cap.read() returned), not exposure time; "
              "camera and USB latency are not measured or removed",
    "imu": "t_s is PC RECEIPT time of the line. device_ms is the controller's own millisecond counter. "
           "The two clocks are NOT synchronised; no offset or drift is estimated",
    "pose": "t_s is the receipt time of the frame the pose was computed from (frame_index), not when processing finished",
    "control": "t_s is PC receipt time of the controller line; device_ms inside is the controller's counter",
}


def software_version(root: Path) -> dict[str, Any]:
    """Git commit and whether the tree had local changes. Never raises."""
    def git(*args: str) -> str | None:
        try:
            done = subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, timeout=5)
            return done.stdout.strip() if done.returncode == 0 else None
        except (OSError, subprocess.SubprocessError):
            return None
    commit = git("rev-parse", "--short", "HEAD")
    status = git("status", "--porcelain", "--untracked-files=no")
    return {"git_commit": commit, "tracked_files_modified": None if status is None else bool(status),
            "note": "most of this repository's newer code is untracked, so the commit alone does not pin it"}


def pose_row(sample_pose: Any, tag_switch: bool) -> dict[str, Any]:
    """A ``BotPose`` (overlay_core) as a pose.csv row. An invalid pose is a gap: no numbers."""
    ok = bool(getattr(sample_pose, "ok", False))
    return {
        "valid": 1 if ok else 0,
        "x_m": f"{sample_pose.x_m:.4f}" if ok else "",
        "y_m": f"{sample_pose.y_m:.4f}" if ok else "",
        "heading_deg": f"{sample_pose.heading_deg:.2f}" if ok else "",
        "orientation": sample_pose.orientation,
        "position_quality": sample_pose.position_quality,
        "used_ids": " ".join(str(i) for i in sample_pose.used_ids),
        "visible_ids": " ".join(str(i) for i in sample_pose.visible_ids),
        "tag_switch": 1 if tag_switch else 0,
        "reason": "" if ok else sample_pose.reason,
    }


@dataclass
class RoundSummary:
    folder: Path
    duration_s: float
    frames_received: int
    frames_written: int
    frames_dropped: int
    write_errors: list[str]
    video_file: str | None
    video_verified: bool
    video_verify_detail: str
    streams: dict[str, dict[str, Any]] = field(default_factory=dict)

    def headline(self) -> str:
        problems = []
        if self.frames_dropped:
            problems.append(f"{self.frames_dropped} frame(s) dropped")
        if self.write_errors:
            problems.append(f"{len(self.write_errors)} write error(s)")
        if not self.video_verified:
            problems.append(f"video check failed: {self.video_verify_detail}")
        tail = "; ".join(problems) if problems else "no dropped frames, no write errors, video opens"
        return f"Saved {self.duration_s:.1f} s, {self.frames_written} frames to {self.folder} ({tail})"


def verify_video(path: Path, expected_frames: int) -> tuple[bool, str]:
    """Reopen the finished video and read its first and last frame."""
    if expected_frames == 0:
        return False, "no frames were written"
    if not path.is_file() or path.stat().st_size == 0:
        return False, "video file missing or empty"
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            return False, "video file does not open"
        count = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
        ok, first = capture.read()
        if not ok or first is None:
            return False, "first frame does not decode"
        if count != expected_frames:
            return False, f"container reports {count} frames, {expected_frames} were written"
        capture.set(cv2.CAP_PROP_POS_FRAMES, max(0, count - 1))
        ok, last = capture.read()
        if not ok or last is None:
            return False, "last frame does not decode"
        return True, f"{count} frames, {first.shape[1]}x{first.shape[0]}"
    finally:
        capture.release()


def new_round_folder(root: Path, label: str = "", now: datetime | None = None) -> Path:
    stamp = (now or datetime.now()).strftime("%Y-%m-%d_%H-%M-%S")
    safe = "".join(ch if ch.isalnum() or ch in "-_" else "-" for ch in label.strip())[:40].strip("-")
    base = f"{stamp}_{safe}" if safe else stamp
    folder = root / base
    suffix = 2
    while folder.exists():
        folder = root / f"{base}-{suffix}"
        suffix += 1
    folder.mkdir(parents=True)
    return folder


class RoundRecorder:
    """Records one round. Create, ``start()``, feed it, ``stop()``."""

    def __init__(self, folder: Path, manifest: dict[str, Any], fps: float = 30.0, codec: str = "MJPG",
                 max_queued_frames: int = 90, clock=time.monotonic, wall=time.time) -> None:
        if codec not in CODECS:
            raise ValueError(f"codec must be one of {sorted(CODECS)}")
        self.folder = Path(folder)
        self.fps = float(fps) if fps and fps > 0 else 30.0
        self.codec = codec
        self.max_queued_frames = max_queued_frames
        self._clock, self._wall = clock, wall
        self._manifest = dict(manifest)
        self._queue: queue.Queue = queue.Queue()
        self._thread = threading.Thread(target=self._run, name="round-recorder", daemon=True)
        self._lock = threading.Lock()
        self._pending_frames = 0
        self._started = False
        self._stopped = False
        self.t0_mono = 0.0
        self.t0_unix = 0.0
        # Counters (written by the writer thread, read by anyone).
        self.frames_received = 0
        self.frames_written = 0
        self.frames_dropped = 0
        self.frames_refused = 0             # reached the writer but could not be written (wrong size, no video file)
        self.rows: dict[str, int] = {name: 0 for name in list(CSV_STREAMS) + list(JSON_STREAMS)}
        self.errors: list[str] = []
        self.video_file: str | None = None
        self.frame_size: tuple[int, int] | None = None
        self._writer: cv2.VideoWriter | None = None
        self._video_failed = False
        self._csv: dict[str, tuple[Any, Any]] = {}
        self._json: dict[str, Any] = {}

    # ---- lifecycle ----

    def start(self) -> None:
        self.t0_mono, self.t0_unix = self._clock(), self._wall()
        self._started = True
        self._write_manifest(final=False)
        self._thread.start()
        self.log_event(self.t0_mono, "recording_start", "Recording started")

    @property
    def active(self) -> bool:
        return self._started and not self._stopped

    def elapsed_s(self) -> float:
        return max(0.0, self._clock() - self.t0_mono) if self._started else 0.0

    def t_s(self, t_mono: float) -> float:
        return t_mono - self.t0_mono

    def _unix(self, t_mono: float) -> float:
        return self.t0_unix + (t_mono - self.t0_mono)

    # ---- inputs (never block on the disk) ----

    def add_frame(self, frame_index: int, t_mono: float, frame: np.ndarray) -> bool:
        """Queue one raw frame. Returns False if it had to be dropped."""
        if not self.active:
            return False
        with self._lock:
            dropped = self._pending_frames >= self.max_queued_frames
            if not dropped:
                self._pending_frames += 1
        self._queue.put(("frame", frame_index, t_mono, None if dropped else frame))
        return not dropped

    def log_imu(self, t_mono: float, device_ms: int, accel_m_s2, gyro_deg_s, sequence: int | None = None, device_us: int | None = None) -> None:
        self._row("imu", {"t_s": f"{self.t_s(t_mono):.4f}", "t_unix": f"{self._unix(t_mono):.4f}", "device_ms": device_ms,
                          "seq": "" if sequence is None else sequence, "device_us": "" if device_us is None else device_us,
                          "ax_m_s2": accel_m_s2[0], "ay_m_s2": accel_m_s2[1], "az_m_s2": accel_m_s2[2],
                          "gx_deg_s": gyro_deg_s[0], "gy_deg_s": gyro_deg_s[1], "gz_deg_s": gyro_deg_s[2]})

    def log_pose(self, t_mono: float, frame_index: int, row: dict[str, Any]) -> None:
        self._row("pose", {"t_s": f"{self.t_s(t_mono):.4f}", "frame_index": frame_index, **row})

    def log_opponent(self, t_mono: float, frame_index: int, row: dict[str, Any]) -> None:
        self._row("opponent", {"t_s": f"{self.t_s(t_mono):.4f}", "frame_index": frame_index, **row})

    def log_control(self, t_mono: float, kind: str, data: dict[str, Any]) -> None:
        self._json_row("control", t_mono, {"kind": kind, **data})

    def log_policy(self, t_mono: float, data: dict[str, Any]) -> None:
        self._json_row("policy", t_mono, data)

    def log_event(self, t_mono: float, kind: str, text: str, **data: Any) -> None:
        self._json_row("events", t_mono, {"kind": kind, "text": text, **data})

    def mark_round_start(self, t_mono: float | None = None) -> None:
        self.log_event(self._clock() if t_mono is None else t_mono, "round_start", "Round start marked by operator")

    def _row(self, stream: str, row: dict[str, Any]) -> None:
        if self.active:
            self._queue.put(("csv", stream, row))

    def _json_row(self, stream: str, t_mono: float, data: dict[str, Any]) -> None:
        if self.active:
            self._queue.put(("json", stream, {"t_s": round(self.t_s(t_mono), 4), "t_unix": round(self._unix(t_mono), 4), **data}))

    def stats(self) -> dict[str, Any]:
        return {"elapsed_s": self.elapsed_s(), "frames_received": self.frames_received,
                "frames_written": self.frames_written, "frames_dropped": self.frames_dropped,
                "queued_frames": self._pending_frames, "errors": list(self.errors), "folder": str(self.folder)}

    def stop(self) -> RoundSummary:
        if not self._started:
            raise RuntimeError("recorder was never started")
        if not self._stopped:
            end = self._clock()
            self.log_event(end, "recording_stop", "Recording stopped")
            self._stopped = True
            self._queue.put(("stop", end))
            self._thread.join(timeout=60)
            if self._thread.is_alive():
                self.errors.append("writer thread did not finish within 60 s")
            self._duration = end - self.t0_mono
        video_path = self.folder / self.video_file if self.video_file else None
        verified, detail = (verify_video(video_path, self.frames_written) if video_path
                            else (False, "no video file was created"))
        self._verified, self._verify_detail = verified, detail
        self._write_manifest(final=True)
        return RoundSummary(self.folder, self._duration, self.frames_received, self.frames_written,
                            self.frames_dropped, list(self.errors), self.video_file, verified, detail, self._streams())

    # ---- writer thread ----

    def _error(self, text: str) -> None:
        if len(self.errors) < 20:
            self.errors.append(text)

    def _run(self) -> None:
        last_flush = self._clock()
        while True:
            try:
                item = self._queue.get(timeout=0.5)
            except queue.Empty:
                item = None
            try:
                if item is not None:
                    if item[0] == "stop":
                        break
                    if item[0] == "frame":
                        self._write_frame(item[1], item[2], item[3])
                    elif item[0] == "csv":
                        self._write_csv(item[1], item[2])
                    else:
                        self._write_json(item[1], item[2])
                if self._clock() - last_flush >= 1.0:
                    self._flush()
                    last_flush = self._clock()
            except Exception as error:  # a write failure must never kill the recording
                self._error(f"{type(error).__name__}: {error}")
        self._close()

    def _open_video(self, frame: np.ndarray) -> None:
        size = (frame.shape[1], frame.shape[0])
        for codec in dict.fromkeys([self.codec, "MJPG"]):
            name, fourcc = CODECS[codec]
            writer = cv2.VideoWriter(str(self.folder / name), cv2.VideoWriter_fourcc(*fourcc), self.fps, size)
            if writer.isOpened():
                self._writer, self.video_file, self.frame_size, self.codec = writer, name, size, codec
                return
            writer.release()
            (self.folder / name).unlink(missing_ok=True)
            self._error(f"video writer did not open with codec {codec}")
        self._video_failed = True

    def _write_frame(self, frame_index: int, t_mono: float, frame: np.ndarray | None) -> None:
        self.frames_received += 1
        written = 0
        if frame is None:
            self.frames_dropped += 1
        else:
            with self._lock:
                self._pending_frames -= 1
            if self._writer is None and not self._video_failed:
                self._open_video(frame)
            if self._writer is not None:
                if (frame.shape[1], frame.shape[0]) != self.frame_size:
                    self._error(f"frame {frame_index} is {frame.shape[1]}x{frame.shape[0]}, video is "
                                f"{self.frame_size[0]}x{self.frame_size[1]}; not written")
                else:
                    self._writer.write(frame)
                    self.frames_written += 1
                    written = 1
            if not written:
                self.frames_refused += 1
        self._write_csv("frames", {"frame_index": frame_index, "t_s": f"{self.t_s(t_mono):.4f}",
                                   "t_unix": f"{self._unix(t_mono):.4f}", "written": written})

    def _write_csv(self, stream: str, row: dict[str, Any]) -> None:
        if stream not in self._csv:
            name, columns = CSV_STREAMS[stream]
            handle = (self.folder / name).open("w", newline="", encoding="utf-8")
            writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
            writer.writeheader()
            self._csv[stream] = (handle, writer)
        self._csv[stream][1].writerow(row)
        self.rows[stream] += 1

    def _write_json(self, stream: str, data: dict[str, Any]) -> None:
        if stream not in self._json:
            self._json[stream] = (self.folder / JSON_STREAMS[stream]).open("w", encoding="utf-8")
        self._json[stream].write(json.dumps(data, separators=(",", ":")) + "\n")
        self.rows[stream] += 1

    def _flush(self) -> None:
        for handle, _ in self._csv.values():
            handle.flush()
        for handle in self._json.values():
            handle.flush()

    def _close(self) -> None:
        try:
            if self._writer is not None:
                self._writer.release()
        except Exception as error:
            self._error(f"closing video: {error}")
        for handle in [pair[0] for pair in self._csv.values()] + list(self._json.values()):
            try:
                handle.close()
            except OSError as error:
                self._error(f"closing {handle.name}: {error}")

    # ---- manifest ----

    def _streams(self) -> dict[str, dict[str, Any]]:
        out: dict[str, dict[str, Any]] = {
            "video": {"present": self.frames_written > 0, "file": self.video_file, "frames": self.frames_written}}
        for stream, (name, _) in CSV_STREAMS.items():
            out[stream] = {"present": self.rows[stream] > 0, "file": name if self.rows[stream] else None, "rows": self.rows[stream]}
        for stream, name in JSON_STREAMS.items():
            out[stream] = {"present": self.rows[stream] > 0, "file": name if self.rows[stream] else None, "rows": self.rows[stream]}
        return out

    def _write_manifest(self, final: bool) -> None:
        manifest = dict(self._manifest)
        manifest.update({
            "format_version": FORMAT_VERSION, "format_doc": FORMAT_DOC,
            "started_local": datetime.fromtimestamp(self.t0_unix).isoformat(timespec="seconds"),
            "t0_unix": self.t0_unix, "timestamps": TIMESTAMP_NOTES, "complete": final,
            "columns": {name: columns for name, (_, columns) in CSV_STREAMS.items()},
        })
        if final:
            manifest.update({
                "duration_s": round(self._duration, 3),
                "video": {"file": self.video_file, "codec": self.codec if self.video_file else None,
                          "container_fps": self.fps,
                          "container_fps_note": "nominal playback rate only; real frame times are in frames.csv",
                          "frame_size": list(self.frame_size) if self.frame_size else None,
                          "overlays_baked_in": False,
                          "frames_received": self.frames_received, "frames_written": self.frames_written,
                          "frames_dropped": self.frames_dropped, "frames_refused": self.frames_refused,
                          "verified_after_stop": self._verified, "verify_detail": self._verify_detail},
                "streams": self._streams(), "write_errors": list(self.errors),
            })
        target = self.folder / "manifest.json"
        temporary = self.folder / "manifest.json.tmp"
        temporary.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
        temporary.replace(target)
