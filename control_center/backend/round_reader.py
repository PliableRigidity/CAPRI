"""Read a recorded round folder (see ``docs/round-recording-format.md``) for review.

Every stream except ``frames.csv`` is optional. A video-only, uncalibrated round loads with the
other streams empty; nothing here raises because a stream is missing.
"""

from __future__ import annotations

import bisect
import csv
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from control_center.backend.round_recorder import POSE_COLUMNS, pose_row

OFFLINE_POSE_FILE = "derived/pose_offline.csv"
OFFLINE_POSE_META = "derived/pose_offline.json"


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue                      # a line cut off by a crash
        if isinstance(record, dict) and "t_s" in record:
            out.append(record)
    return out


def _number(text: str) -> float:
    try:
        return float(text)
    except (TypeError, ValueError):
        return float("nan")


@dataclass
class PoseTrack:
    """Own-bot pose over time. NaN where the pose was unknown: those are gaps, never bridged."""

    source: str                           # "live" (estimated during the round) or "offline" (reconstructed)
    t: np.ndarray
    x: np.ndarray
    y: np.ndarray
    heading_deg: np.ndarray
    valid: np.ndarray
    tag_switch: np.ndarray
    reasons: list[str] = field(default_factory=list)
    used_ids: list[str] = field(default_factory=list)

    @classmethod
    def from_rows(cls, rows: list[dict[str, str]], source: str) -> "PoseTrack":
        valid = np.array([row.get("valid") == "1" for row in rows], bool)
        return cls(source, np.array([_number(row.get("t_s")) for row in rows]),
                   np.array([_number(row.get("x_m")) for row in rows]), np.array([_number(row.get("y_m")) for row in rows]),
                   np.array([_number(row.get("heading_deg")) for row in rows]), valid,
                   np.array([row.get("tag_switch") == "1" for row in rows], bool),
                   [row.get("reason", "") for row in rows], [row.get("used_ids", "") for row in rows])

    def __len__(self) -> int:
        return len(self.t)

    def segments(self, t_end: float | None = None) -> list[np.ndarray]:
        """Runs of consecutive valid poses as (n, 2) arrays. A gap or a tag switch starts a new run."""
        runs: list[np.ndarray] = []
        current: list[tuple[float, float]] = []
        for i in range(len(self.t)):
            if t_end is not None and self.t[i] > t_end:
                break
            if not self.valid[i] or self.tag_switch[i]:
                if len(current) > 0:
                    runs.append(np.array(current))
                current = []
                if not self.valid[i]:
                    continue
            current.append((self.x[i], self.y[i]))
        if current:
            runs.append(np.array(current))
        return runs

    def at(self, t: float, max_age_s: float = 0.25) -> dict[str, Any] | None:
        """The pose row at or just before ``t`` (None if there is none within ``max_age_s``)."""
        i = int(np.searchsorted(self.t, t, side="right")) - 1
        if i < 0 or t - self.t[i] > max_age_s:
            return None
        return {"t_s": float(self.t[i]), "valid": bool(self.valid[i]), "x_m": float(self.x[i]), "y_m": float(self.y[i]),
                "heading_deg": float(self.heading_deg[i]), "reason": self.reasons[i], "used_ids": self.used_ids[i],
                "age_s": float(t - self.t[i]), "source": self.source}

    def gaps(self, min_s: float = 0.3) -> list[tuple[float, float]]:
        """(start, end) of every interval of at least ``min_s`` with no valid pose."""
        out, start = [], None
        for i in range(len(self.t)):
            if not self.valid[i] and start is None:
                start = self.t[i]
            elif self.valid[i] and start is not None:
                if self.t[i] - start >= min_s:
                    out.append((float(start), float(self.t[i])))
                start = None
        if start is not None and len(self.t) and self.t[-1] - start >= min_s:
            out.append((float(start), float(self.t[-1])))
        return out


class RoundData:
    """Everything recorded for one round, indexed by ``t_s``."""

    def __init__(self, folder: Path | str) -> None:
        self.folder = Path(folder)
        manifest_path = self.folder / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"{self.folder} is not a round folder (no manifest.json)")
        self.manifest: dict[str, Any] = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.warnings: list[str] = []
        if not self.manifest.get("complete", False):
            self.warnings.append("recording was not closed cleanly (manifest incomplete); showing what was written")

        frames = _read_csv(self.folder / "frames.csv")
        self.frame_index = np.array([int(row["frame_index"]) for row in frames], int)
        self.frame_t = np.array([_number(row["t_s"]) for row in frames])
        self.frame_written = np.array([row["written"] == "1" for row in frames], bool)
        # Position of each camera frame inside the video file (-1 = not in the file).
        self.frame_video_pos = np.where(self.frame_written, np.cumsum(self.frame_written) - 1, -1)

        imu = _read_csv(self.folder / "imu.csv")
        self.imu_t = np.array([_number(row["t_s"]) for row in imu])
        self.imu_device_ms = np.array([_number(row["device_ms"]) for row in imu])
        self.imu = {name: np.array([_number(row[name]) for row in imu])
                    for name in ("ax_m_s2", "ay_m_s2", "az_m_s2", "gx_deg_s", "gy_deg_s", "gz_deg_s")}

        self.pose_live = PoseTrack.from_rows(_read_csv(self.folder / "pose.csv"), "live")
        self.pose_offline = PoseTrack.from_rows(_read_csv(self.folder / OFFLINE_POSE_FILE), "offline")
        offline_meta = self.folder / OFFLINE_POSE_META
        self.pose_offline_meta = json.loads(offline_meta.read_text(encoding="utf-8")) if offline_meta.is_file() else None
        self.opponent = _read_csv(self.folder / "opponent.csv")

        control = _read_jsonl(self.folder / "control.jsonl")
        self.controller_status = [r for r in control if r.get("kind") == "controller_status"]
        self.pc_sent = [r for r in control if r.get("kind") == "pc_sent"]
        self.link_changes = [r for r in control if r.get("kind") == "link"]
        self._status_t = [r["t_s"] for r in self.controller_status]
        self._pc_sent_t = [r["t_s"] for r in self.pc_sent]
        self.policy = _read_jsonl(self.folder / "policy.jsonl")
        self._policy_t = [r["t_s"] for r in self.policy]
        self.events = _read_jsonl(self.folder / "events.jsonl")

        video = (self.manifest.get("video") or {}).get("file")
        candidates = [video] if video else ["video.avi", "video.mp4"]
        self.video_path = next((self.folder / name for name in candidates if name and (self.folder / name).is_file()), None)
        if self.video_path is None:
            self.warnings.append("no video file in this round")
        self._capture: cv2.VideoCapture | None = None
        self._next_pos = -1

    # ---- timeline ----

    @property
    def duration_s(self) -> float:
        ends = [float(self.frame_t[-1])] if len(self.frame_t) else []
        ends += [float(self.manifest.get("duration_s") or 0.0)]
        ends += [float(e["t_s"]) for e in self.events]
        for times in (self.imu_t, self.pose_live.t, self.pose_offline.t):
            if len(times):
                ends.append(float(np.nanmax(times)))
        ends += self._status_t[-1:]
        return max(ends) if ends else 0.0

    @property
    def calibrated(self) -> bool:
        return bool((self.manifest.get("calibration") or {}).get("points_px"))

    def stream_present(self, name: str) -> bool:
        return bool(((self.manifest.get("streams") or {}).get(name) or {}).get("present"))

    def frame_at(self, t: float) -> int | None:
        """Row (in frames.csv order) of the newest WRITTEN frame at or before ``t``; the first one if earlier."""
        written = np.flatnonzero(self.frame_written)
        if len(written) == 0:
            return None
        i = int(np.searchsorted(self.frame_t[written], t, side="right")) - 1
        return int(written[max(i, 0)])

    @staticmethod
    def _latest(times: list[float], records: list[dict[str, Any]], t: float) -> tuple[dict[str, Any] | None, float | None]:
        i = bisect.bisect_right(times, t) - 1
        if i < 0:
            return None, None
        return records[i], t - times[i]

    def control_at(self, t: float, stale_after_s: float = 0.6) -> dict[str, Any]:
        """What the controller had last reported at ``t``. Stale or absent telemetry is said so, not guessed."""
        record, age = self._latest(self._status_t, self.controller_status, t)
        if record is None:
            return {"freshness": "UNKNOWN", "source": "UNKNOWN", "requested": "UNKNOWN", "armed": None, "age_s": None, "record": None}
        if age > stale_after_s:
            return {"freshness": "STALE", "source": "STALE", "requested": "STALE", "armed": None, "age_s": age, "record": record}
        return {"freshness": "FRESH", "source": record.get("active"), "requested": record.get("requested"),
                "armed": record.get("armed"), "age_s": age, "record": record}

    def pc_sent_at(self, t: float, max_age_s: float = 1.0) -> dict[str, Any] | None:
        record, age = self._latest(self._pc_sent_t, self.pc_sent, t)
        return record if record is not None and age <= max_age_s else None

    def policy_at(self, t: float, max_age_s: float | None = None) -> dict[str, Any] | None:
        """The newest policy record at or before ``t``. The mode holds until the next record (it is logged on change)."""
        record, age = self._latest(self._policy_t, self.policy, t)
        if record is None or (max_age_s is not None and age > max_age_s):
            return None
        return {**record, "age_s": age}

    def best_pose_track(self) -> PoseTrack | None:
        if len(self.pose_live):
            return self.pose_live
        return self.pose_offline if len(self.pose_offline) else None

    # ---- video ----

    def read_frame(self, row: int) -> np.ndarray | None:
        """The raw frame for a frames.csv row, or None (dropped frame, missing or unreadable video)."""
        if self.video_path is None or row is None or not 0 <= row < len(self.frame_video_pos):
            return None
        position = int(self.frame_video_pos[row])
        if position < 0:
            return None
        if self._capture is None:
            self._capture = cv2.VideoCapture(str(self.video_path))
            self._next_pos = 0
        if position != self._next_pos:
            self._capture.set(cv2.CAP_PROP_POS_FRAMES, position)
        ok, frame = self._capture.read()
        self._next_pos = position + 1 if ok else -1
        return frame if ok else None

    def close(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None


def reconstruct_pose_offline(round_folder: Path | str, overlay_config=None, clicks=None, progress=None) -> dict[str, Any]:
    """Re-run marker detection over the recorded video and save the result as an OFFLINE track.

    Uses the calibration stored in the round's manifest unless ``clicks`` is given. The result is
    written to ``derived/pose_offline.csv`` and is always labelled as reconstructed after the round.
    """
    from control_center.backend.live_vision import LiveVision, clicks_from_snapshot
    from control_center.tools.overlay_core import load_config

    data = RoundData(round_folder)
    try:
        config = overlay_config or load_config()
        used = clicks or clicks_from_snapshot(data.manifest.get("calibration"))
        if used is None:
            return {"ok": False, "reason": "this round has no calibration snapshot; calibrate and pass the points to reconstruct"}
        vision = LiveVision(config, used, profile=str(used.meta.get("profile") or "snapshot"))
        target = data.folder / OFFLINE_POSE_FILE
        target.parent.mkdir(exist_ok=True)
        partial = target.with_name(target.name + ".partial")       # replaced only when the whole pass has finished
        rows = valid = 0
        with partial.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=POSE_COLUMNS, extrasaction="ignore")
            writer.writeheader()
            written = np.flatnonzero(data.frame_written)
            for n, row in enumerate(written):
                frame = data.read_frame(int(row))
                if frame is None:
                    continue
                t = float(data.frame_t[row])
                sample = vision.process(int(data.frame_index[row]), t, frame)
                writer.writerow({"t_s": f"{t:.4f}", "frame_index": int(data.frame_index[row]), **pose_row(sample.pose, sample.tag_switch)})
                rows += 1
                valid += int(sample.pose.ok)
                if progress is not None and n % 30 == 0:
                    progress(n, len(written))
        meta = {"source": "reconstructed OFFLINE from the recorded video, after the round",
                "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "calibration": "points given at reconstruction time" if clicks is not None else "the round manifest's calibration snapshot",
                "frames_processed": rows, "valid_poses": valid, "marker_dictionary": config.dictionary}
        partial.replace(target)
        (data.folder / OFFLINE_POSE_META).write_text(json.dumps(meta, indent=2), encoding="utf-8")
        return {"ok": True, **meta}
    finally:
        data.close()
