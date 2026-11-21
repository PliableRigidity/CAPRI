"""LIVE TRACK: camera + calibration + chassis tracker + click-selected opponent. Read-only viewer.

    python -m control_center.tools.live_track --camera N --config config\\home_test_overlay.yaml --profile home_test

No checkpoint, no telemetry, no connection to the robot: this program opens the camera YOU name,
shows what the two trackers make of each frame, and writes ``artifacts/live-track/<time>/trace.jsonl``
(per frame: chassis tracker report, opponent report, timings).

Window: LEFT-CLICK the opponent in the camera picture to select (or reselect) it.
Keys: c clear the opponent selection, q quit.

The per-frame loop is ``run_frames``; it takes any iterable of ``(t, frame)`` and scripted clicks,
which is how the offline tests drive it with SYNTHETIC frames. A real camera is opened only by
``run_camera``, only when this module is run with ``--camera N``.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable, TextIO

import cv2
import numpy as np

from control_center.tools.chassis_tracker import TrackerSettings, live_tracker_settings
from control_center.tools.opponent_select import OpponentSettings
from control_center.tools.arena_setup import with_arena_setup
from control_center.tools.overlay_core import ClickSet, OverlayConfig, load_clicks, load_config, profile_points_path
from control_center.tools.stream_adapter import CameraPipeline, CameraSample, fit_profile

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = REPO_ROOT / "artifacts" / "live-track"
WINDOW = "AutoBots LIVE TRACK (read-only, nothing is sent)"
PANEL_WIDTH = 620

_OWN_LEVEL = {"measured": "ok", "predicted": "note", "stale": "bad", "lost": "bad"}
_OPPONENT_LEVEL = {"tracked": "ok", "ambiguous": "bad", "uncertain": "bad", "lost": "bad", "not_selected": "note"}
_OWN_COLOUR = {"measured": (120, 220, 120), "predicted": (60, 200, 255), "stale": (80, 110, 255)}
_OPPONENT_COLOUR = {"tracked": (60, 60, 255), "ambiguous": (0, 165, 255), "uncertain": (0, 200, 255)}


# ----------------------------------------------------------------------------------------------
# Words and drawing shared with live_shadow and the shadow-stream camera mode. Display only.


def tracking_lines(sample: CameraSample) -> list[tuple[str, str]]:
    """Panel lines (text, level) for the opt-in trackers; empty when neither is in use."""
    lines: list[tuple[str, str]] = []
    own = sample.own_track
    if own is not None:
        state = own["state"]
        age = "" if own["age_s"] is None else f"  last detection {1000 * own['age_s']:.0f} ms ago"
        where = ""
        if own["x_m"] is not None:
            where = f"  ({own['x_m']:.3f}, {own['y_m']:.3f}) m  {own['heading_deg']:.0f} deg  {own['orientation']}"
            if state == "stale":
                where += "  [DISPLAY ONLY]"
        lines.append((f"OWN TRACK   {state.upper():<9} ids {own['used_ids']}{age}{where}", _OWN_LEVEL[state]))
        if own["source"] == "rear marker only":
            lines.append(("            rear marker only: LOWER CONFIDENCE (fixture-tested only)", "bad"))
        if state != "measured":
            lines.append((f"            {own['reason']}", _OWN_LEVEL[state]))
        if own["rejected"]:
            lines.append(("            rejected: " + "; ".join(f"{key}: {value}" for key, value in own["rejected"].items()), "note"))
    opponent = sample.opponent_track
    if opponent is not None:
        state = opponent["state"]
        where = "" if opponent["x_m"] is None else f"  ({opponent['x_m']:.3f}, {opponent['y_m']:.3f}) m  approximate, not height-corrected"
        if opponent.get("image_based"):
            where = ("" if opponent["x_m"] is None else f"  ({opponent['x_m']:.3f}, {opponent['y_m']:.3f}) m  IMAGE-BASED, approximate, not height-corrected")
            if opponent.get("confidence") is not None:
                where += f"  confidence {opponent['confidence']:.2f}"
        lines.append((f"OPPONENT    {state.upper().replace('_', ' '):<12}{where}", _OPPONENT_LEVEL[state]))
        if state != "tracked":
            lines.append((f"            {opponent['reason']}", _OPPONENT_LEVEL[state]))
        if opponent["selection_note"]:
            lines.append((f"            {opponent['selection_note']}", "bad" if "REFUSED" in opponent["selection_note"] else "note"))
    return lines


def draw_tracking(image: np.ndarray, pipeline: CameraPipeline, sample: CameraSample, scale: float = 1.0) -> None:
    """Own footprint (coloured by tracker state) and the selected opponent's box, drawn onto ``image``."""
    mapping = pipeline.mapping
    own = sample.own_track
    if own is not None and own["x_m"] is not None and mapping.valid:
        colour = _OWN_COLOUR.get(own["state"], (190, 190, 190))
        radius = pipeline.clicked.settings.own_radius_m if pipeline.clicked is not None else 0.09
        angles = np.linspace(0.0, 2.0 * math.pi, 40)
        ring = np.array([own["x_m"], own["y_m"]]) + radius * np.column_stack([np.cos(angles), np.sin(angles)])
        pixels = mapping.floor_to_pixel(ring)
        if np.isfinite(pixels).all():
            cv2.polylines(image, [np.round(pixels * scale).astype(np.int32)], True, colour, 1 if own["state"] == "stale" else 2, cv2.LINE_AA)
            anchor = np.round(pixels[0] * scale).astype(int)
            cv2.putText(image, "OWN " + own["state"].upper() + (" (display only)" if own["state"] == "stale" else ""),
                        (int(anchor[0]) + 6, int(anchor[1]) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, 1, cv2.LINE_AA)
    opponent = sample.opponent_track
    if opponent is not None and opponent["box"] is not None:
        x, y, w, h = (int(round(value * scale)) for value in opponent["box"])
        colour = _OPPONENT_COLOUR.get(opponent["state"], (190, 190, 190))
        cv2.rectangle(image, (x, y), (x + w, y + h), colour, 2, cv2.LINE_AA)
        label = "OPPONENT" if opponent["state"] == "tracked" else "OPPONENT? no position"
        if opponent.get("image_based"):
            confidence = "" if opponent.get("confidence") is None else f" {opponent['confidence']:.2f}"
            label = ("OPPONENT" + confidence) if opponent["state"] == "tracked" else f"TRACKING {opponent['state'].upper()}{confidence}: no position"
            centre = opponent.get("centre_px")
            if centre is not None:
                cv2.drawMarker(image, (int(round(centre[0] * scale)), int(round(centre[1] * scale))), colour, cv2.MARKER_CROSS, 14, 2, cv2.LINE_AA)
        cv2.putText(image, label, (x, max(12, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1, cv2.LINE_AA)


class ClickQueue:
    """Left clicks on the camera part of the window, kept with the frame that was on screen at the time."""

    def __init__(self, frame_width: int | None = None) -> None:
        self.frame_width = frame_width
        self.shown: tuple[np.ndarray, float] | None = None
        self.pending: list[tuple[tuple[float, float], np.ndarray | None, float | None]] = []

    def on_mouse(self, event: int, x: int, y: int, _flags: int = 0, _param: Any = None) -> None:
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        if self.frame_width is not None and x >= self.frame_width:
            return                                           # a click on the text panel is not a selection
        frame, t = self.shown if self.shown is not None else (None, None)
        self.pending.append(((float(x), float(y)), frame, t))

    def take(self) -> list[tuple[tuple[float, float], np.ndarray | None, float | None]]:
        pending, self.pending = self.pending, []
        return pending


def apply_clicks(pipeline: CameraPipeline, queue: ClickQueue, frame: np.ndarray, t: float) -> list[dict[str, Any]]:
    """Hand queued clicks to the opponent tracker, each on the frame the user was looking at."""
    events = []
    for pixel, shown, shown_t in queue.take():
        result = pipeline.select_opponent(frame if shown is None else shown, pixel, t if shown_t is None else shown_t)
        events.append({"event": "click", "pixel": [pixel[0], pixel[1]], "accepted": bool(result.ok), "reason": result.reason})
    return events


# ----------------------------------------------------------------------------------------------
# The viewer's own loop


class LiveTrack:
    """Frames in, tracker reports out. No window, no camera: those belong to the caller."""

    def __init__(self, config: OverlayConfig, clicks: ClickSet, tracker_settings: TrackerSettings | None = None,
                 opponent_settings: OpponentSettings | None = None, opponent_mode: str = "click") -> None:
        self.config, self.clicks = config, clicks
        self.opponent_mode = opponent_mode               # "click" (colour segmentation) or "roi" (drag a box, any object)
        self.tracker_settings = tracker_settings or live_tracker_settings(config)
        self.opponent_settings = opponent_settings
        self.pipeline: CameraPipeline | None = None
        self.map_note = ""
        self.fit_level, self.camera_detail = "", ""
        self.frames = 0

    def start(self, image_size: tuple[int, int]) -> None:
        mapping, camera, quality = fit_profile(self.config, self.clicks, image_size)
        self.pipeline = CameraPipeline(self.config, mapping, camera, multi_marker=True, click_opponent=True,
                                       tracker_settings=self.tracker_settings, opponent_settings=self.opponent_settings,
                                       opponent_mode=self.opponent_mode)
        self.map_note = "" if mapping.valid else f"NO FLOOR MAP ({mapping.reason}): calibrate first; tag ids only"
        self.fit_level, self.camera_detail = quality.level, camera.detail

    def step(self, frame: np.ndarray, t: float, events: Iterable[tuple] = ()) -> tuple[CameraSample, dict[str, Any]]:
        """One frame. ``events``: ("click", (x, y)) selects on THIS frame, ("clear",) drops the selection."""
        if self.pipeline is None:
            self.start((frame.shape[1], frame.shape[0]))
        happened: list[dict[str, Any]] = []
        for event in events:
            if event[0] == "click":
                result = self.pipeline.select_opponent(frame, event[1], t)
                happened.append({"event": "click", "pixel": [float(event[1][0]), float(event[1][1])],
                                 "accepted": bool(result.ok), "reason": result.reason})
            elif event[0] == "roi":
                result = self.pipeline.select_roi(frame, event[1], t)
                happened.append({"event": "roi", "rect": [float(v) for v in event[1]], "accepted": bool(result.ok), "reason": result.reason})
            elif event[0] == "clear":
                self.pipeline.clear_opponent()
                happened.append({"event": "clear"})
            else:
                raise ValueError(f"unknown event {event[0]!r}")
        return self.finish(frame, t, happened)

    def finish(self, frame: np.ndarray, t: float, happened: list[dict[str, Any]]) -> tuple[CameraSample, dict[str, Any]]:
        sample = self.pipeline.process(frame, t)
        record = {"frame": self.frames, "t_s": round(t, 6), "markers_decoded": sample.marker_ids,
                  "own_track": sample.own_track, "opponent_track": sample.opponent_track, "events": happened,
                  "processing_ms": {key: round(value, 3) for key, value in sample.timing_ms.items()},
                  "output": "none: display and trace only, nothing is sent to the robot"}
        self.frames += 1
        return sample, record

    def lines(self, sample: CameraSample) -> list[tuple[str, str]]:
        lines = [(f"markers decoded: {sample.marker_ids or 'none'}   dictionary {self.config.dictionary}", "note")]
        lines += tracking_lines(sample)
        lines.append((f"software time this frame: {sample.timing_ms['total']:.1f} ms (camera latency not included)", "note"))
        if self.config.placeholders:
            lines.append(("PLACEHOLDER config values in use: " + "; ".join(self.config.placeholders), "bad"))
        if self.map_note:
            lines.append((self.map_note, "bad"))
        lines.append(("DRAG a box around the opponent to select it   c clear   q quit" if self.opponent_mode == "roi"
                      else "LEFT-CLICK the opponent to select it   c clear   q quit", "note"))
        lines.append(("Output: screen and trace only. There is no connection to the robot.", "note"))
        return lines


def run_frames(track: LiveTrack, frames: Iterable[tuple[float, np.ndarray]],
               events: dict[int, list[tuple]] | Callable[[int, float], Iterable[tuple]] | None = None,
               trace: TextIO | None = None) -> list[dict[str, Any]]:
    """The viewer's loop on any frame source. ``events`` maps a frame number to scripted clicks/clears."""
    records = []
    for index, (t, frame) in enumerate(frames):
        todo = () if events is None else (events(index, t) if callable(events) else events.get(index, ()))
        _, record = track.step(frame, t, todo)
        records.append(record)
        if trace is not None:
            trace.write(json.dumps(record) + "\n")
    return records


def summarise(records: list[dict[str, Any]]) -> dict[str, Any]:
    totals = [record["processing_ms"]["total"] for record in records[1:]]
    own = Counter(record["own_track"]["state"] for record in records)
    opponent = Counter(record["opponent_track"]["state"] for record in records)
    rejected = Counter(marker for record in records for marker in record["own_track"]["rejected"])
    return {"frames": len(records), "own_track_states": dict(own), "opponent_states": dict(opponent),
            "markers_rejected_frames": dict(rejected),
            "events": [event for record in records for event in record["events"]],
            "software_ms_per_frame": ({"median": round(float(np.median(totals)), 2), "p95": round(float(np.percentile(totals, 95)), 2),
                                       "max": round(float(np.max(totals)), 2)} if totals else None),
            "note": "software time on this computer only; camera latency is not measured",
            "output": "screen and trace only; this program has no connection to the robot"}


def run_camera(args: argparse.Namespace) -> dict[str, Any]:
    """Opens a REAL camera. Only reached through ``--camera N`` on the command line."""
    from control_center.tools import arena_overlay          # its camera opener and capture settings, unchanged
    from control_center.tools.shadow_stream_demo import _override_markers, annotate, panel

    config = _override_markers(load_config(args.config), args.dictionary, args.tag_size_mm)
    points_path = Path(args.points) if args.points else profile_points_path(args.profile)
    clicks = load_clicks(points_path) if points_path.exists() else ClickSet()
    if clicks.historical:                                   # checked before any camera is opened
        raise ValueError(f"{points_path} is a HISTORICAL footage profile; it is never used with a live camera")
    if not args.points:                                         # an explicit --points file bypasses the profile on purpose
        config = with_arena_setup(config, args.profile, clicks)       # the profile's saved arena and hazards, if it has any
    source = arena_overlay.CameraSource(args.camera, arena_overlay.camera_settings(args))
    track = LiveTrack(config, clicks)
    out_dir = Path(args.out) / datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    queue = ClickQueue()
    records: list[dict[str, Any]] = []
    cleared: list[dict[str, Any]] = []
    message = ""
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.setMouseCallback(WINDOW, queue.on_mouse)
    try:
        with (out_dir / "trace.jsonl").open("w", encoding="utf-8") as trace:
            while True:
                ok, frame, t = source.read()
                if frame is None:
                    if not ok:
                        message = "camera stopped returning frames"
                        break
                    continue
                if track.pipeline is None:
                    track.start((frame.shape[1], frame.shape[0]))
                queue.frame_width = frame.shape[1]
                happened, cleared = cleared + apply_clicks(track.pipeline, queue, frame, t), []
                sample, record = track.finish(frame, t, happened)
                records.append(record)
                if len(records) <= 30000:
                    trace.write(json.dumps(record) + "\n")
                view = np.hstack([annotate(frame, track.pipeline, sample),
                                  panel(track.lines(sample), PANEL_WIDTH, frame.shape[0], "LIVE TRACK - read-only, nothing is sent")])
                queue.shown = (frame, t)
                cv2.imshow(WINDOW, view)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27) or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                    break
                if key == ord("c"):
                    track.pipeline.clear_opponent()
                    cleared.append({"event": "clear"})
                if args.max_frames is not None and len(records) >= args.max_frames:
                    break
    finally:
        source.close()
        cv2.destroyAllWindows()
    summary = summarise(records)
    summary.update({"input_source": f"LIVE camera {source.description}, profile '{args.profile}'", "config": str(args.config),
                    "tracker_settings": track.tracker_settings.to_report(), "trace": str(out_dir / "trace.jsonl"), "last_message": message})
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m control_center.tools.live_track",
                                     description="Read-only live tracking viewer: chassis tracker + click-selected opponent. Nothing is sent.")
    parser.add_argument("--camera", type=int, required=True, metavar="N", help="live camera index")
    parser.add_argument("--config", default=str(REPO_ROOT / "config" / "home_test_overlay.yaml"), help="overlay config with the arena and the tag layout")
    parser.add_argument("--profile", default="home_test", help="calibration profile (home_test, or live at the venue); never a historical one")
    parser.add_argument("--points", help="clicked-points file, instead of the profile's")
    parser.add_argument("--dictionary", help="override bot.dictionary, e.g. DICT_4X4_50")
    parser.add_argument("--tag-size-mm", type=float, help="override the tag's black-square side")
    parser.add_argument("--width", type=int)
    parser.add_argument("--height", type=int)
    parser.add_argument("--fps", type=float)
    parser.add_argument("--backend", choices=["dshow", "msmf", "auto"])
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--max-frames", type=int)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        summary = run_camera(args)
    except (ValueError, RuntimeError) as error:
        print(f"error: {error}")
        return 2
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
