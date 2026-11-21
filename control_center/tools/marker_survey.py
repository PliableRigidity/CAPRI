"""MARKER SURVEY: which markers does the camera actually decode, in which dictionary, how often? Read-only.

    python -m control_center.tools.marker_survey --camera N [--seconds 10]
    python -m control_center.tools.marker_survey --image PATH
    python -m control_center.tools.marker_survey --video PATH

For each dictionary in a list, every frame is searched for markers and, per (dictionary, id), the tool
counts the frames it was seen in (the detection rate: this is the number that says how intermittent a
marker is), its mean apparent side in pixels and, when ``--config`` with ``--profile`` / ``--points``
gives a valid saved floor calibration, its apparent side in millimetres on the floor plane.

It then prints a table, says which dictionary best explains what was seen, lists ids that were seen
but are not in the config and configured ids that were never seen, and writes
``artifacts/marker-survey/<timestamp>/survey.json``.

It NEVER writes to a config file: what it finds is for you to check against the robot and type in.
A real camera is opened only when YOU pass ``--camera N``. There is no connection to the robot.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Iterator

import cv2
import numpy as np

from control_center.backend.marker_settings import (DICTIONARY_CONFIG_KEY, MARKER_DICTIONARY, PRINTED_ID_RANGE, build_detector,
                                                    dictionary_id, project_dictionary)
from control_center.tools.overlay_core import (CameraEstimate, ClickSet, FloorMapping, OverlayConfig,
                                               lift_to_height, load_clicks, load_config, profile_points_path)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = REPO_ROOT / "artifacts" / "marker-survey"
# The survey deliberately tries OTHER dictionaries too: that is how it notices a config that names the wrong one. The project
# dictionary (marker_settings.MARKER_DICTIONARY) is always the first one tried, whatever list is passed.
DEFAULT_DICTIONARIES = (MARKER_DICTIONARY, "DICT_4X4_100", "DICT_5X5_50", "DICT_6X6_50", "DICT_ARUCO_ORIGINAL",
                        "DICT_APRILTAG_36h11", "DICT_APRILTAG_25h9", "DICT_APRILTAG_16h5")
RARE_SHARE = 0.05              # an id seen in fewer frames than this (of 20 or more) may be a false decode


def _detector(name: str) -> cv2.aruco.ArucoDetector:
    return build_detector(name)            # the same named parameter set as overlay_core.MarkerDetector (marker_settings)


class MarkerSurvey:
    """Accumulates detections frame by frame. No camera, no window, no files: those are the caller's."""

    def __init__(self, dictionaries: Iterable[str] = DEFAULT_DICTIONARIES, mapping: FloorMapping | None = None,
                 camera: CameraEstimate | None = None, config: OverlayConfig | None = None) -> None:
        self.project_dictionary = config.dictionary if config is not None else project_dictionary()
        self.dictionaries = list(dict.fromkeys([self.project_dictionary, *dictionaries]))      # the project dictionary is always tried
        self.detectors = {name: _detector(name) for name in self.dictionaries}
        self.mapping = mapping if mapping is not None and mapping.valid else None
        self.camera, self.config = camera, config
        self.frames = 0
        self.seen: dict[str, dict[int, dict[str, Any]]] = {name: {} for name in self.dictionaries}
        self.last: dict[str, list[tuple[int, np.ndarray]]] = {name: [] for name in self.dictionaries}
        self.image_size: tuple[int, int] | None = None

    def add(self, frame: np.ndarray) -> None:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        self.image_size = (frame.shape[1], frame.shape[0])
        for name, detector in self.detectors.items():
            corners, ids, _ = detector.detectMarkers(gray)
            found: list[tuple[int, np.ndarray]] = []
            counted: set[int] = set()
            for raw, marker_id in zip(corners, [] if ids is None else ids.flatten()):
                points = np.asarray(raw, float).reshape(4, 2)
                marker_id = int(marker_id)
                found.append((marker_id, points))
                entry = self.seen[name].setdefault(marker_id, {"frames": 0, "detections": 0, "side_px": [], "side_floor_mm": [],
                                                               "side_at_height_mm": [], "first_frame": self.frames, "last_frame": self.frames})
                entry["detections"] += 1
                if marker_id not in counted:
                    entry["frames"] += 1
                    counted.add(marker_id)
                entry["last_frame"] = self.frames
                entry["side_px"].append(float(np.mean(np.linalg.norm(points - np.roll(points, -1, axis=0), axis=1))))
                if self.mapping is not None:
                    floor = self.mapping.pixel_to_floor(points)
                    if np.isfinite(floor).all():
                        entry["side_floor_mm"].append(1000.0 * float(np.mean(np.linalg.norm(floor - np.roll(floor, -1, axis=0), axis=1))))
                        spec = None if self.config is None or name != self.config.dictionary else self.config.markers.get(marker_id)
                        if spec is not None and spec.face != "rear" and self.camera is not None and self.camera.position_m is not None:
                            plane = lift_to_height(floor, spec.height_m, self.camera.position_m)
                            entry["side_at_height_mm"].append(1000.0 * float(np.mean(np.linalg.norm(plane - np.roll(plane, -1, axis=0), axis=1))))
            self.last[name] = found
        self.frames += 1

    # ---- results ----------------------------------------------------------------------------
    def _rows(self, name: str) -> list[dict[str, Any]]:
        rows = []
        for marker_id, entry in sorted(self.seen[name].items()):
            rate = entry["frames"] / self.frames if self.frames else 0.0
            rows.append({
                "dictionary": name, "id": marker_id, "frames_seen": entry["frames"], "frames_total": self.frames,
                "detection_rate": round(rate, 4), "seen_more_than_once_in_a_frame": entry["detections"] > entry["frames"],
                "mean_side_px": round(float(np.mean(entry["side_px"])), 2),
                "mean_side_on_floor_plane_mm": round(float(np.mean(entry["side_floor_mm"])), 1) if entry["side_floor_mm"] else None,
                "mean_side_at_configured_height_mm": round(float(np.mean(entry["side_at_height_mm"])), 1) if entry["side_at_height_mm"] else None,
                "first_frame": entry["first_frame"], "last_frame": entry["last_frame"],
                "rare": bool(self.frames >= 20 and rate < RARE_SHARE),
                "outside_printed_id_range": not PRINTED_ID_RANGE[0] <= marker_id <= PRINTED_ID_RANGE[1]})
        return rows

    def report(self) -> dict[str, Any]:
        rows = [row for name in self.dictionaries for row in self._rows(name)]
        scores = {}
        for name in self.dictionaries:
            steady = [row for row in rows if row["dictionary"] == name and not row["rare"]]
            scores[name] = {"ids": [row["id"] for row in steady], "frame_detections": sum(row["frames_seen"] for row in steady),
                            "dictionary_size": int(cv2.aruco.getPredefinedDictionary(dictionary_id(name)).bytesList.shape[0])}
        configured = self.project_dictionary             # the config's dictionary, or the project's (marker_settings) when no config is given
        best, best_note = None, "no marker was decoded in any dictionary"
        top = max((score["frame_detections"] for score in scores.values()), default=0)
        if top > 0:
            tied = [name for name in self.dictionaries if scores[name]["frame_detections"] == top]
            # Equal counts (4X4_50 is the first 50 ids of 4X4_100): prefer the configured one, then the smallest.
            best = configured if configured in tied else min(tied, key=lambda name: scores[name]["dictionary_size"])
            best_note = f"{best} explains the most detections ({top} marker sightings, ids {scores[best]['ids']})"
            if len(tied) > 1:
                best_note += f"; tied with {[name for name in tied if name != best]} (the same ids exist in both)"
        report: dict[str, Any] = {
            "generated": datetime.now().isoformat(timespec="seconds"), "frames": self.frames,
            "image_size": list(self.image_size) if self.image_size else None, "dictionaries_tried": self.dictionaries,
            "floor_calibration_used": self.mapping is not None,
            "size_note": ("side on the floor plane treats a raised marker as if it lay on the floor, so it reads LARGER than the "
                          "printed size; 'at configured height' uses the config's height and the camera position"
                          if self.mapping is not None else "no valid floor calibration given: sizes are in pixels only"),
            "best_dictionary": best, "best_dictionary_note": best_note, "scores": scores, "markers": rows,
            "project_dictionary": configured, "project_dictionary_key": DICTIONARY_CONFIG_KEY,
            "project_dictionary_matches_best": best is None or best == configured,
            "printed_id_range": list(PRINTED_ID_RANGE),
            "rare_note": f"'rare' = seen in under {100 * RARE_SHARE:.0f}% of 20 or more frames: possibly a false decode; not counted for the best dictionary",
            "writes_config": False, "output": "survey file and console only; no config file is written, nothing is sent anywhere"}
        if self.config is not None:
            wanted = {marker_id: spec for marker_id, spec in self.config.markers.items()}
            seen_ids = {row["id"] for row in rows if row["dictionary"] == configured and not row["rare"]}
            report["config"] = {
                "path": None if self.config.path is None else str(self.config.path), "dictionary": configured,
                "dictionary_was_tried": configured in self.dictionaries,
                "configured_ids": {str(marker_id): spec.face for marker_id, spec in sorted(wanted.items())},
                "seen_but_not_configured": sorted(seen_ids - set(wanted)),
                "configured_but_never_seen": sorted(set(wanted) - seen_ids),
                "dictionary_matches_best": best is None or best == configured,
                "placeholders": self.config.placeholders}
        return report

    def annotated(self, frame: np.ndarray, dictionary: str | None) -> np.ndarray:
        """The frame with the last detections of one dictionary drawn on it."""
        image = frame.copy()
        for marker_id, points in self.last.get(dictionary or "", []):
            cv2.polylines(image, [np.round(points).astype(np.int32)], True, (70, 210, 120), 2, cv2.LINE_AA)
            corner = np.round(points[0]).astype(int)
            cv2.circle(image, (int(corner[0]), int(corner[1])), 4, (40, 90, 255), -1)       # the marker's own top-left corner
            centre = np.round(points.mean(axis=0)).astype(int)
            cv2.putText(image, str(marker_id), (int(centre[0]) + 8, int(centre[1]) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(image, f"marker survey: {dictionary or 'no dictionary matched'} (red dot = marker's top-left corner)", (10, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)
        return image


def format_report(report: dict[str, Any]) -> str:
    lines = [f"MARKER SURVEY  {report['frames']} frame(s), image {report['image_size']}, dictionaries tried: {len(report['dictionaries_tried'])}",
             f"{'dictionary':<22}{'id':>4}{'seen':>7}{'of':>7}{'rate':>8}{'side px':>9}{'floor mm':>10}{'at height mm':>14}  note"]
    for row in report["markers"]:
        floor = "-" if row["mean_side_on_floor_plane_mm"] is None else f"{row['mean_side_on_floor_plane_mm']:.1f}"
        height = "-" if row["mean_side_at_configured_height_mm"] is None else f"{row['mean_side_at_configured_height_mm']:.1f}"
        note = ", ".join(word for word, flag in (("RARE (false decode?)", row["rare"]), ("twice in one frame", row["seen_more_than_once_in_a_frame"]),
                                                 ("id outside the printed range", row.get("outside_printed_id_range"))) if flag)
        lines.append(f"{row['dictionary']:<22}{row['id']:>4}{row['frames_seen']:>7}{row['frames_total']:>7}{100 * row['detection_rate']:>7.1f}%"
                     f"{row['mean_side_px']:>9.1f}{floor:>10}{height:>14}  {note}")
    if not report["markers"]:
        lines.append("  (no marker decoded in any dictionary)")
    lines.append(f"BEST DICTIONARY: {report['best_dictionary_note']}")
    if report.get("project_dictionary") is not None:
        lines.append(f"PROJECT DICTIONARY: {report['project_dictionary']} ({report['project_dictionary_key']})"
                     + ("" if report["project_dictionary_matches_best"] else
                        f"  MISMATCH: the detections are explained by {report['best_dictionary']}, not by the project dictionary"))
    lines.append(f"SIZES: {report['size_note']}")
    config = report.get("config")
    if config:
        lines.append(f"CONFIG {config['path']}: dictionary {config['dictionary']}, ids {config['configured_ids']}")
        if not config["dictionary_was_tried"]:
            lines.append("  the config's dictionary was not in the list tried: add it with --dictionaries")
        if not config["dictionary_matches_best"]:
            lines.append(f"  MISMATCH: the config says {config['dictionary']} but {report['best_dictionary']} explains the detections")
        lines.append(f"  seen in {config['dictionary']} but NOT in the config: {config['seen_but_not_configured'] or 'none'}")
        lines.append(f"  configured but NEVER seen: {config['configured_but_never_seen'] or 'none'}")
        if config["placeholders"]:
            lines.append("  config still has PLACEHOLDER values: " + "; ".join(config["placeholders"]))
    lines.append("Nothing was written to any config file. Check these against the robot and edit the config yourself.")
    return "\n".join(lines)


def run_frames(survey: MarkerSurvey, frames: Iterable[np.ndarray]) -> np.ndarray | None:
    """Feed frames from any source (tests use SYNTHETIC ones). Returns the last frame."""
    last = None
    for frame in frames:
        survey.add(frame)
        last = frame
    return last


def write_survey(report: dict[str, Any], out_dir: Path | str, annotated: np.ndarray | None = None) -> Path:
    folder = Path(out_dir)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "survey.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if annotated is not None:
        cv2.imwrite(str(folder / "last_frame_annotated.png"), annotated)
    return path


# ----------------------------------------------------------------------------------------------
# Sources. A real camera is opened ONLY in camera_frames, only when --camera N is given.


def image_frames(path: str) -> Iterator[np.ndarray]:
    frame = cv2.imread(path)
    if frame is None:
        raise RuntimeError(f"could not read image {path}")
    yield frame


def video_frames(path: str, max_frames: int | None) -> Iterator[np.ndarray]:
    capture = cv2.VideoCapture(path)
    if not capture.isOpened():
        raise RuntimeError(f"could not open video {path}")
    try:
        count = 0
        while max_frames is None or count < max_frames:
            ok, frame = capture.read()
            if not ok or frame is None:
                return
            count += 1
            yield frame
    finally:
        capture.release()


def camera_frames(args: argparse.Namespace) -> Iterator[np.ndarray]:
    from control_center.tools import arena_overlay          # its camera opener and capture settings, unchanged

    source = arena_overlay.CameraSource(args.camera, arena_overlay.camera_settings(args))
    print(f"camera open: {source.description}; surveying for {args.seconds:g} s (hold the robot still, then move it, then flip it)")
    started, count = time.monotonic(), 0
    try:
        while time.monotonic() - started < args.seconds and (args.max_frames is None or count < args.max_frames):
            ok, frame, _ = source.read()
            if frame is None:
                if not ok:
                    return
                continue
            count += 1
            yield frame
    finally:
        source.close()


def load_calibration(args: argparse.Namespace, image_size: tuple[int, int]) -> tuple[OverlayConfig | None, FloorMapping | None, CameraEstimate | None, str]:
    """The config and, if a saved calibration fits this picture, its floor mapping. Read only."""
    if not args.config:
        return None, None, None, "no --config given: ids are not compared with a config, sizes are in pixels only"
    config = load_config(args.config)
    points_path = Path(args.points) if args.points else (profile_points_path(args.profile) if args.profile else None)
    if points_path is None or not points_path.exists():
        return config, None, None, f"no saved floor calibration ({points_path or 'no --profile/--points'}): sizes are in pixels only"
    clicks: ClickSet = load_clicks(points_path)
    if clicks.historical and args.camera is not None:
        raise ValueError(f"{points_path} is a HISTORICAL footage profile; it is never used with a live camera")
    if args.profile and not args.points:
        from control_center.tools.arena_setup import with_arena_setup

        config = with_arena_setup(config, args.profile, clicks)
    from control_center.tools.stream_adapter import fit_profile

    mapping, camera, quality = fit_profile(config, clicks, image_size)
    if not mapping.valid:
        return config, None, None, f"saved calibration {points_path} is not valid for this picture ({mapping.reason}): sizes are in pixels only"
    return config, mapping, camera, f"floor calibration {points_path} (fit {quality.level}); camera position {camera.source}"


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.image:
        frames: Iterator[np.ndarray] = image_frames(args.image)
        source = f"image {args.image}"
    elif args.video:
        frames = video_frames(args.video, args.max_frames)
        source = f"video {args.video}"
    else:
        points_path = Path(args.points) if args.points else (profile_points_path(args.profile) if args.profile else None)
        if points_path is not None and points_path.exists() and load_clicks(points_path).historical:      # before any camera is opened
            raise ValueError(f"{points_path} is a HISTORICAL footage profile; it is never used with a live camera")
        frames = camera_frames(args)
        source = f"LIVE camera {args.camera}"
    first = next(frames, None)
    if first is None:
        raise RuntimeError(f"{source} gave no frame")
    config, mapping, camera, calibration_note = load_calibration(args, (first.shape[1], first.shape[0]))
    survey = MarkerSurvey(args.dictionaries or DEFAULT_DICTIONARIES, mapping, camera, config)
    survey.add(first)
    last = run_frames(survey, frames)
    last = first if last is None else last
    report = survey.report()
    report["source"], report["calibration"] = source, calibration_note
    out_dir = Path(args.out) / datetime.now().strftime("%Y%m%d-%H%M%S")
    shown = report["best_dictionary"] or (config.dictionary if config is not None and config.dictionary in survey.dictionaries else None)
    report["survey_file"] = str(out_dir / "survey.json")
    write_survey(report, out_dir, survey.annotated(last, shown))
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m control_center.tools.marker_survey",
                                     description="Read-only survey of the markers the camera decodes. Never writes a config file.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--camera", type=int, metavar="N", help="live camera index (opens a REAL camera)")
    source.add_argument("--image", help="saved frame (PNG/JPG)")
    source.add_argument("--video", help="saved video file")
    parser.add_argument("--seconds", type=float, default=10.0, help="camera: how long to survey (default 10)")
    parser.add_argument("--max-frames", type=int, help="stop after this many frames")
    parser.add_argument("--dictionaries", nargs="+", metavar="DICT", help=f"dictionaries to try (default: {' '.join(DEFAULT_DICTIONARIES)})")
    parser.add_argument("--config", help="overlay config to compare ids with (read only)")
    parser.add_argument("--profile", help="saved calibration profile for millimetre sizes (read only)")
    parser.add_argument("--points", help="clicked-points file, instead of the profile's (read only)")
    parser.add_argument("--width", type=int)
    parser.add_argument("--height", type=int)
    parser.add_argument("--fps", type=float)
    parser.add_argument("--backend", choices=["dshow", "msmf", "auto"])
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = run(args)
    except (ValueError, RuntimeError) as error:
        print(f"error: {error}")
        return 2
    print(format_report(report))
    print(f"calibration: {report['calibration']}")
    print(f"written: {report['survey_file']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
