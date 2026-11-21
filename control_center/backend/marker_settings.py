"""The ONE place that says which ArUco dictionary the AutoBot uses and how the marker detector is configured.

Pure OpenCV and PyYAML: no Qt, no camera, no serial port, no socket.

THE DICTIONARY (single source)
    ``MARKER_DICTIONARY`` = ``DICT_4X4_50`` is the project's dictionary. It is chosen in the overlay configuration by
    the key ``bot.dictionary`` (``config/arena_overlay.yaml``; a profile's overlay config may repeat it, and a test
    checks that every one of them says the same). Everything that detects markers reads that key:

        Command Centre live path   hub -> LiveVision / TrackingEngine / CameraPipeline -> overlay_core.MarkerDetector
        marker survey              tools/marker_survey (it also TRIES other dictionaries, to tell you when the config is wrong)
        screens                    TRACKING / SETUP show ``config.dictionary`` and say when it is not the project dictionary
        legacy window              backend/camera_worker, via ``legacy_dictionary`` below: the overlay key wins; the old
                                   ``april_tag_family`` / ``april_tag.family`` keys of config/control_center.yaml are only
                                   compared with it and a mismatch is reported, never silently used.

THE DETECTOR PARAMETERS (single named set)
    ``LIVE_PARAMETER_SET`` names the parameter set every detector in the repository is built with. The values and the
    synthetic evidence for them are in ``DETECTOR_PARAMETER_SETS`` and docs/aruco-tracking-audit.md. They were tuned on
    SYNTHETIC frames only; nothing here has been checked against the real camera.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import cv2
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
OVERLAY_CONFIG_PATH = REPO_ROOT / "config" / "arena_overlay.yaml"

MARKER_DICTIONARY = "DICT_4X4_50"
DICTIONARY_CONFIG_KEY = "bot.dictionary"

#: The marker ids the printed layout is said to use (top 0-3, bottom 4-7, one rear marker 8). Only a statement of intent: what the
#: tracker accepts is the ``bot.markers`` list of the overlay config, and an id outside that list is reported as UNKNOWN, never used.
PRINTED_ID_RANGE = (0, 8)


# --------------------------------------------------------------------------------------
# Dictionary
# --------------------------------------------------------------------------------------

def dictionary_id(name: str) -> int:
    """cv2.aruco constant for a DICT_* name. Unknown names raise: nothing falls back to another dictionary."""
    if not isinstance(name, str) or not name.startswith("DICT_") or not hasattr(cv2.aruco, name):
        known = sorted(n for n in dir(cv2.aruco) if n.startswith("DICT_"))
        raise ValueError(f"Unknown marker dictionary {name!r}; choose one of {known}")
    return int(getattr(cv2.aruco, name))


def project_dictionary(path: Path | str | None = None) -> str:
    """``bot.dictionary`` of the overlay config file (default ``config/arena_overlay.yaml``); ``MARKER_DICTIONARY`` if absent."""
    source = Path(path) if path else OVERLAY_CONFIG_PATH
    try:
        data = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        return MARKER_DICTIONARY
    name = str((data.get("bot") or {}).get("dictionary", MARKER_DICTIONARY))
    dictionary_id(name)
    return name


def dictionary_status(in_use: str, project: str | None = None) -> tuple[str, str]:
    """(text, level) for a screen: is the dictionary in use the project dictionary?"""
    project = project or project_dictionary()
    if in_use == project:
        return f"{in_use} (the project dictionary, {DICTIONARY_CONFIG_KEY})", "ok"
    return (f"{in_use}: NOT the project dictionary {project}. Markers printed for {project} will not decode with it. "
            f"Check {DICTIONARY_CONFIG_KEY} (or a --dictionary override)"), "bad"


LEGACY_KEYS = ("april_tag_family", "april_tag.family")


def legacy_dictionary(legacy_config: Mapping[str, Any], overlay_path: Path | str | None = None) -> tuple[str, str]:
    """(dictionary the legacy window must use, warning). The overlay key wins; a legacy key that disagrees is only reported.

    ``legacy_config`` is the parsed ``config/control_center.yaml``. A legacy key left at an old value (the first dashboard used
    ``DICT_APRILTAG_36h11``) therefore cannot silently make the legacy window look for markers that were never printed.
    """
    project = project_dictionary(overlay_path)
    stated = []
    if legacy_config.get("april_tag_family"):
        stated.append(("april_tag_family", str(legacy_config["april_tag_family"])))
    family = (legacy_config.get("april_tag") or {}).get("family")
    if family:
        stated.append(("april_tag.family", str(family)))
    differing = [f"{key}: {value}" for key, value in stated if value != project]
    if not differing:
        return project, ""
    return project, (f"config/control_center.yaml still says {', '.join(differing)} but the project dictionary is {project} "
                     f"({DICTIONARY_CONFIG_KEY} in config/arena_overlay.yaml); {project} is used. Delete the old key(s).")


# --------------------------------------------------------------------------------------
# Detector parameters
# --------------------------------------------------------------------------------------

LIVE_PARAMETER_SET = "live_v1"
OPENCV_DEFAULT_SET = "opencv_default"

#: name -> cv2.aruco.DetectorParameters attributes that differ from OpenCV's defaults.
DETECTOR_PARAMETER_SETS: dict[str, dict[str, Any]] = {
    OPENCV_DEFAULT_SET: {},
    "subpix_only": {"cornerRefinementMethod": cv2.aruco.CORNER_REFINE_SUBPIX},       # what the repository used before this audit
    # live_v1: OpenCV's defaults except the five below. Each was chosen by comparing synthetic detection rates (36 px markers under a
    # motion streak, 3 markers per frame, 40 frames per cell; docs/aruco-tracking-audit.md has the table), keeping only changes that
    # raised the detection rate without a single wrong id, extra false decode on 150 marker-free textured frames, or more decodes
    # of other dictionaries' markers. NOT checked on the real camera.
    LIVE_PARAMETER_SET: {
        "cornerRefinementMethod": cv2.aruco.CORNER_REFINE_SUBPIX,   # sub-pixel corners (already used before this audit)
        "adaptiveThreshWinSizeMax": 53,                 # default 23: thresholds up to 53 px, so a marker the camera sees LARGER is
                                                        # still binarised cleanly (steps stay 10 px: 3, 13, 23, 33, 43, 53)
        "perspectiveRemovePixelPerCell": 8,             # default 4: sample each bit from an 8 px cell, so a blurred cell averages more pixels
        "perspectiveRemoveIgnoredMarginPerCell": 0.25,  # default 0.13: ignore the blurred rim of each bit cell when reading it
        "maxErroneousBitsInBorderRate": 0.5,            # default 0.35: tolerate a smeared black border (id bits still need their own match)
    },
}


def detector_parameters(set_name: str = LIVE_PARAMETER_SET) -> "cv2.aruco.DetectorParameters":
    if set_name not in DETECTOR_PARAMETER_SETS:
        raise ValueError(f"unknown detector parameter set {set_name!r}; known: {sorted(DETECTOR_PARAMETER_SETS)}")
    params = cv2.aruco.DetectorParameters()
    for key, value in DETECTOR_PARAMETER_SETS[set_name].items():
        if not hasattr(params, key):
            raise ValueError(f"parameter set {set_name!r}: cv2.aruco.DetectorParameters has no {key!r}")
        setattr(params, key, value)
    return params


def build_detector(dictionary: str = MARKER_DICTIONARY, set_name: str = LIVE_PARAMETER_SET) -> "cv2.aruco.ArucoDetector":
    return cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(dictionary_id(dictionary)), detector_parameters(set_name))


def parameter_report(set_name: str = LIVE_PARAMETER_SET) -> dict[str, Any]:
    """The set's name and every value that differs from OpenCV's default, for traces and the audit document."""
    return {"set": set_name, "differs_from_opencv_default": dict(DETECTOR_PARAMETER_SETS[set_name]),
            "evidence": "synthetic frames only; see docs/aruco-tracking-audit.md"}
