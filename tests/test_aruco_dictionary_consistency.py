"""DICT_4X4_50 end to end: one config key, read by every detector, with no second source that can silently win.

Evidence class: static checks of the source and config files, plus SYNTHETIC frames (control_center/tools/aruco_synth.py). No
camera, serial port or socket is opened. The one piece of real evidence used is the saved output of the user's own marker survey
(artifacts/marker-survey), which is read, never regenerated.
"""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

from control_center.backend.marker_settings import (DETECTOR_PARAMETER_SETS, LIVE_PARAMETER_SET, MARKER_DICTIONARY, OPENCV_DEFAULT_SET,
                                                    build_detector, dictionary_id, dictionary_status, legacy_dictionary,
                                                    parameter_report, project_dictionary)
from control_center.backend.vision import AprilTagTracker
from control_center.tools import aruco_synth as S
from control_center.tools.marker_survey import MarkerSurvey, format_report
from control_center.tools.overlay_core import MarkerDetector, load_config, parse_config

REPO = Path(__file__).resolve().parents[1]


def _dictionary_of(detector) -> np.ndarray:
    return np.asarray(detector.getDictionary().bytesList)


REFERENCE = np.asarray(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50).bytesList)


def _frame(dictionary: str, marker_id: int = 2, side: float = 60.0) -> np.ndarray:
    return S.render([S.PlacedMarker(marker_id, S.square_corners((400, 300), side, 17.0), dictionary=dictionary)])


# ---- the constant and the config files -----------------------------------------------------------------

def test_the_project_dictionary_is_dict_4x4_50_and_the_live_config_says_so():
    assert MARKER_DICTIONARY == "DICT_4X4_50"
    assert project_dictionary() == "DICT_4X4_50"
    live = load_config(REPO / "config" / "arena_overlay.yaml")
    assert live.dictionary == MARKER_DICTIONARY
    # a config with no bot.dictionary key falls back to the same constant, not to another dictionary
    data = yaml.safe_load((REPO / "config" / "arena_overlay.yaml").read_text(encoding="utf-8"))
    del data["bot"]["dictionary"]
    assert parse_config(data).dictionary == MARKER_DICTIONARY


def test_every_overlay_config_in_the_repository_names_the_same_dictionary():
    paths = sorted((REPO / "config").glob("*overlay*.yaml"))
    assert {p.name for p in paths} >= {"arena_overlay.yaml", "home_test_overlay.yaml"}
    for path in paths:
        assert load_config(path).dictionary == MARKER_DICTIONARY, f"{path.name} names another dictionary"


def test_an_unknown_dictionary_name_raises_instead_of_falling_back():
    for bad in ("DICT_BOGUS", "4X4_50", "", "dict_4x4_50"):
        with pytest.raises(ValueError, match="Unknown marker dictionary"):
            dictionary_id(bad)
        with pytest.raises(ValueError):
            build_detector(bad)
    data = yaml.safe_load((REPO / "config" / "arena_overlay.yaml").read_text(encoding="utf-8"))
    data["bot"]["dictionary"] = "DICT_BOGUS"
    with pytest.raises(ValueError):
        parse_config(data)


def test_the_survey_artifacts_from_the_real_camera_say_4x4_50_and_never_36h11():
    """REAL evidence recorded by the user on 21 Nov (marker survey on camera 1): the markers decode as DICT_4X4_50 only."""
    surveys = sorted((REPO / "artifacts" / "marker-survey").glob("*/survey.json"))
    if not surveys:
        pytest.skip("no saved marker survey in artifacts/marker-survey")
    for path in surveys:
        report = json.loads(path.read_text(encoding="utf-8"))
        scores = report["scores"]
        if report["frames"] < 100 or not scores[MARKER_DICTIONARY]["frame_detections"]:
            continue
        assert report["best_dictionary"] == MARKER_DICTIONARY, path
        for other in ("DICT_APRILTAG_36h11", "DICT_APRILTAG_25h9", "DICT_APRILTAG_16h5", "DICT_5X5_50", "DICT_6X6_50"):
            assert scores[other]["frame_detections"] == 0, (path, other)


# ---- the legacy window ---------------------------------------------------------------------------------

def test_the_legacy_config_no_longer_carries_a_dictionary_that_could_win():
    legacy = yaml.safe_load((REPO / "config" / "control_center.yaml").read_text(encoding="utf-8"))
    assert "april_tag_family" not in legacy and "family" not in (legacy.get("april_tag") or {})
    assert legacy_dictionary(legacy) == (MARKER_DICTIONARY, "")


def test_a_stale_legacy_key_is_reported_and_never_wins():
    old = {"april_tag_family": "DICT_APRILTAG_36h11", "april_tag": {"family": "DICT_APRILTAG_36h11"}}
    name, warning = legacy_dictionary(old)
    assert name == MARKER_DICTIONARY and "DICT_APRILTAG_36h11" in warning and "april_tag.family" in warning and MARKER_DICTIONARY in warning
    assert legacy_dictionary({"april_tag_family": MARKER_DICTIONARY}) == (MARKER_DICTIONARY, "")
    name, warning = legacy_dictionary({"april_tag_family": "DICT_NONSENSE"})
    assert name == MARKER_DICTIONARY and "DICT_NONSENSE" in warning


def test_the_legacy_detector_decodes_4x4_50_and_still_decodes_apriltags_when_asked():
    detections, own = AprilTagTracker(MARKER_DICTIONARY, 2).detect(_frame("DICT_4X4_50", 2))
    assert [d.tag_id for d in detections] == [2] and own is not None and own.tag_id == 2
    # legacy 36h11 still works when it is asked for by name ...
    legacy = AprilTagTracker("DICT_APRILTAG_36h11", 7)
    assert [d.tag_id for d in legacy.detect(_frame("DICT_APRILTAG_36h11", 7))[0]] == [7]
    # ... but it used to be what ANY unrecognised name silently became. Now it raises.
    with pytest.raises(ValueError, match="Unknown marker dictionary"):
        AprilTagTracker("DICT_4X4_5O", 0)
    assert AprilTagTracker("DICT_APRILTAG_36h11", 0).detect(_frame("DICT_4X4_50", 2))[0] == []      # the old config could not see the markers


# ---- every detector in the repository ----------------------------------------------------------------------

def _all_detectors():
    config = load_config(REPO / "config" / "arena_overlay.yaml")
    from control_center.backend.live_vision import LiveVision
    from control_center.tools.overlay_core import FloorMapping, CameraEstimate
    from control_center.tools.stream_adapter import CameraPipeline
    from control_center.tools.marker_survey import MarkerSurvey as Survey

    survey = Survey()
    return {
        "MarkerDetector(config.dictionary)": MarkerDetector(config.dictionary)._detector,
        "LiveVision (Command Centre live path)": LiveVision(config).detector._detector,
        "CameraPipeline (TRACKING screen)": CameraPipeline(config, FloorMapping(), CameraEstimate(), multi_marker=True).detector._detector,
        "marker survey, project dictionary": survey.detectors[survey.project_dictionary],
        "legacy window detector": AprilTagTracker(legacy_dictionary({})[0], 0)._detector,
        "build_detector()": build_detector(),
    }


def test_every_detector_in_the_live_and_legacy_paths_is_built_on_dict_4x4_50():
    detectors = _all_detectors()
    assert len(detectors) == 6
    for name, detector in detectors.items():
        assert detector.getDictionary().markerSize == 4, name
        assert np.array_equal(_dictionary_of(detector), REFERENCE), f"{name} does not use DICT_4X4_50"


def test_every_detector_decodes_a_4x4_50_marker_and_rejects_markers_of_other_dictionaries():
    good = _frame("DICT_4X4_50", 3)
    wrong = {name: _frame(name, 3) for name in ("DICT_5X5_50", "DICT_6X6_50", "DICT_ARUCO_ORIGINAL", "DICT_APRILTAG_36h11", "DICT_APRILTAG_16h5")}
    for name, detector in _all_detectors().items():
        _, ids, _ = detector.detectMarkers(cv2.cvtColor(good, cv2.COLOR_BGR2GRAY))
        assert ids is not None and ids.flatten().tolist() == [3], name
        for other, image in wrong.items():
            _, ids, _ = detector.detectMarkers(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
            assert ids is None, f"{name} read a {other} marker as id {ids.flatten().tolist()}"


def test_one_named_parameter_set_is_used_everywhere():
    assert LIVE_PARAMETER_SET in DETECTOR_PARAMETER_SETS and OPENCV_DEFAULT_SET in DETECTOR_PARAMETER_SETS
    report = parameter_report()
    assert report["set"] == LIVE_PARAMETER_SET and "synthetic" in report["evidence"]
    reference = build_detector().getDetectorParameters()
    for name, detector in _all_detectors().items():
        params = detector.getDetectorParameters()
        for key in DETECTOR_PARAMETER_SETS[LIVE_PARAMETER_SET]:
            assert getattr(params, key) == getattr(reference, key), (name, key)
    with pytest.raises(ValueError):
        build_detector(MARKER_DICTIONARY, "no_such_set")


def _python_files(*folders: str) -> list[Path]:
    return [p for folder in folders for p in sorted((REPO / folder).rglob("*.py")) if "__pycache__" not in p.parts]


def test_no_other_file_builds_a_detector_or_chooses_a_dictionary():
    """A detector (or its parameters) built anywhere else would be a second place that can pick another dictionary."""
    builders = {p.relative_to(REPO).as_posix() for p in _python_files("control_center", "scripts")
                if re.search(r"\bArucoDetector\(|\bDetectorParameters\(", p.read_text(encoding="utf-8"))}
    assert builders == {"control_center/backend/marker_settings.py"}, builders
    chooses = {p.relative_to(REPO).as_posix() for p in _python_files("control_center", "scripts")
               if re.search(r"getPredefinedDictionary\(", p.read_text(encoding="utf-8"))}
    allowed = {"control_center/backend/marker_settings.py",                 # the one detector factory
               "control_center/tools/marker_survey.py",                     # asks how many ids a dictionary has
               "control_center/tools/overlay_fixture.py", "control_center/tools/track_fixture.py",       # RENDER markers for fixtures,
               "control_center/tools/aruco_synth.py"}                       # from config.dictionary / by name
    assert chooses <= allowed, sorted(chooses - allowed)
    # and no production module hard-codes an AprilTag family or a DICT_ literal other than the project's (docs/fixtures excepted)
    for path in _python_files("control_center", "scripts"):
        rel = path.relative_to(REPO).as_posix()
        if rel in ("control_center/tools/aruco_synth.py", "control_center/tools/marker_survey.py", "control_center/backend/marker_settings.py",
                   "control_center/backend/vision.py"):
            continue                                   # these list or construct dictionaries by name on purpose
        literals = set(re.findall(r"DICT_[A-Za-z0-9_]+", path.read_text(encoding="utf-8")))
        assert literals <= {"DICT_4X4_50", "DICT_APRILTAG_36h11"}, (rel, literals)
        assert "DICT_APRILTAG_36h11" not in literals or rel.endswith(("shadow_stream_demo.py",)), rel      # help text example only


def test_the_per_profile_saved_setting_aruco_dictionary_is_not_read_by_any_detector():
    """profile_settings has an `aruco_dictionary` slot; nothing may use it to pick a dictionary behind the config's back."""
    readers = []
    for path in _python_files("control_center", "scripts"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        if any(isinstance(node, ast.Constant) and node.value == "aruco_dictionary" for node in ast.walk(tree)):
            readers.append(path.relative_to(REPO).as_posix())
    # profile_settings defines the slot; hub.remember_markers only WRITES the dictionary in force (config bot.dictionary) into it
    # as a record. Neither reads it back to choose a detector dictionary: the config stays the single source.
    assert readers == ["control_center/backend/hub.py", "control_center/backend/profile_settings.py"], readers
    hub_source = (REPO / "control_center" / "backend" / "hub.py").read_text(encoding="utf-8")
    assert 'get_section(self.profile_name, "markers")' not in hub_source and "markers\"][\"aruco_dictionary" not in hub_source


# ---- the survey tool and the screens ---------------------------------------------------------------------------

def test_the_survey_always_tries_the_project_dictionary_first_and_flags_a_mismatch():
    survey = MarkerSurvey(dictionaries=("DICT_5X5_50", "DICT_6X6_50"))               # the caller left it out: it is added
    assert survey.dictionaries[0] == MARKER_DICTIONARY
    assert survey.detectors[MARKER_DICTIONARY].getDictionary().markerSize == 4
    for _ in range(25):
        survey.add(_frame("DICT_4X4_50", 1))
    ok = survey.report()
    assert ok["project_dictionary"] == MARKER_DICTIONARY and ok["project_dictionary_matches_best"] and ok["best_dictionary"] == MARKER_DICTIONARY
    assert "MISMATCH" not in format_report(ok)

    wrong = MarkerSurvey()
    for _ in range(25):
        wrong.add(_frame("DICT_5X5_50", 4))
    report = wrong.report()
    assert report["best_dictionary"] == "DICT_5X5_50" and not report["project_dictionary_matches_best"]
    assert "MISMATCH" in format_report(report) and "project dictionary" in format_report(report).lower()


def test_the_survey_marks_ids_outside_the_printed_range_and_rare_ones():
    survey = MarkerSurvey(dictionaries=(MARKER_DICTIONARY,))
    for index in range(40):
        markers = [S.PlacedMarker(1, S.square_corners((300, 300), 50.0, 5.0))]
        if index < 1:
            markers.append(S.PlacedMarker(17, S.square_corners((800, 300), 50.0, 5.0)))      # seen in 1 of 40 frames
        survey.add(S.render(markers, seed=index))
    rows = {row["id"]: row for row in survey.report()["markers"]}
    assert rows[1]["outside_printed_id_range"] is False and rows[17]["outside_printed_id_range"] is True and rows[17]["rare"] is True
    assert "id outside the printed range" in format_report(survey.report())


def test_screens_say_when_the_dictionary_in_use_is_not_the_project_dictionary():
    text, level = dictionary_status("DICT_4X4_50")
    assert level == "ok" and "project dictionary" in text
    text, level = dictionary_status("DICT_APRILTAG_36h11")
    assert level == "bad" and "NOT the project dictionary" in text and "DICT_4X4_50" in text
