"""Camera signature and comparison rules. Pure data: no camera, no files."""
from __future__ import annotations

import dataclasses

import pytest

from control_center.backend import camera_signature as cs
from control_center.backend.camera_signature import CameraSignature, compare

SAVED = CameraSignature(index=1, device_name="Logitech C525", backend="dshow", width=1280, height=720, fps=30,
                        fourcc="MJPG", extra={"exposure": -6, "auto_exposure": False, "autofocus": False, "focus": 30})


def changed(**kw) -> CameraSignature:
    return dataclasses.replace(SAVED, **kw)


def fields(result) -> dict[str, str]:
    return {d.field: d.severity for d in result.differences}


def test_identical_signatures_match_with_no_differences():
    result = compare(SAVED, CameraSignature.from_dict(SAVED.to_dict()))
    assert result.level == cs.MATCH and result.differences == () and not result.unknown_fields
    assert not result.calibration_void


def test_resolution_change_is_blocking_and_incompatible():
    result = compare(SAVED, changed(width=1920, height=1080))
    assert result.level == cs.INCOMPATIBLE and result.calibration_void
    assert fields(result)["resolution"] == cs.BLOCKING
    assert "1920 x 1080" in result.summary and "1280 x 720" in result.summary and "no longer applies" in result.summary
    assert result.blocking[0].saved == [1280, 720] and result.blocking[0].current == [1920, 1080]


def test_only_one_dimension_changing_is_still_a_resolution_change():
    assert compare(SAVED, changed(height=480)).level == cs.INCOMPATIBLE
    assert compare(SAVED, changed(width=1280, height=1280)).level == cs.INCOMPATIBLE


def test_different_device_name_is_a_warning_and_changed():
    result = compare(SAVED, changed(device_name="Integrated Webcam"))
    assert result.level == cs.CHANGED and fields(result) == {"device_name": cs.WARNING}
    assert "Integrated Webcam" in result.summary and "lens" in result.summary


def test_device_name_comparison_ignores_case_and_spaces():
    assert compare(SAVED, changed(device_name="  logitech c525 ")).level == cs.MATCH


def test_index_change_alone_is_info_when_the_name_matches():
    result = compare(SAVED, changed(index=3))
    assert fields(result) == {"index": cs.INFO}
    assert result.level == cs.MATCH                              # nothing above INFO: the calibration still applies
    assert "same name" in result.differences[0].message


def test_index_change_is_a_warning_when_the_name_is_not_available():
    result = compare(changed(device_name=None), changed(device_name=None, index=3))
    assert fields(result) == {"index": cs.WARNING} and result.level == cs.CHANGED
    assert "device_name" in result.unknown_fields
    assert "not available" in result.differences[0].message


def test_index_change_with_a_different_name_is_a_warning_too():
    result = compare(SAVED, changed(index=0, device_name="Other cam"))
    assert fields(result) == {"device_name": cs.WARNING, "index": cs.WARNING} and result.level == cs.CHANGED


def test_fourcc_change_is_a_warning():
    result = compare(SAVED, changed(fourcc="YUY2"))
    assert fields(result) == {"fourcc": cs.WARNING} and result.level == cs.CHANGED
    assert "field of view" in result.summary


def test_fps_and_backend_changes_are_info_only():
    result = compare(SAVED, changed(fps=15, backend="msmf"))
    assert fields(result) == {"fps": cs.INFO, "backend": cs.INFO}
    assert result.level == cs.MATCH and len(result.differences) == 2
    assert "Minor differences: 2" in result.summary
    assert compare(SAVED, changed(fps=29.8)).differences == ()          # within the tolerance


def test_exposure_and_focus_differences_are_info():
    result = compare(SAVED, changed(extra={"exposure": -3, "auto_exposure": True, "autofocus": False, "focus": 80}))
    assert fields(result) == {"extra.exposure": cs.INFO, "extra.auto_exposure": cs.INFO, "extra.focus": cs.INFO}
    assert result.level == cs.MATCH


def test_autofocus_turned_on_where_it_was_off_is_a_warning():
    result = compare(SAVED, changed(extra={**SAVED.extra, "autofocus": True}))
    assert fields(result) == {"extra.autofocus": cs.WARNING} and result.level == cs.CHANGED
    assert "scale" in result.summary
    assert compare(SAVED, changed(extra={**SAVED.extra, "autofocus": 1})).level == cs.CHANGED        # driver reports 1


def test_autofocus_turned_off_where_it_was_on_is_info():
    on = changed(extra={"autofocus": True})
    result = compare(on, changed(extra={"autofocus": False}))
    assert fields(result) == {"extra.autofocus": cs.INFO} and result.level == cs.MATCH


def test_none_on_either_side_is_unknown_never_a_mismatch():
    result = compare(SAVED, changed(device_name=None, fps=None, fourcc=None, backend=None, index=None, extra={}))
    assert result.differences == () and result.level == cs.MATCH
    assert {"device_name", "fps", "fourcc", "backend", "index", "extra.exposure"} <= set(result.unknown_fields)
    reverse = compare(changed(device_name=None, fourcc=None), SAVED)
    assert reverse.differences == () and "device_name" in reverse.unknown_fields


def test_unknown_picture_size_cannot_be_confirmed():
    result = compare(SAVED, changed(width=None, height=None))
    assert result.level == cs.UNKNOWN and "resolution" in result.unknown_fields
    assert "by eye" in result.summary
    partial = compare(SAVED, changed(height=None))
    assert partial.level == cs.UNKNOWN


def test_missing_signature_on_either_side_is_unknown():
    assert compare(None, SAVED).level == cs.UNKNOWN
    assert compare(SAVED, None).level == cs.UNKNOWN
    assert compare(CameraSignature(), SAVED).level == cs.UNKNOWN
    assert "not open" in compare(SAVED, None).summary


def test_a_warning_still_shows_when_the_picture_size_is_unknown():
    result = compare(SAVED, changed(width=None, height=None, device_name="Other"))
    assert result.level == cs.CHANGED


def test_blocking_wins_over_warnings():
    result = compare(SAVED, changed(width=640, height=480, device_name="Other", fourcc="YUY2"))
    assert result.level == cs.INCOMPATIBLE and len(result.differences) == 3


def test_messages_are_plain_language_and_non_empty():
    result = compare(SAVED, changed(width=640, height=480, device_name="Other", fourcc="YUY2", fps=10, backend="msmf", index=4,
                                    extra={"autofocus": True, "exposure": -1}))
    assert result.differences
    for d in result.differences:
        assert d.message.endswith(".") and len(d.message) > 25
        assert d.severity in cs.SEVERITIES
        assert "None" not in d.message


def test_compare_accepts_dicts_and_dict_round_trip():
    assert compare(SAVED.to_dict(), SAVED.to_dict()).level == cs.MATCH
    assert CameraSignature.from_dict(SAVED.to_dict()) == SAVED
    assert CameraSignature.from_dict(None).is_empty
    assert compare(SAVED, changed(width=800, height=600)).to_dict()["level"] == cs.INCOMPATIBLE


def test_from_settings_prefers_the_actual_size_and_collects_extra_controls():
    settings = {"backend": "DShow", "width": 1280, "height": 720, "fps": 30.0, "fourcc": " MJPG ", "exposure": -6, "autofocus": False}
    sig = CameraSignature.from_settings(settings, index=1, device_name="Logitech C525", actual_size=(640, 480), extra={"focus": 30})
    assert (sig.width, sig.height) == (640, 480)
    assert sig.backend == "dshow" and sig.fourcc == "MJPG" and sig.index == 1
    assert sig.extra == {"exposure": -6, "autofocus": False, "focus": 30}
    requested = CameraSignature.from_settings(settings, index=1)
    assert requested.resolution == (1280, 720) and requested.device_name is None


def test_from_settings_normalises_unknowns_to_none():
    sig = CameraSignature.from_settings({"width": 0, "height": "abc", "fps": 0, "fourcc": "", "backend": ""}, index=-1)
    assert sig.is_empty and sig.resolution is None and sig.fps is None


def test_from_capture_info_matches_the_capture_session_shape():
    info = {"kind": "camera", "description": "camera 1 (dshow, 1280x720)", "index": 1,
            "requested": {"backend": "dshow", "width": 1280, "height": 720, "fps": 30.0, "fourcc": "MJPG"}}
    sig = CameraSignature.from_capture_info(info, (1280, 720))
    assert sig == CameraSignature(index=1, backend="dshow", width=1280, height=720, fps=30.0, fourcc="MJPG")
    assert CameraSignature.from_capture_info(None, None).is_empty


def test_describe_is_one_readable_line():
    assert "Logitech C525" in SAVED.describe() and "1280x720" in SAVED.describe()
    assert "name not available" in CameraSignature(index=2).describe()
