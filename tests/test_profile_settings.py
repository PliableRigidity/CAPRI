"""Per-profile settings store: atomic writes, backup, isolation, forbidden live state, reference picture.

Everything uses pytest's tmp_path; nothing is written under config/. Software tests of file handling only.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import cv2
import numpy as np
import pytest

from control_center.backend import profile_settings as ps
from control_center.backend.profile_settings import ProfileSettingsError, ProfileSettingsStore

T0 = datetime(2026, 11, 21, 10, 0, 0, tzinfo=timezone.utc)


class Clock:
    def __init__(self) -> None:
        self.now = T0

    def __call__(self) -> datetime:
        self.now += timedelta(minutes=1)
        return self.now


@pytest.fixture
def clock() -> Clock:
    return Clock()


@pytest.fixture
def store(tmp_path, clock) -> ProfileSettingsStore:
    return ProfileSettingsStore(tmp_path / "ps", clock=clock)


CAMERA = {"index": 1, "device_name": "Logitech C525", "backend": "dshow", "width": 1280, "height": 720, "fps": 30.0,
          "fourcc": "MJPG", "extra": {"exposure": -6, "auto_exposure": False, "focus": 30, "autofocus": False}}
CONTEXT = {"camera_signature": dict(CAMERA), "resolution": [1280, 720], "frame_reference": None, "arena_setup_id": "abc123",
           "corner_pixels": {"arena_corner_c1": [100.5, 600.0], "arena_corner_c2": [1100.0, 610.0]},
           "calibrated_at": "2026-11-20T18:00:00+00:00", "last_confirmed_at": None}
MARKERS = {"aruco_dictionary": "DICT_4X4_50", "self_tag_id": 0, "layout_name": "antweight_v1", "layout_path": "config/arena_overlay.yaml",
           "layout_hash": "deadbeef", "measured_offsets": {"0": {"forward_m": 0.035, "left_m": 0.03}}, "tag_size_m": None}
WIFI = {"host": "192.168.4.1", "port": 4210, "preferred_ssid": "AutoBots-Bot", "auto_connect": False, "last_endpoint_ok_at": "2026-11-20T18:01:00+00:00"}


def dir_files(store) -> list[str]:
    return sorted(p.name for p in store.directory.iterdir()) if store.directory.exists() else []


def photo(width=1280, height=720, seed=0) -> np.ndarray:
    rng = np.random.RandomState(seed)
    base = cv2.resize(rng.randint(0, 255, (height // 16, width // 16, 3), dtype=np.uint8), (width, height), interpolation=cv2.INTER_CUBIC)
    return base


# ------------------------------------------------------------------------------------------ basics

def test_a_never_saved_profile_reads_as_defaults_without_a_problem(store):
    loaded = store.load("live")
    assert loaded.status == "fresh" and loaded.load_problem == "" and not loaded.present
    assert loaded.sections["camera"]["index"] is None and loaded.sections["camera"]["extra"] == {}
    assert loaded.sections["wifi"]["auto_connect"] is False and loaded.sections["wifi"]["host"] is None
    assert loaded.sections["markers"]["tag_size_m"] is None and loaded.sections["preferences"] == {}
    assert all(v is None for v in loaded.saved_at.values())
    assert not store.directory.exists()                     # reading creates nothing


def test_default_directory_is_config_profile_settings_and_construction_touches_nothing():
    expected = Path(__file__).resolve().parents[1] / "config" / "profile_settings"
    before = expected.exists()
    store = ProfileSettingsStore()
    assert store.directory == expected == ps.DEFAULT_DIRECTORY
    assert expected.exists() == before
    assert store.path_for("live") == expected / "live.json"


def test_every_section_round_trips_and_has_its_own_saved_at(store, clock):
    for name, data in (("camera", CAMERA), ("calibration_context", CONTEXT), ("markers", MARKERS), ("wifi", WIFI),
                       ("preferences", {"overlay_opacity": 0.6, "show_hud": True, "nested": {"a": [1, 2]}})):
        assert store.set_section("live", name, data).ok
    loaded = store.load("live")
    assert loaded.present == frozenset(ps.SECTION_NAMES)
    for name, data in (("camera", CAMERA), ("calibration_context", CONTEXT), ("markers", MARKERS), ("wifi", WIFI)):
        assert store.get_section("live", name) == data
    assert store.get_section("live", "preferences")["nested"] == {"a": [1, 2]}
    stamps = [store.section_saved_at("live", n) for n in ps.SECTION_NAMES]
    assert len(set(stamps)) == len(stamps) and all(s.endswith("+00:00") for s in stamps)       # each section stamped when it was written
    assert stamps == sorted(stamps)


def test_sections_are_independent(store):
    store.set_section("live", "camera", CAMERA)
    before = store.section_saved_at("live", "camera")
    store.set_section("live", "wifi", WIFI)
    assert store.get_section("live", "camera") == CAMERA and store.section_saved_at("live", "camera") == before
    assert not store.has_section("live", "markers")
    assert store.clear_section("live", "wifi").ok
    assert store.get_section("live", "wifi")["host"] is None and store.get_section("live", "camera") == CAMERA


def test_set_replaces_but_update_merges(store):
    store.set_section("live", "wifi", WIFI)
    store.update_section("live", "wifi", {"auto_connect": True})
    merged = store.get_section("live", "wifi")
    assert merged["auto_connect"] is True and merged["host"] == "192.168.4.1"
    store.set_section("live", "wifi", {"host": "10.0.0.2"})
    replaced = store.get_section("live", "wifi")
    assert replaced["host"] == "10.0.0.2" and replaced["port"] is None and replaced["auto_connect"] is False


def test_numpy_values_are_stored_as_plain_json(store):
    store.set_section("live", "camera", {"index": np.int64(2), "fps": np.float32(30.0), "width": np.int32(640), "height": 480,
                                         "extra": {"gain": np.float64(1.5)}})
    raw = json.loads(store.path_for("live").read_text(encoding="utf-8"))
    assert raw["sections"]["camera"]["data"]["index"] == 2
    assert store.get_section("live", "camera")["extra"] == {"gain": 1.5}


def test_wrong_types_and_unknown_keys_are_refused(store):
    for section, bad in (("wifi", {"port": 70000}), ("wifi", {"port": True}), ("wifi", {"auto_connect": "yes"}),
                         ("camera", {"width": -5}), ("camera", {"index": -1}), ("camera", {"extra": [1]}),
                         ("markers", {"tag_size_m": 0}), ("markers", {"colour": "red"}), ("calibration_context", {"resolution": [1280]}),
                         ("calibration_context", {"corner_pixels": [[1, 2, 3]]}), ("camera", {"fps": float("nan")}), ("nope", {})):
        with pytest.raises(ProfileSettingsError):
            store.set_section("live", section, bad)
    assert not store.path_for("live").exists()               # nothing refused was written
    with pytest.raises(ValueError):
        store.get_section("live", "nope")


# ------------------------------------------------------------------------------------------ names and isolation

@pytest.mark.parametrize("bad", ["", "../x", "a/b", "a\\b", "a:b", ".hidden", "ends.", "star*", "q?", "x" * 80, "tab\tname"])
def test_bad_profile_names_are_refused(store, bad):
    with pytest.raises(ProfileSettingsError):
        store.load(bad)
    with pytest.raises(ProfileSettingsError):
        store.set_section(bad, "wifi", WIFI)
    assert not store.directory.exists()


def test_profile_name_rule_agrees_with_arena_setup():
    from control_center.tools.arena_setup import ArenaSetupError, check_profile_name
    for name in ("live", "home_test", "bristol-2026", "", "a/b", "a\\b", "a:b", ".x"):
        try:
            check_profile_name(name)
            theirs = True
        except ArenaSetupError:
            theirs = False
        try:
            ps.check_profile_name(name)
            ours = True
        except ProfileSettingsError:
            ours = False
        assert ours == theirs, name


def test_profiles_never_see_each_others_settings(store):
    store.set_section("home_test", "wifi", {**WIFI, "host": "10.1.1.1"})
    store.set_section("home_test", "camera", {**CAMERA, "index": 0})
    store.save_reference_frame("home_test", photo(640, 360, 1))
    assert store.get_section("live", "wifi")["host"] is None and store.get_section("live", "camera")["index"] is None
    assert store.load_reference_frame("live") is None
    store.set_section("live", "wifi", {**WIFI, "host": "192.168.4.1"})
    assert store.get_section("home_test", "wifi")["host"] == "10.1.1.1"
    assert store.get_section("live", "wifi")["host"] == "192.168.4.1"
    assert store.load_reference_frame("home_test") is not None
    assert sorted(p.name for p in store.directory.glob("home_test*")) != sorted(p.name for p in store.directory.glob("live*"))


def test_a_file_whose_embedded_profile_differs_is_refused(store):
    store.set_section("live", "wifi", WIFI)
    store.directory.joinpath("home_test.json").write_bytes(store.path_for("live").read_bytes())        # live's file under home_test's name
    loaded = store.load("home_test")
    assert loaded.sections["wifi"]["host"] is None and loaded.status == "defaults_after_problem"
    assert "belongs to profile 'live'" in loaded.load_problem
    assert store.get_section("live", "wifi")["host"] == "192.168.4.1"
    assert store.set_section("home_test", "preferences", {"a": 1}).ok              # and the bad file can be replaced by a good one
    assert store.load("home_test").load_problem == ""
    assert store.get_section("live", "wifi") == WIFI


# ------------------------------------------------------------------------------------------ forbidden live state

@pytest.mark.parametrize("key", sorted(ps.FORBIDDEN_KEYS))
def test_forbidden_keys_are_refused_in_free_preferences(store, key):
    with pytest.raises(ValueError, match="never persisted"):
        store.set_section("live", "preferences", {key: 1})
    assert not store.path_for("live").exists()


def test_forbidden_keys_are_refused_at_any_depth_and_in_any_section(store):
    nested = [
        ("preferences", {"ui": {"deep": {"opponent_roi": [1, 2, 3, 4]}}}),
        ("preferences", {"list": [{"fine": 1}, {"Control_Mode": "auto"}]}),
        ("camera", {**CAMERA, "extra": {"armed": True}}),
        ("markers", {**MARKERS, "measured_offsets": {"tracking_state": "locked"}}),
        ("wifi", {**WIFI, "wifi_status": "connected"}),
        ("calibration_context", {**CONTEXT, "camera_signature": {"index": 1, "extra": {"world_coordinates_trusted": True}}}),
        ("preferences", {"authority": "pc"}), ("preferences", {"motors_running": False}), ("preferences", {"last_observation": {}}),
        ("preferences", {"link_state": "up"}), ("preferences", {"observation": {}}), ("preferences", {"opponent_track": []}),
        ("wifi", {**WIFI, "password": "hunter2"}),
    ]
    for section, data in nested:
        with pytest.raises(ProfileSettingsError) as info:
            store.set_section("live", section, data)
        assert "never persisted" in str(info.value)
    assert not store.path_for("live").exists()
    assert {"opponent_roi", "opponent_track", "tracking_state", "control_mode", "authority", "armed", "motors_running", "wifi_status",
            "link_state", "observation", "last_observation"} <= ps.FORBIDDEN_KEYS


def test_forbidden_keys_in_a_hand_edited_file_are_dropped_on_load(store):
    store.set_section("live", "camera", CAMERA)
    doc = json.loads(store.path_for("live").read_text(encoding="utf-8"))
    doc["sections"]["preferences"] = {"saved_at": None, "data": {"armed": True}}
    store.path_for("live").write_text(json.dumps(doc), encoding="utf-8")
    loaded = store.load("live")
    assert loaded.sections["preferences"] == {} and "preferences" in loaded.load_problem
    assert loaded.sections["camera"] == CAMERA                                       # the good section survives


def test_update_section_cannot_smuggle_a_forbidden_key(store):
    store.set_section("live", "preferences", {"a": 1})
    with pytest.raises(ProfileSettingsError):
        store.update_section("live", "preferences", {"armed": False})
    assert store.get_section("live", "preferences") == {"a": 1}


# ------------------------------------------------------------------------------------------ atomic writes and backup

def test_failure_while_replacing_leaves_the_old_file_intact_and_no_temp_files(tmp_path, clock):
    good = ProfileSettingsStore(tmp_path / "ps", clock=clock)
    good.set_section("live", "wifi", WIFI)
    before = good.path_for("live").read_bytes()

    def broken_replace(src, dst):
        if str(dst).endswith("live.json"):
            raise OSError("disk full (simulated)")
        os.replace(src, dst)

    flaky = ProfileSettingsStore(tmp_path / "ps", clock=clock, replace=broken_replace)
    result = flaky.set_section("live", "wifi", {**WIFI, "host": "1.2.3.4"})
    assert not result.ok and "disk full" in result.error and not bool(result)
    assert good.path_for("live").read_bytes() == before
    assert good.get_section("live", "wifi")["host"] == "192.168.4.1"
    assert not [n for n in dir_files(good) if n.endswith(".tmp")]
    assert good.load("live").load_problem == ""


def test_failure_in_fsync_leaves_the_old_file_intact(tmp_path, clock):
    good = ProfileSettingsStore(tmp_path / "ps", clock=clock)
    good.set_section("live", "wifi", WIFI)
    before = good.path_for("live").read_bytes()

    def broken_fsync(fd):
        raise OSError("device removed (simulated)")

    result = ProfileSettingsStore(tmp_path / "ps", clock=clock, fsync=broken_fsync).set_section("live", "wifi", {"host": "9.9.9.9"})
    assert not result.ok
    assert good.path_for("live").read_bytes() == before
    assert not [n for n in dir_files(good) if n.endswith(".tmp")]


def test_a_crash_half_way_through_the_write_never_shows_a_partial_file(tmp_path, clock):
    """The process dies after writing half of the new file to its temporary name: the real file is still the old one."""
    class Crashing(ProfileSettingsStore):
        crash = False

        def _atomic_write(self, path, payload):
            if self.crash and path.name == "live.json":
                temp = self._temp_name(path)
                temp.write_bytes(payload[: len(payload) // 2])           # half a file, left behind like a power cut would
                raise RuntimeError("power cut (simulated)")
            super()._atomic_write(path, payload)

    store = Crashing(tmp_path / "ps", clock=clock)
    store.set_section("live", "wifi", WIFI)
    old = store.path_for("live").read_bytes()
    store.crash = True
    assert not store.set_section("live", "wifi", {"host": "7.7.7.7"}).ok
    assert store.path_for("live").read_bytes() == old
    assert any(n.endswith(".tmp") for n in dir_files(store))              # the leftover is harmless...
    fresh = ProfileSettingsStore(tmp_path / "ps", clock=clock)
    assert fresh.get_section("live", "wifi")["host"] == "192.168.4.1" and fresh.load_problem("live") == ""      # ...and ignored
    assert fresh.set_section("live", "wifi", {"host": "8.8.8.8"}).ok
    assert fresh.get_section("live", "wifi")["host"] == "8.8.8.8"


def test_previous_good_file_is_kept_as_a_rolling_bak(store):
    store.set_section("live", "wifi", {"host": "1.1.1.1"})
    assert not store.backup_path_for("live").exists()                      # nothing to back up yet
    store.set_section("live", "wifi", {"host": "2.2.2.2"})
    store.set_section("live", "wifi", {"host": "3.3.3.3"})
    bak = json.loads(store.backup_path_for("live").read_text(encoding="utf-8"))
    assert bak["sections"]["wifi"]["data"]["host"] == "2.2.2.2"            # only one generation, the previous one
    assert store.get_section("live", "wifi")["host"] == "3.3.3.3"


def test_truncated_file_falls_back_to_the_bak_and_reports_it(store):
    store.set_section("live", "wifi", {"host": "1.1.1.1"})
    store.set_section("live", "camera", CAMERA)                              # main = wifi+camera, bak = wifi only
    raw = store.path_for("live").read_bytes()
    store.path_for("live").write_bytes(raw[: len(raw) // 2])                 # truncated on disk
    loaded = store.load("live")
    assert loaded.used_backup and loaded.status == "restored_from_backup"
    assert "damaged" in loaded.load_problem and "previous good copy" in loaded.load_problem
    assert loaded.sections["wifi"]["host"] == "1.1.1.1" and loaded.sections["camera"]["index"] is None      # the newest change is lost, said so
    assert store.get_section("live", "wifi")["host"] == "1.1.1.1"


def test_garbage_with_no_bak_falls_back_to_defaults_and_reports_it(store):
    store.directory.mkdir(parents=True)
    store.path_for("live").write_bytes(b"\x00\xff not json at all {{{")
    loaded = store.load("live")
    assert loaded.status == "defaults_after_problem" and not loaded.used_backup
    assert "damaged" in loaded.load_problem and "defaults" in loaded.load_problem
    assert loaded.sections["wifi"]["auto_connect"] is False and loaded.sections["camera"]["width"] is None
    assert store.set_section("live", "wifi", WIFI).ok                        # and it recovers by saving
    assert store.load("live").load_problem == "" and store.get_section("live", "wifi") == WIFI


def test_wrong_version_and_non_object_files_are_problems_not_crashes(store):
    store.directory.mkdir(parents=True)
    for payload in (json.dumps({"version": 99, "profile": "live", "sections": {}}), "[1, 2, 3]", "", "null", json.dumps({"version": 1, "profile": "live"})):
        store.path_for("live").write_text(payload, encoding="utf-8")
        loaded = store.load("live")
        assert loaded.load_problem and loaded.sections["wifi"]["host"] is None
    store.path_for("live").write_text(json.dumps({"version": 99, "profile": "live", "sections": {}}), encoding="utf-8")
    assert "format version 99" in store.load("live").load_problem


def test_a_damaged_main_never_overwrites_a_good_bak(store):
    store.set_section("live", "wifi", {"host": "1.1.1.1"})
    store.set_section("live", "wifi", {"host": "2.2.2.2"})                   # bak = 1.1.1.1
    store.path_for("live").write_bytes(b"garbage")
    assert store.set_section("live", "preferences", {"note": "x"}).ok         # loads from the bak, writes a new main
    bak = json.loads(store.backup_path_for("live").read_text(encoding="utf-8"))
    assert bak["sections"]["wifi"]["data"]["host"] == "1.1.1.1"               # still the last GOOD file, not garbage
    now = store.load("live")
    assert now.load_problem == "" and now.sections["wifi"]["host"] == "1.1.1.1" and now.sections["preferences"] == {"note": "x"}


def test_damaged_main_and_damaged_bak_reports_both(store):
    store.set_section("live", "wifi", WIFI)
    store.set_section("live", "wifi", WIFI)
    store.path_for("live").write_bytes(b"x")
    store.backup_path_for("live").write_bytes(b"y")
    loaded = store.load("live")
    assert loaded.status == "defaults_after_problem" and "no usable backup" in loaded.load_problem


def test_missing_main_with_a_bak_restores_and_says_so(store):
    store.set_section("live", "wifi", WIFI)
    store.set_section("live", "wifi", {**WIFI, "host": "5.5.5.5"})
    store.path_for("live").unlink()
    loaded = store.load("live")
    assert loaded.used_backup and "missing" in loaded.load_problem and loaded.sections["wifi"]["host"] == "192.168.4.1"


def test_one_bad_section_does_not_take_the_others_with_it(store):
    store.set_section("live", "camera", CAMERA)
    store.set_section("live", "wifi", WIFI)
    doc = json.loads(store.path_for("live").read_text(encoding="utf-8"))
    doc["sections"]["wifi"]["data"]["port"] = 99999
    store.path_for("live").write_text(json.dumps(doc), encoding="utf-8")
    loaded = store.load("live")
    assert loaded.sections["camera"] == CAMERA and loaded.sections["wifi"]["host"] is None
    assert "section 'wifi' ignored" in loaded.load_problem


# ------------------------------------------------------------------------------------------ reference frame

def test_reference_frame_round_trip_downscales_to_640_greyscale(store):
    frame = photo(1280, 720, 3)
    assert store.save_reference_frame("live", frame).ok
    ref = store.load_reference_frame("live")
    assert ref is not None and ref.image.dtype == np.uint8 and ref.image.ndim == 2
    assert ref.stored_size == (640, 360) and ref.image.shape == (360, 640)
    assert ref.original_size == (1280, 720) and ref.scale == pytest.approx(0.5)
    expected = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (640, 360), interpolation=cv2.INTER_AREA)
    assert np.array_equal(ref.image, expected)
    assert ref.file.startswith("live.reference-") and ref.file.endswith(".png") and (store.directory / ref.file).exists()
    again = ProfileSettingsStore(store.directory).load_reference_frame("live")            # a brand-new object reads the same thing
    assert np.array_equal(again.image, ref.image) and again.original_size == (1280, 720)


def test_small_frames_are_not_upscaled_and_keep_scale_one(store):
    store.save_reference_frame("live", photo(320, 240, 4))
    ref = store.load_reference_frame("live")
    assert ref.stored_size == ref.original_size == (320, 240) and ref.scale == 1.0


def test_grey_and_bgra_inputs_are_accepted(store):
    grey = cv2.cvtColor(photo(800, 600, 5), cv2.COLOR_BGR2GRAY)
    assert store.save_reference_frame("live", grey).ok
    assert store.load_reference_frame("live").original_size == (800, 600)
    assert store.save_reference_frame("live", cv2.cvtColor(photo(800, 600, 5), cv2.COLOR_BGR2BGRA)).ok
    with pytest.raises(ProfileSettingsError):
        store.save_reference_frame("live", np.zeros((0, 0), np.uint8))
    with pytest.raises(ProfileSettingsError):
        store.save_reference_frame("live", "not an image")                                  # type: ignore[arg-type]


def test_saving_a_new_reference_replaces_the_old_file(store):
    store.save_reference_frame("live", photo(640, 360, 1))
    first = store.load_reference_frame("live").file
    store.save_reference_frame("live", photo(640, 360, 2))
    second = store.load_reference_frame("live").file
    assert first != second
    pngs = [n for n in dir_files(store) if n.endswith(".png")]
    assert pngs == [second]


def test_reference_png_is_written_atomically_and_old_one_survives_a_failed_commit(tmp_path, clock):
    good = ProfileSettingsStore(tmp_path / "ps", clock=clock)
    good.save_reference_frame("live", photo(640, 360, 1))
    old = good.load_reference_frame("live")

    def broken_replace(src, dst):
        if str(dst).endswith("live.json"):
            raise OSError("disk full (simulated)")
        os.replace(src, dst)

    result = ProfileSettingsStore(tmp_path / "ps", clock=clock, replace=broken_replace).save_reference_frame("live", photo(640, 360, 2))
    assert not result.ok
    still = good.load_reference_frame("live")
    assert still is not None and still.file == old.file and np.array_equal(still.image, old.image)      # record and file still agree
    assert good.save_reference_frame("live", photo(640, 360, 2)).ok                                       # and the orphan is cleaned up
    assert len([n for n in dir_files(good) if n.endswith(".png")]) == 1


def test_damaged_or_missing_reference_picture_is_none_with_a_reason(store):
    assert store.load_reference_frame("live") is None and store.reference_problem("live") == ""
    store.save_reference_frame("live", photo(640, 360, 1))
    png = store.directory / store.load_reference_frame("live").file
    data = png.read_bytes()
    png.write_bytes(data[: len(data) // 2])
    assert store.load_reference_frame("live") is None and "damaged" in store.reference_problem("live")
    png.unlink()
    assert store.load_reference_frame("live") is None and "missing" in store.reference_problem("live")


def test_calibration_bundle_is_one_write_with_context_and_reference(store):
    frame = photo(1280, 720, 6)
    assert store.save_calibration_bundle("live", CONTEXT, frame).ok
    context = store.get_section("live", "calibration_context")
    ref = store.load_reference_frame("live")
    assert context["frame_reference"] == ref.file and context["corner_pixels"] == CONTEXT["corner_pixels"]
    assert context["arena_setup_id"] == "abc123"
    # a bundle without a frame drops the old reference (it would not match the new corners)
    assert store.save_calibration_bundle("live", {**CONTEXT, "arena_setup_id": "zzz"}, None).ok
    assert store.load_reference_frame("live") is None
    assert store.get_section("live", "calibration_context")["frame_reference"] is None
    assert not [n for n in dir_files(store) if n.endswith(".png")]


def test_reference_cleanup_never_touches_another_profiles_files(store):
    store.save_reference_frame("a", photo(320, 240, 1))
    store.save_reference_frame("a.reference-abcdef012345", photo(320, 240, 2))        # a pathological but legal profile name
    store.save_reference_frame("a", photo(320, 240, 3))
    assert store.load_reference_frame("a.reference-abcdef012345") is not None
    assert store.load_reference_frame("a") is not None


def test_record_pointing_outside_the_profile_is_refused(store):
    store.save_reference_frame("live", photo(320, 240, 1))
    doc = json.loads(store.path_for("live").read_text(encoding="utf-8"))
    doc["reference_frame"]["file"] = "../../secret.png"
    store.path_for("live").write_text(json.dumps(doc), encoding="utf-8")
    assert store.load_reference_frame("live") is None and "does not belong" in store.reference_problem("live")


def test_nothing_is_written_outside_the_given_directory(tmp_path, clock):
    root = tmp_path / "only_here"
    store = ProfileSettingsStore(root / "ps", clock=clock)
    store.set_section("live", "camera", CAMERA)
    store.save_calibration_bundle("live", CONTEXT, photo(640, 360, 1))
    assert {p.relative_to(tmp_path).parts[0] for p in tmp_path.rglob("*")} == {"only_here"}
