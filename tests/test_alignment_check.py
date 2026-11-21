"""Alignment check against a stored reference picture: synthetic images only (no camera, no files).

The scene is a procedurally drawn textured "room" (random blobs, bars and lines). Camera movements are applied to
the full scene with ``cv2.warpAffine`` so nothing but the movement changes between the reference and the current
picture. SOFTWARE TESTS: they show the algorithm behaves on synthetic data, not that a real camera, floor or lighting
will give the same numbers.
"""
from __future__ import annotations

import time

import cv2
import numpy as np
import pytest

from control_center.backend import alignment_check as A

SIZE = (640, 360)                       # width, height of the "camera frame"
CORNERS = [(80.0, 60.0), (560.0, 60.0), (560.0, 300.0), (80.0, 300.0)]
ORIGIN = np.array([[1.0, 0.0, -300.0], [0.0, 1.0, -250.0]])    # where the reference view sits inside the world


def make_world(seed: int = 1, size: tuple[int, int] = (1400, 1000)) -> np.ndarray:
    rng = np.random.RandomState(seed)
    w, h = size
    img = np.full((h, w), 120, np.uint8)
    for _ in range(int(w * h / 1500)):
        kind = rng.randint(0, 3)
        x, y = rng.randint(0, w), rng.randint(0, h)
        value = int(rng.randint(10, 245))
        if kind == 0:
            cv2.circle(img, (x, y), int(rng.randint(2, 14)), value, -1)
        elif kind == 1:
            cv2.rectangle(img, (x, y), (x + int(rng.randint(3, 28)), y + int(rng.randint(3, 28))), value, -1)
        else:
            cv2.line(img, (x, y), (x + int(rng.randint(-40, 40)), y + int(rng.randint(-40, 40))), value, int(rng.randint(1, 4)))
    return cv2.GaussianBlur(img, (3, 3), 0.8)


def compose(delta: np.ndarray, base: np.ndarray = ORIGIN) -> np.ndarray:
    """World -> current view = delta (reference view -> current view) after base (world -> reference view)."""
    a = np.vstack([base, [0, 0, 1]])
    d = np.vstack([delta, [0, 0, 1]])
    return (d @ a)[:2]


def view(world: np.ndarray, matrix: np.ndarray, size: tuple[int, int] = SIZE, noise: float = 1.2, seed: int = 5) -> np.ndarray:
    img = cv2.warpAffine(world, matrix, size, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    sensor = np.random.RandomState(seed).normal(0.0, noise, img.shape)
    return np.clip(img.astype(float) + sensor, 0, 255).astype(np.uint8)


def shift_matrix(dx: float, dy: float) -> np.ndarray:
    return np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]])


def true_max_shift(delta: np.ndarray, corners=CORNERS) -> float:
    pts = np.array(corners)
    moved = pts @ delta[:, :2].T + delta[:, 2]
    return float(np.linalg.norm(moved - pts, axis=1).max())


@pytest.fixture(scope="module")
def world() -> np.ndarray:
    return make_world()


@pytest.fixture(scope="module")
def reference(world) -> np.ndarray:
    return view(world, ORIGIN, seed=1)


def run(reference, current, **kw):
    return A.check_alignment(reference, current, CORNERS, **kw)


def test_identical_frame_matches(reference):
    result = run(reference, reference.copy())
    assert result.status == A.MATCH
    assert result.severity == "none"
    assert result.max_corner_shift_px < 0.5
    assert result.inliers >= A.MIN_INLIERS and result.features >= A.MIN_FEATURES
    assert abs(result.rotation_deg) < 0.05 and abs(result.scale_ratio - 1.0) < 0.001
    assert result.verified and not result.major


def test_new_noise_realisation_still_matches(world, reference):
    current = view(world, ORIGIN, seed=99)               # same scene, different sensor noise
    result = run(reference, current)
    assert result.status == A.MATCH and result.max_corner_shift_px < 1.0


def test_two_pixel_translation_is_a_match_and_is_measured(world, reference):
    delta = shift_matrix(2.0, 0.0)
    result = run(reference, view(world, compose(delta), seed=2))
    assert result.status == A.MATCH
    assert abs(result.max_corner_shift_px - 2.0) < 1.0
    assert result.translation_px[0] == pytest.approx(2.0, abs=1.0)


def test_fifteen_pixel_translation_is_a_major_shift(world, reference):
    delta = shift_matrix(15.0, 0.0)
    result = run(reference, view(world, compose(delta), seed=3))
    assert result.status == A.SHIFTED and result.severity == "major" and result.major
    assert result.max_corner_shift_px == pytest.approx(15.0, abs=1.5)
    assert "right" in result.reason and "Recalibrate" in result.hint


def test_seven_pixel_diagonal_shift_is_minor(world, reference):
    delta = shift_matrix(-5.0, 5.0)                       # ~7.1 px
    result = run(reference, view(world, compose(delta), seed=4))
    assert result.status == A.SHIFTED and result.severity == "minor" and not result.major
    assert result.max_corner_shift_px == pytest.approx(7.07, abs=1.5)


def test_three_degree_rotation_is_shifted_and_measured(world, reference):
    delta = cv2.getRotationMatrix2D((320.0, 180.0), 3.0, 1.0)
    result = run(reference, view(world, compose(delta), seed=6))
    assert result.status == A.SHIFTED
    assert abs(abs(result.rotation_deg) - 3.0) < 0.5
    assert result.max_corner_shift_px == pytest.approx(true_max_shift(delta), abs=2.5)


def test_camera_moved_closer_is_shifted(world, reference):
    delta = cv2.getRotationMatrix2D((320.0, 180.0), 0.0, 1.06)
    result = run(reference, view(world, compose(delta), seed=7))
    assert result.status == A.SHIFTED
    assert result.scale_ratio == pytest.approx(1.06, abs=0.01)
    assert result.max_corner_shift_px == pytest.approx(true_max_shift(delta), abs=2.5)


def test_robots_and_people_in_the_picture_do_not_hide_a_match(world, reference):
    current = view(world, ORIGIN, seed=8)
    cv2.rectangle(current, (120, 100), (300, 250), 15, -1)              # a person-sized dark block
    cv2.rectangle(current, (400, 150), (560, 300), 235, -1)              # a pale block
    cv2.circle(current, (330, 60), 45, 60, -1)                           # a robot seen from above
    covered = float((current != view(world, ORIGIN, seed=8)).mean())
    assert 0.2 < covered < 0.4, covered                                   # about a quarter of the picture is occluded
    result = run(reference, current)
    assert result.status == A.MATCH, result.reason
    assert result.max_corner_shift_px < 2.0


def test_occlusion_does_not_hide_a_real_shift(world, reference):
    delta = shift_matrix(12.0, 4.0)
    current = view(world, compose(delta), seed=9)
    cv2.rectangle(current, (120, 100), (300, 250), 15, -1)
    cv2.rectangle(current, (400, 150), (560, 300), 235, -1)
    result = run(reference, current)
    assert result.status == A.SHIFTED and result.major


def test_dark_frame_is_unverifiable(reference):
    result = run(reference, (reference * 0.05).astype(np.uint8))
    assert result.status == A.UNVERIFIABLE
    assert "dark" in result.reason
    assert result.max_corner_shift_px is None and result.severity is None


def test_heavily_blurred_frame_is_unverifiable(reference):
    result = run(reference, cv2.GaussianBlur(reference, (0, 0), 9.0))
    assert result.status == A.UNVERIFIABLE
    assert result.max_corner_shift_px is None


def test_blank_frame_is_unverifiable(reference):
    result = run(reference, np.full((SIZE[1], SIZE[0]), 120, np.uint8))
    assert result.status == A.UNVERIFIABLE
    assert "blank" in result.reason or "contrast" in result.reason


def test_plain_floor_with_one_marking_is_unverifiable():
    plain = np.full((SIZE[1], SIZE[0]), 110, np.uint8)
    plain = cv2.GaussianBlur(plain + np.random.RandomState(0).randint(0, 4, plain.shape).astype(np.uint8), (5, 5), 2)
    cv2.rectangle(plain, (200, 120), (300, 200), 200, 2)
    result = A.check_alignment(plain, plain.copy(), CORNERS)
    assert result.status == A.UNVERIFIABLE                              # never a MATCH from a featureless floor


def test_different_image_size_is_unverifiable_with_a_resolution_reason(reference):
    small = cv2.resize(reference, (320, 180), interpolation=cv2.INTER_AREA)
    result = run(reference, small)
    assert result.status == A.UNVERIFIABLE
    assert "resolution" in result.reason and "320 x 180" in result.reason


def test_unrelated_scene_is_unverifiable(reference):
    other = view(make_world(seed=77), ORIGIN, seed=10)
    result = run(reference, other)
    assert result.status == A.UNVERIFIABLE


def test_matches_are_concentrated_in_a_corner_is_unverifiable(world):
    """Texture only in one small patch of the picture: a shift measured there says nothing about the far corners."""
    ref = np.full((SIZE[1], SIZE[0]), 110, np.uint8)
    patch = view(world, ORIGIN, seed=1)[0:120, 0:160]
    ref[0:120, 0:160] = patch
    ref = cv2.GaussianBlur(ref, (3, 3), 0.5)
    result = A.check_alignment(ref, ref.copy(), CORNERS)
    assert result.status == A.UNVERIFIABLE


def test_downscaled_reference_gives_original_pixel_shifts():
    big_world = make_world(seed=3, size=(2400, 1700))
    origin = np.array([[1.0, 0.0, -500.0], [0.0, 1.0, -400.0]])
    full_size = (1280, 720)
    reference_full = view(big_world, origin, size=full_size, seed=1)
    stored = cv2.resize(reference_full, SIZE, interpolation=cv2.INTER_AREA)         # what the store keeps
    corners = [(160.0, 120.0), (1120.0, 120.0), (1120.0, 600.0), (160.0, 600.0)]       # original-frame pixels
    same = A.check_alignment(stored, reference_full.copy(), corners, reference_original_size=full_size)
    assert same.status == A.MATCH and same.max_corner_shift_px < 1.0
    moved = view(big_world, compose(shift_matrix(14.0, 0.0), origin), size=full_size, seed=2)
    result = A.check_alignment(stored, moved, corners, reference_scale=0.5)             # scale instead of size
    assert result.status == A.SHIFTED and result.major
    assert result.max_corner_shift_px == pytest.approx(14.0, abs=2.0)                   # ORIGINAL pixels, not 7
    wrong_size = A.check_alignment(stored, cv2.resize(moved, (960, 540)), corners, reference_original_size=full_size)
    assert wrong_size.status == A.UNVERIFIABLE and "resolution" in wrong_size.reason


def test_colour_and_grey_inputs_agree(reference):
    colour = cv2.cvtColor(reference, cv2.COLOR_GRAY2BGR)
    assert run(reference, colour).status == A.MATCH
    assert run(colour, reference).status == A.MATCH


def test_bad_inputs_are_unverifiable_not_exceptions(reference):
    assert A.check_alignment(None, reference, CORNERS).status == A.UNVERIFIABLE
    assert A.check_alignment(reference, None, CORNERS).status == A.UNVERIFIABLE
    assert A.check_alignment(reference, reference, None).status == A.UNVERIFIABLE
    assert A.check_alignment(reference, reference, [("a", "b")]).status == A.UNVERIFIABLE
    assert A.check_alignment(reference, reference, []).status == A.UNVERIFIABLE


def test_corner_mapping_input_is_accepted(reference):
    named = {"arena_corner_c1": CORNERS[0], "arena_corner_c2": CORNERS[1], "arena_corner_c3": CORNERS[2], "arena_corner_c4": CORNERS[3]}
    assert A.check_alignment(reference, reference.copy(), named).status == A.MATCH


def test_results_are_deterministic(world, reference):
    current = view(world, compose(shift_matrix(6.0, -3.0)), seed=11)
    first, second = run(reference, current), run(reference, current)
    assert first == second
    cv2.setRNGSeed(12345)                                                 # outside state must not change the answer
    assert run(reference, current) == first


def test_orb_option_also_works(world, reference):
    assert run(reference, reference.copy(), detector="orb").status == A.MATCH
    result = run(reference, view(world, compose(shift_matrix(15.0, 0.0)), seed=3), detector="orb")
    assert result.status == A.SHIFTED and result.major and result.method == "orb"


def test_thresholds_are_the_documented_ones():
    assert A.MATCH_MAX_SHIFT_PX == 4.0 and A.MINOR_MAX_SHIFT_PX == 10.0


def test_the_whole_suite_of_checks_is_fast(world, reference):
    started = time.perf_counter()
    for seed in range(3):
        run(reference, view(world, compose(shift_matrix(3.0, 0.0)), seed=20 + seed))
    assert (time.perf_counter() - started) / 3 < 1.0
