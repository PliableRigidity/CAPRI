"""Quick, quantitative "has the camera moved since the arena was calibrated?" check.

The Command Centre stores a REFERENCE picture (greyscale, at most 640 px wide) when the arena corners are saved.
On the next start the current camera picture is compared with it: local image features (AKAZE by default, ORB as an
option) are matched with a ratio test, a similarity transform (shift + rotation + uniform scale) is fitted with
RANSAC, and the four saved arena corner pixels are carried through that transform. The result says how far each
corner has moved, in pixels AT THE ORIGINAL FRAME SIZE (the reference may be a downscaled copy; the scale is
handled here).

Result status:

  MATCH         largest corner shift <= ``MATCH_MAX_SHIFT_PX`` (4 px). The calibration very probably still applies.
  SHIFTED       larger than that. ``severity`` is "minor" up to ``MINOR_MAX_SHIFT_PX`` (10 px): look at the overlay and
                decide; "major" above it: recalibrate.
  UNVERIFIABLE  the picture cannot answer the question: too dark, blurred, blank or featureless; too few matching
                features or inliers; the matches cover too little of the picture (or sit too far from the corners);
                the picture size differs from the reference (resolution changed); the reference or the corners are
                unusable. It NEVER reports MATCH in these cases: a person has to look at the overlay.

Things that are deliberately robust: robots, people and other moving things in the current picture (RANSAC ignores
features that do not move with the rest), lighting changes (contrast-limited equalisation before detection).

Limits (stated, not hidden): a similarity transform is a good model for a bumped, nudged or slightly tilted camera
but only an approximation for a large perspective change; those are large shifts anyway. A static floor with almost no
texture and a scene that is mostly the arena floor can come back UNVERIFIABLE. The check proves nothing about the
SCALE of the arena (millimetres), only that the picture has not moved relative to the reference.

Results are deterministic: no randomness besides OpenCV's own, which is seeded (``cv2.setRNGSeed``) on every call.

Synthetic-image tests only. Nothing here opens a camera. It is not a hardware verification.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import cv2
import numpy as np

MATCH, SHIFTED, UNVERIFIABLE = "MATCH", "SHIFTED", "UNVERIFIABLE"
STATUSES = (MATCH, SHIFTED, UNVERIFIABLE)

# ---- thresholds (pixels are at the ORIGINAL frame size)
MATCH_MAX_SHIFT_PX = 4.0            # largest corner shift still called MATCH
MINOR_MAX_SHIFT_PX = 10.0           # above MATCH and up to this: severity "minor"; above: "major" (recalibrate)
MIN_MEAN_BRIGHTNESS = 20.0          # 0-255; darker pictures are too dark to match
MIN_CONTRAST_STD = 6.0              # 0-255 standard deviation; flatter pictures are blank or covered
MIN_FEATURES = 60                   # per picture (reference and current)
MIN_GOOD_MATCHES = 25               # after the ratio test
MIN_INLIERS = 20                    # agreeing with the fitted transform
MIN_INLIER_RATIO = 0.30             # inliers / good matches
MIN_INLIER_SPREAD = 0.35            # inliers must span this fraction of the picture width AND height
MAX_CORNER_TO_INLIER_FRACTION = 0.35    # every corner needs an inlier feature within this fraction of the picture diagonal
RATIO_TEST = 0.75
RANSAC_THRESHOLD_PX = 2.5           # at working (reference) resolution
MAX_FEATURES_KEPT = 1500
RNG_SEED = 0


@dataclass(frozen=True)
class AlignmentResult:
    status: str                                   # MATCH, SHIFTED or UNVERIFIABLE
    max_corner_shift_px: float | None = None      # original-frame pixels; None when UNVERIFIABLE
    mean_corner_shift_px: float | None = None
    rotation_deg: float | None = None             # fitted rotation, + is counter-clockwise in image coordinates (y down: clockwise on screen)
    scale_ratio: float | None = None              # current / reference apparent scale (>1: camera moved closer / zoomed in)
    inliers: int = 0
    features: int = 0                             # the smaller of the two pictures' feature counts
    reason: str = ""                              # plain language: what was found, or why it cannot be judged
    hint: str = ""                                # what to do next
    severity: str | None = None                   # "none", "minor", "major" for MATCH/SHIFTED; None when UNVERIFIABLE
    per_corner_shift_px: tuple[float, ...] = ()
    translation_px: tuple[float, float] | None = None     # mean corner displacement (dx, dy), original-frame pixels
    matches: int = 0                              # good matches after the ratio test
    features_reference: int = 0
    features_current: int = 0
    method: str = ""

    @property
    def verified(self) -> bool:
        """True only when a real comparison was made (MATCH or SHIFTED)."""
        return self.status in (MATCH, SHIFTED)

    @property
    def major(self) -> bool:
        return self.status == SHIFTED and self.severity == "major"

    def to_dict(self) -> dict[str, Any]:
        return {"status": self.status, "severity": self.severity, "max_corner_shift_px": self.max_corner_shift_px,
                "mean_corner_shift_px": self.mean_corner_shift_px, "rotation_deg": self.rotation_deg,
                "scale_ratio": self.scale_ratio, "inliers": self.inliers, "features": self.features, "matches": self.matches,
                "per_corner_shift_px": list(self.per_corner_shift_px),
                "translation_px": None if self.translation_px is None else list(self.translation_px),
                "reason": self.reason, "hint": self.hint, "method": self.method}

    def summary_line(self) -> str:
        if not self.verified:
            return f"alignment could not be checked: {self.reason}"
        return (f"largest corner shift {self.max_corner_shift_px:.1f} px (mean {self.mean_corner_shift_px:.1f}), rotation "
                f"{self.rotation_deg:+.2f} deg, scale x{self.scale_ratio:.3f}, {self.inliers} matched features")


# ---------------------------------------------------------------------------------------------- helpers

def _unverifiable(reason: str, hint: str, **info: Any) -> AlignmentResult:
    return AlignmentResult(UNVERIFIABLE, reason=reason, hint=hint, **info)


_VISUAL = "Look at the arena overlay by eye and press Alignment OK only if its lines sit on the arena edges."


def _to_grey(image: Any) -> np.ndarray | None:
    if not isinstance(image, np.ndarray) or image.ndim not in (2, 3) or image.size == 0:
        return None
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    if image.ndim == 3:
        if image.shape[2] == 1:
            return image[:, :, 0].copy()
        return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY if image.shape[2] == 4 else cv2.COLOR_BGR2GRAY)
    return image


def _corner_array(corner_pixels: Any) -> np.ndarray | None:
    if corner_pixels is None:
        return None
    if isinstance(corner_pixels, Mapping):
        values = [corner_pixels[key] for key in sorted(corner_pixels)]
    else:
        values = list(corner_pixels)
    try:
        array = np.asarray(values, dtype=float).reshape(-1, 2)
    except (TypeError, ValueError):
        return None
    if len(array) == 0 or not np.isfinite(array).all():
        return None
    return array


def _features(grey: np.ndarray, detector: str) -> tuple[list[cv2.KeyPoint], np.ndarray | None]:
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    prepared = clahe.apply(grey)
    if detector == "orb":
        engine = cv2.ORB_create(nfeatures=MAX_FEATURES_KEPT, scaleFactor=1.2, nlevels=8, fastThreshold=10)
        keypoints, descriptors = engine.detectAndCompute(prepared, None)
    else:
        engine = cv2.AKAZE_create()
        keypoints, descriptors = engine.detectAndCompute(prepared, None)
        if descriptors is not None and len(keypoints) > MAX_FEATURES_KEPT:
            order = sorted(range(len(keypoints)), key=lambda i: (-keypoints[i].response, i))[:MAX_FEATURES_KEPT]
            keypoints = [keypoints[i] for i in order]
            descriptors = descriptors[order]
    return list(keypoints), descriptors


def _direction_words(dx: float, dy: float) -> str:
    parts = []
    if abs(dx) >= 1.0:
        parts.append(f"{abs(dx):.0f} px {'right' if dx > 0 else 'left'}")
    if abs(dy) >= 1.0:
        parts.append(f"{abs(dy):.0f} px {'down' if dy > 0 else 'up'}")
    return " and ".join(parts)


# ---------------------------------------------------------------------------------------------- the check

def check_alignment(reference_gray: np.ndarray, current: np.ndarray, corner_pixels: Any, *,
                    reference_original_size: tuple[int, int] | None = None, reference_scale: float | None = None,
                    detector: str = "akaze") -> AlignmentResult:
    """Compare ``current`` with the stored reference and say how far the saved arena corners have moved.

    ``reference_gray``  the stored reference picture (greyscale or colour), possibly a downscaled copy.
    ``current``         the camera picture now, BGR or grey, at the FULL frame size.
    ``corner_pixels``   the saved arena corners in ORIGINAL-frame pixels: a list of (x, y), or a {name: (x, y)} mapping
                        (taken in sorted name order).
    ``reference_original_size``  (width, height) of the frame the reference was made from; or give ``reference_scale``
                        (= stored width / original width). With neither, the reference is taken to be full size.
    ``detector``        "akaze" (default, more precise) or "orb" (faster).

    Returns an ``AlignmentResult``; never raises for bad pictures (that is UNVERIFIABLE with a reason).
    A different picture size from the original is UNVERIFIABLE: the camera's resolution changed.
    """
    cv2.setRNGSeed(RNG_SEED)
    method = detector if detector in ("akaze", "orb") else "akaze"
    ref = _to_grey(reference_gray)
    cur = _to_grey(current)
    if ref is None:
        return _unverifiable("there is no usable reference picture", "Calibrate the arena again, or check the alignment by eye. " + _VISUAL, method=method)
    if cur is None:
        return _unverifiable("the camera has not delivered a usable picture", "Wait for the camera picture, then check again.", method=method)
    corners = _corner_array(corner_pixels)
    if corners is None:
        return _unverifiable("the saved arena corners are missing", "Calibrate the arena (click the four corners).", method=method)

    ref_h, ref_w = ref.shape
    if reference_original_size is not None:
        orig_w, orig_h = int(reference_original_size[0]), int(reference_original_size[1])
    elif reference_scale:
        orig_w, orig_h = int(round(ref_w / reference_scale)), int(round(ref_h / reference_scale))
    else:
        orig_w, orig_h = ref_w, ref_h
    if orig_w <= 0 or orig_h <= 0 or abs(ref_w / orig_w - ref_h / orig_h) > 0.02 * max(ref_w / orig_w, 1e-9):
        return _unverifiable("the reference picture record is inconsistent (its size does not fit the original size)",
                             "Calibrate the arena again to store a new reference picture. " + _VISUAL, method=method)
    scale_x, scale_y = ref_w / orig_w, ref_h / orig_h
    cur_h, cur_w = cur.shape
    if (cur_w, cur_h) != (orig_w, orig_h):
        return _unverifiable(f"the camera picture is {cur_w} x {cur_h} but the reference is {orig_w} x {orig_h}: the camera resolution changed",
                             "Set the camera back to the calibrated resolution, or calibrate the arena again.", method=method)

    for label, picture in (("camera picture", cur), ("reference picture", ref)):
        if float(picture.mean()) < MIN_MEAN_BRIGHTNESS:
            return _unverifiable(f"the {label} is too dark to match", "Turn the lights on (or open the camera shutter), then check again. " + _VISUAL, method=method)
        if float(picture.std()) < MIN_CONTRAST_STD:
            return _unverifiable(f"the {label} is blank or has almost no contrast", "Check the lens is uncovered and the arena is lit. " + _VISUAL, method=method)

    working = cur if (cur_w, cur_h) == (ref_w, ref_h) else cv2.resize(cur, (ref_w, ref_h), interpolation=cv2.INTER_AREA)
    kp_ref, des_ref = _features(ref, method)
    kp_cur, des_cur = _features(working, method)
    features = min(len(kp_ref), len(kp_cur))
    base = dict(features=features, features_reference=len(kp_ref), features_current=len(kp_cur), method=method)
    if des_ref is None or des_cur is None or features < MIN_FEATURES:
        return _unverifiable(f"too few image features to compare ({len(kp_ref)} in the reference, {len(kp_cur)} now; need {MIN_FEATURES}): "
                             "the picture is dark, blurred, or mostly plain floor",
                             "Improve the lighting or focus, then check again. " + _VISUAL, **base)

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    good = []
    for pair in matcher.knnMatch(des_ref, des_cur, k=2):
        if len(pair) == 2 and pair[0].distance < RATIO_TEST * pair[1].distance:
            good.append(pair[0])
    base["matches"] = len(good)
    if len(good) < MIN_GOOD_MATCHES:
        return _unverifiable(f"only {len(good)} features match the reference (need {MIN_GOOD_MATCHES}): the scene looks different, or the picture is "
                             "blurred or badly lit", "Check the camera still looks at the arena. " + _VISUAL, **base)

    src = np.float32([kp_ref[m.queryIdx].pt for m in good])
    dst = np.float32([kp_cur[m.trainIdx].pt for m in good])
    matrix, mask = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=RANSAC_THRESHOLD_PX,
                                               maxIters=3000, confidence=0.995, refineIters=15)
    inliers = 0 if mask is None else int(mask.sum())
    base["inliers"] = inliers
    if matrix is None or inliers < MIN_INLIERS or inliers < MIN_INLIER_RATIO * len(good):
        return _unverifiable(f"the matched features do not agree on one consistent movement ({inliers} of {len(good)} agree; need {MIN_INLIERS} "
                             f"and {100 * MIN_INLIER_RATIO:.0f}%)", "Check the camera still looks at the arena. " + _VISUAL, **base)

    inlier_points = src[mask.ravel() > 0]
    spread_x = float(inlier_points[:, 0].max() - inlier_points[:, 0].min()) / ref_w
    spread_y = float(inlier_points[:, 1].max() - inlier_points[:, 1].min()) / ref_h
    if spread_x < MIN_INLIER_SPREAD or spread_y < MIN_INLIER_SPREAD:
        return _unverifiable(f"the matched features cover too little of the picture ({100 * spread_x:.0f}% x {100 * spread_y:.0f}%): "
                             "a shift measured there cannot be carried to the arena corners", "Check the alignment by eye. " + _VISUAL, **base)

    corners_ref = corners * np.array([scale_x, scale_y])
    diagonal = math.hypot(ref_w, ref_h)
    nearest = np.min(np.linalg.norm(corners_ref[:, None, :] - inlier_points[None, :, :], axis=2), axis=1)
    if float(nearest.max()) > MAX_CORNER_TO_INLIER_FRACTION * diagonal:
        return _unverifiable("a saved arena corner is far from every matched feature, so its movement cannot be measured reliably",
                             "Check the alignment by eye. " + _VISUAL, **base)

    moved = corners_ref @ matrix[:, :2].T + matrix[:, 2]
    shift = (moved - corners_ref) / np.array([scale_x, scale_y])          # back to original-frame pixels
    lengths = np.linalg.norm(shift, axis=1)
    maximum, mean = float(lengths.max()), float(lengths.mean())
    scale = float(math.hypot(matrix[0, 0], matrix[1, 0]))
    rotation = float(math.degrees(math.atan2(matrix[1, 0], matrix[0, 0])))
    dx, dy = float(shift[:, 0].mean()), float(shift[:, 1].mean())
    common = dict(max_corner_shift_px=maximum, mean_corner_shift_px=mean, rotation_deg=rotation, scale_ratio=scale,
                  per_corner_shift_px=tuple(float(v) for v in lengths), translation_px=(dx, dy), **base)
    if maximum <= MATCH_MAX_SHIFT_PX:
        return AlignmentResult(MATCH, reason=f"the arena corners are within {maximum:.1f} px of where they were calibrated (limit {MATCH_MAX_SHIFT_PX:g} px)",
                               hint="Look at the overlay to confirm, then press Alignment OK.", severity="none", **common)
    moved_words = _direction_words(dx, dy)
    detail = (f"the picture has moved relative to the calibration: the arena corners are up to {maximum:.0f} px off"
              + (f" (on average {moved_words})" if moved_words else "")
              + (f", rotated {rotation:+.1f} deg" if abs(rotation) >= 0.3 else "")
              + (f", scale x{scale:.3f}" if abs(scale - 1.0) >= 0.005 else ""))
    if maximum <= MINOR_MAX_SHIFT_PX:
        return AlignmentResult(SHIFTED, reason=detail, severity="minor", **common,
                               hint="A small shift. Check the overlay by eye: if its lines are visibly off the arena edges, recalibrate; otherwise press Alignment OK.")
    return AlignmentResult(SHIFTED, reason=detail, severity="major", **common,
                           hint="The camera has moved. Recalibrate the arena corners before using positions.")
