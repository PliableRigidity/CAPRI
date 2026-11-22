# ArUco tracking audit, 21 Nov 2026

Scope: the Command Centre's live own-robot tracking (dictionary, marker IDs and layout, fusion, states, what the screens show).
Evidence: **SYNTHETIC frames only** (rendered markers, known poses, `control_center/tools/aruco_synth.py`) plus the user's own saved
real-camera outputs in `artifacts/marker-survey` and `artifacts/live-track`, which were read, not regenerated. **No camera, serial
port or socket was opened, and nothing here is hardware verification.**

## 1. Dictionary: DICT_4X4_50 end to end

The Command Centre live path was **already** DICT_4X4_50. The mismatch was in the legacy window only, and it was a trap rather than a bug in the live path.

| Place | Before | Now |
|---|---|---|
| `config/arena_overlay.yaml` `bot.dictionary` | `DICT_4X4_50` | same; **the one key** (`control_center/backend/marker_settings.py` documents the chain) |
| `config/home_test_overlay.yaml` `bot.dictionary` | `DICT_4X4_50`, comment claimed 36h11 | same value, comment fixed, a test fails if the two files ever differ |
| `config/control_center.yaml` `april_tag_family`, `april_tag.family` | `DICT_APRILTAG_36h11` | **removed**; `camera_worker` (legacy window) asks `legacy_dictionary()`: the overlay key wins, a stale legacy key only logs a warning |
| `backend/vision.py` `AprilTagTracker` | any unknown family **silently became 36h11** | any `DICT_*` name works, an unknown name raises |
| `april_test.py` (standalone script; since 22 November `scripts/aruco_camera_check.py`) | hard-coded 36h11 | project dictionary (not run: it opens a camera) |
| detectors built in `overlay_core.MarkerDetector`, `marker_survey`, `AprilTagTracker` | three separate `ArucoDetector(...)` constructions | all via `marker_settings.build_detector()`; a test scans the source so no other place can build one |
| `profile_settings` `markers.aruco_dictionary` | a saved slot nothing reads | unchanged; a test fails if any code starts reading it |

Real-camera evidence that 4X4_50 is right: both marker surveys (294 and 296 frames) decoded ids only in `DICT_4X4_50`/`4X4_100`, **zero** in 36h11, 25h9, 16h5, 5x5, 6x6.
The survey tool now always tries the project dictionary first and prints `MISMATCH` if another one explains the frames.

## 2. IDs 0-8 and the layout

Defined in `config/arena_overlay.yaml` `bot.markers` (read by `overlay_core.parse_config`), nothing else:

* ids **0-3 `face: top`**, ids **4-7 `face: bottom`**, offsets `forward_m = ±0.035`, `left_m = ±0.030`, `yaw_deg = 0`, size 0.040 m, height 0.034 m: **all PLACEHOLDERS, `bot.measured: false`**. No measured mounting data exists in the repository or `artifacts/`.
* **id 8 (rear, `face: rear`) is not in the live config.** It exists only in the test fixture (`track_fixture.REAR_MARKER`, invented numbers). The real camera does decode an id 8, so on screen it is an `UNKNOWN ID` and is ignored, never used. A commented template is in the config; adding it needs real measurements.
* Which face is up is decided only by which ids decode (top ids = upright, bottom ids = inverted). Top and bottom ids visible together give no pose (`conflict`).
* ids outside the list (e.g. 17, 33, 49, valid in the dictionary) are listed as ignored and never used.

## 3. What the tracker does (`control_center/tools/chassis_tracker.py`)

* **Per frame**: each configured marker gives its own chassis pose; the largest mutually consistent set (20 mm, 10 deg, the `pose:` keys) is fused with one rigid fit; the rest are rejected with a reason. A repeated id is rejected (not trusted), a marker whose apparent size is not within 30 % of its configured size is rejected, a marker just outvoted by two or more others is not trusted alone for 3 s, a tie is broken only by the track's prediction.
* **Track**: a motion gate (2.5 m/s, 25 rad/s plus a noise floor) refuses jumps; three agreeing frames at a new place restart the track.
* **States** (a pure function of the query time, so they advance even if frames stop): MEASURED (detection in the newest frame, age <= 50 ms) -> PREDICTED (<= 0.2 s, constant velocity) -> STALE (<= 1.0 s, last measured pose kept **for display only, not valid**) -> LOST (no pose). Every state carries `age_s`.
* **Flipped**: not a separate state; `orientation` (`upright` / `inverted` / `unknown`) and the new `flipped` (True / False / None), held while PREDICTED/STALE, None when LOST. The rear marker (when configured) is read for the orientation the top/bottom markers decide.

## 4. What changed

1. `backend/marker_settings.py` (new): the dictionary, strict name lookup, `legacy_dictionary`, `dictionary_status`, and the named detector parameter set `live_v1` used by every detector.
2. **`live_v1` parameters** (OpenCV defaults except): `adaptiveThreshWinSizeMax 53` (23), `perspectiveRemovePixelPerCell 8` (4), `perspectiveRemoveIgnoredMarginPerCell 0.25` (0.13), `maxErroneousBitsInBorderRate 0.5` (0.35), sub-pixel corners (already used). Cost: +3 ms per 1280x720 frame (9.3 -> 12.6 ms on this PC).
3. **Single-marker start check**: `TrackerSettings.start_confirm_frames` (default 1 = unchanged; the live viewers use 2 via `live_tracker_settings`). A track that would start from ONE marker in ONE frame waits for a second agreeing frame; two or more agreeing markers start at once. Reason: one decoded marker is the weakest evidence and a false decode of an id inside the layout looks identical (with the pre-audit detector 5 of 1470 wrong-dictionary markers were misread, two of them as id 5, inside the layout).
4. `ChassisEstimate.flipped`; a size rejection while `bot.measured` is false now says the size is a PLACEHOLDER and which key to set.
5. **TRACKING screen** (`screens/tracking.py`, `backend/tracking_model.py`): the own-robot panel now shows state, **age**, position, heading, orientation (`UPSIDE DOWN / FLIPPED` spelled out), marker ids in the newest frame, ids the pose rests on, ignored ids, rejections, confidence. The panel and the map marker are **aged on the screen's own clock**: if frames stop, a MEASURED pose turns STALE (display only) then LOST on screen, where before the last report stayed on screen as MEASURED until the 2 s "vision stalled" banner. New banner when markers are decoded but all rejected for size. The dictionary line says when the dictionary in use is not the project's.
6. `backend/live_vision.py`: `VisionSample.own_track` (the tracker report, additive; `pose` is unchanged) for whoever shows it on OPERATE. **OPERATE (`screens/operate.py`, not edited, another engineer's) still shows only the single-frame pose with "last known" amber after 0.5 s: no state, age, ids or confidence.**
7. `marker_survey`: project dictionary always tried first and compared, ids outside 0-8 flagged.

## 5. Findings from the REAL traces (not synthetic)

`artifacts/live-track/*` (home_test profile, camera 1, 28-30 fps): ids 0-8 decode, but the tracker was **LOST on 96-100 % of frames** (the first two sessions decoded no marker at all, cause not recorded) because in the other three it rejected **every decoded marker** as `measures 9-16 mm but is configured as 40 mm`. The size placeholder (`bot.marker_size_m: 0.040`) is far from the printed size, so a stale size, not the dictionary or the detector, is what stopped a pose. (The 9-16 mm is the tracker's own reading through an uncertain home-test calibration; it is **not** a measurement of the marker. Measure it.)
Inside the periods where markers were being shown, any layout marker was decoded in only 43-61 % of frames, each top id in 15-43 %, gaps median 3-4 frames, 90th percentile 16-18 frames (0.55-0.6 s), longest 28 frames. That is real intermittency, longer than the 0.2 s prediction hold, so the pose goes STALE during the long gaps. **The cause is not known from these files** (no frames were kept): hand-held tilt, glare, motion blur from a long exposure in the dim room (the survey photo is dim), and a small apparent size (35-41 px side) are all candidates. Also seen: false decodes `17`, `16`, `34-45` at 14-17 px (all outside the layout, flagged rare).

## 6. SYNTHETIC results (`tests/test_aruco_tracking_conditions.py`; none is a real-camera figure)

* **Detection vs motion streak** (36 px marker, 120 markers per cell, `live_v1` vs the pre-audit sub-pixel-only set): sharp 1.00 / 1.00; blur 6 px 1.00 / 1.00; **blur 10 px 0.78 / 0.62**; blur 12 px 0.31 / 0.11; 26 px marker with 8 px blur 0.50 / 0.33; 20 px marker with 6 px blur 0.61 / 0.48. Wrong ids 0 in both; extra decodes at heavy blur come with corners often >3 px off. The detector gain is real but modest: **blur beyond about a quarter of the marker's side still loses most detections, whatever the parameters**. Fixing exposure, lighting and printing larger markers (if the layout allows) is worth more than any parameter.
* **Wrong dictionaries** (5x5, 6x6, 7x7, ARUCO_ORIGINAL, 36h11, 25h9, 16h5; 1470 markers): 2 misread (ids 32, outside the layout) with `live_v1`, 5 with the old set (two as id 5, inside the layout). Rejected almost always, not always: the single-marker start check is the second line of defence. Marker-free textured floor: 0 false decodes in 150 frames.
* **Partial occlusion / glare**: never a wrong id; but a marker with 10-15 % of its width covered can still decode with corners pulled 4-8 px, i.e. 1.5-5 mm and 2-3 deg of pose error from that marker alone (one marker has nothing to be outvoted by). With other markers present the fused error stayed under 1 mm / 0.5 deg.
* **1, 2, 3, 4 markers**: worst position error 1.25 / 0.30 / 0.10 / 0.06 mm, heading 1.6 / 0.6 / 0.5 / 0.5 deg (ideal pinhole, sharp, placeholder geometry used consistently).
* **Flicker 1 frame in 3** (one marker, 30 fps, blur 3 px, 0.2 m/s): every frame after the 2-frame start is MEASURED/PREDICTED, age <= 67 ms, worst error 0.9 mm.
* **Dropout**: MEASURED -> PREDICTED (<= 0.2 s) -> STALE (<= 1.0 s, never valid, last measured pose) -> LOST, age reported in every state, also when no frame arrives at all; reacquisition immediate with 2+ markers, one extra frame with 1.
* **Inverted** (bottom markers 4-7): reported inverted/flipped, position within 6 mm, held while predicted, None when lost.

## 7. Verified only synthetically (or not at all)

Everything in section 6, and: the rear marker convention, flipped-robot geometry (assumed rolled about the nose axis), height correction with the camera position **derived** from the floor map under a pinhole assumption, the time constants (0.2 s, 1.0 s, 50 ms) against the real camera's frame rate and latency (frame rate was ~29 fps in the traces; latency is unmeasured), the +3 ms cost on the competition laptop, and that `live_v1` raises the real detection rate at all.

## 8. MEASUREMENT CHECKLIST for the operator (every value below is a placeholder or an assumption today)

Set `bot.measured: true` in the overlay config only when items 1-6 are all measured; until then the TRACKING screen shows the banner *PLACEHOLDER config values still in use: marker size/height/offsets (bot.*...)* and the SETUP and ARENA tabs list it too.

| # | Measure | Enter it in | Shown until measured |
|---|---|---|---|
| 1 | **Printed black-square side of each marker (mm), per id** (they may differ) | `bot.marker_size_m` (all) or `size_m` per entry of `bot.markers` in `config/arena_overlay.yaml` (and `home_test_overlay.yaml` for the bench) | PLACEHOLDER banner. **Worse: any marker whose apparent size is not within 30 % of the configured one is rejected, so a wrong size means NO pose** (real traces); banner *MARKERS ARE DECODED BUT REJECTED AS THE WRONG SIZE*, rejection text names `bot.marker_size_m` |
| 2 | **Height above the floor of each marker's surface**, face up, for top and for bottom (and rear centre, upright and inverted) | `bot.marker_height_m.top/bottom` or `height_m` per marker; rear: `height_m`, `height_inverted_m` | PLACEHOLDER banner; position labelled *height-corrected* with the camera source (measured / DERIVED) |
| 3 | **Offset of each marker's centre from the chassis centre**: `forward_m` (toward the nose), `left_m` (looking down, nose away) | `bot.markers[].forward_m/left_m` | PLACEHOLDER banner; wrong offsets show as *markers disagree* / rejected ids and a high marker spread in the confidence inputs |
| 4 | **Which end is the nose** and each marker's rotation (`yaw_deg`, 0 = marker's top edge toward the nose) | `bot.markers[].yaw_deg`; confirm by turning the robot and watching the heading follow the nose | PLACEHOLDER banner; a 180 deg error flips the heading with no other warning |
| 5 | **Marker spacing / chassis-centre definition** (the point the pose means) | follows from 3; the point is the centre of the chassis collision box (observation contract) | none: say which point you measured from |
| 6 | **Which id is on which face**, and whether id 8 exists and where (rear: vertical face) | `bot.markers[].face` (`top`, `bottom`, `rear`); add id 8 from the commented template in the config | id 8 and any other unlisted id: `UNKNOWN ID` rows and *Marker IDs ignored* |
| 7 | **Camera position** (x, y, height above floor, plumb-line from the lens) | `camera.position_m` | PLACEHOLDER *camera position*; pose labelled *DERIVED* (pinhole assumption, principal point at image centre) or *APPROXIMATE* |
| 8 | Lens intrinsics (checkerboard) | `camera.matrix`, `camera.distortion` or the saved lens of the profile | ideal pinhole assumed, silently, no distortion correction |
| 9 | Arena and reference-point positions | `arena.*`, `reference_points[].measured` | PLACEHOLDER banner entries for arena / N reference points |
| 10 | **How it flips** (about which axis) and where the bottom markers really are | bottom entries of `bot.markers`; `chassis_tracker.py` assumes rolled about the nose axis | none beyond item 3's banner |
| 11 | Camera exposure and lighting (the likely cause of the real intermittency): record a short raw clip of the robot moving at the arena and run `marker_survey --video` | camera settings; nothing in the code | the TRACKING marker table shows per-id rate over 5 s; `INTERMITTENT` below 90 % |
| 12 | Assumed, not measured: robot top speed 2.5 m/s and turn rate 25 rad/s (motion gate), hold 0.2 s / lost 1.0 s, agreement 20 mm / 10 deg, size tolerance 30 % | `tracker:` section and `pose:` keys of the overlay config | none: defaults in `TrackerSettings` |

## 9. Tests

    python -m pytest tests/test_aruco_dictionary_consistency.py tests/test_aruco_tracking_conditions.py -q
    python -m pytest tests/test_chassis_tracker.py tests/test_cc_tracking.py tests/test_arena_overlay.py tests/test_opponent_select.py tests/test_roi_tracker.py -q
