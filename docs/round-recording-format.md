# Round recording format (version 1)

Written by `control_center/backend/round_recorder.py`, read by `control_center/backend/round_reader.py`.
One folder per round, under `rounds/` by default:

```
rounds/2026-11-21_14-03-22_heat1/
    manifest.json     setup, which streams were present, counts, errors
    video.avi         the camera frames exactly as captured: no overlays
    frames.csv        one row per camera frame
    imu.csv           raw accelerometer + gyro (only if the controller sent any)
    pose.csv          camera-derived own-bot pose (one row per processed frame, gaps included)
    opponent.csv      opponent tracks (only if a live tracker produced any; none exists yet)
    control.jsonl     controller status, PC-sent commands, telemetry-link changes
    policy.jsonl      policy mode and proposed action (only if a policy ran)
    events.jsonl      operator events and round markers
    derived/          made after the round, never during it (see the end)
```

Only `manifest.json`, `frames.csv`, `events.jsonl` and the video are always there. A file that is
absent means that stream had no data; `manifest.json` → `streams` says so explicitly.

## Time

Every row in every file has `t_s`: **seconds since recording started, on the PC's monotonic clock**.
That is the shared timeline. `t_unix` (PC wall clock at the same instant) is there only to match
other logs.

| Stream | What `t_s` is |
| --- | --- |
| frames | when the PC **received** the frame from the camera driver (`cap.read()` returned). Not the exposure time: camera and USB delay are not measured. |
| imu | when the PC **received** the line. `device_ms` is the controller's own millisecond counter. The two clocks are **not synchronised**; no offset or drift is estimated. |
| pose | the receipt time of the frame the pose was computed from (`frame_index`), not when processing finished. |
| control | when the PC received the controller's line (or read it from the client log). |
| events | when the event happened on the PC. |

## frames.csv

`frame_index, t_s, t_unix, written`

- `frame_index` counts frames since the camera was opened (not since recording started).
- `written` is 1 if the frame is in the video file, 0 if it was dropped (disk too slow) or refused.
- The k-th row with `written=1` is frame k of the video file. The container's frame rate is nominal;
  real timing is this file.

## imu.csv

`t_s, t_unix, device_ms, ax_m_s2, ay_m_s2, az_m_s2, gx_deg_s, gy_deg_s, gz_deg_s, seq, device_us`

- `seq` and `device_us` (the controller's sample counter and its microseconds since boot) are filled when the
  firmware is `ELRS-Handoff-Telemetry`; a jump in `seq` is a gap in what reached the PC.
- Only valid samples are rows. When the controller reports the IMU as not valid it sends no values, and the
  dashboard logs an `imu` event with the reason instead of a row.

- Raw values as the controller sent them: acceleration in m/s², angular rate in deg/s, **sensor axes**.
  Nothing is rotated into a robot frame and the sensor's mounting orientation is not recorded.
- Native rate (whatever the controller sends). Nothing is resampled, filtered or integrated.
- The handoff bench sketch in this repo (`ELRS-Handoff-Bench`) sends no IMU line, so with that
  firmware this file does not exist.

## pose.csv

`t_s, frame_index, valid, x_m, y_m, heading_deg, orientation, position_quality, used_ids, visible_ids, tag_switch, reason`

- Arena-fixed frame of `config/arena_overlay.yaml`: origin at the plain corner, metres, heading in
  degrees counter-clockwise from +x.
- One row per frame the vision thread processed (it skips frames it cannot keep up with).
- `valid=0` is a **gap**: `x_m`, `y_m`, `heading_deg` are empty and `reason` says why (no markers,
  no calibration, conflicting markers). The last good position is never repeated.
- `used_ids` / `visible_ids` are space-separated marker IDs. `tag_switch=1` marks a row where the set
  of markers used changed since the previous valid row: a position step there comes from marker
  offsets, not from the robot moving.
- `position_quality` is `height-corrected` or `approximate`, as in the overlay tool.
- Only written when a floor calibration exists or markers are seen; an uncalibrated round has rows
  with `valid=0`.

## control.jsonl

One JSON object per line. `kind` is one of:

| kind | Fields | Meaning |
| --- | --- | --- |
| `controller_status` | `device_ms, link, armed, requested, active, pc_ok, block, fault, stop, left, right, pc_age_ms, token` | What the **controller reported** (its `ST` line). `active` is the confirmed control source, `requested` is the RadioMaster's mode switch, `left`/`right` are the motor outputs the controller says it is applying. |
| `pc_sent` | `token, throttle, steer` | A drive command the PC client **sent** (from its log). Sent is not applied: compare with `controller_status`. |
| `link` | `text, source` | The telemetry source was opened, closed or failed. |

If more than about 0.6 s passes without a `controller_status`, the control source in that interval is
**unknown** (stale), not "the last value".

## policy.jsonl

`t_s, t_unix, mode, input_source, checkpoint, proposed, note`. `mode` is `OFF`, `SIM` or
`LIVE SHADOW`. `proposed` is what a policy suggested; it is display-only and is never sent.

## events.jsonl

`t_s, t_unix, kind, text` plus kind-specific fields. Kinds include `recording_start`, `round_start`
(the operator's Mark Round Start), `recording_stop`, `mode_requested`, `source_confirmed`, `arm`,
`disarm`, `fault`, `controller_link`, `tracking_lost`, `tracking_reacquired`, `tag_switch`, `camera`,
`calibration`, `policy`.

## manifest.json

`format_version`, `label`, `started_local`, `t0_unix`, `duration_s`, `complete` (false if the
dashboard died before stopping), `software` (git commit), `camera` (source, index, requested and
actual settings), `calibration` (the clicked points, fit errors and quality used during the round,
or `null`), `marker_layout`, `arena`, `controller`, `imu` (units and axis convention), `policy`
(checkpoint identifier if any), `video` (codec, frames received / written / dropped, whether the file
reopened cleanly), `streams` (present or not, row counts), `write_errors`, `timestamps`, `columns`.
A `mock: true` manifest came from the synthetic camera and replayed controller lines, not hardware.

## derived/

`derived/pose_offline.csv` has the same columns as `pose.csv` but was **reconstructed after the
round** by re-running marker detection on the recorded video (`derived/pose_offline.json` says how).
It is never mixed into `pose.csv`.

## Reading it

```python
from control_center.backend.round_reader import RoundData
data = RoundData("rounds/2026-11-21_14-03-22_heat1")
data.control_at(12.5)        # confirmed source, requested mode, armed, or UNKNOWN / STALE
data.pose_live.segments()    # runs of valid poses; gaps are not bridged
frame = data.read_frame(data.frame_at(12.5))
```

or with pandas: `pd.read_csv(folder / "pose.csv")`, `pd.read_json(folder / "control.jsonl", lines=True)`.
