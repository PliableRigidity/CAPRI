# Competition day, 21 November 2026: what the laptop recorded

This note preserves the only machine-recorded evidence from the day of the BEEES antweight competition. The original
files are git-ignored (`logs/`, `config/` per-machine state), so the relevant content is quoted here. The narrative
account of the day is [section 10 of the case study](case-study/README.md#10-competition-day-21-november-2026).

**What this is evidence of:** the Command Centre opened the camera, an arena calibration was clicked, saved and
confirmed, and the own-robot marker tracker did not hold a track.

**What it is not evidence of:** anything about the matches, the robot's driving, the wedge, or where the robot was
relative to the camera at any moment. The log does not record the location of the session; the profile is the LIVE
(competition arena) profile and the time is the morning of the competition. Nothing here was measured with an
instrument.

## 1. Sources

| File (on the development machine, git-ignored) | What it holds |
|---|---|
| `logs/control_center_20261121_103055.log` | Command Centre event log, 10:30:58 to 10:46:52, 56 lines, SHA-256 `a2961a0418d3a978a66491985fe53d361e509e8fc1560b8d5db11ac8f3890886` |
| `logs/sessions/20261121-103058-live/session.jsonl`, `round-001.jsonl` | The same events as JSON lines, plus one 1.4 s round |
| `config/arena_overlay_clicks.json`, `config/arena_overlay_arena.json` | The saved LIVE calibration, 10:35:18 |
| `config/profile_settings/live.json` and its reference picture | Camera mode and the picture the calibration was made on |

## 2. The arena calibration

| | Recorded value |
|---|---|
| Camera | index 1, DirectShow, 1280 x 720, MJPG |
| Saved | 21 November 2026, 10:35:18 (set-up `bc8ccada76`) |
| Method | Four arena corners clicked; pit centres and flipper placed by click; sizes from the design drawing (1150 x 1150 mm, R250 pits, 335 mm flipper legs), no overrides |
| Corner clicks (pixels) | c1 (250, 683), c2 (1010, 676), c3 (928, 72), c4 (352, 97) |
| Operator confirmation | 10:35:42, "Alignment confirmed by the operator; world coordinates trusted for this run" |
| Fit grade | **WEAK**: "no check point clicked: nothing independent confirms the fit (4 fit points always fit exactly)" |
| Click checks | Pit clicks 0.0 mm from the design positions; flipper click 157.9 mm from the design position |

So the arena geometry was established and accepted by the operator. Its accuracy was not independently measured:
no fifth point was clicked, the live profile has no lens-distortion model, and the arena was not tape-checked against
the drawing.

## 3. How large a marker was in the picture

This is an estimate derived from the four saved corner clicks. It is not a measurement of the markers.

The arena floor is 1150 mm on each side. In the calibrated picture its edges span:

| Edge | Length in the picture | Scale at floor level |
|---|---|---|
| c1 to c2 (bottom of the picture, nearest the camera) | 760 px | 0.66 px/mm |
| c2 to c3 (right) | 610 px | 0.53 px/mm |
| c3 to c4 (top of the picture, furthest from the camera) | 577 px | 0.50 px/mm |
| c4 to c1 (left) | 595 px | 0.52 px/mm |

A 4x4 ArUco marker is six cells across including its black border. At 0.50 to 0.66 px/mm:

| Marker side (black square) | Width in the picture | Per marker cell |
|---|---|---|
| 17 mm | 9 to 11 px | 1.4 to 1.9 px |
| 20 mm | 10 to 13 px | 1.7 to 2.2 px |
| 25 mm | 13 to 17 px | 2.1 to 2.8 px |
| 28 mm | 14 to 19 px | 2.3 to 3.1 px |
| 40 mm | 20 to 26 px | 3.3 to 4.4 px |

The printed marker size is **not recorded**. The 17 to 28 mm rows are the sizes the tracker itself inferred for the
markers it decoded that morning (section 4); the 40 mm row is the placeholder in the configuration. If the real markers
are in the 17 to 28 mm range, each one covered roughly 9 to 19 pixels, or about 1.5 to 3 pixels per cell, when the
robot was on the arena floor. That is consistent with the diagnosis that the markers occupied too few pixels to be
decoded reliably. It does not prove it: the estimate ignores the markers' height above the floor, lens distortion,
focus, exposure and motion blur, none of which was recorded.

## 4. The event log

Timestamps are local time on 21 November 2026. Logger names and milliseconds are removed; one local file path is
removed; nothing else is changed. Long messages are the tracker's own wording.

```text
10:30:58  [restore] Remembered camera settings applied
10:31:05  [app] Command Centre started
10:31:05  Control Center started; log=<local path removed>
10:31:05  [camera] Camera OPENING: opening camera
10:31:08  [camera] Camera LIVE: camera 1 (dshow, 1280x720)
10:31:09  [restore] NOT_CALIBRATED: Arena not calibrated
10:34:14  [calibration] Profile 'live': map calibration loaded
10:34:14  [tracking] Tracking restarted for profile 'live': trackers and opponent selection cleared
10:34:15  [restore] NEEDS_CHECK: Calibration loaded and the camera looks unchanged: confirm the alignment
10:34:24  [calibration] Profile 'live': map calibration loaded
10:34:24  [tracking] Tracking restarted for profile 'live': trackers and opponent selection cleared
10:34:24  [restore] NEEDS_CHECK: Calibration loaded and the camera looks unchanged: confirm the alignment
10:34:35  [restore] NEEDS_CHECK: Calibration loaded and the camera looks unchanged: confirm the alignment
10:34:36  [restore] NEEDS_CHECK: Calibration loaded and the camera looks unchanged: confirm the alignment
10:35:18  [calibration] Profile 'live': map calibration loaded
10:35:18  [tracking] Tracking restarted for profile 'live': trackers and opponent selection cleared
10:35:18  [restore] NEEDS_CHECK: Calibration loaded and the camera looks unchanged: confirm the alignment
10:35:42  [restore] Alignment confirmed by the operator; world coordinates trusted for this run
10:35:50  [tracking] ARENA FIT IS WEAK: no check point clicked: nothing independent confirms the fit (4 fit points always fit exactly)
10:35:50  [tracking] PLACEHOLDER config values still in use: 12 reference point position(s) (reference_points); marker size/height/offsets (bot.*; the dictionary is not a placeholder); camera position (camera.position_m)
10:35:50  [tracking] OWN ROBOT LOST: lost: no consistent detection yet; newest frame: every visible marker was rejected: 1: measures 17 mm but is configured as 40 mm (size, height or dictionary wrong); the configured size is a PLACEHOLDER (bot.measured is false): measure the printed black square and set bot.marker_size_m (or size_m per marker) in the overlay config
10:35:51  [tracking] MARKERS ARE DECODED BUT REJECTED AS THE WRONG SIZE (nothing can be tracked): 1: measures 18 mm but is configured as 40 mm (size, height or dictionary wrong); the configured size is a PLACEHOLDER (bot.measured is false): measure the printed black square and set bot.marker_size_m (or size_m per marker) in the overlay config
10:36:28  [tracking] OWN ROBOT STALE (last pose is display only): stale: the last consistent detection is 670 ms old (prediction is allowed for 200 ms); the pose shown is the last measured one, for display only; newest frame: every visible marker was rejected: 0: measures 25 mm but is configured as 40 mm (size, height or dictionary wrong); the configured size is a PLACEHOLDER (bot.measured is false): measure the printed black square and set bot.marker_size_m (or size_m per marker) in the overlay config; 1: measures 26 mm but is configured as 40 mm (size, height or dictionary wrong); the configured size is a PLACEHOLDER (bot.measured is false): measure the printed black square and set bot.marker_size_m (or size_m per marker) in the overlay config; 3: measures 26 mm but is configured as 40 mm (size, height or dictionary wrong); the configured size is a PLACEHOLDER (bot.measured is false): measure the printed black square and set bot.marker_size_m (or size_m per marker) in the overlay config
10:36:29  [tracking] OWN ROBOT LOST: lost: the last consistent detection is 1471 ms old (limit 1000 ms); newest frame: every visible marker was rejected: 1: measures 28 mm but is configured as 40 mm (size, height or dictionary wrong); the configured size is a PLACEHOLDER (bot.measured is false): measure the printed black square and set bot.marker_size_m (or size_m per marker) in the overlay config; 2: measures 25 mm but is configured as 40 mm (size, height or dictionary wrong); the configured size is a PLACEHOLDER (bot.measured is false): measure the printed black square and set bot.marker_size_m (or size_m per marker) in the overlay config; 3: measures 26 mm but is configured as 40 mm (size, height or dictionary wrong); the configured size is a PLACEHOLDER (bot.measured is false): measure the printed black square and set bot.marker_size_m (or size_m per marker) in the overlay config
10:37:04  [tracking] MARKERS ARE DECODED BUT REJECTED AS THE WRONG SIZE (nothing can be tracked): 1: measures 17 mm but is configured as 40 mm (size, height or dictionary wrong); the configured size is a PLACEHOLDER (bot.measured is false): measure the printed black square and set bot.marker_size_m (or size_m per marker) in the overlay config  (+1 more)
10:37:34  [tracking_lost] Own-bot tracking lost: conflict: markers disagree on the bot centre by 158 mm (limit 20 mm)
10:37:49  [tracking_reacquired] Own-bot tracking reacquired after 15.4 s
10:37:49  [tracking_lost] Own-bot tracking lost: no configured bot markers visible
10:37:50  [tracking] OWN ROBOT STALE (last pose is display only): stale: the last consistent detection is 672 ms old (prediction is allowed for 200 ms); the pose shown is the last measured one, for display only; newest frame: no configured bot markers visible
10:37:50  [tracking] OWN ROBOT LOST: lost: the last consistent detection is 1471 ms old (limit 1000 ms); newest frame: no configured bot markers visible
10:37:52  [tracking] MARKERS ARE DECODED BUT REJECTED AS THE WRONG SIZE (nothing can be tracked): 2: measures 18 mm but is configured as 40 mm (size, height or dictionary wrong); the configured size is a PLACEHOLDER (bot.measured is false): measure the printed black square and set bot.marker_size_m (or size_m per marker) in the overlay config
10:38:28  [tracking] tracking restarted: the tab was hidden, so every track and the opponent selection were dropped (click the opponent again)
10:39:09  [tracking] tracking restarted: the tab was hidden, so every track and the opponent selection were dropped (click the opponent again)
10:39:26  [tracking] MARKERS ARE DECODED BUT REJECTED AS THE WRONG SIZE (nothing can be tracked): 5: measures 24 mm but is configured as 40 mm (size, height or dictionary wrong); the configured size is a PLACEHOLDER (bot.measured is false): measure the printed black square and set bot.marker_size_m (or size_m per marker) in the overlay config  (+2 more)
10:39:47  [tracking_reacquired] Own-bot tracking reacquired after 118.4 s
10:39:48  [tracking_lost] Own-bot tracking lost: conflict: markers disagree on the bot centre by 154 mm (limit 20 mm)
10:40:08  [tracking] OWN ROBOT STALE (last pose is display only): stale: the last consistent detection is 623 ms old (prediction is allowed for 200 ms); the pose shown is the last measured one, for display only; newest frame: no configured bot markers visible
10:40:09  [tracking] OWN ROBOT LOST: lost: the last consistent detection is 1470 ms old (limit 1000 ms); newest frame: no configured bot markers visible
10:40:11  [tracking] MARKERS ARE DECODED BUT REJECTED AS THE WRONG SIZE (nothing can be tracked): 0: measures 18 mm but is configured as 40 mm (size, height or dictionary wrong); the configured size is a PLACEHOLDER (bot.measured is false): measure the printed black square and set bot.marker_size_m (or size_m per marker) in the overlay config  (+1 more)
10:43:59  [tracking] OWN ROBOT STALE (last pose is display only): stale: the last consistent detection is 641 ms old (prediction is allowed for 200 ms); the pose shown is the last measured one, for display only; newest frame: inconsistent: markers disagree ([1] vs [2] vs [3]); no usable track to decide
10:43:59  [tracking] OWN ROBOT LOST: lost: the last consistent detection is 1442 ms old (limit 1000 ms); newest frame: inconsistent: markers disagree ([1] vs [2] vs [3]); no usable track to decide
10:44:00  [tracking_reacquired] Own-bot tracking reacquired after 252.9 s
10:44:01  [tracking_lost] Own-bot tracking lost: conflict: markers disagree on the bot centre by 155 mm (limit 20 mm)
10:44:18  [tracking_reacquired] Own-bot tracking reacquired after 17.4 s
10:44:18  [tracking_lost] Own-bot tracking lost: conflict: markers disagree on the bot centre by 155 mm (limit 20 mm)
10:44:19  [tracking_reacquired] Own-bot tracking reacquired after 0.7 s
10:44:19  [tag_switch] Own-bot markers in use changed to [3]; the 135 mm step is a marker-offset effect, not movement
10:44:20  [tracking_lost] Own-bot tracking lost: conflict: markers disagree on the bot centre by 155 mm (limit 20 mm)
10:44:20  [tracking] OWN ROBOT STALE (last pose is display only): stale: the last consistent detection is 676 ms old (prediction is allowed for 200 ms); the pose shown is the last measured one, for display only; newest frame: inconsistent: markers disagree ([0] vs [1] vs [2] vs [3]); no usable track to decide
10:44:21  [tracking] OWN ROBOT LOST: lost: the last consistent detection is 1471 ms old (limit 1000 ms); newest frame: inconsistent: markers disagree ([1] vs [2] vs [3]); no usable track to decide
10:44:21  [tracking_reacquired] Own-bot tracking reacquired after 2.0 s
10:44:22  [tracking_lost] Own-bot tracking lost: no configured bot markers visible
10:45:51  [tracking] MARKERS ARE DECODED BUT REJECTED AS THE WRONG SIZE (nothing can be tracked): 1: measures 17 mm but is configured as 40 mm (size, height or dictionary wrong); the configured size is a PLACEHOLDER (bot.measured is false): measure the printed black square and set bot.marker_size_m (or size_m per marker) in the overlay config  (+1 more)
10:46:31  [tracking] MARKERS ARE DECODED BUT REJECTED AS THE WRONG SIZE (nothing can be tracked): 2: measures 18 mm but is configured as 40 mm (size, height or dictionary wrong); the configured size is a PLACEHOLDER (bot.measured is false): measure the printed black square and set bot.marker_size_m (or size_m per marker) in the overlay config  (+1 more)
10:46:52  Command Centre stopping
10:46:52  Control Center stopped
```

### What the log shows

- **Detection was intermittent, not absent.** Marker IDs 0, 1, 2, 3 and 5 were decoded at various times.
- **Decoded markers were rejected.** The tracker compares each marker's apparent size with the configured size. The
  configured 40 mm is a placeholder (`bot.measured: false`), the markers appeared as 17 to 28 mm, and they were
  rejected as the wrong size.
- **When several markers were accepted they disagreed.** "Markers disagree on the bot centre by 154 to 158 mm (limit
  20 mm)". The marker positions on the chassis in the configuration are also placeholders, so this is expected.
- **No track was held.** Tracking was reacquired after gaps of 15.4 s, 118.4 s, 252.9 s, 17.4 s, 0.7 s and 2.0 s, and
  each time lost again within about a second.
- **No complete observation existed.** The one round started from the Command Centre (10:41:54, MANUAL, stopped by
  the operator after 1.44 s) logged `NO OBSERVATION` six times with IMU `MISSING`, own pose `MISSING` and opponent
  `MISSING`.

### What it does not show

- Where the robot was. The log cannot distinguish "on the arena floor" from "held near the camera".
- Raw per-frame detector output, marker pixel sizes, exposure or focus.
- Anything after 10:46:52. No Command Centre session was logged during the matches.

## 5. Three results that should not be merged

| Stage | Result on 21 November | Basis |
|---|---|---|
| Camera calibration and arena mapping | **Succeeded** (accuracy not independently checked) | Logged, section 2; reported |
| Raw ArUco detection at arena distance | **Unreliable** | Reported; consistent with sections 3 and 4 |
| Pose estimation and continuous localisation | **Not established** | Logged, section 4: placeholder marker configuration, no held track |
