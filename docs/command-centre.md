# Capri Command Centre

One application for the whole competition-day workflow: camera, arena calibration, robot tracking,
hardware and safety checks, live monitoring. You do not need to run any other Python command.

It never arms or hands over the robot, and it starts MANUAL and not transmitting. The RadioMaster and the on-robot
controller do that, with or without this window open; if the laptop crashes, manual driving is unaffected. Wi-Fi motor
commands exist behind a gate that is closed in the current state (see *What is not there*).

## Start it

```powershell
python main_control_center.py
```

That is the only command. It opens the **GUIDED SETUP** tab the first time and, afterwards, the tab and the
profile you used last. Everything below is remembered between runs in `config\command_centre_state.json`
(a damaged file is ignored with a message, never a crash).

| Option | What it does |
| --- | --- |
| `--profile live` / `--profile home_test` | Choose the profile (also remembered for next time). |
| `--tab arena` | Open a tab: `status`, `guided`, `arena`, `tracking`, `connections`, `ai`, `control`, `match`, `operate`, `setup`, `review`, `diagnostics`. |
| `--camera-index 1` | Camera to open (default from `config/control_center.yaml`, or what you set in GUIDED SETUP). |
| `--controller-port COM5` | Listen (read-only) to the controller's status lines on that port. |
| `--controller-log bench.txt` | Follow the log written by `scripts/handoff_bench_client.py --log bench.txt`. |
| `--mock` | No hardware: synthetic camera and replayed controller lines, labelled MOCK everywhere. Remembers nothing. |
| `--video clip.mp4` | Replay a video file in place of the camera, labelled REPLAY. |
| `--review rounds\<folder>` | Open a recorded round; no camera is opened. |
| `--settings-dir D` | Where each profile's remembered camera, calibration context and reference picture go (default `config\profile_settings\`). |
| `--state-file F` | Where progress is remembered (default `config\command_centre_state.json`). |
| `--rounds-dir D:\rounds` | Where round folders go (default `rounds\`). |
| `--legacy` | The first version of the dashboard, unchanged. |

## Profiles: HOME TEST and LIVE

The selector at the top switches profile. Each profile has its **own** calibration, arena, unfinished work
and progress; nothing is shared and nothing carries over.

| | HOME TEST | LIVE |
| --- | --- | --- |
| For | the A4-sheet patch at home | the competition arena |
| Overlay config | `config/home_test_overlay.yaml` | `config/arena_overlay.yaml` |
| Camera corners | `config/overlay_profiles/home_test/clicks.json` | `config/arena_overlay_clicks.json` |
| Arena and hazards (metres) | `config/overlay_profiles/home_test/arena.json` | `config/arena_overlay_arena.json` |
| Sizes | none: you type them | from the design drawing (`config/arena_geometry.yaml`); nothing typed |

Safeguards: an arena file records its profile and refuses to load under another; LIVE refuses the home-test
config; corner clicks made for one arena size are refused with a different size; the older floor-landmark
calibration in SETUP can only save the LIVE landmark file and is blocked while an arena set-up owns the
corners. You cannot switch profile while recording.

## The tabs

Twelve tabs, in the order of a competition day. The status words everywhere are READY / NOT CONFIGURED / WARNING /
FAILED / BLOCKED (SYSTEM STATUS says PASS / NOT TESTED / WARNING / FAIL / BLOCKED: the same five states).

- **SYSTEM STATUS**: every critical component on one page (camera, arena and alignment, own robot, opponent, Wi-Fi,
  manual ELRS and safety checks, AI model, autonomous drive) with why and what to do, the **one next action**, the safe
  state (MANUAL, not transmitting), **Run software checks** and **Copy report**. A PASS is a software check unless the row
  says HARDWARE.
- **GUIDED SETUP**: seven steps (camera, arena, robot tracking, opponent, connections, safety checks, operate), each with a
  status, what is wrong and what to do, buttons to open the right tab, and a camera panel.
- **ARENA**: click-to-calibrate (next section), the remembered calibration and the quick alignment check, advanced overrides.
- **TRACKING**: markers, own robot (MEASURED / PREDICTED / STALE / LOST with age, ids, heading, flipped), the opponent chosen
  by dragging a box, both tracks on the arena, warnings, diagnostics, AI shadow (display only).
- **CONNECTIONS**: the Wi-Fi link to the robot: endpoint, connect / disconnect / reconnect, heartbeat, last command age,
  latency where measurable, firmware identity, the autonomous gate checklist, the hardware acceptance workflow. Starts
  disconnected. Motor transmission stays BLOCKED until acceptance evidence from real hardware exists.
- **AI / MODEL**: the trained checkpoint (identity, compatibility), the verified 19-value observation and what is live or
  missing, the action-to-command mapping table and its limits, a mapping self-check, a diagnostic (simulated) inference.
- **ELRS & CONTROL**: who has control (MANUAL, AUTONOMOUS, SHADOW), link and firmware information with "Not available" where
  the firmware does not report it, the safety-check record, the movement check.
- **MATCH**: MANUAL / AI SHADOW / AUTONOMOUS, Start Round / Stop Round, round timer, prerequisites, intended versus actually
  transmitted throttle and steer, both tracks, authority, faults, session logs. A round starts only when you press Start
  Round. If AUTONOMOUS is requested and anything is missing, the round starts in AI SHADOW and says why.
- **MONITOR** (was OPERATE): camera or map, status rows, **Record Round** (video and data), a short event log.
- **SETUP**: camera, floor-landmark calibration (LIVE only), marker layout, controller and IMU details.
- **REVIEW** and **DIAGNOSTICS**: recorded rounds and the sim-to-real comparison.

Session logs: one folder per run under `logs/sessions/<time>-<profile>/` (`session.jsonl`, `round-001.jsonl`, ...): camera,
tracking, observation validity, decisions, tactics, converted commands, transmission attempts, Wi-Fi status, mode changes,
safety faults, round start and stop. A transmitted command is logged as an attempt; the log never says it was physically
executed (there is no feedback from the motors).

## ARENA tab: marking where the arena and its hazards are

LIVE (the competition arena) asks you for **positions only**. The sizes (1150 x 1150 mm floor, R250 pits, 335 mm
flipper legs) come from the design drawing through `config/arena_geometry.yaml`; see
[arena-geometry.md](arena-geometry.md) for the source and the 1130 vs 1150 mm question. Nothing is typed.

Start with the robot out of the way and the whole arena in view.

1. Press **New set-up**. The current camera frame is frozen (clicks on a live picture are ignored).
2. **Click the four arena corners at floor level, in the order the prompt names:** NEAR-LEFT (bottom-left of
   the picture), NEAR-RIGHT, FAR-RIGHT, FAR-LEFT. For a corner cut away by a pit, click where the two walls
   would meet at floor level (the middle of the quarter circle).
3. **Click the centre of Pit 1, then the centre of Pit 2.** Each click is matched to its nearest corner (the design
   centres a pit on its corner) and the tab says how far your click was from it. The two pits must be on opposite corners.
4. **Click the flipper once** (anywhere on it). Its corner and its direction come from which corner you clicked
   nearest; the legs run along the walls.
5. Look at the picture: outline, pit circles and flipper are projected from the design. **To correct one point,
   click its row** in the table and click again. **Undo** takes back the last step; **Reset** clears this set-up
   (saved files are untouched); **Retry (new frame)** takes a fresh frame and keeps your clicks.
6. Read **Calibration accuracy**: the picture scale on the near and far side (mm per pixel), how far a 1-pixel click error
   moves the map, how far each hazard click was from where the design puts it, and any `UNRELIABLE` reason (a very oblique
   view, or a coarse far side). An unreliable mapping turns the fit WEAK.
7. **Save.** It is refused, with the reason, while anything is missing or inconsistent. Saving also remembers the camera's
   mode and a reference picture for the next start (see Persistence below).
8. **Quick alignment check**, then **Alignment OK, continue** when the saved outline sits on the floor edges, the circles on
   the pits and the shape on the flipper.

**Advanced: measured differences from the design** (collapsed by default; leave empty unless you measured). Arena width or
height, pit radius, flipper leg, or an *inset per side*. A 10 mm inset gives 1130 mm. The override is stored with a reason
beside the design numbers it replaced, and the tab shows "ADVANCED OVERRIDE in use" every time.

**HOME TEST** (a patch of any size) still asks for its own dimensions: corners, size, pits with radius, flipper vertices.
It never uses the competition numbers.

**The camera moved, the arena did not:** **Recalibrate camera**. Only the four corners are asked, named by what they are
(PLAIN, FLIPPER, PIT 1, PIT 2 corner), so it works from any side. The saved hazards are not touched. **The arena changed:**
**Edit arena**: every item starts as saved (violet); click a row to replace it; the camera corners stay. **Continue
unfinished** resumes work interrupted by a restart.

What is rejected: both pit clicks nearest the same or adjacent corners; a pit or flipper click far from any corner; corner
clicks on one spot; a mirrored click order; for typed set-ups sizes outside 0.3 to 3 m and the other limits of the typed flow.

**Frames.** Hazard positions are never taken from screen orientation. From the clicked hazards the tab works out which
corner is plain, flipper and each pit and saves everything in the **simulator's frame** (origin at the plain corner, pits at
(0, H) and (W, 0), flipper at (W, H)); that is the frame the policy's observation uses. Starting from a different corner, or a
camera turned by 90, 180 or 270 degrees, gives the same physical hazards (tested). A layout that cannot be expressed that way
stays in the set-up frame and is flagged `policy_frame_ok: false`.

## Persistence between rounds

Per profile the Command Centre remembers: the arena and hazards, the camera corner clicks (homography), the camera index,
resolution, frame rate and format and a reference picture (`config/profile_settings/<profile>.json` plus the picture), the
marker settings, the Wi-Fi endpoint and the chosen AI model. It never remembers the opponent, a track, armed or authority
states, observations or a Wi-Fi connection status (the store refuses those keys). Settings are written atomically (temporary
file, flush, replace) with one rolling `.bak`; a damaged file falls back to the backup and then to defaults, with the reason
shown. HOME TEST and LIVE never share a file.

Between rounds (reconnect the camera, open the Command Centre):

1. The last LIVE profile loads and the camera opens in the mode it was calibrated in (same index, size, frame rate).
2. The ARENA tab shows **Saved calibration**: whether it loaded, whether the camera settings match those used when it was
   saved, and an automatic check of the live picture against the stored reference picture (largest corner shift in pixels:
   4 px or less matches, up to 10 px is a minor shift, more than that needs recalibration, a dark or featureless picture is
   "cannot verify", never "matches").
3. **Quick alignment check** draws the saved arena over the live picture. If it fits, **Alignment OK, continue** trusts the
   world coordinates for this run only; it is asked again at the next start.
4. A different picture size, a camera that moved, or a corner/arena mismatch means **Recalibrate before trusting world
   coordinates** (the map is flagged NOT confirmed and the AI shadow inputs report it as missing).
5. A camera that is unplugged and plugged back in is retried automatically (1, 2, 4, 5... s); **Reconnect camera** tries
   now. After it returns, the picture is checked again: same place keeps the trust, a changed mode or a knocked camera
   withdraws it.

## TRACKING tab: choosing and following the opponent

The opponent is whatever you drag a rectangle around in the live camera picture: a robot, a ball, a phone. There
is no size limit and nothing to configure; the old 280 mm limit and the YAML thresholds are not used on this path.

- **Select Opponent / Draw Box** arms a crosshair; **drag a rectangle** around the object and release. Dragging
  works at any time. A plain click only reminds you to drag (a click cannot say how big the object is).
- **Reselect Opponent**: drag a new box. The current track stays until the new box is accepted. **Clear Opponent**
  drops the selection. None of this restarts the camera, the calibration or Capri's marker tracking.
- A box is refused only when it is empty (under 4 pixels a side), not a number, entirely outside the picture, or
  centred on our own tagged robot. A box partly outside the picture is cut to the picture.
- **Tracking status** and **Tracking confidence** are shown under the picture, beside the box and the centre cross.

| Status | Meaning |
| --- | --- |
| TRACKED | confidence 0.60 or more: a position is shown |
| TRACKING UNCERTAIN | confidence under 0.60, the object is hidden, or the box overlaps our own robot: **no position is shown**; it recovers by itself if the object is seen again within 0.8 s |
| LOST | uncertain for more than 0.8 s: never picked up again by itself, drag a new box |
| NOT SELECTED | nothing selected |

How it works: OpenCV's CSRT and KCF trackers are used when the installed build has them; this PC's build (4.13, no
contrib) has neither, so the built-in tracker is used: template matching around the predicted position, over
several sizes, with a slowly adapting template; a one-colour object is followed by its colour region. Confidence is
an appearance similarity from 0 to 1 (not a probability), lowered when the best match is hardly better than any
other place in the search window.

**The position is image-based, not a physical measurement.** The centre of the tracked box is projected onto the
floor plane with the saved calibration. An object raised above the floor appears displaced away from the camera's
nadir, so its position is approximate; the display says so. The object's apparent size on the floor is shown as a
diagnostic and never decides whether a selection is accepted.

## What the statuses mean

| Status | Meaning |
| --- | --- |
| Not configured | Nothing has been set up yet, or the input it needs does not exist. |
| Ready | Checked, nothing to warn about. |
| Warning | Usable, but something is weak, stale, unmeasured or still a placeholder. The warning is shown. |
| Failed | Checked and not usable. The tab says what to do. |

A status is decided by the code that checked it, never by the guided page.

## What the control source means

| Shown | Meaning |
| --- | --- |
| RADIO / PC / NONE | The controller's own report, less than 0.6 s old. |
| STALE | The controller was reporting and stopped. The last value is not shown as current. |
| UNKNOWN | No controller telemetry has been received. |

User-facing modes are **MANUAL** (RadioMaster over ELRS), **AUTONOMOUS** (the policy on the laptop, disabled in
this build) and **AI SHADOW** (display only). A PC-originated command is an implementation detail of AUTONOMOUS;
the supervised PC test commands exist only inside the movement check.

## Recording a round

1. Press **Record Round** (F8) before the referee's signal. Only camera frames are required.
2. Press **Mark Round Start** (F9 or Space) at the signal. It adds a timestamp and nothing else.
3. Press **Stop Recording**. The folder and any dropped frames or write errors are shown.

Each round is one folder; the format is in [round-recording-format.md](round-recording-format.md).
At 720p the default MJPG video is about 3 MB per second.

## What is not there

- **Autonomous drive is BLOCKED.** Reasons, all real: the baseline firmware sends no IMU data (the policy needs it, and nothing
  is invented); there is no hardware acceptance record for Wi-Fi motor commands (the 20 November Wi-Fi build restarted and the
  supply collapsed, cause unknown); no firmware has a motors-disabled reception test mode; motor direction and stiction have
  not been checked on the real robot; the policy has never seen live camera data. The CONNECTIONS and SYSTEM STATUS tabs list
  the current exact reasons. No Command Centre action flashes firmware or arms the robot.
- A software emergency stop travels over the same Wi-Fi link as every command: when Wi-Fi is down it cannot reach the robot.
  The firmware's 250 ms stale-command stop and the RadioMaster ARM switch are the backstops.
- The controller link (ELRS & CONTROL) is listen-only; the only transmitting code is the separate Wi-Fi sender, which is
  gated as described above.
- Nothing in this document is a hardware result. See the [test playbook](autobots-test-playbook.md) for what has
  and has not been run on the real camera and robot.

## Appendix: the first dashboard (`--legacy`)

The September 2026 dashboard is still available, unchanged, with `python main_control_center.py --legacy`. These notes
were in the repository README until 22 November 2026. They describe that first window, not the Command Centre above.

It is a local PySide6 application that shows the Logitech C525 feed, runs marker detection off the UI thread, exposes
operator request controls and shows honest placeholders for hardware that is not connected. It is not a command
authority for the robot. `--demo` (legacy window only) shows clearly labelled simulated telemetry; normal mode never
invents telemetry. ESP-NOW, state fusion and command acknowledgement were the September plan and were never built.
TDOA code is retained only as optional legacy code and is not part of the active UI or runtime state. Logs are written
to `logs/`.

**Camera start-up.** If the C525 is not camera index 0, edit `config/control_center.yaml` or pass `--camera-index 1`.
The C525 is configured to use DirectShow because measured first-frame start-up is under one second, against roughly
7 to 9 seconds for MSMF on the development machine. Only the configured backend is tried initially; the default
backend is a single fallback if that attempt genuinely fails. After publishing the first default frame, the camera
performs one controlled same-backend reopen for the configured mode. For this C525, DirectShow requires width, height
and FPS to be requested before MJPG; placing MJPG first silently falls back to 720p YUY2 at roughly 10 FPS. If
validation fails, the application restores the same-backend default mode instead of losing the feed.

**Arena vision workflow in the legacy window.**

1. Press **RECALIBRATE ARENA** and click the four arena corners in the order shown in the event log: near-left,
   near-right, far-right, far-left. The transform is validated and persisted.
2. With an empty arena, press **CAPTURE BACKGROUND**. Capture runs on the vision thread and the panel shows progress.
3. Place the marked robot and the opponent in the arena. Press **AUTO ACQUIRE OPPONENT**, or press **SELECT OPPONENT**
   and drag a rectangle around it.
4. Use **DEBUG VIEW** to cycle through raw, rectified, foreground, candidate and self-exclusion views. Press
   **CLEAR TRACK** to discard the current opponent track.

Pit and flipper coordinates remain `null` or empty in `config/control_center.yaml` until physically measured. The
automatic landmark detector reports candidates only and never replaces a valid manual homography.
