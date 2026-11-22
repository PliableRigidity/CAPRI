# Capri test playbook

> **Read this first (added 22 November 2026).** This playbook was written for Capri V1 on 20 and 21 November 2026,
> before the competition, when the project was still called AutoBots. It is kept as written: the procedures are
> still the right ones to run, and the status statements in it ("not yet run", "the night of 20 November") describe
> that date. The [COMPETITION DAY](#competition-day) section is the procedure as planned; what actually happened on
> 21 November is [section 10 of the case study](case-study/README.md#10-competition-day-21-november-2026).
> The Wi-Fi network name `AutoBot-Handoff`, the firmware identifiers `autobot-handoff-*` and the environment
> variables `AUTOBOT_SKIP_SLOW` and `AUTOBOTS_FOOTAGE_MP4` are real identifiers and keep the old name.

A development and bench-testing guide. **Start with [What to test next](#next)**: it is the main test table, in
dependency order. After changing something, find it in [the index](#index) and rerun the tests it points to.
**Competition day: [follow the short procedure right below](#competition-day).**

Written 2026-11-20 against commit `8b8b7f7` plus the untracked working tree, and **revised that night
after the first hardware session with the Wi-Fi build** (restarts and a collapsing supply on battery; see
[what happened on 20 November](#nov20)). The order of work changed: the wireless build is no longer the
next test. Commands were checked against the code or `--help`. What was only checked in software, and
what does not exist yet, is listed in [Appendix C](#appendix-c).

**One launch command.** From 21 November the whole workflow is one application:

```powershell
python main_control_center.py
```

It opens GUIDED SETUP the first time and then the tab and profile you used last (`--profile live` or
`--profile home_test` to choose; `--mock` to try it with no hardware). Camera, arena calibration, tracking,
the ELRS & CONTROL page and live monitoring are tabs of it; see [command-centre.md](command-centre.md). The
older per-tool commands below still work, but you do not need them on competition day.

**Ground rules of this guide**

- Nothing here arms, flashes or moves the robot from a launcher. Uploading a sketch, switching ARM and
  MODE on the RadioMaster, and starting a drive window are always your own actions.
- Start every motor test with the **wheels raised** and the power cut-off in reach.
- Results from mocks, synthetic frames, recorded footage and the simulator never count as hardware results.
- **A test that passed with the USB cable plugged in says nothing about battery power.** The cable powers the
  board. Battery-only behaviour is its own test ([PWR-1](#pwr-1), [MD-Q](#md-q)).
- **Do not connect USB and the battery together** until [PWR-2](#pwr-2) has an answer.

---

<a id="competition-day"></a>
## COMPETITION DAY: follow this, top to bottom

One application: `python main_control_center.py` (opens on **SYSTEM STATUS**, which always shows the **one next thing to do**).
Everything below was built and tested in software only. **No step of it has been run on the real robot, camera or arena.**
`PASS` on the status page means a *software* check passed unless the row says HARDWARE.

**The robot is always recoverable by hand:** RadioMaster ARM switch OFF, or cut the battery. The laptop can never
override that, and the application always starts in MANUAL, **not transmitting**.

1. **Manual ELRS first (nothing else matters if this fails).** Battery alone, USB unplugged, wheels raised: RadioMaster drives
   both wheels forward / back / left / right, ARM off stops them ([MD-1](#md-1), [MD-Q](#md-q)). `ELRS-Handoff-Bench.ino` stays
   on the robot. Do not flash, do not change the transmitter. If the supply collapses or the board restarts, stop here ([PWR-1](#pwr-1)).
2. **Camera.** Plug the camera in, then open the Command Centre (profile **LIVE**: the box at the top). SYSTEM STATUS row *Camera*:
   it reopens the camera in the mode it was calibrated in. If it is not found, **Reconnect camera** (ARENA tab). Do not change
   resolution unless you intend to recalibrate.
3. **Calibration, no measuring.** ARENA tab > *Saved calibration*: it says whether the calibration loaded, whether the camera
   settings match, and compares the live picture with the stored one.
   - **Quick alignment check**: the saved outline must sit on the floor edges, the circles on the pits, the shape on the flipper.
     Looks right → **Alignment OK, continue**. (Asked again at every start; it is never remembered as trusted.)
   - Row says *Recalibrate*, or the outline is off → **Recalibrate camera** (four corners) or **New set-up**: four corners,
     centre of pit 1, centre of pit 2, one click on the flipper. Sizes come from the design drawing; you type nothing.
     Read *Calibration accuracy* and Save. If the real arena measures differently from the drawing, use *Advanced* once.
4. **Own robot.** Put Capri in view. TRACKING tab: marker IDs seen, position, heading, state MEASURED / PREDICTED / STALE /
   LOST with its age. A stale or lost pose is never shown as current. Marker sizes and offsets still to be measured:
   [aruco-tracking-audit.md](aruco-tracking-audit.md) (checklist at the end).
5. **Opponent, every round.** TRACKING tab: **Select Opponent**, drag a box around it. Tracked / uncertain / lost is shown; the
   position is image-based and approximate. The selection is cleared when a round ends and is never restored after a restart.
6. **AI SHADOW (no motor output, ever).** AI / MODEL tab: **Load model** (`runs/recovery-challenger-v2-seed1/final_model.zip`).
   MATCH tab: choose **AI SHADOW**, **Start Round**. Expect the decisions to be **blocked: IMU unavailable**: the baseline
   firmware sends no IMU data and nothing is invented. Shadow still shows what is missing and whether camera, own robot and
   opponent inputs are fresh. Stop Round when done.
7. **Wi-Fi diagnostics (optional, deliberate).** Only after steps 1-3 are solid, and not with USB and battery together
   ([PWR-2](#pwr-2)). The earlier Wi-Fi firmware restarted the board and the supply collapsed; the cause is unknown
   ([what happened](#nov20)). The Wi-Fi sketch is **not** on the robot and **nothing in the Command Centre flashes it**.
   If you choose to try it, flash by hand following [HO-WIFI-1](#ho-wifi-1), join the robot's network, then CONNECTIONS tab:
   Connect (status requests only), read heartbeat age and firmware identity, run the 60 s stability window. If the board
   restarts or the heartbeat stalls: Disconnect, go back to step 1, restore with [HO-R](#ho-r).
8. **Physical Wi-Fi motor tests: only with your explicit decision**, wheels raised, one person on the RadioMaster, in the
   order the CONNECTIONS acceptance panel lists ([HO-WIFI-2](#ho-wifi-2)). Autonomous stays **BLOCKED** until every required step
   is recorded with HARDWARE evidence. Also verify with wheels raised that each action turns the wheels the way the mapping
   table (AI / MODEL tab) says; **left/right and signs have never been checked on the real robot**.
9. **Between rounds:** Stop Round (neutral, MANUAL), reconnect the camera, reopen the Command Centre if it was closed, redo steps
   3 (quick check) and 5 (new opponent box). No re-measuring unless the status page says the camera moved.

**Blocked for physical autonomy tonight (and why):** no IMU on the working firmware; no hardware Wi-Fi acceptance record (the
earlier Wi-Fi build was unstable); no motors-disabled reception test mode in any firmware; motor direction and stiction
unverified; the policy is a behaviour-cloned pursuit controller trained on exact simulator state, never tested on live
camera data. See the status page row *Autonomous drive* for the current exact list.

---

<a id="next"></a>
<a id="table"></a>
## What to test next, in dependency order

This is the main test table. Each row is one test: its ID links to the full procedure further down.

- **Working on the robot now (your report, night of 20 Nov):** RadioMaster driving on battery alone, on
  `ELRS-Handoff-Bench.ino`, after a battery change and a fresh upload of that sketch. Which of the two
  changes restored it is not known.
- **PASS on the robot earlier:** wired PC driving and both ordinary handoff directions ([HO-USB-1](#ho-usb-1)).
- **Ran on the robot and did not hold:** the Wi-Fi handoff build. It drove from the keyboard, then lost the
  PC link repeatedly and restarted. It is now an experimental route, not the next test.
- **NOT RUN** = ready, waiting for you. **BLOCKED** = a prerequisite has not passed yet (named in the row).
  **PROPOSED** = designed on paper; there is nothing to run yet.
- Evidence in brackets: `[MOCK]` real code on a mocked board or link, `[FIXTURE]` synthetic frames or scripted
  telemetry, `[SIM]` the simulator, `[COMPILE]` builds for the ESP32-C3. None of those is a hardware result.

### 0. ONE APPLICATION: first run, arena calibration, profiles

Everything in this block is done inside `python main_control_center.py`. None of it has been run on the real
camera yet; the software column says what was checked without one.

| Test ID | Prerequisite | Command or physical action | Pass condition | If it fails | State |
| --- | --- | --- | --- | --- | --- |
| [CC-1](#cc-1) | none | `python main_control_center.py --mock` | all twelve tabs open; SYSTEM STATUS lists every component; GUIDED SETUP shows seven steps with statuses; everything is labelled MOCK | `logs\`; a tab that failed to load shows its error and the others still work | PASS `[MOCK]` window built and checked offscreen by tests (the literal command was not opened on a screen) |
| [CC-2](#cc-2) | C525 connected, no other program using it | `python main_control_center.py --profile home_test`; GUIDED SETUP step 1: pick the index, size and fps, press Open camera | preview shows the picture; step 1 reads Ready or Warning with its reason | message under the preview says why; close the Logitech app / browser tabs; try the other index | NOT RUN |
| [ARENA-1](#arena-1) | CC-2 passed; arena or A4 patch fully in view | ARENA tab > New set-up; four corners, size, both pits, flipper, Save, Check saved overlay, tick the box | `SAVED`, then READY; outline on the floor edges, circles on the pits, shape on the flipper | rows of the table say what is missing; fix one point by clicking its row | NOT RUN `[FIXTURE]` on rendered frames only |
| [ARENA-2](#arena-2) | ARENA-1 saved | move or knock the camera; ARENA tab > Recalibrate camera; click the four corners | arena file unchanged (its date does not change); overlay sits on the floor again | New set-up if the arena itself changed | NOT RUN `[FIXTURE]` |
| [ARENA-3](#arena-3) | ARENA-1 saved | close the application; start it again with no options | same profile and tab; ARENA reads Ready (or Warning until you tick the confirmation again if the corners were re-saved) | message says which file could not be read | NOT RUN `[FIXTURE]` |
| [PROF-1](#prof-1) | arena saved for LIVE and a calibration for HOME TEST | switch the profile box LIVE / HOME TEST / LIVE | each shows only its own arena; HOME TEST never shows 1.15 m, LIVE never shows 0.5 m | stop recording first; read the red message next to the box | NOT RUN `[FIXTURE]` on temporary files only |

<a id="nov20"></a>
### What happened on 20 November, in six lines

1. The Wi-Fi build came up and drove the robot untethered from the keyboard (12:48 onward, in the log).
2. Over the next two hours the PC link went stale 12 times (each one latches until ARM off, on), status
   dropped out 56 times, and the controller restarted 3 times while a client was logging. Later it entered a
   restart loop on battery, taking the ELRS receiver down with it. The left wheel moved at resets.
3. Multimeter, as reported: 5.4 V at the unloaded battery; 3.0 V at the buck converter's input and 1.0 V at
   its output in one failing condition; 4.8 V on **both** converter terminals with USB alone.
4. Other batteries and a replacement converter did not settle it. The root cause is **not identified**.
   That Wi-Fi's current draw exposed a marginal supply is a hypothesis, not a finding.
5. A different battery plus a fresh upload of `ELRS-Handoff-Bench.ino` brought battery operation back.
6. The preferred way to get PC commands to the robot is now through the existing ELRS link
   ([relay](#rly)), if the transmitter can accept them. Wi-Fi stays in the repository as route D.

Evidence and what it does and does not show is summarised in the tracked
[case study](case-study/README.md#7-20-november-2026-wi-fi-integration-failures-and-the-decision-to-reuse-elrs).
The underlying local record is `artifacts/integration-2026-11-20/evidence.md` and is intentionally git-ignored.
The narrative is section 7 of the [case study](case-study/README.md).

### Command routes, in the order to pursue them

| Route | Path | State |
| --- | --- | --- |
| RADIO | RadioMaster → ELRS → robot | **Working on battery** (reported). The competition default |
| A | Laptop → USB → ESP32-C3 | Demonstrated (normal path). Tethered. Needs [PWR-2](#pwr-2) before USB and battery are used together |
| B | Laptop → RadioMaster Pocket → ELRS → robot | **PROPOSED.** Nothing built or powered: [relay](#rly) |
| C | Laptop → a second ELRS transmitter → robot | Rejected for now (one transmitter per binding; the Pocket would lose authority) |
| D | Laptop → Wi-Fi → ESP32-C3 | Experimental. Drove, then unstable. Not a default until the supply is understood |

### 1. FIRST, EVERY SESSION: the baseline on battery

USB unplugged. This is the fallback you will compete with; protect it before trying anything else.

| What | Value |
| --- | --- |
| Sketch on the robot | `ELRS-Handoff-Bench\ELRS-Handoff-Bench.ino` |
| SHA-256 | `1A400D3FDF66D0A718D6C96166A6A7A891641EC47A93F31C5FEFEA84D8CB5FBC` |
| Names itself | `HELLO fw=autobot-handoff-bench-1` (visible only over USB) |
| Last built by the IDE | 2026-11-20 21:56:56, 330,000 bytes; the built source matches the file |
| Behind it | `ELRS-Control\ELRS-Control.ino` (`6040BED3…A6DD`), your original, no ARM or MODE switch |

| Test ID | Prerequisite | Command or physical action | Pass condition | If it fails | State |
| --- | --- | --- | --- | --- | --- |
| [PWR-1](#pwr-1) | multimeter; the battery you will compete with | measure the pack at rest; then the converter input and output with the robot powered and idle; then again while the wheels run raised | every reading written down with the time and the pack; converter output steady at its rated voltage while the wheels run | **do not compete on that pack**; try the next; write down what sagged | NOT RUN |
| [MD-Q](#md-q) | PWR-1 readings taken; battery, USB unplugged, wheels raised first | RadioMaster: ARM off, on, sticks | no movement at power-up; forward, reverse, left, right, stop; transmitter off stops the wheels; **no restart in five minutes of driving** | cut power; note what the receiver LED did; [HO-R](#ho-r) | Driving reported working 20 Nov night; not run as the written test |
| [PWR-2](#pwr-2) | robot unpowered; multimeter | find how USB 5 V, the converter and the battery are joined (inspection and continuity) | you can say whether a diode or separate rails keep USB out of the battery side, and it is written down | **keep USB and battery apart**: wired tests run on USB power only | NOT RUN. Gate for every row that needs both |

### 2. ROUTE A: wired handoff and the movement check

The only PC command path with a hardware pass. The movement check answers one question: does the same
commanded sequence move the robot the same **way** in the simulator and in reality?

| Test ID | Prerequisite | Command or physical action | Pass condition | If it fails | State |
| --- | --- | --- | --- | --- | --- |
| [MOVE-1](#move-1) | none | `python scripts\motion_check.py sim` | prints the sequence and the simulator's prediction; writes `artifacts\motion-check\sim-reference.json` | [SW-6](#sw-6) | PASS `[SIM]` |
| [MOVE-C](#move-c) | [PWR-2](#pwr-2) answered; bench sketch on the robot; USB; **wheels raised** | `python scripts\motion_check.py checks --port COM5 --power usb-only` | five checks, each `controller evidence: PASS` and your "yes": stop, disarm, manual takeover, PC link loss, radio loss | cut power; keep `artifacts\motion-check\checks-…\`; do not go on | NOT RUN |
| [MOVE-R](#move-r) | MOVE-C run (pass or not); wheels raised | `python scripts\motion_check.py run --port COM5 --surface raised --power usb-only`, then type `GO` | sequence completes; each wheel turned the way the table says; `compare` reports SIGNS AGREE | a wrong direction is a sign error somewhere between the simulator and the wheels: stop and find it | NOT RUN |
| [MOVE-F](#move-f) | **MOVE-C fully passed within 12 hours on this firmware**; clear floor | `python scripts\motion_check.py run --port COM5 --surface floor --power usb-only`, then `GO` | the robot advances, turns left, turns right and reverses, a few centimetres each, and stops between | ARM off. The tool refuses to start without MOVE-C | BLOCKED: needs MOVE-C |
| [MOVE-3](#move-3) | a run folder with `observed.yaml` filled in | `python scripts\motion_check.py compare artifacts\motion-check\<run folder>` | `SIGNS AGREE on every moving step`; distance and rotation ratios written down | `MISMATCH` lines say which step | NOT RUN |
| [HO-USB-2](#ho-usb-2) | HO-USB-1 passed | keyboard client, failure cases F1–F10 | every loss or disarm stops and latches | stop; record the `ST` line | NOT RUN (MOVE-C covers F5, F7, F8 and the takeover) |

### 3. INDEPENDENT CAMERA TESTS: no robot link, any time

No robot link, no firmware. Close every other program that uses the camera. All commands open camera index 1.

| Test ID | Prerequisite | Command or physical action | Pass condition | If it fails | State |
| --- | --- | --- | --- | --- | --- |
| [CAM-2](#cam-2) | C525 mounted where you will use it | `python scriptsruco_camera_check.py`, hold a tag at the near edge, centre and far corner | whole area in view with margin; tag sharp and decoding at all positions | lower or move the camera; fix focus in the Logitech / Windows camera settings | NOT RUN |
| [CAM-1](#cam-1) | CAM-2 running | show the attached tags **one at a time**, then all together, then all covered | each ID correct and steady for 10 s; `Tags:` equals the number visible; none when covered | wrong dictionary first (printed when `scriptsruco_camera_check.py` starts), then glare or a missing white border | NOT RUN |
| [TAG-0](#cam-3) | CAM-1 passed | measure each tag: black-square side, height above the floor, offset and rotation from the bot centre; enter them in `bot:` of `config\home_test_overlay.yaml` and set `bot.measured: true` | the config holds your numbers, not the placeholders | without this, CAM-3 reports a size conflict or a wrong centre | NOT RUN |
| [CAL-1](#cal-1) | camera fixed; A4 sheet on the test area | `python -m control_center.tools.arena_overlay --camera 1 --config config\home_test_overlay.yaml --profile home_test --calibrate` | `saved 5 point(s)`; `FIT:` is not INSUFFICIENT; outline sits on the sheet | `mapping is mirrored` → click order; redo after any camera move | NOT RUN |
| [CAL-2](#cal-2) | CAL-1 saved; a taped cross measured from the sheet corner | `… --profile home_test --truth X,Y` | held-out check and `truth error` both ≤ 10 mm | error toward or away from the camera → tag height; otherwise redo CAL-1 | NOT RUN |
| [CAM-3](#cam-3) | TAG-0 and CAL-1 done | `python -m control_center.tools.arena_overlay --camera 1 --config config\home_test_overlay.yaml --profile home_test` | green `BOT: x … y … nose heading …` line; heading follows the nose when you turn the bot | `conflict: marker N measures …` → `bot.marker_size_m`; no pose → dictionary or ids | NOT RUN |
| [CAM-4](#cam-4) | CAM-3 passed; bot not touched during the test | same window: all tags, each tag alone, all covered, all uncovered | every single-tag row within 20 mm and 10° of the all-tags row; `unknown` at once when covered; back on the first frame after | a jump when the visible tag changes is a wrong offset in `bot.markers`, not movement | NOT RUN |
| [TRK-1](#trk-1) | CAL-1 saved; tagged object and an unmarked object | `python -m control_center.tools.shadow_stream_demo --camera 1 --config config\home_test_overlay.yaml --profile home_test` (add `--tag-size-mm NN` if TAG-0 is not done) | separated: both positions; touching: opponent `UNKNOWN overlap`; apart again: opponent `OK` within a frame or two | retake the empty reference (SPACE); more contrast | NOT RUN |
| [TAG-S](#tag-s) | CAM-2 running; the robot in view, top up, then bottom up, then rear toward the camera | `python -m control_center.tools.marker_survey --camera 1 --seconds 10` | one dictionary explains the detections; every ID is listed with how often it was seen; you have written down which ID is on which face, including whether the rear one really is 8 | no IDs at all → focus, glare, tag too small in the picture; do TAG-S before TAG-0 | NOT RUN |
| [TRK-2](#trk-2) | **TAG-0 done for all nine markers**; CAL-1 saved | `python -m control_center.tools.live_track --camera 1 --config config\home_test_overlay.yaml --profile home_test` | `OWN TRACK MEASURED` with several ids; covering some tags changes nothing; covering all: `PREDICTED` for 0.2 s, then `STALE`, then `LOST` after 1 s; no pose is ever shown as valid after that | `rejected: <id>: …` names the marker whose configured offset disagrees, with the millimetres | NOT RUN (`[FIXTURE]` SW-7 passes) |
| [TRK-3](#trk-3) | TRK-2 running; an unmarked object on the patch | TRACKING tab: **Select Opponent / Draw Box**, then drag a rectangle around any object; Reselect / Clear Opponent | `TRACKING STATUS: TRACKED` follows it; hidden: `TRACKING UNCERTAIN` with no position; removed for more than half a second: `LOST` until you click again; a click on your own robot or on bare floor is `REFUSED` | lighting or a floor-coloured object: write down what it did | NOT RUN (`[FIXTURE]` SW-7 passes) |

<a id="rly"></a>
### 4. ROUTE B: relay through the RadioMaster Pocket (PROPOSED: nothing to run yet)

Full investigation, with sources: [elrs-relay-feasibility.md](elrs-relay-feasibility.md). **Do not connect
anything to the trainer jack** before the multimeter steps in that note. No sender program exists in this
repository, on purpose.

| Interface on the Pocket | Can a PC send channels in? |
| --- | --- |
| USB-C, joystick mode | **No.** Output only |
| USB-C, serial (VCP) as a trainer input | **No** in released EdgeTX |
| USB-C, serial (VCP) in LUA mode + a mix script on the SD card | Possible on stock firmware, about 20 updates a second; **never demonstrated** |
| DSC trainer port, PPM | Supported by the firmware; **pinout and voltage tolerance undocumented**; needs a microcontroller bridge |
| Module bay CPPM / SBUS; ELRS backpack; unreleased EdgeTX build | Possible in principle; each needs hardware access or firmware changes on the only transmitter |

| Test ID | Prerequisite | Command or physical action | Pass condition | If it fails | State |
| --- | --- | --- | --- | --- | --- |
| RLY-1 | transmitter only, robot **off** | set USB-VCP to LUA, load a mix script, watch the Pocket's channel monitor while the PC sends | channels follow the PC only with the relay switch on; sticks return when it is off; neutral within a quarter of a second of pulling the cable | put USB-VCP back to its previous setting | PROPOSED: no script or sender written |
| RLY-2 | RLY-1; robot on, **disarmed**, on USB power for monitoring | `python scripts\handoff_bench_client.py --port COM5 --monitor` and read `steer=` / `thr=` in the `CH` line while the PC sends known values | sign and size of what arrives match what was sent | fix the mixer weights; never the ARM channel | PROPOSED |
| RLY-3 | RLY-2; battery, wheels raised | relay drive; relay switch off; ARM off; cable pulled; transmitter off | every one stops the wheels | stay on RADIO | PROPOSED |

### 5. LIVE SHADOW (no motor output): BLOCKED

The checkpoint needs 19 values (the TRACKING tab shows the same prerequisites live). Six of them are the gyro and accelerometer. **The baseline firmware sends
no IMU line**, and the only build that does is made from the Wi-Fi sketch. Nothing is invented to fill the
gap: without real IMU values the shadow tools show `IMU MISSING` and `PROPOSED none`.

| What is missing | Unblocked by |
| --- | --- |
| IMU values 0–5 from the robot | a telemetry build that does not depend on Wi-Fi (not written), **or** route D trusted again ([TEL-1](#tel-1), [IMU-1](#imu-1)) |
| IMU axis signs | [IMU-2](#imu-2) |
| Own pose 13–16 | measured marker layout ([TAG-0](#cam-3)), a saved calibration ([CAL-1](#cal-1)), [CAM-4](#cam-4) |
| Opponent 17–18 | [TRK-1](#trk-1) |

| Test ID | Prerequisite | Command or physical action | Pass condition | If it fails | State |
| --- | --- | --- | --- | --- | --- |
| [SHD-1](#shd-1) | none | `python -m pytest tests\test_live_shadow.py -q` | passes: proposals only while every input is valid `[FIXTURE]` | do not go on to SHD-2 | PASS `[FIXTURE]` |
| [SHD-2](#shd-2) | **IMU-2, CAM-4, CAL-2 and TRK-1 passed**; telemetry build on the robot; motors unplugged | terminal 1: `python scripts\telemetry_link.py --wifi`; terminal 2: `python -m control_center.tools.live_shadow --camera 1 --config config\home_test_overlay.yaml --profile home_test`; SPACE on the empty floor, then place the bot and a separate unmarked object | all four input rows `OK` with ages; `MODE LIVE SHADOW`; a `PROPOSED <tactic>` line from the checkpoint | the `BLOCKED BY:` lines name the input and the reason | BLOCKED: no IMU telemetry from the baseline firmware; camera tests not run |
| [SHD-3a](#shd-3) | SHD-2 showing proposals | cover every tag on the bot | `own pose MISSING no configured bot markers visible`; `PROPOSED none`; proposals return when uncovered | a proposal while covered is a bug: keep the trace | BLOCKED: needs SHD-2 |
| [SHD-3b](#shd-3) | SHD-2 showing proposals | slide the unmarked object against the bot, then apart | `opponent UNKNOWN overlap…`; `PROPOSED none` while merged; proposals return after separation | a proposal while merged is a bug | BLOCKED: needs SHD-2 |
| [SHD-3c](#shd-3) | SHD-2 showing proposals | stop `telemetry_link.py` (Ctrl+C), or leave the robot's Wi-Fi; start it again | within a fraction of a second `IMU MISSING stale…` then `telemetry link silent`; `PROPOSED none`; proposals return when it is back | a proposal without fresh IMU is a bug | BLOCKED: needs SHD-2 |


Known limitation, not a failure: while the two objects touch, the opponent position is unknown, so there is no
proposal during contact. That blocks a full round; the milestone here is a first live proposal.

### 6. ROUTE D: Wi-Fi handoff (EXPERIMENTAL: do not retry before [PWR-1](#pwr-1) passes)

Kept, with its tests and its record. Retrying it means flashing over the working baseline, so do it only
with time to restore ([HO-R](#ho-r)) and re-run [MD-Q](#md-q) afterwards.

| What | Value |
| --- | --- |
| Sketch | `ELRS-Handoff-WiFi\ELRS-Handoff-WiFi.ino` (as built 20 Nov 15:07: channel 6, normal transmit power) |
| PC command | `python scripts\handoff_bench_client.py --wifi --log logs\handoff-wifi-2.txt` |
| What its session showed | `python scripts\handoff_log_summary.py logs\handoff-wifi.txt` |
| Software state | `[COMPILE]` builds; `[MOCK]` SW-1 passes. **Hardware: drove, then unstable on battery** |

| Test ID | Prerequisite | Command or physical action | Pass condition | If it fails | State |
| --- | --- | --- | --- | --- | --- |
| [HO-WIFI-1](#ho-wifi-1) | Sketch above uploaded by you; laptop joined to `AutoBot-Handoff` | `python scripts\handoff_bench_client.py --wifi --monitor` | `HELLO fw=autobot-handoff-wifi-1`, then `ST`, `CH` and `NET ap=1 sta=1` lines; radio `lq=` not worse than before | network missing → `WIFI_REDUCED_TX_POWER 1`; joined but silent → allow Python in Windows Firewall; `lq` drops → `WIFI_AP_CHANNEL` 1 or 11 | **RAN 20 Nov:** link came up; `HELLO`, `ST`, `NET` lines and `lq` 98–100 % are in `logs\handoff-wifi.txt` |
| [HO-WIFI-2](#ho-wifi-2) W1–W5 | HO-WIFI-1 passed; battery, USB unplugged, wheels raised | `python scripts\handoff_bench_client.py --wifi --log logs\handoff-wifi.txt`, then ARM on, MODE to PC, SPACE+W/S/D/A | laptop drives forward, reverse, right, left and **stops on every release** | `Blocked by` line in the window says why nothing drives | NO RECORDED PASS: session of 20 Nov was unstable |
| [HO-WIFI-2](#ho-wifi-2) W4, W8 | same window | **RadioMaster → PC:** MODE to PC with nothing held; later again with the throttle and SPACE+W held | wheels still at the switch; with stick and key held: `WAIT_PC_NEUTRAL` until the keys are released, the stick does nothing | cut power; keep `logs\handoff-wifi.txt` | NO RECORDED PASS: session of 20 Nov was unstable |
| [HO-WIFI-2](#ho-wifi-2) W10 | same window | hold SPACE+W (wheels turning), disconnect the laptop from `AutoBot-Handoff` | wheels stop within about a quarter of a second; window says WI-FI LINK LOST | **cut power, then [HO-R](#ho-r)** | NO RECORDED PASS: session of 20 Nov was unstable |
| [HO-WIFI-2](#ho-wifi-2) W11–W12 | W10 done, keys still held | rejoin the network, wait 10 s, release, press SPACE+W; then ARM off, on | no wheel movement until ARM off/on **and** released keys; `Blocked by: REARM_REQUIRED`, `Fault: PC_STALE` | cut power; this is the unintended-motion case | NO RECORDED PASS: session of 20 Nov was unstable |
| [HO-WIFI-2](#ho-wifi-2) W6–W7 | PC driving | **PC → RadioMaster while the laptop is still sending:** hold SPACE+W, MODE to RADIO with sticks centred, keep holding; then use the throttle | wheels stop at once; RADIO HAS CONTROL; wheels follow the stick only | cut power; [HO-R](#ho-r) | NO RECORDED PASS: session of 20 Nov was unstable |
| [HO-WIFI-2](#ho-wifi-2) W9, W13 | PC driving | ARM off while holding SPACE+W; later transmitter off | stop at once on ARM off; stop within about a quarter of a second on transmitter off, then `REARM_REQUIRED` | if the transmitter-off stop fails, check the receiver's failsafe | NO RECORDED PASS: session of 20 Nov was unstable |
| [HO-R](#ho-r) | a failed row above | upload `ELRS-Handoff-Bench\ELRS-Handoff-Bench.ino` (tested USB build), motors unplugged | hash `1A400D3F…5FBC`; [MD-Q](#md-q) passes | upload `RadioMaster_KnownGood.ino` instead | when needed |


### 7. PARKED WITH ROUTE D: telemetry firmware and IMU

A separate build: the wireless sketch plus read-only IMU telemetry. It inherits route D's status.

| What | Value |
| --- | --- |
| Sketch | `ELRS-Handoff-Telemetry\ELRS-Handoff-Telemetry.ino` |
| Regenerate after changing the Wi-Fi password or channel | `python scripts\make_telemetry_sketch.py` |
| Receiver (the one program that owns the robot link) | `python scripts\telemetry_link.py --wifi` |
| IMU wiring it assumes | SDA GPIO6, SCL GPIO7, 100 kHz: the pins of your standalone IMU test of 17 Nov |
| Software state | `[COMPILE]` builds, 1,010,526 bytes (77 %); `[MOCK]` SW-4 passes. **Never flashed.** |

| Test ID | Prerequisite | Command or physical action | Pass condition | If it fails | State |
| --- | --- | --- | --- | --- | --- |
| [TEL-1](#tel-1) | **HO-WIFI-2 passed**; telemetry sketch uploaded by you (motors unplugged) | wheels raised, battery, USB unplugged: `python scripts\telemetry_link.py --wifi --drive`, then rows W5, W10–W12, W6–W7, W9 of HO-WIFI-2 | same results as on the handoff-only build; `fw=autobot-handoff-telemetry-1` | cut power; [TEL-R](#tel-r) back to the handoff-only build | BLOCKED: needs HO-WIFI-2 |
| [IMU-1](#imu-1) | TEL-1 passed; robot flat and still, USB unplugged | `python scripts\telemetry_link.py --wifi` | `IMU ok`, `seq` rising by about 50 a second, `t_us` rising, accel magnitude about 9.8 m/s², gyro within a few deg/s of zero; leaving the Wi-Fi reports no line, rejoining resumes with gaps counted | `IMU NOT VALID: NOT_FOUND` → wiring or pins; values near 1.0 → units; see IMU-1 | BLOCKED: needs TEL-1 |
| [IMU-2](#imu-2) | IMU-1 passed; receiver still running | second terminal: `python -m control_center.tools.live_shadow --imu-check`; flat, turn left, lift nose, lift left side | each printed description matches what you did; then set `sensor_to_body` and `verified: true` in `config\imu_mounting.yaml` | change the matrix until all four agree; a mirrored matrix is refused | BLOCKED: needs IMU-1 |
| [TEL-R](#tel-r) | any failed row here | upload `ELRS-Handoff-WiFi\ELRS-Handoff-WiFi.ino`, motors unplugged | `fw=autobot-handoff-wifi-1`; HO-WIFI-2 W5 and W10 pass again | [HO-R](#ho-r) | when needed |


<a id="aut"></a>
### 8. FIRST CLOSED-LOOP TEST (policy commands reach the motors): BLOCKED

Every row must be a hardware PASS first. Simulated inference and camera shadowing do not count toward any
of them.

| # | Requirement | Shown by | State |
| --- | --- | --- | --- |
| 1 | Human arming and emergency disarm | [MD-Q](#md-q), [MOVE-C](#move-c) disarm | reported working; MOVE-C not run |
| 2 | Positive confirmation of who has control, on the PC | `ST` lines over a link that stays up | route A only (tethered) |
| 3 | Model output maps to the right wheel directions | [MOVE-R](#move-r), [MOVE-F](#move-f), [MOVE-3](#move-3) | NOT RUN |
| 4 | Fresh, valid own pose and opponent position from the camera | [CAM-4](#cam-4), [CAL-2](#cal-2), [TRK-1](#trk-1) | NOT RUN |
| 5 | Valid IMU, as the checkpoint requires | [IMU-1](#imu-1), [IMU-2](#imu-2) | BLOCKED: no IMU telemetry |
| 6 | Motors stop on stale commands, link loss and invalid observations | [MOVE-C](#move-c); [SHD-3](#shd-3) for the observation side | NOT RUN |
| 7 | Bounded speed and duration | firmware PC limit 128 of 255; a runner that stops on an invalid observation | no policy runner exists |
| 8 | No automatic re-arming after a fault | firmware latch ([HO-USB-2](#ho-usb-2), [MOVE-C](#move-c)) | `[MOCK]` only |
| 9 | Manual takeover tested on hardware | [MOVE-C](#move-c) takeover | NOT RUN |
| 10 | Human confirmation immediately before movement | typed `GO`, as in `motion_check.py` | no policy runner exists |

**State: BLOCKED.** There is deliberately no program that sends policy output to the robot.

### 9. OPTIONAL: not a prerequisite for anything above

| Test ID | Prerequisite | Command or physical action | Pass condition | If it fails | State |
| --- | --- | --- | --- | --- | --- |
| [REC-1](#rec-1) | C525; no robot needed | `python main_control_center.py`, Record Round, Mark Round Start, Stop, Review | round folder with video, frames and events; opens in Review | `logs\`; dropped frames → disk speed | NOT RUN |
| [MOVE-2](#move-2) | [MD-Q](#md-q) passed; taped floor | full-speed physical runs on the RadioMaster, hand-timed | every run recorded | — | NOT RUN |

Venue calibration and the match-side checks are in [Appendix A](#appendix-a).

### Reference: software checks

| Test ID | Prerequisite | Command or physical action | Pass condition | If it fails | State |
| --- | --- | --- | --- | --- | --- |
| [SW-1](#sw-1) | MSVC build tools | `python -m pytest tests\test_handoff_bench.py tests\test_handoff_wifi.py -q` | all pass | do not flash that build | PASS `[MOCK]` 133 passed, 112 s (re-run the night of 20 Nov) |
| [SW-8](#sw-8) | none | `python -m pytest tests\test_arena_setup.py tests\test_cc_arena_workflow.py tests\test_cc_guided.py -q` | all pass | do not use the ARENA tab for competition | PASS `[FIXTURE + MOCK]` 119 passed (66 + 25 + 28) |
| [SW-9](#sw-9) | none | `python -m pytest tests\test_cc_control.py -q` | all pass | do not trust the ELRS & CONTROL page | PASS `[MOCK]` 58 passed |
| [SW-11](#sw-11) | none | `python -m pytest tests\test_roi_tracker.py -q` | all pass | do not use drag selection | PASS `[SYNTHETIC]` 39 passed |
| [SW-10](#sw-10) | none | `python -m pytest tests\test_cc_tracking.py -q` | all pass | do not trust the TRACKING page | PASS `[FIXTURE + MOCK]` 42 passed |
| [SW-7](#sw-7) | none | `python -m pytest tests\test_chassis_tracker.py tests\test_opponent_select.py -q` | all pass | do not use `--multi-marker`, `--click-opponent` or `live_track` | PASS `[FIXTURE]` 27 passed |
| [SW-6](#sw-6) | MSVC build tools for the firmware part | `python -m pytest tests\test_motion_check.py -q` | all pass | do not use `motion_check.py` on the robot | PASS `[MOCK + SIM]` 53 passed, 34 s |
| [SW-2](#sw-2) | none | `python -m pytest tests\test_cc_dashboard.py tests\test_cc_review.py tests\test_cc_setup.py tests\test_cc_diagnostics.py tests\test_shadow_stream.py tests\test_arena_overlay.py tests\test_calibration.py -q` | all pass | fix before any camera test | PASS `[MOCK + FIXTURE]` 137 passed, 1 skipped, 168 s |
| [SW-3](#sw-3) | none | `python -m autobot.sim.cli check` | prints `PASS: …` | sim numbers are not usable | PASS `[SIM]` |
| [SW-4](#sw-4) | MSVC build tools | `python -m pytest tests\test_handoff_telemetry.py -q` | all pass | do not flash the telemetry build | PASS `[MOCK]` 130 passed, 20 s |
| [SW-5](#sw-5) | checkpoint in `runs\` | `python -m pytest tests\test_live_shadow.py -q` | all pass | do not start SHD-2 | PASS `[MOCK + FIXTURE]` 11 passed, 59 s |
| [MD-1](#md-1), [MD-2](#md-2), [MD-3](#md-3), [MD-Q](#md-q) | battery, USB unplugged | RadioMaster only, physical | no movement at power-up; drive and stop correct | cut power; [HO-USB-0](#ho-usb-0) | Driving on battery **reported working** the night of 20 Nov; not yet run as the written tests |
| [HO-USB-0](#ho-usb-0) | bench sketch, multimeter, motors unplugged | `python scripts\handoff_bench_client.py --port COM5 --monitor` | all four driver inputs ≈ 3.3 V after every stop | do not reconnect the motors | NOT RUN |
| [HO-USB-1](#ho-usb-1) | bench sketch, USB, wheels raised | `python artifacts\code-snapshots\handoff-usb-tested-2026-11-20\handoff_bench_client.py --port COM5` | PC drives; each handoff brakes then hands over | headline and `Blocked by` line | **PASS (hardware, demonstrated)** |
| [HO-USB-2](#ho-usb-2) | HO-USB-1 passed | same window, failure cases F1–F10 | every loss or disarm stops and latches | stop; record the `ST` line | NOT RUN |

---

## 1. Setup and values you may need to change

Run every command from the repo root:

```powershell
cd path\to\AutoBots
```

Install once (includes `pyserial`, `pygame`, `mujoco`, `PySide6`, `opencv-contrib-python`):

```powershell
python -m pip install -r requirements.txt
```

| Value | Used in this guide | Where it is set |
| --- | --- | --- |
| Robot serial port | `COM5` (example) | find yours with `python scripts\handoff_bench_client.py --list` |
| C525 camera index | `1` | `config\control_center.yaml` → `camera_index`; `--camera N` / `--camera-index N` on the tools; `CAMERA_INDEX` at the top of `scriptsruco_camera_check.py` |
| Camera resolution | 1280×720, MJPG, dshow | `config\control_center.yaml` (`camera_width`, `camera_height`, `camera_fps`, `camera_backend`); `--width --height --fps --backend` on the overlay tools |
| Wi-Fi network / robot address | `AutoBot-Handoff`, `192.168.4.1`, UDP `4210` | `WIFI_AP_SSID`, `WIFI_AP_PASSWORD`, `WIFI_AP_CHANNEL`, `WIFI_UDP_PORT` near the top of `ELRS-Handoff-WiFi\ELRS-Handoff-WiFi.ino` |
| ARM / MODE switches | CH5 / CH6, HIGH = armed, HIGH = PC (defaults, see HO-USB-0) | `ARM_CHANNEL`, `MODE_CHANNEL`, `ARM_WHEN_HIGH`, `PC_WHEN_HIGH` in both handoff sketches |
| PC drive strength | 80 forward/back, 70 turn (of 255); firmware clamps to 128 | `--speed`, `--turn` on the client; `PC_COMMAND_LIMIT` in the sketch |
| Tag dictionary | `DICT_4X4_50` everywhere (arena and home configs; the real camera decoded the printed markers as this). One key, `bot.dictionary`; see `docs/aruco-tracking-audit.md` | `bot.dictionary` in the overlay config you pass (`scriptsruco_camera_check.py` and the legacy window read it from `config/arena_overlay.yaml`); `--dictionary` on the shadow demo only overrides it for one run |
| Tag black-square size | 40 mm placeholder | `bot.marker_size_m` (metres) in the overlay config; `--tag-size-mm` on the shadow demo |
| Tag layout on the bot | placeholders | `bot.markers` and `bot.marker_height_m` in the overlay config |
| Home test area | 0.50 × 0.40 m patch with one A4 sheet | `config\home_test_overlay.yaml`, profile `home_test` |
| Venue calibration | profile `live` | `config\arena_overlay.yaml` + `config\arena_overlay_clicks.json` (does not exist yet) |
| Policy checkpoint | `runs\recovery-challenger-v2-seed1\final_model.zip` (supervised clone, not the PPO run) | default of the shadow tools; `--checkpoint` |
| IMU wiring | SDA GPIO6, SCL GPIO7, 100 kHz, LSM6DS3 at 0x6A or 0x6B | `IMU_SDA_PIN`, `IMU_SCL_PIN`, `IMU_I2C_HZ` in `scripts\make_telemetry_sketch.py` (then regenerate) |
| IMU mounting | identity placeholder, `verified: false` | `config\imu_mounting.yaml`; set after [IMU-2](#imu-2) |
| Telemetry relay | `127.0.0.1:4211` on this PC | `--relay-port` on `telemetry_link.py`, `live_shadow`; `--controller-relay` on the Command Centre |
| Bristol recording | not in the repo | `$env:AUTOBOTS_FOOTAGE_MP4 = "<path to the mp4>"` |

### Firmware and client builds

| Build | File | SHA-256 (start) | State |
| --- | --- | --- | --- |
| Known-good RadioMaster-only | `ELRS-Control\ELRS-Control.ino` = your local known-good `RadioMaster_KnownGood.ino` | `6040BED3…A6DD` | your original, never edited |
| USB handoff, as tested on the bot. **On the robot since the night of 20 Nov: the provisional competition baseline** | `ELRS-Handoff-Bench\ELRS-Handoff-Bench.ino` (copy in `artifacts\code-snapshots\handoff-usb-tested-2026-11-20\`) | `1A400D3F…5FBC` | you demonstrated PC driving and both handoff directions with it over USB, and RadioMaster driving on battery |
| USB client, as tested on the bot | `artifacts\code-snapshots\handoff-usb-tested-2026-11-20\handoff_bench_client.py` | `35130DCD…91B2` | USB only |
| Wi-Fi handoff (route D, experimental) | `ELRS-Handoff-WiFi\ELRS-Handoff-WiFi.ino` | `FECDB0F7…F196` as last built (20 Nov 15:07); it changes with the password | flashed 20 Nov: drove untethered, then dropouts and restarts on battery |
| Current client (USB + Wi-Fi) | `scripts\handoff_bench_client.py` | `3C95C847…7094` | Wi-Fi path and this file's USB path are software-tested only |
| Wi-Fi handoff + IMU telemetry | `ELRS-Handoff-Telemetry\ELRS-Handoff-Telemetry.ino`, generated from the Wi-Fi sketch by `scripts\make_telemetry_sketch.py` | changes with the Wi-Fi sketch | **never flashed**; parked with route D |
| Movement check | `scripts\motion_check.py` (uses the client above for the protocol and the port) | — | software-tested only ([SW-6](#sw-6)) |
| Telemetry link owner | `scripts\telemetry_link.py` (uses the client above unchanged) | — | software-tested only |

Check which files you have before a session, and paste the hashes into the results log:

```powershell
Get-FileHash ELRS-Control\ELRS-Control.ino, ELRS-Handoff-Bench\ELRS-Handoff-Bench.ino, ELRS-Handoff-WiFi\ELRS-Handoff-WiFi.ino, scripts\handoff_bench_client.py | Format-Table Hash, Path -AutoSize
```

On the robot, the running firmware names itself in its `HELLO` line: `fw=autobot-handoff-bench-1` or
`fw=autobot-handoff-wifi-1` or `fw=autobot-handoff-telemetry-1`. The known-good sketch has no `HELLO` line.

Upload settings for every sketch (Arduino IDE, your action, **motors unplugged while uploading**): Board
"ESP32C3 Dev Module", "USB CDC On Boot: Enabled", CPU 160 MHz, Flash 80 MHz QIO 4 MB, default partition.

### How to record a result

Every test ends with a **Log** line giving its ID and evidence class. Copy a row into the
[results log](#results-log) with one of:

| Result | Meaning |
| --- | --- |
| PASS | every pass condition met |
| FAIL | a pass condition was not met; write what you saw |
| BLOCKED | could not be run: something it needs is missing (say what) |
| NOT RUN | not attempted since the last relevant change |

| Evidence class | What produced the result |
| --- | --- |
| HARDWARE | the real robot, radio, camera or network. Only you can produce these |
| MOCK | real code against a mocked board, camera or link |
| FIXTURE | synthetic rendered frames |
| FOOTAGE | the Bristol recording (historical profile, approximate map) |
| SIM | MuJoCo simulator |

---

<a id="index"></a>
## 2. What did I just change?

| I changed… | Rerun, in this order |
| --- | --- |
| **Handoff firmware** (either handoff sketch: arming, mode, stop, timeouts, protocol) | [SW-1](#sw-1) → [HO-USB-0](#ho-usb-0) if the motor or stop path changed → [HO-USB-1](#ho-usb-1) → [HO-USB-2](#ho-usb-2) → [MD-Q](#md-q). For the Wi-Fi sketch add [HO-WIFI-1](#ho-wifi-1) → [HO-WIFI-2](#ho-wifi-2) |
| **Wi-Fi settings** (password, channel, TX power) | [HO-WIFI-1](#ho-wifi-1) → [HO-WIFI-2](#ho-wifi-2) |
| **PC client** (`handoff_bench_client.py`) | [SW-1](#sw-1) → [HO-USB-1](#ho-usb-1) or [HO-WIFI-2](#ho-wifi-2), whichever link you use |
| **Flashed a different sketch / restored the baseline** | [HO-R](#ho-r) → [MD-Q](#md-q) |
| **Chassis, wheels, motors, gearing** | [MD-Q](#md-q) → [MD-3](#md-3) if the shape changed → [MOVE-2](#move-2) → [CAM-4](#cam-4) if tags moved |
| **Wiring or motor driver** | [HO-USB-0](#ho-usb-0) (motors unplugged) → [MD-1](#md-1) → [MD-Q](#md-q) |
| **Battery or power wiring** | [PWR-1](#pwr-1) → [PWR-2](#pwr-2) if the wiring changed → [MD-1](#md-1) → [MD-Q](#md-q) |
| **`motion_check.py`** | [SW-6](#sw-6) → [MOVE-C](#move-c) → [MOVE-R](#move-r) |
| **RadioMaster model, switches or mixer** | [HO-USB-0](#ho-usb-0) step A3 → [MD-1](#md-1) → [MD-Q](#md-q) |
| **Moved, re-aimed or bumped the camera; changed resolution** | [CAL-1](#cal-1) → [CAL-2](#cal-2) → new empty reference in [TRK-1](#trk-1). At the venue: [Appendix A](#appendix-a) |
| **Camera mount height or lens distance** | [CAM-2](#cam-2) → [CAL-1](#cal-1) → [CAL-2](#cal-2) |
| **Tags: reprinted, resized, moved, new dictionary** | [TAG-S](#tag-s) → update `bot.*` in the overlay config → [CAM-1](#cam-1) → [CAM-3](#cam-3) → [CAM-4](#cam-4) → [TRK-2](#trk-2) |
| **`chassis_tracker.py`, `opponent_select.py`, `marker_survey.py`, `live_track.py`** | [SW-7](#sw-7) → [SW-2](#sw-2) → [SW-5](#sw-5) → [TRK-2](#trk-2) → [TRK-3](#trk-3) |
| **Lighting or the floor surface** | [CAM-1](#cam-1) → new empty reference in [TRK-1](#trk-1) |
| **Overlay / vision code or `config\arena_overlay.yaml`** | [SW-2](#sw-2) → [CAM-3](#cam-3) → [CAL-2](#cal-2) |
| **Command Centre code** | [SW-2](#sw-2) → [REC-1](#rec-1) |
| **Shadow adapter, observation contract or checkpoint** | [SW-2](#sw-2) → [SW-5](#sw-5) → [SHD-2](#shd-2) → [SHD-3](#shd-3) |
| **Simulator physics, tactic table or `config\sim.yaml`** | [SW-3](#sw-3) → [SW-6](#sw-6) → [MOVE-1](#move-1) → [MOVE-3](#move-3) |
| **Telemetry sketch or its generator** (`make_telemetry_sketch.py`) | [SW-4](#sw-4) → [TEL-1](#tel-1) → [IMU-1](#imu-1) |
| **Wi-Fi handoff sketch** while the telemetry build is in use | `python scripts\make_telemetry_sketch.py` → [SW-4](#sw-4) → [TEL-1](#tel-1) |
| **IMU moved, re-mounted or rewired** | [IMU-1](#imu-1) → [IMU-2](#imu-2) (set `verified: false` first) → [SHD-2](#shd-2) |
| **`telemetry_link.py`, `live_telemetry.py` or `live_shadow.py`** | [SW-5](#sw-5) → [IMU-1](#imu-1) → [SHD-2](#shd-2) |
| **Nothing, I just want a sanity pass** | [SW-1](#sw-1), [SW-2](#sw-2), [MD-Q](#md-q) |

---

## 3. Test table

Moved to the top: [What to test next, in dependency order](#next). There is one table.

---

<a id="status"></a>
## 4. Where things stand on the night of 2026-11-20

| Area | Evidence so far |
| --- | --- |
| Baseline | HARDWARE (your report, no log): RadioMaster driving on battery alone on `ELRS-Handoff-Bench`, after a battery change and a fresh upload. The margin of the supply is unmeasured ([PWR-1](#pwr-1)) |
| Power | HARDWARE (reported readings): 5.4 V unloaded battery; 3.0 V in and 1.0 V out of the converter in one failing condition; 4.8 V on both converter terminals from USB alone. Root cause **not identified**. USB and battery together: unresolved ([PWR-2](#pwr-2)) |
| Wired handoff (route A) | HARDWARE: wired PC driving and both handoff directions (HO-USB-1). The failure cases have no hardware result |
| Wireless handoff (route D) | HARDWARE: flashed 20 Nov. Drove untethered; then 12 stale-command latches, 56 status dropouts and 3 logged restarts in two hours, and a restart loop afterwards. ELRS link quality stayed 98–100 % while linked. MOCK: SW-1 passes. **Experimental** |
| Relay through the transmitter (route B) | PROPOSED: desk investigation only. No script, no sender, nothing powered |
| Movement check | SIM + MOCK: reference trace and SW-6 pass, including the interlocks against the real bench sketch on a mocked board. **Not run on the robot** |
| Telemetry build | MOCK: SW-4 passes. COMPILE: builds. **Never flashed**; parked with route D. The baseline firmware has no IMU line |
| Camera and tags | Tags are on the robot (four top, four bottom, one rear, as you describe them). Dictionary, IDs, sizes, heights and offsets in both overlay configs are still placeholders. New on 20 Nov, FIXTURE only: a multi-marker tracker with explicit measured / predicted / stale / lost states, a rear-marker model, a click-selected opponent tracker and a marker survey tool ([SW-7](#sw-7)). **None of it has seen the real camera** |
| Calibration | No profile has been saved: neither `config\overlay_profiles\home_test\` nor `config\arena_overlay_clicks.json` exists |
| Live shadow | FIXTURE: SW-5 passes. **Blocked** on real IMU values and the camera tests |
| Closed loop | **Blocked**: see [the prerequisites](#aut). No program sends policy output to the robot |
| Command Centre | MOCK: SW-2 passes. Never run against the real camera or a COM port |
| Simulator | SIM: `check` passes. `calibrate` shows the sim is slower than the physical figures stored in its own code (see [MOVE-2](#move-2)) |

---

## 5. Software checks (no hardware)

<a id="sw-1"></a>
### SW-1 · Handoff firmware and client on a mocked board

Builds both real sketches for the PC and runs them one simulated millisecond at a time, then runs the
real client against them. Needs the MSVC build tools; firmware tests skip without them.

```powershell
python -m pytest tests\test_handoff_bench.py tests\test_handoff_wifi.py -q
```

- **Pass:** all tests pass. On 2026-11-20 (night, with the Wi-Fi sketch as you last built it): `133 passed in 112 s` (60 for the USB sketch, 73 for the Wi-Fi sketch).
- **If it fails:** the test name is the scenario (for example a held key through a handoff). Do not flash
  that build.
- **What it does not show:** anything about real pins, the receiver, USB, the access point or the radio.

**Log:** `SW-1 | MOCK`

<a id="cc-1"></a>
### CC-1 · The window opens with every section (mock)

```powershell
python main_control_center.py --mock
```

Eight tabs: GUIDED SETUP, OPERATE, ARENA, TRACKING, ELRS & CONTROL, SETUP, REVIEW, DIAGNOSTICS. The profile
box is disabled in mock mode; nothing is remembered. **Pass:** each tab shows content or a clear "unavailable"
message; GUIDED SETUP lists five steps with statuses. **Log:** `CC-1 | MOCK`

<a id="cc-2"></a>
### CC-2 · The real camera in GUIDED SETUP

Camera index 1 is the C525 on your PC (set in `config\control_center.yaml`). In GUIDED SETUP the right-hand
panel has the preview, index, picture size and fps. **Open camera with these settings** reopens it and remembers
the choice. Step 1 turns **Ready** when frames arrive at a usable rate and, once the arena corners exist, when
all four corners are inside the picture, the arena fills a fair share of it and a marker would be wide enough to
decode (a placeholder marker size is flagged as an estimate). **Log:** `CC-2 | HARDWARE`

<a id="arena-1"></a>
### ARENA-1 · Calibrate the arena and hazards by clicking

Steps and controls are in [command-centre.md](command-centre.md#arena-tab-marking-where-the-arena-and-its-hazards-are).
LIVE needs **clicks only**: **New set-up**; the four corners at floor level (NEAR-LEFT, NEAR-RIGHT, FAR-RIGHT, FAR-LEFT);
the centre of pit 1; the centre of pit 2; one click on the flipper; read *Calibration accuracy*; **Save**; **Quick
alignment check**; **Alignment OK, continue**. Sizes (1150 mm, R250, 335 mm legs) come from the design drawing
([arena-geometry.md](arena-geometry.md)). HOME TEST still asks for its own typed dimensions.

- **Pass:** the tab says `SAVED`, lists the arena file and the camera-corner file, and after the confirmation the chip reads
  READY. The accuracy lines show the picture scale and how far each hazard click was from the design position.
- **Expect a WARNING chip** until you confirm the overlay by eye, and `fit: WEAK` for an unreliable mapping (a very oblique view,
  a coarse far side) or when you skipped the optional rim checks.
- **If it is refused:** the amber and red lines name the reason (pits not on opposite corners, a click far from any corner, a
  mirrored corner order, a mapping the picture cannot support).
- **Measure once if in doubt:** the draft you typed earlier said 1130 mm; the drawing says 1150 mm. Measure the floor and the
  visible playing area with a tape. Both 1150: nothing to enter. Playing area 1130: *Advanced* > inset 10 mm, with a note.

**Log:** `ARENA-1 | HARDWARE` (note the profile, the picture size, the accuracy lines)

<a id="arena-2"></a>
### ARENA-2 · The camera moved

ARENA tab > **Recalibrate camera**. Four prompts, named PLAIN corner, FLIPPER corner, PIT 1 corner and PIT 2
corner, so the order does not depend on where the camera now stands. Save. **Pass:** `arena geometry left
unchanged`, only the corner file is rewritten, and the saved hazards land on the pits and flipper in the new
picture. If the arena itself changed instead, use **Edit arena** (click a row to replace one item) or a **New
set-up**. **Log:** `ARENA-2 | HARDWARE`

<a id="arena-3"></a>
### ARENA-3 · Restart and reload

Close the application, unplug and replug the camera, start the application again. The same LIVE profile opens, the camera
reopens in its calibrated mode, and ARENA > *Saved calibration* says: calibration loaded, camera settings match, alignment
(pixels). **Quick alignment check**, then **Alignment OK, continue**. Software-tested through a full restart and a simulated
camera disconnect ([tests/test_cc_persistence.py](../tests/test_cc_persistence.py)); never run on the real camera.
**Pass on the day:** a camera that did not move reads *alignment matches* within 4 px and one click reuses the calibration; a
camera that was knocked reads *Recalibrate* and the map is flagged NOT confirmed until you do. **Log:** `ARENA-3 | HARDWARE`

<a id="prof-1"></a>
### PROF-1 · HOME TEST and LIVE never mix

Switch the profile box and watch ARENA, OPERATE and TRACKING. Each loads only its own files (the table in
[command-centre.md](command-centre.md#profiles-home-test-and-live)). SETUP > Floor calibration refuses to save
while HOME TEST is active, or while an arena set-up owns the LIVE corners, and says why.
**Log:** `PROF-1 | HARDWARE`

<a id="sw-8"></a>
### SW-8 · Arena set-up, workflow, profiles and the Command Centre pages

```powershell
python -m pytest tests\test_arena_setup.py tests\test_cc_arena_workflow.py tests\test_cc_guided.py -q
```

- **Pass:** all pass. On 2026-11-21: 119 passed (66 + 25 + 28).
- Covers: pixel clicks to metres through the corner mapping; saving and loading in metres; the simulator frame
  found from the hazards (and the same arena from another side giving the same geometry); profile isolation
  (separate files, a file copied to another profile refused, the home-test config refused for LIVE, stale corner
  clicks refused); undo, reset and correcting one point; invalid, incomplete and inconsistent geometry rejected;
  camera recalibration that leaves the arena file byte-for-byte unchanged; progress and unfinished work
  surviving a restart; the camera and field-of-view checks; the GUIDED SETUP statuses and navigation; profile
  switching in the hub; the older landmark calibration refusing to cross profiles; and a check that the real
  `config\` folder is never written.
- **What it does not show:** anything about the real camera, real lighting, or where you will actually click.

<a id="sw-9"></a>
### SW-9 · ELRS & CONTROL page (mock)

```powershell
python -m pytest tests\test_cc_control.py -q
```

Pure-model and offscreen tests of the control authority display, the NOT AVAILABLE labels, restart detection, the
transport inventory, the closed AUTONOMOUS gate, handoff conflict detection and the motion-check box (its console
launch is patched out; nothing launches by itself). **Log:** `SW-9 | MOCK`

<a id="sw-11"></a>
### SW-11 · Drag-selected opponent tracker (synthetic)

```powershell
python -m pytest tests\test_roi_tracker.py -q
```

- **Pass:** all pass. On 2026-11-21: `39 passed`.
- Covers: only empty, non-numeric or off-picture rectangles refused (and a box on our own robot); objects from a
  12 px ball to a 420 px object followed; size change; flat one-colour objects; a hidden object giving UNCERTAIN
  with no position and then LOST, never adopted again; brief occlusion recovering by itself; a different object in
  the same place not adopted; reselection while tracking, after loss and after clear; the tracked centre converted
  with the calibration; an elevated object labelled image-based; our own robot never becoming the opponent; the
  OpenCV tracker used when one exists (a fake stands in, since this build has none) and a tracker that cannot start
  refusing the box; the box and events inside the pipeline and `LiveTrack`; no size limit in the source.
- **SYNTHETIC only.** Nothing here shows how it behaves on a real camera, real lighting or a real robot.
- Also exercised through the Command Centre's TRACKING tab with real mouse press/release events on the offscreen
  window (`tests\test_cc_tracking.py`).

**Log:** `SW-11 | SYNTHETIC`

<a id="sw-10"></a>
### SW-10 · TRACKING page (fixture)

```powershell
python -m pytest tests\test_cc_tracking.py -q
```

Tracker states, the drag-selected opponent, lost-track warnings, the shadow prerequisites list and the worker
thread, on rendered frames. **Log:** `SW-10 | FIXTURE`

<a id="sw-7"></a>
### SW-7 · Multi-marker tracker, clicked opponent and marker survey on synthetic frames

```powershell
python -m pytest tests\test_chassis_tracker.py tests\test_opponent_select.py -q
```

- **Pass:** all pass. On 2026-11-20: `27 passed`.
- What it covers, on rendered frames only: nine configured markers with random dropouts still give a pose;
  a marker with a wrong configured offset is rejected and named while the pose stays right; two markers
  that disagree with no track give no pose; a total dropout reads PREDICTED, then STALE, then LOST at the
  documented times, also when frames stop arriving altogether; a one-frame jump is rejected and a
  persistent new place is accepted; bottom markers read as inverted; a rear marker alone gives a pose only
  when the camera position is known; a stale or lost pose never reaches the observation. For the opponent:
  a click selects it; a click on the own robot or bare floor is refused; contact gives no position; a lost
  object is not swapped for another one; the survey finds the dictionary and never writes a config.
- Fixture figures (ideal camera, exact calibration; **not** real-camera accuracy): chassis error up to
  2.5 mm measured and 3.8 mm predicted; rear marker alone up to 3.4 mm and 2.5°; clicked opponent about
  36 mm off, the expected bias of a raised object projected to the floor; about 11 ms of software per frame.
- **What it does not show:** detection rate, accuracy, or the click tracker's thresholds under real light.

**Log:** `SW-7 | FIXTURE`

<a id="sw-6"></a>
### SW-6 · Movement check on the simulator and a mocked board

```powershell
python -m pytest tests\test_motion_check.py -q
```

- **Pass:** all pass. On 2026-11-20: `53 passed in 34 s`.
- What it covers: the plan uses the simulator's own tactic table and stays inside the firmware's PC limit;
  the simulator moves the commanded way; nothing but zero is sent before `GO`; the sequence stops for good
  on disarm, MODE away from PC, a latched fault, radio loss, a changed token, status silence, outputs the
  step did not command, and the keyboard; the five safety checks against the **real bench sketch on a mocked
  board**; a floor run is refused until those checks have passed on the same firmware within 12 hours.
- **What it does not show:** that a wheel turns, which way it turns, or anything about the supply.

**Log:** `SW-6 | MOCK + SIM`

<a id="sw-2"></a>
### SW-2 · Command Centre, overlay and shadow on mocks and fixtures

```powershell
python -m pytest tests\test_cc_dashboard.py tests\test_cc_review.py tests\test_cc_setup.py tests\test_cc_diagnostics.py tests\test_shadow_stream.py tests\test_arena_overlay.py tests\test_calibration.py -q
```

- **Pass:** all pass. On 2026-11-20: `137 passed, 1 skipped in 307 s`. The skip is the recorded-footage
  test in `tests\test_shadow_stream.py`, which needs `$env:AUTOBOTS_FOOTAGE_MP4`.
- `tests\test_overlay_footage.py` (recorded footage) was not rerun today.
- **If it fails:** fix before opening the real camera; the live tools share this code.

**Log:** `SW-2 | MOCK + FIXTURE`

<a id="sw-3"></a>
### SW-3 · Simulator self-check

```powershell
python -m autobot.sim.cli check
```

- **Pass:** `PASS: Gymnasium checker, seeded reset, observation separation, shapes, and render-independent trajectory determinism`.
- `tests\test_sim.py` covers more; not rerun today.

**Log:** `SW-3 | SIM`

<a id="sw-4"></a>
### SW-4 · Telemetry firmware on a mocked board

Builds `ELRS-Handoff-Telemetry.ino` and the handoff-only Wi-Fi sketch for the PC with the same harness, plus a
mocked LSM6DS3 on I2C.

```powershell
python -m pytest tests\test_handoff_telemetry.py -q
```

- **Pass:** all tests pass. On 2026-11-20: `130 passed, 20 s`.
- What it shows, on the model only:
  - every USB and Wi-Fi handoff scenario still passes with the IMU present, absent, and with its bus hung;
  - the four motor pins follow exactly the same trace, millisecond by millisecond, as the handoff-only build;
  - a failed or absent sensor is sent as `ok=0 why=…` with no numbers, never as zeros;
  - never more than one I2C transaction per loop pass, each bounded by a 5 ms timeout.
- **If it fails:** do not flash the telemetry build. If the "plus insertions" test fails, regenerate:
  `python scripts\make_telemetry_sketch.py`.

**Log:** `SW-4 | MOCK`

<a id="sw-5"></a>
### SW-5 · Telemetry on the PC and the live shadow path, on mocks and fixtures

```powershell
python -m pytest tests\test_live_shadow.py -q
```

About a minute (it loads the checkpoint). `$env:AUTOBOT_SKIP_SLOW = "1"` skips the checkpoint run.

- **Pass:** all pass. On 2026-11-20: `11 passed, 59 s`.
- What it shows: the sketch's IMU lines parse on the PC (MOCK); the relay delivers them to local listeners and
  accepts nothing back; units and the mounting transform are applied; and on synthetic frames with scripted
  telemetry (FIXTURE) the real checkpoint proposes only while IMU, own pose and opponent are all valid. Covered
  tags, touching objects, a failed IMU, interrupted telemetry and a late datagram each block with their reason,
  and proposals return afterwards.
- It is not evidence about the real camera, IMU or robot.

**Log:** `SW-5 | MOCK + FIXTURE`

---

## 5a. Power checks (group 0)

Added after 20 November. A multimeter and a pen. These exist because the supply collapsed that day and
nobody had a number for it beforehand.

<a id="pwr-1"></a>
### PWR-1 · Battery and rail under load, on battery alone

USB unplugged for all of it. Write every reading down with the time and which pack is fitted.

| # | Do | Record | Stop if |
| --- | --- | --- | --- |
| P1 | Pack disconnected from the robot, rested a few minutes | voltage at the pack | it is well below the pack's nominal voltage: charge it before anything else |
| P2 | Pack connected, robot on, disarmed, transmitter on | pack voltage; converter input; converter output | the output is not at the converter's rated voltage |
| P3 | Wheels raised, ARM on, hold a small throttle for 10 s | the same three, while the wheels run | the output dips, the receiver LED blinks off, or the robot restarts |
| P4 | Wheels raised, full throttle forward for 3 s, then reverse | the same three; lowest value seen | as P3 |
| P5 | Hold each wheel lightly by hand for a second at small throttle (a stall draws the most current) | the same three | as P3 |

- **Pass:** the converter output stays at its rated voltage through P2–P5 and the robot never restarts.
- **If it fails:** that pack, or the wiring to it, is not fit to compete on. The difference between the
  pack reading and the converter-input reading at the same moment tells you where the volts go: a pack
  that sags by itself is the pack; a pack that holds while the converter input sags is the switch, a
  connector or a wire.
- A multimeter averages. A dip shorter than about a fifth of a second will not show; a restart is the
  evidence for those.

**Log:** `PWR-1 | HARDWARE` (the readings, the pack, the time)

<a id="pwr-2"></a>
### PWR-2 · How USB and the battery are joined

Robot unpowered, USB unplugged, battery disconnected. This answers whether a wired test may have the
battery connected at all.

1. Look at where the converter's output goes: the board's 5 V pin or its 3.3 V pin? Note it.
2. Is there a diode between the converter output and the board, or between USB and the board? Note it.
3. Where does the motor driver's supply pin connect: straight to the pack, or to the converter?
4. The reading you already have: with USB alone, 4.8 V on both converter terminals. So USB reaches the
   pack side through the converter, and through it whatever else hangs on that node.

- **If there is no diode and one rail:** treat USB and battery as never to be connected together. Wired
  tests run on `--power usb-only` with the pack unplugged, and the wheels may still turn (the driver can
  be fed from USB through the path above).
- **If a diode or separate rails keep them apart:** write down which, and `--power battery-and-usb` is
  yours to choose. The tool will ask you to confirm it each time.
- This guide does not say that connecting both is safe, or that it is harmful. It has not been measured.

**Log:** `PWR-2 | HARDWARE` (what you found; a photo of the wiring)

---

## 6. Manual drive baseline (group 1)

Battery power, **USB unplugged**, RadioMaster only. Works with any of the three sketches; write down
which one is on the board. On the night of 20 November it is `ELRS-Handoff-Bench` (ARM off then on, MODE = RADIO).
Do [PWR-1](#pwr-1) first: a robot that drives for a minute has not shown that its supply holds.

- Known-good sketch: no ARM switch. It drives once it has a radio link and centred sticks.
- Either handoff sketch: ARM off then on, MODE = RADIO, sticks centred. Without USB you cannot see
  status lines (with the Wi-Fi sketch you can: `--wifi --monitor`, [HO-WIFI-1](#ho-wifi-1)).

No software is involved, so there is no command.

<a id="md-1"></a>
### MD-1 · Startup idle

Setup: wheels raised on a block, power cut-off in reach, USB unplugged.

| # | Do | Pass |
| --- | --- | --- |
| 1 | Transmitter **off**. Switch the robot on. Watch for 10 s. | No wheel moves, including at the instant of power-up |
| 2 | Transmitter on, sticks centred (handoff sketches: ARM off). Watch for 10 s. | No wheel moves |
| 3 | Handoff sketches only: ARM on with sticks centred. | No wheel moves |
| 4 | Switch the robot off and on again with the transmitter on and sticks centred (handoff sketches: ARM still on). | No wheel moves. Handoff sketches then need ARM off, on before they drive |

**If it fails:** cut power. A twitch of the left wheel exactly at power-up or reset points at AIN2 on
GPIO2, a boot strapping pin; measure it with the motors unplugged (HO-USB-0, step A2 of
`ELRS-Handoff-Bench\README.md`). Continuous motion with centred sticks: check stick trims and the channel
calibration (174 / 992 / 1811, deadzone 45).

**Log:** `MD-1 | HARDWARE`

<a id="md-2"></a>
### MD-2 · Upright drive: forward, reverse, left, right, stop

Do it once with the wheels raised, then on the floor.

| # | Do | Pass |
| --- | --- | --- |
| 1 | Small throttle forward, release | Both wheels forward; stop on release |
| 2 | Small throttle reverse, release | Both wheels backward; stop on release |
| 3 | Steer left, release | Left wheel back, right wheel forward (nose turns left on the floor); stop on release |
| 4 | Steer right, release | Mirror of 3; stop on release |
| 5 | Hold a small throttle, then switch the **transmitter off** (wheels raised for this one) | Wheels stop within about a quarter of a second and stay stopped when the transmitter returns |
| 6 | Handoff sketches only: hold a small throttle, ARM off | Wheels stop at once |

Step 5 matters: a reading of the ESP32 core source predicts the known-good sketch may not brake after a
drive command or on radio loss. That prediction has never been measured and does not match your
experience of the robot, so this step is the measurement.

**If it fails:** a wheel turning the wrong way is a swapped motor plug or polarity. No stop on release or
on transmitter off: cut power, do not put the robot on the floor, and do HO-USB-0. With a handoff sketch,
after step 5 the robot needs ARM off, on before it drives again; that is by design.

**Log:** `MD-2 | HARDWARE`

<a id="md-3"></a>
### MD-3 · Inverted drive

No sketch in the repo knows which way up the robot is (none reads the IMU), so the firmware does nothing
different. This test records what the chassis actually does on its back.

Setup: on the floor, robot upside down, same sketch and switches as MD-2.

| # | Do | Record |
| --- | --- | --- |
| 1 | Sticks centred | Both wheels reach the floor? Anything else touching? |
| 2 | Throttle forward, release | Which end leads; stops on release? |
| 3 | Throttle reverse, release | Which end leads; stops on release? |
| 4 | Steer left, release | Turn direction seen from above |
| 5 | Steer right, release | Turn direction seen from above |

- **Pass:** it drives in all four directions, stops on release, and the mapping you wrote down matches
  the last recorded one.
- **Expected from geometry, not measured:** forward stick moves it tail-first, and the turn direction seen
  from above is unchanged.
- **If it fails:** a wheel not reaching the floor is a chassis or wheel-size matter; record it and treat
  inverted driving as unavailable.

**Log:** `MD-3 | HARDWARE`

<a id="md-q"></a>
### MD-Q · Quick rerun (after any chassis, wiring or firmware change)

Battery, USB unplugged, wheels raised first.

1. Power on with sticks centred: no movement (startup idle).
2. Forward, release.
3. Reverse, release.
4. Left, release.
5. Right, release.
6. Hold a small throttle, transmitter off: wheels stop. Transmitter on (handoff sketches: ARM off, on).
7. Five minutes of ordinary driving with the wheels raised: no restart, and the receiver never drops out.

Then repeat 2–5 on the floor. All seven correct = PASS. Anything odd: run MD-1 and MD-2 in full.

**Log:** `MD-Q | HARDWARE`

---

## 7. Wired and wireless handoff (group 2)

USB and Wi-Fi are separate tests with separate results. A USB pass says nothing about Wi-Fi, and the
demonstration you already did covers the normal path only, not the failure cases.

**After 20 November:** the Wi-Fi tests below (HO-WIFI, TEL, IMU) are route D, experimental. They were
written before the session in which that build became unstable on battery, and they are kept as written.
Do not start them before [PWR-1](#pwr-1) passes, and expect to restore the baseline afterwards.

The RadioMaster is the only authority in both sketches: it arms and it selects RADIO or PC. The PC can
only send drive commands.

Client keys: hold **SPACE** (deadman) with **W/A/S/D** or the arrows. **P** stops sending (dead-client
test), P again resumes. **Esc** quits. Clicking another window sends zero.

<a id="ho-usb-0"></a>
### HO-USB-0 · Pin-level stop check, motors unplugged

Run this after changing motor pins, the stop path, driver wiring, or the RadioMaster switches. Unplug both
motors from the DRV8833 outputs first. Multimeter on AIN1 (GPIO3), AIN2 (GPIO2), BIN1 (GPIO4), BIN2 (GPIO5).

```powershell
python scripts\handoff_bench_client.py --list
```

```powershell
python scripts\handoff_bench_client.py --port COM5 --monitor
```

`--monitor` over USB sends nothing. Follow steps A1–A9 in
[ELRS-Handoff-Bench/README.md](../ELRS-Handoff-Bench/README.md#a-motors-disconnected). The three results
that matter:

- **A2:** AIN2 and AIN1 both about 0 V while the reset button is held.
- **A3:** the `CH` line shows `arm=CH5:191:DISARM` / `CH5:1792:ARM` and `mode=CH6:191:RADIO` /
  `CH6:1792:PC` as you flip your switches. If not, edit the four defines and start again.
- **A5, A6, A8:** all four inputs about 3.3 V after a released stick, after ARM off, and after the
  transmitter goes off.

**If it fails:** leave the motors unplugged. Record the `ST` and `CH` lines and the four voltages.

**Log:** `HO-USB-0 | HARDWARE`

<a id="ho-usb-1"></a>
### HO-USB-1 · Wired PC drive and both handoff directions

Prerequisites: `ELRS-Handoff-Bench.ino` on the board (`HELLO fw=autobot-handoff-bench-1`), USB plugged
in, wheels raised, Arduino Serial Monitor closed. **Power:** you ran USB and the battery together for this
test before the readings of 20 November showed USB reaching the battery side of the converter. Until
[PWR-2](#pwr-2) has an answer, run it on USB power with the pack unplugged.

The exact client you tested:

```powershell
python artifacts\code-snapshots\handoff-usb-tested-2026-11-20\handoff_bench_client.py --port COM5 --log logs\handoff-usb.txt
```

The current client in USB mode (same protocol; software-tested only until you run it):

```powershell
python scripts\handoff_bench_client.py --port COM5 --log logs\handoff-usb.txt
```

| # | Do | Pass |
| --- | --- | --- |
| U1 | Transmitter on, ARM off, MODE = RADIO. Open the window. | NOBODY HAS CONTROL, `Blocked by: DISARMED`. Wheels still |
| U2 | ARM on, sticks centred. Small throttle forward, reverse, left, right. | RADIO HAS CONTROL. Wheels follow and stop on release |
| U3 | **RadioMaster → PC**, nothing held: MODE to PC. | PC HAS CONTROL, `Last stop: HANDOFF`, `L=0 R=0` |
| U4 | SPACE+W, release. Then S, A, D. | Slow drive in each direction (`L=80 R=80` forward); stop on release. If 80 does not turn the wheels, restart with `--speed 110` |
| U5 | **PC → RadioMaster**, nothing held: MODE to RADIO. | RADIO HAS CONTROL, `Last stop: HANDOFF`. Sticks drive again |
| U6 | ARM off. | `Blocked by: DISARMED` |

**If it fails:** the headline and the `Blocked by` line say why nothing drives. `NO STATUS FROM THE
BOARD`: wrong port, or the Serial Monitor holds it. `WAIT_PC_NEUTRAL`: release every key once.
`MODE_SWITCH_INVALID` or `ARM_SWITCH_INVALID`: a three-position switch in the middle, or the wrong channel
(HO-USB-0 step A3).

Optional: watch the same session in the Command Centre (it only reads the client's log):

```powershell
python main_control_center.py --controller-log logs\handoff-usb.txt
```

**Log:** `HO-USB-1 | HARDWARE` (already demonstrated once; enter the date and build)

<a id="ho-usb-2"></a>
### HO-USB-2 · Wired failure cases

Same window and setup as HO-USB-1, wheels raised. None of these has a hardware result yet.

| # | Do | Pass |
| --- | --- | --- |
| F1 | ARM on, then power-cycle the board with ARM still on. Push the throttle. | `Blocked by: ARM_ON_AT_BOOT`. Wheels still until ARM off, on |
| F2 | RADIO mode: hold the throttle, ARM off. Still holding, ARM on. Then centre. | Stop at once. `WAIT_RADIO_NEUTRAL` while held. Drives only after centring |
| F3 | PC mode: hold SPACE+W, flip MODE to RADIO **with the throttle pushed**. Centre the stick. | Stop, `Last stop: HANDOFF`. `WAIT_RADIO_NEUTRAL` until centred. The held keys do nothing |
| F4 | Keys still held, flip MODE to PC. Then release the keys. | Stop. `WAIT_PC_NEUTRAL` and "release all keys" until released |
| F5 | PC mode: hold SPACE+W, press P. Press P again with the keys held. | Wheels stop within about a quarter of a second. `Fault: PC_STALE`, `Blocked by: REARM_REQUIRED`. Still stopped after the second P. The radio stick does nothing |
| F6 | ARM off, on. | `WAIT_PC_NEUTRAL` until the keys are released, then PC HAS CONTROL |
| F7 | PC mode: hold SPACE+W, ARM off. | `Blocked by: DISARMED`, `L=0 R=0` although the PC keeps sending |
| F8 | PC mode: hold SPACE+W, transmitter off. Transmitter on. | Stop within about a quarter of a second, `RADIO_LOST`. After it returns: `REARM_REQUIRED` |
| F9 | PC mode, driving: close the window (Esc). Reopen it. | Wheels stop. On reopening: `Fault: PC_STALE`, `REARM_REQUIRED` |
| F10 | PC mode, driving: pull the USB cable. | Wheels stop within about a quarter of a second |

**Pass:** every row. Faults latch: only ARM off then on clears them, and no fault falls back to the other
source.

**If it fails:** cut power, keep the log file, and note the last `ST` line. If F8 does not stop, your
receiver keeps sending frames after the transmitter goes off: stop testing and check the receiver's
failsafe setting.

**Log:** `HO-USB-2 | HARDWARE`

<a id="ho-wifi-1"></a>
### HO-WIFI-1 · Wi-Fi bring-up, listen only

Sketch: `ELRS-Handoff-WiFi\ELRS-Handoff-WiFi.ino`. First flashed on 20 November; the link came up and the
robot drove, then the session became unstable (see [the summary](#nov20)).

1. **Set the password.** Edit `WIFI_AP_PASSWORD` near the top of the sketch (8 to 63 characters). The value
   in the repo is a placeholder.
2. **Upload** (your action): motors unplugged, settings as in section 1. Then reconnect the motors with
   the board unpowered and raise the wheels.
3. **Power the robot from the battery.** USB may stay unplugged from here on.
4. **Join the network.** Laptop Wi-Fi list → `AutoBot-Handoff` → password → Connect. Windows says "No
   internet, secured"; stay connected. Untick "Connect automatically" on your other networks for the test,
   or Windows may switch back. The robot is `192.168.4.1`.
5. Listen (sends `?` once a second, never a drive command):

```powershell
python scripts\handoff_bench_client.py --wifi --monitor
```

- **Pass:** `HELLO fw=autobot-handoff-wifi-1`, then `ST`, `CH` and `NET ap=1 sta=1 peer=192.168.4.2:… ` lines.
  With the transmitter on, the `CH` line's `lq=` and `rc_age=` look the same as they do over USB with Wi-Fi
  off, and MD-Q still passes.
- **If the network does not appear:** set `WIFI_REDUCED_TX_POWER` to `1` in the sketch and upload again
  (present in the sketch, not tried on this robot).
- **If you are joined but nothing prints:** Windows Firewall is probably dropping the replies (the network
  is "Public"); allow Python on it.
- **If `lq` drops or the radio stop misbehaves:** change `WIFI_AP_CHANNEL` to 1 or 11.
- **If the robot resets under load:** battery sag (it boots disarmed). Record it.

**Log:** `HO-WIFI-1 | HARDWARE`

<a id="ho-wifi-2"></a>
### HO-WIFI-2 · Wi-Fi acceptance: battery, USB physically unplugged, wheels raised

Prerequisites: HO-WIFI-1 passed, **USB cable unplugged from the robot**, battery power, wheels raised,
transmitter on, ARM off, MODE = RADIO, laptop joined to `AutoBot-Handoff`.

```powershell
python scripts\handoff_bench_client.py --wifi --log logs\handoff-wifi.txt
```

| # | Do | Pass |
| --- | --- | --- |
| W1 | Look at the window. | NOBODY HAS CONTROL, `Blocked by: DISARMED`, `PC link: Wi-Fi 192.168.4.1:4210 (UDP)`. Wheels still |
| W2 | Hold SPACE+W, release. | Nothing moves (not armed) |
| W3 | ARM on, sticks centred. Drive forward, reverse, left, right on the sticks. Then SPACE+W on the laptop. | RADIO HAS CONTROL; wheels follow the sticks. The laptop does nothing (`PC commands: not accepted`) |
| W4 | **RadioMaster → PC**, nothing held: MODE to PC. | PC HAS CONTROL, wheels still, `robot says the PC input is held by: WIFI` |
| W5 | **Keyboard driving:** SPACE+W, release. S, D, A. | Forward, reverse, right, left at low speed; stop on every release |
| W6 | **PC → RadioMaster with a key held:** hold SPACE+W (wheels turning), MODE to RADIO, sticks centred. Keep holding. | Wheels stop at once. RADIO HAS CONTROL. The held keys do nothing |
| W7 | **Inactive PC sender:** still holding the laptop keys, push the throttle, then centre it. | Wheels follow the stick only and stop when it is centred. Release the keys |
| W8 | **RadioMaster → PC with stick and key held:** hold the throttle and SPACE+W, MODE to PC. Release the keys, then centre the stick. | Wheels stop at the switch. `WAIT_PC_NEUTRAL` until the keys are released, then PC HAS CONTROL. The stick does nothing in PC mode |
| W9 | **Disarm:** hold SPACE+W (wheels turning), ARM off. Keys still held, ARM on. Release. | Stop at once, `DISARMED`. After ARM on: `WAIT_PC_NEUTRAL` until the keys are released |
| W10 | **Loss of the PC link:** hold SPACE+W (wheels turning), disconnect the laptop from `AutoBot-Handoff`. Keep holding. | Wheels stop within about a quarter of a second. WI-FI LINK LOST |
| W11 | **Reconnect:** still holding, rejoin the network. Wait 10 s. Release, then SPACE+W again. | Wheels stay stopped the whole time. NOBODY HAS CONTROL, `Blocked by: REARM_REQUIRED`, `Fault: PC_STALE`. Keys still do nothing |
| W12 | ARM off, on (keys released). | PC HAS CONTROL again. Wheels still |
| W13 | **Radio loss in PC mode:** hold SPACE+W, transmitter off. Transmitter on. ARM off. | Stop within about a quarter of a second. After it returns: `REARM_REQUIRED`. Ends `DISARMED` |

Two shorter link-loss variants, same pass condition as W10–W12: press **P** while driving (client stops
sending), and close the window with **Esc** while driving.

**Pass:** every row, with no wheel movement at any point that a held key or stick from *before* a switch,
disarm, loss or reconnect could have caused.

**If it fails:**

- W10 does not stop the wheels: cut the power and restore the USB version ([HO-R](#ho-r)).
- NO STATUS FROM THE ROBOT while joined: firewall, or Windows moved to another network.
- Wheels move at W11: cut power; this is the unintended-motion case. Keep `logs\handoff-wifi.txt`.
- `other=` climbing in the `NET` line: something is also sending over USB; unplug it.

Only after a full pass with wheels raised: repeat W3–W7 and W10 on the floor in a clear area.

**Log:** `HO-WIFI-2 | HARDWARE`

<a id="tel-1"></a>
### TEL-1 · Telemetry build: wireless regression (queued until HO-WIFI-2 passes)

The telemetry build is the wireless sketch plus IMU reading and two extra lines. This test checks that the
wireless behaviour you passed in HO-WIFI-2 is unchanged on it. Do not run it before HO-WIFI-2 has passed.

1. If you changed the Wi-Fi password or channel in the wireless sketch, regenerate so the telemetry build has
   the same:

```powershell
python scripts\make_telemetry_sketch.py
```

2. **Upload** `ELRS-Handoff-Telemetry\ELRS-Handoff-Telemetry.ino` (your action): motors unplugged, settings as
   in section 1. Reconnect the motors with the board unpowered. Battery on, USB unplugged, wheels raised.
3. Join `AutoBot-Handoff`, then start the link owner with the drive window:

```powershell
python scripts\telemetry_link.py --wifi --drive --log logs\telemetry-regression.txt
```

It is the same window and keys as HO-WIFI-2 (it runs the tested client code on the same link). Run these rows
of [HO-WIFI-2](#ho-wifi-2), in this order:

| # | Rows | Pass |
| --- | --- | --- |
| T1 | W1, W4, W5 | laptop drives and stops on release; afterwards the `HELLO` line in `logs	elemetry-regression.txt` says `fw=autobot-handoff-telemetry-1` |
| T2 | W10, W11, W12 | link loss stops the wheels within about a quarter of a second; no motion on reconnect; ARM off/on needed |
| T3 | W6, W7 | RadioMaster takeover with a key held stops at once; the laptop cannot move the wheels |
| T4 | W9 | ARM off stops at once |

- **Pass:** identical to HO-WIFI-2. The IMU being present, absent or faulty must make no difference here.
- **If it fails:** cut power and go back to the handoff-only build ([TEL-R](#tel-r)). Keep the log.
- **Not measured by the mocked tests:** the IMU read holds the loop for about 1.6 ms fifty times a second (5 ms
  at most if the bus is stuck). T2 and T4 are where that would show.

**Log:** `TEL-1 | HARDWARE`

<a id="imu-1"></a>
### IMU-1 · Untethered six-axis readings, timestamps, units and stale-data reporting

Prerequisites: TEL-1 passed. Robot on battery, **USB unplugged**, flat on the table, not moving. Motors may stay
unplugged. Laptop joined to `AutoBot-Handoff`. Close the drive window first: one program owns the link.

```powershell
python scripts\telemetry_link.py --wifi
```

It sends only `?` once a second, never a drive command, and prints a summary each second:

```
14:03:22  fw=autobot-handoff-telemetry-1  CONTROLLER requested RADIO, confirmed source NONE, disarmed, last stop NONE, applying L=0 R=0
          IMU ok  seq 1042  t_us 20875123  accel +0.12 -0.05 +9.79 m/s^2  gyro +0.31 -0.42 +0.10 deg/s  (sensor axes)
          1042 valid, 0 failed, 0 late/repeated, 0 gaps  |  relay 127.0.0.1:4211: 0 listener(s)
```

| # | Do | Pass |
| --- | --- | --- |
| I1 | Leave the robot flat and still for 10 s | `IMU ok`; `seq` rises by about 50 each second; `t_us` rises by about 1,000,000 each second |
| I2 | Read the accelerometer | one axis near ±9.8 m/s², the other two near 0 (which axis depends on the mounting); the magnitude is 9.3 to 10.3 |
| I3 | Read the gyro | all three within about ±3 deg/s (bias); they move when you turn the robot by hand |
| I4 | Flip MODE and ARM on the RadioMaster (motors unplugged or wheels raised) | the CONTROLLER line follows: requested mode, confirmed source, arm state, last stop |
| I5 | **Stale data:** disconnect the laptop from `AutoBot-Handoff` for 5 s, then rejoin | while away: `no line from the robot`; after: `IMU ok` again and `gaps` has gone up. No values are printed while away |
| I6 | Optional, power off first: unplug the IMU, power on | `IMU NOT VALID: the controller reports NOT_FOUND (no values were sent)`; the robot still drives by RadioMaster |

- **Pass:** I1 to I5.
- **If it fails:**
  - `IMU NOT VALID: … NOT_FOUND` with the sensor fitted: wiring, or SDA/SCL are not GPIO6/GPIO7. Edit
    `IMU_SDA_PIN` / `IMU_SCL_PIN` in `scripts\make_telemetry_sketch.py`, regenerate, upload again.
  - magnitude near 1.0: the values are in g, not m/s²; do not continue.
  - `no IMU line: this firmware does not send IMU telemetry`: the handoff-only build is on the board.
  - `gaps` climbing steadily while joined: Wi-Fi loss; note the rate.
- The two clocks are not synchronised: `t_us` is the controller's, the time at the left is the PC's.

**Log:** `IMU-1 | HARDWARE`

<a id="imu-2"></a>
### IMU-2 · Axis signs: stationary and hand-rotation check

The policy expects the robot body frame: **x toward the nose, y to the robot's left, z up**. The sensor's own
axes depend on how it is mounted, which is not recorded anywhere. This test finds the transform. Until it
passes, `config\imu_mounting.yaml` says `verified: false` and live shadow stays blocked.

Keep the receiver from IMU-1 running. In a second terminal:

```powershell
python -m control_center.tools.live_shadow --imu-check
```

Twice a second it prints the body-frame values and what they mean. Move the robot by hand:

| # | Do | The line must say |
| --- | --- | --- |
| X1 | Flat on the table, upright, still | `STILL AND FLAT: gravity is on body +z` |
| X2 | Turn it to the left (counter-clockwise seen from above), flat on the table | `ROTATING … about body +z -> reads as turning LEFT` |
| X3 | Lift the nose (tail stays down) until the robot stands almost on its tail, then hold it there | while moving: `about body -y -> reads as nose pitching UP`; held: `gravity is mostly on body +x -> reads as nose pointing UP` |
| X4 | Lift the left side until the robot stands almost on its right side, then hold it there | while moving: `about body +x -> … (left side rising)`; held: `gravity is mostly on body +y -> reads as LEFT side up` |

- Held at a small tilt the line only says `tilted`; go past 45° for X3 and X4.
- If a line names a different axis or direction, edit `imu.sensor_to_body` in `config\imu_mounting.yaml` (each
  row says which sensor axis a body axis is, with its sign; examples are in the file), restart the check and
  repeat all four. A mirrored matrix is refused when the file is loaded.
- **Pass:** all four agree. Then set `verified: true` in the file and write the matrix in the results log.
- **If it cannot be made to agree:** two body axes read the same sensor axis, or a value is not moving: go back
  to IMU-1.

**Log:** `IMU-2 | HARDWARE` (note the matrix)

<a id="tel-r"></a>
### TEL-R · Back to the handoff-only wireless build

Upload `ELRS-Handoff-WiFi\ELRS-Handoff-WiFi.ino` (your action), motors unplugged, settings as in section 1.
Then `python scripts\handoff_bench_client.py --wifi --monitor` must show `HELLO fw=autobot-handoff-wifi-1`, and
rows W5 and W10 of HO-WIFI-2 must pass again. Further back: [HO-R](#ho-r).

**Log:** `TEL-R | HARDWARE` (note which sketch is now on the board)

<a id="ho-r"></a>
### HO-R · Restore the tested baseline

Uploading is your action. Motors unplugged, settings as in section 1.

| Go back to | Open and upload | Then |
| --- | --- | --- |
| Tested USB handoff | `ELRS-Handoff-Bench\ELRS-Handoff-Bench.ino` | MD-Q, then HO-USB-1 with the snapshot client |
| RadioMaster only | your local known-good `RadioMaster_KnownGood.ino` | MD-Q |

Confirm the file is the tested one before uploading:

```powershell
(Get-FileHash ELRS-Handoff-Bench\ELRS-Handoff-Bench.ino).Hash
```

Expect `1A400D3FDF66D0A718D6C96166A6A7A891641EC47A93F31C5FEFEA84D8CB5FBC`.

```powershell
(Get-FileHash ELRS-Control\ELRS-Control.ino).Hash
```

Expect `6040BED377E40AFFA3DC4105D23E96EACB788E6E5AFCF4FCAA60CDBFCFDEA6DD`.

**Log:** `HO-R | HARDWARE` (note which sketch is now on the board)

---

## 8. Camera and ArUco (group 3)

Use the mounted C525 and the real tags. Close the Command Centre first: only one program can hold the camera.
None of these tools opens the robot link. None has been run on the real camera yet.

<a id="cam-1"></a>
### CAM-1 · Uncalibrated detection

```powershell
python scriptsruco_camera_check.py
```

- Camera index is hard-coded on line 26 (`cv2.VideoCapture(1, cv2.CAP_DSHOW)`). The dictionary is no longer hard-coded: it is the
  project dictionary (`bot.dictionary` in `config/arena_overlay.yaml`, `DICT_4X4_50`). Edit the camera line if yours differs. `q` quits.
- It runs at the camera's default resolution, not the 1280×720 the other tools request, so tags look
  smaller here than in CAM-3.

| # | Do | Pass |
| --- | --- | --- |
| 1 | One tag flat in the middle of the view | Its ID and a red centre dot, steady for 10 s; `Tags: 1` |
| 2 | Every tag on the bot in view | `Tags:` equals the number visible; no wrong IDs |
| 3 | Cover all tags | `No tags detected` |

**If it fails:** nothing decoded → wrong dictionary first, then focus, glare on glossy paper, or a missing
white border round the tag. Wrong IDs → a different family is being matched.

**Log:** `CAM-1 | HARDWARE`

<a id="cam-2"></a>
### CAM-2 · Focus and field of view at working distance

Mount the C525 where you intend to use it. Keep `scriptsruco_camera_check.py` running.

| Position of the tag | Distance from the lens (measure it) | Decodes steadily? | Sharp? |
| --- | --- | --- | --- |
| Nearest edge of the area | | | |
| Centre | | | |
| Farthest corner | | | |
| Farthest corner, tag tilted as on a moving bot | | | |

- **Pass:** the whole area is in view with margin on every side, and the tag decodes steadily in all four
  rows. Repeat row 3 with the bot upside down if you rely on bottom tags.
- **If it fails:** far tags not decoding → lower or move the camera, print larger tags, or check
  the apparent size with the marker survey (the markers are already `DICT_4X4_50`, the dictionary with the fewest, largest cells; the real camera decoded
  them as that in `artifacts/marker-survey`). Focus hunting → the tools do not set focus; fix it in the Logitech or Windows camera settings.

**Log:** `CAM-2 | HARDWARE`

<a id="cam-3"></a>
### CAM-3 · Calibrated pose

Needs a saved calibration ([CAL-1](#cal-1)) and the real tag values in the `bot:` block of the config you
pass:

| Key | Meaning |
| --- | --- |
| `bot.dictionary` | any `cv2.aruco` `DICT_*` name |
| `bot.marker_size_m` | side of the **black square**, metres |
| `bot.marker_height_m` | height of the tag surface above the floor, per face |
| `bot.markers` | one entry per tag: `id`, `face`, `forward_m`, `left_m`, `yaw_deg` from the bot centre |

`config\home_test_overlay.yaml` assumes a single tag at the centre of the object (ids 0–5, all at offset
zero). For the real bot with several tags, enter the measured layout first.

```powershell
python -m control_center.tools.arena_overlay --camera 1 --config config\home_test_overlay.yaml --profile home_test
```

- **Pass:** a green `BOT: x … m  y … m  nose heading … deg  UPRIGHT` line, a `detection age` in
  milliseconds, and `bot markers seen:` listing the tags in view. Turn the bot over: `INVERTED` (needs
  bottom-face entries in `bot.markers`).
- **If it fails:**
  - `conflict: marker N measures NN mm but is configured as 40 mm` → set `bot.marker_size_m`.
  - `no configured bot markers visible` with tags in view → dictionary, or the ids are not in `bot.markers`.
  - `no valid floor mapping` → run CAL-1.
  - `POSITION: APPROXIMATE` → expected until the camera position is measured or derived.
  - `PLACEHOLDER config in use` → lists the config groups still unmeasured.

Keys: `p` saves a snapshot (PNG + JSON) to `artifacts\vision-overlay-snapshots\`, `h` toggles the status
text, `q` quits.

**Log:** `CAM-3 | HARDWARE`

<a id="cam-4"></a>
### CAM-4 · One tag at a time, cover all, uncover

Same window as CAM-3. Put the bot down and **do not touch it again** until the table is filled; press `p`
at each row to keep the numbers.

| Visible | x (m) | y (m) | heading (deg) |
| --- | --- | --- | --- |
| All tags | | | |
| Only tag … | | | |
| Only tag … | | | |
| Only tag … | | | |
| None (all covered) | `unknown` | `unknown` | `unknown` |
| All uncovered again | | | |

- **Pass:** every single-tag row is within 20 mm and 10° of the all-tags row (the tool's own conflict
  limits, `pose.max_position_disagreement_m` and `pose.max_heading_disagreement_deg`; tighten them if you
  need better). With all tags covered the pose reads `unknown` at once and is not held. It returns on the
  first frame after uncovering, to the same values as before.
- **If the centre or heading jumps when the visible tag changes:** that is an error in `bot.markers`
  (offset or `yaw_deg`), not movement. Re-measure from the bot centre and repeat.
- With `config\arena_overlay.yaml` and the `live` profile, the Command Centre automates this: **SETUP →
  Markers → Single-tag check**, Capture once per tag (20 samples each); it reports the spread between tags.

**Log:** `CAM-4 | HARDWARE`

---

## 9. Home calibration and live tracking (group 4)

A measured flat area with its own profile, `home_test`, stored in
`config\overlay_profiles\home_test\clicks.json`. It never touches the venue profile (`live`).

**Since 21 November the Command Centre's ARENA tab does this click-calibration for either profile** (see
[ARENA-1](#arena-1)): it stores the arena size, both pits and the flipper in metres in a separate `arena.json`
next to the camera corners, and the commands below keep working for the A4-sheet landmark method.

<a id="cal-1"></a>
### CAL-1 · Home calibration

Setup: C525 fixed on its stand, looking down at a plain table or floor at an angle. One sheet of A4
(297 × 210 mm) lying **landscape** in the middle of the view. Optional: rule two pencil lines corner to
corner so the centre is marked.

```powershell
python -m control_center.tools.arena_overlay --camera 1 --config config\home_test_overlay.yaml --profile home_test --calibrate
```

The picture freezes on the first frame and click mode is on. Click the highlighted landmark, **at floor
level**, in this order:

| Order | Landmark | Where |
| --- | --- | --- |
| 1 | `a4_near_left` | sheet corner nearest the camera, left |
| 2 | `a4_near_right` | nearest the camera, right |
| 3 | `a4_far_right` | far side, right |
| 4 | `a4_far_left` | far side, left |
| 5 | `a4_centre` (check point, not used in the fit) | where the two pencil lines cross; press `n` to skip it |

Then `s` to save and `q` to quit. Other keys: `u` undo, `x` clear, `1`–`5` jump to a landmark, `f`
freeze / unfreeze.

- **Pass:** `saved 5 point(s) to clicks.json`, a `MAP: fit 4 pts | held-out check 1 pts: max … mm` line,
  and the drawn outline sitting on the patch. `FIT: WEAK` with "no check point clicked" is expected if you
  skipped step 5.
- **If it fails:** `NOT saved: …` gives the reason. `mapping is mirrored` → corners clicked in the wrong
  order. `fit residual … exceeds 30.0 mm` → a click was off a corner.
- Redo it whenever the camera moves or the resolution changes; the tool refuses points from another
  image size.

**Log:** `CAL-1 | HARDWARE`

<a id="cal-2"></a>
### CAL-2 · Known floor position after fitting

Frame of the home patch: +x to the right, +y away from you, metres. The sheet's near-left corner is at
(0.1015, 0.095), so a mark **a** mm right and **b** mm beyond that corner is at
(0.1015 + a/1000, 0.095 + b/1000). The sheet centre is (0.250, 0.200).

1. Read the `MAP:` line: the held-out check error at `a4_centre`.
2. Tape a cross somewhere **off** the sheet's corners, measure a and b, and stand the tagged object's
   centre on it. Run with its coordinates (example: the sheet centre):

```powershell
python -m control_center.tools.arena_overlay --camera 1 --config config\home_test_overlay.yaml --profile home_test --truth 0.250,0.200
```

- **Pass:** held-out check ≤ 10 mm (`mapping.warn_check_error_mm`) and the `truth error:` line ≤ 10 mm.
  Add a heading as a third number (`--truth 0.250,0.200,90`) to check heading too.
- **If it fails:** an error pointing toward or away from the camera means the tag height
  (`bot.marker_height_m`) is wrong; an error in any direction means the calibration clicks. Redo CAL-1.

**Log:** `CAL-2 | HARDWARE`

<a id="trk-1"></a>
### TRK-1 · Read-only live tracking: tagged object and unmarked object

Needs CAL-1, a tagged object, and an unmarked object that contrasts with the surface. The robot is not
involved and nothing is sent anywhere. Take the A4 sheet and everything else off the patch first.

```powershell
python -m control_center.tools.shadow_stream_demo --camera 1 --config config\home_test_overlay.yaml --profile home_test --tag-size-mm 40
```

Set `--tag-size-mm` to your black-square side; add `--dictionary NAME` only to try a different dictionary for one run (the
default is the project's `DICT_4X4_50`). Keys: **SPACE** take the empty-floor reference, **r** take it again, **q** quit.

| # | Do | Pass |
| --- | --- | --- |
| 1 | **Empty reference:** patch empty, hands out of view, press SPACE | "reference taken: put the objects on the floor" |
| 2 | **Separation:** place both objects at least 25 cm apart | `[13-16] own pose OK ArUco ids […]`, the tagged object's circle labelled `own`; `[17-18] opponent OK unmarked blob T…` on the other |
| 3 | **Tag loss:** cover the tag | own pose `MISSING no configured bot markers visible` on the next frame. Your hand is a blob too, so the opponent row usually reads `UNKNOWN`; that is expected |
| 4 | **Tag recovery:** uncover | own pose `OK` again at once |
| 5 | **Overlap:** slide the objects together | a few centimetres before touching: `OVERLAP: no position`, opponent `UNKNOWN overlap: …`. Own pose stays `OK` while the tag is visible |
| 6 | **Recovery:** slide them about 10 cm apart | opponent `OK` within a frame or two; `own` is on the blob with the tag |
| 7 | **Loss:** lift the unmarked object out | opponent `UNKNOWN no separate unmarked blob …` |
| every step | | `[0-5] IMU MISSING`, `MODE BLOCKED`, `PROPOSED none` |

A trace is saved under `artifacts\shadow-stream\camera-<time>\`. Write down "software time this frame"
and any circle on something that is not an object (a shadow or reflection).

**Streams that must keep shadow mode blocked:**

| Missing or stale | Values | Shown as |
| --- | --- | --- |
| IMU (gyro, accelerometer) | 0–5 | `MISSING no IMU telemetry (none connected)`. Always in this tool: it has no telemetry input ([SHD-2](#shd-2) does) |
| Own pose: no tag, conflict, no map, or older than 0.25 s | 13–16 | `MISSING` with the reason |
| Opponent: overlap, no other blob, more than one other blob, no tag to say which blob is yours, no reference | 17–18 | `UNKNOWN` with the reason |

**If it fails:** a proposal appearing in this mode is a bug (the checkpoint is never loaded here). False
blobs → retake the reference after any lighting or camera change. Objects not picked up → more contrast.
A new track will not start within 0.20 m of an existing one, hence the 25 cm.

**Log:** `TRK-1 | HARDWARE`

<a id="tag-s"></a>
### TAG-S · Which markers are actually on the robot

The robot now carries four markers on top, four underneath and one on the rear face, believed to be ID 8.
Nothing in the configs matches that yet, and none of it has been checked. This tool reads; it never writes
a config.

```powershell
python -m control_center.tools.marker_survey --camera 1 --seconds 10
```

Hold the robot still with the top up, run it; flip it, run it again; then with the rear toward the camera.
Add `--config config\home_test_overlay.yaml --profile home_test` once CAL-1 is saved to get sizes in mm.

- It tries eight dictionaries on every frame and prints, per dictionary and ID: frames seen out of frames
  taken (that percentage **is** the intermittency you have been seeing), the apparent side in pixels, and
  which dictionary best explains the detections. It flags IDs it saw that are not in the config and
  configured IDs it never saw. Output: `artifacts\marker-survey\<time>\survey.json` and an annotated frame.
- **Pass:** you can fill in this table from what it printed and what you measured with a ruler.

| ID | Face (top / bottom / rear) | Black square side (mm) | Height above the floor, that face up (mm) | Forward of chassis centre (mm) | Left of centre (mm) | Rotation (deg) | Seen in % of frames |
| --- | --- | --- | --- | --- | --- | --- | --- |
| | | | | | | | |

Then put those numbers into `bot:` of the overlay config ([TAG-0](#cam-3)). A rear marker is written as, for
example, `{id: 8, face: rear, forward_m: -0.075, left_m: 0.0, yaw_deg: 0.0, size_m: 0.040, height_m: 0.026,
height_inverted_m: 0.026}` with **your** numbers; `height_m` is the height of its centre with the robot
upright. `config\home_test_overlay.yaml` as it stands puts ids 0–5 all at the centre, so with several real
markers in view the tracker will, correctly, report that they disagree.

**Log:** `TAG-S | HARDWARE` (the table)

<a id="trk-2"></a>
### TRK-2 · Multi-marker chassis tracking with explicit states

Needs TAG-0 done for all nine markers and CAL-1 saved. Read-only: no robot link, no checkpoint.

```powershell
python -m control_center.tools.live_track --camera 1 --config config\home_test_overlay.yaml --profile home_test
```

| State on the `OWN TRACK` line | Meaning | May feed the policy's observation? |
| --- | --- | --- |
| `MEASURED` | a consistent set of markers was seen in this frame | yes |
| `PREDICTED` | no measurement now; position carried forward from the last ones, for at most 0.20 s | yes, labelled `PREDICTED nn ms` |
| `STALE` | 0.20 s to 1.0 s since the last measurement; the last pose is drawn for reference only | **no** |
| `LOST` | more than 1.0 s; no pose | **no** |

| # | Do | Pass |
| --- | --- | --- |
| 1 | Robot still, top up | `MEASURED`, the ids of the visible top markers, a steady position |
| 2 | Cover one or two markers with a finger | still `MEASURED`; the position does not step by more than about 20 mm |
| 3 | Cover every marker | `PREDICTED`, then `STALE` within a quarter of a second, then `LOST` at one second |
| 4 | Uncover | `MEASURED` on the next frames |
| 5 | Flip the robot | orientation reads inverted; position stays within about 20 mm of where the chassis centre is |
| 6 | Turn the rear toward the camera so only the rear marker shows | either a pose labelled `rear marker only: LOWER CONFIDENCE`, or a stated reason for rejecting it. Compare with a tape measure |
| 7 | Slide the robot quickly by hand | the track follows; no frozen position anywhere |

- **If it fails:** a `rejected: id: …` line names the marker whose offset in `bot.markers` is wrong. A pose
  that jumps when the visible set changes is a wrong offset or height, not movement.
- The rear-marker maths depends on the camera position worked out from the calibration. It has only ever
  run on rendered frames. Treat row 6 as a measurement, not a pass condition.
- The same tracker in the shadow tools is opt-in: add `--multi-marker` to `live_shadow` or
  `shadow_stream_demo`. Without the flag they behave as before.

**Log:** `TRK-2 | HARDWARE`

<a id="trk-3"></a>
### TRK-3 · Drag-selected opponent (TRACKING tab)

Command Centre > TRACKING, with the camera and calibration of the active profile. No empty-floor picture and no
size setting: you choose the object by dragging a box around it.

| # | Do | Pass |
| --- | --- | --- |
| 1 | Press **Select Opponent / Draw Box**, then drag a rectangle around an unmarked object (a phone, a ball, a robot: try each) | `TRACKING STATUS: TRACKED` with a confidence; a box and a centre cross on the object; a position under OPPONENT |
| 2 | Move the object around, slowly then quickly | the box follows; the confidence stays above 0.60 while it is clearly visible |
| 3 | Cover it with a hand for under a second, then uncover | `TRACKING UNCERTAIN` **with no position** while hidden, `TRACKED` again by itself |
| 4 | Cover it for two seconds | `LOST`; it stays `LOST` when it reappears |
| 5 | Drag a new box (**Reselect Opponent**) around it | `TRACKED` again, without touching the camera or calibration |
| 6 | Drag a box on your own tagged robot | `box REFUSED`, the current track is left alone |
| 7 | Drag a tiny box (a couple of pixels) | refused as empty; nothing else changes |
| 8 | Press **Clear Opponent** | `NOT SELECTED` |
| 9 | Lift the object 10 cm off the floor | still tracked, position labelled image-based; note how far it moves from where the floor point is |

- The position is the centre of the tracked box projected onto the floor plane: **image-based and approximate**,
  not a physical measurement, and not corrected for height. It has no heading.
- **Expect to reselect during a match.** A long push or an occlusion can end in `LOST`; that is deliberate.
  The tracker will not guess, and it will not adopt another object.
- Its thresholds (0.60 confident, 0.35 floor, 0.8 s) have never met real lighting, shadows or a multi-coloured
  robot. Write down what fools it. A plain floor patch that resembles the box is rejected by a distinctiveness
  check, but that has only been tested on synthetic floors.
- The command-line viewers keep the older click-selected tracker (with its size checks) behind
  `--click-opponent`; the Command Centre does not use it.

**Log:** `TRK-3 | HARDWARE`

---

## 10. Movement comparison: sim vs physical (group 5)

One short, fixed sequence, the same in the simulator and on the robot:

```
STOP 1.0 s · ADVANCE 0.6 s · STOP 1.0 s · TURN_LEFT 0.4 s · STOP 1.0 s · TURN_RIGHT 0.4 s · STOP 1.0 s · REVERSE 0.6 s · STOP 1.0 s
```

- The tactics and their wheel commands are the simulator's own (`autobot.sim.env.tactic_to_drive`), scaled
  to **40 %** so the largest command is 102 of 255, under the firmware's PC limit of 128.
- The commands go out in the existing PC format (`D,<tok>,<throttle>,<steer>`, 50 times a second) through the
  tested client's code. Each step is recorded with its tactic, duration, source and time.
- **What is compared:** forward or reverse, the sign of each turn, and roughly how far and how much.
  **What is not expected:** equal distances. The simulator's mass, drive gain and friction are placeholders.
- Sign conventions: simulator wheel commands are −1..1 with **negative = forward**; the firmware's `L=` /
  `R=` are −255..255 with positive = forward; positive steer = right.

`motion_check.py` never arms, never selects PC mode and never clears a fault: those are your switches.
Starting it moves nothing. It sends zero until the firmware itself reports armed, PC selected and outputs at
zero, then waits for you to type `GO`. During a run, Enter or Ctrl+C stops it, and **ARM off always stops
the robot** whatever the laptop is doing.

<a id="move-1"></a>
### MOVE-1 · Simulator reference (no hardware)

```powershell
python scripts\motion_check.py plan
```

```powershell
python scripts\motion_check.py sim
```

Output on 2026-11-20 (`config\sim.yaml`; the frozen experiment config gives the same):

| Step | Sent (throttle, steer) | Firmware L / R | Simulator: forward | Simulator: heading change |
| --- | --- | --- | --- | --- |
| advance 0.6 s | 102, 0 | 102 / 102 | +0.127 m | 0° |
| turn_left 0.4 s | 0, −71 | −71 / 71 | about 0 | +38.5° (left) |
| turn_right 0.4 s | 0, 71 | 71 / −71 | about 0 | −37.0° (right) |
| reverse 0.6 s | −66, 0 | −66 / −66 | −0.076 m | 0° |

`--scale 0.50` is the most the firmware accepts; `--sequence "stop:1.0,advance:0.4,stop:1.0"` changes the
sequence (moves of at most 1 s, each followed by a stop of at least 0.5 s).

**Log:** `MOVE-1 | SIM`

<a id="move-c"></a>
### MOVE-C · Five safety checks, wheels raised

Prerequisites: [PWR-2](#pwr-2) answered. `ELRS-Handoff-Bench` on the robot. USB plugged in, Arduino Serial
Monitor closed. **Wheels off the ground.** Power cut-off in reach. Transmitter on, ARM off.

```powershell
python scripts\handoff_bench_client.py --list
```

```powershell
python scripts\motion_check.py checks --port COM5 --power usb-only
```

It asks you to confirm the wheels are raised, then runs five checks. Before each one: ARM off, MODE = PC,
ARM on, nothing held, then type `GO`.

| # | Check | You do | The controller must report | You must see |
| --- | --- | --- | --- | --- |
| 1 | stop | nothing | the commanded outputs, then `L=0 R=0` within 0.3 s of the stop command | wheels forward 0.6 s, then still |
| 2 | disarm | ARM **off** while the wheels turn | `armed=0`, `stop=DISARM`, `L=0 R=0` | wheels stop at once |
| 3 | takeover | MODE to **RADIO**, sticks centred, while the wheels turn | `req=RADIO`, `stop=HANDOFF`, `L=0 R=0` | wheels stop at once; the sticks then drive |
| 4 | pc_link_loss | nothing: the program stops sending on purpose | `fault=PC_STALE`, `L=0 R=0` within 0.5 s | wheels stop by themselves in about a quarter of a second |
| 5 | radio_loss | transmitter **off** while the wheels turn | `link=0` / `fault=RADIO_LOST`, `L=0 R=0` | wheels stop in about a quarter of a second |

- **Pass:** five lines of `controller evidence: PASS`, five "yes" answers, and `Floor runs are now permitted`.
  The record is `artifacts\motion-check\checks-<time>\checks.json`.
- **If a check fails:** cut power. The line after `controller evidence: FAIL` says what was reported. Check 5
  failing with "status was lost" means the board restarted: that is a power result, go to [PWR-1](#pwr-1).
- If 40 % does not turn the wheels, add `--scale 0.50`.
- After check 4 the firmware holds a `PC_STALE` fault: ARM off, then on.
- **On `--power usb-only` the motors are fed, if at all, from the laptop's USB port backwards through the
  converter** (the 4.8 V reading). That path was never meant to carry motor current. Use a powered hub if
  you have one. Wheels that will not turn, or a board that restarts when they try, are a supply result,
  not a software one: write it down and stop. Whether the wheels turn on USB alone has not been tried.
- `--power battery-and-usb` exists for when [PWR-2](#pwr-2) has shown the two are isolated. The tool asks
  you to confirm that each time and will not start on a "no".

**Log:** `MOVE-C | HARDWARE` (one line per check)

<a id="move-r"></a>
### MOVE-R · The sequence with the wheels raised

Same setup as MOVE-C.

```powershell
python scripts\motion_check.py run --port COM5 --surface raised --power usb-only
```

Arm with MODE on PC, wait for `READY`, type `GO`, and watch each wheel. When `COMPLETE` appears, switch ARM
off. Then open `observed.yaml` in the folder it names and fill in, for each moving step, `left_wheel` and
`right_wheel` as `forward`, `reverse` or `still`.

| Step | Left wheel | Right wheel |
| --- | --- | --- |
| advance | forward | forward |
| turn_left | reverse | forward |
| turn_right | forward | reverse |
| reverse | reverse | reverse |

- **Pass:** [MOVE-3](#move-3) reports `SIGNS AGREE on every moving step (wheel directions only: wheels were raised)`.
- **If a wheel turns the wrong way:** stop. One of the sign conventions between the simulator and the wheels
  is not what the code assumes, and a policy would steer the wrong way.

**Log:** `MOVE-R | HARDWARE`

<a id="move-f"></a>
### MOVE-F · The sequence on the floor

Only after MOVE-C has fully passed on this firmware within the last 12 hours; the tool checks and refuses
otherwise. A clear, flat floor with half a metre free all round. Tape a cross under the robot's centre and
a line along its nose. Film it from above if you can. On USB the cable must be slack and long enough for
about 20 cm of travel and a 40° turn.

```powershell
python scripts\motion_check.py run --port COM5 --surface floor --power usb-only
```

Fill in `observed.yaml`: `direction` (`forward`, `reverse`, `turn_left`, `turn_right`, `none`),
`distance_mm` (how far the centre moved, positive), `rotation_deg` (+ = left), `stopped_when_command_ended`
(`true` / `false`), and `measured_by`.

- **Pass:** `SIGNS AGREE on every moving step`, and the robot stopped between moves.
- Distances and rotations that differ from the simulator are expected. They are the calibration result.

**Log:** `MOVE-F | HARDWARE`

<a id="move-3"></a>
### MOVE-3 · Comparison

```powershell
python scripts\motion_check.py compare artifacts\motion-check\<run folder>
```

It prints, and writes to `comparison.md` in that folder, one row per moving step: what the simulator did,
whether the controller reported the commanded outputs, what you saw, and for a floor run the ratio of
physical to simulated distance and rotation.

| Verdict | Meaning |
| --- | --- |
| `SIGNS AGREE on every moving step` | direction and turn sign match the simulator everywhere |
| `INCOMPLETE: n of m …` | some steps have no observation; nothing is assumed for them |
| `MISMATCH: …` | a wheel, a direction or a rotation sign disagrees, or the controller did not report the command. Do not go on |

The ratios have no pass threshold. Write them in the results log; they are what a later simulator retune
would aim at.

**Log:** `MOVE-3 | SIM + HARDWARE`

<a id="move-2"></a>
### MOVE-2 · Optional: full-speed runs on the RadioMaster, hand-timed

The simulator's `calibrate` command compares exact timed windows with physical figures stored in its own code:

```powershell
python -m autobot.sim.cli calibrate
```

| Window | Command | Simulated | "physical" printed beside it |
| --- | --- | --- | --- |
| 3.00 s | advance | 1.6424 m | 2.0300 m |
| 1.81 s | advance | 0.9863 m | 1.2192 m |
| 4.65 s | turn_left | 3.432 rotations (4.637 rad/s) | 8.25 rotations (11.1 rad/s) |

The repo does not record when or how the "physical" figures were measured. To repeat them: a flat floor
with 3 m clear, two tape lines 1.22 m apart, a phone filming. Full throttle on the RadioMaster from rest
across both lines (time between them from the video); full steer on the spot for about 3 s (rotations ÷
time). Three repeats each. Collisions are left out: discard any run that touches something.

**Log:** `MOVE-2 | HARDWARE`

---

## 11. Recording and shadow mode (group 6)

<a id="rec-1"></a>
### REC-1 · Record → Mark Start → Stop → Review, real camera, no bot

No robot, no COM port. Close any other program using the camera.

```powershell
python main_control_center.py
```

It opens camera index 1 from `config\control_center.yaml`; add `--camera-index 0` if the wrong camera opens.

| # | Do | Pass |
| --- | --- | --- |
| 1 | Look at OPERATE | `Camera live · … fps`. Control source `UNKNOWN`, robot link `Not connected`, IMU `No IMU stream (no controller link)`, opponent `Not available live`, policy `OFF`. No arm, mode or drive control anywhere |
| 2 | Type a label (optional), press **RECORD ROUND** (F8) | Recording indicator and elapsed time |
| 3 | After a few seconds press **MARK ROUND START** (F9 or Space) | Confirmation in green |
| 4 | Wave a hand in view for 10 s, press **STOP RECORDING** | A summary naming the folder under `rounds\`, with dropped frames and write errors if any |
| 5 | Press **Review the round just saved** | REVIEW opens it: the video scrubs, the Events list has recording start, round start and recording stop |

Check the folder: `manifest.json`, `video.avi`, `frames.csv`, `events.jsonl`. No `imu.csv` (no controller)
and no `opponent.csv` (no live tracker) is correct. To reopen a round later without the camera:

```powershell
python main_control_center.py --review rounds\<folder>
```

- **Pass:** all five rows, zero dropped frames (or the count reported honestly), and the round reopens.
- **If it fails:** `Camera lost` or `No camera` → index, or another program holds it. Dropped frames → disk
  speed; MJPG at 720p is about 3 MB per second. Details are in `logs\`.
- The same flow without hardware, for checking the UI only (evidence MOCK, labelled MOCK everywhere):

```powershell
python main_control_center.py --mock
```

**Log:** `REC-1 | HARDWARE` (camera only)

<a id="shd-1"></a>
### SHD-1 · Fixture shadow (synthetic frames, scripted telemetry)

Read-only. Opens no serial port, controller link or camera.

```powershell
python -m pytest tests\test_live_shadow.py -q
```

This is [SW-5](#sw-5). It runs the same loop as the live runner on rendered frames and scripted IMU lines in the
firmware's format, through the real checkpoint. The older stand-alone demo still works and writes a video:

```powershell
python -m control_center.tools.shadow_stream_demo --skip-footage --no-video --out artifacts\shadow-stream-check
```

What the checkpoint expects, field by field:

```powershell
python -m control_center.tools.observation_contract
```

**Log:** `SHD-1 | FIXTURE`

<a id="shd-2"></a>
### SHD-2 · Live shadow: first real proposal (no motor output)

**Prerequisites, all on hardware:** [IMU-2](#imu-2) passed (`verified: true`), [CAM-4](#cam-4), [CAL-2](#cal-2)
and [TRK-1](#trk-1) passed, the telemetry build on the robot, your tag values in `config\home_test_overlay.yaml`.
Robot on battery, **USB unplugged**, **motors unplugged**, standing on the calibrated patch. A separate unmarked
object ready. Laptop joined to `AutoBot-Handoff`. Close the Command Centre and any other camera program.

Terminal 1, the link owner (receive only):

```powershell
python scripts\telemetry_link.py --wifi
```

Terminal 2, the shadow runner (camera + checkpoint; listens to terminal 1 on this PC only):

```powershell
python -m control_center.tools.live_shadow --camera 1 --config config\home_test_overlay.yaml --profile home_test
```

Add `--tag-size-mm NN` or `--dictionary DICT_…` only if you have not put them in the config. Keys: **SPACE**
empty-floor reference, **r** again, **q** quit. The checkpoint takes about a minute to load; the panel says so.

| # | Do | Expect on the panel |
| --- | --- | --- |
| S1 | Floor empty, hands out of view: press SPACE | "reference taken: put the objects on the floor" |
| S2 | Put the robot on the patch | `[13-16] own pose OK ArUco ids […]`; the IMU rows show sensor and body values, `mounting VERIFIED` |
| S3 | Put the unmarked object at least 25 cm away | `[17-18] opponent OK unmarked blob T…` |
| S4 | Wait for "checkpoint loaded" | `[0-5] IMU OK … age NN ms`, then `MODE LIVE SHADOW: proposal only, nothing is sent` and `PROPOSED <tactic> (p = …)` |
| S5 | Move the unmarked object around the robot by hand | the proposed tactic changes with where the object is (for example a turn toward it) |

- **Pass:** at least one `PROPOSED` line with all four input rows `OK`, the checkpoint named as
  `recovery-challenger-v2-seed1`, and the CONTROLLER line still reporting from the robot.
- **If it is blocked:** the `BLOCKED BY:` lines give the input and reason. The common ones:

| Line | Meaning | Fix |
| --- | --- | --- |
| `imu: nothing received from the relay …` | terminal 1 is not running | start `telemetry_link.py` |
| `imu: IMU mounting transform is not verified …` | IMU-2 not done | do IMU-2 |
| `imu: the controller reports the IMU as …` | sensor fault; no values were sent | IMU-1 |
| `imu: stale …` / `telemetry link silent …` | Wi-Fi dropped | rejoin; watch `gaps` |
| `own_pose: no configured bot markers visible` | tag not seen | dictionary, ids, focus |
| `own_pose: tag switch: position stepped …` | tag offsets disagree | re-measure `bot.markers` (CAM-4) |
| `opponent_position: …` | no reference, no other blob, more than one, or touching | SPACE again; one unmarked object only |

- A trace is written to `artifacts\live-shadow\<time>\trace.jsonl` with a `summary.json`.
- At home the positions are metres on the home patch, read in the simulator's 1.15 m arena frame (the patch
  sits in the plain corner). The proposal is real; it is not a proposal for the venue arena.
- There is no motor output: the runner has no socket to the robot, and `telemetry_link.py` without `--drive`
  sends only `?`.

**Log:** `SHD-2 | HARDWARE`

<a id="shd-3"></a>
### SHD-3 · Live shadow: interrupt each input separately

With SHD-2 showing proposals. One interruption per trial; wait for proposals to return before the next.

| Trial | Do | Pass while interrupted | Pass afterwards |
| --- | --- | --- | --- |
| SHD-3a | Cover every tag on the robot | `own pose MISSING no configured bot markers visible`; `PROPOSED none (missing values 13-16)` | proposals return on the first frames after uncovering |
| SHD-3b | Slide the unmarked object against the robot | `opponent UNKNOWN overlap…`; `PROPOSED none (missing values 17-18)`; no position shown for it | proposals return a frame or two after they are about 10 cm apart |
| SHD-3c | Ctrl+C in terminal 1 (or leave the robot's Wi-Fi) | `IMU MISSING stale: the newest IMU sample is … ms old`, then `telemetry link silent …`; `PROPOSED none (missing values 0-5)` | start terminal 1 again: proposals return |

- **Pass:** in every trial there is no `PROPOSED <tactic>` line while the input is invalid, and the reason shown
  is the right one.
- **Limitation to keep on the list:** contact makes the opponent position unknown, so shadow is silent during
  pushing. That is correct behaviour and an open blocker for running a whole round.

**Log:** `SHD-3 | HARDWARE` (one line per trial)

---

<a id="appendix-a"></a>
## Appendix A · Competition day

Only what changes at the venue. Everything else is the procedure linked.

**1. Camera: mount, then calibrate from where it actually stands.** Nothing from home or from the Bristol
recording carries over. Tape two or three crosses on the arena floor, measure each from the x = 0 and y = 0
walls, and put the measured positions into the `tape_centre`, `tape_1` … entries of `reference_points` in
`config\arena_overlay.yaml` before you start (their current positions are placeholders).

```powershell
python -m control_center.tools.arena_overlay --camera 1 --calibrate
```

Click, at floor level, in the order the tool highlights: `corner_origin`, `pitA_rim_x0_wall`,
`pitB_rim_y0_wall`, `flipper_edge_ymax_wall`, `flipper_edge_xmax_wall` (fit), then `pitA_rim_ymax_wall`,
`pitB_rim_xmax_wall`, `tape_centre` and `tape_1`–`tape_4` (check). `n` skips one you cannot see, `t` swaps a point
between fit and check, `s` saves. Same procedure as [CAL-1](#cal-1). The Command Centre's **SETUP → Floor
calibration** tab does the same thing on the same file.

**2. Check a point that was not used in the fit.** The `MAP:` line must show a held-out check within
10 mm, and the outline must sit on the floor edges. Then stand the bot on a taped cross and run
[CAL-2](#cal-2) with `--truth` and without the `--config` / `--profile` options. Then [CAM-4](#cam-4).

**3. Power, then the radio, from the operator position.** [PWR-1](#pwr-1) rows P1–P3 on the pack you will
use. Then, standing where you will stand during the match, [MD-Q](#md-q) with the wheels raised. PC mode
is not planned for matches: no untethered PC route is verified. (Before 20 November this step was the
Wi-Fi acceptance test; that route is now experimental.)

**4. Confirm the stop.** Wheels raised: ARM off stops the wheels; transmitter off stops the wheels; after
either, nothing moves until ARM has been off and on again. Leave MODE on RADIO.

**Pre-match go / no-go**

| # | Check | Go |
| --- | --- | --- |
| 1 | Sketch on the board is the one you tested (results log names it) | yes |
| 2 | MD-Q passed today on this battery, USB unplugged, with no restart | yes |
| 2a | PWR-1 readings for this battery written down; converter output steady with the wheels running | yes |
| 3 | ARM off stops the wheels; transmitter off stops the wheels | both |
| 4 | MODE to RADIO takes over with a laptop key held | yes |
| 5 | PC mode | **not for matches**: stay in RADIO |
| 6 | Calibration saved today, held-out check within 10 mm (if using the camera) | yes, or run without the map |
| 7 | Recording started and the camera row is green (if recording) | optional |

Any "no" on 1–3: do not run. Change the battery first, then restore the firmware ([HO-R](#ho-r)), one at a
time, and write down which one fixed it. Rows 6–7 only limit what you use.

**RadioMaster fallback**

- **During a match:** MODE stays on RADIO. ARM off stops everything. No laptop is needed.
- **Between matches, if the handoff build misbehaves:** upload the known-good sketch ([HO-R](#ho-r)),
  motors unplugged, then MD-Q. It has no ARM or MODE switch.

---

<a id="results-log"></a>
## Appendix B · Results log

### Hardware results

| Date | Firmware / build (sketch + hash start, client) | Test ID | Result | Notes |
| --- | --- | --- | --- | --- |
| 2026-11-20 (early) | `ELRS-Handoff-Bench` 1A400D3F, snapshot client | HO-USB-1 | PASS (demonstrated, your report) | wired PC drive, both handoff directions; USB and battery together |
| 2026-11-20 12:48–14:48 | `ELRS-Handoff-WiFi` (password build), current client | HO-WIFI-1 | RAN: link came up | `HELLO`, `ST`, `NET ap=1 sta=1` and `lq` 98–100 % in `logs\handoff-wifi.txt` |
| 2026-11-20 12:48–14:48 | same | HO-WIFI-2 | **NOT HELD** | drove from the keyboard (PC active about 266 s); 12 `PC_STALE` latches, 56 status dropouts, 3 restarts in the log; restart loop afterwards. No row recorded as a pass |
| 2026-11-20 | Wi-Fi build, battery | power (reported readings) | **FAIL** | 5.4 V unloaded battery; converter 3.0 V in, 1.0 V out in one failing condition; 4.8 V both terminals on USB alone. Cause not identified |
| 2026-11-20 about 21:57 | `ELRS-Handoff-Bench` 1A400D3F, different battery | RadioMaster drive on battery | WORKING (your report; not the written MD tests) | battery and firmware changed together |
| | | PWR-1 | NOT RUN | |
| | | CC-2 | NOT RUN | |
| | | ARENA-1 | NOT RUN | |
| | | ARENA-2 | NOT RUN | |
| | | ARENA-3 | NOT RUN | |
| | | PROF-1 | NOT RUN | |
| | | TAG-S | NOT RUN | |
| | | TRK-2 | NOT RUN | needs TAG-0 for all nine markers |
| | | TRK-3 | NOT RUN | |
| | | PWR-2 | NOT RUN | |
| | | MOVE-C | NOT RUN | |
| | | MOVE-R | NOT RUN | |
| | | MOVE-F | BLOCKED | needs MOVE-C |
| | | HO-USB-2 | NOT RUN | |
| | | MD-1 | NOT RUN | |
| | | MD-2 | NOT RUN | |
| | | MD-3 | NOT RUN | |
| | | CAM-1 | NOT RUN | |
| | | CAL-1 | NOT RUN | |
| | | TRK-1 | NOT RUN | |
| | | MOVE-2 | NOT RUN | |
| | | REC-1 | NOT RUN | |
| | `ELRS-Handoff-Telemetry` | TEL-1 | BLOCKED | parked with route D; sketch never flashed |
| | | IMU-1 | BLOCKED | needs TEL-1 |
| | | IMU-2 | BLOCKED | needs IMU-1; mounting transform unknown |
| | | CAM-2 | NOT RUN | |
| | | CAM-3 | NOT RUN | tag values still placeholders (TAG-0) |
| | | CAM-4 | NOT RUN | |
| | | CAL-2 | NOT RUN | |
| | | SHD-2 | BLOCKED | no IMU telemetry from the baseline firmware; needs IMU-2, CAM-4, CAL-2, TRK-1 |
| | | SHD-3 | BLOCKED | needs SHD-2 |

### Mock, fixture, footage and simulator results

| Date | Build | Test ID | Evidence | Result | Notes |
| --- | --- | --- | --- | --- | --- |
| 2026-11-20 | working tree on `8b8b7f7` | SW-1 | MOCK | PASS | 133 passed, 95 s |
| 2026-11-20 | `ELRS-Handoff-WiFi` 656526CE | compile | COMPILE | PASS | 983,406 bytes, ESP32C3 Dev Module, CDC on boot |
| 2026-11-20 | `ELRS-Handoff-Telemetry` | compile | COMPILE | PASS | 1,010,526 bytes (77 %) |
| 2026-11-20 | `ELRS-Handoff-Telemetry` | SW-4 | MOCK | PASS | 130 passed, 20 s; pin trace equals the handoff-only build |
| 2026-11-20 | `recovery-challenger-v2-seed1` | SW-5 | MOCK + FIXTURE | PASS | 11 passed, 59 s |
| 2026-11-20 | working tree on `8b8b7f7` | SW-2 | MOCK + FIXTURE | PASS | 137 passed, 1 skipped, 168 s |
| 2026-11-20 | working tree on `8b8b7f7` | SW-3 | SIM | PASS | `check` |
| 2026-11-20 | `config\sim.yaml` | MOVE-1 | SIM | PASS | sim slower than the stored physical figures |
| 2026-11-20 | `recovery-challenger-v2-seed1` | SHD-1 | FIXTURE | PASS | 520 proposals, fixture only |
| 2026-11-20 (night) | `ELRS-Handoff-Telemetry` regenerated at 23:57 from the Wi-Fi sketch as you last built it (7F60C809; before that it was stale against your 15:07 edit and one SW-4 test failed) | SW-4 | MOCK | PASS | 130 passed, 22 s. Still never flashed |
| 2026-11-20 (night) | working tree | SW-2 + SW-5 + SW-7 + overlay footage file | MOCK + FIXTURE | PASS | part of one run of 13 test files: 376 passed, 2 skipped (both need the Bristol recording) |
| 2026-11-21 | arena set-up, workflow, GUIDED SETUP, profiles | SW-8 | FIXTURE + MOCK | PASS | 119 passed (66 + 25 + 28) |
| 2026-11-21 | ELRS & CONTROL page | SW-9 | MOCK | PASS | 58 passed |
| 2026-11-21 | TRACKING page | SW-10 | FIXTURE + MOCK | PASS | 42 passed |
| 2026-11-20 (night) | multi-marker tracker, clicked opponent, survey | SW-7 | FIXTURE | PASS | 27 passed; chassis error ≤ 2.5 mm measured, ≤ 3.8 mm predicted on rendered frames |
| 2026-11-20 (night) | `scripts\motion_check.py` | SW-6 | MOCK + SIM | PASS | 53 passed, 34 s |
| 2026-11-20 (night) | `config\sim.yaml`, scale 0.40 | MOVE-1 | SIM | PASS | advance +0.127 m, turn_left +38.5°, turn_right −37.0°, reverse −0.076 m |
| 2026-11-20 (night) | bench sketch 1A400D3F, Wi-Fi sketch FECDB0F7 | SW-1 | MOCK | PASS | 133 passed, 112 s |

---

<a id="appendix-c"></a>
## Appendix C · Missing runners and unverified items

**Missing (not implemented; do not expect a command):**

- Any program that sends policy output to the robot. There is none, on purpose ([prerequisites](#aut)).
- A relay through the transmitter: no Lua script, no PPM bridge and no sender exist ([route B](#rly)).
- IMU telemetry on the baseline firmware. Only the Wi-Fi-based telemetry build sends it.
- A camera-measured comparison in `motion_check.py`: distances and rotations are entered by hand.
- A live opponent tracker in the Command Centre (the blob tracker and the new click-selected tracker run in
  `live_shadow`, `live_track` and the demo only).
- An opponent position during contact: shadow has no proposal while the objects touch.
- Clock synchronisation between the controller and the PC: none; freshness uses PC receipt time.
- Your measured tag dictionary, size, height and offsets, and the IMU mounting transform: not in the repo yet
  (placeholders in `config\home_test_overlay.yaml`, `config\arena_overlay.yaml`, `config\imu_mounting.yaml`).
- A live (camera) mode for `arena_overlay --track`; it is video-only. Live blobs exist only in the shadow
  demo's camera mode.
- A real-camera run of anything in the Command Centre: the ARENA tab, GUIDED SETUP's camera panel and the
  TRACKING tab have only been exercised on synthetic frames and a mock hub.
- A physical check that the arena you calibrate matches the drawing: the LIVE default size (1.150 m, pit radius
  250 mm) is the drawing, offered for confirmation, not a measurement.
- Any inversion handling in firmware.

**Commands that opened nothing real when checked (parsed with `--help` or read in code, never run on hardware):**

- Every `--camera 1` command, `scriptsruco_camera_check.py`, and `main_control_center.py` against the real camera. That
  includes `marker_survey`, `live_track` and the `--multi-marker` / `--click-opponent` options: synthetic
  frames only.
- Every `--port COM5` command against the real robot, including all of `motion_check.py checks` and `run`.
  (The `--wifi` client **was** run on the robot on 20 November.)
- `scripts\telemetry_link.py` and `control_center.tools.live_shadow`: run only against a fake link, loopback and
  synthetic frames.

**Expectations taken from the READMEs and mocked tests, not from the robot:** every pass condition in
MOVE-C, MOVE-R and MOVE-F; every pass condition in
HO-USB-2 and HO-WIFI-1/2; the quarter-second stop times; the inverted-drive prediction in MD-3; all pass
conditions in groups 3 and 4; every pass condition in TEL-1, IMU-1, IMU-2, SHD-2 and SHD-3.
