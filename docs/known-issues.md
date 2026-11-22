# Capri V1: known issues and limitations

Last updated 22 November 2026, the day after the BEEES antweight competition.

This is the list of what did not work, what was never shown to work, and what is still unexplained on Capri V1.
It is a summary: the detailed chronology, measurements and evidence are in the [case study](case-study/README.md).
Ideas for addressing these are in the [roadmap](roadmap.md); nothing in the roadmap is implemented.

**Evidence words used here**

| Word | Meaning |
|---|---|
| **Logged** | Recorded in a file. Where that file is git-ignored (most of `logs/`, `artifacts/`, `runs/`), the relevant lines are quoted in a tracked document and that document is linked. |
| **Reported** | Observed by the author on the real robot or at the event. No log, photo or video is in the repository. |
| **Software only** | Seen in the simulator, on a mocked board or on rendered frames. Says nothing about hardware. |
| **Hypothesis** | A plausible explanation that has not been measured. |

## Summary

| Area | Demonstrated | Not achieved or not demonstrated | Unresolved |
|---|---|---|---|
| Competition, 21 Nov 2026 | Arena mapping; manual ELRS operation; passive wedge effectiveness; the wedge taking spinner damage in place of the electronics | Reliable full-arena ArUco localisation; any physical autonomous operation; reliable Wi-Fi motor control with the policy in the loop | Exact reason for poor marker detection at arena distance; cause of occasional immobilisation |
| Before the competition | RadioMaster driving on battery; laptop keyboard driving over USB with RadioMaster arming and handoff | A stable untethered PC link; a live 19-value observation; the movement check on the robot | Cause of the supply collapse and restarts; cause of the left wheel moving at reset; what fixed battery operation on 20 Nov |

No match results, rankings or placements are recorded anywhere in this repository.

## 1. Perception

### P1. ArUco markers were not reliably detected at arena distance

- **Status:** not achieved. This is the dependency that stopped autonomous operation at the competition.
- **What happened (reported):** during development the camera detected the markers when the robot was held close to it.
  At the event the camera had to sit further back to see the whole arena. With Capri on the arena floor the software
  could not reliably identify its markers. Detection came back when the robot was held about 20 to 30 cm from the camera.
- **What the log adds (logged):** the Command Centre session of 21 November, 10:31 to 10:46, LIVE profile
  ([excerpt](competition-2026-11-21-evidence.md)). Markers were decoded intermittently. Own-robot tracking was reported
  lost for stretches of 15 s, 118 s and 253 s. The log does not record where the robot was at any moment.
- **Most likely explanation (hypothesis):** the printed markers covered too few pixels in the image, together with the
  limits of the Logitech C525 and its optics. An estimate from the saved calibration supports this without proving it:
  the arena floor spans about 0.50 to 0.66 px per millimetre in that view, so a marker a couple of centimetres across
  is only 10 to 20 px wide ([working](competition-2026-11-21-evidence.md#3-how-large-a-marker-was-in-the-picture)).
- **Not recorded:** marker pixel dimensions, printed marker size, exposure, focus, raw detector output per frame.

### P2. Decoded markers could not be turned into a pose: the marker configuration was never measured

- **Status:** not achieved. Independent of P1, and it would have blocked localisation even with perfect detection.
- **Logged:** every marker value in `config/arena_overlay.yaml` (`bot.*`) is a placeholder with `measured: false`:
  40 mm marker size, 34 mm height, and the offsets of markers 0 to 7 from the chassis centre. On 21 November the
  tracker rejected decoded markers because they "measure" 17 to 28 mm against the configured 40 mm, and when several
  markers were visible they disagreed on the chassis centre by about 155 mm against a 20 mm limit.
- **Also:** nine markers are fitted (reported: four on top, four underneath, one on the rear believed to be ID 8).
  ID 8 is not in the configuration and is ignored.
- **Consequence:** camera calibration succeeded, raw detection was intermittent, and pose estimation and continuous
  localisation were never established. These are three separate results.

### P3. The arena calibration was completed but its accuracy was not independently checked

- **Status:** demonstrated with a caveat.
- **Logged:** four arena corners were clicked on a 1280 x 720 picture, pits and flipper placed, and the alignment
  confirmed by the operator at 10:35 on 21 November. The tool graded the fit **WEAK** because no check point was
  clicked: four fit points always fit exactly, so nothing independent confirms the mapping. The saved record also
  shows the flipper click 158 mm from its design position; what that implies for accuracy has not been assessed.
- **Also:** the live profile has no lens-distortion model, and the arena has never been tape-checked against the
  design drawing (`config/arena_geometry.yaml`, 1150 mm square).

### P4. Opponent tracking has only seen rendered frames and one historical recording

- **Status:** not demonstrated on the live camera. The drag-box and click-selected trackers report *lost* or *ambiguous*
  during contact and need a new selection afterwards. See [command-centre.md](command-centre.md).

## 2. Control link and autonomy

### C1. The learned policy has never controlled the physical robot

- **Status:** not achieved. The competition was driven manually; this was a deliberate fallback.
- **Important distinction:** the policy was never deployed in a real match, so nothing here is evidence that the
  policy itself performs badly on hardware. It was not evaluated physically at all.
- **Also:** the checkpoint proposed for the robot (`runs/recovery-challenger-v2-seed1`) is a behaviour-cloned copy of a
  scripted pursuit controller, stored in a Stable-Baselines3 PPO container. PPO training itself did not learn the task
  in the simulator ([case study, section 5](case-study/README.md#5-teaching-blue-to-push-what-failed-and-what-worked);
  [schema](policy-observation-schema.md)). It reads simulator-truth positions. The checkpoint zip predates embedded
  observation metadata and depends on the adjacent `effective_config.json` for its `masked_zero_v1` previous-action
  encoding. Moving the zip without that sidecar can silently select the legacy encoding because the vector shape is
  unchanged; keep the files together and load through the repository compatibility path.

### C2. A complete real-world observation was never assembled

- **Status:** blocked. The policy needs 19 values. Six are IMU readings, and the firmware on the robot sends none.
  Own pose and opponent position depend on P1, P2 and P4.
- **Logged:** the one round started in the Command Centre on 21 November ran in MANUAL for 1.4 s and recorded
  `NO OBSERVATION` six times, with IMU, own pose and opponent all `MISSING`.

### C3. No verified untethered path from the laptop to the motors

- **Demonstrated (reported):** laptop keyboard driving over a USB cable, with the RadioMaster arming and handing over.
- **Not demonstrated:** the same over battery power without a cable in a stable state; any command produced by the
  policy reaching the robot; takeover by the RadioMaster with the policy in the loop.
- **In software only:** the Command Centre's AUTONOMOUS mode sits behind a gate that is closed in this repository
  state. `scripts/motion_check.py` (same commands in simulator and on the robot) has never been run on the robot.
- **Proposed only:** relaying PC commands through the RadioMaster Pocket ([feasibility note](elrs-relay-feasibility.md)).

### C4. Wi-Fi control was intermittent

- **Logged (20 November):** the Wi-Fi handoff build drove the robot untethered, then lost the PC link repeatedly:
  56 status dropouts (longest 4.7 s), 12 latched stale-command faults and 3 firmware restarts in two hours.
- **Status:** kept as an experiment (`ELRS-Handoff-WiFi/`), not a default. See E1 before drawing conclusions about cause.

### C5. Wi-Fi access-point password is a placeholder in source

- `WIFI_AP_PASSWORD` in `ELRS-Handoff-WiFi.ino` and `ELRS-Handoff-Telemetry.ino` is a placeholder committed to the
  repository. Change it before any further use of those builds.

## 3. Electrical

### E1. Supply collapse and restarts on battery: root cause not identified

- **Reported (20 November):** the ESP32-C3 and the ELRS receiver restarted repeatedly, ending in a restart loop.
  Multimeter readings in one failing condition: about 5.4 V at the unloaded battery, 3.0 V at the buck converter input,
  1.0 V at its output. Conditions were not recorded.
- **Not established:** the root cause. A weak battery, a high-resistance connection, a converter outside its range,
  excess load, or a combination all fit. **Wi-Fi has not been shown to be the cause**: it was running at the time, and
  current was never measured.
- **Unresolved:** battery operation came back after a battery change and a firmware change made together, so the fix
  cannot be credited to either. The margin of the working configuration is unmeasured.

### E2. Left wheel moves at reset and upload

- **Reported:** 17 November (under an IMU-only test sketch) and again on 20 November.
- **Hypothesis:** the left motor's reverse input is on GPIO 2, a boot strapping pin. Not measured.

### E3. USB power reaches the battery side of the converter

- **Reported:** with USB connected and the battery disconnected, about 4.8 V appears on both converter terminals.
  Whether USB and battery together is safe is unknown; the current through that path was not measured.

### E4. Stop behaviour of the drive sketch is an open question

- A reading of the ESP32 Arduino core source predicts that `digitalWrite` is ignored on a pin `analogWrite` has claimed,
  which would mean the drive sketch may not brake after driving. That contradicts the author's driving experience.
  A bench step was written for it and has not been run.

### E5. Unrecorded basics

- Robot mass, battery capacity, motor stall current and the as-built wiring are not recorded. The IMU pins used by the
  telemetry build (SDA 6, SCL 7) come from a cached test build and have not been confirmed by eye.
  See the [hardware reference](../hardware/README.md).

## 4. Mechanical

### M1. Occasional immobilisation when something was trapped underneath

- **Reported (competition):** the very low ground clearance that made the wedge effective also let Capri become stuck
  when something was wedged beneath it. How often, and against what, was not recorded.
- **Cause:** not determined. Candidates to investigate: underside geometry, clearance behind the wedge, wheel placement
  and traction, recovery behaviour.

### M2. The wedge is a wear part

- **Reported (competition):** in at least one engagement with a spinner the front-left of the wedge took most of the
  impact damage, and the electronics stayed protected. The wedge is not indestructible and is not currently designed
  to be swapped quickly.

### M3. No active weapon and no self-righting mechanism

- Capri V1 is a passive wedge pusher. Markers are fitted to both faces so that inverted running could be tracked;
  inverted driving itself is not documented as tested.

## 5. Simulator and policy

- **Speed gap to the real robot:** on the calibration command the simulated robot covers 1.64 m where the real one
  covered 2.03 m, and makes 3.43 turns where the real one made 8.25. Not tuned. How the physical figures were measured
  is not recorded.
- **The policy is a chaser, not a strategist:** it reaches the opponent reliably and wins only when its push line
  happens to end in a pit. Most simulated rounds are timeouts.
- **Physics limits:** a robot pinned against a wall is occasionally pushed through it (counted separately, never as a
  win). MuJoCo instability warnings accumulate in `MUJOCO_LOG.TXT`, which is git-ignored and grows without limit.

## 6. Repository and documentation

- **Evidence that is not on GitHub.** `runs/`, `artifacts/` and most of `logs/` are git-ignored. The case study links
  into them, so many of its evidence links work only on the development machine.
- **No photographs or video.** There is no photo of the built robot, the camera set-up, the damaged wedge, or any match.
- **Legacy names in code.** The Python package is `autobot`, firmware identifies itself as `autobot-handoff-*`, and
  the chassis CAD file is `assets/BattleBot v3.2.stl`. These are kept deliberately; see the naming table in the
  [README](../README.md#naming).
- **Firmware sketches are deliberately untouched.** Their contents are pinned by hash in tests or kept byte-identical
  to the sketch that drove the robot, so their comments still carry the old project name.
- **September planning documents are out of date.** [`design-notes-2026-09.md`](design-notes-2026-09.md) and
  [`hardware/list.md`](../hardware/list.md) describe the plan as it stood in September 2026 (ESP-NOW base station,
  AprilTags, IMU on SPI), not the robot as built.
