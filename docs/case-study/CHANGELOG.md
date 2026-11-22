# Case study changelog

One entry per edition. The case study itself lives in [README.md](README.md) and is edited in place; this file records what changed and when.

## 2026-11-18: first edition

State of the project three days before competition day.

- Wrote the full case study: the three parts, the learning story (PPO failures, turning drill, behaviour cloning, DAgger), the live-spin debugging story, verification status, authorship, and appendices (timeline, experiment record, evidence index, demo commands, missing assets).
- Added 17 figures in `figures/`. Six are existing evidence images copied or cropped; the rest are drawn from repository data by the scripts in `tools/`.
- New analysis done for this edition, not covered by any earlier handoff: the 18 November live sessions against the recovery challenger were replayed from their logs (section 6.7). One round of the 20:26 session did not replay faithfully and is excluded.
- Left out for lack of evidence in the repository: a photo of the built robot, the IMU flip plot (the serial log was not saved), any record of the robot driving, and the chassis design history. See Appendix E.
- Section 9 (competition day) is a checklist only. Nothing in it has happened.

## 2026-11-21 (early): Command Centre as one application

Not a new edition of the case study; a record of what changed in the software that day.

- `python main_control_center.py` is now the only command needed: GUIDED SETUP, ARENA, TRACKING and ELRS & CONTROL tabs were added to the Command Centre, with HOME TEST and LIVE profiles that keep separate files, remembered progress and statuses (Not configured / Ready / Warning / Failed).
- The arena and hazards are calibrated by clicking on the camera picture and stored in metres per profile (`arena.json`, camera-independent) apart from the camera's corner clicks (`clicks.json`), so a moved camera is recalibrated without touching the hazards.
- The TRACKING tab selects the opponent by dragging a box around any object (no size limit); the tracker reports TRACKED / UNCERTAIN / LOST with a confidence, never an old position as current, and its position is labelled image-based. Synthetic tests only.
- Operating modes were fixed to MANUAL (RadioMaster) and AUTONOMOUS (disabled), plus AI SHADOW (display only). Nothing here has been run on the real camera or robot. See [docs/command-centre.md](../command-centre.md).

## 2026-11-21 (overnight): one Command Centre from calibration to Wi-Fi gate

Software record only; nothing below has been run on the real camera, arena or robot.

- **Arena.** LIVE calibration is clicks only (four corners, both pit centres, one flipper click); the sizes come from the design
  drawing via `config/arena_geometry.yaml` ([docs/arena-geometry.md](../arena-geometry.md)). The 1130 vs 1150 mm discrepancy was traced
  to the operator's own typed draft (the drawing, simulator and config all say 1150); a measured difference is entered once as a
  labelled override. Rotation, start corner, mirrored order and oblique views are tested.
- **Between rounds.** The calibration, camera mode, marker settings, Wi-Fi endpoint and model choice are remembered per profile
  (atomic writes, `.bak`, no opponent / authority / link status ever stored). A reference picture detects a moved camera; the
  operator confirms alignment each run. Tested through a full restart and a simulated camera disconnect.
- **Policy audit.** `final_model.zip` is behaviour cloning plus DAgger of a pursuit controller saved as an SB3 PPO object, not a
  PPO-trained policy. The verified 19-value schema is in [docs/policy-observation-schema.md](../policy-observation-schema.md); the
  baseline firmware sends no IMU, so live inference is blocked and nothing is invented. This older checkpoint does not
  embed its observation-encoding metadata: it must remain beside `effective_config.json`, or the loader can silently
  fall back to the legacy previous-action encoding while retaining the same vector shape.
- **Command mapping.** One runtime tactic-to-throttle/steer mapping with a conservative default cap (60 of 255), slew limiting,
  neutral-first on reversals and stale-decision stops. The 40 % motion-check values are the supervised scale only.
- **Wi-Fi.** One link manager, three operator modes (MANUAL / AI SHADOW / AUTONOMOUS) with an autonomous gate that is closed in the
  current state (no hardware acceptance record, no IMU, no firmware test mode), a hardware-evidence acceptance workflow and a
  Match tab where a round starts only when the operator presses Start Round. No firmware was flashed or changed.
- **ArUco.** The live path is `DICT_4X4_50` end to end; real-camera traces showed every marker rejected for size because the
  configured 40 mm is a placeholder ([docs/aruco-tracking-audit.md](../aruco-tracking-audit.md), measurement checklist at the end).
- **Interface.** Twelve tabs from SYSTEM STATUS (what is ready, the one next action) to MATCH, a seven-step guided sequence, and
  one log folder per run under `logs/sessions/`.

## 2026-11-20: second edition, the night before the competition

The first day of hardware integration went wrong, and this edition records it.

- **New section 7, "20 November 2026: Wi-Fi integration failures and the decision to reuse ELRS".** It sits in the main narrative after the debugging story. It gives the twelve-step chronology (RadioMaster drive, USB handoff demonstrated, Wi-Fi handoff demonstrated, instability, restarts, the left wheel moving at reset, the supply collapse, the swaps that settled nothing, the recovery, why its cause is unknown, the change of plan, and keeping the Wi-Fi work), the four voltage readings with what they do and do not establish, the trade-offs, both architectures drawn and labelled (attempted, and revised target), and a status table that separates *demonstrated on hardware*, *tested in software*, *proposed* and *blocked / unverified*.
- **The old sections 7 to 10 are now 8 to 11.** Links inside the document were updated. Anything outside it that says "section 9" for competition day now means section 10.
- **Dated updates, not rewrites, in the earlier sections:** the opening status and table, "what I can demonstrate", section 1 (the system figure is left as drawn on 18 November, with a note), section 2 (which firmware is on the robot), section 4 (the markers are now fitted but unmeasured).
- **Tables and appendices:** new rows in "what is verified" (section 8), "who did what" (section 9), the competition-day checklist (section 10), five new lessons (section 11), the timeline, the firmware checks, the evidence index, the demo commands and the missing-evidence list.
- **Corrected a link:** the drive sketch is `ELRS-Control/ELRS-Control.ino`.

New evidence this edition points to:

- Local `artifacts/integration-2026-11-20/evidence.md`: the reported observations and readings, the firmware baseline hash and build time, and what the Wi-Fi client log shows. Reported items are labelled as reported; no agent measured, flashed or drove anything.
- Local `artifacts/integration-2026-11-20/wifi-session-log-summary.txt`, made by `scripts/handoff_log_summary.py` from local `logs/handoff-wifi.txt`: 3 logged restarts, 56 status dropouts, 12 latched stale-command faults, link quality 98 to 100 % while linked.
- [`docs/elrs-relay-feasibility.md`](../elrs-relay-feasibility.md): what the RadioMaster Pocket and EdgeTX can and cannot accept from a PC. Desk investigation only.
- [`scripts/motion_check.py`](../../scripts/motion_check.py) and [`tests/test_motion_check.py`](../../tests/test_motion_check.py): the movement check. 53 tests pass on the simulator and a mocked board. Not run on the robot.
- [`docs/autobots-test-playbook.md`](../autobots-test-playbook.md): reordered after the session. Power checks first, the wired movement check next, the relay as proposed, Wi-Fi as experimental, closed-loop prerequisites listed and blocked.
- `control_center/tools/chassis_tracker.py`, `opponent_select.py`, `marker_survey.py`, `live_track.py` and their tests (`tests/test_chassis_tracker.py`, `tests/test_opponent_select.py`, 27 tests): multi-marker tracking with measured / predicted / stale / lost states, a click-selected opponent tracker, and a read-only survey of the markers actually on the robot. Rendered frames only; nothing has been run on the real camera.

Not in this edition, because it does not exist: any measurement of the supply rail under load with its conditions recorded; any log of the recovered baseline; a physical run of the movement check; anything on the transmitter relay beyond reading.

Pre-edit copies of the four documents changed tonight are in `artifacts/code-snapshots/pre-integration-docs-2026-11-20/`.

## 2026-11-22: third edition, the day after the competition

The competition happened on 21 November. This edition records it and renames the project.

- **The name.** The robot is Capri and this prototype is Capri V1. The title and opening now say so, with a note on
  where the old name (AutoBots) survives. Sections 1 to 9 were not reworded: "the robot" there means Capri V1.
- **Section 10, "Competition day (21 November 2026)", is now the account of the day**, replacing the capture checklist.
  It covers the outcome (manual control; autonomy not attempted), the camera calibration that worked, the marker
  tracking that did not, the decision to drive manually, the wedge, the mechanical limitations, and a table setting
  the planned captures against what is actually in the repository. Its opening table separates *demonstrated*,
  *not achieved or not demonstrated* and *unresolved*.
- **Two new evidence words**, *reported* and *logged*, because almost nothing from the event was recorded. No match
  results, rankings or placements appear, because none were written down.
- **What the laptop logged is preserved** in [`docs/competition-2026-11-21-evidence.md`](../competition-2026-11-21-evidence.md):
  the Command Centre's log for 10:31 to 10:46 quoted in full, the saved calibration, and an estimate of how many pixels
  a marker covered. The log files themselves are git-ignored.
- **A finding the log adds to the account:** besides poor detection at distance, the markers that were decoded were
  rejected because the marker size and positions in the configuration were still placeholders. Detection, pose
  estimation and continuous localisation are reported as three separate results.
- **A clarification about the policy** (10.2): the checkpoint that would have run is a behaviour clone stored in a PPO
  container, not a PPO-trained policy, as the 21 November audit found. It was never deployed, so the competition is
  not evidence about it.
- **Dated updates, not rewrites, elsewhere:** the opening status, "what I can demonstrate", section 4, a note under
  the table in 7.8, seven new rows in section 8, three in section 9, nine new lessons in section 11, the timeline,
  the evidence index and the missing-evidence list.
- **Links corrected** after the repository tidy: the October architecture diagram is now under `docs/images/`, and
  the simulator text that was in the repository README is now `docs/simulator.md`.

Not in this edition, because it does not exist: any photograph or video from the competition; match records; a
measurement of marker pixel size or detection rate against distance; the cause of the immobilisation; any further
work on the 20 November electrical faults. No figure was added or redrawn.

## Next edition

No date set. To add as they are measured, without rewriting sections 1 to 10:

- Answers to the open questions in section 10 (marker detection against distance, the cause of the immobilisation).
- Photographs of Capri V1 as it finished the competition, if any are taken before it is repaired or dismantled.
- Anything in Appendix E that becomes documented.

Capri V2 is not part of this case study. If it is built it gets its own document.
