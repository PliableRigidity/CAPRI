# Capri development changelog

A dated summary of the project, newest first. Each entry is short on purpose: the full account, with measurements
and evidence, is the [case study](docs/case-study/README.md) and its [timeline](docs/case-study/README.md#appendix-a-dated-timeline).
The case study has its own [edition changelog](docs/case-study/CHANGELOG.md).

The project was called AutoBots until 22 November 2026. The robot is named **Capri**, and the prototype described
here is **Capri V1**.

Labels: **Hardware** = seen on the real robot or at the event. **Software** = simulator, mocked board or rendered
frames only. **Reported** = the author's observation, with no saved log, photo or video.

## 2026-11-22: repository cleanup and Capri V1 documentation

Documentation and housekeeping only. No firmware, control, protocol or pin change. Not hardware-tested, because
nothing that runs on the robot changed.

- Project renamed to Capri in documentation and window titles. Code identifiers, firmware identifiers and the
  repository name are unchanged ([naming](README.md#naming)).
- New README; new [hardware reference](hardware/README.md), [known issues](docs/known-issues.md),
  [roadmap](docs/roadmap.md) and [simulator guide](docs/simulator.md).
- Competition results added to the case study (third edition), with the laptop's record of the day preserved in
  [competition-2026-11-21-evidence.md](docs/competition-2026-11-21-evidence.md).
- Code changes are limited to window titles, docstrings, the relocated camera check, and one colour table whose
  repeated keys were removed (checked to give the identical dictionary). The four firmware sketches are byte-identical.
- Root tidied: planning handbook moved to `docs/design-notes-2026-09.md`, October architecture diagrams to
  `docs/images/`, the webcam marker check to `scripts/aruco_camera_check.py`, the EdgeTX test script to
  `scripts/edgetx/`.

## 2026-11-21: BEEES antweight competition (Capri V1)

- **Competed under manual control.** Capri V1 was driven with the RadioMaster Pocket over ExpressLRS. *Reported.*
  No match results, rankings or placements are recorded.
- **Autonomous operation was not attempted in a match.** The decision was made at the event after own-robot tracking
  proved unreliable. The learned policy was not deployed, so it was not evaluated on hardware.
- **Arena mapping worked.** The Command Centre calibrated the arena from the overhead camera at 10:35. *Logged and
  reported.* The fit had no independent check point.
- **ArUco tracking failed at arena distance.** Markers that were detected close to the camera were not reliably
  detected with the robot on the arena floor. The log also shows decoded markers rejected because the marker
  configuration was still placeholder values. Pose estimation and continuous localisation were not established.
- **The passive wedge worked.** Its low leading edge got under many opponents and lifted or displaced them. *Reported.*
- **The wedge absorbed spinner damage** at its front-left; the electronics were protected. *Reported.*
- **The robot occasionally became stuck** with something trapped underneath. Cause not determined. *Reported.*

## 2026-11-20 to 21 (overnight): one Command Centre

*Software.* The Command Centre became the single application for the competition-day workflow: twelve tabs, LIVE and
HOME TEST profiles, click-only arena calibration from the design drawing, calibration remembered between runs, policy
audit, a single tactic-to-command mapping, MANUAL / AI SHADOW / AUTONOMOUS modes with the autonomous gate closed, and
per-run session logs. The audit found that the candidate policy is a behaviour clone, not a PPO-trained policy.

## 2026-11-20: first hardware integration day

- *Hardware.* Laptop keyboard drove the robot over USB with RadioMaster arming and handoff (`ELRS-Handoff-Bench`).
- *Hardware.* Wi-Fi handoff build drove untethered, then lost the link repeatedly and restarted on battery.
- *Reported.* Supply collapse measured in one failing condition; left wheel moved at reset; root cause not identified.
  Wi-Fi has not been shown to be the cause.
- *Hardware.* Battery operation restored after a battery change and a return to the bench sketch, made together.
- Plan changed to relaying PC commands through the transmitter (desk investigation only). Multi-marker tracker,
  click-selected opponent tracker, movement check and log summary written. *Software.*

## 2026-11-19: handoff firmware

*Software.* Standalone handoff sketch with ARM and RADIO/PC switches, a PC keyboard client, and mocked-board tests.

## 2026-11-17 to 18: IMU bench test, firmware review, vision overlay, policy fix

- *Hardware.* LSM6DS3 found on I2C; three flip transitions read. An IMU-only sketch unexpectedly drove the left wheel.
- *Software.* Arena overlay tool; calibration and tracking on a historical recording of a March 2026 event.
- *Software.* A live simulator round exposed a policy that spun on the spot; diagnosed, turned into a regression case,
  and fixed by cloning a simpler teacher (the "recovery challenger").

## 2026-11-14 to 17: simulator and learning experiments

*Software.* MuJoCo arena and robot models. PPO did not learn to push an opponent into a pit across reward, curriculum,
action-space, observation and discount experiments. A turning drill identified the missing skill. Behaviour cloning of
a scripted controller, then balanced cloning with DAgger, produced the first working simulated policy.

## 2026-11-10: manual drive

*Hardware.* RadioMaster-only drive sketch (`ELRS-Control`): the robot drives under manual control.

## 2026-10-11 to 2026-11-03: prediction prototype and architecture

*Software.* Opponent trajectory predictor on synthetic data (smoke runs only). Architecture diagrams
([docs/images](docs/images/)).

## 2026-09-19 to 23: project start

Repository created. Desktop control centre showing the Logitech C525 feed with marker detection. Architecture fixed as
vision-first; ultrasonic positioning dropped from the required build. Project handbook and procurement list written.
