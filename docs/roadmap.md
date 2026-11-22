# Capri roadmap

Written 22 November 2026, after the first competition with Capri V1.

**Everything in this document is an idea or an open question. None of it is implemented, designed in detail or
committed to.** Completed work is described in the [README](../README.md) and the [case study](case-study/README.md);
what is known to be wrong with V1 is in [known issues](known-issues.md).

The next university antweight competition is expected in March 2027. That date is an expectation, not a confirmed
schedule.

## 1. What V1 established

| Finding from V1 | Basis | What it suggests |
|---|---|---|
| A very low passive wedge gets underneath many opponents and can lift or displace them by driving forward | Reported at the competition | Keep the low-profile wedge |
| The wedge took spinner damage and the electronics stayed protected | Reported at the competition | Treat the wedge as a sacrificial, replaceable part |
| The same low clearance sometimes left the robot stuck on something trapped beneath it | Reported; cause undetermined | Investigate the underside before changing it |
| Manual ELRS control was dependable enough to compete | Reported | Keep manual control as the path everything else must not degrade |
| Arena mapping from an angled camera worked at the event | Logged and reported | Keep the calibration workflow |
| Marker detection that worked at close range failed at arena distance | Reported; partly logged | Test perception at the real working distance, early |
| A wireless PC link added on the last day coincided with a supply collapse of unknown cause | Reported and logged | Characterise the power path before adding load |
| The policy only ever ran in the simulator | Repository state | Close the observation and command path step by step before any autonomy claim |

The common thread is the gap between software and synthetic testing and the conditions met in a real arena.

## 2. Close out the V1 unknowns first

These are investigations on the existing robot, not new design. Each answers a question that V2 decisions depend on.

| Question | A way to answer it | Related issue |
|---|---|---|
| How many pixels does a marker need in this camera, and at what distance does detection fail? | Place markers of several printed sizes at measured distances; record pixel size, exposure, focus and detection rate per frame | [P1](known-issues.md#p1-aruco-markers-were-not-reliably-detected-at-arena-distance) |
| Does a measured marker configuration give a stable pose? | Measure printed size, height and offsets of all nine markers; set `bot.measured`; rerun the tracker | [P2](known-issues.md#p2-decoded-markers-could-not-be-turned-into-a-pose-the-marker-configuration-was-never-measured) |
| How accurate is the arena mapping? | Click a fifth check point; tape-check the arena against the drawing | [P3](known-issues.md#p3-the-arena-calibration-was-completed-but-its-accuracy-was-not-independently-checked) |
| Why did the supply collapse? | Log the rail under load on battery alone, with the battery, converter and load written down | [E1](known-issues.md#e1-supply-collapse-and-restarts-on-battery-root-cause-not-identified) |
| Why does the left wheel move at reset? | Multimeter on GPIO 2 during reset, motors disconnected | [E2](known-issues.md#e2-left-wheel-moves-at-reset-and-upload) |
| What traps the robot? | Inspect the underside and the damaged wedge; reproduce on the bench with objects of known height | [M1](known-issues.md#m1-occasional-immobilisation-when-something-was-trapped-underneath) |
| What does the robot weigh, and what do the motors draw? | Scale; stall-current measurement | [E5](known-issues.md#e5-unrecorded-basics) |

Photographs of the robot as it finished the competition, including the wedge damage, would be worth taking before
anything is repaired or dismantled.

## 3. Capri V2: candidate directions

Ideas under consideration. No decision has been made on any of them, and some conflict with each other on mass.

**Mechanical**

- Retain the low-profile wedge that worked.
- Make the wedge a modular, sacrificial assembly that can be replaced between matches.
- Revisit the underside: clearance behind the wedge, and what happens when the robot rides up on debris.
- Improve the drivetrain and traction: wheel placement, tyre material, weight over the wheels.
- Consider an active wedge or lifter mechanism. This would be the first actuator beyond the drive motors and would
  need its own mass, power and control budget.

**Electronics**

- Encoder-equipped N20 motors, giving wheel odometry that neither the camera nor the IMU provides.
- Better power distribution: a characterised regulator with margin, a deliberate USB and battery arrangement, and a
  recorded wiring diagram.
- Move the left motor input off the boot strapping pin, if the reset behaviour is confirmed to come from it.
- A more integrated board in place of hand-wired modules, once the module version is stable.

**Perception and control**

- A camera and fiducial combination chosen from measurement: larger or fewer markers, a different marker family,
  higher resolution or a longer lens, controlled exposure.
- A single, verified way of combining manual and autonomous control, with manual takeover tested on hardware.
- An on-robot source for the inertial part of the observation, in the firmware that actually competes.

## 4. More structured testing

V1's testing was thorough in software and thin on hardware. Directions for the next iteration:

- A hardware results log filled in at the time of each test, not from memory afterwards.
- Photographs and short videos as routine evidence.
- Perception tested at the real camera distance, on the real floor surface, before any dependent work.
- Battery-only acceptance tests before a build counts as working.
- One change at a time when recovering from a fault.
- Match records at events: opponent, result, how it ended, what broke.

The existing [test playbook](autobots-test-playbook.md) already contains most of the individual procedures.

## 5. Other robot concepts

Separate concepts are being explored alongside Capri, including a shuffler (walking drive) and a horizontal spinner.
These are exploratory and are not Capri V2 design decisions. This repository stays focused on Capri and its history;
if another robot is built, whether it belongs here or in its own repository is an open question.

## 6. Naming future versions

| Name | Meaning |
|---|---|
| **Capri** | The robot and the project |
| **Capri V1** | The first prototype, as built for and run at the 21 November 2026 competition |
| **Capri V2**, **V3**, ... | A later iteration with a substantially changed mechanical or electronic design |

Suggested conventions, to adopt when V2 work starts:

- Refer to a specific robot as "Capri V*n*" in documentation; use "Capri" alone for the project.
- Keep each version's as-built hardware reference and known issues in files that name the version.
- Mark the repository state for each competition with a git tag such as `capri-v1-2026-11-21`.
- Keep version-independent tooling (simulator, Command Centre) free of version names.
