# Capri V1 hardware reference

As-built configuration of Capri V1, as far as the repository can support it. Written 22 November 2026.

Every value below is taken from the firmware sketches, the configuration files or the case study, and its source is
given. Values that were never measured or recorded are marked **not recorded**; they are not estimated here.
The September 2026 procurement plan is kept separately in [list.md](list.md) and differs from the robot as built.

## 1. Overview

Capri V1 is a two-wheel differential-drive antweight (150 g class) with a passive wedge and no active weapon.

| Subsystem | Part | Source |
|---|---|---|
| Chassis | Low wedge with a honeycomb scoop at the nose, a wheel at each side. CAD: `assets/BattleBot v3.2.stl` | STL bounds 162.0 x 135.3 x 33.5 mm |
| Drive motors | 2 x N20 brushed gear motors | Case study section 2 |
| Wheels | 33.5 mm diameter, about 132 mm apart | `config/sim.yaml`, recorded there as physical measurements |
| Motor driver | DRV8833 dual H-bridge | Sketches |
| Microcontroller | ESP32-C3 (built as "ESP32C3 Dev Module", USB CDC on boot enabled) | Sketch headers |
| Radio receiver | ExpressLRS receiver, CRSF output over UART | Sketches |
| Transmitter | RadioMaster Pocket (EdgeTX) | Case study |
| IMU | LSM6DS3 on I2C, address 0x6B, WHO_AM_I 0x69 | 17 November bench test, case study section 2 |
| Logic supply | Buck converter from the battery | Case study section 7 |
| Battery | Pack reading about 5.4 V unloaded on 20 November. The September plan lists a 5-cell, 6 V NiMH pack | Reported reading; `list.md` |
| Fiducial markers | Nine printed ArUco 4x4 markers: four on top, four underneath, one on the rear face | Reported; not measured into the configuration |

**Not recorded:** robot mass, wedge material and print settings, leading-edge clearance, battery capacity, motor gear
ratio and stall current, the buck converter model, whether a physical disconnect link is fitted, and the as-built
wiring. There is no photograph or wiring diagram in the repository.

## 2. Pin assignments

Identical in all four sketches (`ELRS-Control`, `ELRS-Handoff-Bench`, `ELRS-Handoff-WiFi`, `ELRS-Handoff-Telemetry`).

| ESP32-C3 GPIO | Connected to | Direction | Notes |
|---|---|---|---|
| 3 | DRV8833 AIN1 | Output, PWM | Left motor, forward |
| 2 | DRV8833 AIN2 | Output, PWM | Left motor, reverse. GPIO 2 is a boot strapping pin; see [known issue E2](../docs/known-issues.md#e2-left-wheel-moves-at-reset-and-upload) |
| 4 | DRV8833 BIN1 | Output, PWM | Right motor, reverse |
| 5 | DRV8833 BIN2 | Output, PWM | Right motor, forward |
| 20 | ELRS receiver TX | Input, UART1 RX | CRSF, 420,000 baud, receive only |
| 6 | LSM6DS3 SDA | I2C | Telemetry sketch only. Taken from a cached test build; **not confirmed by eye** |
| 7 | LSM6DS3 SCL | I2C, 100 kHz | Telemetry sketch only. Same caveat |
| USB | Laptop | USB CDC serial | Status lines; PC drive commands in the handoff sketches |

The right motor's inputs are swapped relative to the left in software (`setRightMotor`), which sets wheel polarity.

## 3. Radio channels

| CRSF channel | Function | Sketches |
|---|---|---|
| 1 | Steering | All |
| 2 | Throttle | All |
| 5 | ARM switch, high = armed | Handoff sketches |
| 6 | MODE switch, high = PC, low = RADIO | Handoff sketches |

Channel calibration: minimum 174, centre 992, maximum 1811, dead zone ±45 around centre. Switches count as low at or
below 600 and high at or above 1400. Mixing is arcade style: left = throttle + steering, right = throttle − steering,
scaled together if either exceeds ±255.

## 4. Safety behaviour in firmware

Read from the sketches. Only the normal driving path has been seen on hardware; the failure cases have no recorded
hardware result.

| Behaviour | Value | Sketch |
|---|---|---|
| Stop when no valid radio frame arrives | 250 ms | All |
| Valid frames required before movement is allowed | 10 consecutive, with sticks centred | All |
| Arming | Physical ARM switch on the transmitter; refused if the switch is already on at boot or after a fault | Handoff sketches |
| Stop when no PC command arrives (PC mode) | 250 ms | Handoff sketches |
| PC command limit | 128 of 255 | Handoff sketches |
| Handoff between RADIO and PC | Requires a neutral command from the new source first | Handoff sketches |
| What the PC can do | Send drive commands only. It cannot arm, select PC mode or clear a fault | Handoff sketches |

Whether the motors actively brake when stopped is an open question
([known issue E4](../docs/known-issues.md#e4-stop-behaviour-of-the-drive-sketch-is-an-open-question)).

## 5. Firmware builds

| Sketch | Role | Hardware status |
|---|---|---|
| [`ELRS-Control`](../ELRS-Control/ELRS-Control.ino) | Original RadioMaster-only drive sketch (10 November). Kept byte-identical as the fallback | Drove the robot |
| [`ELRS-Handoff-Bench`](../ELRS-Handoff-Bench/README.md) | Drive sketch plus ARM and RADIO/PC switches and a USB line protocol. No IMU, no Wi-Fi | On the robot from the night of 20 November. Drove on battery from the RadioMaster and from a laptop keyboard over USB |
| [`ELRS-Handoff-WiFi`](../ELRS-Handoff-WiFi/README.md) | The bench sketch plus a soft access point and UDP PC link | Drove untethered on 20 November, then dropouts and restarts. Experimental |
| `ELRS-Handoff-Telemetry` | The Wi-Fi sketch plus read-only LSM6DS3 telemetry. Generated by `scripts/make_telemetry_sketch.py`; do not edit by hand | Never flashed |

Which sketch was on the robot during the competition was not recorded on the day. `ELRS-Handoff-Bench` was the build on
the robot the night before.

The handoff sketches identify themselves over USB as `autobot-handoff-bench-1`, `autobot-handoff-wifi-1` and
`autobot-handoff-telemetry-1`, and the Wi-Fi builds use the network name `AutoBot-Handoff`. These identifiers predate
the name Capri and are matched by the PC software and its tests, so they are unchanged.

## 6. Power

What is known is four multimeter readings from 20 November, reported from memory without their conditions:

| Where | Condition | Reading |
|---|---|---|
| Battery terminals | No load | About 5.4 V |
| Buck converter input | One failing load condition | About 3.0 V |
| Buck converter output | The same condition | About 1.0 V |
| Both converter terminals | USB powering the ESP32, battery disconnected | About 4.8 V |

The last row means USB power reaches the battery side of the converter. **Do not connect USB and the battery
together** until that path is understood. Details and what the readings do and do not establish are in
[case study section 7.3](../docs/case-study/README.md#73-the-measurements-and-what-they-do-and-do-not-show).

## 7. Off-robot equipment

| Item | Role | Notes |
|---|---|---|
| Logitech C525 webcam | Overhead arena camera | Used at 1280 x 720 over DirectShow. Marker detection at arena distance was unreliable ([P1](../docs/known-issues.md#p1-aruco-markers-were-not-reliably-detected-at-arena-distance)) |
| Laptop (Windows) | Command Centre, vision, simulator | |
| RadioMaster Pocket | Manual control, arming, mode selection | The only command path used at the competition |

## 8. Markers

| Property | Value | Status |
|---|---|---|
| Dictionary | ArUco `DICT_4X4_50` | Confirmed by two real-camera surveys ([audit](../docs/aruco-tracking-audit.md)) |
| IDs | 0 to 3 top, 4 to 7 bottom in the configuration; an ID 8 was decoded and is believed to be the rear marker | IDs as fitted not verified against the configuration |
| Printed size | Configured as 40 mm | **Placeholder, not measured** |
| Height above the floor | Configured as 34 mm | **Placeholder, not measured** |
| Position of each marker on the chassis | Configured ±35 mm, ±30 mm | **Placeholder, not measured** |

Configuration: `config/arena_overlay.yaml`, block `bot`, with `measured: false`.

## 9. Arena

From the design drawing, via `config/arena_geometry.yaml`: floor 1150 x 1150 mm, two R250 quarter-circle pits on
opposite corners, one flipper corner with 335 mm legs. Not tape-checked against the arena as built. See
[arena-geometry.md](../docs/arena-geometry.md).
