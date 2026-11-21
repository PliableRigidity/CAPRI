# Relaying PC commands through the RadioMaster Pocket: feasibility

Written 20 November 2026, the night before the competition. **Nothing in this note has been tried on the
transmitter or the robot.** It is a desk investigation of what the Pocket and EdgeTX can accept, so that
the first hardware attempt starts from facts and not from forum memory.

Status words used here: **source** = read in EdgeTX or ExpressLRS source or official documentation;
**reported** = stated by users or third-party projects, not checked; **unknown** = no primary source
found. EdgeTX references are to release 2.12.5. Links are at the end.

## Why this route

The robot already has a radio link that works on battery power: Pocket → ExpressLRS → receiver → ESP32-C3.
If the laptop's commands travel over that same link, the robot needs no second radio and no new
firmware: its existing RADIO mode reads steering on channel 1 and throttle on channel 2 whoever moved
them. Arming stays on a physical switch on the Pocket (channel 5), exactly as now.

```
TARGET (not built, not verified)

  Laptop: camera + policy -> throttle / steer
          |
          |  one of the inputs in section 2
          v
  RadioMaster Pocket: mixer chooses STICKS or PC with a physical switch; ARM switch stays physical
          |
      ExpressLRS (the existing, working link)
          v
  Receiver -> ESP32-C3 (ELRS-Handoff-Bench, RADIO mode) -> DRV8833 -> motors
```

## 1. What the Pocket has

| Port | Status | Notes |
| --- | --- | --- |
| USB-C data port | source | joystick, storage or serial (VCP) mode, chosen when plugged in |
| USB-C charge port | source | charging only |
| 3.5 mm headphone jack | source | audio |
| "DSC" trainer port | source | present on the hardware and in the Pocket firmware build (trainer in, trainer out, plug detect) |
| Nano external module bay | source | carries CPPM / SBUS trainer input only with the external module off |
| AUX serial pads | unknown | a UART exists in the firmware build; no document shows it reaching a connector or pad |

**Trainer jack pinout: unknown.** No RadioMaster or EdgeTX document gives tip / ring / sleeve for the
Pocket. Reports for a different RadioMaster radio contradict each other. **Input voltage tolerance:
unknown.** No primary source says whether the input takes 3.3 V only or tolerates 5 V.

The trainer port and the headphone jack are two different sockets. Read the labels on the radio itself
before measuring anything; a review places the trainer port on the top edge beside the data USB-C port and
the headphone jack on the bottom (reported, not checked against the hardware).

EdgeTX trainer modes compiled into the Pocket build (source), and what each would need:

| Trainer mode | Usable on a stock Pocket? |
| --- | --- |
| Master/Jack | Yes: PPM in on the DSC port. This is the PPM route in section 2 |
| Slave/Jack | Yes: PPM **out** on the DSC port. Useful only for measuring what the port does (section 4) |
| Master/SBUS Module, Master/CPPM Module | Only with the external module off; the signal goes in on module-bay pins |
| Master/Serial | Needs a serial port set to "SBUS Trainer". USB-VCP is excluded; no other port is known to be reachable |
| Master/Bluetooth, Slave/Bluetooth | No: needs a Bluetooth build and module |
| Master/Multi | No: needs a multiprotocol module |
| Master/CRSF | Needs ELRS 4 on the internal module and trainer data from the ELRS backpack |

## 2. Ways to get PC channels in

| Route | Supported? | What it needs | Biggest unknown |
| --- | --- | --- | --- |
| **USB joystick mode** | **No.** It is an output: the radio shows up as a game controller (source) | nothing: not usable | n/a |
| **USB serial (VCP) as SBUS trainer** | **No.** EdgeTX refuses that assignment: "Telemetry input & SBUS trainer on VCP is not yet supported" (source) | nothing: not usable | n/a |
| **USB serial (VCP) in LUA mode + a Lua mix script** | Possible on stock firmware (source for each part; no working example found) | USB-VCP set to LUA; a script in `/SCRIPTS/MIXES`; two mixer lines behind a switch | Never demonstrated by anyone I could find. The script runs about 20 times a second, not at mixer rate |
| **PPM into the DSC trainer port** | Supported in firmware (Master/Jack) (source) | A microcontroller making PPM, a 3.5 mm plug, a level clamp | Which contact is signal and which is ground; whether the input survives more than 3.3 V |
| **CPPM / SBUS into the module bay** | Supported in firmware with the external module off (source) | Access to bay pins | Bay pin mapping, levels, SBUS polarity |
| **Unreleased EdgeTX "CRSF trainer over USB-VCP"** | Open pull request, not merged; unofficial builds exist (reported) | Flashing non-release firmware on the transmitter | Unmerged code on the only transmitter, the night before |
| **ExpressLRS TX-backpack trainer input** | Feature exists for head trackers (source) | ELRS 4, backpack 1.5.2+, EdgeTX 2.11+, and a custom ESP-NOW sender | The sender is undocumented; three firmware versions to line up |
| **Second ELRS transmitter driven by the PC** | Not as a relay. One active transmitter per binding phrase (source); a bare ESP32-C3 has no ELRS radio chip (source) | An RF module | The Pocket would lose override and disarm authority |
| **ELRS Airport** | **No.** It replaces the control link with a data link (source) | nothing: not usable | n/a |

So two candidates keep stock firmware on the transmitter: **the Lua script over USB-C** and **PPM into
the trainer jack**. Neither is verified.

## 3. How the pilot keeps authority in either candidate

These are EdgeTX behaviours read from source, to be confirmed on the Pocket's own screen before any use:

- PC values arrive as ordinary mixer sources (a Lua script's outputs, or `TR1`..`TR16` for trainer input).
- A mixer line with that source, set to **Replace** and gated by a **physical switch**, sits under the
  stick line for the same channel. Switch off: sticks. Switch on: PC.
- **Channel 5 (arm) must stay on its physical switch with no PC line at all.** The PC must never be able
  to arm.
- Trainer inputs have their own calibration: the radio's Trainer page stores the incoming centre with
  "Cal" and scales all inputs with a multiplier, and each `TR` source is `(input − centre) × 2`. The
  alternative to hand-written mixer lines is the "Trainer" special function, which applies the Trainer
  page's per-stick mode (off, add, or replace, with a weight) while its switch is on. Either way, what
  reaches the robot has to be checked end to end: send a known value from the PC and read `steer=` and
  `thr=` in the robot's `CH` line (playbook, RLY-2). The robot's firmware ignores the first ±45 counts
  around centre, so small PC values will read as zero unless the mapping allows for it.
- If trainer input stops, EdgeTX disables lines that use it after **1 second** and the stick lines take
  over. One second is long for this robot: at even 0.2 m/s it travels 0.2 m. A Lua script has no built-in
  timeout at all and must zero its own outputs when frames stop.
- The robot's firmware cannot tell PC from pilot on this route. Its own protections that still apply:
  disarm, the 250 ms radio-loss stop, and the arm-off-then-on rule after a loss. Its PC-specific
  protections (token, neutral-first, stale-command latch) do **not** apply, because it never sees a "PC".

That last point is the real cost of this route: the handoff logic moves from firmware that was tested
against a mocked board into a transmitter mixer and a script that have not been tested at all.

## 4. If the trainer jack is tried: what must be known before any wire is connected

**Do not connect anything to the jack yet.** The order matters, and the first steps use only a multimeter.

1. **Pinout, transmitter off, nothing plugged in but a bare 3.5 mm plug or breakout.** Continuity from
   each contact to the transmitter's ground (the USB-C shell is ground). The contact that reads near zero
   ohms is ground. Record it.
2. **What the port puts out, transmitter on, trainer mode Slave/Jack, nothing connected to the PC.**
   Measure each non-ground contact to ground with a multimeter, then with a scope or logic analyser if one
   is available. In slave mode the port outputs the radio's own PPM, so this shows the idle level, the
   voltage swing and the polarity the radio itself uses. Record the voltage. This is the best available
   guide to what the input expects.
3. **Only if the output swing is at or below 3.3 V:** plan a 3.3 V source (an ESP32-C3 GPIO is 3.3 V) with
   a series resistor of 1 kΩ or more so a wrong contact cannot pass damaging current. If the swing is
   higher or cannot be measured, stop: the input level is unknown.
4. **What EdgeTX will accept as PPM** (source): time between successive edges of the same kind 800 to
   2200 µs per channel (1500 µs is centre); a gap between frames strictly longer than 4 ms and shorter
   than 19 ms; up to 16 channels. Pulse width itself is not checked. Send 8 channels in a 22.5 ms frame so
   the gap stays inside the window. Either polarity decodes if the timing is right.
5. **Bench order, robot powered off throughout:** PPM source running on its own and checked on a scope or
   logic analyser first; then connected to the Pocket with trainer mode Master/Jack; then the Pocket's
   channel monitor watched while the source sweeps one channel. Only when the monitor follows, and the
   switch and the 1 s loss behaviour are seen on the monitor, does the robot get powered: disarmed first,
   then armed with the wheels raised.

A spare ESP32-C3 can generate PPM. That does not make the route ready: steps 1 to 3 are measurements
nobody has made.

## 5. If the Lua route is tried: what must be true

- USB mode "Serial (VCP)" when the cable is plugged in, and **Hardware → Serial ports → USB-VCP = LUA**.
  On radios with an internal ELRS module this setting is normally CLI, which ELRS passthrough flashing
  uses; it has to be put back afterwards.
- **While USB-VCP is in CLI mode, bytes sent to the COM port go to the radio's command line.** A PC
  program must therefore send nothing until it has received a banner that only the Lua script prints, and
  must stop the moment the banner stops. No sender exists in this repository yet, deliberately: writing
  one before the receiving side is confirmed on the radio would be sending arbitrary bytes at a port.
- The script reads with `serialRead()`, holds its last valid command for a short time (about 150 ms), and
  outputs zero otherwise. It exposes two outputs (throttle, steer) as mixer sources.
- About 20 updates a second, with jitter. The simulator chooses a tactic every 20 ms, so the policy's
  decisions would be held for two or three steps each. The effect of that hold on the policy has not been
  measured.

## 6. Recommendation

1. **Tomorrow, drive by RadioMaster.** The relay is not a competition-day feature. Neither candidate has
   been powered once.
2. **First relay experiment, when there is time: the Lua route, transmitter only, robot off.** It risks no
   hardware and needs no wiring. Pass condition: the Pocket's channel monitor follows PC values only while
   the relay switch is on, returns to the sticks when it is off, and returns to neutral within a quarter of
   a second of the USB cable being pulled.
3. **The trainer jack second**, starting from the multimeter steps in section 4.
4. Do not flash unreleased EdgeTX or a different ELRS version on the only transmitter before an event.

## 7. Interfaces at a glance

| Interface | Verdict for PC → Pocket control |
| --- | --- |
| USB-C, joystick mode | Unsupported (output only) |
| USB-C, storage mode | Unsupported (the radio stops running) |
| USB-C, serial as SBUS trainer | Unsupported in released EdgeTX |
| USB-C, serial in LUA mode with a mix script | Possible, unverified, about 20 Hz |
| DSC trainer port, PPM | Supported by firmware; pinout and voltage unknown; needs a bridge |
| Module bay, CPPM / SBUS | Supported by firmware with the module off; pins unknown |
| ELRS backpack trainer input | Exists; needs version upgrades and an undocumented sender |
| Second ELRS transmitter | Not a relay; removes the Pocket's authority |
| ELRS Airport | Unsupported for this purpose |

## Sources

- RadioMaster Pocket quick start guide (port labels): mirror at xcopter.com/download/Pocket_User_Manual.pdf
- EdgeTX 2.12.5, `radio/src/targets/taranis/hal.h` (Pocket trainer pins), `radio/src/gui/gui_common.cpp`
  (`isSerialModeAvailable`, `isTrainerModeAvailable`), `radio/src/serial.cpp` (Lua receive buffer),
  `radio/src/lua/api_general.cpp` (`serialRead`, `serialWrite`, `setSerialBaudrate`),
  `radio/src/targets/common/arm/stm32/trainer_driver.cpp` (PPM decoding), `radio/src/trainer.cpp` and
  `radio/src/mixer.cpp` (1 s trainer validity, lines disabled on loss): github.com/EdgeTX/edgetx
- EdgeTX manual, trainer and special functions pages: manual.edgetx.org
- EdgeTX pull request 7630, "support CRSF Trainer over USB-VCP" (open): github.com/EdgeTX/edgetx/pull/7630
- ExpressLRS: trainer input (expresslrs.org/software/trainer-input), Airport
  (expresslrs.org/software/airport), team racing page for the one-transmitter rule
  (expresslrs.org/software/teamracing), receiver-as-transmitter pull request 2826
- Head-tracker project wiring note recommending a 3.3 V clamp on PPM (third party, reported):
  headtracker.gitbook.io
