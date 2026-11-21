# ELRS-Handoff-Bench

Bench-test sketch for the ESP32-C3 robot plus a Windows keyboard client. It tests who has control
(RadioMaster or PC), what happens at a handoff, and whether the motors stop on disarm and link loss.

> **Status, night of 20 November 2026.** This sketch is the firmware on the robot and the provisional
> competition baseline (SHA-256 `1a400d3f…5fbc`, last built by the Arduino IDE at 21:56:56). Ishaan has
> driven the robot with it from the laptop keyboard over USB, switched control in both directions, and,
> after a battery change and a fresh upload, driven it from the RadioMaster on battery alone. The failure
> cases in the bench sequence (steps A1 to A15, B3 to B6) have **no recorded hardware result**. The story
> of that day, including why the Wi-Fi build came off the robot, is section 7 of
> [the case study](../docs/case-study/README.md); the current test order is in
> [the playbook](../docs/autobots-test-playbook.md).
>
> **Power warning added 20 November:** with USB alone, about 4.8 V was measured on both terminals of the
> buck converter, so USB reaches the battery side. Do not run USB and the battery together until that
> path is understood (playbook, PWR-2). Section B below was written before that reading.

The text below is as written on 19 November, before anything was flashed. Everything marked "software" was
compiled or run on the PC against a model of the board. No IMU, vision or autonomous control.

| File | What it is |
| --- | --- |
| `ELRS-Handoff-Bench.ino` | the sketch (complete, standalone) |
| `../scripts/handoff_bench_client.py` | PC keyboard client and listen-only monitor |
| `../tests/test_handoff_bench.py` | all the software tests |
| `host_test/` | mocked board and test scenarios; the Arduino build ignores this folder |
| `../ELRS-Control/ELRS-Control.ino` | your working sketch, untouched (SHA-256 `6040bed3...a6dd`) |

## What is the same as ELRS-Control.ino

Motor pins (AIN1/AIN2 = GPIO3/GPIO2, BIN1/BIN2 = GPIO4/GPIO5), receiver on UART1 RX GPIO20 at 420000 baud,
CH1 steering and CH2 throttle, calibration 174/992/1811, deadzone 45, the mixer and its scaling, wheel
polarity, the 10-frame qualification and the 250 ms frame timeout. The test build fails if any of those
constants differs from the original, and a test compares the two sketches at 676 stick positions.

## What is different

- **ARM and MODE switches.** CH5 arms, CH6 selects RADIO or PC, HIGH = armed and HIGH = PC. These are
  defaults, not confirmed: see "Finding your switches". Four `#define`s at the top of the sketch change them.
- **Neutral means zero.** Control is granted only when both normalised sticks are exactly 0 (inside the
  ±45 deadzone), not the original's wider ±70 "centred" check.
- **RC frame length is checked.** A type 0x16 frame that is not 24 long is counted (`len=`) and not decoded.
- **Link quality.** A fresh receiver report of 0 % counts as radio lost, on top of the 250 ms timeout.
  Reports are never required.
- **Stop path.** See below.
- **USB output.** Three line types instead of the original's debug text.

## Rules the sketch enforces

- Boots disarmed. Arming needs the ARM switch seen **off, then on** over a live radio link.
- The PC can only send drive commands. Nothing it sends arms, selects PC mode or clears a fault.
- A switch value between the bands (600 to 1400) or outside 100 to 1950 is invalid and never grants drive.
- Every handoff brakes first, forgets the old command and changes the token. The new source must then
  show a neutral input that arrives **after** the switch: sticks at zero, or a `0,0` PC command carrying
  the new token. A held stick or key does nothing until it is released.
- Disarm or radio loss stops in either mode. In PC mode, 250 ms without an accepted command also stops.
- Radio loss and PC loss latch. Motion needs ARM off, then on. There is no fall-back to the other source.

## Diagnostics

`ST` is sent on every change (at most 50 a second) and every 0.2 s:

```
ST t=11340 tok=1511506142 link=1 armed=0 req=RADIO act=NONE pc_ok=0 block=DISARMED fault=NONE stop=NONE L=0 R=0 pc_age=-1
```

| Field | Meaning |
| --- | --- |
| `link` | 1 = qualified radio link |
| `armed` | 1 = armed by the ARM switch |
| `req` | what the MODE switch asks for: `RADIO`, `PC`, or `NONE` (no link or invalid) |
| `act` | the source whose commands reach the motors now: `RADIO`, `PC` or `NONE` |
| `pc_ok` | 1 = the firmware is accepting PC drive commands |
| `block` | why nothing is driving (table below) |
| `fault` | latched cause that needs ARM off then on: `RADIO_LOST`, `PC_STALE`, `ARM_SWITCH_INVALID` |
| `stop` | cause of the most recent stop: `DISARM`, `RADIO_LOST`, `PC_STALE`, `HANDOFF`, `ARM_SWITCH_INVALID`, `MODE_SWITCH_INVALID` |
| `L`, `R` | motor outputs, -255..255 |
| `pc_age` | ms since the last well-formed PC command, -1 = never |

| `block` | Meaning |
| --- | --- |
| `NONE` | a source has control |
| `NO_RADIO` | no qualified radio link since boot |
| `RADIO_LOST` | the radio link is down |
| `DISARMED` | ARM switch off; ready to arm |
| `ARM_ON_AT_BOOT` | ARM switch on but never seen off: turn it off, then on |
| `REARM_REQUIRED` | a fault latched: turn ARM off, then on |
| `ARM_SWITCH_INVALID` / `MODE_SWITCH_INVALID` | that channel is not clearly low or high |
| `WAIT_RADIO_NEUTRAL` | armed, RADIO selected, sticks not at zero yet |
| `WAIT_PC_NEUTRAL` | armed, PC selected, no fresh `0,0` command yet |

`CH` shows raw radio values, four times a second while disarmed and once a second while armed:

```
CH rc_age=0 frames=2689 crc=0 len=0 lq=-1 ch1=992 ch2=992 ch5=191 ch6=191 ch7=992 ch8=992 ch9=992 ch10=992 arm=CH5:191:DISARM mode=CH6:191:RADIO steer=0 thr=0 pc_bad=0
```

`HELLO` is sent at boot and when the PC sends `?`.

## PC protocol

One command line, sent 50 times a second:

```
D,<tok>,<throttle>,<steer>\n
```

- `tok` is the `tok=` value from the latest `ST` line.
- `throttle` and `steer` are integers -255..255 in the same units as the normalised sticks (positive =
  forward, positive = right). The firmware clamps both to ±128 (`PC_COMMAND_LIMIT`).
- A command counts only while armed, with PC selected and the current token. Malformed lines, `\r\n`
  endings, wrong tokens and lines over 39 characters are ignored and do not keep the link alive.

There was no serial protocol in the repo to reuse (`control_center/backend/robot_link.py` is a stub, and
Notes.md §10.4 is a design for the ESP-NOW link). The line format and token idea come from the earlier
prototype in `Documents\Arduino\ELRS_Control` (`drive_handoff.h`, `pc_drive.py`), reworked without the IMU.

## Stop path: what the core source says

The original stops with `digitalWrite(pin, HIGH)` and drives with `analogWrite` on the same pins. In the
installed ESP32 core (3.3.0):

- `analogWrite` attaches the pin to the LEDC peripheral (`esp32-hal-ledc.c`).
- `digitalWrite` then does nothing on that pin: `__digitalWrite` in `esp32-hal-gpio.c` only acts on a pin
  registered as GPIO, and otherwise logs "IO is not set as GPIO" (hidden at Core Debug Level: None).
- `ledcWrite(pin, 255)` at 8 bits is constant high.

So, by the source, a pin that has carried PWM does not reach the brake level in the original. The model
of the original on this core (scenario `original_stop_path_model`) gives:

| Step | AIN1 GPIO3 | AIN2 GPIO2 | BIN1 GPIO4 | BIN2 GPIO5 |
| --- | --- | --- | --- | --- |
| after start-up, sticks never moved | HIGH | HIGH | HIGH | HIGH |
| stick released, only forward used so far | PWM 6/256 | HIGH | HIGH | PWM 6/256 |
| stick released, both directions used | PWM 6/256 | PWM 7/256 | PWM 7/256 | PWM 6/256 |
| full forward, after both directions used | HIGH | PWM 7/256 | PWM 7/256 | HIGH |
| transmitter off at full forward | HIGH | PWM 7/256 | PWM 7/256 | HIGH |

Read literally, that is nearly full reverse after the first forward push, coasting instead of braking
later, and no stop when the transmitter goes off. This is a prediction from source code. It has not been
measured, and it does not match your experience that RadioMaster driving works, so treat it as a question
for the multimeter (step A9), not as a fact about the robot.

**In this sketch only:** every motor pin level is written with `analogWrite` (255 = constant high, 0 =
constant low) and `digitalWrite` is never used on a motor pin. Drive is unchanged: PWM on one input, the
other input low. On the model, all four inputs are constant high after forward, reverse, a spin, ARM off
and transmitter off.

Still to measure, because neither the core's real behaviour nor the pins have been observed:

1. All four driver inputs at about 3.3 V after a drive command followed by a stop (step A5).
2. The inputs during reset and upload. GPIO2 (AIN2) is a strapping pin; if the board pulls it high in
   reset, the left motor is driven then, whatever the sketch (step A2).
3. If this sketch brakes where the original coasted, the robot will stop more sharply than you are used to.

## Finding your switches (nothing to confirm before flashing)

The RadioMaster must send a two-position switch on CH5 and another on CH6 (EdgeTX: one mixer line each).
ELRS treats CH5 as its own arm channel and sends it as a two-position value, with high meaning armed.
Watch the `CH` line with the robot disarmed:

- flip a switch and see which of `ch5`..`ch10` moves, and to what value (two-position: about 191 and 1792);
- `arm=CH5:1792:ARM` and `mode=CH6:191:RADIO` show how the firmware reads them;
- a three-position switch in the middle shows `INVALID`;
- if your switches are on other channels or reversed, edit `ARM_CHANNEL`, `MODE_CHANNEL`,
  `ARM_WHEN_HIGH`, `PC_WHEN_HIGH` and upload again.

## PC client

Install (once):

```powershell
python -m pip install pyserial
```

Find the port:

```powershell
python scripts/handoff_bench_client.py --list
```

Listen only (sends nothing; use this to find your switches):

```powershell
python scripts/handoff_bench_client.py --port COM5 --monitor
```

Keyboard drive window:

```powershell
python scripts/handoff_bench_client.py --port COM5
```

- Hold **SPACE** (deadman) with **W/A/S/D** or the arrow keys. Releasing SPACE, releasing the direction
  keys or clicking another window sends zero.
- Default command 80 forward/back and 70 turn, out of 255. `--speed` and `--turn` change them; the
  firmware still clamps to 128.
- The window shows the active source and whether the firmware is accepting PC commands, with the reason
  when it is not.
- **P** stops sending altogether, to test the stale-command stop. P again resumes.
- `--log bench.txt` records every received line and every change of command.
- Close Arduino's Serial Monitor first: only one program can hold the port.

## Software tests

```powershell
python -m pytest tests/test_handoff_bench.py -q
```

60 tests, about a minute; the firmware ones need the MSVC build tools and are skipped without them.

- **Firmware on a mocked board, 24 scenarios, one simulated millisecond per loop:** boot with no radio;
  ARM already on at power-up (RADIO and PC); drive output equal to the original at 676 stick positions;
  stop-path pin levels; disarm in each mode; handoff in each direction with stick and key held; MODE
  flapping; stale PC commands; malformed and wrong-token lines; client reconnecting with the old command
  and token; radio loss in each mode; 0 % link quality; noise instead of frames; wrong-length RC frames;
  invalid and swept switch values; PC trying to arm; PC limit; USB not being read; diagnostic rates.
- **Rules checked on every simulated millisecond:** a wheel is driven only with a radio frame in the last
  250 ms, ARM on and seen off on this link, a valid MODE, a neutral input since the last switch change,
  and an output equal to the active source's command.
- **The real client against the real sketch, 5 runs:** keys held before arming and through both handoffs,
  client killed and restarted with the key held, pause key, focus loss, disarm and radio loss.
- **16 deliberately broken copies of the sketch** (old stop path, no neutral check, no token check, PC
  stale falling back to RADIO, faults not latching, long timeouts, PC able to arm, and others) each fail
  at least one scenario. Four changed shared constants each fail the build.

Compile for the board, no upload (Arduino IDE's bundled arduino-cli 1.5.0, esp32 core 3.3.0, your IDE's
options with `CDCOnBoot=cdc`): 329,850 bytes (25 %), no warnings. The untouched original also still
compiles (326,770 bytes). With USB CDC On Boot disabled the new sketch refuses to build, on purpose.

None of this shows what the real pins, receiver or USB port do.

## Bench sequence

Uploading is your step. Arduino IDE: open `ELRS-Handoff-Bench.ino`, Board "ESP32C3 Dev Module",
"USB CDC On Boot: Enabled", Upload.

### A. Motors disconnected

Unplug both motors from the DRV8833 outputs before uploading, and keep them unplugged for all of A.
Multimeter on the driver inputs: AIN1 (GPIO3), AIN2 (GPIO2), BIN1 (GPIO4), BIN2 (GPIO5).

| # | Do | Expect |
| --- | --- | --- |
| A1 | Upload. Transmitter off. Start `--monitor`. | `ST ... link=0 armed=0 act=NONE block=NO_RADIO`. All four inputs about 3.3 V. |
| A2 | Hold the board's reset button and read AIN2 and AIN1. Release. | Both about 0 V in reset. If AIN2 is about 3.3 V with AIN1 at 0 V, the left motor is driven during every reset and upload: note it. After release: all four about 3.3 V again, `block=NO_RADIO`. |
| A3 | Transmitter on, ARM off. Flip each switch. | `link=1 block=DISARMED`. `CH` line: `arm=CH5:191:DISARM` / `CH5:1792:ARM`, `mode=CH6:191:RADIO` / `CH6:1792:PC`. If not, fix the four defines and repeat from A1. |
| A4 | ARM on, then power-cycle the board (ARM still on). Push the throttle. | `armed=0 block=ARM_ON_AT_BOOT`, `L=0 R=0`, all four about 3.3 V. |
| A5 | MODE = RADIO. ARM off, then on, sticks centred. Push throttle forward, hold, release. Then reverse, release. | `armed=1 act=RADIO block=NONE`. Forward: `L` and `R` positive, AIN2 and BIN1 about 0 V, AIN1 and BIN2 above 0 V. **Released: `L=0 R=0` and all four about 3.3 V.** Same after reverse. This is the stop-path measurement. |
| A6 | Hold the throttle and turn ARM off. | `armed=0 block=DISARMED stop=DISARM`, all four about 3.3 V. |
| A7 | Still holding the throttle, ARM on. Then centre the stick. | `block=WAIT_RADIO_NEUTRAL`, `L=0 R=0` while held. Centred: `act=RADIO`. |
| A8 | Hold the throttle and switch the transmitter off. Switch it on again. | Within about a quarter of a second: `link=0 block=RADIO_LOST fault=RADIO_LOST stop=RADIO_LOST`, all four about 3.3 V. Back on: `link=1 armed=0 block=REARM_REQUIRED`. ARM off, on, centre: `act=RADIO`. If it does not stop, your receiver keeps sending frames after loss: stop here. |
| A9 | Optional, original sketch only: flash it (see Restore), push forward once, release, read the inputs. | All four about 3.3 V means the original does brake and my reading of the core is wrong. AIN1 near 0 V with AIN2 at 3.3 V means it is right. |
| A10 | Stop `--monitor`. Start the drive window. ARM on in RADIO, then MODE to PC with no keys held. | `req=PC`, then "PC HAS CONTROL", `pc_ok=1`, `L=0 R=0`. |
| A11 | Hold SPACE+W. Release. | `L=80 R=80`, inputs as in A5 forward. Released: `L=0 R=0`, all four about 3.3 V. |
| A12 | Hold SPACE+W and flip MODE to RADIO with the throttle pushed. Centre the stick. Flip back to PC, keys still held. Release the keys. | Each flip: `L=0 R=0`, `stop=HANDOFF`. `block=WAIT_RADIO_NEUTRAL` until the stick is centred. `block=WAIT_PC_NEUTRAL` and "release all keys" until the keys are released. |
| A13 | In PC mode hold SPACE+W and press P. Press P again, keys still held. | Within about a quarter of a second: `armed=0 block=REARM_REQUIRED fault=PC_STALE stop=PC_STALE`, all four about 3.3 V. After the second P: still `REARM_REQUIRED`. Push the radio stick: nothing. ARM off, on: `WAIT_PC_NEUTRAL` until the keys are released. |
| A14 | In PC mode with SPACE+W held, turn ARM off. Then repeat with the transmitter switched off instead. | `block=DISARMED`; then `block=RADIO_LOST`. `L=0 R=0` both times, although the PC keeps sending. |
| A15 | Close the drive window while driving, reopen it. | `fault=PC_STALE block=REARM_REQUIRED` (a closed client is a lost PC), and `t=` keeps counting. If `t=` restarts from a small number, opening the port resets the board; note it (it boots disarmed either way). |

### B. Motors connected, wheels off the ground

Only after A is clean. Reconnect the motors with the board unpowered. Power cut-off in reach. The PC steps
need USB and the motor battery together: Notes.md §7 warns against that if the ESP32 is fed 3.3 V
directly without a diode, so check how your board is powered first.

| # | Do | Expect |
| --- | --- | --- |
| B1 | Power up, transmitter on, ARM off. | Wheels still. `block=DISARMED`. |
| B2 | Arm in RADIO. Small throttle forward, release. Reverse, release. Left, right. | Both wheels forward, then back; turning as before. Each release stops the wheels, possibly more sharply than with the original. |
| B3 | Small throttle, ARM off. Re-arm with the stick held, then centre. | Wheels stop at once and stay stopped until the stick has been centred. |
| B4 | Small throttle, transmitter off. | Wheels stop within about a quarter of a second and stay stopped when the transmitter returns, until ARM off and on. |
| B5 | MODE to PC, drive window, SPACE+W, release. Then S, A, D. | Wheels turn slowly (80 of 255) in the matching direction and stop on release. If 80 does not move them, restart with `--speed 110`. |
| B6 | Repeat A12, A13 and A14. | Wheels stop at every handoff, on P, on ARM off and on transmitter off, and never start while a stick or key is still held from before. |

## Restoring the original sketch

Motors disconnected. Arduino IDE: open your local known-good `RadioMaster_KnownGood.ino` sketch.
(read-only, identical to `ELRS-Control/ELRS-Control.ino` in this repo), Board "ESP32C3 Dev Module",
"USB CDC On Boot: Enabled", Upload. To check the file first:

```powershell
(Get-FileHash "ELRS-Control\ELRS-Control.ino").Hash
```

It should print `6040BED377E40AFFA3DC4105D23E96EACB788E6E5AFCF4FCAA60CDBFCFDEA6DD`.

## Still to confirm on the RadioMaster and the robot

- Which channels your ARM and MODE switches are on, and which position is high (A3).
- That both are two-position, or that you only ever use the end positions.
- What the receiver sends when the transmitter goes off (A8).
- Driver input levels in reset (A2) and after a stop (A5).
- Whether opening the COM port resets the board (A15).
- Whether 80 of 255 is enough to turn the wheels (B5).
