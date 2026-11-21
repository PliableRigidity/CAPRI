# ELRS-Handoff-WiFi

The USB handoff sketch you tested (`ELRS-Handoff-Bench`), plus one thing: the laptop can send the
same drive commands over Wi-Fi, so PC mode works on battery with the USB cable unplugged.

> **Status, night of 20 November 2026: EXPERIMENTAL, not on the robot.** This sketch was flashed and run
> on 20 November. The access point came up and the laptop drove the robot untethered. Over the next two
> hours the client's log shows 12 latched stale-command faults, 56 status dropouts and 3 firmware
> restarts; later the robot restarted in a loop on battery and the ELRS receiver power-cycled with it.
> Readings taken that day showed the supply collapsing in a failing condition (about 3.0 V into the buck
> converter and 1.0 V out). The root cause was not identified, and it has **not** been shown that Wi-Fi
> caused it: its extra current draw exposing a marginal supply is a hypothesis. While the radio link was
> up, the receiver reported 98 to 100 % link quality throughout.
>
> The robot was returned to `ELRS-Handoff-Bench`. This sketch is kept, with its tests, as route D. Do not
> make it the default before the supply passes the playbook's PWR-1 on battery and the acceptance test
> below passes in full. Evidence: `../artifacts/integration-2026-11-20/evidence.md`; summary of the log:
> `python scripts\handoff_log_summary.py logs\handoff-wifi.txt`.

The text below is as written before the first upload. "Software" means compiled, or run on the PC
against a model of the board, the access point and the UDP socket.

| File | What it is |
| --- | --- |
| `ELRS-Handoff-WiFi.ino` | the sketch |
| `../scripts/handoff_bench_client.py` | the same PC client, now with `--wifi` (USB still `--port COMx`) |
| `../tests/test_handoff_wifi.py` | the software tests |
| `host_test/` | mocked board, access point and socket; the Arduino build ignores this folder |
| `../ELRS-Handoff-Bench/` | the tested USB version, untouched |
| `../ELRS-Control/ELRS-Control.ino` | your original, untouched |

## What changed from the tested USB sketch

- The robot starts a password-protected access point (`AutoBot-Handoff`, one laptop at most) and
  listens on UDP port 4210. No venue Wi-Fi and no extra hardware.
- A Wi-Fi command goes through the same function as a USB command: same token check, same neutral
  rule, same 250 ms timeout, same clamp to 128, same mixer. There is no second motor path.
- **One link at a time.** Whichever of USB or Wi-Fi supplies the neutral command holds the PC input
  until the next handoff, disarm or fault. Commands from the other one are ignored and do not keep the
  link alive.
- Wi-Fi commands carry a number that counts up, so a repeated or late datagram is never applied.
- Status lines also go to the laptop over Wi-Fi, plus one new line, `NET`.
- The radio's serial buffer is larger (1024 bytes), so a busy moment in the Wi-Fi stack does not drop
  radio bytes.

Unchanged, character for character (a test checks it): radio parsing, calibration, mixer, motor
writes, arming, the handoff rules, the stop rules and the 500 ms start-up. Nothing waits for USB or for
a laptop: with neither present it is a RadioMaster-only robot.

## Set the Wi-Fi password first

Near the top of the sketch:

```
#define WIFI_AP_SSID      "AutoBot-Handoff"
#define WIFI_AP_PASSWORD  "..."        <- change this: 8 to 63 characters
#define WIFI_AP_CHANNEL   6            <- 1, 6 or 11
```

The password in the repo is a placeholder. Anyone who knows it can take the one laptop slot.

## Upload

Arduino IDE, same settings as the USB version: Board "ESP32C3 Dev Module", "USB CDC On Boot: Enabled",
CPU 160 MHz, Flash 80 MHz QIO 4 MB, default partition scheme. Motors unplugged while uploading.
Compiled size: 983,406 bytes (75 % of the default partition).

## Join the robot's network

1. Power the robot. USB is not needed.
2. On the laptop: Wi-Fi list → `AutoBot-Handoff` → enter the password → Connect.
   Windows will say "No internet, secured". That is expected; stay connected.
3. Untick "Connect automatically" for your other networks for the test, or Windows may switch back
   to one that has internet.
4. The robot is `192.168.4.1`.

## Drive

```powershell
python scripts/handoff_bench_client.py --wifi
```

Listen only (sends a status request once a second, never a drive command):

```powershell
python scripts/handoff_bench_client.py --wifi --monitor
```

USB still works as before, with either sketch:

```powershell
python scripts/handoff_bench_client.py --port COM5
```

The window and keys are the same as over USB: the RadioMaster arms and selects PC, then hold SPACE
with W/A/S/D. Use `--wifi` or `--port`, not both.

## What you will see

`ST` lines are the same as the USB version. The new line:

```
NET ap=1 sta=1 peer=192.168.4.2:50123 pc_via=WIFI rx=1234 bad=0 old=0 other=0
```

| Field | Meaning |
| --- | --- |
| `ap` | 1 = access point and UDP port are up |
| `sta` | laptops joined (0 or 1) |
| `peer` | where status is being sent over Wi-Fi, or `NONE` |
| `pc_via` | which link holds the PC input: `NONE`, `USB` or `WIFI` |
| `rx`, `bad`, `old` | datagrams received, malformed, repeated or late |
| `other` | commands ignored because the other link holds the input |

| Situation | Client headline | `ST` contains |
| --- | --- | --- |
| Joined, disarmed | NOBODY HAS CONTROL | `armed=0 block=DISARMED` |
| Armed, MODE = PC, keys released | PC HAS CONTROL | `act=PC pc_ok=1`, `NET … pc_via=WIFI` |
| Wi-Fi dropped | WI-FI LINK LOST | (nothing arrives) |
| Wi-Fi back | NOBODY HAS CONTROL | `armed=0 block=REARM_REQUIRED fault=PC_STALE stop=PC_STALE` |
| RadioMaster takes over | RADIO HAS CONTROL | `req=RADIO act=RADIO pc_ok=0 stop=HANDOFF` |

After a Wi-Fi drop the robot stays stopped until you turn ARM off then on **on the RadioMaster** and
release the laptop keys. That is the same latch as the USB version, and it is the "deliberate
re-enable": the laptop cannot clear it.

## Acceptance test: battery, USB unplugged, wheels raised

Do the tested USB bench checks first if anything about the wiring changed. Start with the transmitter
on, ARM off, MODE = PC, laptop joined, client running with `--wifi`.

| # | Do | Expect |
| --- | --- | --- |
| 1 | Look at the client. | NOBODY HAS CONTROL, "Blocked by: DISARMED", "PC link: Wi-Fi 192.168.4.1:4210". Wheels still. |
| 2 | Hold SPACE+W. | Nothing moves (not armed). |
| 3 | Release the keys. RadioMaster: ARM on. | PC HAS CONTROL. Wheels still. |
| 4 | SPACE+W, release. SPACE+S, release. SPACE+D, release. SPACE+A, release. | Forward, reverse, turn right, turn left at low speed; wheels stop at each release. |
| 5 | Hold SPACE+W. With the wheels turning, disconnect the laptop from `AutoBot-Handoff` (keep holding). | Wheels stop within about a quarter of a second. Client: WI-FI LINK LOST. |
| 6 | Still holding the keys, reconnect to `AutoBot-Handoff`. Wait 10 s. | Wheels stay stopped. Client: NOBODY HAS CONTROL, "Blocked by: REARM_REQUIRED", "Fault: PC_STALE". |
| 7 | Release the keys. Press SPACE+W again. | Still nothing: the fault is latched. |
| 8 | Release. RadioMaster: ARM off, then on. | PC HAS CONTROL again. Wheels still. |
| 9 | Hold SPACE+W (wheels turning). RadioMaster: MODE to RADIO, sticks centred. Keep holding the keys. | Wheels stop at once. Client: RADIO HAS CONTROL. The laptop's forward command does nothing. |
| 10 | Still holding the laptop keys, push the RadioMaster throttle, then centre it. | Wheels follow the stick only, and stop when it is centred. |
| 11 | RadioMaster: ARM off. | Wheels stopped, "Blocked by: DISARMED". |

If step 5 does not stop the wheels, cut the power and restore the USB version.

## Not verified on hardware

- That the access point comes up and the laptop can join it. Some ESP32-C3 Super Mini boards need
  lower transmit power to work at all; `WIFI_REDUCED_TX_POWER 1` in the sketch does that (compiled, never tried).
- Whether Wi-Fi on the robot disturbs the ELRS receiver beside it (same 2.4 GHz band). Watch `lq=` on
  the `CH` line and the radio stop behaviour; try channel 1 or 11 if it drops.
- Whether Windows Firewall lets the status replies in (the network is "Public" by default). If the
  client stays on NO STATUS FROM THE ROBOT while joined, allow Python on that network.
- Command delay and loss over the real link, and range.
- Battery sag when the Wi-Fi transmitter and the motors draw together (a brown-out resets the board;
  it boots disarmed).
- Real stop time in step 5.

## Software tests

```powershell
python -m pytest tests/test_handoff_wifi.py -q
```

- All 24 scenarios of the USB version, run against this sketch, plus 9 Wi-Fi scenarios: drive with USB
  dead from power-up; link drop while commanding motion, reconnect, replay of every old datagram;
  RadioMaster takeover while the laptop keeps sending; disarm and radio loss in Wi-Fi PC mode; USB and
  Wi-Fi never commanding together; late, repeated, foreign and malformed datagrams; a datagram flood
  not delaying the radio stop; access point failing to start; status following the laptop.
- The real client against the real sketch over the simulated link, including the acceptance sequence
  above.
- The client's UDP code over the loopback interface.
- 11 deliberately broken copies of the Wi-Fi additions and the 16 of the USB version are each caught.

## Restore the tested USB version

Open `ELRS-Handoff-Bench\ELRS-Handoff-Bench.ino` (unchanged, SHA-256 `1a400d3f…5fbc`) and upload it
with the same board settings, motors unplugged. The client command is the USB one above. A copy of
the client exactly as you tested it is in
`artifacts\code-snapshots\handoff-usb-tested-2026-11-20\handoff_bench_client.py`.

To go all the way back, upload `Documents\Arduino\RadioMaster_KnownGood\RadioMaster_KnownGood.ino`
(identical to `ELRS-Control\ELRS-Control.ino`).
