"""Summarise a handoff client log (`handoff_bench_client.py --log FILE`): what the controller itself reported.

    python scripts\\handoff_log_summary.py logs\\handoff-wifi.txt

Read-only. It opens one text file and prints counts; it talks to nothing.

What it can show: when the controller's own clock (`t=` in its ST lines) went backwards, which means
the firmware restarted; how often status stopped arriving; which faults latched; whether the motors
were commanded at the moment of a restart; the radio link quality the receiver reported.

What it cannot show: voltages, why a restart happened, or anything that happened while no client was
logging. A restart with nobody listening leaves no line here.
"""

from __future__ import annotations

import collections
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

STATUS = re.compile(r"^(\d+\.\d+) RX ST t=(\d+) tok=(\d+) link=(\d) armed=(\d) req=(\w+) act=(\w+) pc_ok=(\d) "
                    r"block=(\w+) fault=(\w+) stop=(\w+) L=(-?\d+) R=(-?\d+) pc_age=(-?\d+)")
CHANNELS = re.compile(r"^(\d+\.\d+) RX CH rc_age=(-?\d+) .*?lq=(-?\d+)")
ANY = re.compile(r"^(\d+\.\d+) (RX|TX) (.*)")
SESSION_GAP_S = 5.0          # a longer silence in the file is a new client session, not a link dropout
SILENCE_S = 0.6              # the firmware sends ST at least every 0.2 s


def clock(unix: float) -> str:
    return datetime.fromtimestamp(unix, timezone.utc).strftime("%H:%M:%S")


def summarise(path: Path) -> dict:
    sessions: list[list[float]] = []
    restarts, silences = [], []
    faults = collections.Counter()
    fault_times: dict[str, list[str]] = collections.defaultdict(list)
    blocks_after_restart = collections.Counter()
    quality = collections.Counter()
    frame_age = collections.Counter()
    firmware = collections.Counter()
    recent: collections.deque[str] = collections.deque(maxlen=4)
    previous_any = previous_rx = previous_uptime = previous_status_at = None
    previous_fault, link, after_restart = "NONE", None, 0
    lines = nonzero_sent = driving_pc = driving_radio = 0
    pc_seconds = 0.0
    with path.open(encoding="utf-8", errors="replace") as handle:
        for raw in handle:
            entry = ANY.match(raw)
            if not entry:
                continue
            lines += 1
            at = float(entry.group(1))
            if previous_any is None or at - previous_any > SESSION_GAP_S:
                sessions.append([at, at])
                previous_rx = previous_uptime = previous_status_at = None
                recent.clear()
            sessions[-1][1] = previous_any = at
            if entry.group(2) == "TX":
                parts = entry.group(3).split(",")
                nonzero_sent += len(parts) >= 4 and (parts[2] != "0" or parts[3] != "0")
                continue
            if previous_rx is not None and at - previous_rx > SILENCE_S:
                silences.append((previous_rx, at - previous_rx))
            previous_rx = at
            if entry.group(3).startswith("HELLO "):
                found = re.search(r"fw=(\S+)", entry.group(3))
                firmware[found.group(1) if found else "?"] += 1
            channels = CHANNELS.match(raw)
            if channels:
                age, lq = int(channels.group(2)), int(channels.group(3))
                if link == "1":
                    quality["not reported" if lq < 0 else "0 %" if lq == 0 else "1-89 %" if lq < 90 else "90-100 %"] += 1
                    frame_age["none yet" if age < 0 else "20 ms or less" if age <= 20 else "21-100 ms" if age <= 100 else "over 100 ms"] += 1
                continue
            status = STATUS.match(raw)
            if not status:
                continue
            uptime, link = int(status.group(2)), status.group(4)
            if previous_uptime is not None and uptime + 1000 < previous_uptime:
                restarts.append({"at": at, "uptime_before_s": previous_uptime / 1000.0, "uptime_after_s": uptime / 1000.0,
                                 "motors_commanded_just_before": any(" L=0 R=0 " not in line for line in recent),
                                 "last_line_before": recent[-1] if recent else ""})
                after_restart = 40
            elif after_restart:
                after_restart -= 1
                blocks_after_restart[status.group(9)] += 1
            fault = status.group(10)
            if fault != "NONE" and previous_fault == "NONE":
                faults[fault] += 1
                fault_times[fault].append(clock(at))
            previous_fault = fault
            moving = (status.group(12), status.group(13)) != ("0", "0")
            driving_pc += moving and status.group(7) == "PC"
            driving_radio += moving and status.group(7) != "PC"
            if status.group(7) == "PC" and previous_status_at is not None:
                pc_seconds += min(at - previous_status_at, 0.25)
            previous_uptime = uptime
            previous_status_at = at
            recent.append(raw.strip().split(" ", 1)[1])
    return {"lines": lines, "sessions": sessions, "restarts": restarts, "silences": silences, "faults": faults,
            "fault_times": fault_times, "blocks_after_restart": blocks_after_restart, "quality": quality,
            "frame_age": frame_age, "firmware": firmware, "nonzero_sent": nonzero_sent, "driving_pc": driving_pc,
            "driving_radio": driving_radio, "pc_seconds": pc_seconds}


def report(path: Path) -> str:
    data = summarise(path)
    if not data["sessions"]:
        return f"{path}: no client log lines found"
    first, last = data["sessions"][0][0], data["sessions"][-1][1]
    day = datetime.fromtimestamp(first, timezone.utc).strftime("%Y-%m-%d")
    out = [f"Log: {path}", f"Date {day} (UTC), {clock(first)} to {clock(last)}, {data['lines']} lines",
           "Firmware named in HELLO lines: " + (", ".join(sorted(data["firmware"])) or "none seen"), "",
           f"Client sessions (a gap of more than {SESSION_GAP_S:.0f} s in the file starts a new one): {len(data['sessions'])}"]
    out += [f"  {clock(start)} to {clock(end)}  ({end - start:6.1f} s)" for start, end in data["sessions"]]
    out += ["", f"Status lines with the motors commanded: {data['driving_pc']} with PC as the source, "
                f"{data['driving_radio']} with the RadioMaster as the source",
            f"Non-zero drive commands sent by the PC: {data['nonzero_sent']}",
            f"Time with PC as the active source: about {data['pc_seconds']:.0f} s", "",
            f"Firmware restarts seen (the controller's clock went back by more than 1 s inside a session): {len(data['restarts'])}"]
    for item in data["restarts"]:
        out.append(f"  {clock(item['at'])}  after {item['uptime_before_s']:.1f} s of uptime; motors commanded in the last "
                   f"four status lines before it: {'YES' if item['motors_commanded_just_before'] else 'no'}")
        out.append(f"      last line before: {item['last_line_before']}")
    if data["restarts"]:
        out.append("  state reported in the 40 status lines after each restart: "
                   + ", ".join(f"{name} x{count}" for name, count in data["blocks_after_restart"].most_common()))
    long = [item for item in data["silences"] if item[1] > 2.0]
    out += ["", f"Status silences longer than {SILENCE_S} s inside a session: {len(data['silences'])} "
                f"({len(long)} longer than 2 s)"]
    out += [f"  {clock(at)}  {seconds:.1f} s" for at, seconds in sorted(data["silences"], key=lambda item: -item[1])[:8]]
    out += ["", "Latched faults (each needs ARM off, then on): "
            + (", ".join(f"{name} x{count}" for name, count in data["faults"].items()) or "none")]
    for name, times in data["fault_times"].items():
        out.append(f"  {name} at " + ", ".join(times))
    out += ["", "Radio link while the controller reported link=1 (from its CH lines):",
            "  link quality: " + (", ".join(f"{name}: {count}" for name, count in sorted(data["quality"].items())) or "no CH lines"),
            "  age of the newest radio frame: " + (", ".join(f"{name}: {count}" for name, count in sorted(data["frame_age"].items())) or "no CH lines")]
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 1:
        print(__doc__)
        return 2
    path = Path(args[0])
    if not path.is_file():
        print(f"no such file: {path}")
        return 2
    print(report(path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
