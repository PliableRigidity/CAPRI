"""Session and round logs for the Command Centre: what the system saw, decided, converted and tried to send.

One folder per application run (``logs/sessions/<timestamp>/``) holding ``session.jsonl`` and, for each round,
``round-001.jsonl`` ... Each line is one JSON event with both a monotonic and a wall-clock time. Events are flushed as
they are written, so a crash or a power cut loses at most the line being written; a half-written last line is
ignored by the reader.

What is NOT claimed: a transmitted command is logged as a transmission attempt (``sent`` True/False and why not), and
``physical_execution`` is always ``"unknown: no evidence"``. There is no feedback from the motors in this system, so
nothing in this log says the robot moved. High-rate kinds (observation, tracking) are sampled to a maximum rate;
decisions, command plans, transmissions, mode changes, faults and round boundaries are never dropped.

No Qt, no hardware. The clock and the root folder are injectable.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator

log = logging.getLogger(__name__)
DEFAULT_ROOT = Path(__file__).resolve().parents[2] / "logs" / "sessions"
NO_EVIDENCE = "unknown: no evidence"

# kind -> minimum seconds between two logged events of that kind (0 = every event is kept)
SAMPLED = {"camera": 1.0, "tracking": 0.2, "observation": 0.2, "wifi_status": 1.0}
KINDS = ("session", "camera", "tracking", "observation", "decision", "command_plan", "transmit", "wifi_status", "mode", "override",
         "safety", "round", "note", "error")


class SessionLog:
    def __init__(self, root: Path | str | None = None, *, clock: Callable[[], float] = time.monotonic,
                 wall_clock: Callable[[], datetime] | None = None, label: str = "") -> None:
        self.root = Path(root) if root else DEFAULT_ROOT
        self._clock = clock
        self._wall = wall_clock or (lambda: datetime.now(timezone.utc))
        self._lock = threading.Lock()
        stamp = self._wall().strftime("%Y%m%d-%H%M%S")
        self.folder = self.root / (stamp + (f"-{label}" if label else ""))
        counter = 1
        while self.folder.exists():
            counter += 1
            self.folder = self.root / (stamp + (f"-{label}" if label else "") + f"-{counter}")
        self.folder.mkdir(parents=True, exist_ok=True)
        self._files: dict[str, Any] = {}
        self._last: dict[str, float] = {}
        self.round_number = 0
        self.round_active = False
        self.dropped = 0
        self.errors = 0
        self.closed = False
        self.log("session", "start", folder=str(self.folder))

    # ---- writing
    def _open(self, name: str):
        handle = self._files.get(name)
        if handle is None:
            handle = open(self.folder / name, "a", encoding="utf-8", newline="\n")
            self._files[name] = handle
        return handle

    def log(self, kind: str, event: str = "", **data: Any) -> bool:
        """Append one event. Returns False when it was sampled out or the log is closed. Never raises."""
        if self.closed:
            return False
        now = self._clock()
        with self._lock:
            interval = SAMPLED.get(kind, 0.0)
            if interval and now - self._last.get(kind, -1e9) < interval:
                self.dropped += 1
                return False
            self._last[kind] = now
            record = {"t_mono": round(now, 4), "t_wall": self._wall().isoformat(), "kind": kind, "event": event, **data}
            if self.round_active:
                record["round"] = self.round_number
            line = json.dumps(record, default=str, separators=(",", ":"))
            try:
                for name in ("session.jsonl",) + ((f"round-{self.round_number:03d}.jsonl",) if self.round_active else ()):
                    handle = self._open(name)
                    handle.write(line + "\n")
                    handle.flush()
            except (OSError, ValueError) as error:
                self.errors += 1
                log.error("session log write failed: %s", error)
                return False
        return True

    # ---- the things the Command Centre records
    def command(self, plan: Any, *, mode: str, authority: str) -> None:
        """A command plan and whether a datagram was written. ``plan`` is a control_modes.CommandPlan or a dict."""
        data = plan if isinstance(plan, dict) else {k: getattr(plan, k, None) for k in (
            "tactic", "intended_throttle", "intended_steer", "transmitted_throttle", "transmitted_steer", "sent", "why_not")}
        self.log("command_plan", mode=mode, authority=authority, **data)
        self.log("transmit", sent=bool(data.get("sent")), throttle=data.get("transmitted_throttle"), steer=data.get("transmitted_steer"),
                 why_not=data.get("why_not"), physical_execution=NO_EVIDENCE)

    def mode_change(self, old: str, new: str, ok: bool, reasons: list[str] | None = None) -> None:
        self.log("mode", f"{old} -> {new}" if ok else f"{old} -> {new} REFUSED", ok=ok, reasons=list(reasons or []))

    def fault(self, what: str, **data: Any) -> None:
        self.log("safety", what, **data)

    def start_round(self, mode: str, **data: Any) -> int:
        self.round_number += 1
        self.round_active = True
        self.log("round", "start", mode=mode, **data)
        return self.round_number

    def stop_round(self, reason: str, **data: Any) -> None:
        if self.round_active:
            self.log("round", "stop", reason=reason, **data)
        self.round_active = False
        handle = self._files.pop(f"round-{self.round_number:03d}.jsonl", None)
        if handle is not None:
            try:
                handle.close()
            except OSError:
                pass

    def close(self, reason: str = "application closed") -> None:
        if self.closed:
            return
        if self.round_active:
            self.stop_round("session closed during the round")
        self.log("session", "stop", reason=reason, dropped_samples=self.dropped, write_errors=self.errors)
        self.closed = True
        for handle in self._files.values():
            try:
                handle.close()
            except OSError:
                pass
        self._files.clear()


# ---- reading (for the screens)

def read_events(path: Path | str) -> Iterator[dict[str, Any]]:
    """Events of one log file. A damaged or half-written line is skipped, never an error."""
    try:
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                try:
                    yield json.loads(line)
                except ValueError:
                    continue
    except OSError:
        return


def tail(path: Path | str, count: int = 200) -> list[dict[str, Any]]:
    events = list(read_events(path))
    return events[-count:]


def list_sessions(root: Path | str | None = None) -> list[Path]:
    base = Path(root) if root else DEFAULT_ROOT
    return sorted((p for p in base.glob("*") if p.is_dir()), reverse=True) if base.exists() else []


def describe(event: dict[str, Any]) -> str:
    """One readable line for the screen."""
    when = str(event.get("t_wall", ""))[11:19]
    rest = {k: v for k, v in event.items() if k not in ("t_mono", "t_wall", "kind", "event", "round")}
    detail = ", ".join(f"{k}={v}" for k, v in rest.items())
    return f"{when} {event.get('kind', '')}: {event.get('event', '')} {detail}".strip()
